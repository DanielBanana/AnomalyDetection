"""How a tiled model's tile scores become one image score, and the image threshold.

Background, measurements and the reasoning behind the choices are archived in
documentation/source/decisions/tile_score_merge.rst of the AOI Inline repository.

Score merge (tiling config ``score_merge``):

- ``max``: every tile score is scaled by the good validation images of its tile
  position, ``z = (score - mean_good) / std_good`` -- how far above normal that
  spot is, whatever is printed there -- and the image score is the highest ``z``
  of its tiles. "Highest tile above the threshold" is "at least one tile above
  the threshold", so a defect in one of many tiles is not averaged away.
- ``mean``: the image score is the mean of the tile scores (anomalib's merge,
  used before 2026-10-14). A local defect counts only 1/(number of tiles).

The scaling is fitted when a model is calibrated (validation run of
EvalTiledEnsemble) and saved as ``tile_scaling.json`` next to ``stats.json``.
Test runs and inspection read the merge from that file, not from the config:
the image threshold only fits the merge it was calibrated with. Without the
file (a model calibrated before ``max`` existed) the merge is ``mean``.

Image threshold (tiling config ``image_threshold``):

- ``f1``: the threshold with the best F1 on the validation images (anomalib's
  F1AdaptiveThreshold, computed by AOIStatisticsJob in jobs.py).
- ``miss_rate``: at most ``max_miss_rate`` of the defective validation images
  may fall below the threshold (see miss_rate_threshold).

Either way the threshold is then mapped to 0.5 by the normalization, so
inspection keeps comparing with 0.5.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

MERGE_MAX = "max"
MERGE_MEAN = "mean"
SCORE_MERGES = (MERGE_MAX, MERGE_MEAN)
THRESHOLD_F1 = "f1"
THRESHOLD_MISS_RATE = "miss_rate"
IMAGE_THRESHOLDS = (THRESHOLD_F1, THRESHOLD_MISS_RATE)

TILE_SCALING_FILENAME = "tile_scaling.json"
# Lower bound of a tile position's spread, so that a position whose good
# scores happen to be (nearly) identical does not blow its z up.
MIN_SPREAD = 1e-6


def tile_key(tile_index: Tuple[int, int]) -> str:
    """"3_5" for tile (3, 5), as in the checkpoint name model3_5.ckpt."""
    return f"{tile_index[0]}_{tile_index[1]}"


def fit_tile_scaling(tile_batches: Dict[Tuple[int, int], Iterable[Any]]) -> Dict[str, Dict[str, float]]:
    """Mean and spread of every tile position's scores over its good tiles.

    ``tile_batches`` maps a tile position to its prediction batches (with
    ``pred_score`` and ``gt_label``; the label of a tile comes from its own part
    of the ground-truth mask, see TileCollater in ensemble_tiling.py). Good
    tiles of anomalous images count too: they show good surface. Returns
    {"3_5": {"mean", "std", "good"}}. A position without any good tile gets
    mean 0 and std 1 (its scores are then used as they are) and a warning."""
    scaling: Dict[str, Dict[str, float]] = {}
    for tile_index, batches in tile_batches.items():
        good = []
        for batch in batches:
            scores = torch.as_tensor(batch.pred_score, dtype=torch.float64).flatten().cpu()
            labels = torch.as_tensor(batch.gt_label).flatten().bool().cpu()
            good.append(scores[~labels])
        values = torch.cat(good) if good else torch.empty(0, dtype=torch.float64)
        if values.numel() == 0:
            logger.warning(f"Tile {tile_index}: no good validation tile to scale it by; its scores are used unscaled.")
            scaling[tile_key(tile_index)] = {"mean": 0.0, "std": 1.0, "good": 0}
            continue
        std = float(values.std(unbiased=False)) if values.numel() > 1 else 0.0
        scaling[tile_key(tile_index)] = {"mean": float(values.mean()), "std": max(std, MIN_SPREAD), "good": int(values.numel())}
    return scaling


def save_tile_scaling(directory: Path, score_merge: str, scaling: Optional[Dict[str, Dict[str, float]]]) -> Path:
    """Write <directory>/tile_scaling.json: the merge the model is calibrated
    with, and (for max) every tile position's scaling. Written for "mean" too,
    so that a scaling left by an earlier calibration is not taken for this
    one's."""
    path = Path(directory) / TILE_SCALING_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"score_merge": score_merge, "tiles": scaling or {}}, f, ensure_ascii=False, indent=2)
    return path


def load_tile_scaling(directory: Optional[Path]) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """The merge a model was calibrated with, and its tile scaling, from
    <directory>/tile_scaling.json. ("mean", {}) if there is no such file -- a
    model calibrated before the max merge existed."""
    path = Path(directory) / TILE_SCALING_FILENAME if directory is not None else None
    if path is None or not path.is_file():
        return MERGE_MEAN, {}
    with path.open("r", encoding="utf-8") as f:
        saved = json.load(f)
    return saved.get("score_merge", MERGE_MEAN), saved.get("tiles", {})


def merge_scores(batch_data: Dict[Tuple[int, int], Any], score_merge: str, scaling: Dict[str, Dict[str, float]]) -> Tensor:
    """The image scores of one batch from its tiles' ``pred_score``: the
    highest scaled tile score (``max``) or the mean (``mean``). A tile position
    missing from ``scaling`` is used unscaled."""
    scores = [torch.as_tensor(tile.pred_score, dtype=torch.float32).flatten().cpu() for tile in batch_data.values()]
    if score_merge != MERGE_MAX:
        return torch.stack(scores).mean(dim=0)
    scaled = []
    for tile_index, score in zip(batch_data.keys(), scores):
        fit = scaling.get(tile_key(tile_index), {"mean": 0.0, "std": 1.0})
        scaled.append((score - fit["mean"]) / fit["std"])
    return torch.stack(scaled).max(dim=0).values


def miss_rate_threshold(scores: Tensor, labels: Tensor, max_miss_rate: float) -> Optional[float]:
    """The highest image threshold (anomalous = score >= threshold) that misses
    at most ``max_miss_rate`` of the defective images.

    With the defect scores sorted a_1 <= ... <= a_n and k = floor(m * n) misses
    allowed, the cut is a_(k+1): exactly the k lowest defects fall below it.
    The threshold is put halfway between a_(k+1) and the next lower score of
    any image, which keeps every validation verdict but leaves a margin for a
    slightly weaker defect. None if there is no defective image."""
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
    labels = torch.as_tensor(labels).flatten().bool()
    defects = torch.sort(scores[labels]).values
    if defects.numel() == 0:
        return None
    allowed = math.floor(max_miss_rate * defects.numel() + 1e-9)
    cut = float(defects[min(allowed, defects.numel() - 1)])
    below = scores[scores < cut]
    return cut if below.numel() == 0 else (cut + float(below.max())) / 2
