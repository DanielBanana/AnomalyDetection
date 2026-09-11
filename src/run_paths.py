"""
Canonical, manager-owned conventions for where a tiled-ensemble training run's artifacts
live on disk, and how its per-tile identifiers are named.

These live in their own module rather than on ``AnomalyDetectionManager`` in manager.py:
manager.py imports from ``tiling.tiled_ensemble``, which imports from
``tiling.ensemble_engine`` - if ensemble_engine.py imported these conventions back from
manager.py, that would be a circular import. Keeping them here lets manager.py, the tiled
ensemble pipeline, and the engine all depend on one shared source of truth instead of each
re-deriving (or hardcoding) the same paths/names independently.
"""

from pathlib import Path
from typing import Optional

def resolve_output_dir(
    baseOutputDir: Path,
    datasetName: str,
    modelName: str,
    runId: str,
    category: Optional[str] = None,
    tiling: bool = True,
    runKind: str = "training",
) -> Path:
    """
    Where new artifacts for a run should be written:
    base/dataset/[category]/model/[tiled]/<runKind>/runs/<runId>

    Parameters
    ----------
    baseOutputDir : Path
        Base of the output directories usually .../results
    datasetName : str
        Name of the dataset to save the results for
    modelName : str
        Name of the model to save the results for
    category : Optional[str] (optional)
        Name of the category in the dataset the results are fore. Default is `None`
    tiling : bool (optional)
        Is a TiledEnsemble model used? I.e. is there a model for each tile being trained because the images are too large. Default is `True`
    runKind : str (optional)
        "training" (train()/eval() -- anything that produces or refines a
        checkpoint) or "inference" (a predict-only run against an already
        trained checkpoint, e.g. shift_inspect/product_inference) -- keeps
        trained-model runs and one-off prediction runs apart on disk instead
        of mixing them under the same runs/ folder. Default "training".

    Returns
    -------
    _name_ : Path
        The path where results are stored
    """
    if runKind not in ("training", "inference"):
        raise ValueError(f"runKind must be 'training' or 'inference', got {runKind!r}")
    path = baseOutputDir / datasetName
    if category is not None:
        path = path / category
    path = path / modelName
    if tiling:
        path = path / "tiled"
    path = path / runKind
    return path / "runs" / runId

def resolve_checkpoint_paths(trainingDir: Path) -> Path:
    """Where checkpoints live within a given (already-resolved) training run directory."""
    return trainingDir / "checkpoints"


def resolve_stats_path(trainingDir: Path) -> Path:
    """Where stats.json (calibrated anomaly thresholds, written by eval()) lives
    within a given (already-resolved) training run directory."""
    return trainingDir / "stats.json"


def resolve_calibration_scores_path(trainingDir: Path) -> Path:
    """Where calibration_scores.json (per-sample anomaly scores + ground-truth
    labels from the val/test split, written by AD_Worker._handle_calibrate)
    lives within a given (already-resolved) training run directory --
    alongside stats.json, which only keeps the two threshold scalars. This is
    what lets the GUI's score-distribution histogram be redrawn later from
    disk without rerunning calibration."""
    return trainingDir / "calibration_scores.json"


def resolve_wandb_manifest_dir(trainingDir: Path) -> Path:
    """Where each tile's W&B run-id manifest lives within a given training run directory."""
    return trainingDir / "wandb_runs"


def run_id_from_training_dir(trainingDir: Path) -> str:
    """
    The run id for a given (already-resolved) training run directory.

    ``AnomalyDetectionManager.resolve_output_dir`` names every run directory after its
    run id (``.../runs/<runId>``), so ``trainingDir.name`` already *is* the run id - this
    makes that contract explicit and gives other modules one place to depend on instead
    of re-deriving it themselves.
    """
    return trainingDir.name


def tile_checkpoint_filename(tile_index: tuple[int, int]) -> str:
    """Canonical per-tile checkpoint filename (without suffix), used across the tiled ensemble."""
    i, j = tile_index
    return f"model{i}_{j}"


def tile_wandb_run_id(run_id: str, tile_index: tuple[int, int]) -> str:
    """Canonical, deterministic per-tile W&B run id, grouped under ``run_id``."""
    return f"{run_id}_{tile_checkpoint_filename(tile_index)}"
