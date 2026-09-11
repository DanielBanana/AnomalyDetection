"""
run_registry.py

Utilities for treating each training attempt as its own identifiable, inspectable,
and reproducible "run" on disk, rather than overwriting the same checkpoint/config
location every time a model is retrained.

Expected directory layout produced by these helpers:

    results/<dataset>/<category>/<model>/tiled/runs/<run_id>/
        manifest.yaml           # resolved config snapshot + config_hash + timestamp
        metrics.yaml            # written by eval(), if/when it runs (optional)
        configs/
            Models/<model>.yaml
            Trainer/<trainer>.yaml
            Tiling/<tiling>.yaml
            Engine/...
        checkpoints/
            model0_0.ckpt
            model0_1.ckpt
            ...

Nothing in this file talks to fiftyone, wandb, or anomalib's Engine directly -
it only deals with paths, YAML, and config objects, so it can be unit tested and
reused (e.g. from a notebook or a future web backend) without pulling in the
full training stack.
"""

from __future__ import annotations

import logging
import csv
import hashlib
import json
import math
import re
import shutil
import uuid
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid a hard import-time dependency / circular import; these are only
    # needed for type hints.
    from setup import (
        ModelConfig,
        TrainerConfig,
        DataModuleConfig,
        TilingPipelineConfig,
        DatasetSession,
    )

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Run identity
# --------------------------------------------------------------------------- #

def generate_run_id(label: Optional[str] = None) -> str:
    """Sortable, human-readable, always-unique run identifier.

    Format: YYYYMMDD-HHMMSS[_label]_<8 hex chars>
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    if label:
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
        return f"{timestamp}_{safe_label}_{short_id}"
    return f"{timestamp}_{short_id}"


def hash_config(config_dict: Dict[str, Any]) -> str:
    """Stable short hash of an effective config dict, for detecting whether
    an identical configuration has already been run.
    """
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Path resolution (run-aware)
# --------------------------------------------------------------------------- #

def resolve_output_dir(
    baseOutputDir: Path,
    datasetName: str,
    modelName: str,
    runId: str,
    category: Optional[str] = None,
    tiling: bool = True,
) -> Path:
    """Where a specific run's artifacts should be written:
    base/dataset/[category]/model/[tiled]/runs/run_id
    """
    path = baseOutputDir / datasetName
    if category is not None:
        path = path / category
    path = path / modelName
    if tiling:
        path = path / "tiled"
    return path / "runs" / runId


def resolve_checkpoint_paths(
    runDir: Path,
    ckptFileName: str = "best",
    ckptSuffix: str = ".ckpt",
) -> Tuple[Path, Path]:
    """Checkpoint directory/path within a given (already-resolved) run directory.

    `ckptPath` is only meaningful for a non-tiled model (single checkpoint file).
    For tiled models, use `ckptDir` together with `check_tiled_checkpoints_exist`.
    """
    ckptDir = runDir / "checkpoints"
    ckptPath = ckptDir / (ckptFileName + ckptSuffix)
    return ckptDir, ckptPath


# --------------------------------------------------------------------------- #
# Config file copying (single source of truth for "which files were used")
# --------------------------------------------------------------------------- #

@dataclass
class RunConfigFiles:
    """The set of on-disk config files actually used for a run, anchored to
    `configDir`. `copy_to` mirrors each file's path *relative to configDir*
    into `runDir/configs/...`, reusing whatever subfolder structure the
    configs already live in instead of guessing a subfolder per kind of file.
    This is also exactly the layout `reproduce_run` expects to find.
    """
    configDir: Path
    modelConfigPath: Optional[Path] = None
    trainerConfigPath: Optional[Path] = None
    tilingConfigPath: Optional[Path] = None
    inferencerConfigPath: Optional[Path] = None
    preProcessorPath: Optional[Path] = None
    postProcessorPath: Optional[Path] = None
    evaluatorPath: Optional[Path] = None

    def _paths(self) -> List[Path]:
        return [p for p in (
            self.modelConfigPath,
            self.trainerConfigPath,
            self.tilingConfigPath,
            self.inferencerConfigPath,
            self.preProcessorPath,
            self.postProcessorPath,
            self.evaluatorPath,
        ) if p is not None]

    def copy_to(self, runDir: Path) -> List[Path]:
        """Copy every configured file into `runDir/configs`, mirroring its
        path relative to `configDir` -- e.g. configs/Models/InpFormer.yaml,
        configs/Tiling/TiledEnsemble.yaml, configs/Trainer/Training_
        InpFormer.yaml, configs/Engine/PreProcessor.yaml, reusing whatever
        subfolder structure the configs already live in. Only files truly
        outside `configDir` fall back to just their filename.

        Both `configDir` and each path are resolved (made absolute,
        symlinks/".." collapsed) before computing the relative path -- a
        caller passing one of them relative and the other absolute (e.g.
        Manager.configDir's own default is the relative Path("configs"),
        while resolve_product_config_path always returns an absolute,
        already-`.resolve()`d path) would otherwise make relative_to raise
        ValueError for every single file and silently flatten the whole
        configs/ tree, which is exactly the bug this guards against.

        Missing files (a stale path pointing at something since moved or
        deleted) are skipped with a warning rather than raising -- losing
        one archived config shouldn't abort an otherwise-successful run.
        """
        logger.info(f"Copying files to configs folder of run @ {runDir}")
        configsDir = runDir / "configs"
        resolvedConfigDir = self.configDir.resolve()
        destinations: List[Path] = []
        for path in self._paths():
            if not path.exists():
                logger.warning(f"Config file {path} no longer exists; skipping its archived copy for this run.")
                continue
            resolvedPath = path.resolve()
            try:
                relative = resolvedPath.relative_to(resolvedConfigDir)
            except ValueError:
                relative = Path(path.name)
            destination = configsDir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolvedPath, destination)
            destinations.append(destination)
        return destinations


# --------------------------------------------------------------------------- #
# Checkpoint copying (so a run that only *reads* a model still carries its
# own copy of the exact weights it used)
# --------------------------------------------------------------------------- #

def copy_checkpoints(srcCkptDir: Path, runDir: Path) -> Path:
    """Copy every checkpoint file from `srcCkptDir` into `runDir/checkpoints`.

    Used by eval() to pull the checkpoints being evaluated (from the training
    run they came from) into the evaluation run's own directory, so the run
    that determined a threshold stays self-contained and inspectable even if
    the original training run is later deleted or retrained over.

    No-op (destination created but left empty) if `srcCkptDir` doesn't exist.
    Returns the destination checkpoints directory.
    """
    destDir = runDir / "checkpoints"
    destDir.mkdir(parents=True, exist_ok=True)
    if not srcCkptDir.exists():
        logger.warning(f"No checkpoint directory to copy at {srcCkptDir}; skipping.")
        return destDir

    logger.info(f"Copying checkpoints from {srcCkptDir} to {destDir}")
    print(f"Copying checkpoints from {srcCkptDir} to {destDir}")
    for ckptFile in srcCkptDir.glob("*.ckpt"):
        shutil.copy2(ckptFile, destDir / ckptFile.name)
    return destDir


# --------------------------------------------------------------------------- #
# Tiled checkpoint completeness
# --------------------------------------------------------------------------- #

_CKPT_NAME_RE = re.compile(r"^model(\d+)_(\d+)\.ckpt$")


def compute_tile_grid(
    image_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> Tuple[int, int]:
    """Number of (rows, cols) tiles produced for a given image size.

    NOTE: verify this formula against anomalib's Tiler implementation for your
    installed version; this is the single place that assumption lives so it
    only needs correcting in one spot if it diverges.
    """
    n_rows = math.ceil((image_size[0] - tile_size[0]) / stride[0]) + 1
    n_cols = math.ceil((image_size[1] - tile_size[1]) / stride[1]) + 1
    return n_rows, n_cols


def expected_tiled_checkpoint_paths(
    ckptDir: Path,
    tilingPipelineConfig: "TilingPipelineConfig",
) -> List[Path]:
    """All checkpoint file paths expected for a fully trained tiled ensemble."""
    n_rows, n_cols = compute_tile_grid(
        tilingPipelineConfig.image_size,
        tilingPipelineConfig.tile_size,
        tilingPipelineConfig.stride,
    )
    return [ckptDir / f"model{r}_{c}.ckpt" for r in range(n_rows) for c in range(n_cols)]


def check_tiled_checkpoints_exist(
    ckptDir: Path,
    tilingPipelineConfig: "TilingPipelineConfig",
) -> Tuple[bool, List[Path]]:
    """Returns (all_present, missing_paths). Empty missing_paths means fully trained."""
    expected = expected_tiled_checkpoint_paths(ckptDir, tilingPipelineConfig)
    if not ckptDir.exists():
        return False, expected
    missing = [p for p in expected if not p.exists()]
    return (len(missing) == 0), missing


def find_unexpected_checkpoint_files(
    ckptDir: Path,
    tilingPipelineConfig: "TilingPipelineConfig",
) -> List[Path]:
    """Checkpoint-looking files present that fall outside the current config's
    tile grid - usually leftovers from a differently-shaped tiling config that
    previously wrote into the same directory.
    """
    if not ckptDir.exists():
        return []
    n_rows, n_cols = compute_tile_grid(
        tilingPipelineConfig.image_size,
        tilingPipelineConfig.tile_size,
        tilingPipelineConfig.stride,
    )
    unexpected = []
    for f in ckptDir.glob("model*_*.ckpt"):
        m = _CKPT_NAME_RE.match(f.name)
        if not m or int(m.group(1)) >= n_rows or int(m.group(2)) >= n_cols:
            unexpected.append(f)
    return unexpected


# --------------------------------------------------------------------------- #
# Effective-config serialization + manifest writing
# --------------------------------------------------------------------------- #

def _safe(value: Any) -> Any:
    """Recursively convert a value into something YAML/JSON-serializable.
    Non-serializable objects (nn.Module instances, callback objects, enums,
    etc.) are recorded by class name + repr only - enough to know what ran,
    not enough to reconstruct the live object. Reconstruction instead relies
    on the raw YAML files copied alongside the manifest.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return {"__repr__": str(value), "__class__": type(value).__name__}


def serialize_effective_config(
    trainerConfig: "TrainerConfig",
    modelConfig: "ModelConfig",
    datamoduleConfig: "DataModuleConfig",
    tilingPipelineConfig: "TilingPipelineConfig",
    datasetSession: "DatasetSession",
) -> Dict[str, Any]:
    """Flatten the current, fully-resolved settings into a YAML/JSON-safe dict.
    This is what actually ran - independent of whether the source YAML files
    or code defaults change afterwards.
    """
    return {
        "trainer": _safe(trainerConfig.to_kwargs()),
        "model": {
            "name": modelConfig.name,
            # "configPath": modelConfig,
            # "config": _safe(modelConfig.to_dict()),
            "preProcessorPath": _safe(modelConfig.preProcessorPath),
            "postProcessorPath": _safe(modelConfig.postProcessorPath),
            "evaluatorPath": _safe(modelConfig.evaluatorPath),
        },
        "datamodule": _safe(datamoduleConfig.to_dict()),
        "tiling": _safe(tilingPipelineConfig.to_dict()),
        "dataset": {
            "name": datasetSession.datasetName,
            "category": datasetSession.category,
        },
    }


def write_run_manifest(
    runDir: Path,
    effective_config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Writes manifest.yaml capturing the resolved config, its hash, and a
    timestamp. Returns the manifest path.
    """
    manifest:Dict[str,Any] = {
        "config": effective_config,
        "config_hash": hash_config(effective_config),
        "created_at": datetime.now().isoformat(),
        **(extra or {}),
    }
    runDir.mkdir(parents=True, exist_ok=True)
    manifestPath = runDir / "manifest.yaml"
    with open(manifestPath, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return manifestPath


def write_run_metrics(runDir: Path, metrics: Dict[str, float]) -> Path:
    """Writes/overwrites metrics.yaml for a run (called after eval())."""
    metricsPath = runDir / "metrics.yaml"
    with open(metricsPath, "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)
    return metricsPath


def read_run_metrics(runDir: Path) -> Dict[str, float]:
    """Reads the metrics recorded for a run. The tiled-ensemble eval pipeline
    (`tiling/jobs.py`'s MetricsJob) writes a single-row `metric_results.csv`
    (header = metric names) into the run directory; `metrics.yaml` is an
    older/alternate format some callers may still write. csv wins if both
    are present, since it's what the actual pipeline produces.

    Prefer metric_results_test.csv over metric_results_val.csv and
    metric_results.csv, and prefer metric_results_val.csv over metric_results.csv.
    """
    for csv_name in ["metric_results_test.csv", "metric_results_val.csv", "metric_results.csv"]:
        csvPath = runDir / csv_name
        if csvPath.exists():
            with open(csvPath, newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return {}
            return {k: float(v) for k, v in rows[0].items()}

    yamlPath = runDir / "metrics.yaml"
    if yamlPath.exists():
        with open(yamlPath) as f:
            return yaml.safe_load(f) or {}

    return {}


def find_existing_runs_with_config(baseOutputDir: Path, config_hash: str) -> List[Path]:
    """Scan for prior runs whose manifest has this exact config hash."""
    matches:List[Path] = []
    for manifestPath in baseOutputDir.glob("**/runs/*/manifest.yaml"):
        try:
            with open(manifestPath) as f:
                manifest = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not read manifest {manifestPath}: {e}")
            continue
        if manifest and manifest.get("config_hash") == config_hash:
            matches.append(manifestPath.parent)
    return matches


# --------------------------------------------------------------------------- #
# RunSummary / browsing
# --------------------------------------------------------------------------- #

@dataclass
class RunSummary:
    """A read-only, disk-derived summary of a single training run - enough to
    list, filter, and compare runs without loading models or datasets.
    """
    runId: str
    runDir: Path
    datasetName: str
    category: Optional[str]
    modelName: str
    createdAt: str
    configHash: str
    isComplete: bool
    missingCheckpoints: List[Path] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["runDir"] = str(self.runDir)
        d["missingCheckpoints"] = [str(p) for p in self.missingCheckpoints]
        return d

    def __str__(self) -> str:
        status = "complete" if self.isComplete else f"INCOMPLETE ({len(self.missingCheckpoints)} missing)"
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items()) if self.metrics else "no metrics"
        return (
            f"[{self.runId}] {self.modelName} on {self.datasetName}"
            f"{f'/{self.category}' if self.category else ''} "
            f"- {status} - {metrics_str} - {self.createdAt}"
        )


def _tiling_config_from_manifest_dict(tiling_dict: Dict[str, Any]) -> "TilingPipelineConfig":
    """Rebuild a TilingPipelineConfig from the manifest's serialized tiling
    section. Imported lazily to avoid a hard dependency at module import time.
    """
    from setup import TilingPipelineConfig  # local import: see module docstring
    return TilingPipelineConfig.extract_tiling_pipeline_params({"tiling": tiling_dict})


def list_runs(
    baseOutputDir: Path,
    productName: Optional[str] = None,
    modelName: Optional[str] = None,
    onlyComplete: bool = False,
) -> List[RunSummary]:
    """Scan baseOutputDir for all runs (via their manifest.yaml), optionally
    filtered by product/category and/or model name.

    Returns summaries sorted newest-first.
    """
    summaries: List[RunSummary] = []

    files = baseOutputDir.glob("**/runs/*/manifest.yaml")

    for manifestPath in files:
        print(f"Checking {manifestPath}")
        logger.debug(f"Checking {manifestPath}")
        runDir = manifestPath.parent
        try:
            with open(manifestPath) as f:
                manifest = yaml.safe_load(f)
        except Exception as e:
            print(f"Skipping unreadable manifest {manifestPath}: {e}")
            logger.warning(f"Skipping unreadable manifest {manifestPath}: {e}")
            continue

        cfg = manifest.get("config", {})
        dataset_cfg = cfg.get("dataset", {})
        model_cfg = cfg.get("model", {})
        _category = dataset_cfg.get("category")
        if isinstance(_category, list):
            category = _category[0]
        else:
            category = _category
        model_name = model_cfg.get("name", "unknown")

        if productName is not None and category != productName:
            print(f"found category ({category}) != wanted product name ({productName}). Skipping")
            logger.debug(f"found category ({category}) != wanted product name ({productName}). Skipping")
            continue
        if modelName is not None and model_name != modelName:
            print(f"found model name ({model_name}) != wanted model name ({modelName}). Skipping")
            logger.debug(f"found model name ({model_name}) != wanted model name ({modelName}). Skipping")
            continue

        try:
            tilingCfg = _tiling_config_from_manifest_dict(cfg.get("tiling", {}))
            complete, missing = check_tiled_checkpoints_exist(runDir / "checkpoints", tilingCfg)
            print("Checkpoints complete!")
            logger.debug("Checkpoints complete!")
        except Exception as e:
            print(f"Could not evaluate checkpoint completeness for {runDir}: {e}")
            logger.warning(f"Could not evaluate checkpoint completeness for {runDir}: {e}")
            complete, missing = False, []

        try:
            metrics = read_run_metrics(runDir)
            print(f"Found metrics: {metrics}")
            logger.debug(f"Found metrics: {metrics}")
        except Exception as e:
            print(f"Could not read metrics for {runDir}: {e}")
            logger.warning(f"Could not read metrics for {runDir}: {e}")
            metrics = {}

        if onlyComplete and not complete:
            print(f"Wanted only complete tile checkpoints. Some checkpoints are missing ({missing}). Skipping")
            logger.debug(f"Wanted only complete tile checkpoints. Some checkpoints are missing ({missing}). Skipping")
            continue

        summaries.append(RunSummary(
            runId=runDir.name,
            runDir=runDir,
            datasetName=dataset_cfg.get("name", "unknown"),
            category=category,
            modelName=model_name,
            createdAt=manifest.get("created_at", ""),
            configHash=manifest.get("config_hash", ""),
            isComplete=complete,
            missingCheckpoints=missing,
            metrics=metrics,
        ))

    return sorted(summaries, key=lambda r: r.createdAt, reverse=True)


def best_run(
    baseOutputDir: Path,
    productName: Optional[str],
    metric: str,
    mode: str = "max",
    modelName: Optional[str] = None,
) -> Optional[RunSummary]:
    """Convenience wrapper around list_runs: the best complete run for a
    product/category by a given metric. Returns None if no complete run has
    that metric recorded.
    """
    candidates = [
        r for r in list_runs(baseOutputDir, productName=productName, modelName=modelName, onlyComplete=True)
        if metric in r.metrics
    ]
    if not candidates:
        logger.debug(f"No candiates for best run determination found.")
        return None
    key = lambda r: r.metrics[metric]
    return max(candidates, key=key) if mode == "max" else min(candidates, key=key)


DEFAULT_BEST_METRIC = "image_F1Score"


def resolve_run_dir(
    baseOutputDir: Path,
    category: Optional[str],
    modelName: Optional[str],
    selection: str = "latest",
    metric: str = DEFAULT_BEST_METRIC,
) -> Tuple[Path | None, str | None]:
    """
    Pick a single run directory for a product+model, so a `Product` config
    doesn't have to hardcode `trainingDir`/`weights_path` for a run that a
    later retrain would make stale.

    `selection` is `"latest"` (newest complete run) or `"best"` (highest
    `metric` among complete runs, read from each run's `metric_results.csv`).

    Parameters
    ----------
    baseOutputDir : Path
        Output directory for training or inference runs. New runs should go here and here we look for old runs
    category : Optional[str]
        For what kind of product are we training
    modelName : Optional[str]
        Name of the model we are training
    selection : str (optional)
        What kind of selection criteria we are using to find a previous run. Either `latest` or `best`. Default is `"latest"`.
    metric : str (optional)
        What kind of metric ist used to determine best. Default is `DEFAULT_BEST_METRIC`

    Returns
    -------
    _name_ : Path
        Path to the determined run directory.

    Raises
    ------
    _name_ : ValueError
        Selection criterion is neither `latest` nor `best`
    """
    print(f"Trying to find previous run under {baseOutputDir} for the {modelName} model and the {category} product")
    if selection == "latest":
        runs = list_runs(baseOutputDir, productName=category, modelName=modelName, onlyComplete=True)
        if not runs:
            print(f"There has been found no previous latest run for {baseOutputDir}/<datasetName>/{category}/{modelName}. \n If you are training a new model this is assumed to be the case.")
            logger.info(f"There has been found no previous latest run been found for {baseOutputDir}/<datasetName>/{category}/{modelName}. \n If you are training a new model this is assumed to be the case.")
            return None, None
        return runs[0].runDir, runs[0].modelName
    elif selection == "best":
        run = best_run(baseOutputDir, productName=category, metric=metric, mode="max", modelName=modelName)
        if run is None:
            print(f"There has been found no previous best run for {baseOutputDir}/<datasetName>/{category}/{modelName}. \n If you are training a new model this is assumed to be the case.")
            logger.info(f"There has been found no previous best run been found for {baseOutputDir}/<datasetName>/{category}/{modelName}. \n If you are training a new model this is assumed to be the case.")
            return None, None
        return run.runDir, run.modelName
    else:
        raise ValueError(f"selection must be 'latest' or 'best', got {selection!r}")


# --------------------------------------------------------------------------- #
# Reproducing a run
# --------------------------------------------------------------------------- #

def reproduce_run(runDir: Path) -> Dict[str, Any]:
    """Reconstruct the config objects needed to rerun training (or just load
    the run's checkpoint for inference -- see manager.AnomalyDetectionManager.
    loadFromRunDir) exactly as it happened, using the FROZEN config copies
    inside runDir/configs -- never the live/shared config tree, which may
    have changed since the run happened.

    Returns a dict with keys: modelConfig, modelConfigPath, trainerConfig,
    trainerConfigPath, datamoduleConfig, tilingPipelineConfig,
    tilingConfigPath, inferencerConfig, inferencerConfigPath, datasetName,
    category, original_manifest. inferencerConfig/inferencerConfigPath fall
    back to the trainer config/path if no separate inferencer config was
    archived (an older run, or one that never copied one) -- same "loaded
    is better than nothing" precedent as tilingPipelineConfig falling back
    to the manifest's own recorded tiling section below. datasetName/
    category come from the manifest (the only place they're recorded) and
    are None if it has neither.

    Raises FileNotFoundError if the run directory doesn't have a manifest or
    is missing its frozen config copies.
    """
    # Local imports: keep this module importable without the full config stack
    # loaded, for callers that only want path/listing utilities.
    from setup import ModelConfig, TrainerConfig, DataModuleConfig, TilingPipelineConfig

    manifestPath = runDir / "manifest.yaml"
    if not manifestPath.exists():
        raise FileNotFoundError(f"No manifest.yaml in {runDir}; cannot reproduce this run.")
    with open(manifestPath) as f:
        manifest = yaml.safe_load(f)

    frozenConfigDir = runDir / "configs"
    if not frozenConfigDir.exists():
        raise FileNotFoundError(
            f"No frozen configs directory at {frozenConfigDir}; this run cannot be "
            f"reproduced (it was likely created before config copying was enabled)."
        )

    def _first_or_raise(pattern: str, kind: str) -> Path:
        matches = sorted(frozenConfigDir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No frozen {kind} config found under {frozenConfigDir / pattern}")
        if len(matches) > 1:
            logger.warning(f"Multiple {kind} configs found in {frozenConfigDir}; using {matches[0]}")
        return matches[0]

    modelYaml = _first_or_raise("Models/*.yaml", "model")

    # Trainer and inferencer configs both land under configs/Trainer/ (see
    # setup.loadProductFromYaml, which resolves both with subdir="Trainer")
    # -- "Training_*.yaml" is the trainer config by the same naming
    # convention every model's own config uses (configs/Trainer/Training_
    # <model>.yaml); whatever else is archived alongside it is the
    # inferencer config, if one was copied for this run at all.
    trainerCandidates = sorted((frozenConfigDir / "Trainer").glob("*.yaml"))
    if not trainerCandidates:
        raise FileNotFoundError(f"No frozen trainer config found under {frozenConfigDir / 'Trainer'}")
    trainerYaml = next((p for p in trainerCandidates if p.name.startswith("Training_")), trainerCandidates[0])
    inferencerCandidates = [p for p in trainerCandidates if p != trainerYaml]
    inferencerYaml = inferencerCandidates[0] if inferencerCandidates else None

    tilingYaml = frozenConfigDir / "Tiling"
    tilingYamlPath = next(iter(sorted(tilingYaml.glob("*.yaml"))), None) if tilingYaml.exists() else None

    modelConfig = ModelConfig.from_yaml(modelYaml, config_dir=frozenConfigDir)
    trainerConfig = TrainerConfig.load_trainer_config_from_yaml(trainerYaml)
    datamoduleConfig = DataModuleConfig.load_datamodule_config_from_yaml(trainerYaml)
    tilingPipelineConfig = (
        TilingPipelineConfig.load_tiling_pipeline_config_from_yaml(tilingYamlPath)
        if tilingYamlPath is not None
        else _tiling_config_from_manifest_dict(manifest.get("config", {}).get("tiling", {}))
    )
    inferencerConfig = (
        TrainerConfig.load_trainer_config_from_yaml(inferencerYaml) if inferencerYaml is not None else trainerConfig
    )

    datasetSection = manifest.get("config", {}).get("dataset", {})
    datasetName = datasetSection.get("name")
    _category = datasetSection.get("category")
    category = _category[0] if isinstance(_category, list) and _category else _category

    return {
        "modelConfig": modelConfig,
        "modelConfigPath": modelYaml,
        "trainerConfig": trainerConfig,
        "trainerConfigPath": trainerYaml,
        "datamoduleConfig": datamoduleConfig,
        "tilingPipelineConfig": tilingPipelineConfig,
        "tilingConfigPath": tilingYamlPath,
        "inferencerConfig": inferencerConfig,
        "inferencerConfigPath": inferencerYaml if inferencerYaml is not None else trainerYaml,
        "datasetName": datasetName,
        "category": category,
        "original_manifest": manifest,
    }


if __name__ == "__main__":
    # Minimal smoke test / usage example against a results/ tree, if present.
    resultsDir = Path("results")
    if resultsDir.exists():
        runs = list_runs(resultsDir)
        print(f"Found {len(runs)} run(s) under {resultsDir}:")
        for r in runs:
            print(f"  {r}")
    else:
        print(f"No {resultsDir} directory found; nothing to list.")