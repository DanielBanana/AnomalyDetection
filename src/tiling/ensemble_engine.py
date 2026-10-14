"""
Amended version of the TiledEnsembleEngine from anomalib. Changed setup of callbacks and workspace for different file access path.
"""
#
# Created on Fri Feb 06 2026
#
# Copyright (c) 2026 TH Nuernberg - Daniel Pommer
#

# GENERAL
import json
import wandb

from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

# ANOMALIB
from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_engine import TiledEnsembleEngine
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.loggers import AnomalibWandbLogger
from anomalib.utils.path import create_versioned_dir
from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint, EarlyStopping

# OWN CODE
from run_paths import (
    resolve_checkpoint_paths,
    resolve_wandb_manifest_dir,
    run_id_from_training_dir,
    tile_checkpoint_filename,
    tile_wandb_run_id,
)
from tiling.gui_training_callback import GUITrainingProgressCallback
from tiling.gui_predict_callback import GUIPredictProgressCallback

# PYTORCH LIGHTNING
if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import Logger


class TileContext:
    """Everything a per-tile callback/logger adjuster needs to relocate its object.

    Built once per engine from settings manager.py already owns (`default_root_dir` is the
    manager's run directory, threaded down unchanged; the run id and every derived path/name
    are read from run_paths.py, not invented here) - see run_paths.py's docstring for why
    those live in their own module instead of directly on the manager.

    total_tiles comes from the engine (see AOITiledEnsembleEngine.__init__), which itself
    gets it from get_ensemble_engine/TrainModelJob - ultimately the same tiler.num_tiles
    TrainModelJobGenerator.generate_jobs() already computes and logs before the tile loop.
    """

    def __init__(self, tile_index: tuple[int, int], default_root_dir: Path, total_tiles: int = 1) -> None:
        self.tile_index = tile_index
        self.default_root_dir = default_root_dir
        self.total_tiles = total_tiles
        self.tile_id = tile_checkpoint_filename(tile_index)
        self.run_id = run_id_from_training_dir(default_root_dir)
        self.checkpoint_dir = resolve_checkpoint_paths(default_root_dir)
        self.wandb_manifest_dir = resolve_wandb_manifest_dir(default_root_dir)


def _adjust_model_checkpoint(callback: LightningModelCheckpoint, ctx: TileContext) -> "Callback":
    """Clone a user/config-provided ModelCheckpoint with this tile's dirpath/filename.

    Every tile shares the same callbacks battery from the manager's trainerConfig (same
    dirpath/filename as configured) - without this, every tile would try to write to the
    exact same checkpoint file. Every other setting (monitor, mode, save_top_k, ...) is kept
    as configured.
    """
    return ModelCheckpoint(
        dirpath=ctx.checkpoint_dir,
        filename=ctx.tile_id,
        auto_insert_metric_name=False,
        monitor=callback.monitor,
        mode=callback.mode,
        save_top_k=callback.save_top_k,
        save_last=callback.save_last,  # type: ignore[arg-type]  # runtime value is always bool | "link" | None, same constraint as the constructor
        every_n_epochs=callback.every_n_epochs,
    )

def _adjust_early_stopping_checkpoint(callback: EarlyStopping, ctx: TileContext) -> "Callback":
    """Clone a user/config-provided ModelCheckpoint with this tile's dirpath/filename.

    Every tile shares the same callbacks battery from the manager's trainerConfig (same
    dirpath/filename as configured) - without this, every tile would try to write to the
    exact same checkpoint file. Every other setting (monitor, mode, save_top_k, ...) is kept
    as configured.
    """
    return EarlyStopping(
        monitor=callback.monitor,
        min_delta=callback.min_delta,
        patience=callback.patience,
        verbose=callback.verbose,
        mode=callback.mode,
        stopping_threshold=callback.stopping_threshold,
        divergence_threshold=callback.divergence_threshold,
        check_on_train_epoch_end=callback._check_on_train_epoch_end,
        log_rank_zero_only=callback.log_rank_zero_only
    )


def _adjust_gui_progress_callback(callback: GUITrainingProgressCallback, ctx: TileContext) -> "Callback":
    """Set total_tiles on the config-provided callback from the tiler, in place.

    Unlike _adjust_model_checkpoint/_adjust_early_stopping_checkpoint above, this must return
    the SAME instance rather than a clone: the callback tracks tiles_completed cumulatively
    across the whole run (see its own docstring), and handing back a fresh instance every
    tile would reset that counter to 0 each time instead of accumulating it. Mutating
    total_tiles in place is safe here precisely because it's the one piece of tile-grid
    information the callback couldn't have known at YAML-parse time (see
    GUITrainingProgressCallback's docstring) - everything else about it is already correct
    from construction.
    """
    callback.total_tiles = ctx.total_tiles
    # Which tile, and where its run is: for the Loss chart's tile names and the
    # tile_metrics.csv written after the last tile (see GUITrainingProgressCallback).
    callback.tile_index = ctx.tile_index
    callback.run_dir = ctx.default_root_dir
    return callback


def _adjust_gui_predict_progress_callback(callback: GUIPredictProgressCallback, ctx: TileContext) -> "Callback":
    """Set total_tiles on the config-provided callback from the tiler, in place.

    Predict-side counterpart to _adjust_gui_progress_callback above, same
    reasoning: must return the SAME instance, since GUIPredictProgressCallback
    tracks tiles_completed cumulatively across the whole predict run (see its
    own docstring) rather than resetting itself every tile.
    """
    callback.total_tiles = ctx.total_tiles
    return callback


def _adjust_wandb_logger(run_logger: AnomalibWandbLogger, ctx: TileContext) -> "Logger":
    """Relocate a user/config-provided AnomalibWandbLogger to this tile's own W&B run.

    Every tile shares the same logger config (same project/entity, no id/name/group set) -
    without this, every tile's Trainer would log to the *same* W&B run and clobber each
    other's metrics. This id is the only source of truth for which W&B run belongs to which
    tile - nothing downstream (the notebook included) re-derives it from a formula; it's
    written once, here, to a manifest file per tile so callers can just read it back.

    `_setup_anomalib_callbacks` re-runs on every `Engine._setup_trainer` call, including
    when the tiled ensemble's val-predict step *reuses* the same already-trained engine
    (and therefore the same logger object) later, in a separate phase after every tile has
    finished training. If `_experiment` is already attached, this is that re-entry - the
    run is already correctly configured, so return as-is.

    Otherwise this is a genuinely new logger for a new tile. `WandbLogger.experiment` would
    normally lazily call `wandb.init()`, but it also *silently reuses* `wandb.run` if one is
    already active in the process instead of honouring this logger's own `id`/`group` - and
    since tiles train sequentially in the same process (SerialRunner) with no `wandb.finish()`
    in between, every tile after the first would otherwise log into the first tile's run.
    Calling `wandb.init(..., reinit="create_new")` here, ourselves, opens a genuinely separate,
    concurrently-open run and attaches it directly - sidestepping that reuse check entirely,
    without having to finish (and thus permanently lose) any other tile's still-needed run.
    """
    if run_logger._experiment is not None:  # noqa: SLF001
        return run_logger

    wandb_run_id = tile_wandb_run_id(ctx.run_id, ctx.tile_index)

    run_logger._wandb_init["id"] = wandb_run_id  # noqa: SLF001
    run_logger._wandb_init["name"] = wandb_run_id  # noqa: SLF001
    run_logger._wandb_init["group"] = ctx.run_id  # noqa: SLF001
    run_logger._wandb_init["dir"] = str(ctx.default_root_dir)  # noqa: SLF001
    run_logger._wandb_init["reinit"] = "create_new"  # noqa: SLF001
    run_logger._id = wandb_run_id  # noqa: SLF001
    run_logger._save_dir = str(ctx.default_root_dir)  # noqa: SLF001

    run_logger._experiment = wandb.init(**run_logger._wandb_init)  # noqa: SLF001

    ctx.wandb_manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": wandb_run_id,
        "group": ctx.run_id,
        "project": run_logger._wandb_init.get("project"),  # noqa: SLF001
        "entity": run_logger._wandb_init.get("entity"),  # noqa: SLF001
    }
    (ctx.wandb_manifest_dir / f"{ctx.tile_id}.json").write_text(json.dumps(manifest, indent=2))

    return run_logger


# Registry of tile-aware adjusters, keyed by the callback/logger type they apply to. Extend
# this to support a new tile-aware callback/logger type without touching
# _setup_anomalib_callbacks itself - anything without a registered adjuster passes through
# to the trainer unchanged.
#
# Checked against lightning's base ModelCheckpoint, not anomalib's subclass: a yaml-configured
# `class_path: lightning.pytorch.callbacks.ModelCheckpoint` is not an instance of anomalib's
# ModelCheckpoint (a separate subclass), so registering on the anomalib type would silently
# fail to match it.
_CALLBACK_ADJUSTERS: list[tuple[type, Callable[..., Any]]] = [
    (LightningModelCheckpoint, _adjust_model_checkpoint),
    (EarlyStopping, _adjust_early_stopping_checkpoint),
    (GUITrainingProgressCallback, _adjust_gui_progress_callback),
    (GUIPredictProgressCallback, _adjust_gui_predict_progress_callback),
]
_LOGGER_ADJUSTERS: list[tuple[type, Callable[..., Any]]] = [
    (AnomalibWandbLogger, _adjust_wandb_logger),
]


class AOITiledEnsembleEngine(TiledEnsembleEngine):

    def __init__(self, tile_index: tuple[int, int], total_tiles: int = 1, **kwargs) -> None:
        self.total_tiles = total_tiles
        super().__init__(tile_index, **kwargs)

    @staticmethod
    def setup_ensemble_workspace(args: dict, versioned_dir: bool = True) -> Path:
        """Set up the workspace at the beginning of tiled ensemble training.

        Args:
            args (dict): Tiled ensemble config dict.
            versioned_dir (bool, optional): Whether to create a versioned directory.
                Defaults to ``True``.

        Returns:
            Path: path to new workspace root dir
        """

        return create_versioned_dir(args["rootDir"]) if versioned_dir else args["rootDir"] / "latest"

    def _setup_anomalib_callbacks(self) -> None:
        """Make the manager/config-provided callbacks and logger tile-aware.

        The callbacks battery and logger themselves come entirely from the manager's
        trainerConfig (the yaml `callbacks:`/`logger:` keys) - this method's only job is to
        walk whatever was handed down and, for any type with a registered adjuster (see
        `_CALLBACK_ADJUSTERS`/`_LOGGER_ADJUSTERS` above), replace it with a tile-scoped clone.
        Everything else passes through unchanged. A ModelCheckpoint and TimerCallback are
        still guaranteed present (added with defaults if the config didn't include one),
        matching anomalib's own default behaviour.
        """
        ctx = TileContext(self.tile_index, self._cache.args["default_root_dir"], total_tiles=self.total_tiles)

        adjusted_callbacks: list[Callback] = []
        for callback in self._cache.args["callbacks"]:
            adjuster = next((fn for t, fn in _CALLBACK_ADJUSTERS if isinstance(callback, t)), None)
            adjusted_callbacks.append(adjuster(callback, ctx) if adjuster else callback)

        if not any(isinstance(c, LightningModelCheckpoint) for c in adjusted_callbacks):
            adjusted_callbacks.append(_adjust_model_checkpoint(ModelCheckpoint(), ctx))
        if not any(isinstance(c, TimerCallback) for c in adjusted_callbacks):
            adjusted_callbacks.append(TimerCallback())

        self._cache.args["callbacks"] = adjusted_callbacks

        run_logger = self._cache.args.get("logger")
        adjuster = next((fn for t, fn in _LOGGER_ADJUSTERS if isinstance(run_logger, t)), None)
        if adjuster is not None:
            self._cache.args["logger"] = adjuster(run_logger, ctx)
