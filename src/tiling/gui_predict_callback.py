"""
Lightning Callback that reports tiled-ensemble prediction progress to the GUI.

Mirrors gui_training_callback.py's GUITrainingProgressCallback, but for the
predict phase: a tiled-ensemble predict run never calls trainer.fit() (see
InferenceJob.run in tiling/tiled_ensemble.py, which calls engine.predict()
directly), so this reacts to the predict loop's own hooks
(on_predict_start/on_predict_batch_end/on_predict_end) instead of the fit
loop's. Reports through core.command_dispatcher.post_worker_update under
its own "predict_progress" kind -- same channel training progress uses, kept
separate so the GUI doesn't have to guess which run a "progress" update
belongs to. GUI.py's _onPredictProgress is what consumes it.
"""

from typing import TYPE_CHECKING, Any, Dict

from lightning.pytorch.callbacks import Callback

from core.command_dispatcher import post_worker_update

if TYPE_CHECKING:
    import lightning.pytorch as pl


class GUIPredictProgressCallback(Callback):
    """Reports per-batch/per-tile prediction progress via
    post_worker_update("predict_progress", ...).

    Tiled-ensemble prediction builds a fresh Trainer per tile (one
    InferenceJob per tile location -- see InferenceJobGenerator.generate_jobs
    in tiling/tiled_ensemble.py), but every tile's InferenceJob is handed the
    same trainer_args dict for the whole predict run (InferenceTiledEnsemble.run
    builds it once via _pipeline_trainer_kwargs and reuses it across every
    tile), so this callback instance -- like GUITrainingProgressCallback's --
    passes through unchanged tile to tile. That's what lets it track tile
    count across the whole run instead of resetting itself every tile.

    total_tiles doesn't need to be set by hand: it's not knowable at
    YAML-parse time (this callback is instantiated from the inferencer
    config alone), but InferenceJobGenerator does know it by the time it
    builds each tile's InferenceJob -- ensemble_engine.py registers a
    GUIPredictProgressCallback adjuster (see
    _adjust_gui_predict_progress_callback) that sets it from the tiler's
    real tile count automatically, in place, before every tile predicts.

    Parameters
    ----------
    total_tiles : int, optional
        How many tiles this run will predict, in total. Defaults to 1 (a
        non-tiled run); tiled-ensemble runs get this overwritten
        automatically -- see class docstring above.
    """

    def __init__(self, total_tiles: int = 1) -> None:
        super().__init__()
        if total_tiles < 1:
            raise ValueError(f"total_tiles must be >= 1, got {total_tiles}")
        self.total_tiles = total_tiles
        self.tiles_completed = 0
        self._total_batches = 1

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """A new trainer.predict() call means a new tile has started."""
        num_batches = getattr(trainer, "num_predict_batches", None)
        if isinstance(num_batches, (list, tuple)):
            num_batches = num_batches[0] if num_batches else None
        self._total_batches = int(num_batches) if num_batches else 1
        self._report(batch=0, batch_progress=0.0)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """0-100% for the current tile is relative to its total predict
        batch count (established in on_predict_start)."""
        batch_progress = min((batch_idx + 1) / self._total_batches, 1.0)
        self._report(batch=batch_idx + 1, batch_progress=batch_progress)

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """This tile's predict pass is done -- always counts as fully
        complete before the next tile's on_predict_start.

        Reports *before* incrementing tiles_completed, for the same reason
        GUITrainingProgressCallback.on_fit_end does: _report's own formula
        adds batch_progress on top of tiles_completed to get
        global_progress, so tiles_completed here must still mean "tiles
        finished before this one" -- bumping it first would double-count
        the tile that just finished.
        """
        self._report(batch=self._total_batches, batch_progress=1.0)
        self.tiles_completed = min(self.tiles_completed + 1, self.total_tiles)

    def _report(self, batch: int, batch_progress: float) -> None:
        current_tile = min(self.tiles_completed + 1, self.total_tiles)
        global_progress = min((self.tiles_completed + batch_progress) / self.total_tiles, 1.0)
        payload: Dict[str, Any] = {
            "tile": current_tile,
            "total_tiles": self.total_tiles,
            "batch": batch,
            "total_batches": self._total_batches,
            "tile_progress": batch_progress,
            "global_progress": global_progress,
        }
        post_worker_update("predict_progress", payload)
