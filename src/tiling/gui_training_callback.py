"""
Lightning Callback that reports tiled-ensemble training progress to the GUI.

Doesn't touch or replace the console's own TQDMProgressBar -- this is a
second, independent reporter hooked into the same Lightning training loop,
alongside whatever other callbacks (EarlyStopping, ModelCheckpoint, ...)
the trainer config already lists. Reports through
core.command_dispatcher.post_worker_update, the same channel every other
worker-to-GUI fact in this app goes through (src/ is on sys.path
alongside submodule_AnomalyDetection/src, same as every other cross-package
import in this app -- see gui_main.py/cli_main.py's own sys.path setup); GUI.py's
_onTrainProgress is what consumes it (three progress bars: tiles overall,
epochs within whichever tile is currently training, and batches within the
epoch that is running).
"""

import time
from typing import TYPE_CHECKING, Any, Dict

from lightning.pytorch.callbacks import Callback

from core.command_dispatcher import post_worker_update

if TYPE_CHECKING:
    import lightning.pytorch as pl


class GUITrainingProgressCallback(Callback):
    """Reports per-batch/per-epoch/per-tile training progress via
    post_worker_update("progress", ...).

    Tiled-ensemble training builds a fresh Trainer per tile (see
    AOITiledEnsembleEngine._setup_anomalib_callbacks in
    tiling/ensemble_engine.py), but that method only replaces callback
    *types* it has a registered adjuster for -- everything else, this
    callback included, passes through unchanged as the same instance on
    every tile. That's what lets it track tile count across the whole run
    instead of resetting itself every tile.

    total_tiles doesn't need to be set by hand for a tiled-ensemble run:
    it's not knowable at YAML-parse time (this callback is instantiated
    from the trainer config alone, well before anything reads the tiling
    config), but AOITiledEnsembleEngine *does* know it by the time it
    actually builds each tile's Trainer -- ensemble_engine.py registers a
    GUITrainingProgressCallback adjuster (see _adjust_gui_progress_callback)
    that sets it from the tiler's real tile count automatically, in place,
    before every tile trains. The init_arg below only matters for a
    non-tiled (single-model) run, where total_tiles=1 is simply correct
    and there's no engine to override it.

    Parameters
    ----------
    total_tiles : int, optional
        How many tiles this run will train, in total. Defaults to 1 (a
        non-tiled run); tiled-ensemble runs get this overwritten
        automatically -- see class docstring above.
    """

    def __init__(self, total_tiles: int = 1) -> None:
        super().__init__()
        if total_tiles < 1:
            raise ValueError(f"total_tiles must be >= 1, got {total_tiles}")
        self.total_tiles = total_tiles
        self.tiles_completed = 0
        # When the last per-batch report went out (time.monotonic()), see
        # on_train_batch_end.
        self._lastBatchReport = 0.0

    # At most this many per-batch reports a second: an epoch of many fast
    # batches would otherwise fill the GUI's update queue faster than it
    # is drawn. The last batch of an epoch is always reported.
    MIN_BATCH_REPORT_INTERVAL = 0.2

    @staticmethod
    def _totalBatches(trainer: "pl.Trainer") -> int:
        """Batches in one training epoch; 0 if Lightning does not know
        (an iterable dataset without a length reports infinity)."""
        total = getattr(trainer, "num_training_batches", 0)
        return int(total) if isinstance(total, (int, float)) and 0 < total < float("inf") else 0

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """A new trainer.fit() call means a new tile has started."""
        self._report(trainer, tile_progress=0.0, batch=0)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        """How far the running epoch is. The part of the epoch that is
        done also counts towards the tile's (and so the overall) progress,
        so those bars move during a long epoch instead of standing still
        until it ends."""
        total = self._totalBatches(trainer)
        done = batch_idx + 1
        now = time.monotonic()
        if done < total and now - self._lastBatchReport < self.MIN_BATCH_REPORT_INTERVAL:
            return
        self._lastBatchReport = now
        max_epochs = trainer.max_epochs or 1
        epoch_progress = min(done / total, 1.0) if total else 0.0
        tile_progress = min((trainer.current_epoch + epoch_progress) / max_epochs, 1.0)
        self._report(trainer, tile_progress=tile_progress, batch=done)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """0-100% for the current tile is relative to its max_epochs --
        EarlyStopping just means fewer of these calls happen before
        on_fit_end, not a different denominator."""
        max_epochs = trainer.max_epochs or 1
        tile_progress = min((trainer.current_epoch + 1) / max_epochs, 1.0)
        self._report(trainer, tile_progress=tile_progress, batch=self._totalBatches(trainer))

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Reached max_epochs or EarlyStopping cut it short -- either way
        this tile is done, so it always counts as fully complete before
        the next tile's on_fit_start.

        Reports *before* incrementing tiles_completed: _report's own
        formula adds tile_progress on top of tiles_completed to get
        global_progress, so tiles_completed here must still mean "tiles
        finished before this one" -- bumping it first would double-count
        the tile that just finished (it'd be both +1 in tiles_completed
        *and* contribute tile_progress=1.0 on top of that).
        """
        self._report(trainer, tile_progress=1.0, batch=self._totalBatches(trainer))
        self.tiles_completed = min(self.tiles_completed + 1, self.total_tiles)

    def _report(self, trainer: "pl.Trainer", tile_progress: float, batch: int) -> None:
        """`batch` is how many batches of the running epoch are done;
        epoch_progress is that as a share of the epoch's batches (0.0 if
        their number is not known)."""
        current_tile = min(self.tiles_completed + 1, self.total_tiles)
        global_progress = min((self.tiles_completed + tile_progress) / self.total_tiles, 1.0)
        total_batches = self._totalBatches(trainer)
        payload: Dict[str, Any] = {
            "tile": current_tile,
            "total_tiles": self.total_tiles,
            "epoch": trainer.current_epoch + 1,
            "max_epochs": trainer.max_epochs,
            "batch": batch,
            "total_batches": total_batches,
            "epoch_progress": min(batch / total_batches, 1.0) if total_batches else 0.0,
            "tile_progress": tile_progress,
            "global_progress": global_progress,
        }
        post_worker_update("progress", payload)
