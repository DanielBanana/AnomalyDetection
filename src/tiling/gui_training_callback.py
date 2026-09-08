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
_onTrainProgress is what consumes it (two progress bars: tiles overall, and
epochs within whichever tile is currently training).
"""

from typing import TYPE_CHECKING, Any, Dict

from lightning.pytorch.callbacks import Callback

from core.command_dispatcher import post_worker_update

if TYPE_CHECKING:
    import lightning.pytorch as pl


class GUITrainingProgressCallback(Callback):
    """Reports per-epoch/per-tile training progress via
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

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """A new trainer.fit() call means a new tile has started."""
        self._report(trainer, tile_progress=0.0)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """0-100% for the current tile is relative to its max_epochs --
        EarlyStopping just means fewer of these calls happen before
        on_fit_end, not a different denominator."""
        max_epochs = trainer.max_epochs or 1
        tile_progress = min((trainer.current_epoch + 1) / max_epochs, 1.0)
        self._report(trainer, tile_progress=tile_progress)

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
        self._report(trainer, tile_progress=1.0)
        self.tiles_completed = min(self.tiles_completed + 1, self.total_tiles)

    def _report(self, trainer: "pl.Trainer", tile_progress: float) -> None:
        current_tile = min(self.tiles_completed + 1, self.total_tiles)
        global_progress = min((self.tiles_completed + tile_progress) / self.total_tiles, 1.0)
        payload: Dict[str, Any] = {
            "tile": current_tile,
            "total_tiles": self.total_tiles,
            "epoch": trainer.current_epoch + 1,
            "max_epochs": trainer.max_epochs,
            "tile_progress": tile_progress,
            "global_progress": global_progress,
        }
        post_worker_update("progress", payload)
