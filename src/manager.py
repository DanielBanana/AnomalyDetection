"""
Main class for managing the Industrial Anomaly Detection (IAD)
"""
# GENERAL
import os
import yaml
import shutil
import datetime
import logging
import logging.config
import sys
import threading
import warnings
from datetime import datetime
from logging.config import dictConfig
from enum import IntFlag, auto
from pathlib import Path
from typing import Any, Callable, List, Tuple, Optional, Dict
from dataclasses import dataclass
from typing import Optional
from enum import Enum

# ANOMALIB
from anomalib.models.components import AnomalibModule
from anomalib.loggers import AnomalibWandbLogger
from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.visualization import ImageVisualizer

# PYTROCH LIGHTNING
from lightning.pytorch import Callback
# from lightning.pytorch.callbacks import TQDMProgressBar

# OWN FILES
from setup import create_model
from tiling.tiled_ensemble import TrainTiledEnsemble, EvalTiledEnsemble, InferenceTiledEnsemble
from tiling.tilingCheckpoints import checkTiledCheckpointsExist
from run_registry import generate_run_id, serialize_effective_config, write_run_manifest, RunConfigFiles, copy_checkpoints, reproduce_run
from run_paths import resolve_checkpoint_paths, resolve_stats_path, resolve_wandb_manifest_dir, resolve_output_dir
from setup import (
    DataModuleConfig,
    TrainerConfig,
    ModelConfig,
    TilingPipelineConfig,
    Product, loadProductFromYaml,
    DatasetConfig,
    DatasetSession,
    SetupError,
    DatamoduleError,
    DatasetSessionError,
    ModelError,
    TilingPipelineError
)
# Windows WDDM adds ~100-200 KB of C stack per CUDA call, exhausting the 1 MB main
# thread stack that python.exe ships with.  Run GPU-heavy pipelines in a thread that
# has a generous stack so the kernel cannot crash mid-training.
_PIPELINE_STACK_SIZE = 64 * 1024 * 1024  # 64 MB


def _run_in_large_stack(fn: Callable) -> None:
    """Run fn() in a new thread with a 64 MB stack, re-raising any exception."""
    exc: list = []

    def _target():
        try:
            fn()
        except Exception as e:
            exc.append(e)

    prev = threading.stack_size(_PIPELINE_STACK_SIZE)
    try:
        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join()
    finally:
        threading.stack_size(prev)

    if exc:
        raise exc[0]

os.environ["TRUST_REMOTE_CODE"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="openvino.runtime")

# --- pure path resolution, no side effects, independently testable ---



@dataclass
class RunContext:
    runName: str
    outputDir: Path

class ManagerError(Exception):
    """Base class for all AnomalyDetectionManager errors."""

class ManagerReadinessError(ManagerError):
    """Raised when an action is requested before its prerequisites are met.
    Carries `missing` so a frontend can render actionable messages without
    parsing the exception text.
    """
    def __init__(self, message: str, missing: Optional[List[str]] = None):
        super().__init__(message)
        self.missing = missing or []

class NoModelLoadedError(ManagerReadinessError): ...
class NoDatasetLoadedError(ManagerReadinessError): ...
class TilingNotConfiguredError(ManagerReadinessError): ...
class CalibrationNotFoundError(ManagerReadinessError): ...
class CheckpointNotFoundError(ManagerReadinessError): ...
class RunPreparedError(ManagerReadinessError): ...

class ConfigError(ManagerError):
    """Raised for malformed or unresolvable configuration (YAML, paths, etc.)."""
class Action(str, Enum):
    SETUP_TILING = "setup_tiling"
    TRAIN = "train"
    EVAL = "eval"
    INFERENCE = "inference"
class ManagerReadiness(IntFlag):
    """
    An accumulating checklist of prerequisites the Manager has satisfied so
    far (model loaded, dataset loaded, ...), used to decide whether
    train()/eval()/inference()/setupTiling() can run yet. Not mutually
    exclusive -- multiple flags accumulate over time as prerequisites are
    met (see _REQUIREMENTS) -- so this is a different kind of "state" than
    a single-value session/board status (see core.status.HardwareStatus,
    workers.duetboard.duet_control.BoardStatus).

    Parameters
    ----------
    IntFlag : _type_
        Support for integer based flags
    """
    NONE = 0
    MODEL_LOADED = auto()
    DATASET_LOADED = auto()
    TILING_CONFIGURED = auto()
    RUN_PREPARED = auto()      # paths resolved, callbacks/logger set up
    CHECKPOINT_AVAILABLE = auto()
    CALIBRATED = auto()        # stats.json (anomaly thresholds) available -- produced by
                                # eval(), not train(); required before inference() so scores
                                # can actually be normalized/thresholded

# Create the general logger
logger = logging.getLogger(__name__)

# Shared with the per-run debug.log/info.log handlers (see
# AnomalyDetectionManager._attach_run_log_handlers) so run-directory logs are
# formatted the same way as the general ones.
_DETAILED_LOG_FORMAT = "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s"
_DETAILED_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"

# dictConfig() replaces the root logger's handlers wholesale, so calling it
# more than once per process would silently drop file logging for anything
# that happened before the *last* call. AnomalyDetectionManager instances are
# created repeatedly within a single console session (every load_product /
# train_product / inference), so configuration must happen at most once.
_logging_configured = False

def _default_logging_config() -> None:
    """Fallback root logging config used when no logging.yaml is found."""
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(levelname)s: %(message)s"},
            "detailed": {
                "format": _DETAILED_LOG_FORMAT,
                "datefmt": _DETAILED_LOG_DATEFMT,
            },
        },
        "handlers": {
            "stdout": {"class": "logging.StreamHandler", "level": "INFO", "formatter": "simple", "stream": sys.stdout},
            "stderr": {"class": "logging.StreamHandler", "level": "WARNING", "formatter": "simple", "stream": sys.stderr},
            "infoFile": {
                "class": "logging.handlers.RotatingFileHandler", "level": "INFO", "formatter": "detailed",
                "filename": "info.log", "maxBytes": 10000000, "backupCount": 3, "encoding": "utf-8",
            },
            "debugFile": {
                "class": "logging.handlers.RotatingFileHandler", "level": "DEBUG", "formatter": "detailed",
                "filename": "debug.log", "maxBytes": 10000000, "backupCount": 3, "encoding": "utf-8",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["stderr", "stdout", "debugFile", "infoFile"]},
    }
    logging.config.dictConfig(logging_config)

def configure_logging(logDir: Path, configDir: Path, logConfigFile: Optional[Path] = None) -> None:
    """
    Attach the project's file/console log handlers to the root logger.

    Call this once, as early as possible in the process (before any hardware
    or worker threads start logging) -- e.g. from main.py, not lazily from the
    first AnomalyDetectionManager construction, otherwise everything logged
    before that first construction never reaches the log files. Safe to call
    repeatedly; only the first call actually touches the root logger.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    if not os.path.exists(logDir):
        os.makedirs(logDir)

    if logConfigFile is None:
        logConfigFile = configDir / "Logging" / "logging.yaml"

    if not Path.exists(logConfigFile):
        _default_logging_config()
        return

    with open(logConfigFile) as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    handlers: Dict[str, Dict[str, Any]] | None = config.get("handlers", None)
    if handlers is not None:
        for handlerName in handlers.keys():
            filename = handlers[handlerName].get("filename", None)
            if filename is not None:
                config["handlers"][handlerName]["filename"] = logDir / filename
    dictConfig(config=config)



class AnomalyDetectionManager:
    # Each action declares what state it needs. Single source of truth —
    # used both to raise clear errors and to answer "can I do X yet?"
    _REQUIREMENTS: Dict[str, ManagerReadiness] = {
        "setupTiling": ManagerReadiness.MODEL_LOADED,
        "train": ManagerReadiness.MODEL_LOADED | ManagerReadiness.DATASET_LOADED | ManagerReadiness.TILING_CONFIGURED,
        "eval": ManagerReadiness.MODEL_LOADED | ManagerReadiness.DATASET_LOADED | ManagerReadiness.TILING_CONFIGURED | ManagerReadiness.CHECKPOINT_AVAILABLE,
        "inference": ManagerReadiness.MODEL_LOADED | ManagerReadiness.TILING_CONFIGURED | ManagerReadiness.CHECKPOINT_AVAILABLE | ManagerReadiness.CALIBRATED
    }

    _STATE_DESCRIPTIONS: Dict[ManagerReadiness, str] = {
        ManagerReadiness.MODEL_LOADED: "Load a model with generateModel()",
        ManagerReadiness.DATASET_LOADED: "Load a dataset (pass a DatasetSession)",
        ManagerReadiness.TILING_CONFIGURED: "Configure tiling with setupTiling()",
        ManagerReadiness.CHECKPOINT_AVAILABLE: "No checkpoint found — train first or point at an existing training run",
        ManagerReadiness.CALIBRATED: "No thresholds found — run eval() to calibrate before inference",
        ManagerReadiness.RUN_PREPARED: "Internal run preparation failed. Check with program designer" # TODO when does this actually happen, can the user actually do something about it occuring.
    }

    _STATE_ERROR_CLASSES: Dict[ManagerReadiness, type[ManagerReadinessError]] = {
        ManagerReadiness.MODEL_LOADED: NoModelLoadedError,
        ManagerReadiness.DATASET_LOADED: NoDatasetLoadedError,
        ManagerReadiness.TILING_CONFIGURED: TilingNotConfiguredError,
        ManagerReadiness.CHECKPOINT_AVAILABLE: CheckpointNotFoundError,
        ManagerReadiness.CALIBRATED: CalibrationNotFoundError,
        ManagerReadiness.RUN_PREPARED: RunPreparedError
    }

    # Priority order when several things are missing at once — report the
    # earliest step in the pipeline first, since fixing it is usually a
    # prerequisite for the others anyway.
    _STATE_PRIORITY: List[ManagerReadiness] = [
        ManagerReadiness.MODEL_LOADED,
        ManagerReadiness.DATASET_LOADED,
        ManagerReadiness.TILING_CONFIGURED,
        ManagerReadiness.CHECKPOINT_AVAILABLE,
        ManagerReadiness.CALIBRATED,
        ManagerReadiness.RUN_PREPARED
    ]
        
    def __init__(self, 
                 outputDir: Path = Path("results"),
                 configDir: Path = Path("configs"),
                 ) -> None:
        
        self.readiness: ManagerReadiness = ManagerReadiness.NONE

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        self.baseOutputDir:Path = outputDir
        self.outputDir:Path = outputDir
        self.runDir:Path|None = None
        self.configDir:Path = configDir
        self.ckptDir: Optional[Path] = None

        # Extra debug.log/info.log handlers mirroring the root logger into the
        # currently active run's directory, on top of the general log files.
        # See _attach_run_log_handlers / _detach_run_log_handlers.
        self._runLogHandlers: List[logging.Handler] = []

        # modelConfig/tilingPipelineConfig are kept here because they're
        # actively derived into real objects (self.model via generateModel(),
        # tiling setup via setupTiling()) -- not passive copies. Config file
        # *paths* (modelConfigPath, trainerConfigPath, tilingConfigPath,
        # inferencerConfigPath) are deliberately NOT stored on Manager:
        # Product (setup.Product) is their source of truth, and Manager
        # never acted on its own copies of them (see train()'s
        # modelConfigPath/etc. parameters, used only for run archival).
        self.model: Optional[AnomalibModule] = None
        self.modelConfig: Optional[ModelConfig] = None
        self.modelTrainingDir: Optional[Path] = None

        self.tilingPipelineConfig: Optional[TilingPipelineConfig] = None
        self.isTilingSetup = False

        self.datasetSession: Optional[DatasetSession] = None

        # Sibling of outputDir ("logs" next to "results"), not nested inside
        # it -- these are the general, whole-session log files (as opposed
        # to _attach_run_log_handlers' per-run debug.log/info.log, which
        # deliberately do live inside each run's own directory alongside
        # its other artifacts). Computed from outputDir.parent rather than
        # a hardcoded path so a caller that points outputDir somewhere
        # other than the default "results" still gets "logs" alongside it.
        self.logDir: Path = self.outputDir.parent / "logs"
        configure_logging(self.logDir, self.configDir)

    def __repr__(self):
        return "\n".join(f"{key}={value}" for key, value in self.__dict__.items())

    @property
    def wandbManifestDir(self) -> Path:
        """Where per-tile W&B run-id manifests for the current run live (see run_paths.py)."""
        return resolve_wandb_manifest_dir(self.outputDir)

    def attachDatasetSession(self, datasetSession: Optional[DatasetSession] = None) -> DatasetSession:
        """
        Attach a DatasetSession instance to the current manager and set the appropriate state flag.
        If no datasetSession is provided, resolve and return the currently attached one.

        Parameters
        ----------
        datasetSession : Optional[DatasetSession]
            DatasetSession instance with the interesting stats, or None to use the
            already-attached session.

        Returns
        -------
        DatasetSession
            The dataset session attached to the manager.
        """
        if datasetSession is not None:
            self.datasetSession = datasetSession
            self.readiness |= ManagerReadiness.DATASET_LOADED
            logger.info(
                f"Attached dataset '{datasetSession.datasetName}' (category={datasetSession.category[0] if datasetSession.category else None}) to manager."
            )

        if self.datasetSession is None:
            raise NoDatasetLoadedError(
                "No dataset attached. Call attachDatasetSession() or pass datasetSession=...",
                missing=["A dataset attached via attachDatasetSession() or a datasetSession argument"],
            )
        return self.datasetSession

    def now(self):
        """
        Return a format Date and time string for saving things to disk

        Returns
        -------
        dateAndTime : str
            formated date and time string of the current time
        """
        dateAndTime: str = datetime.now().strftime("%Y%m%d-%H%M")
        return dateAndTime

    def _apply_visualizer_output_dir(self, outputDir: Path) -> None:
        """
        Change the output directory for the visualizer according to the current outputDir

        Parameters
        ----------
        outputDir : Path
            The current outputDir where the images with the training results should be returned to
        """
        if self.model is not None and isinstance(self.model.visualizer, ImageVisualizer):
            self.model.visualizer.output_dir = outputDir / "images"
        else:
            logger.info("No model set; cannot adjust visualizer output_dir.")

    def _attach_run_log_handlers(self, runDir: Path) -> None:
        """
        Mirror root-logger output into `<runDir>/debug.log` and `<runDir>/info.log`,
        in addition to the general log files under self.logDir. Only one run's
        handlers are active at a time -- starting a new run detaches the
        previous run's handlers first.

        Parameters
        ----------
        runDir : Path
            Directory of the run that should receive its own copy of the logs
        """
        self._detach_run_log_handlers()

        formatter = logging.Formatter(_DETAILED_LOG_FORMAT, datefmt=_DETAILED_LOG_DATEFMT)
        debugHandler = logging.FileHandler(runDir / "debug.log", encoding="utf-8")
        debugHandler.setLevel(logging.DEBUG)
        debugHandler.setFormatter(formatter)
        infoHandler = logging.FileHandler(runDir / "info.log", encoding="utf-8")
        infoHandler.setLevel(logging.INFO)
        infoHandler.setFormatter(formatter)

        root = logging.getLogger()
        root.addHandler(debugHandler)
        root.addHandler(infoHandler)
        self._runLogHandlers = [debugHandler, infoHandler]

    def _detach_run_log_handlers(self) -> None:
        """Stop mirroring logs into the previously active run's directory, if any."""
        if not self._runLogHandlers:
            return
        root = logging.getLogger()
        for handler in self._runLogHandlers:
            root.removeHandler(handler)
            handler.close()
        self._runLogHandlers = []

    def has_readiness(self, flag: ManagerReadiness) -> bool:
        """
        Check if the manager's readiness includes a specific flag like CHECKPOINT_AVAILABLE

        Parameters
        ----------
        flag : ManagerReadiness
            ManagerReadiness IntFlag to check

        Returns
        -------
        _name_ : bool
            Flag contained in readiness? True or False
        """
        return flag in self.readiness
    
    def get_missing_requirements(self, action: str) -> List[str]:
        """
        Human-readable list of what's missing before `action` can run.
        Empty list means the action is currently valid.

        Parameters
        ----------
        action : str
            action we want to perform

        Returns
        -------
        _name_ : List[str]
            List of States that are missing for the given action as description for reading

        Raises
        ------
        _name_ : ValueError
            If the name of the action is unknown. See AnomalyDetectionManager._REQUIREMENTS
        """
        required = self._REQUIREMENTS.get(action)
        if required is None:
            raise ValueError(f"Unknown action '{action}'. Known actions: {list(self._REQUIREMENTS)}")
        missing_flags = required & ~self.readiness
        return [
            desc for flag, desc in self._STATE_DESCRIPTIONS.items()
            if flag in missing_flags
        ]

    def can_run(self, action: str) -> bool:
        """
        Check if the manager can run a certain action listed under _REQUIREMENTS (train, eval, inference, setupTiling)

        Parameters
        ----------
        action : str
            string of action listed under _REQUIREMENTS

        Returns
        -------
        _name_ : bool
            Can the selected action be run? (True or False)
        """
        return not self.get_missing_requirements(action)

    def _require(self, action: str) -> None:
        """
        Raises the exception type matching the first (in pipeline order) missing
        requirement for `action`. `.missing` on the exception still lists everything
        missing, not just the one that determined the exception type.

        Parameters
        ----------
        action : str
            action to perform. See AnomalyDetectionManager._REQUIREMENTS

        Raises
        ------
        _name_ : ValueError
            If action is not known
        _name_ : error_cls
            If the state misses flags for the chosen action 
        """
        required = self._REQUIREMENTS.get(action)
        if required is None:
            raise ValueError(f"Unknown action '{action}'. Known actions: {list(self._REQUIREMENTS)}")

        missing_flags = required & ~self.readiness
        if not missing_flags:
            return

        missing_descriptions = [
            self._STATE_DESCRIPTIONS[flag] for flag in self._STATE_PRIORITY if flag in missing_flags
        ]
        first_missing_flag = next(flag for flag in self._STATE_PRIORITY if flag in missing_flags)
        error_cls = self._STATE_ERROR_CLASSES[first_missing_flag]

        raise error_cls(
            f"Cannot run '{action}': missing: {', '.join(missing_descriptions)}",
            missing=missing_descriptions,
        )
    
    def loadModelConfig(self, modelConfigPath: Path) -> ModelConfig:
        modelConfig = ModelConfig.from_yaml(modelConfigPath)
        self.modelConfig = modelConfig
        return modelConfig
    
    def generateModel(self, modelConfig: ModelConfig) -> None:
        """
        Create a model for the manager to manage. Either from a config or a path to a config yaml-file.

        Parameters
        ----------
        modelConfigPath : Optional[Path] (optional)
            Path to the config file. Has to be a yaml file. Default is `None`
        modelConfig : Optional[ModelConfig] (optional)
            Configuration either loaded from a file or written manually. Default is `None`

        Raises
        ------
        _name_ : ConfigError
            Neither ConfigFile nor Config where given.
        _name_ : ConfigError
            The model creation process failed for some reason.
        """

        model = create_model(modelConfig.name, modelConfig.to_dict())
        if model is None:
            raise ConfigError(f"Model creation failed for config: {modelConfig}")

        self.model = model
        self.readiness |= ManagerReadiness.MODEL_LOADED
        # loading a new model invalidates any previous training/tiling/calibration readiness
        self.readiness &= ~(ManagerReadiness.TILING_CONFIGURED | ManagerReadiness.CHECKPOINT_AVAILABLE | ManagerReadiness.CALIBRATED)
        logger.info(f"Successfully loaded model {self.model.name}")

    def setupTiling(self, tilingPipelineConfig: TilingPipelineConfig) -> bool:
        """
        Setup the tiling mechanism for a TiledEnsemble anomaly detection model. 
        The ensemble contains multiple models, one for each tile.

        Parameters
        ----------
        tilingPipelineConfig : TilingPipelineConfig
            Configuration file for the tilgin mechanism. Usually loaded from a TiledEnsemble.yaml file

        Returns
        -------
        _name_ : bool
            True if the tiling setup was successful
        """
        self._require("setupTiling")
        if tilingPipelineConfig.root_dir is None:
            tilingPipelineConfig.root_dir = self.outputDir
        self.tilingPipelineConfig = tilingPipelineConfig
        self.isTilingSetup = True
        self.readiness |= ManagerReadiness.TILING_CONFIGURED
        logger.info(f"Tiling configured: {tilingPipelineConfig}")
        return self.isTilingSetup
    
    def _runConfigFiles(
        self,
        modelConfigPath: Optional[Path],
        trainerConfigPath: Optional[Path],
        tilingConfigPath: Optional[Path],
        inferencerConfigPath: Optional[Path],
    ) -> RunConfigFiles:
        """
        The set of config files in use for the run being prepared, anchored
        to `self.configDir`. See `RunConfigFiles.copy_to` for how these get
        laid out under a run directory.

        Takes the four paths as parameters rather than reading them off
        `self` -- Manager doesn't keep its own copy of Product's config
        paths (Product is the source of truth for those; see setup.Product),
        so the caller (train()) passes through whatever it was given.

        Returns
        -------
        _name_ : RunConfigFiles
            Config file paths for the run being prepared.
        """
        preProcessorPath = postProcessorPath = evaluatorPath = None
        if self.modelConfig is not None:
            preProcessorPath = self.modelConfig.preProcessorPath
            postProcessorPath = self.modelConfig.postProcessorPath
            evaluatorPath = self.modelConfig.evaluatorPath

        return RunConfigFiles(
            configDir=self.configDir,
            modelConfigPath=modelConfigPath,
            trainerConfigPath=trainerConfigPath,
            tilingConfigPath=tilingConfigPath,
            inferencerConfigPath=inferencerConfigPath,
            preProcessorPath=preProcessorPath,
            postProcessorPath=postProcessorPath,
            evaluatorPath=evaluatorPath,
        )

    @classmethod
    def loadProduct(cls, productConfigPath: Path, outputPath: Path, configDir: Path) -> Tuple["AnomalyDetectionManager", Product]:
        """
        Loads a Product configuration from disk, which should contain everything for a training run

        Parameters
        ----------
        productConfigPath : Path
            Configuration to the specific product, i.e. MDF6 or cable (from MVTecAD)
        outputPath : Path
            Where the output path of the manager should be
        configDir : Path
            Where the configuration files can be found. Usually a dir named 'configs'

        Returns
        -------
        _name_ : Tuple[AnomalyDetectionManager, Product]
            A tuple of the created manager class and a product that contains all important information about the product.
        """
        product = loadProductFromYaml(productConfigPath, config_dir=configDir, baseOutputDir=outputPath)
        manager = cls(outputDir=outputPath, configDir=configDir)
        manager.generateModel(modelConfig=product.modelConfig)
        manager.setupTiling(product.tilingPipelineConfig)
        manager.modelTrainingDir = product.modelTrainingDir

        # loadProduct points at a model/weights path that presumably already has a
        # checkpoint and, if it was ever calibrated, a stats.json alongside it.
        if manager.modelTrainingDir is not None:
            manager.ckptDir = resolve_checkpoint_paths(manager.modelTrainingDir)
            if manager.ckptDir.exists():
                manager.readiness |= ManagerReadiness.CHECKPOINT_AVAILABLE
            if resolve_stats_path(manager.modelTrainingDir).exists():
                manager.readiness |= ManagerReadiness.CALIBRATED
        manager._apply_visualizer_output_dir(outputPath)

        return manager, product

    @classmethod
    def loadFromRunDir(cls, runDir: Path, baseOutputDir: Path) -> Tuple["AnomalyDetectionManager", Product]:
        """
        Bootstrap a manager+product straight from a completed run's own
        frozen config copies (see run_registry.reproduce_run/
        RunConfigFiles.copy_to), instead of a live product YAML -- lets a
        caller point at any run_list/best_model_info result and actually
        use it (e.g. for inference) even without ever calling loadProduct
        first, or having loaded a different product before.

        configDir is anchored at `runDir/configs` (the frozen copies), not
        the live configs/ tree -- consistent with why runs archive their
        own config copies in the first place: anything this manager does
        afterwards (e.g. archiving further config copies of its own,
        should it ever train/eval again) stays pinned to this run's own
        settings, immune to the live tree having changed since.

        dataset.split isn't recorded anywhere in a run's manifest/frozen
        configs (only name/category are, see reproduce_run) -- the
        resulting Product's datasetConfig.splits is a placeholder. Harmless
        for loading a checkpoint to run inference with (this exists for),
        since dataset_load/DatasetSession.select_category are a separate,
        operator-driven step that don't consult it.

        Raises FileNotFoundError if `runDir` doesn't have a manifest or
        frozen config copies (see reproduce_run), or ValueError if the
        manifest has no dataset name/category recorded (a run this old
        predates the manifest carrying that at all).
        """
        reproduced = reproduce_run(runDir)
        if reproduced["datasetName"] is None or reproduced["category"] is None:
            raise ValueError(
                f"Run manifest at {runDir} has no dataset name/category recorded; "
                f"cannot bootstrap a product from it."
            )

        configDir = runDir / "configs"
        manager = cls(outputDir=baseOutputDir, configDir=configDir)
        manager.generateModel(modelConfig=reproduced["modelConfig"])
        manager.setupTiling(reproduced["tilingPipelineConfig"])
        manager.modelTrainingDir = runDir
        manager.ckptDir = resolve_checkpoint_paths(runDir)
        if manager.ckptDir.exists():
            manager.readiness |= ManagerReadiness.CHECKPOINT_AVAILABLE
        if resolve_stats_path(runDir).exists():
            manager.readiness |= ManagerReadiness.CALIBRATED
        manager._apply_visualizer_output_dir(baseOutputDir)

        datasetName = reproduced["datasetName"]
        datasetConfig = DatasetConfig(
            name=datasetName,
            category=reproduced["category"],
            splits=("train", "test"),
            path=(Path("datasets") / datasetName).resolve(),
        )
        product = Product(
            name=reproduced["category"],
            modelConfig=reproduced["modelConfig"],
            modelConfigPath=reproduced["modelConfigPath"],
            modelTrainingDir=runDir,
            tilingPipelineConfig=reproduced["tilingPipelineConfig"],
            tilingConfigPath=reproduced["tilingConfigPath"],
            trainerConfig=reproduced["trainerConfig"],
            trainerConfigPath=reproduced["trainerConfigPath"],
            datamoduleConfig=reproduced["datamoduleConfig"],
            inferencerConfig=reproduced["inferencerConfig"],
            inferencerConfigPath=reproduced["inferencerConfigPath"],
            datasetConfig=datasetConfig,
            datasetPath=datasetConfig.path,
            # Every run resolve_output_dir creates for this codebase is a
            # tiled-ensemble one (tiling=True is hardcoded there) -- never
            # read elsewhere (see Product.enableTiling), so a fixed True
            # here is simpler and more reliable than trying to infer it
            # from tilingPipelineConfig, which reproduce_run doesn't
            # otherwise mark as enabled/disabled.
            enableTiling=True,
            selection="latest",
        )
        return manager, product

    def loadCheckpoint(self, path:Optional[Path], tilingPipelineConfig: TilingPipelineConfig):
        """
        Check if the checkpoints for a tiledEnsemble run (eval, inference) are available at the expected directory

        Parameters
        ----------
        path : Path
            Path where to check
        tilingPipelineConfig : TilingPipelineConfig
            Tiling config needed to what kind of files to expect

        Raises
        ------
        _name_ : CheckpointNotFoundError
            If not all expected checkpoints are found.
        """
        if path is None:
            if self.ckptDir is not None:
                path = self.ckptDir
                logger.info(f"Loading checkpoint from manger.ckptDir: {self.ckptDir}")
            else:
                raise AttributeError("Either path needs to be given to loadCheckpoint or ckptDir needs to be set for manager.")
        complete, missingCkpt = checkTiledCheckpointsExist(ckptDir=path, tilingPipelineConfig=tilingPipelineConfig)
        if complete:
            self.readiness |= ManagerReadiness.CHECKPOINT_AVAILABLE
        else:
            missing:List[str] = [str(path) for path in missingCkpt]
            raise CheckpointNotFoundError(message=f"Follwing Checkpoints are missing from {path}", missing=missing)

    def _trainTiledModel(self,
                         datasetSession:DatasetSession,
                         trainerConfig: TrainerConfig,
                         datamoduleConfig:DataModuleConfig,
                         modelConfig:ModelConfig,
                         tilingPipelineConfig:TilingPipelineConfig):
        """
        Train a tiled ensemble model with the given training parameters on the given data with, considering the tiling.

        Parameters
        ----------
        datasetSession : DatasetSession
            On which data to train the model
        trainerConfig : TrainerConfig
            How the training should go concerning the training itself. (E.g. max iterations)
        datamoduleConfig : DataModuleConfig
            How the training should go concerning the data. (E.g. batch size)
        modelConfig : ModelConfig
            What kind of model to train
        tilingPipelineConfig : TilingPipelineConfig
            How the tiling process works (E.g. tile size, stride,...)
        """
        try:
            datamodule = datasetSession.setupDatamodule(datamoduleConfig, self.outputDir)
            logger.info(f"Datamodule set up successfully: {datamodule}")
        except DatamoduleError as e:
            logger.error(f"Error occurred while setting up datamodule: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            raise e

        gtAvail:bool = True if len(datasetSession.FO_Dataset.exists("ground_truth"))>0 else False

        logger.info(f"Dataset used for training: {datasetSession.FO_Dataset}")

        self.setupTiling(tilingPipelineConfig)

        trainPipeline = TrainTiledEnsemble(rootDir=self.outputDir,
                                           datamodule=datamodule,
                                           dataModuleConfig=datamoduleConfig,
                                           FO_Dataset=datasetSession.FO_Dataset,
                                           gtAvail=gtAvail,
                                           tilingPipelineConfig=tilingPipelineConfig,
                                           modelConfig=modelConfig,
                                           trainerConfig=trainerConfig)
       
        _run_in_large_stack(trainPipeline.run)

    def _evalTiledModel(self,
                         datasetSession:DatasetSession,
                         evalConfig: TrainerConfig,
                         datamoduleConfig:DataModuleConfig,
                         modelConfig:ModelConfig,
                         tilingPipelineConfig:TilingPipelineConfig):
        """
        Evaluate a trained tiled ensemble model on a dataset. Also creates a stats.json that desribes how the threshold
        should be set. (Threshold setting is independent of the model, in the sense that the model just gives a score for each image)
        We expect the need checkpoint weights to be set via a previous training or loadCheckpoint()

        Parameters
        ----------
        datasetSession : DatasetSession
            On which data to evaluate the model
        evalConfig : TrainerConfig
            How the evaluation should go concerning the evaluation itself. (E.g. device)
        datamoduleConfig : DataModuleConfig
            How the evaluation should go concerning the data. (E.g. batch size)
        modelConfig : ModelConfig
            What kind of model to train
        tilingPipelineConfig : TilingPipelineConfig
            How the tiling process works (E.g. tile size, stride,...)
        """

        datamodule = datasetSession.setupDatamodule(datamoduleConfig, self.outputDir)
        gtAvail:bool = True if len(datasetSession.FO_Dataset.exists("ground_truth"))>0 else False

        logger.info(f"Dataset used for training: {datasetSession.FO_Dataset}")

        self.setupTiling(tilingPipelineConfig)

        evalPipeline = EvalTiledEnsemble(rootDir=self.outputDir,
                                           datamodule=datamodule,
                                           dataModuleConfig=datamoduleConfig,
                                           FO_Dataset=datasetSession.FO_Dataset,
                                           gtAvail=gtAvail,
                                           tilingPipelineConfig=tilingPipelineConfig,
                                           modelConfig=modelConfig,
                                           evalConfig=evalConfig,
                                           ckptPath=self.ckptDir)

        _run_in_large_stack(evalPipeline.run)

    def _inferenceTiledModel(self,
                             trainingDir: Path,
                             ckptDir: Path,
                             datasetSession: DatasetSession,
                             dataModuleConfig: DataModuleConfig,
                             tilingPipelineConfig:TilingPipelineConfig,
                             inferencerConfig: TrainerConfig,
                             modelConfig: ModelConfig):
        """
        Predict/Inference a trained tiled ensemble model on a dataset. Also needs a stats.json that desribes how the threshold
        is set. (Threshold setting is independent of the model, in the sense that the model just gives a score for each image)
        We expect the need checkpoint weights to be set via a previous training or loadCheckpoint()

        Parameters
        ----------
        trainingDir : Path
            Directory where the training took place that contains the stats.json file
        ckptDir : Path
            Path to the directory of checkpoint files to use
        datasetSession : DatasetSession
            On which data to evaluate the model
        evalConfig : TrainerConfig
            How the evaluation should go concerning the evaluation itself. (E.g. device)
        datamoduleConfig : DataModuleConfig
            How the evaluation should go concerning the data. (E.g. batch size)
        modelConfig : ModelConfig
            What kind of model to train
        tilingPipelineConfig : TilingPipelineConfig
            How the tiling process works (E.g. tile size, stride,...)
        """

        datamodule = datasetSession.setupDatamodule(dataModuleConfig, self.outputDir)

        logger.info(f"Dataset used for training: {datasetSession.FO_Dataset}")

        self.setupTiling(tilingPipelineConfig)

        assert datasetSession.AL_PredictDataset is not None

        inferencerPipeline = InferenceTiledEnsemble(root_dir=self.outputDir,
                                          trainingDir=trainingDir,
                                          ckptDir=ckptDir,
                                          inferenceDataset=datasetSession.AL_PredictDataset,
                                          dataset=datasetSession.FO_Dataset,
                                          datamodule=datamodule,
                                          dataModuleConfig=dataModuleConfig,
                                          tilingPipelineConfig=tilingPipelineConfig,
                                          inferencerConfig=inferencerConfig,
                                          modelConfig=modelConfig
                                          )

        _run_in_large_stack(inferencerPipeline.run)

    def _prepareRun(
        self,
        trainerConfig:TrainerConfig,
        modelConfig: ModelConfig,
        datasetSession: DatasetSession,
        datamoduleConfig: DataModuleConfig,
        tilingPipelineConfig: TilingPipelineConfig,
        runLabel: Optional[str] = None,
        runId: Optional[str] = None,
        datasetNameOverride: Optional[str] = None,
        categoryOverride: Optional[str] = None,
        runKind: str = "training",
    ) -> RunContext:
        """
        Shared setup for train()/eval()/inference(): resolve paths, wire up tiling,
        callbacks and the W&B logger. Output dir doubles as the checkpoint dir
        since these runs write and read within the same location.

        Parameters
        ----------
        modelConfig : ModelConfig
            Descibes an anomaly detection model
        datasetSession : DatasetSession
            Contains the dataset
        tilingPipelineConfig : TilingPipelineConfig
            How should the image be tiled?
        runId : Optional[str] (optional)
            If given, used directly as the run id instead of generating a fresh
            one -- lets repeated calls (e.g. one inference() per image during a
            shift session) share a single output directory instead of each
            getting its own. See AD_Worker's shift_* commands.
        datasetNameOverride, categoryOverride : Optional[str] (optional)
            Use these instead of datasetSession.datasetName/.category[0] for
            resolve_output_dir's path -- for predict-only sessions,
            datasetSession.datasetName/.category are shaped by whatever's
            actually on disk where the images were captured (an ad hoc
            per-call name, an incidental folder name), not the product's
            real dataset/name -- data-selection still needs the real
            per-sample values (setupDatamodule etc. read datasetSession
            itself, untouched), only the *results path* should instead read
            like training's (see AD_Worker._handle_inference/_handle_shift_
            inspect, the only callers that pass these).
        runKind : str (optional)
            Forwarded to resolve_output_dir -- "training" (default, used by
            train()/eval()) or "inference" (inference() passes this
            explicitly), so trained-model runs and predict-only runs land in
            separate subtrees instead of sharing one runs/ folder.

        Returns
        -------
        _name_ : RunContext
            Run name, output directory, checkpoint directory, (checkpoint path, not applicable for tiled)
        """
        runId = runId if runId is not None else generate_run_id(runLabel)
        resolvedDatasetName = datasetNameOverride if datasetNameOverride is not None else datasetSession.datasetName
        resolvedCategory = categoryOverride if categoryOverride is not None else (datasetSession.category[0] if datasetSession.category else None)
        outputDir = resolve_output_dir(
            baseOutputDir=self.baseOutputDir, datasetName=resolvedDatasetName,
            modelName=modelConfig.name, runId=runId, category=resolvedCategory, tiling=True,
            runKind=runKind,
        )

        effective_config = serialize_effective_config(trainerConfig, modelConfig, datamoduleConfig, tilingPipelineConfig, datasetSession)
        print(effective_config)
        write_run_manifest(outputDir, effective_config)

        # Logged before attaching the run's own handlers, so this pointer only
        # ends up in the general log files, not in the run's own debug/info.log.
        logger.info(f"Run '{runId}' started. Detailed logs for this run are also written to {outputDir} (debug.log, info.log).")
        self._attach_run_log_handlers(outputDir)

        self.outputDir, self.runId = outputDir, runId
        self._apply_visualizer_output_dir(outputDir)
        self.setupTiling(tilingPipelineConfig)
        self.readiness |= ManagerReadiness.RUN_PREPARED

        return RunContext(runName=runId, outputDir=outputDir)

    def train(
        self,
        trainerConfig: TrainerConfig,
        modelConfig: ModelConfig,
        datamoduleConfig: DataModuleConfig,
        tilingPipelineConfig: TilingPipelineConfig,
        datasetSession: Optional[DatasetSession] = None,
        modelConfigPath: Optional[Path] = None,
        trainerConfigPath: Optional[Path] = None,
        tilingConfigPath: Optional[Path] = None,
        inferencerConfigPath: Optional[Path] = None,
        runLabel: Optional[str] = None,
    ) -> RunContext:
        """
        Train the current model on `datasetSession`, creating a new run
        directory.

        The four `*ConfigPath` arguments are only used to archive raw config
        YAMLs alongside the run's manifest (see RunConfigFiles.copy_to) --
        Manager doesn't keep its own copy of them, Product does (see
        setup.Product), so pass `product.modelConfigPath` etc. through here.

        runLabel : Optional[str]
            Forwarded to _prepareRun -> generate_run_id (run_registry.py),
            which slots it into the run id as YYYYMMDD-HHMMSS_<label>_<hash>
            -- the date prefix and hash suffix are generate_run_id's own and
            always present regardless of this; leave unset for the plain
            YYYYMMDD-HHMMSS_<hash> form. AD_Worker._handle_train is what
            actually supplies a value here (the operator's own run label, or
            the product name if they didn't type one).

        Returns
        -------
        RunContext
            The newly created training run (runName, outputDir). Callers
            that also hold a Product for this manager should write
            `ctx.outputDir` onto `product.modelTrainingDir` themselves --
            Manager doesn't reach into Product to update it, so that sync
            stays explicit and happens in exactly one place.
        """
        datasetSession = self.attachDatasetSession(datasetSession)
        self._require("train")
        ctx = self._prepareRun(trainerConfig, modelConfig, datasetSession, datamoduleConfig=datamoduleConfig, tilingPipelineConfig=tilingPipelineConfig, runLabel=runLabel)
        if not ManagerReadiness.RUN_PREPARED in self.readiness:
            raise ManagerReadinessError(f"Cannot run 'train': missing: {self._STATE_DESCRIPTIONS[ManagerReadiness.RUN_PREPARED]}",
                        missing=[self._STATE_DESCRIPTIONS[ManagerReadiness.RUN_PREPARED]])
        self._runConfigFiles(
            modelConfigPath=modelConfigPath, trainerConfigPath=trainerConfigPath,
            tilingConfigPath=tilingConfigPath, inferencerConfigPath=inferencerConfigPath,
        ).copy_to(ctx.outputDir)
        # raw YAMLs alongside the manifest
        self.ckptDir = resolve_checkpoint_paths(ctx.outputDir)

        self._trainTiledModel(
            datasetSession=datasetSession, trainerConfig=trainerConfig,
            datamoduleConfig=datamoduleConfig, modelConfig=modelConfig,
            tilingPipelineConfig=tilingPipelineConfig,
        )
        complete, missing = checkTiledCheckpointsExist(self.ckptDir, tilingPipelineConfig)
        if complete:
            # train() produces a checkpoint, not thresholds -- CALIBRATED
            # only comes from a subsequent eval() call.
            self.modelTrainingDir = ctx.outputDir
            self.readiness |= ManagerReadiness.CHECKPOINT_AVAILABLE
        else:
            logger.warning(f"Training finished but {len(missing)} tile checkpoint(s) missing: {missing}")

        self.readiness &= ~(ManagerReadiness.RUN_PREPARED)

        return ctx

    def eval(
        self,
        evalConfig: TrainerConfig,
        modelConfig: ModelConfig,
        datamoduleConfig: DataModuleConfig,
        datasetSession: DatasetSession,
        tilingPipelineConfig: TilingPipelineConfig,
        modelTrainingDir: Optional[Path] = None,
        modelConfigPath: Optional[Path] = None,
        trainerConfigPath: Optional[Path] = None,
        tilingConfigPath: Optional[Path] = None,
        inferencerConfigPath: Optional[Path] = None,
    ) -> RunContext:
        """
        Evaluate the current model on the current dataset, determining the
        anomaly thresholds (stats.json). Tiled ensemble only for now.

        The eval run this creates is self-contained: copy_checkpoints() pulls
        the checkpoints being evaluated into the run's own directory
        alongside the stats.json the run produces. self.modelTrainingDir is
        updated to point at *that* directory rather than the original
        training run, so a subsequent inference() call on this same manager
        (with modelTrainingDir omitted) finds both checkpoints/ and
        stats.json in one place, and the original training run can be
        retrained over or deleted without invalidating this calibrated one.

        Returns
        -------
        Path
            The new, self-contained calibrated run directory. Callers that
            also hold a Product for this manager should write this onto
            `product.modelTrainingDir` themselves -- Manager doesn't reach
            into Product to update it, so that sync stays explicit and
            happens in exactly one place.
        """
        datasetSession = self.attachDatasetSession(datasetSession)
        if modelTrainingDir is None:
            if self.modelTrainingDir is None:
                raise AttributeError("Need a modelTrainingDir and neither attribute nor class atrribute are available")
            modelTrainingDir = self.modelTrainingDir
        self.ckptDir = resolve_checkpoint_paths(modelTrainingDir)
        complete, missing = checkTiledCheckpointsExist(self.ckptDir, tilingPipelineConfig)
        if not complete:
            raise CheckpointNotFoundError(
                f"Incomplete checkpoint set in {self.ckptDir}: missing: {[p.name for p in missing]}; ",
                missing=[f"Checkpoint file {p.name}" for p in missing],
            )
        else:
            self.readiness |= ManagerReadiness.CHECKPOINT_AVAILABLE
        self._require("eval")
        ctx = self._prepareRun(evalConfig, modelConfig, datasetSession, datamoduleConfig, tilingPipelineConfig)
        if not ManagerReadiness.RUN_PREPARED in self.readiness:
            raise ManagerReadinessError(f"Cannot run 'train': missing: {self._STATE_DESCRIPTIONS[ManagerReadiness.RUN_PREPARED]}",
                        missing=[self._STATE_DESCRIPTIONS[ManagerReadiness.RUN_PREPARED]])
        self._runConfigFiles(
            modelConfigPath=modelConfigPath, trainerConfigPath=trainerConfigPath,
            tilingConfigPath=tilingConfigPath, inferencerConfigPath=inferencerConfigPath,
        ).copy_to(ctx.outputDir)
        copy_checkpoints(self.ckptDir, ctx.outputDir)
        self._evalTiledModel(
            datasetSession=datasetSession, evalConfig=evalConfig,
            datamoduleConfig=datamoduleConfig, modelConfig=modelConfig,
            tilingPipelineConfig=tilingPipelineConfig,
        )

        # Verify the eval run directory actually ended up self-contained
        # (checkpoints copied in above, stats.json just written by the
        # pipeline's statistics job) before treating it as the new canonical
        # modelTrainingDir -- don't just trust the pipeline calls above
        # succeeded silently.
        newCkptDir = resolve_checkpoint_paths(ctx.outputDir)
        complete, missing = checkTiledCheckpointsExist(newCkptDir, tilingPipelineConfig)
        if not complete:
            raise CheckpointNotFoundError(
                f"Checkpoints failed to copy into the eval run at {newCkptDir}: missing: {[p.name for p in missing]}",
                missing=[f"Checkpoint file {p.name}" for p in missing],
            )
        statsPath = resolve_stats_path(ctx.outputDir)
        if not statsPath.exists():
            raise CalibrationNotFoundError(
                f"eval() completed but no stats.json was written to {statsPath}.",
                missing=["stats.json"],
            )

        self.modelTrainingDir = ctx.outputDir
        self.ckptDir = newCkptDir
        self.readiness |= ManagerReadiness.CHECKPOINT_AVAILABLE | ManagerReadiness.CALIBRATED

        self.readiness &= ~(ManagerReadiness.RUN_PREPARED)

        return ctx

    def inference(
        self,
        inferencerConfig: TrainerConfig,
        modelConfig: ModelConfig,
        datamoduleConfig: DataModuleConfig,
        datasetSession: DatasetSession,
        tilingPipelineConfig: TilingPipelineConfig,
        modelTrainingDir: Optional[Path] = None,
        modelConfigPath: Optional[Path] = None,
        trainerConfigPath: Optional[Path] = None,
        tilingConfigPath: Optional[Path] = None,
        inferencerConfigPath: Optional[Path] = None,
        datasetNameOverride: Optional[str] = None,
        categoryOverride: Optional[str] = None,
    ) -> None:
        """Run inference: write results under the *current* dataset's output dir,
        but read the checkpoint from `trainingDir` (a prior, separate run).
        This replaces the old adjustPaths(..., adjustCheckpoints=False) workaround —
        the two directories are now resolved independently and explicitly.

        `runId`, if given, pins the output directory instead of generating a
        fresh one -- pass the same value across repeated calls (e.g. one call
        per image during a shift session) to have them all write into a single
        shared results directory.

        datasetNameOverride, categoryOverride : forwarded to _prepareRun --
        see its own docstring. A predict-only datasetSession's own
        datasetName/category reflect an ad hoc per-call name and whatever
        folder the captures happened to sit in, not the product's actual
        dataset/name, so results otherwise land under a differently-shaped,
        much less meaningful path than train()/eval()'s
        <datasetName>/<productName>/<modelName>/tiled/runs/<runId>. Passing
        these (see AD_Worker._handle_inference/_handle_shift_inspect, which
        supply the product's real datasetConfig.name/product.name) makes
        inference's results path match that same shape.
        """
        datasetSession = self.attachDatasetSession(datasetSession)
        if modelTrainingDir is not None:
            self.modelTrainingDir = modelTrainingDir
        else:
            if self.modelTrainingDir is None:
                raise AttributeError("Need a modelTrainingDir and neither attribute nor class atrribute are available")
            modelTrainingDir = self.modelTrainingDir

        self.ckptDir = resolve_checkpoint_paths(modelTrainingDir)
        complete, missing = checkTiledCheckpointsExist(self.ckptDir, tilingPipelineConfig)
        if not complete:
            raise CheckpointNotFoundError(
                f"Incomplete checkpoint set in {self.ckptDir}: missing {[p.name for p in missing]}",
                missing=[f"Checkpoint file {p.name}" for p in missing],
            )

        self._require("inference")
        ctx = self._prepareRun(
            inferencerConfig, modelConfig, datasetSession, datamoduleConfig, tilingPipelineConfig,
            datasetNameOverride=datasetNameOverride, categoryOverride=categoryOverride,
            runKind="inference",
        )

        if not ManagerReadiness.RUN_PREPARED in self.readiness:
            raise ManagerReadinessError(f"Cannot run 'train': missing: {self._STATE_DESCRIPTIONS[ManagerReadiness.RUN_PREPARED]}",
                        missing=[self._STATE_DESCRIPTIONS[ManagerReadiness.RUN_PREPARED]])
        
        self._runConfigFiles(
            modelConfigPath=modelConfigPath, trainerConfigPath=trainerConfigPath,
            tilingConfigPath=tilingConfigPath, inferencerConfigPath=inferencerConfigPath,
        ).copy_to(ctx.outputDir)
        copy_checkpoints(self.ckptDir, ctx.outputDir)

        self._inferenceTiledModel(
            trainingDir=modelTrainingDir, 
            ckptDir=self.ckptDir, 
            datasetSession=datasetSession,
            dataModuleConfig=datamoduleConfig, 
            tilingPipelineConfig=tilingPipelineConfig,
            inferencerConfig=inferencerConfig, 
            modelConfig=modelConfig,
        )

        self.readiness &= ~(ManagerReadiness.RUN_PREPARED)


    def exportResults(self, exportPath:Path) -> None:
        # Create the destination directory if it doesn't exist
        os.makedirs(exportPath, exist_ok=True)

        # Copy each item in the source directory
        for item in os.listdir(self.outputDir):
            src_path = os.path.join(self.outputDir, item)
            dst_path = os.path.join(exportPath, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)