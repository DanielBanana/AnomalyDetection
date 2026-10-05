"""
Helpful functions for setting up an anomaly detection model
"""
# GENERAL
import os
import sys
import logging
import yaml
import datetime
import importlib
import shutil
import numpy as np
import torchvision.transforms.v2 as T
import uuid
import platform
import torch

from dataclasses import asdict, dataclass, field
from pathlib import Path
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Type
from enum import Enum
from torchvision.transforms.v2 import Transform
from contextlib import contextmanager

# ANOMALIB
from anomalib.metrics import F1Score, AUPR, AUROC, F1AdaptiveThreshold, AUPIMO, AUPRO, PGn, PBn, ManualThreshold, F1Max
from anomalib.data import MVTecAD, BTech, Visa, Kolektor, Folder, PredictDataset
from anomalib.models import Padim, EfficientAd, Dsr, ReverseDistillation, Fastflow, Patchcore, Stfpm, InpFormer, AnomalyVFM, Glass
from anomalib.models.components import AnomalibModule
from anomalib.callbacks import ModelCheckpoint, GraphLogger, TimerCallback
from anomalib.loggers import AnomalibTensorBoardLogger, AnomalibWandbLogger
from anomalib.metrics import Evaluator
from anomalib.pre_processing import PreProcessor
from anomalib.data.utils.split import ValSplitMode, TestSplitMode
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage, ThresholdingStage
from anomalib.visualization import ImageVisualizer
from anomalib.metrics import AUROC, AUPR, F1AdaptiveThreshold, F1Score

# PYTORCH LIGHTNING
from lightning.pytorch.callbacks import TQDMProgressBar

# OWN FILES
from settings import DATASETS, CATEGORIES, MODELS, DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG
from data.anomaly_datasets import importDataset, exportDataset, FODataModule, FODataset, importPredictDataset
from run_registry import resolve_run_dir
from tiling.post_processor import AOIPostProcessor
from settings import DEFAULT_FIELDS_CONFIG, DEFAULT_OVERLAY_FIELDS_CONFIG, DEFAULT_TEXT_CONFIG

# FIFTYONE
import fiftyone.core.dataset as fod
import fiftyone as fo
import fiftyone.zoo as foz # zoo datasets and models
import fiftyone.brain as fob # ML methods
from fiftyone import ViewField as F # helper for defining views

logger = logging.getLogger(__name__)

class SetupError(Exception):
    """Base class for all AnomalyDetectionManager errors."""

class DatamoduleError(SetupError): ...
class DatasetSessionError(SetupError): ...
class ModelError(SetupError): ...
class TilingPipelineError(SetupError): ...
class ProductError(RuntimeError): ...


class VisualizerType(Enum):
    train = 0
    valNoGT = 1
    valGT = 2

def getVisualizer(vtype:VisualizerType, fieldSize:Tuple[int,int]):
    match vtype:
        case VisualizerType.train:
            visualizer = ImageVisualizer(# output_dir=prediction_path,
                        fields=["image"],
                        overlay_fields=[("image", ["anomaly_map"])],
                        field_size=fieldSize,
                        fields_config=DEFAULT_FIELDS_CONFIG,
                        overlay_fields_config=DEFAULT_OVERLAY_FIELDS_CONFIG,
                        text_config=DEFAULT_TEXT_CONFIG)
            return visualizer
        case VisualizerType.valNoGT:
            visualizer = ImageVisualizer(# output_dir=prediction_path,
                        fields=["image", "pred_mask"],
                        overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],
                        field_size=fieldSize,
                        fields_config=DEFAULT_FIELDS_CONFIG,
                        overlay_fields_config=DEFAULT_OVERLAY_FIELDS_CONFIG,
                        text_config=DEFAULT_TEXT_CONFIG)
            return visualizer

        case VisualizerType.valGT:
            visualizer = ImageVisualizer(# output_dir=prediction_path,
                        fields=["image", "gt_mask", "pred_mask"],
                        overlay_fields=[("image", ["anomaly_map"]), ("image", ["gt_mask"]), ("image", ["pred_mask"])],
                        field_size=fieldSize,
                        fields_config=DEFAULT_FIELDS_CONFIG,
                        overlay_fields_config=DEFAULT_OVERLAY_FIELDS_CONFIG,
                        text_config=DEFAULT_TEXT_CONFIG)
            return visualizer

def find_first_file(directory:Path, target_filename:str) -> Path|None:
    target_lower = target_filename.lower()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower() == target_lower:
                return Path(os.path.join(root, file))
    return None  # Return None if the file is not found

@contextmanager
def exclude_from_logger():
    original_stdout = sys.stdout
    sys.stdout = sys.__stdout__  # Restore original stdout
    try:
        yield
    finally:
        sys.stdout = original_stdout  # Restore logger stdout

# def load_transform_from_yaml(configPath:Path):
#     with open(configPath, "r") as f:
#         config:Dict[str,Any] = yaml.safe_load(f)

#     transform_list:List[Transform] = []
#     for step in config["transform"]:
#         transform_name = step["name"]
#         transform_args = step["args"]

#         # Get the transform class from torchvision.transforms
#         transform_class = getattr(T, transform_name)
#         # Instantiate the transform with the provided arguments
#         transform:Transform = transform_class(**transform_args)
#         transform_list.append(transform)

#     # Compose all transforms into a pipeline
#     return T.Compose(transform_list)


# def load_metrics_from_yaml(configPath: Path) -> tuple[List[Any], List[Any]]:
#     with open(configPath, "r") as f:
#         config:Dict[str,List[Dict[str,Any]]] = yaml.safe_load(f)

#     # Map metric types to their classes
#     metric_classes: Dict[str,Any] = {
#         "AUROC": AUROC,
#         "AUPR": AUPR,
#         "F1AdaptiveThreshold": F1AdaptiveThreshold,
#         "F1Score": F1Score,
#     }

#     def instantiate_metrics(metrics_config:List[Dict[str,Any]]) -> List[Any]:
#         metrics:List[Any] = []
#         for metric_config in metrics_config:
#             metric_type = metric_config["type"]
#             fields = metric_config["fields"]
#             prefix = metric_config["prefix"]
#             # Instantiate the metric
#             metric = metric_classes[metric_type](fields=fields, prefix=prefix)
#             metrics.append(metric)
#         return metrics

#     val_metrics:List[Any] = instantiate_metrics(config["val_metrics"])
#     test_metrics:List[Any]  = instantiate_metrics(config["test_metrics"])

#     return val_metrics, test_metrics


def getDefaultEvaluator():
    val_metrics, test_metrics = define_metrics()
    evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
    return evaluator

def define_metrics() -> Tuple[List[Any], List[Any]]:
    # val metrics (needed for early stopping)
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    image_aupr = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_aupr = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    image_f1score = F1AdaptiveThreshold(fields=["pred_score", "gt_label"], prefix="image_")
    val_metrics:List[Any] = [image_auroc, pixel_auroc, image_aupr, pixel_aupr, image_f1score]

    # test_metrics
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    image_aupr = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_aupr = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    test_metrics:List[Any]  = [image_auroc, image_f1score, pixel_auroc, pixel_f1score, image_aupr, pixel_aupr]
    
    return val_metrics, test_metrics

def mapNameToModule(modelName: str) -> Optional[Type[AnomalibModule]]:
    if modelName.lower() == "efficientad-s":
        # model = EfficientAd(visualizer=visualizer, model_size="small", post_processor=postProcessor)
        model = EfficientAd
    elif modelName.lower() == "efficientad-m":
        model = EfficientAd
    elif modelName.lower() == "dsr":
        model = Dsr
    elif modelName.lower() == "reversedistillation":
        model = ReverseDistillation
    elif modelName.lower() == "reverse_distillation":
        model = ReverseDistillation
    elif modelName.lower() == "rd":
        model = ReverseDistillation
    elif modelName.lower() == "stfpm":
        model = Stfpm
    elif modelName.lower() == "fastflow":
        model = Fastflow
    elif modelName.lower() == "fast_flow":
        model = Fastflow
    elif modelName.lower() == "patchcore":
        model = Patchcore
    elif modelName.lower() == "padim":
        model = Padim
    elif modelName.lower() == "InpFormer":
        model = InpFormer
    else:
        KeyError(f"Model {modelName} not found! \n Available models are: {', '.join(MODELS)}")
        model=None
    return model

def create_model(modelName:str, modelConfig:dict[str,Any]) -> Optional[AnomalibModule]:
    if modelName == "EfficientAd":
        # model = EfficientAd(visualizer=visualizer, model_size="small", post_processor=postProcessor)
        model = EfficientAd(**modelConfig)
    elif modelName == "dsr":
        model = Dsr(**modelConfig)
    elif modelName == "reversedistillation":
        model = ReverseDistillation(**modelConfig)
    elif modelName == "reverse_distillation":
        model = ReverseDistillation(**modelConfig)
    elif modelName == "rd":
        model = ReverseDistillation(**modelConfig)
    elif modelName == "stfpm":
        model = Stfpm(**modelConfig)
    elif modelName == "Stfpm":
        model = Stfpm(**modelConfig)
    elif modelName == "fastflow":
        model = Fastflow(**modelConfig)
    elif modelName == "fast_flow":
        model = Fastflow(**modelConfig)
    elif modelName == "patchcore":
        model = Patchcore(**modelConfig)
    elif modelName == "Patchcore":
        model = Patchcore(**modelConfig)
    elif modelName == "padim":
        model = Padim(**modelConfig)
    elif modelName == "Padim":
        model = Padim(**modelConfig)
    elif modelName == "InpFormer":
        model = InpFormer(**modelConfig)
    else:
        print(f"Model {modelName} not found! \n Available models are: {', '.join(MODELS)}")
        model = None
    return model

from typing import Tuple, Dict, Any

# def setupTensorboardLoggingAndCallbacks(logDir, runName, versionName, version) -> Tuple[AnomalibTensorBoardLogger, Dict[str, Any]]:
#     checkpointDir = os.path.join(logDir, runName, versionName, "checkpoints")
#     if not os.path.exists(checkpointDir):
#         os.mkdir(checkpointDir)
#     checkpointCallback = ModelCheckpoint(
#         dirpath=checkpointDir,
#         filename="best",
#         monitor="image_F1AdaptiveThreshold",  # val_loss not found?
#         verbose=True,
#         save_top_k=1,  # Save only the best model
#         mode="min",  # Save the model with the minimum training loss
#     )
    
#     checkpointCallback.FILE_EXTENSION = ".pt"  # Set the file extension for checkpoints
    
#     graphCallback = GraphLogger()
#     timerCallback = TimerCallback()
#     progressBar = TQDMProgressBar(refresh_rate=50)
    
#     callbacks = {
#         "progress_bar": progressBar,
#         "checkpoint": checkpointCallback,
#         "graph": graphCallback,
#         "timer": timerCallback
#     }
    
#     tblogger = AnomalibTensorBoardLogger(
#         save_dir=logDir,
#         name=runName,
#         version=version)
    
#     return tblogger, callbacks, os.path.join(checkpointDir, "best.pt")


# class LoggerWriter:
#     def __init__(self, logger, log_level=logging.INFO):
#         self.logger = logger
#         self.log_level = log_level
#         self.linebuf = ''

#     def write(self, buf):
#         for line in buf.rstrip().splitlines():
#             self.logger.log(self.log_level, line.rstrip())

#     def flush(self):
#         pass

# class LoggerStdin:
#     def __init__(self, logger, log_level=logging.INFO):
#         self.logger = logger
#         self.log_level = log_level
#         self.builtin_stdin = sys.stdin

#     def readline(self):
#         line = self.builtin_stdin.readline()
#         self.logger.log(self.log_level, f"Input: {line.rstrip()}")
#         return line

#     def __getattr__(self, attr):
#         return getattr(self.builtin_stdin, attr)

# def find_first_file(directory, target_filename):
#     target_lower = target_filename.lower()
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.lower() == target_lower:
#                 return os.path.join(root, file)
#     return None  # Return None if the file is not found

def get_device():
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def generate_unique_id() -> str:
    return str(uuid.uuid4())

def load_transform_from_yaml(config_path: Path):
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    transform_list: List[T.Transform] = []
    for step in config.get("transform", []):
        transform_name = step["name"]
        transform_args = step.get("args", {})
        transform_class = getattr(T, transform_name)
        transform_list.append(transform_class(**transform_args))

    return T.Compose(transform_list)

def load_metrics_from_yaml(config_path: Path) -> tuple[List[Any], List[Any]]:
    logger.info(f"Loading metrics from {config_path}")
    with open(config_path, "r") as f:
        config: Dict[str, List[Dict[str, Any]]] = yaml.safe_load(f)

    metric_classes: Dict[str, Any] = {
        "AUROC": getattr(__import__("anomalib.metrics", fromlist=["AUROC"]), "AUROC"),
        "AUPR": getattr(__import__("anomalib.metrics", fromlist=["AUPR"]), "AUPR"),
        "AUPRO": getattr(__import__("anomalib.metrics", fromlist=["AUPRO"]), "AUPRO"),
        "AUPIMO": getattr(__import__("anomalib.metrics", fromlist=["AUPIMO"]), "AUPIMO"),
        "F1AdaptiveThreshold": getattr(__import__("anomalib.metrics", fromlist=["F1AdaptiveThreshold"]), "F1AdaptiveThreshold"),
        "ManualThreshold": getattr(__import__("anomalib.metrics", fromlist=["ManualThreshold"]), "ManualThreshold"),
        "F1Score": getattr(__import__("anomalib.metrics", fromlist=["F1Score"]), "F1Score"),
        "F1Max": getattr(__import__("anomalib.metrics", fromlist=["F1Max"]), "F1Max"),
        "PBn": getattr(__import__("anomalib.metrics", fromlist=["PBn"]), "PBn"),
        "PGn": getattr(__import__("anomalib.metrics", fromlist=["PGn"]), "PGn"),
    }

#     - Area Under Curve (AUC) metrics:
#     - ``AUROC``: Area Under Receiver Operating Characteristic curve
#     - ``AUPR``: Area Under Precision-Recall curve
#     - ``AUPRO``: Area Under Per-Region Overlap curve
#     - ``AUPIMO``: Area Under Per-Image Missed Overlap curve

# - F1-score metrics:
#     - ``F1Score``: Standard F1 score
#     - ``F1Max``: Maximum F1 score across thresholds

# - Threshold metrics:
#     - ``F1AdaptiveThreshold``: Finds optimal threshold by maximizing F1 score
#     - ``ManualThreshold``: Uses manually specified threshold

# - Other metrics:
#     - ``AnomalibMetric``: Base class for custom metrics
#     - ``AnomalyScoreDistribution``: Analyzes score distributions
#     - ``BinaryPrecisionRecallCurve``: Computes precision-recall curves
#     - ``Evaluator``: Combines multiple metrics for evaluation
#     - ``MinMax``: Normalizes scores to [0,1] range
#     - ``PBn``: Presorted bad with n% good samples misclassified
#     - ``PGn``: Presorted good with n% bad samples missed
#     - ``PRO``: Per-Region Overlap score
#     - ``PIMO``: Per-Image Missed Overlap score

    def instantiate_metrics(metrics_config: List[Dict[str, Any]]) -> List[Any]:
        metrics: List[Any] = []
        for metric_config in metrics_config:
            metric_type = metric_config["type"]
            fields = metric_config["fields"]
            prefix = metric_config["prefix"]
            metrics.append(metric_classes[metric_type](fields=fields, prefix=prefix))
        return metrics

    val_metrics: List[Any] = instantiate_metrics(config.get("val_metrics", []))
    test_metrics: List[Any] = instantiate_metrics(config.get("test_metrics", []))
    logger.info(f"Loaded val_metrics: {val_metrics}")
    logger.info(f"Loaded test_metrics: {test_metrics}")

    return val_metrics, test_metrics


def get_default_evaluator() -> Evaluator:
    val_metrics, test_metrics = define_metrics()
    return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)


def resolve_config_path(path_str: str, model_yaml_path: Path, config_dir: Optional[Path] = None) -> Path:
    logger.info(f"Resolving config path: {path_str}; config_dir: {config_dir}")
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates: List[Path] = []
    if config_dir is not None:
        candidates.append(config_dir / path)
        candidates.append(config_dir / "Engine" / path)

    candidates.append(model_yaml_path.parent / path)
    if len(model_yaml_path.parents) >= 2:
        candidates.append(model_yaml_path.parents[1] / path)
    if len(model_yaml_path.parents) >= 3:
        candidates.append(model_yaml_path.parents[2] / path)

    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Success; Returning {candidate.resolve()}")
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to resolve path '{path_str}' from model YAML '{model_yaml_path}'"
    )


def copy_file_to_dir(source_path: Path, copy_dir: Path, base_dir: Optional[Path] = None) -> Path:
    logger.info(f"Trying to copy files from {source_path} to {copy_dir}")
    try:
        relative_path = source_path.relative_to(base_dir) if base_dir is not None else source_path.name
    except Exception:
        relative_path = source_path.name

    destination = copy_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    logger.info(f"Success!; Copied to {destination}")
    return destination


def parse_pre_processor(init_args: Dict[str, Any], model_yaml_path: Path, config_dir: Optional[Path]) -> tuple[Any, Optional[Path]]:
    pre_processor_value = init_args.pop("pre_processor_path", None)
    if pre_processor_value is None and "pre_processor" in init_args:
        pre_processor_value = init_args["pre_processor"]

    if pre_processor_value is False:
        return False, None
    if isinstance(pre_processor_value, str):
        resolved = resolve_config_path(pre_processor_value, model_yaml_path, config_dir)
        return PreProcessor(load_transform_from_yaml(resolved)), resolved
    return PreProcessor(), None


def parse_post_processor(init_args: Dict[str, Any], model_yaml_path: Path, config_dir: Optional[Path]) -> tuple[Any, Optional[Path]]:
    post_processor_value = init_args.pop("post_processor_path", None)
    if post_processor_value is None and "post_processor" in init_args:
        post_processor_value = init_args["post_processor"]

    if post_processor_value is False:
        return False, None
    if isinstance(post_processor_value, str):
        resolved = resolve_config_path(post_processor_value, model_yaml_path, config_dir)
        with open(resolved, "r") as f:
            post_processor_config = yaml.safe_load(f)
        return AOIPostProcessor(**post_processor_config), resolved
    return AOIPostProcessor(), None


def parse_evaluator(init_args: Dict[str, Any], model_yaml_path: Path, config_dir: Optional[Path]) -> tuple[Any, Optional[Path]]:
    evaluator_value = init_args.pop("evaluator_path", None)
    if evaluator_value is None and "evaluator" in init_args:
        evaluator_value = init_args["evaluator"]

    if evaluator_value is False:
        return False, None
    if isinstance(evaluator_value, str):
        resolved = resolve_config_path(evaluator_value, model_yaml_path, config_dir)
        return Evaluator(*load_metrics_from_yaml(resolved)), resolved
    return get_default_evaluator(), None


def parse_model_init_args(
    model_yaml_path: Path,
    config_dir: Optional[Path] = None,
    copy_dir: Optional[Path] = None,
) -> tuple[str, Dict[str, Any], Optional[Path], Optional[Path], Optional[Path]]:
    logger.info(f"Parsing model init_args from {model_yaml_path}" + f"; Copying to {copy_dir}" if copy_dir is not None else "")
    with open(model_yaml_path, "r") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("YAML model config must contain a mapping at the top level.")

    model_section = config.get("model")
    if not isinstance(model_section, dict):
        raise ValueError("YAML model config must contain a 'model' section as a mapping.")
    
    class_path:str = model_section.get("class_path", "")

    init_args = dict(model_section.get("init_args", {}))
    if not isinstance(init_args, dict):
        raise TypeError("'model.init_args' must be a mapping.")

    if copy_dir is not None:
        if config_dir is None:
            raise ValueError("copy_dir requires config_dir to be provided")
        copy_file_to_dir(model_yaml_path, copy_dir, base_dir=config_dir)

    pre_processor, pre_processor_path = parse_pre_processor(init_args, model_yaml_path, config_dir)
    post_processor, post_processor_path = parse_post_processor(init_args, model_yaml_path, config_dir)
    evaluator, evaluator_path = parse_evaluator(init_args, model_yaml_path, config_dir)

    init_args["pre_processor"] = pre_processor
    init_args["post_processor"] = post_processor
    init_args["evaluator"] = evaluator

    if copy_dir is not None:
        if pre_processor_path is not None:
            copy_file_to_dir(pre_processor_path, copy_dir, base_dir=config_dir)
        if post_processor_path is not None:
            copy_file_to_dir(post_processor_path, copy_dir, base_dir=config_dir)
        if evaluator_path is not None:
            copy_file_to_dir(evaluator_path, copy_dir, base_dir=config_dir)

    logger.info(f"Successfully parsed config: class_path:{class_path}, init_args: {init_args}")

    return class_path, init_args, pre_processor_path, post_processor_path, evaluator_path

def loadModelConfig(configDir:Path, modelConfigPath:Path, copyDir:Path|None=None, vtype:VisualizerType|None=None, fieldSize:Tuple[int,int]|None=None) -> Tuple[Dict[str,Any], Path|None, Path|None, Path|None]:
    """Load YAML model config and resolve engine config references."""
    class_path, init_args, pre_processor_path, post_processor_path, evaluator_path = parse_model_init_args(
        modelConfigPath,
        config_dir=configDir,
        copy_dir=copyDir,
    )

    if vtype is None:
        init_args["visualizer"] = False
    else:
        if fieldSize is None:
            raise ValueError(f"If Visualiser is to be created a fieldSize needs to be given; {fieldSize} not allowed")
        init_args["visualizer"] = getVisualizer(vtype=vtype, fieldSize=fieldSize)

    with open(modelConfigPath, 'r') as f:
        model_config = yaml.safe_load(f)

    model_config = dict(model_config)
    model_config["model"]["init_args"] = init_args

    return model_config, pre_processor_path, post_processor_path, evaluator_path

def resolve_product_config_path(path_str: str, product_yaml_path: Path, config_dir: Optional[Path] = None, subdir: Optional[str] = None) -> Path:
    """Resolve a path relative to the product yaml path and the general config path. Searches in subdir if available

    Args:
        path_str (str): Name of a file or relative path to a file from the product config yaml
        product_yaml_path (Path): Path to the product configuration YAML file
        config_dir (Optional[Path], optional): Path to the general configuration directory. Defaults to None.
        subdir (Optional[str], optional): Name of the subdirectory to search in. Defaults to None.

    Raises:
        FileNotFoundError: If the resolved path does not exist.

    Returns:
        Path: The resolved path.
    """
    if Path(path_str).is_absolute():
        return Path(path_str)

    candidates: List[Path] = []
    if config_dir is not None:
        if subdir:
            candidates.append(config_dir / subdir / path_str)
        candidates.append(config_dir / path_str)
    candidates.append(product_yaml_path.parent / path_str)
    if subdir is not None:
        candidates.append(product_yaml_path.parent / subdir / path_str)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to resolve product-config path '{path_str}' from '{product_yaml_path}'"
    )

@dataclass
class Image():
    """Class that holds an image to process
    """
    image: NDArray[np.float32]|None = None
    imageSegments: List[NDArray[np.float32]]|None = None

    def addImage(self, image:NDArray[np.float32]):
        self.image = image

    def addImageSegments(self, segments:NDArray[np.float32]):
        segments = np.asarray(segments)
        self.Image = np.concat(segments, axis=0)



                                    # rotational angle of the bounding box of the error; in degree; 0 is parallel to x axis to the right, 90 is parallel to y axis upwards

@dataclass
class DatasetConfig:
    """Dataset parameters"""
    name:str
    category:str
    splits: Tuple[str,str]|Tuple[str]
    path: Path

    @classmethod
    def load_dataset_config(cls, config: Dict[str, Any], product_yaml_path: Path) -> "DatasetConfig":
        name = config.get("name")
        category = config.get("category")
        splits = config.get("split") or config.get("splits")
        path_value = config.get("path")

        if not isinstance(name, str) or not name:
            raise ValueError("dataset.name must be a non-empty string")
        if not isinstance(category, str) or not category:
            raise ValueError("dataset.category must be a non-empty string")
        if not isinstance(splits, list) or not all(isinstance(item, str) for item in splits):
            raise ValueError("dataset.split must be a list of strings")

        if isinstance(path_value, str) and path_value:
            dataset_path = Path(path_value)
            if not dataset_path.is_absolute():
                dataset_path = (product_yaml_path.parent / dataset_path).resolve()
        else:
            dataset_path = (Path("datasets") / name).resolve()

        return DatasetConfig(name=name, category=category, splits=tuple(splits), path=dataset_path)

@dataclass
class DataModuleConfig:
    val_split_mode: ValSplitMode
    test_split_mode: TestSplitMode
    train_batch_size: Optional[int | str] = None  # int, or "auto" to probe the largest batch size that fits in GPU memory
    eval_batch_size: Optional[int | str] = None  # int, or "auto" (see train_batch_size)
    train_augmentations: Optional[List[Any]] = None
    val_augmentations: Optional[List[Any]] = None
    test_augmentations: Optional[List[Any]] = None
    augmentations: Optional[List[Any]] = None
    val_split_ratio: Optional[float] = None
    test_split_ratio: Optional[float] = None
    seed: Optional[int] = None
    num_workers: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d:Dict[str,Any] = {}
        for k,v in asdict(self).items():
            if v is None:
                continue
            if isinstance(v,Enum):
                d[k] = v.value
            else:
                d[k] = v
        return d

    @classmethod
    def extract_datamodule_config(cls, config: Dict[str, Any]) -> "DataModuleConfig":
        """Create a DataModuleConfig instance from a dictionary."""
        return cls(
            train_batch_size=config.get("train_batch_size"),
            eval_batch_size=config.get("eval_batch_size"),
            train_augmentations=config.get("train_augmentations"),
            val_augmentations=config.get("val_augmentations"),
            test_augmentations=config.get("test_augmentations"),
            augmentations=config.get("augmentations"),
            val_split_mode=ValSplitMode(config.get("val_split_mode", ValSplitMode.NONE)),
            val_split_ratio=config.get("val_split_ratio"),
            test_split_mode=TestSplitMode(config.get("test_split_mode", TestSplitMode.NONE)),
            test_split_ratio=config.get("test_split_ratio"),
            seed=config.get("seed"),
            # accept both "num_workers" and the legacy "max_workers" key
            num_workers=config.get("num_workers", config.get("max_workers")),
        )

    @classmethod
    def load_datamodule_config_from_yaml(cls, yaml_path: Path) -> "DataModuleConfig":
        """Load DataModuleConfig from a YAML file."""
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls.extract_datamodule_config(config)

def clipEmbedding(dataset:fod.Dataset):
    model = foz.load_zoo_model(
        "clip-vit-base32-torch"
    )  # load the CLIP model from the zoo

    # Compute embeddings for the dataset
    dataset.compute_embeddings(
        model=model, embeddings_field="clip_embeddings", batch_size=64
    )

    # Dimensionality reduction using UMAP on the embeddings
    fob.compute_visualization(
        dataset, embeddings="clip_embeddings", method="umap", brain_key="clip_vis"
    )

def resnetEmbedding(dataset:fod.Dataset):
    model = foz.load_zoo_model(
        "resnet50-imagenet-torch"
    )  # load the ResNet50 model from the zoo

    # Compute embeddings for the dataset — this might take a while on a CPU
    dataset.compute_embeddings(model=model, embeddings_field="resnet50_embeddings")

    # Dimensionality reduction using UMAP on the embeddings
    fob.compute_visualization(
        dataset,
        embeddings="resnet50_embeddings",
        method="umap",
        brain_key="resnet50_vis",
    )

@dataclass
class DatasetSession:
    """Manages FiftyOne dataset viewing and session lifecycle.
    Separate concern: how data is explored, not how models interact with it.
    """
    FO_DatasetOriginal: fo.Dataset
    FO_Dataset: fo.Dataset
    datasetName: str
    categories: List[str]
    category: Optional[List[str]] = None
    config: Optional[DatasetConfig] = None
    path: Optional[Path] = None
    # FO_DatasetView: Optional[fo.DatasetView] = None
    AL_PredictDataset: Optional[PredictDataset] = None
    currentSession: Optional[fo.Session] = None  # fo.Session

    def __len__(self):
        return self.FO_Dataset.__len__

    @classmethod
    def loadDatasetFromConfig(cls, config:DatasetConfig, overwrite:bool=True, merge:bool=False, split:Optional[Tuple[str]]=None) -> "DatasetSession":
        session = cls.loadDatasetFromDisk(datasetPath=config.path,
                                          datasetName=config.name, 
                                          overwrite=overwrite,
                                          merge=merge,
                                          split=config.splits if split is None else split)
        session.config=config
        return session  

    @classmethod
    def loadDatasetFromDatabase(cls, datasetName: str) -> "DatasetSession":
        """Load a dataset from MongoDB and track it here."""
        if fo.dataset_exists(name=datasetName):
            FO_Dataset = fo.load_dataset(name=datasetName)
            try:
                session:DatasetSession = cls(
                    datasetName=datasetName,
                    categories=list(FO_Dataset.distinct("category.label")),
                    FO_DatasetOriginal=FO_Dataset,
                    FO_Dataset=FO_Dataset
                )
            except:
                logger.error(f"Could not create DatasetSession.")
                raise ValueError
            
            logger.info(f"Loaded dataset '{datasetName}' from database!")
            return session
        else:
            logger.error(f"Dataset '{datasetName}' does not exist in database")
            raise FileNotFoundError()

    @classmethod
    def loadDatasetFromDisk(cls, datasetPath: Path, datasetName:str,  overwrite:bool=True, merge:bool=False, split:Tuple[str,...] = ("train", "test")) -> "DatasetSession":
        """Load a dataset from a given path on the disk and give it a name for the Voxel51 MongoDB.
        Existing dataset in the database can be overwritten or merged with.
        

        Arguments:
            datasetPath -- directory where to find the dataset

        Keyword Arguments:
            datasetName -- Name of the dataset for the database (default: {""})
            overwrite -- overwrite a potential dataset in the database with the same name? (default: {True})
            merge -- merge with a potential dataset in the database with the same name? (default: {False})
            split -- Does the data contain training, testing both or prediction data (default: {("train", "test")})
        """
        datasetSession: DatasetSession|None = None

        AL_PredictDataset:Optional[PredictDataset] = None

        if split == ("pred",):
            FO_Dataset, AL_PredictDataset = importPredictDataset(datasetPath, name=datasetName, overwrite=overwrite)
        else:
            if overwrite and merge:
                logger.info("Overwrite and merge should not both be true. Overwrite is ignored...")
                overwrite = False

            if not datasetPath.exists():
                raise FileNotFoundError(f"Dataset {datasetPath} does not exit")
            elif fo.dataset_exists(datasetName) and not overwrite and not merge:
                logger.info(f"Dataset '{datasetName}' already exists in database")
                logger.info("Loading from database")
                datasetSession = cls.loadDatasetFromDatabase(datasetName=datasetName)  
                FO_Dataset = datasetSession.FO_Dataset  
            elif fo.dataset_exists(datasetName) and overwrite:
                logger.info(f"Dataset '{datasetName}' already exists in database")
                logger.info("Overwriting")
                FO_Dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName,
                    overwrite=overwrite,
                    split=split
                )
            elif fo.dataset_exists(datasetName) and merge:
                logger.info(f"Dataset '{datasetName}' already exists in FiftyOne database")
                logger.info(f"Importing '{datasetName}' dataset from disk")
                now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
                dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName+"_"+str(now),
                    overwrite=False,
                    split=split
                )
                logger.info(f"Importing '{datasetName}' dataset from FiftyOne database")
                FO_Dataset = fo.load_dataset(datasetName) # TODO Safety
                logger.info(f"Merging both datasets")
                FO_Dataset.merge_samples(dataset)
            else:
                logger.info(f"Loading {datasetName} dataset from disk")
                FO_Dataset, _ = importDataset(
                    path=datasetPath,
                    name=datasetName,
                    overwrite=overwrite,
                    split=split
                )

        if datasetSession is not None:
            return datasetSession
        else:
            if FO_Dataset is not None:
                session = DatasetSession(
                    datasetName=datasetName,
                    FO_DatasetOriginal=FO_Dataset,
                    FO_Dataset=FO_Dataset,
                    AL_PredictDataset=AL_PredictDataset,
                    categories=FO_Dataset.distinct("category.label"),
                    path=datasetPath
                    )
                logger.info(f"There are {session.FO_Dataset.count()} images in the {datasetName} dataset.")
                logger.info(f"There are {len(session.categories)} categorie(s) in the {datasetName} dataset.")
                logger.info(session.categories)
                # session.name = datasetName
                return session
            else:
                logger.warning(f"Import was not successfull")
                raise FileNotFoundError()

    def select_category(self, category: str) -> fo.Dataset | None:
        """Select a category and create a view."""
        

        if category == "all":
            self.category = None
            self.FO_DatasetView:fo.DatasetView = self.FO_Dataset.exists("file_path")
        else:
            if category not in (self.categories or []):
                raise AttributeError(f"Category {category} not found! Available: {self.categories}")
            self.category = [category]
            self.FO_DatasetView:fo.DatasetView = self.FO_Dataset.filter_labels("category", F("label").is_in([category]))
        
        logger.info(f"Selected category: {category}, {len(self.FO_DatasetView)} images")

        if self.FO_DatasetView is not None:
            try:
                FO_Dataset_:fo.Dataset = self.FO_DatasetView.clone(name=f"{self.datasetName}-{category}", persistent=True)
            except ValueError as e:

                logger.error(e)
                logger.info(f"Deleting {self.datasetName}-{category} from database and reloading.")
                fo.delete_dataset(f"{self.datasetName}-{category}")
                FO_Dataset_:fo.Dataset = self.FO_DatasetView.clone(name=f"{self.datasetName}-{category}", persistent=True)
            finally:
                if isinstance(FO_Dataset_, fo.Dataset):
                    FO_Dataset = FO_Dataset_
                else:
                    raise ValueError("FO_Dataset_ is not bound. Check for problems with dataset/datasetView (category/product selection)")
                self.FO_Dataset = FO_Dataset
                return self.FO_Dataset
        else:
            return None

        # return FO_Dataset

    def launchSession(self) -> Any:
        """Launch the FiftyOne app for exploration."""

        # if self.FO_DatasetView is not None: # Category is set which generates a datasetview
        #     # we make a separate dataset out of the View since some operations do not work on dataset views
        #     try:
        #         FO_Dataset_ = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
        #     except ValueError as e:
        #         logger.error(e)
        #         logger.info(f"Deleting {self.datasetName}-{self.category} from database and reloading.")
        #         fo.delete_dataset(f"{self.datasetName}-{self.category}")
        #         FO_Dataset_ = self.FO_DatasetView.clone(name=f"{self.datasetName}-{self.category}", persistent=False)
        #     finally:
        #         if isinstance(FO_Dataset_, fo.Dataset):
        #             FO_Dataset = FO_Dataset_
        #         else:
        #             raise ValueError("FO_Dataset_ is not bound. Check for problems with dataset/datasetView (category/product selection)")
        # else:
        #     FO_Dataset = self.FO_Dataset

        self.currentSession = fo.launch_app(self.FO_Dataset)
        logger.info(f"Session: {self.currentSession.server_address}:{self.currentSession.server_port}")
        return self.currentSession

    def get_dataset_for_operation(self) -> fo.Dataset:
        """Returns the dataset to use for training/eval: the view if a category is selected, else full dataset."""
        if self.FO_DatasetView is not None:
            return self.FO_DatasetView
        if self.FO_Dataset is None:
            raise AttributeError("No dataset loaded")
        return self.FO_Dataset
    
    def setupDatamodule(self, datamoduleConfig:DataModuleConfig, outputPath:Path) -> FODataModule:
        """Setup a datamodule from the dataset for running a model on the data

        Arguments:
            outputPath: Path

        Returns:
            _description_
        """

        if self.datasetName == "":
            logger.warning("Dataset name is empty. Using 'unnamedDataset' as dataset name.")
            self.datasetName = "unnamedDataset"
        dm_kwargs = datamoduleConfig.to_dict()
        # "auto" batch sizes are resolved later, per tile, by the tiled ensemble pipeline
        # (it needs the model and tile size to probe GPU memory). Use a safe placeholder
        # here so datamodule construction doesn't receive a non-int batch size.
        for key in ("train_batch_size", "eval_batch_size"):
            if dm_kwargs.get(key) == "auto":
                dm_kwargs[key] = 2
        datamodule = FODataModule(name=self.datasetName, samples=self.FO_Dataset, root=outputPath, **dm_kwargs)
        try:
            datamodule.setup()
            self.datamodule = datamodule
            return datamodule
        except ValueError as e:
            logger.error(f"dm_kwargs: {dm_kwargs}, datamodule: {datamodule}")
            logger.error(f"Name: {self.datasetName}, Root: {outputPath}")
            logger.error(f"Dataset: {self.FO_Dataset}")
            logger.error(f"Error occurred while setting up datamodule: {e}")
            raise DatamoduleError(f"Error occurred while setting up datamodule. Datamodule exists but not setup: {e}")
        except Exception as e:
            raise DatamoduleError(f"Unexpected error occurred while setting up datamodule: {e}")
            
    def generateEmbedding(self) -> None:
        """Generate an embedding of the dataset into a 2d space to visually inspect the data. Opens a voxel51 session

        Raises:
            AttributeError: Needs a dataset to be set
        """

        with exclude_from_logger():
            clipEmbedding(self.FO_Dataset)
        logger.info("Finished embedding computation.")
        logger.info("Please reload the FiftyOne app to see the new visualizations.")
        logger.info("Data already has embedding")
        logger.info("You find the visualizations by clicking the '+' next to Samples and choosing Embeddings.")

    def save(self) -> None:
        """
        Saves the current status of the dataset to the database
        """

        logger.info(f"Saving current state of the dataset {self.datasetName} to the database")
        self.FO_Dataset.save()

@dataclass 
class SeamSmoothingConfig:
    apply: bool = True
    sigma: int = 2
    width: float = 0.1

@dataclass
class TilingPipelineConfig:
    image_size: Tuple[int, int]
    tile_size: Tuple[int, int]
    stride: Tuple[int, int]
    seam_smoothing: SeamSmoothingConfig 
    root_dir:Optional[Path] = None
    normalization_stage: NormalizationStage = NormalizationStage.IMAGE
    thresholding_stage: ThresholdingStage = ThresholdingStage.IMAGE

    def to_dict(self) -> Dict[str, Any]:
        d:Dict[str,Any] = {}
        for k,v in asdict(self).items():
            if v is None:
                continue
            if isinstance(v,Enum):
                d[k] = v.value
            else:
                d[k] = v
        return d

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  image_size={self.image_size}\n"
            f"  tile_size={self.tile_size}\n"
            f"  stride={self.stride}\n"
            f"  seam_smoothing={self.seam_smoothing}\n"
            f"  root_dir={self.root_dir}\n"
            f"  normalization_stage={self.normalization_stage}\n"
            f"  thresholding_stage={self.thresholding_stage}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def _parse_enum(cls, enum_cls: type[Enum], value: Any, default: Enum) -> Enum:
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            return enum_cls(value)
        return default

    @classmethod
    def _parse_tuple(cls, value: Any, name: str) -> Tuple[int, int]:
        try: 
            len(value)
        except:
            logger.error("Length can no be determined.")
            raise ValueError(f"Length can no be determined.")
        if isinstance(value, (List, Tuple)):
            return (int(value[0]), int(value[1]))
        raise ValueError(f"{name} must be a tuple/list of length 2. Got {value}")

    @classmethod
    def _parse_seam_smoothing(cls, value: Any) -> SeamSmoothingConfig:
        if isinstance(value, SeamSmoothingConfig):
            return value
        if isinstance(value, dict):
            return SeamSmoothingConfig(
                apply=bool(value.get("apply", True)),
                sigma=int(value.get("sigma", 2)),
                width=float(value.get("width", 0.1)),
            )
        raise TypeError(f"seam_smoothing must be a SeamSmoothingConfig or a dict. Instead got {type(value)}")

    @classmethod
    def extract_tiling_pipeline_params(cls, config: Dict[str, Any]) -> "TilingPipelineConfig":
        """Create a TilingPipelineConfig instance from a dictionary."""
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        tiling:Dict[str,Any]|None = config.get("tiling")
        if not isinstance(tiling, dict):
            raise TypeError("tiling config must be a dict")

        seam_smoothing:Optional[Dict[str,Any]] = tiling.get("SeamSmoothing", tiling.get("seam_smoothing", None))
        if seam_smoothing is None:
            Warning("Seem smoothing has not been found in manifest.yaml")
            seam_smoothing = {
                "apply" : True,
                "sigma" : 2,
                "width" : 0.1,
            }

        return cls(
            root_dir=config.get("root_dir", None),
            image_size=cls._parse_tuple(tiling.get("image_size"), "image_size"),
            tile_size=cls._parse_tuple(tiling.get("tile_size"), "tile_size"),
            stride=cls._parse_tuple(tiling.get("stride"), "stride"),
            normalization_stage=NormalizationStage(cls._parse_enum(NormalizationStage, tiling.get("normalization_stage"), NormalizationStage.IMAGE)),
            thresholding_stage=ThresholdingStage(cls._parse_enum(ThresholdingStage, tiling.get("thresholding_stage"), ThresholdingStage.IMAGE)),
            seam_smoothing=cls._parse_seam_smoothing(seam_smoothing),
        )

    @classmethod
    def load_tiling_pipeline_config_from_yaml(cls, yaml_path: Path) -> "TilingPipelineConfig":
        """Load TilingPipelineConfig from a YAML file."""
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls.extract_tiling_pipeline_params(config)


@dataclass
class BaseModelConfig:
    backbone: Optional[str] = None
    pre_trained: Optional[bool] = None
    pre_processor: Optional[Any] = False
    post_processor: Optional[Any] = False
    evaluator: Optional[Any] = False
    visualizer: Optional[Any] = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BaseModelConfig":
        # if not isinstance(config, dict):
        #     raise TypeError("config must be a dict")

        kwargs: Dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            if key in config:
                kwargs[key] = config[key]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def load_model_config_from_yaml(
        cls,
        yaml_path: str | Path,
        config_dir: Optional[Path] = None,
        copy_dir: Optional[Path] = None,
    ) -> "BaseModelConfig":
        """Load BaseModelConfig from a model YAML file and resolve engine config references."""
        model_yaml_path = Path(yaml_path)
        if config_dir is None:
            config_dir = model_yaml_path.parents[1] if len(model_yaml_path.parents) > 1 else model_yaml_path.parent

        class_path, init_args, _, _, _ = parse_model_init_args(
            model_yaml_path,
            config_dir=config_dir,
            copy_dir=copy_dir,
        )

        return cls.from_dict(init_args)

@dataclass
class CFAConfig(BaseModelConfig):
    gamma_c: int = 1
    gamma_d: int = 1
    num_nearest_neighbors: int = 3
    num_hard_negative_features: int = 3
    radius: float = 1e-5


@dataclass
class FastFlowConfig(BaseModelConfig):
    flow_steps: int = 8
    conv3x3_only: bool = False
    hidden_ratio: float = 1.0


@dataclass
class PadimConfig(BaseModelConfig):
    layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])
    n_features: Optional[int] = None


@dataclass
class PatchcoreConfig(BaseModelConfig):
    layers: List[str] = field(default_factory=lambda: ["layer2", "layer3"])
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9


@dataclass
class ReverseDistillationConfig(BaseModelConfig):
    layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])
    anomaly_map_mode: Optional[Any] = None


@dataclass
class STFPMConfig(BaseModelConfig):
    layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])


@dataclass
class InpFormerConfig(BaseModelConfig):
    encoder_name: str = "dinov2reg_vit_base_14"
    target_layers: Optional[List[int]] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9])
    fuse_layer_encoder: Optional[List[List[int]]] = field(default_factory=lambda: [[0, 1, 2, 3], [4, 5, 6, 7]])
    fuse_layer_decoder: Optional[List[List[int]]] = field(default_factory=lambda: [[0, 1, 2, 3], [4, 5, 6, 7]])
    remove_class_token: bool = True
    inp_num: int = 6


@dataclass
class GlassConfig(BaseModelConfig):
    input_shape: Tuple[int, int] = (288, 288)
    anomaly_source_path: Optional[str] = None
    pretrain_embed_dim: int = 1536
    target_embed_dim: int = 1536
    patchsize: int = 3
    patchstride: int = 1
    layers: Optional[List[str]] = field(default_factory=lambda: ["layer2", "layer3"])
    pre_projection: int = 1
    discriminator_layers: int = 2
    discriminator_hidden: int = 1024
    learning_rate: float = 0.0001
    step: int = 20
    svd: int = 0
    gaussian_noise_std: float = 0.015
    radius_quantile: float = 0.75
    focal_loss_quantile_threshold: float = 0.5
    mining: bool = True


MODEL_CONFIG_CLASSES: Dict[str, type[BaseModelConfig]] = {
    "cfa": CFAConfig,
    "fastflow": FastFlowConfig,
    "padim": PadimConfig,
    "patchcore": PatchcoreConfig,
    "reversedistillation": ReverseDistillationConfig,
    "reverse_distillation": ReverseDistillationConfig,
    "stfpm": STFPMConfig,
    "inpformer": InpFormerConfig,
    "glass": GlassConfig,
}


@dataclass
class ModelConfig:
    name: str
    config: BaseModelConfig
    preProcessorPath: Optional[Path] = None
    postProcessorPath: Optional[Path] = None
    evaluatorPath: Optional[Path] = None

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        config_dir: Optional[Path] = None,
        copy_dir: Optional[Path] = None,
    ) -> "ModelConfig":
        model_yaml_path = Path(yaml_path)
        if config_dir is None:
            config_dir = (
                model_yaml_path.parents[1]
                if len(model_yaml_path.parents) > 1
                else model_yaml_path.parent
            )

        class_path, init_args, preProcessorPath, postProcessorPath, evaluatorPath = parse_model_init_args(
            model_yaml_path,
            config_dir=config_dir,
            copy_dir=copy_dir,
        )

        # Requires "name" to be present in the YAML's init_args
        # model_name = init_args.get("class_path")
        if not class_path:
            raise ValueError("YAML must contain a 'name' field")

        model_cls = MODEL_CONFIG_CLASSES.get(class_path.lower())
        if model_cls is None:
            raise ValueError(
                f"Unsupported model '{class_path}'. "
                f"Supported: {', '.join(sorted(MODEL_CONFIG_CLASSES))}"
            )

        return cls(name=class_path,
                   config=model_cls.from_dict(init_args),
                   preProcessorPath=preProcessorPath,
                   postProcessorPath=postProcessorPath,
                   evaluatorPath=evaluatorPath)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        # if not isinstance(config, dict):
        #     raise TypeError("config must be a dict")

        model_name = config.get("name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model config requires a non-empty name")

        config = config.get("config", {})
        # if not isinstance(config, dict):
        #     raise TypeError("model config must be a dict")

        model_cls = MODEL_CONFIG_CLASSES.get(model_name.lower())
        if model_cls is None:
            raise ValueError(f"Unsupported model '{model_name}'. Supported models: {', '.join(sorted(MODEL_CONFIG_CLASSES))}")

        return cls(name=model_name,
                   config=model_cls.from_dict(config))

    def to_dict(self) -> Dict[str, Any]:
        return self.config.to_dict()
        # return {
        #     "name": self.name,
        #     "config": self.config.to_dict(),
        # }


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: Any = "auto"
    num_nodes: int = 1
    precision: Optional[int] = None
    logger: Optional[Any] = None
    callbacks: Optional[List[Any]] = None
    fast_dev_run: bool = False
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Optional[Any] = None
    limit_train_batches: Optional[Any] = None
    limit_val_batches: Optional[Any] = None
    limit_test_batches: Optional[Any] = None
    limit_predict_batches: Optional[Any] = None
    overfit_batches: float = 0.0
    val_check_interval: Optional[float] = None
    check_val_every_n_epoch: int = 1
    num_sanity_val_steps: Optional[int] = None
    log_every_n_steps: Optional[int] = None
    enable_checkpointing: Optional[bool] = None
    enable_progress_bar: Optional[bool] = None
    enable_model_summary: Optional[bool] = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    deterministic: Optional[bool] = None
    benchmark: Optional[bool] = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    profiler: Optional[Any] = "simple"
    detect_anomaly: bool = False
    barebones: bool = False
    plugins: Optional[Any] = None
    sync_batchnorm: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: Optional[str] = None
    # enable_autolog_hparams: bool = True
    # model_registry: Optional[Any] = None
    n_parallel_jobs: Optional[int] = None  # tiles to train in parallel; None = one per GPU

    @staticmethod
    def instantiate_callback(callback_spec: Dict[str, Any]) -> Any:
        if not isinstance(callback_spec, dict):
            raise TypeError("callback spec must be a dict")

        class_path = callback_spec.get("class_path")
        if not isinstance(class_path, str):
            raise ValueError("callback spec requires a non-empty class_path string")

        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        callback_cls = getattr(module, class_name)

        init_args = callback_spec.get("init_args", {})
        if not isinstance(init_args, dict):
            raise TypeError("callback init_args must be a dict")

        return callback_cls(**init_args)

    @classmethod
    def instantiate_callbacks(cls, callbacks: Optional[List[Any]]) -> Optional[List[Any]]:
        if callbacks is None:
            return None

        instantiated: List[Any] = []
        for callback in callbacks:
            if isinstance(callback, dict) and "class_path" in callback:
                instantiated.append(cls.instantiate_callback(callback))
            else:
                instantiated.append(callback)
        return instantiated

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainerConfig":
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        kwargs: Dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            if key in config:
                if key == "callbacks":
                    kwargs[key] = cls.instantiate_callbacks(config[key])
                else:
                    kwargs[key] = config[key]
        return cls(**kwargs)

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a dict suitable for passing into Trainer(**kwargs)."""
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def extract_trainer_config(cls, config: Dict[str, Any]) -> "TrainerConfig":
        """Create a TrainerConfig instance from a dictionary."""
        return cls.from_dict(config)

    @classmethod
    def load_trainer_config_from_yaml(cls, yaml_path: Path) -> "TrainerConfig":
        """Load TrainerConfig from a YAML file."""
        with open(yaml_path, 'r') as file:
            config:dict[str,Any] = yaml.safe_load(file)
        if "accelerator" in config.keys():
            if config["accelerator"] == "auto":
                config["accelerator"] = get_device()
        return cls.extract_trainer_config(config)



TModelConfig = TypeVar("TModelConfig", bound=BaseModelConfig)

@dataclass
class Product:
    name: str
    modelConfig: ModelConfig
    modelConfigPath: Path
    modelTrainingDir: Optional[Path]
    tilingPipelineConfig: TilingPipelineConfig
    tilingConfigPath:Optional[Path]
    trainerConfig: TrainerConfig
    trainerConfigPath: Path
    datamoduleConfig: DataModuleConfig
    inferencerConfig: TrainerConfig
    inferencerConfigPath: Path
    datasetConfig: DatasetConfig
    datasetPath:Path
    imageCrop: Optional[Tuple[int, int, int, int]] = None  # (x_min, y_min, x_max, y_max)
    # id: str = generate_unique_id()
    enableTiling: bool = True
    selection: str = "latest"  # how modelWeightsPath/modelTrainingDir were picked when not pinned in the YAML: "latest" or "best"

    def __str__(self) -> str:
        string: str = ""
        for k,v in self.__dict__.items():
            string += f"{k}: {v}\n"
        return string

    def __repr__(self):
        return "\n".join(f"{key}={value}" for key, value in self.__dict__.items())

    def refresh_training_dir(self, baseOutputDir: Path) -> Optional[Path]:
        """
        Re-resolve modelTrainingDir against the current state of `baseOutputDir`,
        e.g. after a train()/eval() call may have changed which run counts as
        "latest"/"best". Only re-runs the resolution lookup - no config YAML is
        re-parsed and no model is rebuilt. Updates self.modelTrainingDir in place.

        If the registry lookup finds nothing (e.g. no runs/*/manifest.yaml
        exists yet under baseOutputDir), the existing modelTrainingDir is left
        untouched instead of being cleared -- this keeps an explicitly pinned
        `model.trainingDir` (set in the product yaml, bypassing the registry
        entirely) working across repeated inference_prep calls.

        Parameters
        ----------
        baseOutputDir : Path
            Root results directory to search for runs (e.g. manager.baseOutputDir)

        Returns
        -------
        _name_ : Optional[Path]
            The freshly resolved training directory, or the existing
            modelTrainingDir if no complete run for this product/model/selection
            was found in the registry.
        """
        try:
            resolvedRunDir, modelName = resolve_run_dir(
                baseOutputDir=Path(baseOutputDir),
                category=self.datasetConfig.category,
                modelName=self.modelConfig.name,
                selection=self.selection,
            )
        except FileNotFoundError:
            resolvedRunDir = None

        if resolvedRunDir is None:
            logger.info(
                f"refresh_training_dir: no complete '{self.selection}' run found for "
                f"{self.modelConfig.name}/{self.datasetConfig.category} under {baseOutputDir}; "
                f"keeping existing modelTrainingDir={self.modelTrainingDir}"
            )
            return self.modelTrainingDir

        self.modelTrainingDir = resolvedRunDir
        # self.modelConfig.name = modelName
        return resolvedRunDir

    def loadTrainer(self, trainerConfigPath:Path) -> None:
        self.trainerConfig:TrainerConfig = TrainerConfig.load_trainer_config_from_yaml(trainerConfigPath)
        self.datamoduleConfig:DataModuleConfig = DataModuleConfig.load_datamodule_config_from_yaml(trainerConfigPath)
        self.trainerConfigPath = trainerConfigPath

    def loadInferencer(self, inferencerConfigPath:Path) -> None:
        self.inferencer:TrainerConfig = TrainerConfig.load_trainer_config_from_yaml(inferencerConfigPath)
        self.datamoduleConfig:DataModuleConfig = DataModuleConfig.load_datamodule_config_from_yaml(inferencerConfigPath)
        self.inferencerConfigPath = inferencerConfigPath

    def loadTiling(self, tilingConfigPath:Path) -> None:
        self.tilingPipelineConfig:TilingPipelineConfig = TilingPipelineConfig.load_tiling_pipeline_config_from_yaml(tilingConfigPath)
        self.tilingConfigPath = tilingConfigPath

def loadProductFromYaml(product_yaml_path: Path, config_dir: Optional[Path] = None, baseOutputDir: Optional[Path] = None) -> Product:
    with product_yaml_path.open("r", encoding="utf-8") as f:
        product_config = yaml.safe_load(f)

    return loadProductFromConfig(product_config, product_yaml_path, config_dir=config_dir, baseOutputDir=baseOutputDir)


def loadProductFromConfig(product_config: Dict[str, Any], product_yaml_path: Path, config_dir: Optional[Path] = None, baseOutputDir: Optional[Path] = None) -> Product:
    """
    Build a Product from an already parsed product description -- the same
    mapping a configs/Products/*.yaml holds (product, model, tiling, trainer,
    inferencer, dataset), wherever it came from. loadProductFromYaml reads it
    from a file; AD_Worker's product_load_catalog builds it from the product's
    catalog row instead (see core/database.py's products table).

    Parameters
    ----------
    product_config : Dict[str, Any]
        The product description, in the product YAML's layout.
    product_yaml_path : Path
        Where the description lives, or would live -- only used as an anchor
        to resolve relative config/dataset paths against, so the file itself
        does not have to exist when the description did not come from one.
    config_dir : Optional[Path]
        General configuration directory (usually 'configs').
    baseOutputDir : Optional[Path]
        Results root, to resolve the 'latest'/'best' run from when the
        description does not pin model.trainingDir.
    """
    training_dir: Path | None
    modelName: str | None
    modelConfigFile: str | None
    selection: str

    if not isinstance(product_config, dict):
        raise TypeError("product YAML must contain a mapping at the top level")

    if config_dir is None:
        config_dir = product_yaml_path.parent.parent
        logger.info(f"Config dir not given for loadProductFromYaml. Choosing the double parent directory of the product.yaml: {config_dir}.")

    product_name:str = product_config.get("product")
    if not isinstance(product_name, str) or not product_name:
        raise ValueError("product field must be a non-empty string")

    model_section = product_config.get("model")
    if not isinstance(model_section, dict):
        raise TypeError("model config must be a dict")

    # If the key 'config' is present, validate it's a non-empty string.
    # If the key is missing entirely, leave modelConfigFile as None for the caller to handle.
    if isinstance(model_section, dict) and "config" in model_section:
        modelConfigFile = model_section["config"]
        if not isinstance(modelConfigFile, str) or not modelConfigFile:
            raise ValueError("model.config must be a non-empty string")
    else:
        modelConfigFile = None

    if isinstance(model_section, dict) and "name" in model_section:
        modelName = model_section["name"]
        if not isinstance(modelName, str) or not modelName:
            raise ValueError("model.name must be a non-empty string")
    else:
        modelName = None

    selection = model_section.get("selection", "latest")
    if selection not in ("latest", "best"):
        raise ValueError(f"model.selection must be 'latest' or 'best', got {selection!r}")

    # with model_config_path.open("r", encoding="utf-8") as f:
    #     model_yaml = yaml.safe_load(f)
    # if not isinstance(model_yaml, dict):
    #     raise TypeError("model YAML must contain a mapping at the top level")

    # model_class_path = model_yaml.get("model", {}).get("class_path")
    # if not isinstance(model_class_path, str) or not model_class_path:
    #     raise ValueError("model YAML must contain model.class_path")

    # normalized_model_key = model_class_path.lower().replace("_", "")
    # model_cls = MODEL_CONFIG_CLASSES.get(normalized_model_key)
    # if model_cls is None:
    #     raise ValueError(f"Unsupported model '{model_class_path}' in model YAML")


    # modelConfig = model_cls.load_model_config_from_yaml(model_config_path, config_dir=config_dir)
    dataset_section = product_config.get("dataset", None)

    if dataset_section is None:
        raise AttributeError("No dataset section found in product yaml.")
    category = str(dataset_section.get("category", None)) if isinstance(dataset_section, dict) else None

    training_dir = model_section.get("trainingDir", None)

    _modelName: str | None = None
    if training_dir is None:
        if baseOutputDir is None:
            raise ValueError(
                "trainingDir is not set in the product and baseOutputDir is not given to loadProductFromYaml function."
                "trainingDir must be given explicitly, or baseOutputDir must be given so the "
                f"'{selection}' run can be resolved automatically"
            )
        # resolve_run_dir returns a tuple (training_dir, modelName)

        training_dir, _modelName = resolve_run_dir(
            baseOutputDir=Path(baseOutputDir),
            category=category,
            modelName=modelName,
            selection=selection,
        )

    # True only when the user gave neither model.name nor model.config and let
    # `selection` pick the model itself (not just which run of a pinned model
    # to use) -- the "I don't care which model, just give me the best one"
    # case. The trainer section is model-specific (hyperparameters like
    # max_epochs/batch size differ per model), so whatever the yaml has under
    # `trainer.config` can't be trusted here: it was written for whatever
    # model the author had in mind, not necessarily the one selection ends up
    # resolving to. See the trainer_config_name derivation below.
    model_auto_resolved = modelConfigFile is None and modelName is None

    if modelConfigFile is None:
        if modelName is None:
            if _modelName is None:
                raise ValueError(
                    f"modelConfigFile and modelName are not set in the product and could not be resolved from selection {selection}."
                    "Either modelConfigFile or modelName must be given explicitly, or selection must be given so the "
                    f"'{selection}' run can be resolved automatically"
                )
            else:
                modelName = _modelName
                modelConfigFile = f"{modelName}.yaml"
        else:
            modelConfigFile = f"{modelName}.yaml"
    else:
        if modelName is None:
            modelName = Path(modelConfigFile).stem
        else:
            if modelName != Path(modelConfigFile).stem:
                raise Warning(
                    f"modelConfigFile '{modelConfigFile}' and modelName '{modelName}' are inconsistent. "
                )

    model_config_path = resolve_product_config_path(modelConfigFile, product_yaml_path, config_dir, subdir="Models")

    modelConfig:ModelConfig = ModelConfig.from_yaml(model_config_path, config_dir=config_dir)


    # if weights_path
    # model_weights_path = Path(weights_path)
    # if not model_weights_path.is_absolute():
    #     model_weights_path = model_weights_path.resolve()

    model_training_dir:Path|None
    if training_dir is not None:
        model_training_dir = Path(training_dir)
        if not model_training_dir.is_absolute():
            model_training_dir = model_training_dir.resolve()
    else:
        print("No training directory found. Either because it is not given and no dir could be found automatically or because given one does not exist.")
        logger.info("No training directory found. Either because it is not given and no dir could be found automatically or because given one does not exist.")
        model_training_dir = None

    tiling_section = product_config.get("tiling", {})
    if not isinstance(tiling_section, dict):
        raise TypeError("tiling config must be a dict")
    enable_tiling = bool(tiling_section.get("enable", False))
    if enable_tiling:
        tiling_config_name = tiling_section.get("config")
        if not isinstance(tiling_config_name, str) or not tiling_config_name:
            raise ValueError("tiling.config must be a non-empty string when tiling.enable is true")
        tiling_config_path = resolve_product_config_path(tiling_config_name, product_yaml_path, config_dir, subdir="Tiling")
        tiling_pipeline_config = TilingPipelineConfig.load_tiling_pipeline_config_from_yaml(tiling_config_path)
    else:
        tiling_pipeline_config = TilingPipelineConfig(
            image_size=(0, 0),
            tile_size=(0, 0),
            stride=(0, 0),
        )
        tiling_config_path = None

    trainer_section = product_config.get("trainer", {})
    if not isinstance(trainer_section, dict):
        raise TypeError("trainer config must be a dict")
    trainer_config_name = trainer_section.get("config")

    if model_auto_resolved:
        # Model type was picked by `selection`, not pinned by the user, so
        # derive the matching trainer config from the resolved model name
        # (same "<Name>.yaml" convention as the model config itself) instead
        # of trusting a possibly stale/mismatched trainer.config in the yaml.
        resolved_trainer_config_name = f"Training_{modelName}.yaml"
        if isinstance(trainer_config_name, str) and trainer_config_name and trainer_config_name != resolved_trainer_config_name:
            logger.info(
                f"model.selection resolved model '{modelName}' automatically; ignoring "
                f"trainer.config '{trainer_config_name}' from the product yaml (trainer "
                f"settings are model-specific) and using '{resolved_trainer_config_name}' instead."
            )
        trainer_config_name = resolved_trainer_config_name
    elif not isinstance(trainer_config_name, str) or not trainer_config_name:
        raise ValueError("trainer.config must be a non-empty string")

    trainer_config_path = resolve_product_config_path(trainer_config_name, product_yaml_path, config_dir, subdir="Trainer")
    trainer_config = TrainerConfig.load_trainer_config_from_yaml(trainer_config_path)
    datamoduleConfig = DataModuleConfig.load_datamodule_config_from_yaml(trainer_config_path)
    datasetConfig = DatasetConfig.load_dataset_config(dataset_section, product_yaml_path)
    datasetPath = datasetConfig.path

    inferencer_section = product_config.get("inferencer", None)
    if inferencer_section is None:
        raise ValueError("Need Inferencer for product. add `inferencer` section with path to `inferencer.yaml`")
    else:
        if not isinstance(inferencer_section, dict):
            # logger.warning("Inferencer section given but section is not a dict.")
            raise TypeError("inferencer config must be a dict")
        inferencer_config_name = inferencer_section.get("config")
        if not isinstance(inferencer_config_name, str) or not inferencer_config_name:
            raise ValueError("inferencer.config must be a non-empty string")
        inferencer_config_path = resolve_product_config_path(inferencer_config_name, product_yaml_path, config_dir, subdir="Trainer")
        inferencer_config = TrainerConfig.load_trainer_config_from_yaml(inferencer_config_path)

    return Product(
        name=product_name,
        modelConfig=modelConfig,
        modelConfigPath=model_config_path,
        modelTrainingDir=model_training_dir,
        tilingPipelineConfig=tiling_pipeline_config,
        tilingConfigPath=tiling_config_path,
        trainerConfig=trainer_config,
        trainerConfigPath=trainer_config_path,
        datamoduleConfig=datamoduleConfig,
        inferencerConfig=inferencer_config,
        inferencerConfigPath=inferencer_config_path,
        datasetConfig=datasetConfig,
        datasetPath=datasetPath,
        enableTiling=enable_tiling,
        selection=selection,
    )

