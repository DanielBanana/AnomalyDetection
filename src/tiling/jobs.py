"""Collection of obs for the tiles ensemble"""

# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Changed by Daniel Pommer, TH Nuremberg, 2026

# GENERAL
from enum import Enum
import json
import logging
import fiftyone as fo
import numpy as np
import pandas as pd
import torch

from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from tqdm import tqdm
from typing import Any, Tuple, Dict, List
from torch import Tensor, nn
from torchvision.tv_tensors import Image
from torchvision.transforms.v2 import Compose, Resize, Transform
from lightning import LightningModule, Trainer
from torchvision.tv_tensors import Mask

from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage
from anomalib.metrics.evaluator import Evaluator




# ANOMALIB
from anomalib.post_processing import PostProcessor
from anomalib.visualization import ImageVisualizer, visualize_image_item
from anomalib.visualization.image.item_visualizer import (
    DEFAULT_FIELDS_CONFIG,
    DEFAULT_OVERLAY_FIELDS_CONFIG,
    DEFAULT_TEXT_CONFIG,
)
from anomalib.data import ImageBatch, ImageItem
from anomalib.pre_processing.utils.transform import get_exportable_transform
from anomalib.utils.path import generate_output_filename
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage, PredictData as InferenceData
from anomalib.pipelines.tiled_ensemble.components.thresholding import ThresholdingJob
from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler
from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import get_ensemble_tiler, get_threshold_values
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS, PREV_STAGE_RESULT
from anomalib.models.components import AnomalibModule
from anomalib.utils.normalization.min_max import normalize

# OWN CODE
from data.anomaly_datasets import FODataModule
from setup import create_model, TilingPipelineConfig, ModelConfig

logger = logging.getLogger(__name__)

class Split(Enum):
    """Enum for dataset splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    INFERENCE = "inference"


class AOIStatisticsJob(Job):
    """Job for calculation statistics about the quality of evaluations.

    Args:
        predictions: Object containing ensemble predictions.
        root_dir:
    """

    name = "Statistics"

    def __init__(self, prev_stage_result: List[ImageBatch], root_dir: Path) -> None:
        self.predictions = prev_stage_result
        self.root_dir = root_dir

    def run(self, task_id: int | None = None) -> Tuple[List[ImageBatch], Dict[str,Any]]:
        """Run job that calculates statistics needed in post-processing steps.

        Args:
            task_id: Not used in this case

        Returns:
            dict: Statistics dict with min, max and threshold values.
        """
        del task_id  # not needed here

        logger.info("Starting post-processing statistics job calculation.")
        logger.info("Calculating image/pixel thresholds, pred_score and anomaly_map")

        logger.debug(f"{self.name}: {self.predictions[0].image[0]}")

        post_processor = PostProcessor()
        for batch in tqdm(self.predictions, desc="Stats calculation"):
            # update minmax and thresholds
            post_processor.on_validation_batch_end(trainer=Trainer(), pl_module=LightningModule(), outputs=batch)

        post_processor.on_validation_epoch_end(trainer=Trainer(), pl_module=LightningModule(),)

        # return stats with save path that is later used to save statistics.
        return self.predictions, {
            "minmax": {
                "pred_score": {
                    "min": post_processor.image_min.item(),
                    "max": post_processor.image_max.item(),
                },
                "anomaly_map": {
                    "min": post_processor.pixel_min.item(),
                    "max": post_processor.pixel_max.item(),
                },
            },
            "image_threshold": post_processor.image_threshold.item(),
            "pixel_threshold": post_processor.pixel_threshold.item(),
            "save_path": (self.root_dir / "stats.json"),
        }
    
    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            dict: statistics dictionary.
        """
        return results[0]   # take first element since Statistics job is not run in parallel

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Save statistics to file system."""
        
        res:dict[str,Any] = results[1] # the run job is returning something else than the statistics, assume the statistics are the second thing thing

        # get and remove path from stats dict
        stats_path: Path = res.pop("save_path")
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"StatisticsJob: Save results to path: {stats_path}")

        # save statistics next to weights
        with stats_path.open("w", encoding="utf-8") as stats_file:
            json.dump(res, stats_file, ensure_ascii=False, indent=4)
    
class AOIStatisticsJobGenerator(JobGenerator):
    """Generate StatisticsJob.

    Args:
        root_dir (Path): Root directory where statistics file will be saved (in weights folder).
    """

    def __init__(
        self,
        root_dir: Path,
    ) -> None:
        self.root_dir = root_dir

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIStatisticsJob

    def generate_jobs(
        self,
        args: Dict[str,Any] | None = None,
        prev_stage_result: List[ImageBatch] | None = None,
    ) -> Generator[AOIStatisticsJob, None, None]:
        """Return a generator producing a single stats calculating job.

        Args:
            args: Not used here.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[StatisticsJob, None, None]: StatisticsJob generator.
        """
        del args  # not needed here

        yield AOIStatisticsJob(
            prev_stage_result=prev_stage_result,
            root_dir=self.root_dir,
        )



# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - prediction merging job."""

class AOIMergeJob(Job):
    """Job for merging tile-level predictions into image-level predictions.

    Args:
        predictions (EnsemblePredictions): Object containing ensemble predictions.
        tiler (EnsembleTiler): Ensemble tiler used for untiling.
    """

    name = "Merge"

    def __init__(self, predictions: EnsemblePredictions, tiler: EnsembleTiler) -> None:
        super().__init__()
        self.predictions = predictions
        self.tiler = tiler

    def run(self, task_id: int | None = None) -> List[ImageBatch]:
        """Run merging job that merges all batches of tile-level predictions into image-level predictions.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: List of merged predictions.
        """
        del task_id  # not needed here
        logger.info("Starting merging job to combine tile results.")

        merger = PredictionMergingMechanism(self.predictions, self.tiler)
        merged_predictions:List[ImageBatch] = [
            merger.merge_tile_predictions(batch_idx)
            for batch_idx in tqdm(range(merger.num_batches), desc="Prediction merging")
        ]
        logger.debug(f"{self.name}: number of batches: {merger.num_batches}")
        # logger.debug(f"{self.name}: Sample merged predition: {merged_predictions[0].image[0]}")
        
        return merged_predictions  # noqa: RET504

    @staticmethod
    def collect(results: List[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: List of prediction batches.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing to save in this job."""


class AOIMergeJobGenerator(JobGenerator):
    """Generate MergeJob."""

    def __init__(self, tilingPipelineConfig: TilingPipelineConfig) -> None:
        super().__init__()
        self.tilingPipelineConfig = tilingPipelineConfig

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIMergeJob

    def generate_jobs(
        self,
        args: Dict[str, Any] | None = None,
        prev_stage_result: EnsemblePredictions | None = None,
    ) -> Generator[AOIMergeJob, None, None]:
        """Return a generator producing a single merging job.

        Args:
            args (dict): Tiled ensemble pipeline args.
            prev_stage_result (EnsemblePredictions): Ensemble predictions from predict step.

        Returns:
            Generator[AOIMergeJob, None, None]: AOIMergeJob generator
        """
        del args  # args not used here

        tiler = get_ensemble_tiler(self.tilingPipelineConfig.to_dict())
        if prev_stage_result is not None:
            yield AOIMergeJob(prev_stage_result, tiler)
        else:
            msg = "Merging job requires tile level predictions from previous step."
            raise ValueError(msg)


# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class used as mechanism to merge ensemble predictions from each tile into complete whole-image representation."""
class PredictionMergingMechanism:
    """Class used for merging the data predicted by each separate model of tiled ensemble.

    Tiles are stacked in one tensor and untiled using Ensemble Tiler.
    Boxes from tiles are either stacked or generated anew from anomaly map.
    Labels are combined with OR operator, meaning one anomalous tile -> anomalous image.
    Scores are averaged across all tiles.

    Args:
        ensemble_predictions (EnsemblePredictions): Object containing predictions on tile level.
        tiler (EnsembleTiler): Tiler used to transform tiles back to image level representation.

    Example:
        >>> from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler
        >>> from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
        >>>
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> data = EnsemblePredictions()
        >>> merger = PredictionMergingMechanism(data, tiler)
        >>>
        >>> # we can then start merging procedure for each batch
        >>> merger.merge_tile_predictions(0)
    """

    def __init__(self, ensemble_predictions: EnsemblePredictions, tiler: EnsembleTiler) -> None:
        assert ensemble_predictions.num_batches > 0, "There should be at least one batch for each tile prediction."
        assert (0, 0) in ensemble_predictions.get_batch_tiles(
            0,
        ), "Tile prediction dictionary should always have at least one tile"

        self.ensemble_predictions = ensemble_predictions
        self.num_batches = self.ensemble_predictions.num_batches

        self.tiler = tiler

    def merge_tiles(self, batch_data: dict, tile_key: str) -> Tensor:
        """Merge tiles back into one tensor and perform untiling with tiler.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.
            tile_key (str): Key used in prediction dictionary for tiles that we want to merge.

        Returns:
            Tensor: Tensor of tiles in original (stitched) shape.
        """
        # batch of tiles with index (0, 0) always exists, so we use it to get some basic information
        first_tiles = getattr(batch_data[0, 0], tile_key)
        batch_size = first_tiles.shape[0]
        device = first_tiles.device

        single_channel = False
        if len(first_tiles.shape) == 3:
            single_channel = True
            # in some cases, we don't have channels but just B, H, W
            merged_size = [
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            ]
        else:
            # some tiles also have channels
            num_channels = first_tiles.shape[1]
            merged_size = [
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                int(num_channels),
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            ]

        # create new empty tensor for merged tiles
        merge_buffer = torch.zeros(size=merged_size, device=device)

        # insert tile into merged tensor at right locations
        for (tile_i, tile_j), tile_data in batch_data.items():
            merge_buffer[tile_i, tile_j, ...] = getattr(tile_data, tile_key)

        if single_channel:
            # add channel as tiler needs it
            merge_buffer = merge_buffer.unsqueeze(3)

        # stitch tiles back into whole, output is [B, C, H, W]
        merged_output = self.tiler.untile(merge_buffer)

        if single_channel:
            # remove previously added channels
            merged_output = merged_output.squeeze(1)

        return merged_output

    def merge_labels_and_scores(self, batch_data: dict) -> dict[str, Tensor]:
        """Join scores and their corresponding label predictions from all tiles for each image.

        Label merging is done by rule where one anomalous tile in image results in whole image being anomalous.
        Scores are averaged over tiles.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.

        Returns:
            dict[str, Tensor]: Dictionary with "pred_labels" and "pred_scores"
        """
        # create accumulator with same shape as original
        labels = torch.zeros(batch_data[0, 0].pred_label.shape, dtype=torch.bool)
        scores = torch.zeros(batch_data[0, 0].pred_score.shape)

        for curr_tile_data in batch_data.values():
            curr_labels = curr_tile_data.pred_label
            curr_scores = curr_tile_data.pred_score

            labels = labels.logical_or(curr_labels)
            scores += curr_scores

        scores /= self.tiler.num_tiles

        return {"pred_label": labels, "pred_score": scores}

    @staticmethod
    def merge_gt_labels(batch_data: dict) -> Any:
        """The image's ground-truth label from its tiles' labels: anomalous if any
        tile is. Each tile is labelled by its own part of the mask (TileCollater in
        ensemble_tiling.py), so tile (0, 0) alone no longer tells; "any tile" gives
        back exactly the image's label. None if the tiles carry no label."""
        labels = [tile.gt_label for tile in batch_data.values() if getattr(tile, "gt_label", None) is not None]
        if not labels:
            return None
        merged = labels[0].bool()
        for label in labels[1:]:
            merged = merged.logical_or(label.bool())
        return merged.to(labels[0].dtype)

    def merge_tile_predictions(self, batch_index: int) -> ImageBatch:
        """Join predictions from ensemble into whole image level representation for batch at index batch_index.

        Args:
            batch_index (int): Index of current batch.

        Returns:
            dict[str, Tensor | list]: List of merged predictions for specified batch.
        """
        current_batch_data = self.ensemble_predictions.get_batch_tiles(batch_index)

        # take first tile as base prediction, keep items that are the same over all tiles:
        # image_path, mask_path
        merged_predictions = {
            "image_path": current_batch_data[0, 0].image_path,
            "gt_label": self.merge_gt_labels(current_batch_data),
        }
        if hasattr(current_batch_data[0, 0], "mask_path"):
            merged_predictions["mask_path"] = current_batch_data[0, 0].mask_path
        
        tiled_data = ["image"]

        # What if the batch does not have a ground truth (gt) mask??
        if hasattr(current_batch_data[0, 0], "gt_mask") and current_batch_data[0,0].gt_mask is not None:
            tiled_data += ["gt_mask"]
        if hasattr(current_batch_data[0, 0], "anomaly_map") and current_batch_data[0, 0].anomaly_map is not None:
            tiled_data += ["anomaly_map"]
        if hasattr(current_batch_data[0, 0], "pred_mask") and current_batch_data[0, 0].pred_mask is not None:
            tiled_data += ["pred_mask"]

        # merge all tiled data
        for t_key in tiled_data:
            if hasattr(current_batch_data[0, 0], t_key):
                # print(t_key)
                merged_predictions[t_key] = self.merge_tiles(current_batch_data, t_key)

        # label and score merging
        merged_scores_and_labels = self.merge_labels_and_scores(current_batch_data)
        merged_predictions["pred_label"] = merged_scores_and_labels["pred_label"]
        merged_predictions["pred_score"] = merged_scores_and_labels["pred_score"]

        return ImageBatch(**merged_predictions)


# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - metrics calculation job."""

class AOIMetricsCalculationJob(Job):
    """Job for image and pixel metrics calculation.

    Args:
        accelerator (str): Accelerator (device) to use.
        predictions (list[Any]): List of batch predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    name = "Metrics"

    def __init__(
        self,
        accelerator: str,
        prev_stage_result: Tuple[List[ImageBatch],Any] | List[ImageBatch],
        root_dir: Path,
        evaluator: Evaluator,
        split: Split,
        saveName: str = "metric_results.csv"
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        if isinstance(prev_stage_result, Tuple):
            self.predictions = prev_stage_result[0]
            self.rest = prev_stage_result[1:]
        else:
            self.predictions = prev_stage_result
            self.rest = None
        self.root_dir = root_dir
        self.evaluator = evaluator
        self.saveName = saveName
        self.split = split


    def run(self, task_id: int | None = None) -> Tuple[List[ImageBatch],Dict[str,Any]]:
        """Run a job that calculates image and pixel level metrics.

        Args:
            task_id: Not used in this case.

        Returns:
            dict[str, float]: Dictionary containing calculated metric values.
        """
        del task_id  # not needed here

        logger.info(f"Starting {self.name}!.")
        logger.debug(f"{self.name}: Sample: {self.predictions[0].image}")
        logger.debug(f"{self.name}: Sample: {self.predictions[0].anomaly_map}")


        if self.split == Split.TEST:
            for batch in tqdm(self.predictions, desc="Calculating metrics"):
                self.evaluator.on_test_batch_end(None, None, None, batch=batch, batch_idx=0)

                # compute all metrics on specified accelerator
                metrics_dict:Dict[str,Any] = {}
                for metric in self.evaluator.test_metrics:
                    metric.to(self.accelerator)
                    metrics_dict[metric.name] = metric.compute().item()
                    metric.cpu()

        else:
            # split has to be Split.VAL
            for batch in tqdm(self.predictions, desc="Calculating metrics"):
                self.evaluator.on_validation_batch_end(None, None, None, batch=batch, batch_idx=0)

            # compute all metrics on specified accelerator
            metrics_dict:Dict[str,Any] = {}
            for metric in self.evaluator.val_metrics:
                metric.to(self.accelerator)
                metrics_dict[metric.name] = metric.compute().item()
                metric.cpu()

        for name, value in metrics_dict.items():
            print(f"{name}: {value:.4f}")

        # save path used in `save` method
        metrics_dict["save_path"] = self.root_dir / self.saveName

        return self.predictions, metrics_dict

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: list of predictions.
        """
        # take the first element as the SerialRunner this Job is run with creates a list of Runs (there is only one in this case)
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Save metrics values to csv."""
        logger.info("Saving metrics to csv.")

        # get and remove path from stats dict
        results = results[1] # Results are a tuple of presdictions and the metric stats we want to save
        results_path: Path = results.pop("save_path")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        df_dict = {k: [v] for k, v in results.items()}
        metrics_df = pd.DataFrame(df_dict)
        metrics_df.to_csv(results_path, index=False)

class AOIMetricsCalculationJobGenerator(JobGenerator):
    """
    Generates the Job to Calculate the metrics to determine the quality of the anomaly detection model
    + thresholds

    Parameters
    ----------
    JobGenerator : class
        Original JobGenerator class we inherit from
    """

    def __init__(
        self,
        accelerator: str,
        root_dir: Path,
        modelConfig: ModelConfig,
        tile_size: Tuple[int,int],
        split:Split,
        saveName: str = "metric_results.csv"
    ) -> None:
        """
        Generates the Job to Calculate the metrics to determine the quality of the anomaly detection model
        + thresholds

        Parameters
        ----------
        accelerator : str
            On what kind of hardware to run the job
        root_dir : Path
            In which directory the results are saved
        modelConfig : ModelConfig
            Configuration dataclass object the describes the anomaly detection model 
        tile_size : Tuple[int,int]
            Size of the tile we split the images into
        saveName : str (optional)
            Name of the file that contains the result metrics like AUROC or F1 score. Default is `"metric_results.csv"`
        """
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.modelConfig = modelConfig
        self.tile_size = tile_size
        self.saveName = saveName
        self.split = split

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIMetricsCalculationJob

    def generate_jobs(
        self,
        args: Dict[str,Any] | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[AOIMetricsCalculationJob, None, None]:
        """Make a generator that yields a single metrics calculation job.

        Args:
            args: ensemble run config.
            prev_stage_result: ensemble predictions from previous step.

        Returns:
            Generator[MetricsCalculationJob, None, None]: MetricsCalculationJob generator
        """
        del args  # args not used here

        model = get_ensemble_model(self.modelConfig, normalization_stage=NormalizationStage.IMAGE, input_size=self.tile_size)

        if model.evaluator is not None:
            yield AOIMetricsCalculationJob(
                accelerator=self.accelerator,
                prev_stage_result=prev_stage_result,
                root_dir=self.root_dir,
                evaluator=cast(Evaluator,model.evaluator),
                saveName=self.saveName,
                split=self.split
            )
        else:
            msg = "Model passed to tiled ensemble has no evaluator module which is required to calculate metrics."
            raise RuntimeError(msg)
        

def get_ensemble_model(
    modelConfig: ModelConfig,
    input_size: Tuple[int,int],
    normalization_stage: NormalizationStage,
) -> AnomalibModule:
    """Get model prepared for ensemble training.

    Args:
        model_args (dict): tiled ensemble model configuration.
        input_size (int | tuple[int, int]): individual model input size.
        normalization_stage (NormalizationStage): stage when normalization performed.

    Returns:
        AnomalyModule: model with input_size setup
    """



    # # Take evaluator out of model args if possible
    # post_processor:PostProcessor|bool = model_args["init_args"].pop("post_processor", False)
    # pre_processor:PreProcessor|bool = model_args["init_args"].pop("pre_processor", False)        # TODO Find a way to preserve settings while also overwriting input size with tiling size
    # evaluator:Evaluator|bool = model_args["init_args"].pop("evaluator", False)
    # visualizer = model_args["init_args"].pop("visualizer", False) 
    # if isinstance(visualizer, ImageVisualizer):
    #     visualizer.field_size = input_size

    # if not isinstance(evaluator, Evaluator):
    #     image_auroc_val = AUROC(fields=["pred_score", "gt_label"], prefix="image_val_")
    #     pixel_auroc_val = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_val_")
    #     image_auroc_test = AUROC(fields=["pred_score", "gt_label"], prefix="image_test_")
    #     pixel_auroc_test = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_test_")
    #     evaluator = Evaluator(val_metrics=[image_auroc_val, pixel_auroc_val], test_metrics=[image_auroc_test, pixel_auroc_test])

    # if isinstance(post_processor, PostProcessor):
    #     # set model normalisation only if the stage is set to tile level (but thresholding is always applied)
    #     post_processor.enable_normalization = normalization_stage == NormalizationStage.TILE
        
    # # first make temporary model to get object
    # temp_model = get_model(model_args)
    # logger.info(f"Configuring model {temp_model.__class__.__name__} for ensemble with input size {input_size} and {model_args}")

    # # create custom pre_proc with correct input size
    # # since we can't modify input_size directly (needed during instantiation by some models like FastFlow)
    # logger.info(f"Configuring pre-processor for ensemble model with input size {input_size}")
    # _pre_processor = temp_model.configure_pre_processor(image_size=input_size, crop_size=input_size[0])
    
    # name = model_args["class_path"]

    model: AnomalibModule|None = create_model(modelConfig.name, modelConfig=modelConfig.to_dict())
    assert model is not None
    
    # _pre_processor = tmp_model.configure_pre_processor(image_size=input_size)

    if isinstance(model.visualizer, ImageVisualizer):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        model.visualizer.field_size = input_size
    if isinstance(model.post_processor, PostProcessor):
        model.post_processor.enable_normalization = normalization_stage == NormalizationStage.TILE

    # model: AnomalibModule = get_model(name, pre_processor=_pre_processor, visualizer=visualizer, evaluator=evaluator, post_processor=post_processor, **model_args["init_args"])
    if model.pre_processor is not None:
        model_pre_processor: nn.Module = model.pre_processor

        # drop Resize in all cases since it gets copied to datamodule, and we don't want that!
        pre_transforms = model_pre_processor.transform
        if isinstance(pre_transforms, Resize):
            update_transform = []
        elif isinstance(pre_transforms, Compose):
            update_transform = Compose([
                transform for transform in pre_transforms.transforms if not isinstance(transform, Resize)
            ])
        else:
            update_transform = pre_transforms

        # elif pre_transforms is not None:
        #     update_transform = pre_transforms
        # else:
        #     update_transform = []

        model_pre_processor.transform = update_transform
        model_pre_processor.export_transform = get_exportable_transform(update_transform)

    # model_args["init_args"]["post_processor"] = post_processor
    # model_args["init_args"]["pre_processor"] = pre_processor
    # model_args["init_args"]["evaluator"] = evaluator
    # model_args["init_args"]["visualizer"] = visualizer

    return model

# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - visualization job."""
class AOIVisualizationJob(Job):
    """Job for visualization of predictions.

    Args:
        predictions (list[Any]): list of image-level predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        data_args (Dict): data args used to get data name and category name.
    """

    name = "VisualizeOnDisk"

    def __init__(self, prev_stage_result: Tuple[List[ImageBatch], dict[str, Any]] | List[ImageBatch], root_dir: Path, datasetName:str, category:str,  visualisationArgs:dict[str,Any], predMaskImage:bool=False) -> None:
        super().__init__()
        # self.predictions = prev_stage_result
        if isinstance(prev_stage_result, Tuple):
            self.predictions = prev_stage_result[0]
            self.rest = prev_stage_result[1:]
        else:
            self.predictions = prev_stage_result
        self.predMaskImage = predMaskImage # If this is true the prediction mask is saved as a standalone image
        self.root_dir = root_dir / "images"

        self.fields = visualisationArgs.get("fields", None)
        if self.fields is None:
            self.fields = ["image", "pred_mask", "gt_mask"]
        self.overlay_fields = visualisationArgs.get("overlay_fields", None)
        if self.overlay_fields is None:
            self.overlay_fields = [("image", ["pred_mask"]), ("image", ["anomaly_map"])]
        self.field_size = visualisationArgs.get("field_size", None)
        if self.field_size is None:
            self.field_size = [256,256]
            logger.warning(f"Field size was not given for VisualisationJob. The Visualisation is probaly not right; size of (256,256) is assumed")

        self.fields_config = visualisationArgs.get("fields_config", None)
        if self.fields_config is None:
            self.fields_config = DEFAULT_FIELDS_CONFIG

        self.overlay_fields_config = visualisationArgs.get("overlay_fields_config", None)
        if self.overlay_fields_config is None:
            self.overlay_fields_config = DEFAULT_OVERLAY_FIELDS_CONFIG

        self.text_config = visualisationArgs.get("text_config", None)
        if self.text_config is None:
            self.text_config = DEFAULT_TEXT_CONFIG

        self.datasetName = datasetName
        self.category = category
        # if self.datasetName is None:da
        #     # if not specified, take class name
        #     self.datasetName = data_args["class_path"].split(".")[-1]
        # self.category = dataModuleConfig

    def run(self, task_id: int | None = None) -> list[Any]:
        """Run job that visualizes all prediction data.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: Unchanged predictions.
        """
        del task_id  # not needed here

        logger.info("Starting visualization.")

        for batch in tqdm(self.predictions, desc="Visualisation"):
            for data in batch:
                # logger.debug(f"{self.name}: item: {data}")
                # for item in batch:
                image = visualize_image_item(
                    data,
                    fields=self.fields,
                    overlay_fields=self.overlay_fields,
                    field_size=self.field_size,
                    fields_config=self.fields_config,
                    overlay_fields_config=self.overlay_fields_config,
                    text_config=self.text_config,
                )

                # Get the dataset name and category to save the image
                filename = generate_output_filename(
                    input_path=data.image_path or "",
                    output_path=self.root_dir,
                    dataset_name=self.datasetName,
                    category=self.category,
                )
                logger.debug(f"{self.name}: filename: {filename}")

                if image is not None:
                    # Save the image to the specified filename
                    image.save(filename)

                if self.predMaskImage:
                    predMask = visualize_image_item(
                        data,
                        fields=["pred_mask"],
                        overlay_fields=None,
                        field_size=self.field_size,
                        fields_config=self.fields_config,
                        overlay_fields_config=self.overlay_fields_config,
                        text_config={"enable": False},
                    )
                    if predMask is not None:
                        # Save the image to the specified filename
                        newStem = f"{filename.stem}_predMask"
                        predMask.save(filename.with_stem(newStem))

        return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: Unchanged list of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """This job doesn't save anything."""


class AOIVisualizationJobGenerator(JobGenerator):
    """Generate VisualizationJob.

    Args:
        root_dir (Path): Root directory where images will be saved (root/images).
    """

    def __init__(self, root_dir: Path, datasetName:str, category:str, visualisationArgs:dict[str,Any], predMaskImage:bool=False) -> None:
        self.root_dir = root_dir
        self.datasetName = datasetName
        self.category = category
        self.visualisationArgs = visualisationArgs
        self.predMaskImage = predMaskImage

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIVisualizationJob

    def generate_jobs(
        self,
        args: Dict[str,Any] | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[AOIVisualizationJob, None, None]:
        """Return a generator producing a single visualization job.

        Args:
            args: Ensemble run args.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[VisualizationJob, None, None]: VisualizationJob generator
        """
        del args  # args not used here

        if prev_stage_result is not None:
            yield AOIVisualizationJob(prev_stage_result, root_dir=self.root_dir, datasetName=self.datasetName, category=self.category, visualisationArgs=self.visualisationArgs, predMaskImage=self.predMaskImage)
        else:
            msg = "Visualization job requires tile level predictions from previous step."
            raise ValueError(msg)
        
class AOIFiftyOneVisJob(Job):
    """Job for visualization of predictions.

    Args:
        predictions (list[Any]): list of image-level predictions.
        root_dir (Path): Root directory for images.
        data_args (Dict): data args used to get data name and category name.
    """

    name = "51Visualize"

    def __init__(self, prev_stage_result: Tuple[List[ImageBatch], dict[str, Any]] | List[ImageBatch], FO_Dataset:fo.Dataset, datamodule: FODataModule, modelName:str, split:Split) -> None:
        super().__init__()
        if isinstance(prev_stage_result, Tuple):
            self.predictions = prev_stage_result[0]
            self.rest = prev_stage_result[1:]
        else:
            self.predictions = prev_stage_result
            self.rest = None
            
        self.FO_Dataset = FO_Dataset
        self.datamodule = datamodule
        self.modelName = modelName
        self.datasetName = datamodule.name
        self.category:str = datamodule.category # dataArgs["init_args"].get("category",FO_Dataset.first().category.label)
        self.split = split
        logger.debug(f"{self.name}: Dataset name: {self.datasetName}")
        logger.debug(f"{self.name}: Selected Category: {self.category}")
        logger.debug(f"{self.name}: Model name: {self.modelName}")

    def run(self, task_id: int | None = None) -> Tuple[List[ImageBatch], dict[str, Any]] | List[ImageBatch]:
        """Run job that visualizes all prediction data.

        Args:
            task_id: Not used in this case.

        Returns:
            Tuple[List[ImageBatch], dict[str, Any]] | List[ImageBatch]: Unchanged predictions.
        """
        del task_id  # not needed here
        logger.info("Starting visualisation for Fiftyone.")
        logger.debug(f"{self.name}: Sample: {self.predictions[0].image[0]}")

        for batch in tqdm(self.predictions, desc="51 Visualisation"):
            for data in batch:
                path = data.image_path
                sample:fo.Sample = self.FO_Dataset[path]
                conf = data.pred_score.item()
                anomaly = "anomaly" if data.pred_label.item() else "normal"

                sample[f"pred_anomaly_score_{self.modelName}"] = conf
                sample[f"pred_anomaly_{self.modelName}"] = fo.Classification(label=anomaly)
                heatmap = data.anomaly_map.to("cpu")
                sample[f"pred_anomaly_map_{self.modelName}"] = fo.Heatmap(map=heatmap.data.numpy().squeeze()*255, range=[0,255])
                try:
                    mask = data.pred_mask.to("cpu")
                    sample[f"pred_defect_mask_{self.modelName}"] = fo.Segmentation(mask=mask.data.numpy().squeeze().astype(np.uint8)*255)
                except:
                    logger.error("Segmentation prediction mask not available.")

                enum_values = {getattr(member, "value", member) for member in [Split.VAL, Split.TEST]}

                if hasattr(sample, 'tags'):
                    if isinstance(sample.tags, list):
                        split_value = self.split.value
                        new_tags: list[str] = []
                        split_replaced = False
                    
                        for tag in sample.tags: # TODO ignore
                            if isinstance(tag, str) and tag in enum_values:
                                if not split_replaced:
                                    new_tags.append(split_value)
                                    split_replaced = True
                            else:
                                new_tags.append(tag)

                        if not split_replaced:
                            new_tags.append(split_value)
                        if "predicted" not in new_tags:
                            new_tags.append("predicted")

                        sample.tags = new_tags
                    else:
                        sample.tags = ["predicted", self.split.value]
                else:
                    sample.tags = ["predicted", self.split.value]
                sample.save()
        # self.FO_Dataset.tag_samples("predicted")
        # self.FO_Dataset.tags.append("predicted")

        if self.rest is not None:
            return self.predictions, self.rest
        else:
            return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: Unchanged list of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """This job doesn't save anything."""


class AOIFiftyOneVisJobGenerator(JobGenerator):
    """Generate VisualizationJob.

    Args:
        root_dir (Path): Root directory where images will be saved (root/images).
    """

    def __init__(self, FO_Dataset:fo.Dataset, datamodule:FODataModule, modelName:str, split:Split) -> None:
        self.datamodule = datamodule
        self.FO_Dataset = FO_Dataset
        self.modelName = modelName
        self.split = split

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIFiftyOneVisJob

    def generate_jobs(
        self,
        args: Dict[str, Any] | None = None,
        prev_stage_result: List[ImageBatch] | None = None,
    ) -> Generator[AOIFiftyOneVisJob, None, None]:
        """Return a generator producing a single visualization job.

        Args:
            args: Ensemble run args.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[VisualizationJob, None, None]: VisualizationJob generator
        """
        del args  # args not used here

        if prev_stage_result is not None:
            yield AOIFiftyOneVisJob(prev_stage_result, self.FO_Dataset, datamodule=self.datamodule, modelName=self.modelName, split=self.split)
        else:
            msg = "Visualization job requires tile level predictions from previous step."
            raise ValueError(msg)
        
# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - normalization job."""
class AOINormalizationJob(Job):
    """Job for normalization of predictions.

    Args:
        predictions (list[Any]): List of predictions.
        root_dir (Path): Root directory containing statistics needed for normalization.
    """

    name = "Normalize"

    def __init__(self, prev_stage_result: Tuple[List[ImageBatch], Dict[str,Any]]|List[ImageBatch], root_dir: Path) -> None:
        super().__init__()
        # if prev_stage_result is not None:
        if isinstance(prev_stage_result, Tuple):
            self.predictions = prev_stage_result[0]
            self.rest = prev_stage_result[1:]
        else:
            self.predictions = prev_stage_result
            self.rest = None
        self.root_dir = root_dir

    def run(self, task_id: int | None = None):
        """Run normalization job which normalizes image, pixel and box scores.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: List of normalized predictions.
        """
        del task_id  # not needed here

        # load all statistics needed for normalization
        stats_path = self.root_dir / "stats.json"
        logger.info(f"{self.name}: Reading stats from file {stats_path}")
        with stats_path.open("r") as f:
            stats = json.load(f)
        minmax = stats["minmax"]
        image_threshold = stats["image_threshold"]
        pixel_threshold = stats["pixel_threshold"]

        logger.info("Starting normalization.")
        logger.info(f"Unnormalized image threshold is {image_threshold}")
        logger.info(f"Unnormalized pixel threshold is {pixel_threshold}")
        logger.debug(f"{self.name}: Unnormalized sample anomaly map: {self.predictions[0].anomaly_map[0]}")
        updatedPredictions: List[ImageBatch] = []
        for batch in tqdm(self.predictions, desc="Normalizing"):
            updatedDataList: List[ImageItem] = []
            for data in batch:
                if data.pred_score is not None:
                    data.update(pred_score=Tensor(normalize(
                        data.pred_score,
                        image_threshold,
                        minmax["pred_score"]["min"],
                        minmax["pred_score"]["max"],
                    )))
                if hasattr(data, "anomaly_map") and data.anomaly_map is not None:
                    data.update(anomaly_map = Mask(normalize(
                        torch.as_tensor(data.anomaly_map),
                        pixel_threshold,
                        float(minmax["anomaly_map"]["min"]),
                        float(minmax["anomaly_map"]["max"]),
                    )))
                updatedDataList.append(data)
            updatedBatch: ImageBatch = ImageBatch.collate(updatedDataList)
            updatedPredictions.append(updatedBatch)
        self.predictions = updatedPredictions
        logger.debug(f"{self.name}: Normalized sample anomaly map: {self.predictions[0].anomaly_map[0]}")
        logger.debug(f"{self.name}: Normalized (updated) sample anomaly map: {updatedPredictions[0].anomaly_map[0]}")


        logger.info("Normalized anomaly_map and pred_score to 0-1. Threshold of 0.5 is now expected")

        if self.rest is not None:
            return self.predictions, self.rest
        else:
            return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: List of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing is saved in this job."""


class AOINormalizationJobGenerator(JobGenerator):
    """Generate NormalizationJob.

    Args:
        root_dir (Path): Root directory where statistics are saved.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOINormalizationJob

    def generate_jobs(
        self,
        args: Dict[str,Any] | None = None,
        prev_stage_result: List[ImageBatch] | None = None,
    ) -> Generator[AOINormalizationJob, None, None]:
        """Return a generator producing a single normalization job.

        Args:
            args: not used here.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[NormalizationJob, None, None]: NormalizationJob generator.
        """
        del args  # not needed here
        if prev_stage_result is None:
            logger.info(f"NormalizationJobGenerator received 'None' as previous results. Check pipeline!")
            raise ValueError
        else:
            yield AOINormalizationJob(prev_stage_result, self.root_dir)

class AOIThresholdingJob(ThresholdingJob):
    """Job used to threshold predictions, producing labels from scores.

    Args:
        predictions (list[Any]): List of predictions.
        image_threshold (float): Threshold used for image-level thresholding.
        pixel_threshold (float): Threshold used for pixel-level thresholding.
    """

    name = "Threshold"

    def __init__(self, prev_stage_result: Tuple[List[ImageBatch], Dict[str,Any]]|List[ImageBatch], image_threshold: float, pixel_threshold: float) -> None:
        # if prev_stage_result is not None:
        if isinstance(prev_stage_result, Tuple):
            self.predictions = prev_stage_result[0]
            self.rest = prev_stage_result[1]
        else:
            self.predictions = prev_stage_result
            self.rest = None
        # else:
        #     self.predictions = None
        #     self.rest = None
        #     logger.info(f"Predictions in {self.name} job is None. This is likely an implementation error")

            super().__init__(self.predictions, image_threshold, pixel_threshold)
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold

    def run(self, task_id: int | None = None) -> Tuple[List[ImageBatch], Dict[str,Any]] | List[ImageBatch]:
        """Run job that produces prediction labels from scores.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: List of thresholded predictions.
        """
        del task_id  # not needed here

        logger.info("Starting thresholding.")
        logger.info(f"Image threshold is {self.image_threshold}")
        logger.info(f"Pixel threshold is {self.pixel_threshold}")
        logger.info(f"Number of predictions {len(self.predictions)}")

        updatedPredictions: List[ImageBatch] = []
        for batch in tqdm(self.predictions, desc="Thresholding"):
            updatedDataList: List[ImageItem] = []
            for data in batch:
                if hasattr(data, "pred_score") and data.pred_score is not None:
                    data.pred_label = data.pred_score >= self.image_threshold
                    # logger.info(f"score: {data.pred_score} - threshold {self.image_threshold}")
                    # print(f"score: {data.pred_score} - threshold {self.image_threshold}")

                if hasattr(data, "anomaly_map") and data.anomaly_map is not None:
                    data.pred_mask = data.anomaly_map >= self.pixel_threshold
                    # print(f"anomaly_map: {data.anomaly_map} - threshold {self.pixel_threshold}")
                updatedDataList.append(data)
            updatedBatch: ImageBatch = ImageBatch.collate(updatedDataList)
            updatedPredictions.append(updatedBatch)
        self.predictions = updatedPredictions

        if self.rest is not None:
            return self.predictions, self.rest
        else:
            return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: List of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing is saved in this job."""


class AOIThresholdingJobGenerator(JobGenerator):
    """Generate ThresholdingJob.

    Args:
        root_dir (Path): Root directory containing post-processing stats.
    """

    def __init__(self, root_dir: Path, normalization_stage: NormalizationStage) -> None:
        self.root_dir = root_dir
        self.normalization_stage = normalization_stage

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOIThresholdingJob

    def generate_jobs(
        self,
        args: Dict[str,Any] | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[AOIThresholdingJob, None, None]:
        """Return a generator producing a single thresholding job.

        Args:
            args: ensemble run args.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[AOIThresholdingJob, None, None]: AOIThresholdingJob generator.
        """
        del args  # args not used here

        # get threshold values base on normalization
        image_threshold, pixel_threshold = get_threshold_values(self.normalization_stage, self.root_dir)

        yield AOIThresholdingJob(
            prev_stage_result=prev_stage_result,
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
        )

def get_threshold_values(normalization_stage: NormalizationStage, root_dir: Path) -> Tuple[float, float]:
    """Get threshold values for image and pixel level predictions.

    If normalization is not used, get values based on statistics obtained from validation set.
    If normalization is used, both image and pixel threshold are 0.5

    Args:
        normalization_stage (NormalizationStage): ensemble run args, used to get normalization stage.
        root_dir (Path): path to run root where stats file is saved.

    Returns:
        Tuple[float, float]: image and pixel threshold.
    """
    if normalization_stage == NormalizationStage.NONE:
        logger.info("Normalization is not used. Obtain thresholds from validation set")
        stats_path = root_dir / "stats.json"
        with stats_path.open("r") as f:
            stats = json.load(f)
        image_threshold = stats["image_threshold"]
        pixel_threshold = stats["pixel_threshold"]
    else:
        logger.info("Normalization is used. both image and pixel threshold are 0.5.")
        # normalization transforms the scores so that threshold is at 0.5
        image_threshold = 0.5
        pixel_threshold = 0.5

    return image_threshold, pixel_threshold


from anomalib.models.components import GaussianBlur2d

class AOISmoothingJob(Job):
    """Job for smoothing the area around the tile seam.

    Args:
        accelerator (str): Accelerator used for processing.
        predictions (list[Any]): List of image-level predictions.
        width_factor (float):  Factor multiplied by tile dimension to get the region around seam which will be smoothed.
        filter_sigma (float): Sigma of filter used for smoothing the seams.
        tiler (EnsembleTiler): Tiler object used to get tile dimension data.
    """

    name = "SeamSmoothing"

    def __init__(
        self,
        accelerator: str,
        predictions: List[ImageBatch],
        width_factor: float,
        filter_sigma: float,
        tiler: EnsembleTiler,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.predictions = predictions

        # offset in pixels of region around tile seam that will be smoothed
        self.height_offset = int(tiler.tile_size_h * width_factor)
        self.width_offset = int(tiler.tile_size_w * width_factor)
        self.tiler = tiler

        self.seam_mask = self.prepare_seam_mask()

        self.blur = GaussianBlur2d(sigma=filter_sigma)

    def prepare_seam_mask(self) -> torch.Tensor:
        """Prepare boolean mask of regions around the part where tiles seam in ensemble.

        Returns:
            torch.Tensor: Representation of boolean mask where filtered seams should be used.
        """
        img_h, img_w = self.tiler.image_size
        stride_h, stride_w = self.tiler.stride_h, self.tiler.stride_w

        mask = torch.zeros(img_h, img_w, dtype=torch.bool)

        # prepare mask strip on vertical seams
        curr_w = stride_w
        while curr_w < img_w:
            start_i = curr_w - self.width_offset
            end_i = curr_w + self.width_offset
            mask[:, start_i:end_i] = 1
            curr_w += stride_w

        # prepare mask strip on horizontal seams
        curr_h = stride_h
        while curr_h < img_h:
            start_i = curr_h - self.height_offset
            end_i = curr_h + self.height_offset
            mask[start_i:end_i, :] = True
            curr_h += stride_h

        return mask

    def run(self, task_id: int | None = None) -> List[ImageBatch]:
        """Run smoothing job.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: List of predictions.
        """
        del task_id  # not needed here
        logger.debug(f"{self.name}: Sample: {self.predictions[0].image[0]}")

        updatedPredictions: List[ImageBatch] = []
        for batch in tqdm(self.predictions, desc="Seam smoothing"):
            updatedDataList: List[ImageItem] = []
            for data in batch:
                if data.anomaly_map is not None:
                    # move to specified accelerator for faster execution
                    data.anomaly_map = data.anomaly_map.to(self.accelerator)
                    smoothed = self.blur(data.anomaly_map.unsqueeze(0).unsqueeze(0))
                    data.anomaly_map[self.seam_mask] = smoothed[0, 0, self.seam_mask]
                    data.anomaly_map = data.anomaly_map.cpu()
                else:
                    logger.debug(f"{self.name}: Anomaly map is None. No smoothing done.")
                updatedDataList.append(data)
            updatedBatch: ImageBatch = ImageBatch.collate(updatedDataList)
            updatedPredictions.append(updatedBatch)
        self.predictions = updatedPredictions

        return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: List of predictions.
        """
        # take the first element as result of run() is packed into a list.
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing to save in this job."""
class AOISmoothingJobGenerator(JobGenerator):
    """Generate SmoothingJob."""

    def __init__(self, accelerator: str, tilingPipelineConfig: TilingPipelineConfig) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.tilingPipelineConfig = tilingPipelineConfig
        # self.data_args = data_args

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return AOISmoothingJob

    def generate_jobs(
        self,
        args: Dict[str, Any] | None = None,
        prev_stage_result: List[ImageBatch] | None = None,
    ) -> Generator[AOISmoothingJob, None, None]:
        """Return a generator producing a single seam smoothing job.

        Args:
            args: Tiled ensemble pipeline args.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[SmoothingJob, None, None]: SmoothingJob generator
        """
        # if args is None:
        #     msg = "SeamSmoothing job requires config args"
        #     raise ValueError(msg)
        del args
        # tiler is used to determine where seams appear
        tiler = get_ensemble_tiler(self.tilingPipelineConfig.to_dict())
        if prev_stage_result is not None:
            yield AOISmoothingJob(
                accelerator=self.accelerator,
                predictions=prev_stage_result,
                width_factor=self.tilingPipelineConfig.seam_smoothing.width,
                filter_sigma=self.tilingPipelineConfig.seam_smoothing.sigma,
                tiler=tiler,
            )
        else:
            msg = "Join smoothing job requires tile level predictions from previous step."
            raise ValueError(msg)