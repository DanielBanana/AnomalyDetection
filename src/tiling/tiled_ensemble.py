"""
Implements the 
"""
# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# GENERAL
import fiftyone as fo # base library and app
import logging
import torch
import json
import os
import contextlib
import functools

from typing import List, Any, Dict, Literal
from collections.abc import Generator
from itertools import product
from pathlib import Path
from lightning import seed_everything, Trainer
from torch.utils.data import DataLoader
from jsonargparse import ArgumentParser, Namespace
from torchvision.transforms.v2 import Compose, Resize, Transform


# ANOMALIB
from anomalib.data import AnomalibDataModule, ImageBatch, get_datamodule, AnomalibDataModule, PredictDataset as InferenceDataset
from anomalib.data.utils import ValSplitMode, TestSplitMode

from anomalib.models import AnomalibModule, get_model
from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform
from anomalib.post_processing import PostProcessor
from anomalib.metrics import F1Score, AUPR, AUROC, F1AdaptiveThreshold
from anomalib.metrics.evaluator import Evaluator
from anomalib.utils.logging import redirect_logs

# ANOMALIB PIPELINES
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage, PredictData as InferenceData, ThresholdingStage
from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_engine import TiledEnsembleEngine
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
from anomalib.pipelines.tiled_ensemble.components import (
    # MetricsCalculationJobGenerator,
    NormalizationJobGenerator,
    ThresholdingJobGenerator,
    VisualizationJobGenerator,
    StatisticsJobGenerator
)
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT

# OWN CODE
from setup import DataModuleConfig, ModelConfig, TrainerConfig, TilingPipelineConfig
from data.anomaly_datasets import importDataset, FODataModule, FODataset, AnomalibDataset
from tiling.ensemble_engine import AOITiledEnsembleEngine
from tiling.ensemble_tiling import EnsembleTiler, TileCollater
from tiling.jobs import (
    AOIStatisticsJobGenerator,
    AOIMergeJobGenerator,
    AOINormalizationJobGenerator,
    AOIMetricsCalculationJobGenerator,
    AOIVisualizationJobGenerator,
    AOIFiftyOneVisJobGenerator,
    AOIThresholdingJobGenerator,
    AOISmoothingJobGenerator,
    get_ensemble_model,
    Split
)
import platform

class PipelineError(RuntimeError): ...

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    # Enables TensorFloat-32 for float32 matmuls/convs on Tensor Core GPUs (Ampere+,
    # incl. the RTX 3070 this pipeline targets). float32 tensors are the common case
    # here (no AMP/precision override is configured), so without this Lightning's own
    # startup warning is right: those ops run at full fp32 precision and leave the
    # Tensor Cores unused. "high" trades a small amount of matmul precision (TF32,
    # ~10 bits mantissa vs fp32's 23) for a substantial throughput gain; anomalib
    # models' tolerance for this is the same as any other fp32-trained CNN/ViT.
    torch.set_float32_matmul_precision("high")

def get_device():
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def _pipeline_trainer_kwargs(trainerConfig: TrainerConfig) -> Dict[str, Any]:
    """Build the trainer_args dict handed to each stage's Job/JobGenerator.

    TrainModelJob/PredictJob thread this straight into ``parse_trainer_kwargs`` ->
    ``lightning.pytorch.Trainer(**kwargs)``, so everything here must be a real Trainer
    kwarg - drop n_parallel_jobs, which is our own bolted-on config field.
    """
    kwargs = trainerConfig.to_kwargs()
    kwargs.pop("n_parallel_jobs", None)
    return kwargs


# Cache of probed batch sizes, keyed by (accelerator, model class, tile shape, which
# batch-size arg, upper limit). All tiles in an ensemble share the same tile shape, so
# probing once per unique key instead of once per tile avoids repeating the (slow) OOM
# search.
_BATCH_SIZE_PROBE_CACHE: dict[tuple[Any, ...], int] = {}

# Batch size used when the probe itself fails: small, so that it also fits the large
# tiles of the line camera's images, not only the 256 px tiles of MVTec.
PROBE_FALLBACK_BATCH_SIZE = 4
# Largest batch size the probe tries at all.
PROBE_MAX_BATCH_SIZE = 64
# Default of a trainer config's max_train_batch_size (see DataModuleConfig in setup.py).
DEFAULT_MAX_TRAIN_BATCH_SHARE = 0.125


def max_train_batch_size(n_train_images: int, share: float | None) -> int:
    """Upper limit of a probed ("auto") training batch size: `share` of the training
    images (DEFAULT_MAX_TRAIN_BATCH_SHARE if None), at least 1, at most
    PROBE_MAX_BATCH_SIZE. Every tile model sees every training image once per epoch,
    so this keeps at least about 1/share optimizer steps per epoch."""
    share = DEFAULT_MAX_TRAIN_BATCH_SHARE if share is None else share
    return max(1, min(PROBE_MAX_BATCH_SIZE, int(n_train_images * share)))


@contextlib.contextmanager
def _trust_own_checkpoints():
    """Temporarily default ``torch.load(weights_only=...)`` to ``False``.

    torch>=2.6 defaults ``weights_only=True``, which refuses to unpickle non-tensor
    objects (e.g. anomalib's ``PreProcessor``) bundled in a full Lightning checkpoint.
    Only use this around loads of checkpoints we wrote ourselves moments earlier in
    this same process - never around loading a checkpoint of unknown/external origin.

    Overrides ``weights_only`` even when the caller passes it: Lightning's
    checkpoint restore calls ``torch.load(..., weights_only=None)`` explicitly, and
    None means torch's strict default. Neither ``kwargs.setdefault`` nor a
    ``functools.partial`` default (what this used to do) gets past an explicit
    argument, so the probe's restore kept failing.
    """
    original_load = torch.load

    @functools.wraps(original_load)
    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


def probe_optimal_batch_size(
    model: AnomalibModule,
    datamodule: AnomalibDataModule,
    accelerator: str,
    root_dir: Path,
    batch_arg_name: str,
    method: Literal["fit", "validate", "test", "predict"],
    max_val: int = PROBE_MAX_BATCH_SIZE,
) -> int:
    """Empirically find the largest batch size that fits in memory via binary search.

    Runs against a deep-copied model and a throwaway ``Trainer`` so the real
    training/prediction run (and its checkpoints/workspace) is left untouched. This
    is preferred over a closed-form memory estimate because actual GPU usage depends
    on cuDNN's chosen conv algorithm, framework overhead and fragmentation, which
    aren't reliably predictable from model size and tile shape alone.

    Args:
        model: Model to probe (used as a template; a deep copy is actually run).
        datamodule: Ensemble datamodule already configured for this tile (tiling
            collate function and resize already applied).
        accelerator: Accelerator to probe on. Probing is skipped (and a small
            conservative default returned) unless this is "cuda" - a single GPU is
            the resource under contention here, so that's the only case worth the
            probe cost.
        root_dir: Root dir to scratch-write the prober's temporary checkpoint into.
        batch_arg_name: "train_batch_size" or "eval_batch_size" - which datamodule
            attribute the probe should search over and write the result back to.
        method: Lightning method to probe with ("fit", "validate" or "test"). Use
            "fit" for train_batch_size (exercises the real backward/optimizer step
            for gradient-trained models), and "validate"/"test" for eval_batch_size
            (matches whichever split predict will actually run on).
        max_val: Upper bound on the search, to avoid testing unrealistically large
            batch sizes. For training see max_train_batch_size. Below 2 there is
            nothing to search: max_val itself is returned.

    Returns:
        int: Largest batch size found to fit in memory (>= 1).
    """
    input_size = getattr(model, "input_size", None)
    cache_key = (
        accelerator,
        model.__class__.__name__,
        tuple(input_size) if input_size is not None else None,
        batch_arg_name,
        method,
        max_val,
    )
    if cache_key in _BATCH_SIZE_PROBE_CACHE:
        return _BATCH_SIZE_PROBE_CACHE[cache_key]
    if max_val < 2:
        return max(1, max_val)

    if accelerator != "cuda":
        fallback = min(8, max_val)
        logger.info(f"Batch size probing skipped on accelerator '{accelerator}'; using conservative default of {fallback}.")
        _BATCH_SIZE_PROBE_CACHE[cache_key] = fallback
        return fallback

    from copy import deepcopy
    from lightning.pytorch import Trainer as PLTrainer
    from lightning.pytorch.tuner.tuning import Tuner

    probe_model = deepcopy(model)
    probe_trainer = PLTrainer(
        accelerator=accelerator,
        devices=1,
        default_root_dir=root_dir / ".batch_size_probe",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=1,

    )
    tuner = Tuner(probe_trainer)

    optimal: int | None
    try:
        # Tuner.scale_batch_size checkpoints the probe model/trainer before searching and
        # restores it afterwards, so the search's real forward/backward trials don't leave
        # the (discarded) probe model in a partially-updated state. That checkpoint bundles
        # non-tensor objects (e.g. anomalib's PreProcessor), which torch>=2.6's strict
        # `weights_only=True` default refuses to unpickle. The checkpoint is one we wrote
        # ourselves a few lines above, in this same process, so trusting it here (unlike an
        # arbitrary external checkpoint) is safe - relax just this restore accordingly.
        with _trust_own_checkpoints():
            optimal = tuner.scale_batch_size(
                probe_model,
                datamodule=datamodule,
                method=method,
                mode="binsearch",
                batch_arg_name=batch_arg_name,
                init_val=2,
                steps_per_trial=1,
                max_trials=6,
                margin=0.1,
                max_val=max_val,    
            )
    except Exception:
        optimal = min(PROBE_FALLBACK_BATCH_SIZE, max_val)
        logger.exception(f"Batch size probe for '{batch_arg_name}' failed; falling back to {optimal}.")
    finally:
        del probe_model, probe_trainer, tuner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    optimal = max(1, min(optimal or 1, max_val))
    logger.info(f"Batch size probe: {batch_arg_name} ({method}, at most {max_val}) -> {optimal}")
    _BATCH_SIZE_PROBE_CACHE[cache_key] = optimal
    return optimal

class TrainTiledEnsemble(Pipeline):
    """Tiled ensemble training pipeline."""

    def __init__(self, rootDir:Path,
                 datamodule:FODataModule,
                 dataModuleConfig:DataModuleConfig,
                 FO_Dataset:fo.Dataset,
                 gtAvail:bool,
                 tilingPipelineConfig:TilingPipelineConfig,
                 modelConfig:ModelConfig,
                 trainerConfig:TrainerConfig) -> None:
        self.rootDir:Path = rootDir
        self.datamodule = datamodule
        self.dataModuleConfig = dataModuleConfig
        self.FO_Dataset:fo.Dataset|None = FO_Dataset
        self.gtAvail:bool = gtAvail #TODO add function that sets this value
        self.tilingPipelineConfig=tilingPipelineConfig
        self.modelConfig=modelConfig
        self.trainerConfig=trainerConfig
        self.visualisationArgs:dict[str,Any] = {
            "field_size": self.tilingPipelineConfig.image_size,
            "fields": ["image", "pred_mask"] if not self.gtAvail else ["image", "gt_mask", "pred_mask"],
            "overlay_fields": [("image", ["anomaly_map"]), ("image", ["pred_mask"])] if not self.gtAvail else [("image", ["anomaly_map"]), ("image", ["gt_mask"]), ("image", ["pred_mask"])]
        }

        logger.debug(f"TrainTiledEnsemble: DataModuleConfig: {dataModuleConfig}")
        logger.debug(f"TrainTiledEnsemble: tilingPipelineConfig: {tilingPipelineConfig}")
        logger.debug(f"TrainTiledEnsemble: modelConfig: {modelConfig}")
        logger.debug(f"TrainTiledEnsemble: trainerConfig: {trainerConfig}")
        logger.debug(f"TrainTiledEnsemble: rootDir: {rootDir}")
        logger.debug(f"TrainTiledEnsemble: visualisationARgs: {self.visualisationArgs}")

    def _setup_runners(self, args: Dict[str,Any]) -> List[Runner]:
        """Setup the runners for the pipeline.

        This pipeline consists of training and validation steps:
        Training models > prediction on val data > merging val data >
        > (optionally) smoothing seams > calculation of post-processing statistics

        Returns:
            List[Runner]: List of runners executing tiled ensemble train + val jobs.
        """

        seed:int = args.get("seed", 42)
        logger.info("TrainTiledPipeline: No seed given in arguments, using 42")


        runners: List[Runner] = []
        valSplitMode:ValSplitMode = self.dataModuleConfig.val_split_mode

        # 1. train
        train_job_generator = TrainModelJobGenerator(
            seed=seed,
            accelerator=self.trainerConfig.accelerator,
            root_dir=self.rootDir,
            tilingPipelineConfig=self.tilingPipelineConfig,
            dataModuleConfig=self.dataModuleConfig,
            modelConfig=self.modelConfig,
            datamodule=self.datamodule,
            normalization_stage=self.tilingPipelineConfig.normalization_stage,
        )
        n_gpus = torch.cuda.device_count()
        n_parallel = min(self.trainerConfig.n_parallel_jobs or n_gpus, n_gpus) if n_gpus > 0 else 1
        if self.trainerConfig.n_parallel_jobs and self.trainerConfig.n_parallel_jobs > n_gpus:
            logger.warning(
                f"n_parallel_jobs ({self.trainerConfig.n_parallel_jobs}) exceeds available GPUs ({n_gpus}); "
                f"clamping to {n_parallel} to avoid multiple processes contending for the same device.",
            )
        if self.trainerConfig.accelerator == "cuda" and n_parallel > 1:
            runners.append(
                ParallelRunner(
                    train_job_generator,
                    n_jobs=n_parallel,
                ),
            )
        else:
            runners.append(
                SerialRunner(
                    train_job_generator,
                ),
            )

        if valSplitMode == ValSplitMode.NONE:
            logger.warning("No validation set provided, skipping statistics calculation.")
            return runners

        # mode = InferenceData.VAL

        # # 2. predict using validation data
        # predict_job_generator = InferenceJobGenerator(
        #     data_source=mode,
        #     seed=seed,
        #     accelerator=self.trainerConfig.accelerator,
        #     root_dir=self.rootDir,
        #     tilingPipelineConfig=self.tilingPipelineConfig,
        #     dataModuleConfig=self.dataModuleConfig,
        #     datamodule=self.datamodule,
        #     modelConfig=self.modelConfig,
        #     normalization_stage=self.tilingPipelineConfig.normalization_stage,
        #     inferenceDataset=None
        # )

        # if self.trainerConfig.accelerator == "cuda" and n_parallel > 1:
        #     runners.append(
        #         ParallelRunner(predict_job_generator, n_jobs=n_parallel),
        #     )
        # else:
        #     runners.append(
        #         SerialRunner(predict_job_generator),
        #     )

        # # 3. merge predictions
        # runners.append(SerialRunner(AOIMergeJobGenerator(tilingPipelineConfig=self.tilingPipelineConfig)))

        # # 4. (optional) smooth seams
        # if self.tilingPipelineConfig.seam_smoothing.apply:
        #     runners.append(
        #         SerialRunner(
        #             AOISmoothingJobGenerator(accelerator="cpu", tilingPipelineConfig=self.tilingPipelineConfig),
        #         ),
        #     )

        # # 5. calculate statistics used for inference
        # runners.append(SerialRunner(AOIStatisticsJobGenerator(self.rootDir)))

        # # 6. (optional) normalize
        # if self.tilingPipelineConfig.normalization_stage == NormalizationStage.IMAGE:
        #     runners.append(SerialRunner(AOINormalizationJobGenerator(self.rootDir)))
            
        # # 7. (optional) threshold to get labels from scores
        # if self.tilingPipelineConfig.thresholding_stage == ThresholdingStage.IMAGE:
        #     runners.append(SerialRunner(AOIThresholdingJobGenerator(self.rootDir, self.tilingPipelineConfig.normalization_stage)))
        
        # # 8. calculate accuracy metrics
        # runners.append(
        #     SerialRunner(
        #         AOIMetricsCalculationJobGenerator(
        #             accelerator=self.trainerConfig.accelerator,
        #             root_dir=self.rootDir,
        #             modelConfig=self.modelConfig,
        #             tile_size=self.tilingPipelineConfig.tile_size,
        #             split=mode
        #         ),
        #     ),
        # )

        # # 9. Visualise on disk
        # runners.append(SerialRunner(AOIVisualizationJobGenerator(root_dir=self.rootDir/mode.value, datasetName=self.datamodule.name, category=self.datamodule.category, visualisationArgs=self.visualisationArgs, predMaskImage=True)))

        # # 9 (optional) Associate the results back with the fiftyone dataset where they come from so they can be visualised
        # if self.FO_Dataset is not None:
        #     runners.append(SerialRunner(AOIFiftyOneVisJobGenerator(FO_Dataset=self.FO_Dataset, datamodule=self.datamodule, modelName=self.modelConfig.name, split=mode)))

        return runners

    def run(self, args: Namespace | None = None) -> None:
        """Run the pipeline.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.
        """
        runners:List[Runner] = self._setup_runners({})
        previous_results: PREV_STAGE_RESULT = None
        for runner in runners:
            try:
                job_args = _pipeline_trainer_kwargs(self.trainerConfig)
                previous_results = runner.run(job_args or {}, previous_results)
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the runner.")
                print(
                    f"There were some errors when running {runner.generator.job_class.name} with"
                    f" {runner.__class__.__name__}."
                    # f" Please check {logFile} for more details.",
                )

class EvalTiledEnsemble(Pipeline):
    """Tiled ensemble evaluation pipeline.

    Args:
        root_dir (Path): Path to root dir of run that contains checkpoints.
    """

    def __init__(self, rootDir:Path, datamodule:FODataModule, dataModuleConfig:DataModuleConfig, FO_Dataset:fo.Dataset, gtAvail:bool, tilingPipelineConfig: TilingPipelineConfig, evalConfig:TrainerConfig, modelConfig: ModelConfig, ckptPath:Path|None=None, ) -> None:
        self.rootDir:Path = rootDir
        self.datamodule = datamodule
        self.dataModuleConfig = dataModuleConfig
        self.tilingPipelineConfig = tilingPipelineConfig
        self.modelConfig = modelConfig
        self.evalConfig = evalConfig
        self.FO_Dataset = FO_Dataset
        self.gtAvail:bool = gtAvail #TODO add function that sets this value
        self.ckptPath = ckptPath

    def _setup_runners(self, args: dict[str,Any]) -> List[Runner]:
        """Set up the runners for the pipeline.

        This pipeline consists of jobs used to test/evaluate tiled ensemble:
        Prediction on test data > merging of predictions > (optional) seam smoothing
        > (optional) Normalization > (optional) Thresholding
        > Visualisation of predictions > Metrics calculation.

        Returns:
            List[Runner]: List of runners executing tiled ensemble testing jobs.
        """
        runners: List[Runner] = []

        seed:int = args.get("seed", 42)
        if self.ckptPath is None:
            logger.info(f"No ckptPath given in arguments to {self.__class__}. Going over rootDir: {self.rootDir}/checkpoints")
            ckptPath:Path = self.rootDir / "checkpoints"
        else:
            ckptPath:Path = self.ckptPath
            logger.info(f"Loading Checkpoints from path: {ckptPath}")

        normalization_stage = NormalizationStage(args.get("normalization_stage", NormalizationStage.IMAGE))
        thresholding_stage = ThresholdingStage(args.get("thresholding_stage", ThresholdingStage.IMAGE))

        visualisationArgs:dict[str,Any] = {
            "field_size": self.tilingPipelineConfig.image_size,
            "fields": ["image", "pred_mask"] if not self.gtAvail else ["image", "gt_mask", "pred_mask"],
            "overlay_fields": [("image", ["anomaly_map"]), ("image", ["pred_mask"])] if not self.gtAvail else [("image", ["anomaly_map"]), ("image", ["gt_mask"]), ("image", ["pred_mask"])]
        }
        
        validationSplit: ValSplitMode = self.dataModuleConfig.val_split_mode
        testSplit: TestSplitMode = self.dataModuleConfig.test_split_mode
        modes: List[Split] = []

        logger.info(f"Validation split for evaluation pipeline is set to: {validationSplit}")
        if validationSplit == ValSplitMode.NONE:
            logger.info("This means no Evaluation is done on the to determine the decision tresholds")
        else:
            logger.info("Evaluating performance of trained model to determine decision thresholds")
            modes.append(Split.VAL)

        logger.info(f"Validation split for evaluation pipeline is set to: {testSplit}")
        if testSplit == TestSplitMode.NONE:
            logger.info("This means no Testing is done on the determine the quality of the threshold on unseen data")
        else:
            if validationSplit is ValSplitMode.NONE:
                raise PipelineError(f"Can not run test if no validation ran before hand.")
            else:
                logger.info("Testing to determine quality on unseen data")
                modes.append(Split.TEST)

        statsDir = self.ckptPath.parent if self.ckptPath is not None and self.rootDir is None else self.rootDir

        for mode in modes:

            inferenceJobGenerator = InferenceJobGenerator(
                data_source=mode,
                seed=seed,
                accelerator=self.evalConfig.accelerator,
                root_dir=self.rootDir,
                tilingPipelineConfig=self.tilingPipelineConfig,
                dataModuleConfig=self.dataModuleConfig,
                datamodule=self.datamodule,
                modelConfig=self.modelConfig,
                normalization_stage=normalization_stage,
                ckptPath=ckptPath,
                inferenceDataset=None
            )

            # 1. predict using test data
            _nEvalJobs = torch.cuda.device_count()
            if self.evalConfig.accelerator == "cuda" and _nEvalJobs > 1:
                runners.append(
                    ParallelRunner(
                        inferenceJobGenerator,
                        n_jobs=_nEvalJobs,
                    ),
                )
            else:
                runners.append(
                    SerialRunner(
                        inferenceJobGenerator,
                    ),
                )
            # 2. merge predictions
            runners.append(SerialRunner(AOIMergeJobGenerator(tilingPipelineConfig=self.tilingPipelineConfig)))

            # 3. (optional) smooth seams
            if self.tilingPipelineConfig.seam_smoothing.apply:
                runners.append(
                    SerialRunner(
                        AOISmoothingJobGenerator(accelerator="cpu", tilingPipelineConfig=self.tilingPipelineConfig),
                    ),
                )

            if mode == Split.VAL:
                # 5. calculate statistics used for inference
                runners.append(SerialRunner(AOIStatisticsJobGenerator(statsDir)))

            # 4. (optional) normalize
            if normalization_stage == NormalizationStage.IMAGE:
                logger.info(f"Taking stats for Nomalization from: {statsDir}")
                runners.append(SerialRunner(AOINormalizationJobGenerator(statsDir)))

            # 5. (optional) threshold to get labels from scores
            if thresholding_stage == ThresholdingStage.IMAGE:
                logger.info(f"Taking stats for Thresholding from: {statsDir}")
                runners.append(SerialRunner(AOIThresholdingJobGenerator(statsDir, normalization_stage)))

            # 6. calculate accuracy metrics
            runners.append(
                SerialRunner(
                    AOIMetricsCalculationJobGenerator(
                        accelerator=self.evalConfig.accelerator,
                        root_dir=self.rootDir,
                        modelConfig=self.modelConfig,
                        tile_size=self.tilingPipelineConfig.tile_size,
                        saveName=f"metric_results_{mode.value}.csv",
                        split=mode
                    ),
                ),
            )

            # 7. Visualise on disk
            runners.append(SerialRunner(AOIVisualizationJobGenerator(root_dir=self.rootDir/mode.value, datasetName=self.datamodule.name, category=self.datamodule.category, visualisationArgs=visualisationArgs, predMaskImage=True)))

            # 8. Visualize predictions in 51
            runners.append(SerialRunner(AOIFiftyOneVisJobGenerator(FO_Dataset=self.FO_Dataset, datamodule=self.datamodule, modelName=self.modelConfig.name, split=mode)))

        return runners
    
    # def setDatamodule(self, datamodule: FODataModule):
    #     self.datamodule = datamodule
    #     self.datamoduleArgs = {
    #         "init_args": {
    #             "name": datamodule.name,
    #             "root": datamodule.root,
    #             "category": datamodule.category,
    #             "train_batch_size": datamodule.train_batch_size,
    #             "eval_batch_size": datamodule.eval_batch_size,
    #             "num_workers": datamodule.num_workers,
    #             "train_augmentations": datamodule.train_augmentations,
    #             "val_augmentations": datamodule.val_augmentations,
    #             "test_augmentations": datamodule.test_augmentations,
    #             "augmentations": None,
    #             "test_split_mode": datamodule.test_split_mode,
    #             "test_split_ratio": datamodule.test_split_ratio,
    #             "val_split_mode": datamodule.val_split_mode,
    #             "val_split_ratio": datamodule.val_split_ratio,
    #         }
    #     }

    # def setFODataset(self, dataset:FODataset):
    #     self.dataset = dataset

    def run(self, args: Namespace | None = None) -> None:
        """Run the pipeline.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.
        """
        # if args is None:
        # pipeline_args = self._get_args(args)
        runners:List[Runner] = self._setup_runners({})
        # redirect_logs(logFile) # dont know what it does
        previous_results: PREV_STAGE_RESULT = None

        for runner in runners:
            try:
                job_args = _pipeline_trainer_kwargs(self.evalConfig)
                previous_results = runner.run(job_args or {}, previous_results)
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the runner.")
                print(
                    f"There were some errors when running {runner.generator.job_class.name} with"
                    f" {runner.__class__.__name__}."
                    # f" Please check {logFile} for more details.",
                )

class InferenceTiledEnsemble(Pipeline):
    """Tiled ensemble evaluation pipeline.

    Args:
        root_dir (Path): Path to root dir of run that contains checkpoints.
    """

    def __init__(self,
                 root_dir: Path,
                 trainingDir:Path,
                 ckptDir:Path,
                 inferenceDataset:InferenceDataset,
                 dataset:fo.Dataset,
                 datamodule:FODataModule,
                 dataModuleConfig:DataModuleConfig,
                 tilingPipelineConfig:TilingPipelineConfig,
                 inferencerConfig:TrainerConfig,
                 modelConfig:ModelConfig,
                 gtAvail:bool=False) -> None:
        """
        _summary_

        Parameters
        ----------
        root_dir : Path
            Root directory for prediction pipeline
        trainingDir : Path
            Directory where the model was trained for the  stats.json file for the threshold data
        ckptDir : Path
            Directory where the model weights were stored during training
        inferenceDataset : InferenceDataset
            Dataset that contains the data for which a prediction should be made
        dataset : fo.Dataset
            Dataset the prediction data belongs to; ususally the training images (i.e. dataset is all images of a specific product) both training data and prediction data belong to that general dataset
        datamodule : FODataModule
            Datamodule for how to handle the data in the prediction dataset (e.g. batch size)
        dataModuleConfig : DataModuleConfig
            Config for the datamodule (TODO ist that really needed here if we have the datamodule already?)
        tilingPipelineConfig : TilingPipelineConfig
            Config on how to tile the images in the prediction dataset. Should usually be identical to the config during training
        inferencerConfig : TrainerConfig
            Config for the pytorch.lightning trainer class that does the inferencing
        modelConfig : ModelConfig
            Config for the model that is inferenced
        gtAvail : bool (optional)
            Is ground truth data available. Default is `False` (TODO Remove since prediction dataset has no ground truth)
        """
        self.root_dir = Path(root_dir)                                          # Where this pipeline stores results from
        self.trainingDir = Path(trainingDir)   
        self.ckptDir = ckptDir
        logger.debug(f"Current working directory (cwd): {os.getcwd()}")                                 # Where the training pipeline stored results like threshold and normalization stats
        logger.info(f"Root directory for prediction Pipeline: {root_dir}")
        logger.info(f"Checkpoint directory: {ckptDir}")
        logger.info(f"Stats directory: {trainingDir}")
        self.dataset:fo.Dataset = dataset
        self.inferenceDataset = inferenceDataset
        self.datamodule:FODataModule = datamodule
        self.dataModuleConfig = dataModuleConfig
        self.tilingPipelineConfig = tilingPipelineConfig
        self.inferencerConfig = inferencerConfig
        self.modelConfig = modelConfig
        self.gtAvail = gtAvail

    def _setup_runners(self, args: dict[str,Any]) -> List[Runner]:
        """Set up the runners for the pipeline.

        This pipeline consists of jobs used to test/evaluate tiled ensemble:
        Prediction on test data > merging of predictions > (optional) seam smoothing
        > (optional) Normalization > (optional) Thresholding
        > Visualisation of predictions > Metrics calculation.

        Returns:
            List[Runner]: List of runners executing tiled ensemble testing jobs.
        """

        logger.info("Setting up runners")

        runners: List[Runner] = []
   
        seed:int = int(args.get("seed", 0))        

        visualisationArgs:dict[str,Any] = {
            "field_size": self.tilingPipelineConfig.image_size,
            "fields": ["image", "pred_mask"] if not self.gtAvail else ["image", "gt_mask", "pred_mask"],
            "overlay_fields": [("image", ["anomaly_map"]), ("image", ["pred_mask"])] if not self.gtAvail else [("image", ["anomaly_map"]), ("image", ["gt_mask"]), ("image", ["pred_mask"])]
        }

        logger.debug(self.dataset)

        logger.debug("Setting up JobGenerators")

        inferenceJobGenerator = InferenceJobGenerator(
            Split.INFERENCE,
            seed=seed,
            accelerator=self.inferencerConfig.accelerator,
            root_dir=self.root_dir,
            tilingPipelineConfig=self.tilingPipelineConfig,
            dataModuleConfig=self.dataModuleConfig,
            modelConfig=self.modelConfig,
            normalization_stage=self.tilingPipelineConfig.normalization_stage,
            datamodule=self.datamodule,
            inferenceDataset=self.inferenceDataset,
            ckptPath=self.ckptDir
        )

        # 1. predict using test data
        _n_inf_jobs = torch.cuda.device_count()
        if self.inferencerConfig.accelerator == "cuda" and _n_inf_jobs > 1:
            runners.append(
                ParallelRunner(
                    inferenceJobGenerator,
                    n_jobs=_n_inf_jobs,
                ),
            )
        else:
            runners.append(
                SerialRunner(
                    inferenceJobGenerator,
                ),
            )
        # 2. merge predictions
        runners.append(SerialRunner(AOIMergeJobGenerator(tilingPipelineConfig=self.tilingPipelineConfig)))

        # 3. (optional) smooth seams
        if self.tilingPipelineConfig.seam_smoothing.apply:
            runners.append(
                SerialRunner(
                    AOISmoothingJobGenerator(accelerator="cpu", tilingPipelineConfig=self.tilingPipelineConfig),
                ),
            )

        # 4. (optional) normalize
        if self.tilingPipelineConfig.normalization_stage == NormalizationStage.IMAGE:
             runners.append(SerialRunner(AOINormalizationJobGenerator(self.trainingDir)))

        # 5. (optional) threshold to get labels from scores
        if self.tilingPipelineConfig.thresholding_stage == ThresholdingStage.IMAGE:
            runners.append(SerialRunner(AOIThresholdingJobGenerator(self.trainingDir, self.tilingPipelineConfig.normalization_stage)))

        runners.append(SerialRunner(AOIVisualizationJobGenerator(root_dir=self.root_dir, datasetName=self.datamodule.name, category=self.datamodule.category, visualisationArgs=visualisationArgs, predMaskImage=True)))

        # # 6. visualize predictions
        # if self.dataset is not None:
        runners.append(SerialRunner(AOIFiftyOneVisJobGenerator(FO_Dataset=self.dataset, datamodule=self.datamodule, modelName=self.modelConfig.name, split=Split.INFERENCE)))

        # # calculate metrics
        # runners.append(
        #     SerialRunner(
        #         AOIMetricsCalculationJobGenerator(
        #             accelerator=accelerator,
        #             root_dir=self.root_dir,
        #             model_args=modelArgs,
        #         ),
        #     ),
        # )

        logger.debug("Finished JobGenerator setup!")

        return runners
    
    # def setDatamodule(self, datamodule: FODataModule):
    #     self.datamodule = datamodule
    #     self.datamoduleArgs = {
    #         "init_args": {
    #             "name": datamodule.name,
    #             "root": datamodule.root,
    #             "category": datamodule.category,
    #             "train_batch_size": datamodule.train_batch_size,
    #             "eval_batch_size": datamodule.eval_batch_size,
    #             "num_workers": datamodule.num_workers,
    #             "train_augmentations": datamodule.train_augmentations,
    #             "val_augmentations": datamodule.val_augmentations,
    #             "test_augmentations": datamodule.test_augmentations,
    #             "augmentations": None,
    #             "test_split_mode": datamodule.test_split_mode,
    #             "test_split_ratio": datamodule.test_split_ratio,
    #             "val_split_mode": datamodule.val_split_mode,
    #             "val_split_ratio": datamodule.val_split_ratio,
    #         }
    #     }

    # def setFODataset(self, dataset:FODataset):
    #     self.dataset = dataset

    def run(self, args: Namespace | None = None) -> None:
        """Run the pipeline.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.
        """
        # if args is None:
        # pipeline_args = self._get_args(args)
        runners:List[Runner] = self._setup_runners({})
        # redirect_logs(logFile) # dont know what it does
        previous_results: PREV_STAGE_RESULT = None

        for runner in runners:
            try:
                job_args = _pipeline_trainer_kwargs(self.inferencerConfig)
                previous_results = runner.run(job_args or {}, previous_results)
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the runner.")
                print(
                    f"There were some errors when running {runner.generator.job_class.name} with"
                    f" {runner.__class__.__name__}."
                    #  f" Please check {logFile} for more details.",
                )


"""Tiled ensemble - ensemble training job."""
class TrainModelJob(Job):
    """Job for training of individual models in the tiled ensemble.

    Args:
        accelerator (str): Accelerator (device) to use.
        seed (int): Random seed for reproducibility.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        tile_index (tuple[int, int]): Index of tile that this model processes.
        total_tiles (int): Total number of tiles in this ensemble run (see
            TrainModelJobGenerator.generate_jobs' tiler.num_tiles) -- threaded down to
            AOITiledEnsembleEngine so GUITrainingProgressCallback (if configured) can report
            accurate tile-progress without needing total_tiles set by hand in the trainer yaml.
        normalization_stage (str): Normalization stage flag.
        metrics (dict): metrics dict with pixel and image metric names.
        trainer_args (dict| None): Additional arguments to pass to the trainer class.
        model (AnomalyModule): Model to train.
        datamodule (AnomalibDataModule): Datamodule with all dataloaders.

    """

    name = "Trainer"

    def __init__(
        self,
        accelerator: str,
        seed: int,
        root_dir: Path,
        tile_index: tuple[int, int],
        total_tiles: int,
        normalization_stage: str,
        trainer_args: dict[str,Any] | None,
        model: AnomalibModule,
        datamodule: AnomalibDataModule,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        if self.accelerator == "auto":
            self.accelerator = get_device()
        self.seed = seed
        self.root_dir = root_dir
        self.tile_index = tile_index
        self.total_tiles = total_tiles
        self.normalization_stage = normalization_stage
        self.trainer_args = trainer_args
        self.model = model
        self.datamodule = datamodule

    def run(
        self,
        task_id: int | None = None,
    ) -> AOITiledEnsembleEngine:
        """Run train job that fits the model for given tile location.

        Args:
            task_id: Passed when job is ran in parallel.

        Returns:
            AOITiledEnsembleEngine: Engine containing trained model.
        """
        devices: str | List[int] = "auto"
        if task_id is not None:
            n_gpus = torch.cuda.device_count()
            gpu_id = task_id % n_gpus if n_gpus > 0 else 0
            devices = [gpu_id]
            logger.info(f"Running job {self.model.__class__.__name__} with task_id {task_id} → device {gpu_id}")

        logger.info(f"Running {self.__class__}")
        logger.info("Training for tile at position %s,", self.tile_index)
        seed_everything(self.seed)

        # create engine for specific tile location and fit the model
        logger.info(f"Creating engine for tile {self.tile_index} on device {devices}, accelerator: {self.accelerator}, trainer_args: {self.trainer_args}")
        engine = get_ensemble_engine(
            tile_index=self.tile_index,
            total_tiles=self.total_tiles,
            accelerator=self.accelerator,
            devices=devices,
            root_dir=self.root_dir,
            trainer_args=self.trainer_args,
        )

        logger.info(f"Fitting model for tile {self.tile_index} on device {devices}, accelerator: {self.accelerator}")
        logger.info(f"{self.name}: Engine: {engine}, Model {self.model}, datamodule: {self.datamodule}")
        # logger.info(f"Batches: {len(self.datamodule.train_data)} - Batchsize = {self.dataloader.batch_size}")

        engine.fit(model=self.model, datamodule=self.datamodule)
        # move model to cpu to avoid memory issues as the engine is returned to be used in validation phase
        engine.model.cpu()

        return engine

    @staticmethod
    def collect(results: List[AOITiledEnsembleEngine]) -> dict[tuple[int, int], AOITiledEnsembleEngine]:
        """Collect engines from each tile location into a dict.

        Returns:
            dict[tuple[int, int], AOITiledEnsembleEngine]: Dict has form {tile_index: AOITiledEnsembleEngine}
        """
        return {r.tile_index: r for r in results}

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Skip as checkpoints are already saved by callback."""


class TrainModelJobGenerator(JobGenerator):
    """Generator for training job that train model for each tile location.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    def __init__(
        self,
        seed: int,
        accelerator: str,
        root_dir: Path,
        tilingPipelineConfig: TilingPipelineConfig,
        dataModuleConfig: DataModuleConfig|None,
        modelConfig: ModelConfig,
        datamodule: AnomalibDataModule|None,
        normalization_stage: NormalizationStage,
    ) -> None:
        self.seed = seed
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.tilingPipelineConfig = tilingPipelineConfig
        self.dataModuleConfig = dataModuleConfig
        self.modelConfig = modelConfig
        self.datamodule = datamodule
        self.normalization_stage = normalization_stage
        try:
            if dataModuleConfig is None:
                assert datamodule is not None
            if datamodule is None:
                assert dataModuleConfig is not None
        except AssertionError:
            logger.error(f"TrainModelGenerator: Either data_args or datamodule needs to be available.")


    @property
    def job_class(self) -> type:
        """Return the job class."""
        return TrainModelJob

    def generate_jobs(
        self,
        args: dict[str,Any] | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[TrainModelJob, None, None]:
        """Generate training jobs for each tile location.

        Args:
            args (dict): Dict with config passed to training.
            prev_stage_result (None): Not used here.

        Returns:
            Generator[TrainModelJob, None, None]: TrainModelJob generator
        """
        del prev_stage_result  # Not needed for this job
        if args is None:
            msg = "TrainModels job requires config args"
            raise ValueError(msg)

        # tiler used for splitting the image and getting the tile count
        tiler = get_ensemble_tiler(self.tilingPipelineConfig.to_dict())

        logger.info(
            f"Tiled ensemble training started. Separate models will be trained for {tiler.num_tiles} tile locations.",
        )
        # go over all tile positions
        for tile_index in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
            # prepare datamodule with custom collate function that only provides specific tile of image
            datamodule = get_ensemble_datamodule(
                data_config=args,
                image_size=self.tilingPipelineConfig.image_size,
                tiler=tiler,
                tile_index=tile_index,
                datamodule=self.datamodule
            )
            model = get_ensemble_model(
                modelConfig=self.modelConfig,
                normalization_stage=self.normalization_stage,
                input_size=self.tilingPipelineConfig.tile_size,
            )

            if self.dataModuleConfig is not None and self.dataModuleConfig.train_batch_size == "auto":
                resolved_accelerator = self.accelerator if self.accelerator != "auto" else get_device()
                datamodule.train_batch_size = probe_optimal_batch_size(
                    model=model,
                    datamodule=datamodule,
                    accelerator=resolved_accelerator,
                    root_dir=self.root_dir,
                    batch_arg_name="train_batch_size",
                    method="fit",
                    max_val=max_train_batch_size(
                        len(datamodule.train_data), self.dataModuleConfig.max_train_batch_size,
                    ),
                )

            # pass root_dir to engine so all models in ensemble have the same root dir
            yield TrainModelJob(
                accelerator=self.accelerator,
                seed=self.seed,
                root_dir=self.root_dir,
                tile_index=tile_index,
                total_tiles=tiler.num_tiles,
                normalization_stage=self.normalization_stage,
                trainer_args=args,
                model=model,
                datamodule=datamodule,
            )

"""Tiled ensemble - ensemble prediction job."""
class InferenceJob(Job):
    """Job for generating predictions with individual models in the tiled ensemble.

    Args:
        accelerator (str): Accelerator (device) to use.
        seed (int): Random seed for reproducibility.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        tile_index (tuple[int, int]): Index of tile that this model processes.
        normalization_stage (str): Normalization stage flag.
        dataloader (DataLoader): Dataloader to use for training (either val or test).
        model (AnomalyModule): Model to train.
        engine (AOITiledEnsembleEngine | None):
            engine from train job. If job is used standalone, instantiate engine and model from checkpoint.
        ckpt_path (Path | None): Path to checkpoint to be loaded if engine doesn't contain correct weights.
        trainer_args (dict | None): `{"callbacks": [...]}` only (see
            InferenceJobGenerator.generate_jobs' predict_trainer_args for why it's just
            callbacks, e.g. a configured GUIPredictProgressCallback -- see
            tiling/gui_predict_callback.py -- and not the rest of the source config's
            kwargs) to hand to get_ensemble_engine when this job builds its own engine (the
            `engine is None` case below). Only meaningful there -- a reused, already-built
            engine (predict chained right after training) already has its own callbacks.
        total_tiles (int): Total tile count for this predict run (see
            InferenceJobGenerator.generate_jobs), forwarded to get_ensemble_engine so
            GUIPredictProgressCallback (if configured) knows the true denominator for its
            tile-progress fraction without needing it set by hand in the inferencer yaml.

    """

    name = "Predict"

    def __init__(
        self,
        accelerator: str,
        seed: int,
        root_dir: Path,
        tile_index: tuple[int, int],
        normalization_stage: str,
        dataloader: DataLoader[FODataset|InferenceDataset],
        model: AnomalibModule | None,
        engine: AOITiledEnsembleEngine | None,
        ckpt_path: Path | None,
        trainer_args: dict[str, Any] | None = None,
        total_tiles: int = 1,
    ) -> None:
        super().__init__()
        if engine is None and ckpt_path is None:
            msg = "Either engine or checkpoint must be provided to predict job."
            raise ValueError(msg)

        self.accelerator = accelerator
        if self.accelerator == "auto":
            self.accelerator = get_device()
        self.seed = seed
        self.root_dir = root_dir
        self.tile_index = tile_index
        self.normalization_stage = normalization_stage
        self.dataloader = dataloader
        self.model = model
        self.engine = engine
        self.ckpt_path = ckpt_path
        self.trainer_args = trainer_args
        self.total_tiles = total_tiles

    def run(
        self,
        task_id: int | None = None,
    ) -> tuple[tuple[int, int], Any | None]:
        """Predict job that predicts the data with specific model for given tile location.

        Args:
            task_id: Passed when job is ran in parallel.

        Returns:
            tuple[tuple[int, int], List[Any]]: Tile index, List of predictions.
        """
        devices: str | List[int] = "auto"
        if task_id is not None:
            n_gpus = torch.cuda.device_count()
            gpu_id = task_id % n_gpus if n_gpus > 0 else 0
            devices = [gpu_id]
            logger.info(f"Running job {self.model.__class__.__name__} with task_id {task_id} → device {gpu_id}")

        logger.info("Start of predicting for tile at position %s,", self.tile_index)
        seed_everything(self.seed)

        if self.engine is None:
            # in case predict is invoked separately from train job, make new engine instance
            self.engine = get_ensemble_engine(
                tile_index=self.tile_index,
                accelerator=self.accelerator,
                devices=devices,
                root_dir=self.root_dir,
                trainer_args=self.trainer_args,
                total_tiles=self.total_tiles,
            )

        logger.info(f"Batches: {len(self.dataloader)} - Batchsize = {self.dataloader.batch_size}")

        predictions:list[Any]|list[list[Any]]|None = self.engine.predict(model=self.model, dataloaders=self.dataloader, ckpt_path=self.ckpt_path)

        # also return tile index as it's needed in collect method

        return self.tile_index, predictions

    @staticmethod
    def collect(results: List[tuple[tuple[int, int], List[Any]]]) -> EnsemblePredictions:
        """Collect predictions from each tile location into the predictions class.

        Returns:
            EnsemblePredictions: Object containing all predictions in form ready for merging.
        """
        storage = EnsemblePredictions()

        for tile_index, predictions in results:
            storage.add_tile_prediction(tile_index, predictions)

        return storage

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """This stage doesn't save anything."""

class InferenceJobGenerator(JobGenerator):
    """Generator for predict job that uses individual models to predict for each tile location.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
        data_source (Split): Whether to predict on validation set. If false use test set.
    """

    def __init__(
        self,
        data_source: Split,
        seed: int,
        accelerator: str,
        root_dir: Path,
        tilingPipelineConfig: TilingPipelineConfig,
        dataModuleConfig: DataModuleConfig,
        modelConfig: ModelConfig,
        normalization_stage: NormalizationStage,
        datamodule:AnomalibDataModule|None,
        inferenceDataset:InferenceDataset|None,
        ckptPath:Path|None=None
    ) -> None:
        self.data_source = data_source
        self.seed = seed
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.tilingPipelineConfig = tilingPipelineConfig
        self.dataModuleConfig = dataModuleConfig
        self.modelConfig = modelConfig
        self.normalization_stage = normalization_stage
        self.datamodule = datamodule
        self.ckptPath = ckptPath
        self.inferenceDataset = inferenceDataset

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return InferenceJob

    def generate_jobs(
        self,
        args: dict[str,Any] | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[InferenceJob, None, None]:
        """Generate predict jobs for each tile location.

        Args:
            args (dict): Trainer kwargs from the inferencer config (see
                _pipeline_trainer_kwargs(self.inferencerConfig) in InferenceTiledEnsemble.run) --
                threaded into each tile's InferenceJob as trainer_args so a configured
                GUIPredictProgressCallback (or any other predict-time callback) actually
                reaches the Trainer this generator's jobs build, instead of being dropped here.
            prev_stage_result (dict[tuple[int, int], AOITiledEnsembleEngine] | None):
                if called after train job this contains engines with individual models, otherwise load from checkpoints.

        Returns:
            Generator[PredictJob, None, None]: PredictJob generator.
        """
        # tiler used for splitting the image and getting the tile count
        tiler = get_ensemble_tiler(self.tilingPipelineConfig.to_dict())

        logger.info(
            "Tiled ensemble predicting started using Using ckpt_path%s data.",
            self.data_source.value,
        )

        # Only forward `callbacks` (e.g. a configured GUIPredictProgressCallback) out of
        # `args`, not the whole dict -- InferenceJobGenerator is shared by both standalone
        # predict (args built from the inferencer config) and EvalTiledEnsemble's
        # calibration val/test passes (args built from the *training* config, which is
        # where the project's wandb logger normally lives). Forwarding `args` wholesale
        # would hand that wandb logger to every per-tile predict engine built here too --
        # something this generator never did before (get_ensemble_engine's callers here
        # always passed trainer_args=None) -- and each of this generator's separate
        # generate_jobs() calls (once per data_source mode in EvalTiledEnsemble, e.g. val
        # then test) deep-copies its own fresh logger instance via _pipeline_trainer_kwargs,
        # so two modes visiting the same tile_index collide over the same wandb run id
        # (see ensemble_engine.py's _adjust_wandb_logger). Stripping logger here keeps
        # predict-time callbacks working without resurrecting that collision.
        predict_trainer_args = {"callbacks": args["callbacks"]} if args and args.get("callbacks") else None
        # go over all tile positions
        for tile_index in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
            # prepare datamodule with custom collate function that only provides specific tile of image
            datamodule = get_ensemble_datamodule(
                data_config=self.dataModuleConfig.to_dict(),
                image_size=self.tilingPipelineConfig.image_size,
                tiler=tiler,
                tile_index=tile_index,
                datamodule=self.datamodule
            )

            # check if predict step is positioned after training
            if prev_stage_result and tile_index in prev_stage_result:
                engine = prev_stage_result[tile_index]
                # model is inside engine in this case
                if isinstance(engine, AOITiledEnsembleEngine):
                    model = engine.model
                else:
                    raise AttributeError(f"prev_stage_result does not contain engine")
                ckpt_path = None
                logger.info(f"Using model from previous training job. No loading from checkpoint")
            else:
                # any other case - predict is called standalone
                engine = None
                # we need to make new model instance as it's not inside engine
                model = get_ensemble_model(
                    modelConfig=self.modelConfig,
                    normalization_stage=self.normalization_stage,
                    input_size=self.tilingPipelineConfig.tile_size
                )
                tile_i, tile_j = tile_index
                # prepare checkpoint path for model on current tile location
                
                # ckpt_path = self.root_dir / "weights" / "lightning" / f"model{tile_i}_{tile_j}.ckpt"
                if self.ckptPath is None:
                    ckpt_path = self.root_dir / "checkpoints" / f"model{tile_i}_{tile_j}.ckpt"
                else:
                    ckpt_path = self.ckptPath / f"model{tile_i}_{tile_j}.ckpt"
                logger.info(f"Loading checkpoint from ckpt_path: {ckpt_path}. No Model from previous training job available.")
                

            # pick the dataloader based on predict data
            if self.inferenceDataset:
                logger.info(f"Using a specified dataset for dataloader (usually the case during inference): {self.inferenceDataset}")
                dataloader:DataLoader[InferenceDataset] = DataLoader(self.inferenceDataset, collate_fn=datamodule.external_collate_fn, pin_memory=True)
            else:
                logger.info(f"Creating a dataloader from a datamodule. Usually the case during the validation or test step.")
                if self.dataModuleConfig is not None and self.dataModuleConfig.eval_batch_size == "auto":
                    resolved_accelerator = self.accelerator if self.accelerator != "auto" else get_device()
                    datamodule.eval_batch_size = probe_optimal_batch_size(
                        model=model,
                        datamodule=datamodule,
                        accelerator=resolved_accelerator,
                        root_dir=self.root_dir,
                        batch_arg_name="eval_batch_size",
                        method="validate" if self.data_source == Split.VAL else "test",
                    )


                dataloader = datamodule.test_dataloader()
                if self.data_source == Split.VAL:
                    dataloader = datamodule.val_dataloader()
                    print(f"Using validation data")
                else:
                    print(f"Using test data")

            # pass root_dir to engine so all models in ensemble have the same root dir
            yield InferenceJob(
                accelerator=self.accelerator,
                seed=self.seed,
                root_dir=self.root_dir,
                tile_index=tile_index,
                normalization_stage=self.normalization_stage,
                model=model,
                dataloader=dataloader,
                engine=engine,
                ckpt_path=ckpt_path,
                trainer_args=predict_trainer_args,
                total_tiles=tiler.num_tiles,
            )

class FOPredictJob(Job):
    """Job for generating predictions with individual models in the tiled ensemble.

    Args:
        accelerator (str): Accelerator (device) to use.
        seed (int): Random seed for reproducibility.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        tile_index (tuple[int, int]): Index of tile that this model processes.
        normalization_stage (str): Normalization stage flag.
        dataloader (DataLoader): Dataloader to use for training (either val or test).
        model (AnomalyModule): Model to train.
        engine (AOITiledEnsembleEngine | None):
            engine from train job. If job is used standalone, instantiate engine and model from checkpoint.
        ckpt_path (Path | None): Path to checkpoint to be loaded if engine doesn't contain correct weights.

    """

    name = "Predict"

    def __init__(
        self,
        accelerator: str,
        seed: int,
        root_dir: Path,
        tile_index: tuple[int, int],
        normalization_stage: str,
        dataloader:DataLoader[FODataset],
        foDataset:FODataset,
        model: AnomalibModule | None,
        engine: AOITiledEnsembleEngine | None,
        ckpt_path: Path | None,
        key: str,
    ) -> None:
        super().__init__()
        if engine is None and ckpt_path is None:
            msg = "Either engine or checkpoint must be provided to predict job."
            raise ValueError(msg)

        self.accelerator = accelerator
        self.seed = seed
        self.root_dir = root_dir
        self.tile_index = tile_index
        self.normalization_stage = normalization_stage
        self.foDataset = foDataset
        self.model = model
        self.engine = engine
        self.ckpt_path = ckpt_path
        self.key = key

    def run(
        self,
        task_id: int | None = None,
    ) -> tuple[tuple[int, int], Any | None]:
        """Predict job that predicts the data with specific model for given tile location.

        Args:
            task_id: Passed when job is ran in parallel.

        Returns:
            tuple[tuple[int, int], List[Any]]: Tile index, List of predictions.
        """
        devices: str | List[int] = "auto"
        if task_id is not None:
            n_gpus = torch.cuda.device_count()
            gpu_id = task_id % n_gpus if n_gpus > 0 else 0
            devices = [gpu_id]
            logger.info(f"Running job {self.model.__class__.__name__} with task_id {task_id} → device {gpu_id}")

        logger.info("Start of predicting for tile at position %s,", self.tile_index)
        seed_everything(self.seed)

        if self.engine is None:
            # in case predict is invoked separately from train job, make new engine instance
            self.engine = get_ensemble_engine(
                tile_index=self.tile_index,
                accelerator=self.accelerator,
                devices=devices,
                root_dir=self.root_dir,
            )

        for sample in self.foDataset.iter_samples(autosave=True, progress=True):
            predictions = self.engine.predict(model=self.model, data_path=sample.filepath, ckpt_path=self.ckpt_path)

            output = predictions[0]

            conf = output.pred_score.item()
            anomaly = "anomaly" if output.pred_label else "normal"

            sample[f"pred_anomaly_score_{self.key}"] = conf
            sample[f"pred_anomaly_{self.key}"] = fo.Classification(label=anomaly)
            sample[f"pred_anomaly_map_{self.key}"] = fo.Heatmap(map=output.anomaly_map.data.numpy().squeeze()*255, range=[0,255])
            sample[f"pred_defect_mask_{self.key}"] = fo.Segmentation(mask=output.pred_mask.data.numpy().squeeze().astype(np.int16)*255)

        # also return tile index as it's needed in collect method
        return self.tile_index, predictions

    @staticmethod
    def collect(results: List[tuple[tuple[int, int], List[Any]]]) -> EnsemblePredictions:
        """Collect predictions from each tile location into the predictions class.

        Returns:
            EnsemblePredictions: Object containing all predictions in form ready for merging.
        """
        storage = EnsemblePredictions()

        for tile_index, predictions in results:
            storage.add_tile_prediction(tile_index, predictions)

        return storage

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """This stage doesn't save anything."""

class FOPredictJobGenerator(JobGenerator):
    """Generator for predict job that uses individual models to predict for each tile location.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
        data_source (Split): Whether to predict on validation set. If false use test set.
    """

    def __init__(
        self,
        data_source: Split,
        seed: int,
        accelerator: str,
        root_dir: Path,
        tilingPipelineConfig: TilingPipelineConfig,
        dataModuleConfig: DataModuleConfig,
        modelConfig: ModelConfig,
        normalization_stage: NormalizationStage,
        datamodule:FODataModule,
        dataset:FODataset
    ) -> None:
        self.data_source = data_source
        self.seed = seed
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.tilingPipelineConfig = tilingPipelineConfig
        self.dataModuleConfig = dataModuleConfig
        self.modelConfig = modelConfig
        self.normalization_stage = normalization_stage
        self.datamodule_ = datamodule
        self.dataset = dataset

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return FOPredictJob

    def generate_jobs(
        self,
        args: Dict[str,Any] | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[FOPredictJob, None, None]:
        """Generate predict jobs for each tile location.

        Args:
            args (dict): Dict with config passed to training.
            prev_stage_result (dict[tuple[int, int], AOITiledEnsembleEngine] | None):
                if called after train job this contains engines with individual models, otherwise load from checkpoints.

        Returns:
            Generator[PredictJob, None, None]: PredictJob generator.
        """
        del args  # args not used here

        # tiler used for splitting the image and getting the tile count
        tiler = get_ensemble_tiler(self.tiling_args)

        logger.info(
            "Tiled ensemble predicting started using %s data.",
            self.data_source.value,
        )
        # go over all tile positions
        for tile_index in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
            # prepare datamodule with custom collate function that only provides specific tile of image
            datamodule = get_ensemble_datamodule(
                data_config=self.dataModuleConfig.to_dict(),
                image_size=self.tilingPipelineConfig.image_size,
                tiler=tiler,
                tile_index=tile_index,
                datamodule=self.datamodule_
            )

            # check if predict step is positioned after training
            if prev_stage_result and tile_index in prev_stage_result:
                engine = prev_stage_result[tile_index]
                # model is inside engine in this case
                model = engine.model
                ckpt_path = None
            else:
                # any other case - predict is called standalone
                engine = None
                # we need to make new model instance as it's not inside engine
                model = get_ensemble_model(
                    modelConfig=self.modelConfig,
                    normalization_stage=self.normalization_stage,
                    input_size=self.tilingPipelineConfig.tile_size
                )
                tile_i, tile_j = tile_index
                # prepare checkpoint path for model on current tile location
                # ckpt_path = self.root_dir / "weights" / "lightning" / f"model{tile_i}_{tile_j}.ckpt"
                ckpt_path = self.root_dir / "checkpoints" / f"model{tile_i}_{tile_j}.ckpt"

            # pick the dataloader based on predict data
            dataloader = datamodule.test_dataloader()
            if self.data_source == Split.VAL:
                dataloader = datamodule.val_dataloader()

            # pass root_dir to engine so all models in ensemble have the same root dir
            yield FOPredictJob(
                accelerator=self.accelerator,
                seed=self.seed,
                root_dir=self.root_dir,
                tile_index=tile_index,
                normalization_stage=self.normalization_stage,
                model=model,
                dataloader=dataloader,
                engine=engine,
                ckpt_path=ckpt_path,
                foDataset=self.dataset,
                key=self.modelConfig.name
            )

"""Helper functions for the tiled ensemble training."""

def get_ensemble_datamodule(
    data_config: dict[str,Any],
    image_size: int | tuple[int, int],
    tiler: EnsembleTiler,
    tile_index: tuple[int, int],
    datamodule: AnomalibDataModule|None = None,
) -> AnomalibDataModule:
    """Get Anomaly Datamodule adjusted for use in ensemble.

    Datamodule collate function gets replaced by TileCollater in order to tile all images before they are passed on.

    Args:
        data_config: tiled ensemble data configuration.
        image_size (int | tuple[int, int]): full effective image size of tiled ensemble.
        tiler (EnsembleTiler): Tiler used to split the images to tiles for use in ensemble.
        tile_index (tuple[int, int]): Index of the tile in the split image.

    Returns:
        AnomalibDataModule: Anomalib Lightning DataModule
    """
    if datamodule == None:
        datamodule = get_datamodule(data_config)
        datamodule.setup()

    # add tiled ensemble image_size transform to datamodule
    setup_transforms(datamodule, image_size=image_size)
    datamodule.external_collate_fn = TileCollater(tiler, tile_index, default_collate_fn=ImageBatch.collate)
    # manually set setup, so later setup doesn't override the transforms...
    datamodule._is_setup = True  # noqa: SLF001

    return datamodule

def setup_transforms(datamodule: AnomalibDataModule, image_size: int | tuple[int, int]) -> None:
    """Modify datamodule resize transforms so the effective ensemble image_size is correct.

    Args:
        datamodule: datamodule where resize transform will be setup.
        image_size (int | tuple[int, int]): tiled ensemble input image size

    """
    resize_transform = Resize(image_size)

    for subset_name in ["train", "val", "test"]:
        default_aug = getattr(datamodule, f"{subset_name}_augmentations", None)

        if isinstance(default_aug, Resize):
            msg = f"Conflicting resize shapes found between dataset augmentations and tiled ensemble size. \
                You are using a Resize transform in your input data augmentations. Please be aware that the \
                tiled ensemble image size is determined by tiling config. The final effective input size as \
                seen by individual model will be determined by the tile_size. To change \
                the effective ensemble input size, please change the image_size in the tiling config. \
                Augmentations: {default_aug.size}, Tiled ensemble base size: {image_size}"
            logger.warning(msg)
            augmentations = resize_transform
        elif isinstance(default_aug, Compose):
            augmentations = Compose([*default_aug.transforms, resize_transform])
        elif isinstance(default_aug, Transform):
            augmentations = Compose([default_aug, resize_transform])
        else:
            augmentations = resize_transform
        # add augmentations with resize to datamodule and datasets, ensuring that output images match effective size
        setattr(datamodule, f"{subset_name}_augmentations", augmentations)
        data_subset = getattr(datamodule, f"{subset_name}_data", None)
        if data_subset is not None:
            data_subset.augmentations = augmentations

def get_ensemble_tiler(tiling_args: Dict[str,Any]) -> EnsembleTiler:
    """Get tiler used for image tiling and to obtain tile dimensions.

    Args:
        tiling_args: tiled ensemble tiling configuration.

    Returns:
        EnsembleTiler: tiler object.
    """
    tiler = EnsembleTiler(
        tile_size=tiling_args["tile_size"],
        stride=tiling_args["stride"],
        image_size=tiling_args["image_size"],
    )

    return tiler  # noqa: RET504


def parse_trainer_kwargs(trainer_args: dict | None) -> Namespace | dict:
    """Parse trainer args and instantiate all needed elements.

    Transforms config into kwargs ready for Trainer, including instantiation of callback etc.

    Args:
        trainer_args (dict): Trainer args dictionary.

    Returns:
        dict: parsed kwargs with instantiated elements.
    """
    if not trainer_args:
        return {}

    # try to get trainer args, if not present return empty
    parser = ArgumentParser()

    parser.add_class_arguments(Trainer, fail_untyped=False, instantiate=False, sub_configs=True)
    config = parser.parse_object(trainer_args)
    objects = parser.instantiate_classes(config)

    return objects  # noqa: RET504

def get_ensemble_engine(
    tile_index: tuple[int, int],
    accelerator: str,
    devices: List[int] | str | int,
    root_dir: Path,
    trainer_args: dict | None = None,
    total_tiles: int = 1,
) -> AOITiledEnsembleEngine:
    """Prepare engine for ensemble training or prediction.

    This method makes sure correct normalization is used, prepares metrics and additional trainer kwargs..

    Args:
        tile_index (tuple[int, int]): Index of tile that this model processes.
        accelerator (str): Accelerator (device) to use.
        devices (List[int] | str | int): device IDs used for training.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        trainer_args (dict): Trainer args dictionary. Empty dict if not present.
        total_tiles (int): Total tile count for this run (see TrainModelJob). Only meaningful
            for training -- callers on the predict-only path (no total tile count to give)
            can leave this at its default; it's just forwarded to
            GUITrainingProgressCallback via AOITiledEnsembleEngine, which only ever reports
            progress from inside a trainer.fit() call.

    Returns:
        AOITiledEnsembleEngine: set up engine for ensemble training/prediction.
    """
    # parse additional trainer args and callbacks if present in config
    trainer_kwargs = parse_trainer_kwargs(trainer_args)
    # remove keys that we already have
    trainer_kwargs.pop("accelerator", None)
    trainer_kwargs.pop("default_root_dir", None)
    trainer_kwargs.pop("devices", None)

    logger.info("Engine: Accelerator: %s, devices: %s, trainer_kwargs: %s", accelerator, devices, trainer_kwargs)

    # create engine for specific tile location
    engine = AOITiledEnsembleEngine(
        tile_index=tile_index,
        total_tiles=total_tiles,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=root_dir,
        **trainer_kwargs,
    )

    return engine  # noqa: RET504


def get_threshold_values(normalization_stage: NormalizationStage, root_dir: Path) -> tuple[float, float]:
    """Get threshold values for image and pixel level predictions.

    If normalization is not used, get values based on statistics obtained from validation set.
    If normalization is used, both image and pixel threshold are 0.5

    Args:
        normalization_stage (NormalizationStage): ensemble run args, used to get normalization stage.
        root_dir (Path): path to run root where stats file is saved.

    Returns:
        tuple[float, float]: image and pixel threshold.
    """
    if normalization_stage == NormalizationStage.NONE:
        stats_path = root_dir /"checkpoints" / "stats.json"
        with stats_path.open("r") as f:
            stats = json.load(f)
        image_threshold = stats["image_threshold"]
        pixel_threshold = stats["pixel_threshold"]
    else:
        # normalization transforms the scores so that threshold is at 0.5
        image_threshold = 0.5
        pixel_threshold = 0.5

    return image_threshold, pixel_threshold

