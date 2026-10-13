"""Data loading stage — downloads data, builds datasets and DataLoaders."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from metrics.experiment import ExperimentTracker
from metrics.schemas import StageMetrics, now_iso


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class DataLoadingStage:
    """Pipeline stage that downloads data and builds PyTorch DataLoaders."""

    STAGE_NAME = "data_loading"

    def __init__(
        self,
        config,  # PipelineConfig
        tracker: ExperimentTracker,
        checkpoint_emitter=None,
    ):
        self.config = config
        self.tracker = tracker
        self.checkpoint_emitter = checkpoint_emitter
        # These are populated after run()
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.train_set = None
        self.test_set = None

    @property
    def name(self) -> str:
        return self.STAGE_NAME

    def run(self) -> StageMetrics:
        """Run data loading and return metrics."""
        metrics = self.tracker.start_stage(self.STAGE_NAME)
        self._emit_checkpoint("stage_start", {"stage": self.STAGE_NAME})

        try:
            result = self.execute()
            metrics.metrics.update(result)
            self.tracker.end_stage(self.STAGE_NAME, status="success")
            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("stage_end", result)
            print(
                f"  ✅ Data loading: {result['train_samples']} train, "
                f"{result['test_samples']} test samples "
                f"({result['load_time_s']:.1f}s)"
            )
            return metrics

        except Exception as exc:
            self.tracker.log_event(
                self.STAGE_NAME, "error", f"Data loading failed: {exc}"
            )
            self.tracker.end_stage(self.STAGE_NAME, status="failed")
            metrics.error_message = str(exc)
            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("anomaly_detected", {"error": str(exc)})
            raise

    def execute(self) -> dict:
        """Download data and build DataLoaders. Returns timing metrics."""
        t0 = time.time()
        pin_memory = self.config.device == "cuda"

        # Check if non-CIFAR DataModule or custom dataset is active
        if getattr(self.config, "data_module", None) is not None or getattr(self.config, "dataset", "cifar10").lower() != "cifar10":
            from pipeline.data_module import create_data_module
            dm = create_data_module(self.config)
            dm.setup("fit")
            self.train_loader = dm.train_dataloader(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=pin_memory,
            )
            self.test_loader = dm.val_dataloader(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=pin_memory,
            )
            self.train_set = getattr(dm, "train_dataset", getattr(dm, "train_set", None))
            self.test_set = getattr(dm, "val_dataset", getattr(dm, "test_set", None))
            train_samples = dm.train_samples
            test_samples = dm.val_samples
            load_time = time.time() - t0

            self.tracker.log_metric(self.STAGE_NAME, "train_samples", train_samples)
            self.tracker.log_metric(self.STAGE_NAME, "test_samples", test_samples)
            self.tracker.log_metric(self.STAGE_NAME, "load_time_s", round(load_time, 2))
            self.tracker.log_metric(self.STAGE_NAME, "batch_size", self.config.batch_size)

            return {
                "train_samples": train_samples,
                "test_samples": test_samples,
                "load_time_s": round(load_time, 2),
                "batch_size": self.config.batch_size,
            }

        from datasets.cifar10 import CIFAR10Parquet, download_split
        cache_dir = Path(self.config.cache_dir)

        # Build transforms
        train_tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        eval_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        self.tracker.log_event(self.STAGE_NAME, "info", "Downloading/loading parquet shards...")

        # Download splits
        train_path = download_split("train", cache_dir)
        test_path = download_split("test", cache_dir)

        # Build datasets
        train_set = CIFAR10Parquet(train_path, transform=train_tfm)
        test_set = CIFAR10Parquet(test_path, transform=eval_tfm)

        # Apply subset if configured
        if self.config.subset is not None:
            train_set = Subset(train_set, range(min(self.config.subset, len(train_set))))
            test_set = Subset(test_set, range(min(self.config.subset, len(test_set))))
            self.tracker.log_event(
                self.STAGE_NAME,
                "info",
                f"Subset mode: train={len(train_set)}, test={len(test_set)}",
            )

        self.train_set = train_set
        self.test_set = test_set

        # Build DataLoaders
        pin_memory = self.config.device == "cuda"
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
        )

        load_time = time.time() - t0
        train_samples = len(train_set)
        test_samples = len(test_set)

        self.tracker.log_metric(self.STAGE_NAME, "train_samples", train_samples)
        self.tracker.log_metric(self.STAGE_NAME, "test_samples", test_samples)
        self.tracker.log_metric(self.STAGE_NAME, "load_time_s", round(load_time, 2))
        self.tracker.log_metric(self.STAGE_NAME, "batch_size", self.config.batch_size)

        return {
            "train_samples": train_samples,
            "test_samples": test_samples,
            "load_time_s": round(load_time, 2),
            "batch_size": self.config.batch_size,
        }

    def _emit_checkpoint(self, event_type: str, data: dict | None = None) -> None:
        """Emit an agent checkpoint if emitter is available."""
        if self.checkpoint_emitter is not None:
            from agents.checkpoints import AgentCheckpoint
            checkpoint = AgentCheckpoint(
                stage=self.STAGE_NAME,
                event_type=event_type,
                timestamp=now_iso(),
                metrics_snapshot=data or {},
                anomalies_detected=[],
                agent_diagnosis=None,
                recommended_action=None,
                action_taken=None,
            )
            self.checkpoint_emitter.emit(checkpoint)
