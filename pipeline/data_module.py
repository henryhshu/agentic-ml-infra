"""AutoPARK DataModule Abstraction.

Decouples data validation, partitioning, transformation, and loader creation
from the core AutoPARK orchestration and self-healing engine. Allows ML engineers
to plug in arbitrary vision, NLP, audio, or tabular datasets with zero friction.
"""
from __future__ import annotations

import io
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms


class BaseDataModule(ABC):
    """Abstract Base Class for AutoPARK Data Modules."""

    name: str = "base_datamodule"
    num_classes: int = 10
    class_names: list[str] | None = None

    def __init__(self, cache_dir: str | Path = "data", subset: int | None = None):
        self.cache_dir = Path(cache_dir)
        self.subset = subset
        self.is_setup = False
        self._train_samples: int = 0
        self._val_samples: int = 0

    @abstractmethod
    def setup(self, stage: str | None = None) -> None:
        """Download, parse, and partition dataset splits."""
        pass

    @abstractmethod
    def validate_schema(self) -> tuple[bool, list[str]]:
        """Validate dataset schema, row counts, and sample decodability.
        
        Returns:
            (is_valid, list_of_issues_or_warnings)
        """
        pass

    @abstractmethod
    def train_dataloader(
        self, batch_size: int, num_workers: int = 0, pin_memory: bool = False
    ) -> DataLoader:
        """Construct training DataLoader."""
        pass

    @abstractmethod
    def val_dataloader(
        self, batch_size: int, num_workers: int = 0, pin_memory: bool = False
    ) -> DataLoader:
        """Construct validation / evaluation DataLoader."""
        pass

    @property
    def train_samples(self) -> int:
        return self._train_samples

    @property
    def val_samples(self) -> int:
        return self._val_samples


class CIFAR10DataModule(BaseDataModule):
    """Production CIFAR-10 DataModule backed by Hugging Face Parquet shards."""

    name: str = "cifar10"
    num_classes: int = 10
    class_names: list[str] = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)

    def __init__(self, cache_dir: str | Path = "data/cifar10", subset: int | None = None):
        super().__init__(cache_dir, subset)
        self.train_set: Dataset | None = None
        self.test_set: Dataset | None = None
        self.train_path: Path | None = None
        self.test_path: Path | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.is_setup:
            return

        from datasets.cifar10 import CIFAR10Parquet, download_split

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.train_path = download_split("train", self.cache_dir)
        self.test_path = download_split("test", self.cache_dir)

        train_tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        eval_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        train_set = CIFAR10Parquet(self.train_path, transform=train_tfm)
        test_set = CIFAR10Parquet(self.test_path, transform=eval_tfm)

        if self.subset is not None:
            train_set = Subset(train_set, range(min(self.subset, len(train_set))))
            test_set = Subset(test_set, range(min(self.subset, len(test_set))))

        self.train_set = train_set
        self.test_set = test_set
        self._train_samples = len(train_set)
        self._val_samples = len(test_set)
        self.is_setup = True

    def validate_schema(self) -> tuple[bool, list[str]]:
        import pyarrow.parquet as pq
        from PIL import Image

        issues: list[str] = []
        try:
            self.setup()
            for split_name, path in [("train", self.train_path), ("test", self.test_path)]:
                if not path or not path.exists():
                    issues.append(f"Parquet shard {split_name} missing at {path}")
                    continue
                meta = pq.read_metadata(str(path))
                cols = set(meta.schema.names)
                if not {"img", "label"}.issubset(cols):
                    issues.append(f"{split_name} schema missing required columns: got {cols}")
            
            # Verify sample decode
            if self.train_set and len(self.train_set) > 0:
                x, y = self.train_set[0]
                if not isinstance(x, torch.Tensor) or not isinstance(y, int):
                    issues.append(f"Invalid decoded sample type: ({type(x)}, {type(y)})")
        except Exception as e:
            issues.append(f"CIFAR10 validation failed with exception: {e}")

        return (len(issues) == 0, issues)

    def train_dataloader(
        self, batch_size: int, num_workers: int = 0, pin_memory: bool = False
    ) -> DataLoader:
        self.setup()
        return DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def val_dataloader(
        self, batch_size: int, num_workers: int = 0, pin_memory: bool = False
    ) -> DataLoader:
        self.setup()
        return DataLoader(
            self.test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class SyntheticVisionDataModule(BaseDataModule):
    """Lightweight, offline Synthetic Vision DataModule.
    
    Generates synthetic image tensors and balanced categorical labels.
    Requires zero external downloads, providing 100% deterministic testing and
    enabling new MLE pipelines to bootstrap instantly without external HuggingFace/S3 dependencies.
    """

    name: str = "synthetic"

    def __init__(
        self,
        cache_dir: str | Path = "data/synthetic",
        subset: int | None = None,
        num_classes: int = 10,
        train_samples: int = 1000,
        val_samples: int = 200,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        class_names: list[str] | None = None,
    ):
        super().__init__(cache_dir, subset)
        self.num_classes = num_classes
        self._target_train = train_samples if subset is None else min(subset, train_samples)
        self._target_val = val_samples if subset is None else min(subset, val_samples)
        self.image_shape = image_shape
        self.class_names = class_names or [f"category_{i}" for i in range(num_classes)]
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.is_setup:
            return

        torch.manual_seed(42)
        n_train = self._target_train
        n_val = self._target_val

        # Create structured synthetic images (separable signal + noise)
        x_train = torch.randn(n_train, *self.image_shape) * 0.1
        y_train = torch.randint(0, self.num_classes, (n_train,))
        for c in range(self.num_classes):
            mask = (y_train == c)
            if mask.any():
                x_train[mask, 0, :4, :4] += (c + 1.0) * 0.5

        x_val = torch.randn(n_val, *self.image_shape) * 0.1
        y_val = torch.randint(0, self.num_classes, (n_val,))
        for c in range(self.num_classes):
            mask = (y_val == c)
            if mask.any():
                x_val[mask, 0, :4, :4] += (c + 1.0) * 0.5

        self.train_dataset = TensorDataset(x_train, y_train)
        self.val_dataset = TensorDataset(x_val, y_val)
        self._train_samples = n_train
        self._val_samples = n_val
        self.is_setup = True

    def validate_schema(self) -> tuple[bool, list[str]]:
        self.setup()
        issues: list[str] = []
        if len(self.train_dataset) == 0:
            issues.append("Synthetic train dataset is empty")
        if len(self.val_dataset) == 0:
            issues.append("Synthetic val dataset is empty")
        x, y = self.train_dataset[0]
        if tuple(x.shape) != self.image_shape:
            issues.append(f"Sample shape mismatch: expected {self.image_shape}, got {x.shape}")
        if not (0 <= int(y) < self.num_classes):
            issues.append(f"Label {y} outside bounds [0, {self.num_classes})")
        return (len(issues) == 0, issues)

    def train_dataloader(
        self, batch_size: int, num_workers: int = 0, pin_memory: bool = False
    ) -> DataLoader:
        self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def val_dataloader(
        self, batch_size: int, num_workers: int = 0, pin_memory: bool = False
    ) -> DataLoader:
        self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class CustomDataModule(BaseDataModule):
    """Wraps pre-existing PyTorch Datasets or custom functions directly."""

    name: str = "custom"

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_classes: int = 10,
        class_names: list[str] | None = None,
        validator_fn: Callable[[], tuple[bool, list[str]]] | None = None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.class_names = class_names
        self.validator_fn = validator_fn
        self._train_samples = len(train_dataset)
        self._val_samples = len(val_dataset)
        self.is_setup = True

    def setup(self, stage: str | None = None) -> None:
        pass

    def validate_schema(self) -> tuple[bool, list[str]]:
        if self.validator_fn:
            return self.validator_fn()
        issues: list[str] = []
        if len(self.train_dataset) == 0:
            issues.append("Training dataset is empty")
        return (len(issues) == 0, issues)

    def train_dataloader(
        self, batch_size: int, num_workers: int = 0, pin_memory: bool = False
    ) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def val_dataloader(
        self, batch_size: int, num_workers: int = 0, pin_memory: bool = False
    ) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


def create_data_module(config: Any) -> BaseDataModule:
    """Factory creating the appropriate DataModule for a given PipelineConfig."""
    if hasattr(config, "data_module") and config.data_module is not None:
        if isinstance(config.data_module, BaseDataModule):
            return config.data_module
        if callable(config.data_module):
            return config.data_module()

    dataset_name = getattr(config, "dataset", "cifar10").lower()
    subset = getattr(config, "subset", None)
    cache_dir = getattr(config, "cache_dir", "data/cifar10")

    if dataset_name == "cifar10":
        return CIFAR10DataModule(cache_dir=cache_dir, subset=subset)
    elif dataset_name in ("synthetic", "synthetic_vision"):
        num_classes = getattr(config, "num_classes", 10)
        return SyntheticVisionDataModule(
            cache_dir=cache_dir,
            subset=subset,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported built-ins: 'cifar10', 'synthetic'. "
            f"Or supply a custom BaseDataModule instance via config.data_module."
        )
