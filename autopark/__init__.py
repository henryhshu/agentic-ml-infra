"""AutoPARK: Autonomous Agentic Orchestration and Self-Healing for Distributed Machine Learning Infrastructure.

High-level Python SDK providing plug-and-play adoption for any PyTorch workload.
"""
from __future__ import annotations

from typing import Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pipeline.config import PipelineConfig
from pipeline.data_module import (
    BaseDataModule,
    CIFAR10DataModule,
    CustomDataModule,
    SyntheticVisionDataModule,
    create_data_module,
)
from pipeline.registry import get_model, list_models, register_model
from pipeline.runner import PipelineRunner


class AutoParkRunner:
    """High-Level Developer SDK for running AutoPARK on arbitrary ML tasks.
    
    Examples:
        >>> # Method 1: Config with registered model and built-in/synthetic dataset
        >>> config = PipelineConfig(model="tiny", dataset="synthetic", enable_agentic=True)
        >>> runner = AutoParkRunner(config=config)
        >>> result = runner.run()

        >>> # Method 2: Pass custom PyTorch DataLoaders & Model directly (< 10 lines)
        >>> runner = AutoParkRunner.from_loaders(
        ...     model=my_model,
        ...     train_loader=my_train_loader,
        ...     val_loader=my_val_loader,
        ...     enable_agentic=True
        ... )
        >>> result = runner.run()
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        model: nn.Module | Callable[..., nn.Module] | str | None = None,
        data_module: BaseDataModule | None = None,
        **config_overrides: Any,
    ):
        if config is None:
            self.config = PipelineConfig(**config_overrides)
        else:
            self.config = config
            for k, v in config_overrides.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        if model is not None:
            if isinstance(model, str):
                self.config.model = model
            elif callable(model) or isinstance(model, nn.Module):
                self.config.model_fn = model
                self.config.model = getattr(model, "__name__", "custom_model")

        if data_module is not None:
            self.config.data_module = data_module
            self.config.dataset = data_module.name
            if hasattr(data_module, "num_classes"):
                self.config.num_classes = data_module.num_classes

        self._runner = PipelineRunner(self.config)

    @classmethod
    def from_loaders(
        cls,
        model: nn.Module | Callable[..., nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = 10,
        class_names: list[str] | None = None,
        enable_agentic: bool = False,
        epochs: int = 1,
        lr: float = 0.01,
        **kwargs: Any,
    ) -> AutoParkRunner:
        """Construct an AutoParkRunner directly from existing PyTorch DataLoaders."""
        train_ds = train_loader.dataset
        val_ds = val_loader.dataset

        dm = CustomDataModule(
            train_dataset=train_ds,
            val_dataset=val_ds,
            num_classes=num_classes,
            class_names=class_names,
        )

        batch_size = kwargs.pop("batch_size", getattr(train_loader, "batch_size", 128) or 128)
        config = PipelineConfig(
            model=getattr(model, "__name__", "custom_model"),
            dataset="custom",
            num_classes=num_classes,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            enable_agentic=enable_agentic,
            data_module=dm,
            model_fn=model,
            **kwargs,
        )

        return cls(config=config, model=model, data_module=dm)

    def run(self) -> dict[str, Any]:
        """Execute the end-to-end AutoPARK pipeline."""
        return self._runner.run()


__all__ = [
    "AutoParkRunner",
    "PipelineConfig",
    "BaseDataModule",
    "CIFAR10DataModule",
    "SyntheticVisionDataModule",
    "CustomDataModule",
    "create_data_module",
    "register_model",
    "get_model",
    "list_models",
]
