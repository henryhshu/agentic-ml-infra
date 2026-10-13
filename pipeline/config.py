"""Pipeline configuration module for AutoPARK.

Defines `PipelineConfig`, which acts as the shared-RAM state container and
knob-actuation surface manipulated dynamically by `RemediatorAgent` during
autonomous self-healing without requiring pod restarts or cold boots.
"""
from __future__ import annotations
import argparse
import os
import torch
from dataclasses import dataclass, asdict, field


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PipelineConfig:
    """AutoPARK Shared Runtime Configuration & Actuation Knob Interface.

    Contains all training hyperparameters, hardware bindings, and runtime flags.
    Mutated in shared memory by the RemediatorAgent upon anomaly detection to
    adjust learning rates, batch sizes, gradient clipping, or worker restart state.
    """
    model: str = 'tiny'
    epochs: int = 5
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 0
    ray_workers: int = 2
    use_gpu: bool = True
    device: str = field(default_factory=get_default_device)
    enable_agentic: bool = False
    max_retries: int = 3
    enable_tuning: bool = False
    tune_num_samples: int = 4
    subset: int | None = None
    cache_dir: str = 'data/cifar10'
    log_every: int = 50
    run_id: str | None = None
    s3_bucket: str | None = field(default_factory=lambda: os.environ.get("S3_BUCKET_NAME"))
    max_grad_norm: float | None = None  # Set by remediator to enable gradient clipping
    fault_injection: str | None = None  # Set by benchmark harness to inject controlled faults
    accum_steps: int = 1  # Micro-batch gradient accumulation steps
    preserve_effective_batch: bool = True  # Invariant effective batch preservation during remediation
    dataset: str = 'cifar10'  # Dataset identifier ('cifar10', 'synthetic', etc.)
    num_classes: int = 10  # Number of prediction target classes
    data_module: Any = field(default=None, repr=False)  # Optional BaseDataModule instance or factory
    model_fn: Any = field(default=None, repr=False)  # Optional custom model builder callable

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size invariant across micro-batch downscaling."""
        return self.batch_size * self.accum_steps

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PipelineConfig:
        d = vars(args)
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        d = {}
        for k in self.__dataclass_fields__:
            if k in ("data_module", "model_fn"):
                continue
            d[k] = getattr(self, k)
        if self.data_module is not None:
            d["data_module_name"] = getattr(self.data_module, "name", type(self.data_module).__name__)
        return d
