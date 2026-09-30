from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict

@dataclass
class PipelineConfig:
    model: str = 'tiny'
    epochs: int = 5
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 0
    ray_workers: int = 2
    use_gpu: bool = True
    device: str = 'mps'
    enable_agentic: bool = False
    max_retries: int = 3
    enable_tuning: bool = False
    tune_num_samples: int = 4
    subset: int | None = None
    cache_dir: str = 'data/cifar10'
    log_every: int = 50
    run_id: str | None = None
    max_grad_norm: float | None = None  # Set by remediator to enable gradient clipping
    fault_injection: str | None = None  # Set by benchmark harness to inject controlled faults

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PipelineConfig:
        d = vars(args)
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)
