from __future__ import annotations
import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

def now_iso() -> str:
    """Returns current time in ISO format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

@dataclass
class Event:
    timestamp: str
    event_type: str
    stage: str
    message: str
    data: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        return cls(**d)

@dataclass
class StageMetrics:
    stage_name: str
    start_time: str
    end_time: str | None = None
    duration_s: float | None = None
    status: Literal['pending', 'running', 'success', 'failed', 'retried', 'skipped'] = 'pending'
    metrics: dict = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    retries: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['events'] = [e.to_dict() if isinstance(e, Event) else e for e in self.events]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StageMetrics:
        d = d.copy()
        if 'events' in d:
            d['events'] = [Event.from_dict(e) if isinstance(e, dict) else e for e in d['events']]
        return cls(**d)

@dataclass
class DataValidationMetrics(StageMetrics):
    num_train_samples: int = 0
    num_test_samples: int = 0
    schema_valid: bool = False
    sample_decode_ok: bool = False
    data_quality_score: float = 0.0

@dataclass
class TrainingMetrics(StageMetrics):
    epochs_completed: int = 0
    final_train_loss: float | None = None
    final_test_accuracy: float | None = None
    best_test_accuracy: float | None = None
    best_epoch: int | None = None
    per_epoch: list[dict] = field(default_factory=list)

@dataclass
class TuningMetrics(StageMetrics):
    num_trials: int = 0
    best_config: dict | None = None
    best_trial_accuracy: float | None = None
    all_trial_results: list[dict] = field(default_factory=list)

@dataclass
class EvaluationMetrics(StageMetrics):
    test_accuracy: float | None = None
    test_loss: float | None = None
    per_class_accuracy: dict | None = None
    inference_throughput_sps: float | None = None
    confusion_matrix: list[list[int]] | None = None
