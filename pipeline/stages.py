from __future__ import annotations
import traceback
from enum import Enum
from abc import ABC, abstractmethod
from typing import Any

from metrics.schemas import StageMetrics
from metrics.experiment import ExperimentTracker
from pipeline.config import PipelineConfig

class Stage(Enum):
    DATA_VALIDATION = 'data_validation'
    DATA_LOADING = 'data_loading'
    TRAINING = 'training'
    TUNING = 'tuning'
    EVALUATION = 'evaluation'

class StageRunner(ABC):
    def __init__(self, config: PipelineConfig, tracker: ExperimentTracker, checkpoint_emitter: Any | None = None):
        self.config = config
        self.tracker = tracker
        self.checkpoint_emitter = checkpoint_emitter

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def run(self) -> StageMetrics:
        pass

    def _emit_checkpoint(self, event_type: str, data: dict | None = None):
        if self.checkpoint_emitter is not None:
            if hasattr(self.checkpoint_emitter, 'emit'):
                self.checkpoint_emitter.emit(event_type, data)
            elif callable(self.checkpoint_emitter):
                self.checkpoint_emitter(event_type, data)

    def _run_with_tracking(self) -> StageMetrics:
        stage = self.tracker.start_stage(self.name)
        max_retries = self.config.max_retries
        attempts = 0
        
        while attempts <= max_retries:
            try:
                metrics_dict = self.execute()
                stage.metrics.update(metrics_dict)
                self.tracker.end_stage(self.name, status='success')
                return stage
            except Exception as e:
                attempts += 1
                stage.retries += 1
                self.tracker.log_event(
                    self.name, 
                    "error", 
                    f"Stage failed attempt {attempts}: {str(e)}", 
                    data={"traceback": traceback.format_exc()}
                )
                if attempts > max_retries:
                    stage.error_message = str(e)
                    self.tracker.end_stage(self.name, status='failed')
                    raise e
                else:
                    self.tracker.log_event(self.name, "retry", f"Retrying stage ({attempts}/{max_retries})")
        return stage

    @abstractmethod
    def execute(self) -> dict:
        pass
