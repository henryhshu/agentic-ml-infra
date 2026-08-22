from __future__ import annotations
import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Any

from metrics.schemas import StageMetrics, Event, now_iso

class ExperimentTracker:
    def __init__(self, run_id: str | None = None, base_dir: str | Path = 'experiments'):
        self.base_dir = Path(base_dir)
        if run_id is None:
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            rand_hex = uuid.uuid4().hex[:4]
            self.run_id = f"run_{now_str}_{rand_hex}"
        else:
            self.run_id = run_id
            
        self.run_dir = self.base_dir / self.run_id
        self.checkpoints_dir = self.run_dir / 'checkpoints'
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.config: dict[str, Any] = {}
        self.stages: dict[str, StageMetrics] = {}
        self.start_time = now_iso()
        self.end_time: str | None = None
        self.duration_s: float | None = None
        
        self._update_symlink()

    def _update_symlink(self):
        latest_link = self.base_dir / 'latest'
        if latest_link.exists() or latest_link.is_symlink():
            try:
                latest_link.unlink()
            except OSError:
                pass
        try:
            latest_link.symlink_to(self.run_dir.name)
        except OSError:
            pass

    def start_stage(self, stage_name: str, metrics_cls: type = StageMetrics) -> StageMetrics:
        stage = metrics_cls(
            stage_name=stage_name,
            start_time=now_iso(),
            status='running'
        )
        self.stages[stage_name] = stage
        return stage

    def end_stage(self, stage_name: str, status: str = 'success'):
        if stage_name in self.stages:
            stage = self.stages[stage_name]
            stage.end_time = now_iso()
            stage.status = status # type: ignore
            start_dt = datetime.datetime.fromisoformat(stage.start_time)
            end_dt = datetime.datetime.fromisoformat(stage.end_time)
            stage.duration_s = (end_dt - start_dt).total_seconds()
            self.save_stage(stage_name)
            self.save_summary()

    def log_metric(self, stage_name: str, key: str, value: Any):
        if stage_name in self.stages:
            self.stages[stage_name].metrics[key] = value

    def log_event(self, stage_name: str, event_type: str, message: str, data: dict | None = None):
        if stage_name in self.stages:
            event = Event(
                timestamp=now_iso(),
                event_type=event_type,
                stage=stage_name,
                message=message,
                data=data
            )
            self.stages[stage_name].events.append(event)

    def save_stage(self, stage_name: str):
        if stage_name in self.stages:
            stage = self.stages[stage_name]
            out_path = self.run_dir / f"{stage_name}_metrics.json"
            with open(out_path, 'w') as f:
                json.dump(stage.to_dict(), f, indent=2)

    def save_summary(self):
        out_path = self.run_dir / "experiment_summary.json"
        end_time = self.end_time or now_iso()
        
        start_dt = datetime.datetime.fromisoformat(self.start_time)
        end_dt = datetime.datetime.fromisoformat(end_time)
        duration_s = (end_dt - start_dt).total_seconds()
        
        summary = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": end_time,
            "duration_s": duration_s,
            "config": self.config,
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
            "status": getattr(self, 'status', None) or ("completed" if self.end_time else "running")
        }
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def set_config(self, config: dict):
        self.config = config
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.save_summary()

    def get_run_dir(self) -> Path:
        return self.run_dir
