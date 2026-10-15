from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

@dataclass
class AgentCheckpoint:
    stage: str
    event_type: str
    timestamp: str
    metrics_snapshot: dict[str, Any]
    anomalies_detected: list[str]
    agent_diagnosis: str | None
    recommended_action: str | None
    action_taken: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCheckpoint:
        return cls(**data)

class CheckpointEmitter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, checkpoint: AgentCheckpoint) -> None:
        filename = f"{checkpoint.stage}_{checkpoint.event_type}_{checkpoint.timestamp}.json"
        # Sanitize filename
        filename = filename.replace(":", "-").replace(" ", "_")
        file_path = self.checkpoints_dir / filename
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            print(f"🤖 [Checkpoint] {checkpoint.stage} ({checkpoint.event_type}): {len(checkpoint.anomalies_detected)} anomalies detected.")
        except Exception as e:
            print(f"⚠️ [Checkpoint] Failed to emit checkpoint {file_path.name}: {e}")

    def load_checkpoints(self, stage: str | None = None) -> list[AgentCheckpoint]:
        checkpoints = []
        if not self.checkpoints_dir.exists():
            return checkpoints
            
        for path in self.checkpoints_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cp = AgentCheckpoint.from_dict(data)
                    if stage is None or cp.stage == stage:
                        checkpoints.append(cp)
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint {path}: {e}")
        return sorted(checkpoints, key=lambda c: c.timestamp)
