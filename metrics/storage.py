from __future__ import annotations
import os
import json
from pathlib import Path

class MetricsStore:
    def __init__(self, base_dir: str | Path = 'experiments'):
        self.base_dir = Path(base_dir)

    def list_experiments(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        
        dirs = [d.name for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        return sorted(dirs)

    def load_experiment(self, run_id: str) -> dict:
        summary_path = self.base_dir / run_id / "experiment_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                return json.load(f)
        return {}

    def load_stage(self, run_id: str, stage_name: str) -> dict:
        stage_path = self.base_dir / run_id / f"{stage_name}_metrics.json"
        if stage_path.exists():
            with open(stage_path, 'r') as f:
                return json.load(f)
        return {}

    def compare_experiments(self, run_ids: list[str], metric_key: str = 'final_test_accuracy') -> list[dict]:
        results = []
        for run_id in run_ids:
            exp = self.load_experiment(run_id)
            if not exp:
                continue
            
            stages = exp.get("stages", {})
            val = None
            
            for stage_name, stage_data in stages.items():
                if metric_key in stage_data.get("metrics", {}):
                    val = stage_data["metrics"][metric_key]
                elif metric_key in stage_data:
                    val = stage_data[metric_key]

            res = {
                "run_id": run_id,
                metric_key: val,
                "duration_s": exp.get("duration_s")
            }
            results.append(res)
        return results

    def get_latest(self) -> dict | None:
        latest_link = self.base_dir / "latest"
        if latest_link.exists() and latest_link.is_symlink():
            run_id = latest_link.readlink().name
            return self.load_experiment(run_id)
        return None
