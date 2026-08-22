from __future__ import annotations

from typing import Any
from agents.base import AgentBase

class ClusterHealthAgent(AgentBase):
    def observe(self, event: dict[str, Any]) -> list[str]:
        anomalies = []
        cluster_info = self._check_ray_status()
        
        worker_count = event.get('worker_count', cluster_info.get('worker_count', 0))
        expected_workers = event.get('expected_workers', 4)
        
        if worker_count < expected_workers:
            anomalies.append('worker_shortage')
            
        if event.get('task_failed'):
            anomalies.append('task_failure')
            
        memory_usage = event.get('memory_usage', cluster_info.get('memory_usage', 0))
        if memory_usage > 0.9:
            anomalies.append('memory_pressure')
            
        return anomalies

    def diagnose(self, anomalies: list[str], context: dict[str, Any]) -> str:
        if not anomalies:
            return ""
        
        diag = "Cluster issues detected: " + ", ".join(anomalies)
        if self.is_llm_available:
            llm_diag = self._call_llm("Diagnose cluster health", f"Anomalies: {anomalies}")
            diag += f" | {llm_diag}"
            
        return diag

    def recommend_action(self, diagnosis: str, context: dict[str, Any]) -> str | None:
        if 'worker_shortage' in diagnosis:
            return 'restart_workers'
        if 'task_failure' in diagnosis:
            return 'retry_stage'
        if 'memory_pressure' in diagnosis:
            return 'reduce_batch_size'
        return None

    def _check_ray_status(self) -> dict[str, Any]:
        try:
            import ray # type: ignore
            if ray.is_initialized():
                return {
                    'worker_count': len(ray.nodes()),
                    'memory_usage': 0.5 # Mock value since real util takes more work
                }
        except ImportError:
            self._log("Ray not available to check status.", level='warning')
        except Exception as e:
            self._log(f"Error checking ray status: {e}", level='error')
        
        return {'worker_count': 1, 'memory_usage': 0.0}
