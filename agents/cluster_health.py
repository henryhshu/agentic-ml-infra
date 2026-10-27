"""Cluster health agent module for AutoPARK.

Monitors distributed Ray and Kubernetes infrastructure state, detecting
worker node shortages, AWS Spot instance preemptions, cluster memory pressure,
and unexpected distributed task failures.
"""
from __future__ import annotations

from typing import Any
from agents.base import AgentBase


class ClusterHealthAgent(AgentBase):
    """AutoPARK Distributed Cluster Health Agent.

    Monitors distributed runtime infrastructure health across KubeRay and AWS EKS.
    Flags worker node shortages, transient EC2 Spot interruptions, cluster memory
    saturation (>90%), and task execution interruptions.
    """

    def observe(self, event: dict[str, Any]) -> list[str]:
        anomalies = []
        cluster_info = self._check_cluster_status()

        worker_count = event.get('worker_count', cluster_info.get('worker_count', 0))
        expected_workers = event.get('expected_workers', cluster_info.get('expected_workers', 1))
        
        if worker_count < expected_workers:
            anomalies.append('worker_shortage')
            
        if event.get('task_failed'):
            anomalies.append('task_failure')
            
        memory_usage = event.get('memory_usage', cluster_info.get('memory_usage', 0))
        if memory_usage > 0.9:
            anomalies.append('memory_pressure')
            
        return anomalies

    def diagnose(self, anomalies: list[str], context: dict[str, Any]) -> str:
        """Diagnose cluster health issues directly using Claude Platform on AWS."""
        if not anomalies:
            return ""

        system_prompt = (
            "You are an ML Cluster Infrastructure Agent running on Claude Platform on AWS. "
            "Analyze cluster telemetry and provide a concise, one-sentence diagnosis of the operational failure."
        )
        user_prompt = f"Anomalies detected: {anomalies}. Cluster context: {context}."
        return self._call_llm(system_prompt, user_prompt, max_tokens=150)

    def recommend_action(self, diagnosis: str, context: dict[str, Any]) -> str | None:
        if 'worker_shortage' in diagnosis:
            return 'restart_workers'
        if 'task_failure' in diagnosis:
            return 'retry_stage'
        if 'memory_pressure' in diagnosis:
            return 'reduce_batch_size'
        return None

    def _check_cluster_status(self) -> dict[str, Any]:
        """Measured ranks (DDP heartbeats) and host memory utilisation."""
        info: dict[str, Any] = {'worker_count': 1, 'expected_workers': 1, 'memory_usage': 0.0}
        try:
            import psutil
            info['memory_usage'] = psutil.virtual_memory().percent / 100.0
        except Exception:
            pass
        from pipeline.distributed import get_context
        ctx = get_context()
        if ctx is not None:
            info['worker_count'] = 1 + ctx.live_worker_count()
            info['expected_workers'] = ctx.world_size
        return info
