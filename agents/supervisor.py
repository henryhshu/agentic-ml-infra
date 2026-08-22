from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING
from agents.base import AgentBase

if TYPE_CHECKING:
    from metrics.experiment import ExperimentTracker
    from pipeline.config import PipelineConfig
    from agents.checkpoints import CheckpointEmitter

class SupervisorAgent(AgentBase):
    SYSTEM_PROMPT = (
        "You are the Training Supervisor Agent. Monitor training metrics, cluster health, "
        "and data quality to ensure successful model training. Determine when to automatically "
        "remediate issues or when to escalate them."
    )
    
    def __init__(
        self, 
        name: str, 
        tracker: ExperimentTracker, 
        config: PipelineConfig, 
        checkpoint_emitter: CheckpointEmitter | None = None,
        sub_agents: dict[str, AgentBase] | None = None
    ) -> None:
        super().__init__(name, tracker, config, checkpoint_emitter)
        self.incidents: list[dict[str, Any]] = []
        self.sub_agents: dict[str, AgentBase] = sub_agents or {}

    def register_agent(self, name: str, agent: AgentBase) -> None:
        self.sub_agents[name] = agent

    def route_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        event_type = event.get('type', '')
        agent = None
        if event_type == 'training_metrics':
            agent = self.sub_agents.get('training_watch')
        elif event_type == 'cluster_event':
            agent = self.sub_agents.get('cluster_health')
        elif event_type == 'data_event':
            agent = self.sub_agents.get('data_quality')
            
        if not agent:
            return None
            
        anomalies = agent.observe(event)
        if anomalies:
            self._log(f"Anomalies detected by {agent.name}: {anomalies}", level='warning')
            return {
                'agent': agent.name,
                'anomalies': anomalies
            }
        return None

    def handle_anomaly(self, stage: str, anomalies: list[str], context: dict[str, Any]) -> str:
        self._log(f"Handling anomalies for stage {stage}: {anomalies}", level='warning')
        
        diagnoses = []
        for agent_name, agent in self.sub_agents.items():
            if agent_name == 'remediator':
                continue
            diagnosis = agent.diagnose(anomalies, context)
            if diagnosis:
                diagnoses.append(f"{agent_name}: {diagnosis}")
                
        overall_diagnosis = " | ".join(diagnoses) or "Unknown issue"
        
        remediator = self.sub_agents.get('remediator')
        action_taken = None
        
        if self.should_auto_remediate(overall_diagnosis, anomalies) and remediator:
            action = remediator.recommend_action(overall_diagnosis, context)
            if action:
                result = getattr(remediator, 'execute_action')(action, context, approved_by=self.name)
                action_taken = result.get('action')
        
        incident = {
            'stage': stage,
            'anomalies': anomalies,
            'diagnosis': overall_diagnosis,
            'action_taken': action_taken,
            'context': context
        }
        self.incidents.append(incident)
        return overall_diagnosis

    def should_auto_remediate(self, diagnosis: str, anomalies: list[str]) -> bool:
        # returns True for known-safe remediations (retry, reduce batch size)
        safe_remediations = {'nan_loss_detected', 'worker_shortage', 'memory_pressure', 'gradient_explosion'}
        return any(anom in safe_remediations for anom in anomalies)

    def save_incidents(self) -> None:
        if not hasattr(self.tracker, 'run_dir'):
            return
        path = self.tracker.run_dir / "incidents.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.incidents, f, indent=2)
            self._log(f"Saved {len(self.incidents)} incidents.", level="success")
        except Exception as e:
            self._log(f"Failed to save incidents: {e}", level="error")

    def observe(self, event: dict[str, Any]) -> list[str]:
        return []
        
    def diagnose(self, anomalies: list[str], context: dict[str, Any]) -> str:
        return ""
        
    def recommend_action(self, diagnosis: str, context: dict[str, Any]) -> str | None:
        return None
