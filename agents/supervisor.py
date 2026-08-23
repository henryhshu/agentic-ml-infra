from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING
from agents.base import AgentBase

if TYPE_CHECKING:
    from metrics.experiment import ExperimentTracker
    from pipeline.config import PipelineConfig
    from agents.checkpoints import CheckpointEmitter

# Canonical anomaly types used across the agent fleet.
ANOMALY_TYPES = {
    "nan_loss_detected",
    "loss_plateau",
    "accuracy_regression",
    "gradient_explosion",
    "worker_shortage",
    "task_failure",
    "memory_pressure",
    "schema_error",
    "corrupt_data",
    "label_imbalance",
    "missing_columns",
}

# Anomalies that are safe for automatic remediation without human escalation.
SAFE_REMEDIATIONS = {
    "nan_loss_detected",
    "gradient_explosion",
    "worker_shortage",
    "memory_pressure",
    "loss_plateau",
}


def normalize_anomalies(raw_anomalies: list[str]) -> list[str]:
    """Map raw exception strings or errors to canonical anomaly type names."""
    keyword_map = {
        "nan": "nan_loss_detected",
        "inf": "nan_loss_detected",
        "diverge": "nan_loss_detected",
        "plateau": "loss_plateau",
        "regression": "accuracy_regression",
        "gradient": "gradient_explosion",
        "grad_norm": "gradient_explosion",
        "worker": "worker_shortage",
        "memory": "memory_pressure",
        "oom": "memory_pressure",
        "schema": "schema_error",
        "corrupt": "corrupt_data",
    }
    normalized: list[str] = []
    for raw in raw_anomalies:
        if raw in ANOMALY_TYPES:
            normalized.append(raw)
            continue
        lower = raw.lower()
        matched = False
        for keyword, anomaly_type in keyword_map.items():
            if keyword in lower:
                if anomaly_type not in normalized:
                    normalized.append(anomaly_type)
                matched = True
                break
        if not matched:
            normalized.append(raw)
    return normalized


class SupervisorAgent(AgentBase):
    """Orchestrates the agent fleet, routes pipeline telemetry, and synthesizes in-the-loop decisions."""

    SYSTEM_PROMPT = (
        "You are the Lead ML Training Supervisor Agent running in-the-loop on AWS. "
        "Your mission is to monitor training metrics, data partitions, and cluster health. "
        "Coordinate autonomous recovery actions when anomalies arise and provide concise diagnostic assessments."
    )

    def __init__(
        self,
        name: str,
        tracker: ExperimentTracker,
        config: PipelineConfig,
        checkpoint_emitter: CheckpointEmitter | None = None,
        sub_agents: dict[str, AgentBase] | None = None,
    ) -> None:
        super().__init__(name, tracker, config, checkpoint_emitter)
        self.incidents: list[dict[str, Any]] = []
        self.sub_agents: dict[str, AgentBase] = sub_agents or {}

    def register_agent(self, name: str, agent: AgentBase) -> None:
        self.sub_agents[name] = agent

    def route_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Route an event to the appropriate sub-agent and return any anomalies found."""
        event_type = event.get("type", "")

        # Route training metrics
        if event_type in ("training_metrics", "epoch_complete"):
            tw = self.sub_agents.get("training_watch")
            if tw:
                anomalies = tw.observe(event)
                # In-the-loop Claude epoch analysis
                if hasattr(tw, "analyze_epoch") and "metrics" in event:
                    tw.analyze_epoch(event["metrics"])
                if anomalies:
                    self._log(f"Anomalies detected by {tw.name}: {anomalies}", level="warning")
                    return {"agent": tw.name, "anomalies": anomalies}
            return None

        # Route data events
        elif event_type in ("data_event", "data_quality"):
            dq = self.sub_agents.get("data_quality")
            if dq:
                anomalies = dq.observe(event)
                if anomalies:
                    self._log(f"Anomalies detected by {dq.name}: {anomalies}", level="warning")
                    return {"agent": dq.name, "anomalies": anomalies}
            return None

        # Route cluster events
        elif event_type in ("cluster_event", "cluster_health"):
            ch = self.sub_agents.get("cluster_health")
            if ch:
                anomalies = ch.observe(event)
                if anomalies:
                    self._log(f"Anomalies detected by {ch.name}: {anomalies}", level="warning")
                    return {"agent": ch.name, "anomalies": anomalies}
            return None

        # Route stage completion
        elif event_type == "stage_complete":
            stage_name = event.get("stage", "unknown")
            attempt = event.get("attempt", 0)
            self._log(f"Stage '{stage_name}' successfully finished (attempt {attempt + 1})", level="success")
            ch = self.sub_agents.get("cluster_health")
            if ch:
                cluster_anomalies = ch.observe(event)
                if cluster_anomalies:
                    return {"agent": ch.name, "anomalies": cluster_anomalies}
            return None

        return None

    def handle_anomaly(
        self,
        stage: str,
        anomalies: list[str],
        context: dict[str, Any],
    ) -> str:
        """Coordinate diagnosis across agents and decide on remediation."""
        canonical = normalize_anomalies(anomalies)
        self._log(f"Handling anomalies for stage '{stage}': {canonical}", level="warning")

        # Gather diagnoses from relevant subagents
        diagnoses: list[str] = []
        for agent_name, agent in self.sub_agents.items():
            if agent_name == "remediator":
                continue
            diagnosis = agent.diagnose(canonical, context)
            if diagnosis:
                diagnoses.append(f"[{agent_name}] {diagnosis}")

        overall_diagnosis = " | ".join(diagnoses) or "Unclassified pipeline failure."

        # Remediate or Escalate
        remediator = self.sub_agents.get("remediator")
        action_taken = None

        if self.should_auto_remediate(overall_diagnosis, canonical) and remediator:
            action = remediator.recommend_action(overall_diagnosis, context)
            if action:
                self._log(f"Auto-remediating: executing '{action}'", level="info")
                result = remediator.execute_action(action, self.config)
                action_taken = result.get("action")
        else:
            self._log(f"Escalating anomaly to human/operator — manual intervention advised for: {canonical}", level="warning")

        incident = {
            "stage": stage,
            "anomalies": canonical,
            "raw_anomalies": anomalies,
            "diagnosis": overall_diagnosis,
            "auto_remediated": action_taken is not None,
            "action_taken": action_taken,
        }
        self.incidents.append(incident)
        return overall_diagnosis

    def should_auto_remediate(self, diagnosis: str, anomalies: list[str]) -> bool:
        """Check whether all anomalies fall under safe auto-remediation policies."""
        return any(anom in SAFE_REMEDIATIONS for anom in anomalies)

    def save_incidents(self) -> None:
        """Save structured incident audit trail to JSON."""
        if not hasattr(self.tracker, "run_dir"):
            return
        path = self.tracker.run_dir / "incidents.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.incidents, f, indent=2)
            self._log(f"Saved {len(self.incidents)} incidents to {path.name}", level="success")
        except Exception as e:
            self._log(f"Failed to save incidents: {e}", level="error")

    def observe(self, event: dict[str, Any]) -> list[str]:
        return []

    def diagnose(self, anomalies: list[str], context: dict[str, Any]) -> str:
        return ""

    def recommend_action(self, diagnosis: str, context: dict[str, Any]) -> str | None:
        return None
