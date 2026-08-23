from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from agents.base import AgentBase

if TYPE_CHECKING:
    from pipeline.config import PipelineConfig


# Maps canonical anomaly types to the remediation action that fixes them.
ANOMALY_TO_ACTION = {
    "nan_loss_detected": "reduce_learning_rate",
    "gradient_explosion": "enable_gradient_clipping",
    "memory_pressure": "reduce_batch_size",
    "worker_shortage": "restart_workers",
    "loss_plateau": "reduce_learning_rate",
}


class RemediatorAgent(AgentBase):
    """Executes approved remediation actions that mutate the live PipelineConfig.

    Every action is audit-logged so the full chain of automated decisions
    can be inspected after the run.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_log: list[dict[str, Any]] = []
        self.approved_actions: set[str] = {
            'retry_stage',
            'reduce_batch_size',
            'reduce_learning_rate',
            'enable_gradient_clipping',
            'restart_workers',
            'skip_and_continue',
        }

    def execute_action(self, action: str, config: PipelineConfig) -> dict[str, Any]:
        """Execute a remediation action by mutating the live PipelineConfig.

        Args:
            action: The action name (must be in self.approved_actions).
            config: The live PipelineConfig object — mutations here
                    propagate to the next retry attempt.

        Returns:
            A dict with action, status, details, and config_diff.
        """
        result: dict[str, Any] = {'action': action, 'status': 'failed', 'details': '', 'config_diff': {}}

        if action not in self.approved_actions:
            result['details'] = f"Action '{action}' not in approved actions."
            self._log(result['details'], level='error')
            self._record(action, result)
            return result

        self._log(f"Executing action: {action}", level='info')

        if action == 'reduce_learning_rate':
            old_lr = config.lr
            config.lr = old_lr * 0.1
            result['status'] = 'success'
            result['details'] = f'Reduced LR from {old_lr} to {config.lr}'
            result['config_diff'] = {'lr': {'old': old_lr, 'new': config.lr}}

        elif action == 'reduce_batch_size':
            old_bs = config.batch_size
            config.batch_size = max(1, old_bs // 2)
            result['status'] = 'success'
            result['details'] = f'Reduced batch size from {old_bs} to {config.batch_size}'
            result['config_diff'] = {'batch_size': {'old': old_bs, 'new': config.batch_size}}

        elif action == 'enable_gradient_clipping':
            config.max_grad_norm = 1.0
            result['status'] = 'success'
            result['details'] = 'Enabled gradient clipping with max_norm=1.0'
            result['config_diff'] = {'max_grad_norm': {'old': None, 'new': 1.0}}

        elif action == 'restart_workers':
            result['status'] = 'logged'
            result['details'] = 'Restart request logged (no local restart).'

        elif action == 'retry_stage':
            result['status'] = 'retry_requested'
            result['details'] = 'Stage will be retried with current config.'

        elif action == 'skip_and_continue':
            result['status'] = 'skipped'
            result['details'] = 'Skipped current stage.'

        log_level = 'success' if result['status'] in ('success', 'retry_requested', 'logged', 'skipped') else 'error'
        self._log(f"Action '{action}' → {result['status']}: {result['details']}", level=log_level)
        self._record(action, result)
        return result

    def _record(self, action: str, result: dict[str, Any]) -> None:
        """Append to audit log."""
        self.action_log.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,
            'status': result['status'],
            'details': result['details'],
            'config_diff': result.get('config_diff', {}),
        })

    def save_audit_log(self) -> None:
        if not hasattr(self.tracker, 'run_dir'):
            return
        path = self.tracker.run_dir / "action_log.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.action_log, f, indent=2)
            self._log("Audit log saved successfully.", level='success')
        except Exception as e:
            self._log(f"Failed to save audit log: {e}", level='error')

    def observe(self, event: dict[str, Any]) -> list[str]:
        return []

    def diagnose(self, anomalies: list[str], context: dict[str, Any]) -> str:
        return ""

    def recommend_action(self, diagnosis: str, context: dict[str, Any]) -> str | None:
        """Map diagnosis to the best remediation action."""
        # Check for canonical anomaly types in the diagnosis string
        for anomaly_type, action in ANOMALY_TO_ACTION.items():
            if anomaly_type in diagnosis:
                return action
        return 'retry_stage'
