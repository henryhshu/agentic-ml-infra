from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from agents.base import AgentBase

class RemediatorAgent(AgentBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_log: list[dict[str, Any]] = []
        self.approved_actions: set[str] = {
            'retry_stage', 
            'reduce_batch_size', 
            'adjust_learning_rate', 
            'enable_gradient_clipping'
        }

    def execute_action(self, action: str, context: dict[str, Any], approved_by: str = 'supervisor') -> dict[str, Any]:
        result = {'action': action, 'status': 'failed', 'details': ''}
        
        if action not in self.approved_actions and action not in ('restart_workers', 'skip_and_continue'):
            result['details'] = f"Action {action} not in approved actions."
            self._log(result['details'], level='error')
            return result
            
        self._log(f"Executing action: {action} (approved by: {approved_by})", level='info')
        
        if action == 'retry_stage':
            result['status'] = 'retry_requested'
            result['details'] = 'Stage will be retried'
        elif action == 'reduce_batch_size':
            if 'batch_size' in context:
                context['batch_size'] = max(1, context['batch_size'] // 2)
            result['status'] = 'success'
            result['details'] = 'Reduced batch size.'
        elif action == 'adjust_learning_rate':
            if 'lr' in context:
                context['lr'] *= 0.1
            result['status'] = 'success'
            result['details'] = 'Adjusted learning rate.'
        elif action == 'enable_gradient_clipping':
            context['max_grad_norm'] = 1.0
            result['status'] = 'success'
            result['details'] = 'Enabled gradient clipping.'
        elif action == 'restart_workers':
            result['status'] = 'logged'
            result['details'] = 'Restart request logged (no local restart).'
            self._log("Restarting workers...", level='info')
        elif action == 'skip_and_continue':
            result['status'] = 'skipped'
            result['details'] = 'Skipped current stage.'
            
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'approved_by': approved_by,
            'status': result['status'],
            'details': result['details']
        }
        self.action_log.append(log_entry)
        
        log_level = 'success' if result['status'] in ('success', 'retry_requested', 'logged', 'skipped') else 'error'
        self._log(f"Action {action} completed with status: {result['status']}", level=log_level)
        return result

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
        return None
