from __future__ import annotations

import math
from typing import Any
from agents.base import AgentBase

class TrainingWatchAgent(AgentBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

    def observe(self, event: dict[str, Any]) -> list[str]:
        anomalies = []
        
        metrics = event.get('metrics', {})
        loss = metrics.get('loss')
        accuracy = metrics.get('accuracy')
        grad_norm = metrics.get('grad_norm')
        
        if loss is not None:
            if math.isnan(loss) or math.isinf(loss):
                anomalies.append('nan_loss_detected')
            self.loss_history.append(loss)
            
            if len(self.loss_history) >= 5:
                recent_losses = self.loss_history[-5:]
                if max(recent_losses) - min(recent_losses) < 1e-4:
                    anomalies.append('loss_plateau')
                    
        if accuracy is not None:
            if self.accuracy_history:
                best_acc = max(self.accuracy_history)
                if best_acc - accuracy > 0.05:
                    anomalies.append('accuracy_regression')
            self.accuracy_history.append(accuracy)
            
        if grad_norm is not None and grad_norm > 100:
            anomalies.append('gradient_explosion')
            
        return anomalies

    def diagnose(self, anomalies: list[str], context: dict[str, Any]) -> str:
        if not anomalies:
            return ""
            
        diagnosis_parts = []
        for anomaly in anomalies:
            if anomaly == 'nan_loss_detected':
                diagnosis_parts.append("Loss is NaN or Inf, indicating divergence.")
            elif anomaly == 'loss_plateau':
                diagnosis_parts.append("Loss has plateaued over the last 5 epochs.")
            elif anomaly == 'accuracy_regression':
                diagnosis_parts.append("Accuracy dropped by more than 5% from best.")
            elif anomaly == 'gradient_explosion':
                diagnosis_parts.append("Gradient norm exceeded 100.")
                
        if self.is_llm_available:
            msg = self._call_llm(
                "Diagnose training anomalies", 
                f"Anomalies: {anomalies}. History: {self.loss_history[-3:]}"
            )
            diagnosis_parts.append(f"LLM Insight: {msg}")
            
        return " | ".join(diagnosis_parts)

    def recommend_action(self, diagnosis: str, context: dict[str, Any]) -> str | None:
        if 'divergence' in diagnosis or 'NaN' in diagnosis:
            return 'retry_with_lower_lr'
        if 'plateau' in diagnosis:
            return 'adjust_learning_rate'
        if 'regression' in diagnosis:
            return 'investigate_data_pipeline'
        if 'Gradient norm' in diagnosis:
            return 'enable_gradient_clipping'
        return None
