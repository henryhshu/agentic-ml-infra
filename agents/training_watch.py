from __future__ import annotations

import math
from typing import Any
from agents.base import AgentBase


class TrainingWatchAgent(AgentBase):
    """Monitors live training metrics (loss, accuracy, gradients) and detects anomalies in real-time."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []
        self.grad_norm_history: list[float] = []

    def observe(self, event: dict[str, Any]) -> list[str]:
        """Inspect streaming training metrics from an epoch and identify anomalies."""
        anomalies: list[str] = []

        metrics = event.get("metrics", {})
        loss = metrics.get("loss")
        accuracy = metrics.get("accuracy")
        grad_norm = metrics.get("grad_norm")

        if loss is not None:
            if math.isnan(loss) or math.isinf(loss):
                anomalies.append("nan_loss_detected")
            self.loss_history.append(loss)

            if len(self.loss_history) >= 5:
                recent_losses = self.loss_history[-5:]
                if max(recent_losses) - min(recent_losses) < 1e-4:
                    anomalies.append("loss_plateau")

        if accuracy is not None:
            if self.accuracy_history:
                best_acc = max(self.accuracy_history)
                if best_acc - accuracy > 0.05:
                    anomalies.append("accuracy_regression")
            self.accuracy_history.append(accuracy)

        if grad_norm is not None:
            self.grad_norm_history.append(grad_norm)
            if grad_norm > 100.0:
                anomalies.append("gradient_explosion")

        return anomalies

    def analyze_epoch(self, metrics: dict[str, Any]) -> str | None:
        """Provide real-time Claude in-the-loop analysis for an epoch during training."""
        if not self.is_llm_available:
            return None

        epoch = metrics.get("epoch", len(self.loss_history))
        train_loss = metrics.get("loss", metrics.get("train_loss"))
        test_acc = metrics.get("accuracy", metrics.get("test_accuracy"))
        lr = metrics.get("learning_rate")
        grad_norm = metrics.get("grad_norm", metrics.get("avg_grad_norm"))

        system_prompt = (
            "You are a real-time ML Training Diagnostic Agent watching an active training loop on AWS. "
            "Give a concise, single-sentence analytical feedback on the convergence, loss progress, "
            "and learning rate behavior."
        )
        user_prompt = (
            f"Epoch {epoch}: train_loss={train_loss:.4f}, test_accuracy={test_acc:.4f}, "
            f"lr={lr}, avg_grad_norm={grad_norm:.2f}. "
            f"Model: {self.config.model}, Batch size: {self.config.batch_size}."
        )

        return self._call_llm(system_prompt, user_prompt, max_tokens=150)

    def diagnose(self, anomalies: list[str], context: dict[str, Any]) -> str:
        """Diagnose root cause using Claude with recent training history context."""
        if not anomalies:
            return ""

        rule_diagnoses = []
        for anomaly in anomalies:
            if anomaly == "nan_loss_detected":
                rule_diagnoses.append("Loss diverged to NaN/Inf due to unstable gradients or excessive learning rate.")
            elif anomaly == "loss_plateau":
                rule_diagnoses.append("Loss plateaued across recent epochs, indicating stagnant optimization.")
            elif anomaly == "accuracy_regression":
                rule_diagnoses.append("Test accuracy dropped significantly from peak performance.")
            elif anomaly == "gradient_explosion":
                rule_diagnoses.append("Gradient norm exceeded numerical stability threshold.")

        rule_summary = " | ".join(rule_diagnoses)

        if self.is_llm_available:
            system_prompt = (
                "You are an expert ML Training Diagnostic Agent on AWS. Analyze the observed training anomalies "
                "and recent metric trajectory. In 1-2 concise sentences, provide the probable root cause "
                "and recommended parameter adjustment."
            )
            history_summary = {
                "recent_losses": self.loss_history[-5:],
                "recent_accuracies": self.accuracy_history[-5:],
                "recent_grad_norms": self.grad_norm_history[-5:],
                "current_lr": self.config.lr,
                "batch_size": self.config.batch_size,
            }
            user_prompt = (
                f"Anomalies detected: {anomalies}.\n"
                f"Context: {context}.\n"
                f"Training telemetry: {history_summary}."
            )
            llm_insight = self._call_llm(system_prompt, user_prompt, max_tokens=200)
            return f"{rule_summary} | Claude Diagnosis: {llm_insight}"

        return rule_summary

    def recommend_action(self, diagnosis: str, context: dict[str, Any]) -> str | None:
        """Map diagnosis to a safe automated remediation action."""
        if "nan" in diagnosis.lower() or "diverge" in diagnosis.lower():
            return "reduce_learning_rate"
        if "plateau" in diagnosis.lower():
            return "reduce_learning_rate"
        if "gradient" in diagnosis.lower() or "explosion" in diagnosis.lower():
            return "enable_gradient_clipping"
        if "regression" in diagnosis.lower():
            return "retry_stage"
        return None
