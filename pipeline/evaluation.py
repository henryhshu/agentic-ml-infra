"""Evaluation stage — final model assessment with comprehensive metrics."""
from __future__ import annotations

import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics.experiment import ExperimentTracker
from metrics.schemas import EvaluationMetrics, now_iso


class EvaluationStage:
    """Pipeline stage that runs final model evaluation with detailed metrics."""

    STAGE_NAME = "evaluation"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    def __init__(
        self,
        config,  # PipelineConfig
        tracker: ExperimentTracker,
        model: nn.Module,
        test_loader: DataLoader,
        checkpoint_emitter=None,
        supervisor=None,
    ):
        self.config = config
        self.tracker = tracker
        self.model = model
        self.test_loader = test_loader
        self.checkpoint_emitter = checkpoint_emitter
        self.supervisor = supervisor

    @property
    def name(self) -> str:
        return self.STAGE_NAME

    def run(self) -> EvaluationMetrics:
        """Run evaluation and return metrics."""
        metrics = self.tracker.start_stage(self.STAGE_NAME, metrics_cls=EvaluationMetrics)
        self._emit_checkpoint("stage_start", {"stage": self.STAGE_NAME})

        try:
            result = self.execute()
            metrics.test_accuracy = result["test_accuracy"]
            metrics.test_loss = result["test_loss"]
            metrics.per_class_accuracy = result["per_class_accuracy"]
            metrics.inference_throughput_sps = result["inference_throughput_sps"]
            metrics.confusion_matrix = result["confusion_matrix"]
            metrics.metrics.update(result)
            self.tracker.end_stage(self.STAGE_NAME, status="success")
            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("stage_end", {
                "test_accuracy": result["test_accuracy"],
                "throughput": result["inference_throughput_sps"],
            })

            if self.supervisor is not None:
                self.supervisor.route_event({
                    "type": "evaluation_metrics",
                    "stage": self.STAGE_NAME,
                    "metrics": {
                        "test_accuracy": result["test_accuracy"],
                        "test_loss": result["test_loss"],
                        "inference_throughput_sps": result["inference_throughput_sps"],
                    },
                })

            print(
                f"  ✅ Evaluation: accuracy={result['test_accuracy']:.4f}, "
                f"throughput={result['inference_throughput_sps']:.0f} samples/sec"
            )
            return metrics

        except Exception as exc:
            self.tracker.log_event(
                self.STAGE_NAME, "error", f"Evaluation failed: {exc}"
            )
            self.tracker.end_stage(self.STAGE_NAME, status="failed")
            metrics.error_message = str(exc)
            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("anomaly_detected", {"error": str(exc)})
            raise

    def execute(self) -> dict:
        """Run comprehensive evaluation. Returns detailed metrics dict."""
        device = torch.device(self.config.device)
        criterion = nn.CrossEntropyLoss()

        self.tracker.log_event(self.STAGE_NAME, "info", "Starting final evaluation...")

        # Initialize confusion matrix
        confusion = [[0] * self.NUM_CLASSES for _ in range(self.NUM_CLASSES)]
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(device, non_blocking=True)
                labels = torch.as_tensor(labels, device=device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Update confusion matrix and per-class counts
                for pred, true in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                    confusion[true][pred] += 1
                    per_class_total[true] += 1
                    if pred == true:
                        per_class_correct[true] += 1

        eval_time = time.time() - t0
        test_accuracy = correct / total if total > 0 else 0.0
        test_loss = total_loss / total if total > 0 else 0.0
        throughput = total / eval_time if eval_time > 0 else 0.0

        # Per-class accuracy
        per_class_accuracy = {}
        for class_idx in range(self.NUM_CLASSES):
            class_name = self.CLASS_NAMES[class_idx]
            class_total = per_class_total.get(class_idx, 0)
            class_correct = per_class_correct.get(class_idx, 0)
            per_class_accuracy[class_name] = (
                round(class_correct / class_total, 4) if class_total > 0 else 0.0
            )

        # Log per-class results
        self.tracker.log_event(
            self.STAGE_NAME,
            "per_class_results",
            "Per-class accuracy breakdown",
            data=per_class_accuracy,
        )

        # Print per-class breakdown
        print("  ── Per-class accuracy ──")
        for name, acc in sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(acc * 20)
            print(f"    {name:<12s}  {acc:.4f}  {bar}")

        self.tracker.log_metric(self.STAGE_NAME, "test_accuracy", round(test_accuracy, 6))
        self.tracker.log_metric(self.STAGE_NAME, "test_loss", round(test_loss, 6))
        self.tracker.log_metric(self.STAGE_NAME, "inference_throughput_sps", round(throughput, 1))
        self.tracker.log_metric(self.STAGE_NAME, "eval_time_s", round(eval_time, 2))

        return {
            "test_accuracy": round(test_accuracy, 6),
            "test_loss": round(test_loss, 6),
            "per_class_accuracy": per_class_accuracy,
            "inference_throughput_sps": round(throughput, 1),
            "confusion_matrix": confusion,
            "eval_time_s": round(eval_time, 2),
            "total_samples": total,
        }

    def _emit_checkpoint(self, event_type: str, data: dict | None = None) -> None:
        """Emit an agent checkpoint if emitter is available."""
        if self.checkpoint_emitter is not None:
            from agents.checkpoints import AgentCheckpoint
            checkpoint = AgentCheckpoint(
                stage=self.STAGE_NAME,
                event_type=event_type,
                timestamp=now_iso(),
                metrics_snapshot=data or {},
                anomalies_detected=[],
                agent_diagnosis=None,
                recommended_action=None,
                action_taken=None,
            )
            self.checkpoint_emitter.emit(checkpoint)
