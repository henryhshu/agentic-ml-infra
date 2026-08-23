"""Training stage — runs PyTorch training via Ray Train with metrics and anomaly detection."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics.experiment import ExperimentTracker
from metrics.schemas import TrainingMetrics, now_iso


class TrainingStage:
    """Pipeline stage that runs model training with per-epoch metrics and anomaly detection."""

    STAGE_NAME = "training"

    def __init__(
        self,
        config,  # PipelineConfig
        tracker: ExperimentTracker,
        train_loader: DataLoader,
        test_loader: DataLoader,
        train_set_size: int,
        checkpoint_emitter=None,
        supervisor=None,
    ):
        self.config = config
        self.tracker = tracker
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_set_size = train_set_size
        self.checkpoint_emitter = checkpoint_emitter
        self.supervisor = supervisor  # SupervisorAgent or None
        # Populated after run()
        self.model: nn.Module | None = None
        self.best_accuracy: float = 0.0
        self.best_epoch: int = 0

    @property
    def name(self) -> str:
        return self.STAGE_NAME

    def run(self) -> TrainingMetrics:
        """Run training and return metrics."""
        metrics = self.tracker.start_stage(self.STAGE_NAME, metrics_cls=TrainingMetrics)
        self._emit_checkpoint("stage_start", {
            "model": self.config.model,
            "epochs": self.config.epochs,
            "lr": self.config.lr,
            "batch_size": self.config.batch_size,
        })

        try:
            result = self.execute()
            metrics.epochs_completed = result["epochs_completed"]
            metrics.final_train_loss = result["final_train_loss"]
            metrics.final_test_accuracy = result["final_test_accuracy"]
            metrics.best_test_accuracy = result["best_test_accuracy"]
            metrics.best_epoch = result["best_epoch"]
            metrics.per_epoch = result["per_epoch"]
            metrics.metrics.update(result)
            self.tracker.end_stage(self.STAGE_NAME, status="success")
            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("stage_end", {
                "final_accuracy": result["final_test_accuracy"],
                "best_accuracy": result["best_test_accuracy"],
            })
            print(
                f"  ✅ Training: {result['epochs_completed']} epochs, "
                f"best_acc={result['best_test_accuracy']:.4f} (epoch {result['best_epoch']})"
            )
            return metrics

        except _TrainingAnomalyError as exc:
            self.tracker.log_event(
                self.STAGE_NAME, "anomaly", str(exc), data=exc.context
            )
            self.tracker.end_stage(self.STAGE_NAME, status="failed")
            metrics.error_message = str(exc)
            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("anomaly_detected", {
                "error": str(exc),
                "anomaly_type": exc.anomaly_type,
            })
            raise
        except Exception as exc:
            self.tracker.log_event(
                self.STAGE_NAME, "error", f"Training failed: {exc}"
            )
            self.tracker.end_stage(self.STAGE_NAME, status="failed")
            metrics.error_message = str(exc)
            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("anomaly_detected", {"error": str(exc)})
            raise

    def execute(self) -> dict:
        """Run the training loop. Returns comprehensive metrics dict."""
        from models.resnet import build_resnet18
        from models.tiny_cnn import build_tiny_cnn

        builders = {"resnet18": build_resnet18, "tiny": build_tiny_cnn}
        device = torch.device(self.config.device)

        # Build model
        model = builders[self.config.model](num_classes=10).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        self.tracker.log_event(
            self.STAGE_NAME,
            "info",
            f"Model: {self.config.model}, params={param_count:,}, device={device}",
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )

        # ── Fault injection: Simulated Memory Pressure / OOM ──
        if self.config.fault_injection in ("batch_oom", "memory_pressure") and self.config.batch_size >= 256:
            raise _TrainingAnomalyError(
                "memory_pressure",
                f"Simulated CUDA Out of Memory: failed to allocate memory for batch_size={self.config.batch_size}",
                {"batch_size": self.config.batch_size},
            )

        per_epoch: list[dict] = []
        best_acc = 0.0
        best_epoch = 0
        num_batches = len(self.train_loader)

        for epoch in range(self.config.epochs):
            epoch_num = epoch + 1
            t_epoch = time.time()

            # ── Training pass ──
            model.train()
            running_loss = 0.0
            total_grad_norm = 0.0
            grad_samples = 0

            for step, (images, labels) in enumerate(self.train_loader, start=1):
                images = images.to(device, non_blocking=True)
                labels = torch.as_tensor(labels, device=device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)

                # ── Fault injection: NaN / Extreme LR divergence ──
                if self.config.fault_injection in ("nan_divergence", "nan_loss") and self.config.lr >= 0.5:
                    raise _TrainingAnomalyError(
                        "nan_loss",
                        f"NaN/Inf loss detected at epoch {epoch_num}, step {step}: divergent loss with excessive lr={self.config.lr}",
                        {"epoch": epoch_num, "step": step, "lr": self.config.lr, "loss": float("nan")},
                    )

                # ── NaN/Inf detection (fast-fail) ──
                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    raise _TrainingAnomalyError(
                        "nan_loss",
                        f"NaN/Inf loss detected at epoch {epoch_num}, step {step}",
                        {"epoch": epoch_num, "step": step, "loss": loss.item()},
                    )

                loss.backward()

                # Track gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

                # ── Fault injection: Gradient explosion (healed by gradient clipping) ──
                if self.config.fault_injection == "gradient_explosion" and self.config.max_grad_norm is None and step == 1:
                    total_norm = 999.9
                    raise _TrainingAnomalyError(
                        "gradient_explosion",
                        f"Gradient explosion detected: grad_norm={total_norm:.1f} exceeds stability threshold",
                        {"epoch": epoch_num, "step": step, "grad_norm": total_norm},
                    )

                total_grad_norm += total_norm
                grad_samples += 1

                # Apply gradient clipping if enabled (set by remediator)
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_grad_norm
                    )

                optimizer.step()
                running_loss += loss.item() * images.size(0)

                if step % self.config.log_every == 0 or step == num_batches:
                    print(
                        f"    epoch {epoch_num:02d}  step {step:>4d}/{num_batches}  "
                        f"loss={loss.item():.4f}  grad_norm={total_norm:.2f}"
                    )

            scheduler.step()
            train_loss = running_loss / self.train_set_size
            avg_grad_norm = total_grad_norm / max(grad_samples, 1)

            # ── Evaluation pass ──
            model.eval()
            correct = total = 0
            eval_loss = 0.0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images = images.to(device, non_blocking=True)
                    labels = torch.as_tensor(labels, device=device)
                    outputs = model(images)
                    eval_loss += criterion(outputs, labels).item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            test_acc = correct / total if total > 0 else 0.0
            test_loss = eval_loss / total if total > 0 else 0.0
            epoch_time = time.time() - t_epoch
            current_lr = scheduler.get_last_lr()[0]

            # Track best
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch_num

            epoch_metrics = {
                "epoch": epoch_num,
                "train_loss": round(train_loss, 6),
                "test_loss": round(test_loss, 6),
                "test_accuracy": round(test_acc, 6),
                "learning_rate": current_lr,
                "avg_grad_norm": round(avg_grad_norm, 4),
                "epoch_time_s": round(epoch_time, 2),
            }
            per_epoch.append(epoch_metrics)

            # Log per-epoch metrics
            self.tracker.log_metric(
                self.STAGE_NAME, f"epoch_{epoch_num}_accuracy", test_acc
            )
            self.tracker.log_event(
                self.STAGE_NAME,
                "epoch_complete",
                f"epoch {epoch_num}/{self.config.epochs}  "
                f"loss={train_loss:.4f}  test_acc={test_acc:.4f}",
                data=epoch_metrics,
            )

            # Emit epoch boundary checkpoint
            self._emit_checkpoint("epoch_boundary", epoch_metrics)

            print(
                f"  epoch {epoch_num:02d}/{self.config.epochs}  "
                f"loss={train_loss:.4f}  test_acc={test_acc:.4f}  "
                f"lr={current_lr:.6f}  ({epoch_time:.1f}s)"
            )

            # ── Route epoch metrics to agent fleet (if agentic) ──
            if self.supervisor is not None:
                agent_result = self.supervisor.route_event({
                    "type": "training_metrics",
                    "stage": self.STAGE_NAME,
                    "metrics": {
                        "loss": train_loss,
                        "accuracy": test_acc,
                        "grad_norm": avg_grad_norm,
                        "epoch": epoch_num,
                        "learning_rate": current_lr,
                    },
                })
                if agent_result and agent_result.get("anomalies"):
                    self._log_agent_anomalies(agent_result["anomalies"], epoch_num)

        self.model = model
        self.best_accuracy = best_acc
        self.best_epoch = best_epoch

        return {
            "epochs_completed": self.config.epochs,
            "final_train_loss": per_epoch[-1]["train_loss"] if per_epoch else None,
            "final_test_accuracy": per_epoch[-1]["test_accuracy"] if per_epoch else None,
            "best_test_accuracy": round(best_acc, 6),
            "best_epoch": best_epoch,
            "per_epoch": per_epoch,
            "param_count": param_count,
            "total_training_time_s": sum(e["epoch_time_s"] for e in per_epoch),
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

    def _log_agent_anomalies(self, anomalies: list[str], epoch: int) -> None:
        """Log anomalies detected by the agent fleet."""
        for anomaly in anomalies:
            self.tracker.log_event(
                self.STAGE_NAME, "agent_anomaly",
                f"Agent detected: {anomaly} at epoch {epoch}",
            )
            self._emit_checkpoint("anomaly_detected", {
                "anomaly_type": anomaly,
                "epoch": epoch,
                "source": "agent_fleet",
            })
            print(f"  ⚠️  Agent anomaly: {anomaly} (epoch {epoch})")


class _TrainingAnomalyError(Exception):
    """Raised when a training anomaly is detected that requires intervention."""

    def __init__(self, anomaly_type: str, message: str, context: dict | None = None):
        super().__init__(message)
        self.anomaly_type = anomaly_type
        self.context = context or {}
