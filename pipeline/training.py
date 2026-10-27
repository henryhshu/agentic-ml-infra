"""Training stage — runs PyTorch training via Ray Train with metrics and anomaly detection."""
from __future__ import annotations

import json
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
        self.step_telemetry: list[dict] = []

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
        """Run the training loop, as one DDP rank of a fresh process group when distributed."""
        from pipeline.distributed import get_context

        ctx = get_context()
        if ctx is None:
            return self._execute(None)
        spec = None
        if ctx.rank == 0:
            spec = self.config.to_dict()
            spec["workers_restarted"] = getattr(self.config, "workers_restarted", False)
        ctx.open_attempt(spec)
        closed = False
        try:
            return self._execute(ctx)
        except _TrainingAnomalyError:
            raise
        except Exception as exc:
            t_failed = time.time()
            ctx.close_attempt()
            closed = True
            if ctx.rank == 0:
                lost = ctx.worker_lost_error(exc)
                if lost is not None:
                    # measured: collective error on rank 0 -> liveness check confirmed the lost ranks
                    self.tracker.timeline.append({
                        "type": "worker_lost", "stage": self.STAGE_NAME, "start": t_failed, "end": time.time(),
                        "duration_s": time.time() - t_failed, "lost": lost.lost,
                    })
                    self.tracker.log_event(self.STAGE_NAME, "worker_lost", str(lost))
                    raise lost from exc
            raise
        finally:
            if not closed:
                ctx.close_attempt()

    def _execute(self, ctx) -> dict:
        """Run the training loop. Returns comprehensive metrics dict."""
        from pipeline.registry import get_model

        device = torch.device(self.config.device)
        num_classes = getattr(self.config, "num_classes", 10)

        # Build model via custom model_fn or registered builder
        if getattr(self.config, "model_fn", None) is not None:
            if isinstance(self.config.model_fn, nn.Module):
                model = self.config.model_fn.to(device)
            elif callable(self.config.model_fn):
                try:
                    model = self.config.model_fn(num_classes=num_classes).to(device)
                except TypeError:
                    model = self.config.model_fn().to(device)
            else:
                raise TypeError(f"Invalid model_fn type: {type(self.config.model_fn)}")
        else:
            model = get_model(self.config.model, num_classes=num_classes).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        self.tracker.log_event(
            self.STAGE_NAME,
            "info",
            f"Model: {self.config.model}, params={param_count:,}, device={device}",
        )

        train_loader, test_loader = self.train_loader, self.test_loader
        train_sampler = None
        if ctx is not None:
            # Global batch = config.batch_size, split evenly across ranks, so the optimization
            # matches the single-process run while every step all-reduces gradients.
            from torch.nn.parallel import DistributedDataParallel
            from torch.utils.data.distributed import DistributedSampler

            model = DistributedDataParallel(model)
            per_rank = max(1, self.config.batch_size // ctx.world_size)
            train_sampler = DistributedSampler(
                self.train_loader.dataset, ctx.world_size, ctx.rank, shuffle=True, seed=0
            )
            train_loader = DataLoader(self.train_loader.dataset, batch_size=per_rank, sampler=train_sampler,
                                      num_workers=self.config.num_workers)
            test_loader = DataLoader(
                self.test_loader.dataset, batch_size=per_rank,
                sampler=DistributedSampler(self.test_loader.dataset, ctx.world_size, ctx.rank, shuffle=False),
                num_workers=self.config.num_workers,
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

        # ── Fault injection: Structural Architecture Tensor Dimension Mismatch (Failure Frontier) ──
        if self.config.fault_injection == "architecture_tensor_mismatch":
            raise RuntimeError(
                "Internal PyTorch Architecture Defect: mat1 and mat2 shapes cannot be multiplied (128x512 and 1024x10). "
                "Linear projection layer in model architecture has dimension mismatch."
            )

        # ── Fault injection: Simulated Memory Pressure / OOM (Compound or standalone) ──
        if self.config.fault_injection in (
            "batch_oom", "memory_pressure", "compound_oom_and_explosion", "cascaded_lr_and_memory", "triple_cascade_oom_lr_grad"
        ) and self.config.batch_size >= 256:
            self._record_step_telemetry(0, 1, None, None, event="injected_oom")
            raise _TrainingAnomalyError(
                "memory_pressure",
                f"Simulated CUDA Out of Memory: failed to allocate memory for batch_size={self.config.batch_size}",
                {"batch_size": self.config.batch_size},
            )

        # ── Fault injection: Simulated Spot Node Preemption / Worker Task Failure ──
        # (single-process only; distributed runs kill a real worker process mid-epoch below)
        if (ctx is None and self.config.fault_injection == "spot_worker_churn"
                and not getattr(self.config, "workers_restarted", False)):
            self._record_step_telemetry(0, 1, None, None, event="injected_spot_interruption")
            raise _TrainingAnomalyError(
                "worker_shortage",
                "Ray cluster worker shortage: worker node preempted by EC2 Spot interruption during distributed training",
                {"worker_count": 0, "expected_workers": 2, "task_failed": True, "event": "spot_interruption"},
            )

        per_epoch: list[dict] = []
        best_acc = 0.0
        best_epoch = 0
        num_batches = len(train_loader)

        for epoch in range(self.config.epochs):
            epoch_num = epoch + 1
            t_epoch = time.time()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # ── Training pass ──
            model.train()
            running_loss = 0.0
            seen = 0
            total_grad_norm = 0.0
            grad_samples = 0
            optimizer.zero_grad()

            for step, (images, labels) in enumerate(train_loader, start=1):
                # ── Fault injection: worker process loss (distributed). The last rank is
                # SIGKILLed at epoch 1, step 3 until the remediator restarts the workers,
                # modelling a node that keeps failing until it is replaced. ──
                if (ctx is not None and ctx.rank == ctx.world_size - 1 and epoch_num == 1 and step == 3
                        and self.config.fault_injection == "spot_worker_churn"
                        and not getattr(self.config, "workers_restarted", False)):
                    ctx.crash()

                images = images.to(device, non_blocking=True)
                labels = torch.as_tensor(labels, device=device)

                # ── Fault injection: Catastrophic Label Noise (Tier 5 Statistical Trap) ──
                if self.config.fault_injection == "catastrophic_label_noise":
                    mask = torch.rand(labels.shape) < 0.90
                    labels = labels.clone()
                    labels[mask] = torch.randint(0, 10, (mask.sum().item(),), device=labels.device)

                loss = criterion(model(images), labels)
                if ctx is not None:
                    # global mean loss, so every rank takes the same NaN/anomaly decisions
                    loss_sum, n_sum = ctx.all_reduce_sum([loss.item() * labels.size(0), labels.size(0)])
                    loss_value = loss_sum / n_sum
                else:
                    loss_value = loss.item()

                # ── Fault injection: Tier 7 Thrashing Bimodal Trap (Control Frontier) ──
                if self.config.fault_injection == "thrashing_bimodal_trap":
                    if self.config.lr >= 0.01:
                        self._record_step_telemetry(step, epoch_num, loss=round(loss_value, 5), grad_norm=None, event="injected_nan")
                        raise _TrainingAnomalyError(
                            "nan_loss",
                            f"Loss exploded to NaN at epoch {epoch_num}, step {step}: unstable gradient trajectory with lr={self.config.lr:.4f} >= 0.01",
                            {"epoch": epoch_num, "step": step, "lr": self.config.lr, "loss": float("nan")},
                        )
                    else:
                        self._record_step_telemetry(step, epoch_num, loss=round(loss_value, 5), grad_norm=None, event="injected_plateau")
                        raise _TrainingAnomalyError(
                            "loss_plateau",
                            f"Loss stagnated in non-convex saddle at epoch {epoch_num}, step {step}: dead gradient flow with lr={self.config.lr:.6f} < 0.01",
                            {"epoch": epoch_num, "step": step, "lr": self.config.lr, "loss": 2.30},
                        )

                # ── Fault injection: NaN / Extreme LR divergence (Cascaded or standalone) ──
                if self.config.fault_injection in (
                    "nan_divergence", "nan_loss", "cascaded_lr_and_memory", "triple_cascade_oom_lr_grad"
                ) and self.config.lr >= 0.5:
                    self._record_step_telemetry(step, epoch_num, loss=round(loss_value, 5), grad_norm=None, event="injected_nan")
                    raise _TrainingAnomalyError(
                        "nan_loss",
                        f"NaN/Inf loss detected at epoch {epoch_num}, step {step}: divergent loss with excessive lr={self.config.lr}",
                        {"epoch": epoch_num, "step": step, "lr": self.config.lr, "loss": float("nan")},
                    )

                # ── NaN/Inf detection (fast-fail) ──
                if math.isnan(loss_value) or math.isinf(loss_value):
                    self._record_step_telemetry(step, epoch_num, loss=loss_value, grad_norm=None, event="nan_loss")
                    raise _TrainingAnomalyError(
                        "nan_loss",
                        f"NaN/Inf loss detected at epoch {epoch_num}, step {step}",
                        {"epoch": epoch_num, "step": step, "loss": loss_value},
                    )

                # ── Gradient Accumulation Guard: Loss Scaling & Boundary Gating ──
                accum_steps = max(1, getattr(self.config, "accum_steps", 1))
                is_accum_boundary = (step % accum_steps == 0) or (step == num_batches)
                scaled_loss = loss / accum_steps

                # Distributed no_sync() handling: suppress DDP all-reduce on non-boundary steps
                if hasattr(model, "no_sync") and not is_accum_boundary:
                    with model.no_sync():
                        scaled_loss.backward()
                else:
                    scaled_loss.backward()

                # Track gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

                # ── Fault injection: Gradient explosion (healed by gradient clipping) ──
                if self.config.fault_injection in (
                    "gradient_explosion", "compound_oom_and_explosion", "triple_cascade_oom_lr_grad"
                ) and self.config.max_grad_norm is None and step == 1:
                    measured_norm = total_norm
                    total_norm = 999.9  # injected fault signature reported to the agents
                    self._record_step_telemetry(step, epoch_num, loss=round(loss_value, 5), grad_norm=round(measured_norm, 4), event="injected_gradient_explosion")
                    raise _TrainingAnomalyError(
                        "gradient_explosion",
                        f"Gradient explosion detected: grad_norm={total_norm:.1f} exceeds stability threshold",
                        {"epoch": epoch_num, "step": step, "grad_norm": total_norm},
                    )

                # ── Complex Fault injection: Delayed Mid-Epoch Gradient Shock (step 2 explosion) ──
                if self.config.fault_injection == "delayed_stochastic_explosion" and self.config.max_grad_norm is None and step == 2:
                    measured_norm = total_norm
                    total_norm = 842.1  # injected fault signature reported to the agents
                    self._record_step_telemetry(step, epoch_num, loss=round(loss_value, 5), grad_norm=round(measured_norm, 4), event="injected_gradient_explosion")
                    raise _TrainingAnomalyError(
                        "gradient_explosion",
                        f"Stealth gradient shock detected at step 2: grad_norm={total_norm:.1f} exploded after step 1 normal loss",
                        {"epoch": epoch_num, "step": step, "grad_norm": total_norm, "delayed": True},
                    )

                total_grad_norm += total_norm
                grad_samples += 1

                # ── Combined Optimizer Update & Gradient Clearing at Accumulation Boundary ──
                if is_accum_boundary:
                    # Apply gradient clipping if enabled (set by remediator)
                    if self.config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item() * images.size(0)
                seen += images.size(0)

                # Record healthy step telemetry
                self._record_step_telemetry(
                    step=step,
                    epoch_num=epoch_num,
                    loss=round(loss_value, 5),
                    grad_norm=round(total_norm, 4),
                    event="step",
                )

                if step % self.config.log_every == 0 or step == num_batches:
                    print(
                        f"    epoch {epoch_num:02d}  step {step:>4d}/{num_batches}  "
                        f"loss={loss_value:.4f}  grad_norm={total_norm:.2f}"
                    )

            scheduler.step()
            if ctx is not None:
                running_loss, seen = ctx.all_reduce_sum([running_loss, seen])
            train_loss = running_loss / max(seen, 1)
            avg_grad_norm = total_grad_norm / max(grad_samples, 1)

            # ── Evaluation pass ──
            model.eval()
            correct = total = 0
            eval_loss = 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device, non_blocking=True)
                    labels = torch.as_tensor(labels, device=device)
                    outputs = model(images)
                    eval_loss += criterion(outputs, labels).item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            if ctx is not None:
                correct, total, eval_loss = ctx.all_reduce_sum([correct, total, eval_loss])
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
            abort = None
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
                    if "catastrophic_underperformance" in agent_result["anomalies"]:
                        abort = "catastrophic_underperformance"
            if ctx is not None:
                # the agents run on rank 0 only; every rank follows rank 0's decision
                abort = ctx.broadcast(abort)
            if abort == "catastrophic_underperformance":
                raise _TrainingAnomalyError(
                    "catastrophic_underperformance",
                    f"Model failed to converge: test accuracy {test_acc:.4f} is at or below random guessing threshold (0.15). Suspected label corruption or representational collapse.",
                    {"accuracy": test_acc, "loss": train_loss, "epoch": epoch_num},
                )

        self.model = model.module if ctx is not None else model
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

    def _record_step_telemetry(
        self,
        step: int,
        epoch_num: int,
        loss: float | None,
        grad_norm: float | None,
        event: str = "step",
        details: dict | None = None,
    ) -> None:
        """Record high-frequency step telemetry (measured values only) for analysis."""
        rss_mb = None
        try:
            import psutil
            rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        except Exception:
            pass

        point = {
            "timestamp": time.time(),
            "epoch": epoch_num,
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "memory_mb": rss_mb,
            "lr": self.config.lr,
            "batch_size": self.config.batch_size,
            "max_grad_norm": self.config.max_grad_norm,
            "event": event,
            "details": details or {},
        }
        self.step_telemetry.append(point)
        self._persist_telemetry()

    def _persist_telemetry(self) -> None:
        """Persist step telemetry to run directory as structured JSON."""
        if not hasattr(self.tracker, "run_dir") or not self.tracker.run_dir:
            return
        telemetry_file = Path(self.tracker.run_dir) / "step_telemetry.json"
        try:
            existing = []
            if telemetry_file.exists():
                try:
                    existing = json.loads(telemetry_file.read_text(encoding="utf-8"))
                except Exception:
                    existing = []
            seen = {p.get("timestamp") for p in existing}
            combined = existing + [p for p in self.step_telemetry if p.get("timestamp") not in seen]
            telemetry_file.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        except Exception:
            pass

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
