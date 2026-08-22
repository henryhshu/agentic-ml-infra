"""Hyperparameter tuning stage — runs Ray Tune with ASHA scheduler."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

from metrics.experiment import ExperimentTracker
from metrics.schemas import TuningMetrics, now_iso


class TuningStage:
    """Pipeline stage that performs hyperparameter search using Ray Tune."""

    STAGE_NAME = "tuning"

    def __init__(
        self,
        config,  # PipelineConfig
        tracker: ExperimentTracker,
        checkpoint_emitter=None,
    ):
        self.config = config
        self.tracker = tracker
        self.checkpoint_emitter = checkpoint_emitter
        # Populated after run()
        self.best_config: dict | None = None

    @property
    def name(self) -> str:
        return self.STAGE_NAME

    def run(self) -> TuningMetrics:
        """Run hyperparameter tuning and return metrics."""
        metrics = self.tracker.start_stage(self.STAGE_NAME, metrics_cls=TuningMetrics)
        self._emit_checkpoint("stage_start", {
            "num_samples": self.config.tune_num_samples,
        })

        try:
            result = self.execute()
            metrics.num_trials = result["num_trials"]
            metrics.best_config = result["best_config"]
            metrics.best_trial_accuracy = result["best_trial_accuracy"]
            metrics.all_trial_results = result["all_trial_results"]
            metrics.metrics.update(result)
            self.tracker.end_stage(self.STAGE_NAME, status="success")
            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("stage_end", {
                "best_accuracy": result["best_trial_accuracy"],
                "best_config": result["best_config"],
            })
            print(
                f"  ✅ Tuning: best_acc={result['best_trial_accuracy']:.4f} "
                f"from {result['num_trials']} trials"
            )
            return metrics

        except Exception as exc:
            self.tracker.log_event(
                self.STAGE_NAME, "error", f"Tuning failed: {exc}"
            )
            self.tracker.end_stage(self.STAGE_NAME, status="failed")
            metrics.error_message = str(exc)
            self.tracker.save_stage(self.STAGE_NAME)
            self._emit_checkpoint("anomaly_detected", {"error": str(exc)})
            raise

    def execute(self) -> dict:
        """Run Ray Tune hyperparameter search."""
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler

        self.tracker.log_event(self.STAGE_NAME, "info", "Initializing Ray Tune...")

        # Define search space
        search_space = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([64, 128, 256]),
            "weight_decay": tune.loguniform(1e-5, 1e-2),
            "model": self.config.model,
            "epochs": self.config.epochs,
            "device": self.config.device,
            "cache_dir": self.config.cache_dir,
            "subset": self.config.subset,
            "num_workers": self.config.num_workers,
        }

        # ASHA scheduler for early stopping of bad trials
        scheduler = ASHAScheduler(
            metric="test_accuracy",
            mode="max",
            max_t=self.config.epochs,
            grace_period=1,
            reduction_factor=2,
        )

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2)

        self.tracker.log_event(
            self.STAGE_NAME,
            "info",
            f"Starting {self.config.tune_num_samples} trials with ASHA scheduler",
        )

        tuner = tune.Tuner(
            _tune_train_fn,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=self.config.tune_num_samples,
                max_concurrent_trials=1,
            ),
            run_config=tune.RunConfig(
                name="cifar10_tune",
            ),
        )

        results = tuner.fit()

        # Collect all trial results
        all_trial_results = []
        for i, result in enumerate(results):
            trial_result = {
                "trial_id": i,
                "config": {
                    "lr": result.config.get("lr"),
                    "batch_size": result.config.get("batch_size"),
                    "weight_decay": result.config.get("weight_decay"),
                },
                "test_accuracy": result.metrics.get("test_accuracy", 0.0) if result.metrics else 0.0,
                "status": "completed" if not result.error else "errored",
            }
            all_trial_results.append(trial_result)
            self.tracker.log_event(
                self.STAGE_NAME,
                "trial_complete",
                f"Trial {i}: acc={trial_result['test_accuracy']:.4f}",
                data=trial_result,
            )
            self._emit_checkpoint("epoch_boundary", trial_result)

        # Find best result
        best_result = results.get_best_result(metric="test_accuracy", mode="max")
        best_config = {
            "lr": best_result.config.get("lr"),
            "batch_size": best_result.config.get("batch_size"),
            "weight_decay": best_result.config.get("weight_decay"),
        }
        best_accuracy = best_result.metrics.get("test_accuracy", 0.0) if best_result.metrics else 0.0

        self.best_config = best_config

        self.tracker.log_metric(self.STAGE_NAME, "num_trials", len(all_trial_results))
        self.tracker.log_metric(self.STAGE_NAME, "best_trial_accuracy", best_accuracy)
        self.tracker.log_metric(self.STAGE_NAME, "best_config", best_config)

        return {
            "num_trials": len(all_trial_results),
            "best_config": best_config,
            "best_trial_accuracy": round(best_accuracy, 6),
            "all_trial_results": all_trial_results,
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


def _tune_train_fn(config: dict) -> None:
    """Training function for Ray Tune trials. Runs inside a Ray worker."""
    import io
    import math

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    from ray import tune

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from datasets.cifar10 import CIFAR10Parquet, download_split
    from models.resnet import build_resnet18
    from models.tiny_cnn import build_tiny_cnn

    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    builders = {"resnet18": build_resnet18, "tiny": build_tiny_cnn}
    model_name = config.get("model", "tiny")
    epochs = config.get("epochs", 5)
    lr = config["lr"]
    batch_size = config["batch_size"]
    weight_decay = config["weight_decay"]
    device = torch.device(config.get("device", "cpu"))
    cache_dir = config.get("cache_dir", "data/cifar10")
    subset = config.get("subset")
    num_workers = config.get("num_workers", 0)

    # Build transforms
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Load data
    train_path = download_split("train", cache_dir)
    test_path = download_split("test", cache_dir)
    train_set = CIFAR10Parquet(train_path, transform=train_tfm)
    test_set = CIFAR10Parquet(test_path, transform=eval_tfm)

    if subset is not None:
        train_set = Subset(train_set, range(min(subset, len(train_set))))
        test_set = Subset(test_set, range(min(subset, len(test_set))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Build model
    model = builders[model_name](num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_set_size = len(train_set)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, device=device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                tune.report({"test_accuracy": 0.0, "train_loss": float("inf"), "epoch": epoch + 1})
                return
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = torch.as_tensor(labels, device=device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc = correct / total if total > 0 else 0.0
        train_loss = running_loss / train_set_size

        tune.report({"test_accuracy": test_acc, "train_loss": train_loss, "epoch": epoch + 1})
