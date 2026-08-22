"""Pipeline runner — orchestrates all stages with optional agentic monitoring."""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from metrics.experiment import ExperimentTracker
from metrics.schemas import now_iso
from pipeline.config import PipelineConfig


class PipelineRunner:
    """Orchestrates the full ML pipeline: validation → loading → training → [tuning] → evaluation.

    When agentic monitoring is enabled, an agent fleet watches each stage,
    detects anomalies, and can trigger intelligent retries.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracker = ExperimentTracker(
            run_id=config.run_id,
            base_dir="experiments",
        )
        self.tracker.set_config(config.to_dict())
        self.checkpoint_emitter = None
        self.supervisor = None

        # Set up agentic monitoring if enabled
        if config.enable_agentic:
            self._init_agents()

    def _init_agents(self) -> None:
        """Initialize the agent monitoring fleet."""
        from agents.checkpoints import CheckpointEmitter
        from agents.supervisor import SupervisorAgent
        from agents.training_watch import TrainingWatchAgent
        from agents.cluster_health import ClusterHealthAgent
        from agents.data_quality import DataQualityAgent
        from agents.remediator import RemediatorAgent

        self.checkpoint_emitter = CheckpointEmitter(self.tracker.run_dir)

        # Create agents
        self.supervisor = SupervisorAgent(
            name="supervisor",
            tracker=self.tracker,
            config=self.config,
            checkpoint_emitter=self.checkpoint_emitter,
        )

        training_watch = TrainingWatchAgent(
            name="training_watch",
            tracker=self.tracker,
            config=self.config,
            checkpoint_emitter=self.checkpoint_emitter,
        )
        cluster_health = ClusterHealthAgent(
            name="cluster_health",
            tracker=self.tracker,
            config=self.config,
            checkpoint_emitter=self.checkpoint_emitter,
        )
        data_quality = DataQualityAgent(
            name="data_quality",
            tracker=self.tracker,
            config=self.config,
            checkpoint_emitter=self.checkpoint_emitter,
        )
        remediator = RemediatorAgent(
            name="remediator",
            tracker=self.tracker,
            config=self.config,
            checkpoint_emitter=self.checkpoint_emitter,
        )

        self.supervisor.register_agent("training_watch", training_watch)
        self.supervisor.register_agent("cluster_health", cluster_health)
        self.supervisor.register_agent("data_quality", data_quality)
        self.supervisor.register_agent("remediator", remediator)

        print("🤖 Agentic monitoring fleet initialized:")
        print("   • SupervisorAgent — orchestrates fleet, routes anomalies")
        print("   • TrainingWatchAgent — monitors loss/accuracy/gradients")
        print("   • ClusterHealthAgent — watches Ray cluster status")
        print("   • DataQualityAgent — validates data pipeline")
        print("   • RemediatorAgent — executes approved remediation actions")

    def run(self) -> dict:
        """Execute the full pipeline. Returns the experiment summary."""
        run_id = self.tracker.run_id
        agentic_label = "🤖 AGENTIC" if self.config.enable_agentic else "📊 STANDARD"
        print(f"\n{'='*70}")
        print(f"  {agentic_label} ML Pipeline — Run {run_id}")
        print(f"{'='*70}")
        print(f"  Model: {self.config.model}  |  Epochs: {self.config.epochs}  |  "
              f"LR: {self.config.lr}  |  Batch: {self.config.batch_size}")
        if self.config.subset:
            print(f"  Subset: {self.config.subset} samples")
        if self.config.enable_tuning:
            print(f"  Tuning: {self.config.tune_num_samples} trials (ASHA scheduler)")
        print(f"{'='*70}\n")

        t_start = time.time()
        pipeline_status = "success"
        stages_run = []

        try:
            # ── Stage 1: Data Validation ──
            print("📋 Stage 1/5: Data Validation")
            dv_metrics = self._run_stage_with_retry("data_validation", self._run_data_validation)
            stages_run.append("data_validation")

            # ── Stage 2: Data Loading ──
            print("\n📦 Stage 2/5: Data Loading")
            dl_stage = self._run_stage_with_retry("data_loading", self._run_data_loading)
            stages_run.append("data_loading")

            # ── Stage 3: Training ──
            print(f"\n🏋️ Stage 3/5: Training ({self.config.epochs} epochs)")
            train_result = self._run_stage_with_retry("training", self._run_training)
            stages_run.append("training")

            # ── Stage 4: Tuning (optional) ──
            if self.config.enable_tuning:
                print(f"\n🔧 Stage 4/5: Hyperparameter Tuning ({self.config.tune_num_samples} trials)")
                self._run_stage_with_retry("tuning", self._run_tuning)
                stages_run.append("tuning")
            else:
                print("\n⏭️  Stage 4/5: Tuning (skipped — use --tune to enable)")
                self.tracker.start_stage("tuning")
                self.tracker.end_stage("tuning", status="skipped")
                self.tracker.save_stage("tuning")

            # ── Stage 5: Evaluation ──
            print("\n📊 Stage 5/5: Final Evaluation")
            self._run_stage_with_retry("evaluation", self._run_evaluation)
            stages_run.append("evaluation")

        except Exception as exc:
            pipeline_status = "failed"
            print(f"\n❌ Pipeline failed at stage: {exc}")
            traceback.print_exc()

        # ── Save final summary ──
        total_time = time.time() - t_start
        self.tracker.end_time = now_iso()
        self.tracker.status = pipeline_status
        self.tracker.save_summary()

        # Save agent artifacts
        if self.supervisor is not None:
            self.supervisor.save_incidents()
            # Save remediator audit log
            remediator = self.supervisor.sub_agents.get("remediator")
            if remediator is not None:
                remediator.save_audit_log()

        # ── Print final summary ──
        print(f"\n{'='*70}")
        print(f"  Pipeline {pipeline_status.upper()} — {total_time:.1f}s total")
        print(f"  Stages: {', '.join(stages_run)}")
        print(f"{'='*70}")

        # Print coherent metrics summary from all stages
        self._print_metrics_summary()

        print(f"\n  📁 Full results: experiments/{run_id}/")
        print(f"{'='*70}\n")

        return {
            "run_id": run_id,
            "status": pipeline_status,
            "duration_s": round(total_time, 2),
            "stages_run": stages_run,
            "results_dir": str(self.tracker.run_dir),
        }
    def _print_metrics_summary(self) -> None:
        """Print a coherent summary table of metrics across all stages."""
        stages = self.tracker.stages
        print()
        rows: list[tuple[str, str, str]] = []

        # Data Validation
        dv = stages.get("data_validation")
        if dv and dv.status == "success":
            score = dv.metrics.get("data_quality_score", 0)
            n_train = dv.metrics.get("num_train_samples", "?")
            n_test = dv.metrics.get("num_test_samples", "?")
            rows.append(("Data Validation", f"{score:.0%} quality", f"{n_train} train / {n_test} test"))

        # Data Loading
        dl = stages.get("data_loading")
        if dl and dl.status == "success":
            t = dl.metrics.get("load_time_s", "?")
            ts = dl.metrics.get("train_samples", "?")
            bs = dl.metrics.get("batch_size", "?")
            rows.append(("Data Loading", f"{t}s load time", f"{ts} samples, batch={bs}"))

        # Training
        tr = stages.get("training")
        if tr and tr.status == "success":
            acc = tr.metrics.get("best_test_accuracy")
            loss = tr.metrics.get("final_train_loss")
            ep = tr.metrics.get("best_epoch", "?")
            t = tr.metrics.get("total_training_time_s", "?")
            acc_s = f"{acc:.4f}" if acc is not None else "n/a"
            loss_s = f"{loss:.4f}" if loss is not None else "n/a"
            rows.append(("Training", f"best_acc={acc_s} (epoch {ep})", f"loss={loss_s}, {t}s"))

        # Tuning
        tu = stages.get("tuning")
        if tu and tu.status == "success":
            best = tu.metrics.get("best_trial_accuracy")
            n = tu.metrics.get("num_trials", "?")
            cfg = tu.metrics.get("best_config", {})
            best_s = f"{best:.4f}" if best is not None else "n/a"
            cfg_s = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in cfg.items()) if cfg else "n/a"
            rows.append(("Tuning", f"best_acc={best_s} ({n} trials)", cfg_s))
        elif tu and tu.status == "skipped":
            rows.append(("Tuning", "skipped", ""))

        # Evaluation
        ev = stages.get("evaluation")
        if ev and ev.status == "success":
            acc = ev.metrics.get("test_accuracy")
            loss = ev.metrics.get("test_loss")
            thr = ev.metrics.get("inference_throughput_sps")
            acc_s = f"{acc:.4f}" if acc is not None else "n/a"
            loss_s = f"{loss:.4f}" if loss is not None else "n/a"
            thr_s = f"{thr:.0f}" if thr is not None else "n/a"
            rows.append(("Evaluation", f"accuracy={acc_s}", f"loss={loss_s}, {thr_s} samples/s"))

        # Print table
        if rows:
            col1_w = max(len(r[0]) for r in rows) + 2
            col2_w = max(len(r[1]) for r in rows) + 2
            for stage_name, metric, detail in rows:
                line = f"  {stage_name:<{col1_w}}  {metric:<{col2_w}}"
                if detail:
                    line += f"  {detail}"
                print(line)

    def _run_stage_with_retry(self, stage_name: str, stage_fn, max_retries: int | None = None):
        """Run a stage with automatic retry logic, optionally mediated by agents."""
        retries = max_retries if max_retries is not None else self.config.max_retries
        last_error = None

        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    print(f"  🔄 Retry {attempt}/{retries} for {stage_name}...")
                    self.tracker.log_event(
                        stage_name, "retry",
                        f"Retry attempt {attempt}/{retries}",
                        data={"attempt": attempt, "previous_error": str(last_error)},
                    )
                    if self.checkpoint_emitter is not None:
                        from agents.checkpoints import AgentCheckpoint
                        self.checkpoint_emitter.emit(AgentCheckpoint(
                            stage=stage_name,
                            event_type="retry_initiated",
                            timestamp=now_iso(),
                            metrics_snapshot={"attempt": attempt},
                            anomalies_detected=[str(last_error)] if last_error else [],
                            agent_diagnosis=None,
                            recommended_action="retry_stage",
                            action_taken="retry_stage",
                        ))

                result = stage_fn()

                # If agentic, let agents observe the successful completion
                if self.supervisor is not None:
                    self.supervisor.route_event({
                        "type": "stage_complete",
                        "stage": stage_name,
                        "status": "success",
                        "attempt": attempt,
                    })

                return result

            except Exception as exc:
                last_error = exc
                self.tracker.log_event(
                    stage_name, "error",
                    f"Stage failed (attempt {attempt + 1}): {exc}",
                )

                # If agentic, let agents diagnose and potentially remediate
                if self.supervisor is not None:
                    anomalies = [str(exc)]
                    context = {
                        "stage": stage_name,
                        "attempt": attempt + 1,
                        "error": str(exc),
                        "config": self.config.to_dict(),
                    }
                    action = self.supervisor.handle_anomaly(stage_name, anomalies, context)
                    print(f"  🤖 Agent recommendation: {action}")

                if attempt >= retries:
                    print(f"  ❌ Stage {stage_name} failed after {retries + 1} attempts")
                    raise

        return None  # Unreachable but satisfies type checker

    # ── Stage implementations ──

    def _run_data_validation(self):
        from pipeline.data_validation import DataValidationStage
        stage = DataValidationStage(
            self.config, self.tracker, self.checkpoint_emitter
        )
        return stage.run()

    def _run_data_loading(self):
        from pipeline.data_loading import DataLoadingStage
        stage = DataLoadingStage(
            self.config, self.tracker, self.checkpoint_emitter
        )
        result = stage.run()
        # Store loaders for subsequent stages
        self._train_loader = stage.train_loader
        self._test_loader = stage.test_loader
        self._train_set = stage.train_set
        self._test_set = stage.test_set
        return result

    def _run_training(self):
        from pipeline.training import TrainingStage
        train_set_size = len(self._train_set)
        stage = TrainingStage(
            self.config,
            self.tracker,
            self._train_loader,
            self._test_loader,
            train_set_size,
            self.checkpoint_emitter,
        )
        result = stage.run()
        self._trained_model = stage.model
        self._best_accuracy = stage.best_accuracy
        return result

    def _run_tuning(self):
        from pipeline.tuning import TuningStage
        stage = TuningStage(
            self.config, self.tracker, self.checkpoint_emitter
        )
        result = stage.run()
        if stage.best_config:
            print(f"  🔧 Best config: {stage.best_config}")
        return result

    def _run_evaluation(self):
        from pipeline.evaluation import EvaluationStage
        stage = EvaluationStage(
            self.config,
            self.tracker,
            self._trained_model,
            self._test_loader,
            self.checkpoint_emitter,
        )
        return stage.run()
