"""AutoPARK Comparative Benchmark & Efficacy Harness.

Executes controlled fault injection scenarios across paired modes:
  1. Standard Baseline (--no-agentic): Fixed retry loops, static configurations.
  2. AutoPARK Pipeline (--agentic): Real-time observation, Claude in-the-loop
     cognitive triage, and closed-loop knob/plugin remediations.

All timing figures are derived from wall-clock timestamps recorded during the run
(`timing.json`: stage attempts, incident-handling windows, every Claude call;
`action_log.json`: every remediation; `step_telemetry.json`: every optimizer step).
Nothing in this module is estimated or hardcoded.

MTTR decomposition for a run whose first failure starts at t_fault:
  T_det     execution time of failed attempts until the anomaly was raised
  T_reason  Claude call time inside incident-handling windows
  T_act     time spent executing remediation actions
  T_handle_other  remaining incident-handling time (prompt building, logging)
  T_recover time from the start of the successful attempt to its first healthy
            optimizer step (DataLoader rebuild, model init, first step)
  residual  untracked gaps between the above intervals
MTTR = T_det + T_reason + T_act + T_handle_other + T_recover + residual, exactly.
For runs that never recover, the same decomposition measures time-to-termination.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.scenarios import BenchmarkScenario, PRESET_SCENARIOS
from pipeline.config import PipelineConfig
from pipeline.runner import PipelineRunner

TIMING_KEYS = ("t_det", "t_reason", "t_act", "t_handle_other", "t_recover", "residual")


def compute_incident_timing(
    timeline: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Decompose measured run intervals into MTTR components (seconds)."""
    attempts = [e for e in timeline if e["type"] == "attempt"]
    handles = [e for e in timeline if e["type"] == "handle"]
    failed = [a for a in attempts if a["status"] == "failed"]

    def llm_time(start: float, end: float) -> float:
        return sum(c["duration_s"] for c in llm_calls if c["start"] >= start and c["end"] <= end)

    out: dict[str, Any] = {
        "llm_calls": len(llm_calls),
        "llm_time_s": sum(c["duration_s"] for c in llm_calls),
        "input_tokens": sum(c.get("input_tokens") or 0 for c in llm_calls),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in llm_calls),
        "monitor_llm_s": sum(llm_time(a["start"], a["end"]) for a in attempts),
        "incident": bool(failed),
    }
    if not failed:
        return out

    t_fault = failed[0]["start"]
    stage = failed[0]["stage"]
    stage_failed = [a for a in failed if a["stage"] == stage]
    success = next(
        (a for a in attempts if a["stage"] == stage and a["status"] == "success" and a["start"] >= t_fault),
        None,
    )
    if success is not None:
        healthy = [p["timestamp"] for p in steps if p.get("event") == "step" and p["timestamp"] >= success["start"]]
        t_end = min(healthy) if healthy else success["end"]
        t_recover = t_end - success["start"]
    else:
        t_end = max(e["end"] for e in timeline if e["stage"] == stage)
        t_recover = 0.0

    window = [h for h in handles if h["stage"] == stage and h["start"] >= t_fault and h["end"] <= t_end]
    t_det = sum(a["duration_s"] for a in stage_failed)
    t_reason = sum(llm_time(h["start"], h["end"]) for h in window)
    t_act = sum(
        a.get("duration_s") or 0.0 for a in actions if a.get("start") is not None and t_fault <= a["start"] <= t_end
    )
    t_handle = sum(h["duration_s"] for h in window)
    total = t_end - t_fault

    out.update({
        "stage": stage,
        "recovered": success is not None,
        "mttr_s": total,
        "t_det": t_det,
        "t_det_llm": sum(llm_time(a["start"], a["end"]) for a in stage_failed),
        "t_reason": t_reason,
        "t_act": t_act,
        "t_handle_other": t_handle - t_reason - t_act,
        "t_recover": t_recover,
        "residual": total - (t_det + t_handle + t_recover),
        "failed_attempts": len(stage_failed),
        "incidents_handled": len(window),
    })
    return out


def classify_outcome(status: str, incidents: list[dict[str, Any]], remediations: list[dict[str, Any]]) -> str:
    """Classify an agentic run from its recorded artifacts.

    healed     pipeline succeeded after >=1 successfully actuated remediation
    passed     pipeline succeeded with no incident
    recovered  pipeline succeeded after incidents but without a successful remediation
    escalated  pipeline failed and >=1 incident was escalated (no safe remediation)
    exhausted  pipeline failed although every incident was auto-remediated (retry budget spent)
    failed     pipeline failed with no incident recorded
    """
    actuated = [r for r in remediations if r.get("status") in ("success", "retry_requested")]
    if status == "success":
        if not incidents:
            return "passed"
        return "healed" if actuated else "recovered"
    if not incidents:
        return "failed"
    if any(not i.get("auto_remediated") for i in incidents):
        return "escalated"
    return "exhausted"


@dataclass
class SingleModeResult:
    """Result of a scenario execution in one mode (baseline or agentic)."""

    mode: str  # 'baseline' or 'agentic'
    status: str  # 'success' or 'failed'
    duration_s: float
    attempts: int
    final_stage: str
    error_message: str | None = None
    final_accuracy: float | None = None
    remediations_applied: list[dict[str, Any]] = field(default_factory=list)
    incidents: list[dict[str, Any]] = field(default_factory=list)
    claude_diagnoses: list[str] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    run_dir: str | None = None


@dataclass
class ScenarioBenchmarkResult:
    """Paired evaluation of one repeat of a scenario comparing baseline vs agentic."""

    scenario: BenchmarkScenario
    repeat: int
    baseline: SingleModeResult
    agentic: SingleModeResult
    outcome: str
    outcome_matches_expected: bool


@dataclass
class BenchmarkSuiteResult:
    """All repeats of all executed scenarios plus per-scenario aggregates."""

    timestamp: str
    environment: dict[str, Any]
    scenario_results: list[ScenarioBenchmarkResult] = field(default_factory=list)
    aggregates: list[dict[str, Any]] = field(default_factory=list)


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def aggregate(results: list[ScenarioBenchmarkResult]) -> list[dict[str, Any]]:
    """Per-scenario summary statistics over repeats."""
    by_id: dict[str, list[ScenarioBenchmarkResult]] = {}
    for r in results:
        by_id.setdefault(r.scenario.id, []).append(r)

    rows = []
    for sid, rs in by_id.items():
        sc = rs[0].scenario
        outcomes: dict[str, int] = {}
        for r in rs:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        timed = [r.agentic.timing for r in rs if r.agentic.timing.get("incident")]
        rows.append({
            "id": sid,
            "expected_outcome": sc.expected_outcome,
            "n": len(rs),
            "matches_expected": sum(r.outcome_matches_expected for r in rs),
            "outcomes": outcomes,
            "baseline_status": {s: sum(r.baseline.status == s for r in rs) for s in ("success", "failed")},
            "baseline_duration_s": _stats([r.baseline.duration_s for r in rs]),
            "baseline_accuracy": _stats([r.baseline.final_accuracy for r in rs if r.baseline.final_accuracy is not None]),
            "agentic_duration_s": _stats([r.agentic.duration_s for r in rs]),
            "agentic_accuracy": _stats([r.agentic.final_accuracy for r in rs if r.agentic.final_accuracy is not None]),
            "mttr_s": _stats([t["mttr_s"] for t in timed]),
            **{k: _stats([t[k] for t in timed]) for k in TIMING_KEYS},
            "t_det_llm": _stats([t["t_det_llm"] for t in timed]),
            "llm_calls": _stats([float(r.agentic.timing.get("llm_calls", 0)) for r in rs]),
            "input_tokens": _stats([float(r.agentic.timing.get("input_tokens", 0)) for r in rs]),
            "output_tokens": _stats([float(r.agentic.timing.get("output_tokens", 0)) for r in rs]),
            "actions_per_run": [[a.get("action") for a in r.agentic.remediations_applied] for r in rs],
        })
    return rows


class BenchmarkHarness:
    """Executes benchmark scenarios and measures agentic efficacy."""

    def __init__(
        self,
        model: str = "tiny",
        epochs: int = 2,
        subset: int = 500,
        max_retries: int = 3,
        base_dir: str = "experiments/benchmarks",
        device: str | None = None,
        repeats: int = 1,
    ):
        from pipeline.config import get_default_device

        self.model = model
        self.epochs = epochs
        self.subset = subset
        self.max_retries = max_retries
        self.device = device or get_default_device()
        self.repeats = repeats
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run_scenario(self, scenario: BenchmarkScenario, repeat: int = 0) -> ScenarioBenchmarkResult:
        """Run a single scenario in both Baseline (Standard) and Agentic modes."""
        print(f"\n{'='*75}")
        print(f"🧪 BENCHMARK SCENARIO: {scenario.title}  (repeat {repeat + 1}/{self.repeats})")
        print(f"   ID: {scenario.id}  |  Category: {scenario.category}  |  Expected: {scenario.expected_outcome}")
        print(f"{'='*75}")

        print(f"\n[1/2] 🔴 Running Standard Baseline (--no-agentic)...")
        baseline_result = self._execute_mode(scenario, enable_agentic=False)
        print(f"  → Baseline Result: status={baseline_result.status.upper()}, attempts={baseline_result.attempts}, time={baseline_result.duration_s:.1f}s")

        print(f"\n[2/2] 🤖 Running Agentic Pipeline (--agentic)...")
        agentic_result = self._execute_mode(scenario, enable_agentic=True)
        print(f"  → Agentic Result: status={agentic_result.status.upper()}, attempts={agentic_result.attempts}, time={agentic_result.duration_s:.1f}s")

        outcome = classify_outcome(
            agentic_result.status, agentic_result.incidents, agentic_result.remediations_applied
        )
        print(f"  → Outcome: {outcome} (expected {scenario.expected_outcome})")

        return ScenarioBenchmarkResult(
            scenario=scenario,
            repeat=repeat,
            baseline=baseline_result,
            agentic=agentic_result,
            outcome=outcome,
            outcome_matches_expected=outcome == scenario.expected_outcome,
        )

    def _execute_mode(self, scenario: BenchmarkScenario, enable_agentic: bool) -> SingleModeResult:
        """Execute a single run of the pipeline for a scenario."""
        mode_str = "agentic" if enable_agentic else "baseline"

        config_kwargs: dict[str, Any] = {
            "model": self.model,
            "epochs": self.epochs,
            "batch_size": 128,
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "num_workers": 0,
            "device": self.device,
            "use_gpu": self.device in ("mps", "cuda"),
            "enable_agentic": enable_agentic,
            "max_retries": self.max_retries,
            "enable_tuning": False,
            "subset": self.subset,
            "log_every": 50,
            "fault_injection": scenario.fault_type,
        }
        config_kwargs.update(scenario.initial_overrides)
        config = PipelineConfig(**config_kwargs)

        runner = PipelineRunner(config)
        t_start = time.time()
        error_msg = None
        try:
            summary = runner.run()
            status = summary.get("status", "success")
            final_stage = summary["stages_run"][-1] if summary.get("stages_run") else "none"
        except Exception as exc:
            status = "failed"
            error_msg = str(exc)
            final_stage = "error"
        duration = time.time() - t_start

        remediations: list[dict[str, Any]] = []
        incidents: list[dict[str, Any]] = []
        claude_diagnoses: list[str] = []
        if enable_agentic and runner.supervisor is not None:
            incidents = runner.supervisor.incidents
            remediator = runner.supervisor.sub_agents.get("remediator")
            if remediator is not None and hasattr(remediator, "action_log"):
                remediations = remediator.action_log
            claude_diagnoses = [inc.get("diagnosis", "") for inc in incidents if inc.get("diagnosis")]

        steps: list[dict[str, Any]] = []
        telemetry_file = Path(runner.tracker.run_dir) / "step_telemetry.json"
        if telemetry_file.exists():
            try:
                steps = json.loads(telemetry_file.read_text(encoding="utf-8"))
            except Exception:
                steps = []

        timing = compute_incident_timing(runner.tracker.timeline, runner.tracker.llm_calls, remediations, steps)
        attempts = sum(1 for e in runner.tracker.timeline if e["type"] == "attempt")

        final_accuracy = None
        ev = runner.tracker.stages.get("evaluation")
        if ev is not None and ev.status == "success":
            final_accuracy = ev.metrics.get("test_accuracy")

        return SingleModeResult(
            mode=mode_str,
            status=status,
            duration_s=round(duration, 3),
            attempts=attempts,
            final_stage=final_stage,
            error_message=error_msg,
            final_accuracy=final_accuracy,
            remediations_applied=remediations,
            incidents=incidents,
            claude_diagnoses=claude_diagnoses,
            timing=timing,
            run_dir=str(runner.tracker.run_dir),
        )

    def run_suite(self, scenarios: list[BenchmarkScenario] | None = None) -> BenchmarkSuiteResult:
        """Run all or specified benchmark scenarios `repeats` times and aggregate."""
        import datetime
        import os
        import platform

        import torch

        target_scenarios = scenarios or list(PRESET_SCENARIOS.values())
        results: list[ScenarioBenchmarkResult] = []

        print(f"\n{'#'*75}")
        print(f"🚀 STARTING AGENTIC ML BENCHMARK SUITE ({len(target_scenarios)} scenarios x {self.repeats} repeats)")
        print(f"   Model: {self.model}  |  Epochs: {self.epochs}  |  Subset: {self.subset} samples  |  Device: {self.device}")
        print(f"{'#'*75}")

        for repeat in range(self.repeats):
            for sc in target_scenarios:
                results.append(self.run_scenario(sc, repeat))

        environment = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": self.device,
            "model": self.model,
            "epochs": self.epochs,
            "subset": self.subset,
            "max_retries": self.max_retries,
            "repeats": self.repeats,
            "llm_model": os.environ.get("ANTHROPIC_MODEL") or os.environ.get("CLAUDE_MODEL") or "claude-haiku-4-5-20251001",
            "world_size": int(os.environ.get("AUTOPARK_WORLD_SIZE", "1")),
            "dist_backend": "gloo" if int(os.environ.get("AUTOPARK_WORLD_SIZE", "1")) > 1 else None,
        }
        return BenchmarkSuiteResult(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            environment=environment,
            scenario_results=results,
            aggregates=aggregate(results),
        )
