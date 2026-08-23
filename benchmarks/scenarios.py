"""Benchmark scenarios and preset fault injection catalog for testing agentic ML efficacy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkScenario:
    """Definition of a benchmark fault scenario."""

    id: str
    title: str
    category: str  # 'training_instability', 'resource_pressure', 'data_quality', 'optimizer_stagnation', 'baseline'
    description: str
    fault_type: str | None
    initial_overrides: dict[str, Any]
    expected_baseline_outcome: str
    expected_agentic_outcome: str
    target_stage: str
    is_recoverable: bool  # True if the fault can be automatically fixed; False if safe early halt is expected


# Preset catalog of benchmark scenarios
PRESET_SCENARIOS: dict[str, BenchmarkScenario] = {
    "nan_divergence": BenchmarkScenario(
        id="nan_divergence",
        title="Learning Rate Divergence (NaN Loss)",
        category="training_instability",
        description=(
            "Simulates an unstable initial hyperparameter setup (excessive LR=5.0) "
            "causing training loss to explode to NaN on step 1."
        ),
        fault_type="nan_divergence",
        initial_overrides={"lr": 5.0, "fault_injection": "nan_divergence"},
        expected_baseline_outcome="Fails: Crashes with NaN loss repeatedly across all retry attempts.",
        expected_agentic_outcome="Self-heals: TrainingWatchAgent detects NaN loss, Claude diagnoses LR divergence, Remediator reduces LR by 10x, retry succeeds.",
        target_stage="training",
        is_recoverable=True,
    ),
    "gradient_explosion": BenchmarkScenario(
        id="gradient_explosion",
        title="Unclipped Gradient Shock (Exploding Norm)",
        category="training_instability",
        description=(
            "Simulates severe gradient explosion during backward pass without gradient clipping."
        ),
        fault_type="gradient_explosion",
        initial_overrides={"fault_injection": "gradient_explosion", "max_grad_norm": None},
        expected_baseline_outcome="Fails: Crashes with gradient explosion error across all retries.",
        expected_agentic_outcome="Self-heals: Agent flags gradient explosion, Claude recommends clipping, Remediator enables max_grad_norm=1.0, retry succeeds.",
        target_stage="training",
        is_recoverable=True,
    ),
    "batch_oom": BenchmarkScenario(
        id="batch_oom",
        title="Memory Pressure (Simulated Batch OOM)",
        category="resource_pressure",
        description=(
            "Simulates an out-of-memory allocation error caused by an oversized initial batch_size=512."
        ),
        fault_type="batch_oom",
        initial_overrides={"batch_size": 512, "fault_injection": "batch_oom"},
        expected_baseline_outcome="Fails: Crashes with CUDA/MPS Out of Memory on every attempt with batch_size=512.",
        expected_agentic_outcome="Self-heals: Agent detects memory pressure, Claude diagnoses batch size, Remediator halves batch size (512->256->128), retry succeeds.",
        target_stage="training",
        is_recoverable=True,
    ),
    "corrupt_data": BenchmarkScenario(
        id="corrupt_data",
        title="Data Partition Corruption / Schema Mismatch",
        category="data_quality",
        description=(
            "Simulates a malformed dataset partition with invalid schema and corrupt parquet bytes."
        ),
        fault_type="corrupt_data",
        initial_overrides={"fault_injection": "corrupt_data"},
        expected_baseline_outcome="Fails: Data validation crashes; retries fail blindly.",
        expected_agentic_outcome="Early Diagnosis: DataQualityAgent identifies schema mismatch, Claude provides root-cause diagnosis, pipeline safely halts preventing wasted training compute.",
        target_stage="data_validation",
        is_recoverable=False,
    ),
    "loss_plateau": BenchmarkScenario(
        id="loss_plateau",
        title="Optimizer Stagnation / Dead Learning Rate",
        category="optimizer_stagnation",
        description=(
            "Simulates a near-zero learning rate (lr=1e-8) resulting in zero convergence."
        ),
        fault_type="loss_plateau",
        initial_overrides={"lr": 1e-8},
        expected_baseline_outcome="Silent failure: Completes all epochs with random accuracy (~10%), wasting compute without alerts.",
        expected_agentic_outcome="In-the-loop Diagnosis: Claude in-the-loop flags stagnant loss trajectory and negligible learning rate in real-time.",
        target_stage="training",
        is_recoverable=True,
    ),
    "clean_run": BenchmarkScenario(
        id="clean_run",
        title="Reference Baseline (Clean Pipeline)",
        category="baseline",
        description=(
            "Standard healthy pipeline run with optimal hyperparameters to verify baseline overhead."
        ),
        fault_type=None,
        initial_overrides={"lr": 0.1, "batch_size": 128, "fault_injection": None},
        expected_baseline_outcome="Passes: Completes all stages successfully.",
        expected_agentic_outcome="Passes: Completes all stages with Claude in-the-loop telemetry confirming healthy convergence.",
        target_stage="evaluation",
        is_recoverable=True,
    ),
}


def get_scenario(scenario_id: str) -> BenchmarkScenario:
    """Retrieve a scenario by ID."""
    if scenario_id not in PRESET_SCENARIOS:
        valid = ", ".join(PRESET_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_id}'. Available scenarios: {valid}")
    return PRESET_SCENARIOS[scenario_id]


def list_scenarios() -> list[BenchmarkScenario]:
    """Return all registered benchmark scenarios."""
    return list(PRESET_SCENARIOS.values())
