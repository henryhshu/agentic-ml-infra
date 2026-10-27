"""AutoPARK Fault Injection Catalog & Progressive Difficulty Testbench.

Defines the complete 7-tier benchmark escalation matrix designed to map the
empirical Self-Healing Frontier of autonomous ML platforms:
  - Tier 1: Isolated Hyperparameter Faults (NaN Loss, Batch OOM)
  - Tier 2: Multi-Fault Cascades (Compound OOM + Gradient Shock, Memory + Runaway LR)
  - Tier 3: Complex Deceptive & Infrastructure (Mid-Epoch Delayed Explosion, Spot Churn)
  - Tier 4: Triple Cascaded Instability (OOM -> Runaway LR -> Grad Explosion)
  - Tier 5: Statistical Silent Traps (90% Catastrophic Label Noise)
  - Tier 6: Structural Architecture Frontier (PyTorch Tensor Dimension Mismatch)
  - Tier 7: Dynamical Control Frontier (Antagonistic Bimodal Thrashing Trap)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkScenario:
    """Specification for an AutoPARK fault injection scenario.

    Attributes:
        id: Unique scenario identifier string.
        title: Human-readable scenario title.
        category: Failure class taxonomy (e.g., 'training_instability', 'multi_fault_cascade').
        description: Technical explanation of the injected defect.
        fault_type: Key matching the runtime trigger in pipeline stages.
        initial_overrides: Config parameter mutations applied before run start.
        expected_baseline_outcome: Anticipated unassisted baseline behavior.
        expected_agentic_outcome: Anticipated AutoPARK autonomous resolution or escalation.
        target_stage: Primary pipeline stage where the fault manifests.
        is_recoverable: True if in-memory self-healing is possible; False if safe escalation is expected.
        expected_outcome: Outcome the agentic run is scored against: 'healed', 'escalated',
            'exhausted', or 'passed' (fault-free run with no false-positive remediation).
    """

    id: str
    title: str
    category: str
    description: str
    fault_type: str | None
    initial_overrides: dict[str, Any]
    expected_baseline_outcome: str
    expected_agentic_outcome: str
    target_stage: str
    is_recoverable: bool
    expected_outcome: str = "healed"


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
        expected_outcome="escalated",
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
        expected_outcome="escalated",
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
        expected_outcome="passed",
    ),
    "compound_oom_and_explosion": BenchmarkScenario(
        id="compound_oom_and_explosion",
        title="Cascaded Multi-Fault: Memory OOM + Gradient Shock",
        category="multi_fault_cascade",
        description=(
            "Simulates concurrent faults: oversized batch_size=512 triggers memory exhaustion; "
            "once resolved (batch_size halved to 128), unclipped gradient explosion hits on backward pass."
        ),
        fault_type="compound_oom_and_explosion",
        initial_overrides={"batch_size": 512, "max_grad_norm": None, "fault_injection": "compound_oom_and_explosion"},
        expected_baseline_outcome="Fails: Crashes continuously across retries without adapting batch size or clipping.",
        expected_agentic_outcome="Self-heals: Agent resolves memory pressure first (halving batch_size 512->256->128), then detects gradient explosion and enables clipping (max_grad_norm=1.0), succeeding on attempt 3.",
        target_stage="training",
        is_recoverable=True,
    ),
    "delayed_stochastic_explosion": BenchmarkScenario(
        id="delayed_stochastic_explosion",
        title="Complex Deceptive Fault: Mid-Epoch Delayed Gradient Shock",
        category="complex_instability",
        description=(
            "Deceptive failure mode: step 1 passes with normal loss (~2.3), but step 2 suffers "
            "sudden numerical explosion due to unclipped weight updates."
        ),
        fault_type="delayed_stochastic_explosion",
        initial_overrides={"max_grad_norm": None, "fault_injection": "delayed_stochastic_explosion"},
        expected_baseline_outcome="Fails: Crashes mid-epoch on step 2 across all retries.",
        expected_agentic_outcome="Self-heals: Agent identifies mid-epoch gradient shock, Claude recommends clipping, Remediator enables max_grad_norm=1.0, retry completes full epoch.",
        target_stage="training",
        is_recoverable=True,
    ),
    "cascaded_lr_and_memory": BenchmarkScenario(
        id="cascaded_lr_and_memory",
        title="Cascaded Multi-Fault: Memory Pressure + Divergent Learning Rate",
        category="multi_fault_cascade",
        description=(
            "Simulates dual misconfiguration: oversized batch_size=256 coupled with runaway learning rate lr=5.0."
        ),
        fault_type="cascaded_lr_and_memory",
        initial_overrides={"batch_size": 256, "lr": 5.0, "fault_injection": "cascaded_lr_and_memory"},
        expected_baseline_outcome="Fails: Crashes with OOM on attempt 1, then would crash with NaN loss.",
        expected_agentic_outcome="Self-heals: Claude diagnoses memory pressure -> halves batch size, then diagnoses LR divergence -> reduces LR by 10x twice, successfully converging.",
        target_stage="training",
        is_recoverable=True,
    ),
    "spot_worker_churn": BenchmarkScenario(
        id="spot_worker_churn",
        title="Complex Infrastructure Fault: Spot Node Task Preemption",
        category="cluster_fault",
        description=(
            "Simulates an AWS Spot instance worker preemption during distributed training, throwing transient task failure."
        ),
        fault_type="spot_worker_churn",
        initial_overrides={"fault_injection": "spot_worker_churn"},
        expected_baseline_outcome="Fails: Unhandled task interruption terminates pipeline.",
        expected_agentic_outcome="Self-heals: ClusterHealthAgent catches task failure, Claude diagnoses worker churn, Remediator triggers retry_stage, subsequent attempt completes.",
        target_stage="training",
        is_recoverable=True,
    ),
    "triple_cascade_oom_lr_grad": BenchmarkScenario(
        id="triple_cascade_oom_lr_grad",
        title="Tier 4 Triple Cascade: OOM + Runaway LR + Gradient Explosion",
        category="multi_fault_cascade",
        description=(
            "Simulates extreme compounding failure: oversized batch_size=256 triggers OOM, "
            "followed by runaway LR=5.0 triggering loss explosion, followed by unclipped gradient explosion."
        ),
        fault_type="triple_cascade_oom_lr_grad",
        initial_overrides={"batch_size": 256, "lr": 5.0, "max_grad_norm": None, "fault_injection": "triple_cascade_oom_lr_grad"},
        expected_baseline_outcome="Fails: Crashes across all retries at different failure points.",
        expected_agentic_outcome="Self-heals: Agent fleet iteratively solves memory (halves batch_size), LR divergence (reduces LR twice), and gradient clipping (enables max_grad_norm=1.0).",
        target_stage="training",
        is_recoverable=True,
    ),
    "catastrophic_label_noise": BenchmarkScenario(
        id="catastrophic_label_noise",
        title="Tier 5 Statistical Trap: 90% Ground-Truth Label Inversion",
        category="statistical_trap",
        description=(
            "Simulates corrupted training labels (90% noise) causing loss to plateau and accuracy to flatline near random guessing (~10%)."
        ),
        fault_type="catastrophic_label_noise",
        initial_overrides={"fault_injection": "catastrophic_label_noise"},
        expected_baseline_outcome="Silent Failure: Completes training blindly with useless weights (~10% accuracy) without alerting operator.",
        expected_agentic_outcome="In-the-loop Diagnosis: TrainingWatchAgent identifies accuracy stagnation and inability to converge, Claude alerts operator to data corruption.",
        target_stage="training",
        is_recoverable=False,
        expected_outcome="escalated",
    ),
    "architecture_tensor_mismatch": BenchmarkScenario(
        id="architecture_tensor_mismatch",
        title="Tier 6 Failure Frontier: Model Architecture Tensor Shape Defect",
        category="structural_code_defect",
        description=(
            "Simulates a non-remediable structural PyTorch bug: linear projection shape mismatch (128x512 vs 1024x10). Cannot be resolved by PipelineConfig mutations."
        ),
        fault_type="architecture_tensor_mismatch",
        initial_overrides={"fault_injection": "architecture_tensor_mismatch"},
        expected_baseline_outcome="Fails: Crashes with PyTorch shape mismatch across retries.",
        expected_agentic_outcome="Safe Escalation: Supervisor and Claude diagnose structural code defect, recognize it cannot be auto-remediated, and safely escalate to engineer.",
        target_stage="training",
        is_recoverable=False,
        expected_outcome="escalated",
    ),
    "thrashing_bimodal_trap": BenchmarkScenario(
        id="thrashing_bimodal_trap",
        title="Tier 7 Control Frontier: Antagonistic Thrashing / Bimodal Optimization Trap",
        category="control_thrashing",
        description=(
            "Simulates an ill-conditioned optimization landscape where any lr >= 0.01 triggers instant NaN explosion, "
            "but any lr < 0.01 triggers instant loss stagnation. The agent enters a paradoxical thrashing loop until retry budget exhaustion."
        ),
        fault_type="thrashing_bimodal_trap",
        initial_overrides={"lr": 0.05, "fault_injection": "thrashing_bimodal_trap"},
        expected_baseline_outcome="Fails: Crashes with NaN divergence on all retries.",
        expected_agentic_outcome="Self-Healing Inability: Agent attempts remediation (reducing LR), which immediately triggers opposite failure (plateau/stagnation), demonstrating the autonomous control frontier.",
        target_stage="training",
        is_recoverable=False,
        expected_outcome="exhausted",
    ),
}


# Scenarios reported in the paper (Levels 1-7) plus the fault-free overhead reference.
PAPER_SUITE: list[str] = [
    "nan_divergence",
    "batch_oom",
    "compound_oom_and_explosion",
    "cascaded_lr_and_memory",
    "delayed_stochastic_explosion",
    "spot_worker_churn",
    "triple_cascade_oom_lr_grad",
    "catastrophic_label_noise",
    "architecture_tensor_mismatch",
    "thrashing_bimodal_trap",
    "clean_run",
]


def get_scenario(scenario_id: str) -> BenchmarkScenario:
    """Retrieve a scenario by ID."""
    if scenario_id not in PRESET_SCENARIOS:
        valid = ", ".join(PRESET_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_id}'. Available scenarios: {valid}")
    return PRESET_SCENARIOS[scenario_id]


def list_scenarios() -> list[BenchmarkScenario]:
    """Return all registered benchmark scenarios."""
    return list(PRESET_SCENARIOS.values())
