# AutoPARK: Autonomous Closed-Loop Self-Healing ML Platform with Modular Actuation Plugins

> **Architecture**: Split-Plane (In-Container Ray Workers + Claude on AWS Cognitive Orchestrator)  
> **Core Substrate**: The **PARK Stack** (PyTorch · Arrow · Ray · Kubernetes)  
> **Cognitive Engine**: Claude Platform on AWS (`claude-haiku-4-5-20251001` via `anthropic[aws]`)  
> **Target Environment**: Local development & Amazon EKS (`park-ml-cluster`, Spot `c6i.xlarge` nodes)

---

## 🌟 Overview & Scientific Motivation

Traditional machine learning infrastructure relies on **passive observability** (dashboards like Grafana, Prometheus, or W&B that alert engineers to crashes) paired with **blind retry loops** (Kubernetes pod restarts or Airflow retries that re-run identical broken configurations). When complex compounding failures strike—such as memory pressure, learning rate divergence, unclipped gradient shocks, or AWS Spot node preemptions—engineers are paged to manually triage stack traces, waste hours of GPU compute, and edit configurations.

**AutoPARK** bridges this gap by introducing **closed-loop autonomous self-healing with modular actuation plugins**:

1. **Active Actuation ("Agents with Hands")**: Rather than merely reporting errors, AutoPARK agents hold extensible actuation tools that reach across runtime memory, dataset partitions, cluster health, and codebase definitions.
2. **Split-Plane Architecture**: Decouples deterministic high-throughput execution (in-container Ray/Kubernetes workers) from high-level cognitive reasoning (cloud-hosted LLMs).
3. **Zero Cold-Boot Overhead**: Anomaly remediation occurs directly in shared RAM (`PipelineConfig`), re-instantiating DataLoaders and adjusting optimizers in **15–30 seconds** without requiring Kubernetes pod evictions or container cold reboots.
4. **The Self-Healing Frontier**: Empirically maps the exact boundary where autonomous in-memory self-healing succeeds, where non-remediable structural/statistical defects safely escalate to engineers, and where dynamical optimization traps reach control limits.

```text
               ┌────────────────────────────────────────────────────────┐
               │              AUTOPARK CONTROL ARCHITECTURE             │
               └────────────────────────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌─────────────────────────────────┐                         ┌─────────────────────────────────┐
│     AUTONOMOUSLY REMEDIABLE     │                         │         SAFE ESCALATION         │
│        (Tiers 1, 2, 3, 4)       │                         │          (Tiers 5 & 6)          │
├─────────────────────────────────┤                         ├─────────────────────────────────┤
│ • Batch OOMs                    │                         │ • Statistical Silent Traps      │
│ • Runaway Learning Rates (NaN)  │                         │   (90% Catastrophic Noise)      │
│ • Gradient Explosions           │                         │ • Structural Code Defect        │
│ • EC2 Spot Node Preemptions     │                         │   (Tensor Shape Mismatch)       │
└─────────────────────────────────┘                         └─────────────────────────────────┘
         ▲                                                                   ▲
         │                                                                   │
 [In-RAM Hyperparameter Space]                                 [Safe Operator Triage Space]
```

---

## 🚀 Quick Start

```bash
# 1. Activate your Python environment
conda activate agentic-ml-infra

# 2. Run a standard pipeline smoke test (non-agentic baseline)
python run_pipeline.py --model tiny --epochs 2 --subset 500 --no-tune

# 3. Run with AutoPARK agentic monitoring & closed-loop self-healing
python run_pipeline.py --model tiny --epochs 5 --subset 500 --agentic --no-tune

# 4. Run automated test suite & mathematical accumulation parity proofs
pytest tests/ -v

# 5. Run an automated benchmark scenario (e.g. Tier 1 Batch OOM)
python run_benchmark.py --scenario batch_oom --subset 500 --epochs 1 --device cpu

# 6. Run the complete multi-tier benchmark suite
python run_benchmark.py --all --subset 500 --epochs 1 --device cpu

# 7. Run custom dataset & architecture (e.g. Synthetic Vision with MLP)
python run_pipeline.py --dataset synthetic --model mlp --epochs 1

# 8. Inspect 4-pillar audit of autonomous changes from any past run ID or log
python run_pipeline.py --inspect-run latest

# 9. Roll back to pre-intervention configuration and re-execute pipeline
python run_pipeline.py --rollback-run latest
```

---

## 🛠️ Prerequisites & Setup

| Tool | Minimum Version | Purpose |
|------|-----------------|---------|
| [Conda](https://docs.conda.io/) / Python | Python ≥ 3.10 | Runtime environment |
| PyTorch & torchvision | ≥ 2.3.0 | Deep learning framework |
| Ray | ≥ 2.35.0 | Distributed compute & tuning |
| Anthropic AWS SDK | `anthropic[aws]` | Direct connection to Claude Platform on AWS |

### Install Dependencies

```bash
conda create -n agentic-ml-infra python=3.10 -y
conda activate agentic-ml-infra
pip install -r requirements.txt
```

### Authentication for Claude Platform on AWS

All agents communicate via the **Anthropic AWS SDK** (`AnthropicAWS`) targeting **Claude Platform on AWS**:

```bash
# Add to .env in repository root:
ANTHROPIC_AWS_API_KEY=aws-external-anthropic-api-key-...
ANTHROPIC_AWS_WORKSPACE_ID=wrkspc_...
AWS_REGION=us-east-1
HF_TOKEN=hf_...
```

*Note: Default reasoning model is `claude-haiku-4-5-20251001` for cost-efficient, sub-second cognitive triage.*

---

## 🤖 The AutoPARK Agent Fleet & Actuation Plugins

When `--agentic` is enabled, `PipelineRunner` initializes a specialized agent fleet that watches telemetry in real-time and intervenes when anomalies occur:

```
                    ┌─────────────────────────┐
                    │    SupervisorAgent       │
                    │  • Fleet Orchestration  │
                    │  • Cognitive Synthesis  │
                    │  • Escalation Authority │
                    └────┬──────┬──────┬───────┘
                         │      │      │
              ┌──────────┘      │      └──────────┐
              ▼                 ▼                  ▼
    ┌─────────────────┐ ┌──────────────┐ ┌────────────────┐
    │TrainingWatchAgent│ │ClusterHealth │ │DataQualityAgent│
    │ • Loss / Grads   │ │ • Spot Churn │ │ • Schema       │
    │ • Accuracy Floor │ │ • Workers    │ │ • Partitions   │
    └────────┬────────┘ └──────┬───────┘ └───────┬────────┘
             │                 │                  │
             └─────────┐      │      ┌────────────┘
                       ▼      ▼      ▼
                 ┌─────────────────────────┐
                 │    RemediatorAgent       │
                 │  • In-RAM Knob Mutation │
                 │  • Worker Restart Flags │
                 │  • Full Audit Logging   │
                 └─────────────────────────┘
```

### Agent Roles & Responsibilities

| Agent | Module | Core Responsibilities |
|---|---|---|
| **SupervisorAgent** | [`agents/supervisor.py`](agents/supervisor.py) | Coordinates the sub-agent fleet, routes stage events, calls Claude on AWS for unified root-cause synthesis, enforces `SAFE_REMEDIATIONS` bounds, and records incidents in `incidents.json`. |
| **TrainingWatchAgent** | [`agents/training_watch.py`](agents/training_watch.py) | Watches per-epoch metrics (loss, accuracy, gradient norms). Intercepts NaN/Inf loss, gradient explosions (>100.0), vanishing gradients (<1e-7), loss plateaus, and catastrophic underperformance (accuracy ≤ 15% random chance floor). |
| **ClusterHealthAgent** | [`agents/cluster_health.py`](agents/cluster_health.py) | Monitors KubeRay and AWS EKS cluster health, worker availability, memory pressure, and AWS Spot instance interruptions. |
| **DataQualityAgent** | [`agents/data_quality.py`](agents/data_quality.py) | Validates Apache Arrow / Parquet dataset partitions, schema conformance, sample decodability, and class distribution integrity. |
| **RemediatorAgent** | [`agents/remediator.py`](agents/remediator.py) | Executes approved knob mutations directly on live `PipelineConfig` in shared RAM: `reduce_learning_rate`, `reduce_batch_size` (with invariant effective batch Accumulation Guard), `enable_gradient_clipping`, `restart_workers`, `convert_batchnorm_to_groupnorm`, `synthesize_code_patch`. Audit logs every action to `action_log.json`. |

---

## 🧪 The 7-Tier Fault Injection Matrix & Benchmarks

The AutoPARK testbench evaluates platform efficacy across escalating tiers of failure difficulty, ranging from isolated hyperparameter issues to structural code bugs and dynamical optimization paradoxes:

```text
[ ZONE 1: FULLY AUTONOMOUS SELF-HEALING ]
│
├── Tier 1: Basic Single-Fault (NaN Loss, Batch OOM)
│     └── [PASSED: 100% Healed in 14.2s - 16.6s via LR & Batch Size mutations]
│
├── Tier 2: Multi-Fault Cascades (Compound OOM + Gradient Shock)
│     └── [PASSED: 100% Healed in 24.9s via sequential Batch halving + Grad clipping]
│
├── Tier 3: Complex Deceptive & Infra (Mid-Epoch Stealth Shock, Spot Worker Churn)
│     └── [PASSED: 100% Healed in 9.4s - 17.0s via dynamic clipping & worker restart]
│
└── Tier 4: Triple Cascaded Instability (OOM → Runaway LR → Grad Explosion)
      └── [PASSED: 100% Healed in 31.0s via 4 sequential in-RAM config mutations]
      │
      │ ─── TRANSITION TO OPERATOR ESCALATION ───
      │
[ ZONE 2: SAFE REJECTION & ESCALATION (NON-REMEDIABLE) ]
│
├── Tier 5: Statistical Silent Traps (90% Catastrophic Label Noise)
│     └── [ESCALATED: Caught at epoch boundary via Accuracy Floor; 0 wasted retries]
│
└── Tier 6: Structural Architecture Frontier (PyTorch Tensor Dimension Mismatch)
      └── [ESCALATED: Claude isolates code bug in model definition; 0 invalid mutations]
      │
      │ ─── TRANSITION TO DYNAMICAL CONTROL FRONTIER ───
      │
[ ZONE 3: DYNAMICAL CONTROL FRONTIER (SELF-HEALING BREAKS) ]
│
└── Tier 7: Antagonistic Thrashing / Bimodal Optimization Paradox
      └── [FAILED: LR >= 0.01 causes NaN; LR < 0.01 causes Dead Saddle; budget exhausted]
```

### Empirical Benchmark Results on AWS EKS (2 × Spot `c6i.xlarge` Nodes)

Live test results evaluated against the `park-ml-cluster` running PyTorch 2.3 and KubeRay 1.1.0 on AWS EKS:

| Tier | Scenario ID | Pathology | Baseline Status | AutoPARK Status | Actions Applied | Verdict |
|---|---|---|---|---|---|---|
| **Tier 1** | `nan_divergence` | Excessive LR (`5.0`) &rarr; NaN loss | ❌ Crashed (3/3 retries) | ✅ **Success** (16.6s) | `reduce_learning_rate` × 2 | 🏆 **Self-Healed** |
| **Tier 1** | `batch_oom` | Oversized batch (`512`) &rarr; OOM | ❌ Crashed (3/3 retries) | ✅ **Success** (14.2s) | `reduce_batch_size` × 2 | 🏆 **Self-Healed** |
| **Tier 2** | `compound_oom_and_explosion` | Memory OOM + gradient explosion | ❌ Crashed (4/4 retries) | ✅ **Success** (24.9s) | `reduce_batch_size` × 2, `enable_gradient_clipping` | 🏆 **Self-Healed** |
| **Tier 2** | `cascaded_lr_and_memory` | Batch OOM (`256`) + Runaway LR (`5.0`) | ❌ Crashed (4/4 retries) | ✅ **Success** (28.4s) | `reduce_batch_size`, `reduce_learning_rate` × 2 | 🏆 **Self-Healed** |
| **Tier 3** | `delayed_stochastic_explosion` | Step 1 normal (2.30), Step 2 spike (842.1) | ❌ Crashed (3/3 retries) | ✅ **Success** (9.4s) | `enable_gradient_clipping` (norm=1.0) | 🏆 **Self-Healed** |
| **Tier 3** | `spot_worker_churn` | AWS Spot preemption (`workers=0`) | ❌ Crashed (3/3 retries) | ✅ **Success** (17.0s) | `restart_workers` | 🏆 **Self-Healed** |
| **Tier 4** | `triple_cascade_oom_lr_grad` | Memory OOM &rarr; Runaway LR &rarr; Grad shock | ❌ Crashed (4/4 retries) | ✅ **Success** (31.0s) | 4 sequential in-RAM mutations | 🏆 **Self-Healed** |
| **Tier 5** | `catastrophic_label_noise` | 90% label noise (loss ~2.3, acc ~8%) | ⚠️ Silent Waste (2.9s) | 🛡️ **Escalated** (36.3s) | Refused hyperparameter mutation | 🛡️ **Diagnosed Early** |
| **Tier 6** | `architecture_tensor_mismatch` | Layer mismatch (`128×512` vs `1024×10`) | ❌ Crashed (3/3 retries) | 🛡️ **Escalated** (27.8s) | Refused invalid mutation | 🛡️ **Diagnosed Early** |
| **Tier 7** | `thrashing_bimodal_trap` | $LR \ge 0.01 \implies \text{NaN}$; $LR < 0.01 \implies \text{Saddle}$ | ❌ Crashed (3/3 retries) | ❌ **Failed** (30.5s) | Thrashed across opposing bounds | ⚠️ **Control Limit** |

---

## 🔄 Reversion & Rollback Manifest: Full Auditability & Rapid Rollback

When an autonomous remediation agent intervenes in a training pipeline, engineers must have complete visibility and immediate reversion capability. AutoPARK automatically generates a **Reversion and Rollback Manifest** (`reversion_manifest.json` and human-readable `reversion_manifest.md`) in each run directory.

### The 4-Pillar Incident & Intervention Audit

Every manifest articulates four key questions:
1. 🔴 **Issues Detected**: Complete incident records including the execution stage (`training`, `data_loading`), anomaly classification (e.g. `memory_pressure`), and raw hardware/runtime error signatures.
2. 🛠️ **What Was Changed**: Complete table of mutations performed by the Remediator agent, including exact parameter diffs (e.g. `batch_size: 512 → 256`, `accum_steps: 1 → 2`) and any generated AST patches.
3. 🧠 **Why It Was Changed**: The raw cognitive root-cause diagnosis emitted by Claude Haiku 4.5 on AWS Bedrock explaining why the failure occurred and why the remediation was chosen.
4. 🔄 **How To Change It Back**:
   - **One-Command CLI Rollback**: Exact command to rerun the pipeline restoring the pre-intervention state.
   - **Manual Python Override**: Code snippet to set original parameters on `PipelineConfig`.
   - **Restoration Table**: Exact side-by-side comparison of mutated vs. restored parameters.

### Multi-Hop Chained Mutation Resolution

If the agent undertakes multiple chained interventions in a single run (e.g., halving batch size twice: $512 \to 256 \to 128$), an inverse of the last action would only roll back to $256$. AutoPARK tracks chronological parameter history to resolve the **earliest known pre-intervention baseline** ($512$), ensuring that rollback always restores the engineer's true intended configuration.

### CLI Usage & Backward Compatibility with Old Run IDs

AutoPARK dynamically constructs manifests from old log artifacts (`action_log.json`, `incidents.json`, `config.json`) for historical runs:

```bash
# Inspect the 4-pillar audit of the most recent run (or any run ID)
python run_pipeline.py --inspect-run latest
python run_pipeline.py --inspect-run run_20261012_213426_5cae

# Execute a rollback run: restores original parameters and re-executes
python run_pipeline.py --rollback-run latest
python run_pipeline.py --rollback-run run_20261012_213426_5cae
```

---

## 🔌 Friction-Minimizing Pluggable Abstractions (DataModule, Registry & SDK)

AutoPARK is designed so any ML engineer can adopt the platform for their custom dataset (vision, NLP, audio, tabular) and model architecture with zero pipeline rewrites:

### 1. The `DataModule` Abstraction (`pipeline/data_module.py`)
Decouples data validation, shard reading, and DataLoader construction:
- `BaseDataModule`: Abstract interface defining `setup()`, `validate_schema()`, `train_dataloader()`, `val_dataloader()`.
- `CIFAR10DataModule`: Production Parquet CIFAR-10 data loader with zero-copy caching.
- `SyntheticVisionDataModule`: Zero-network, offline synthetic vision data generator for deterministic testing.
- `CustomDataModule`: Wraps arbitrary PyTorch `Dataset`s with custom schema validators.

### 2. Pluggable Model Registry (`pipeline/registry.py`)
Eliminates hardcoded model dictionaries:
```python
from autopark import register_model

@register_model("my_vit")
def build_custom_vit(num_classes=10, **kwargs):
    return timm.create_model("vit_base_patch16_224", num_classes=num_classes)
```

### 3. The 10-Line Developer SDK (`AutoParkRunner`)
Wrap pre-existing PyTorch models and DataLoaders directly:
```python
import torch
from autopark import AutoParkRunner

runner = AutoParkRunner.from_loaders(
    model=my_model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=10,
    epochs=5,
    enable_agentic=True,  # Enables closed-loop Claude self-healing & AST patching
)
results = runner.run()
```

---

## 📁 Repository Structure

```
agentic-ml-infra/
├── run_pipeline.py            # Primary CLI — full 5-stage AutoPARK pipeline (--inspect-run, --rollback-run, --dataset)
├── run_benchmark.py           # Benchmark CLI — progressive fault injection suite
├── main.py                    # Standalone single-node training loop
│
├── autopark/                  # High-level Python SDK
│   └── __init__.py            # AutoParkRunner, from_loaders convenience builder, exports
│
├── pipeline/                  # Pipeline stage implementations
│   ├── config.py              # PipelineConfig (shared RAM knob-actuation interface)
│   ├── runner.py              # PipelineRunner — stage lifecycle & agentic retry loop
│   ├── data_module.py         # BaseDataModule, CIFAR10, SyntheticVision, CustomDataModule
│   ├── registry.py            # Model registry (@register_model, get_model, list_models)
│   ├── reversion.py           # Reversion & Rollback Manifest generator, multi-hop resolver, and CLI inspector
│   ├── data_validation.py     # Stage 1: Schema & sample decodability (pluggable via DataModule)
│   ├── data_loading.py        # Stage 2: DataLoader builder & batch allocation (pluggable via DataModule)
│   ├── training.py            # Stage 3: PyTorch training loop with in-loop triggers & model registry
│   ├── tuning.py              # Stage 4: Ray Tune ASHA hyperparameter search
│   └── evaluation.py          # Stage 5: Final evaluation with dynamic class naming
│
├── agents/                    # AutoPARK Agent Monitoring & Actuation Fleet
│   ├── base.py                # AgentBase ABC — AnthropicAWS integration & LLM calls
│   ├── supervisor.py          # SupervisorAgent — cognitive routing & escalation
│   ├── training_watch.py      # TrainingWatchAgent — real-time loss/gradient/acc gates
│   ├── cluster_health.py      # ClusterHealthAgent — Ray & AWS Spot preemption monitor
│   ├── data_quality.py        # DataQualityAgent — schema & partition validator
│   ├── remediator.py          # RemediatorAgent — shared-RAM knob mutator & audit log
│   ├── checkpoints.py         # CheckpointEmitter & AgentCheckpoint schemas
│   └── plugins/               # Modular Actuation Plugins ("Agents with Hands")
│       └── ast_patcher.py     # ASTCodeSynthesizer — BatchNorm->GroupNorm & dimension repair
│
├── tests/                     # Unit Tests & Mathematical Parity Testbench
│   ├── conftest.py            # Pytest configuration & environment hooks
│   ├── test_abstractions.py   # Unit tests for DataModule, Model Registry, and AutoParkRunner SDK
│   ├── test_ast_patcher.py    # Unit tests & <50ms AST synthesis performance benchmark
│   ├── test_accumulation_parity.py # Exact mathematical parity proofs for Accumulation Guard
│   └── test_reversion_manifest.py  # Unit tests for 4-pillar Reversion & Rollback Manifests
│
├── benchmarks/                # Fault Injection Benchmark Framework
│   ├── scenarios.py           # Catalog of 14 preset scenarios across Tiers 1–7
│   ├── harness.py             # BenchmarkHarness — paired Baseline vs. AutoPARK evaluation
│   ├── report.py              # Markdown & JSON telemetry report generator
│   └── reports/               # Persisted benchmark reports and execution runs
│
├── models/                    # Neural Network Backbones
│   ├── resnet.py              # ResNet18 adapted for CIFAR-10 (~11M params)
│   └── tiny_cnn.py            # TinyCNN for rapid smoke testing (~40K params)
│
├── aws/                       # Production AWS Cloud Deployment
│   ├── README.md              # Complete AWS EKS & KubeRay deployment guide
│   ├── eks-cluster.yaml       # eksctl cluster specification (Spot c6i.xlarge nodes)
│   ├── iam_roles.yaml         # CloudFormation template for IRSA service accounts
│   ├── s3_config.py           # S3 dataset & experiment metrics store
│   └── ecr_push.sh            # Container build & ECR push automation
│
├── kubernetes/                # Kubernetes manifests
│   ├── ray-cluster.yaml       # KubeRay RayCluster CRD configuration
│   └── pipeline-job.yaml      # Kubernetes Job specification for AutoPARK
│
└── experiments/               # Structured per-run results, logs, and checkpoints
```

---

## ☁️ Local → AWS Cloud Deployment

AutoPARK runs identically on a local developer laptop and on AWS EKS:

| Feature | Local Environment | Production AWS EKS |
|---|---|---|
| **Compute Hardware** | Apple Silicon (MPS) or CPU | Amazon EC2 Spot (`c6i.xlarge` / `g4dn.xlarge`) |
| **Kubernetes Substrate** | `kind` / `minikube` | Amazon EKS (`park-ml-cluster`) |
| **Distributed Ray** | In-process Ray instance | KubeRay Operator (`RayCluster` CRD) |
| **Storage Backend** | Local disk cache (`data/`) | Amazon S3 Parquet Shards via Arrow S3FS |
| **Cognitive Agent LLM** | `AnthropicAWS` via `.env` | `AnthropicAWS` with EKS IRSA credentials |

*For complete step-by-step instructions on setting up EKS, IAM IRSA roles, and S3 datasets, see [`aws/README.md`](aws/README.md).*
