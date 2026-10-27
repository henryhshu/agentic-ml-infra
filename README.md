# AutoPARK: Autonomous Agentic Orchestration and Self-Healing for Distributed Machine Learning Infrastructure

> **Architecture**: Split-Plane (In-Container Ray Workers + Claude on AWS Cognitive Orchestrator)  
> **Core Substrate**: The **PARK Stack** (PyTorch · Arrow · Ray · Kubernetes)  
> **Cognitive Engine**: Claude Platform on AWS (`claude-haiku-4-5-20251001` via `anthropic[aws]`)  
> **Target Environment**: Local development & Amazon EKS (`park-ml-cluster`, Spot `c6i.xlarge` nodes)

---

## 🌟 Overview & Scientific Motivation

Traditional machine learning infrastructure relies on **passive observability** (dashboards like Grafana, Prometheus, or W&B that alert engineers to crashes) paired with **blind retry loops** (Kubernetes pod restarts or Airflow retries that re-run identical broken configurations). When complex compounding failures strike—such as memory pressure, learning rate divergence, unclipped gradient shocks, or AWS Spot node preemptions—engineers are paged to manually triage stack traces, waste hours of GPU compute, and edit configurations.

**AutoPARK** bridges this gap by introducing **closed-loop autonomous self-healing with modular actuation plugins**:

1. **Active Actuation ("Agents with Hands")**: Rather than merely reporting errors, AutoPARK agents hold extensible actuation tools that reach across runtime memory, dataset partitions, cluster health, and codebase definitions.
2. **Agent / Execution Separation**: Deterministic detectors run inside the pipeline; Claude (cloud-hosted) is called only to diagnose incidents and choose an allow-listed action, plus one short analysis per epoch.
3. **Live Remediation Across Ranks**: The Remediator mutates rank 0's live `PipelineConfig`; the failed stage re-runs and every DDP rank receives the new config in the next attempt (no pod restart; the stage itself restarts from its beginning).
4. **Multi-Node Training**: the training stage runs as PyTorch DDP (gloo), one rank per node, with a fresh process group per attempt and heartbeat-based detection of lost workers (`pipeline/distributed.py`).
5. **Measured, Not Claimed**: Every MTTR number below is derived from wall-clock instrumentation of stage attempts, Claude calls, and remediations (`timing.json`).

```text
Measured on a 4-node EKS cluster (PyTorch DDP, one rank per node), 7-level testbench, 5 repeats per scenario:
  Levels 1–3  configuration faults + real worker loss → healed 30/30 runs, median MTTR 6.4–19.3 s
  Level 4     triple cascade               → exhausted 5/5 (needs 5 attempts; budget allows 4)
  Levels 5–6  label noise / shape defect   → escalated 10/10
  Level 7     bimodal LR trap              → exhausted 5/5 (as designed)
  No fault    clean run                    → passed 5/5, +50% wall time from per-epoch LLM monitoring
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

# 5. Run an automated benchmark scenario (e.g. Level 1 Batch OOM)
python run_benchmark.py --scenario batch_oom --subset 5000 --epochs 3 --device cpu

# 6. Reproduce the paper results (Levels 1–7 + clean_run, 5 repeats, live Claude calls)
python run_benchmark.py --suite paper --repeats 5 --subset 5000 --epochs 3 --device cpu
python -m benchmarks.plot_analytical_graphs   # figures from the latest results JSON

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

*Note: Default reasoning model is `claude-haiku-4-5-20251001` for cost-efficient triage; measured Claude call latency is 0.6–2.9 s per call (five calls per remediated incident).*

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
| **ClusterHealthAgent** | [`agents/cluster_health.py`](agents/cluster_health.py) | Reads measured DDP rank liveness (worker heartbeats) and host memory; diagnoses worker loss and cluster memory pressure. |
| **DataQualityAgent** | [`agents/data_quality.py`](agents/data_quality.py) | Validates Apache Arrow / Parquet dataset partitions, schema conformance, sample decodability, and class distribution integrity. |
| **RemediatorAgent** | [`agents/remediator.py`](agents/remediator.py) | Executes approved knob mutations directly on live `PipelineConfig` in shared RAM: `reduce_learning_rate`, `reduce_batch_size` (with invariant effective batch Accumulation Guard), `enable_gradient_clipping`, `restart_workers`, `convert_batchnorm_to_groupnorm`, `synthesize_code_patch`. Audit logs every action to `action_log.json`. |

---

## 🧪 The 7-Level Fault Injection Testbench & Measured Results

Faults are defined in `benchmarks/scenarios.py` and fired by `pipeline/training.py`. Injected OOM, NaN, gradient-explosion and shape errors are exceptions raised on every rank while a configuration condition holds (e.g. simulated OOM while `batch_size >= 256`). Worker loss (`spot_worker_churn`) is a real `SIGKILL` of rank 3's process at epoch 1, step 3 of every attempt until `restart_workers` runs; the surviving ranks' collectives fail and rank 0's heartbeat check identifies the lost rank. Label noise (Level 5) is real data corruption.

### Measured results: 4-node Amazon EKS cluster, PyTorch DDP

Cluster: EKS 1.34, `us-east-1d`, four on-demand `c6i.xlarge` nodes, one DDP rank per node (gloo over the VPC network). Workload: TinyCNN (57k params), CIFAR-10 5,000-image subset, 3 epochs, global batch 128 (32 per rank), `max_retries=3`, Claude Haiku 4.5 via Claude Platform on AWS. Source: [`benchmarks/reports/eks/eks-20261027T131911Z/`](benchmarks/reports/eks/eks-20261027T131911Z/) (`bundle/results.json`, per-run traces, per-rank cgroup metrics, pod logs, [`analysis.md`](benchmarks/reports/eks/eks-20261027T131911Z/analysis.md)). MTTR = $T_{\text{det}} + T_{\text{reason}} + T_{\text{act}} + T_{\text{recover}} + T_{\text{other}}$ (definitions in [`benchmarks/harness.py`](benchmarks/harness.py)); for unrecovered runs it is time to termination.

| Level | Scenario | Agentic outcome | $T_{\text{det}}$ s | $T_{\text{reason}}$ s | MTTR median [min–max] s | Baseline | Agentic wall s |
|---|---|---|---|---|---|---|---|
| 1 | `nan_divergence` | healed 5/5 | 0.24 | 12.2 | 12.5 [12.0–14.0] | failed 5/5 (3.1 s) | 29.8 |
| 1 | `batch_oom` | healed 5/5 | 0.21 | 12.9 | 13.2 [13.0–13.7] | failed 5/5 (2.9 s) | 31.3 |
| 2 | `compound_oom_and_explosion` | healed 5/5 | 0.35 | 18.7 | 19.2 [18.3–19.4] | failed 5/5 (2.7 s) | 36.7 |
| 2 | `cascaded_lr_and_memory` | healed 5/5 | 0.35 | 18.8 | 19.3 [18.3–19.9] | failed 5/5 (2.8 s) | 37.1 |
| 3 | `delayed_stochastic_explosion` | healed 5/5 | 0.18 | 6.1 | 6.4 [6.2–6.9] | failed 5/5 (3.1 s) | 24.4 |
| 3 | `spot_worker_churn` (real worker kill) | healed 5/5 | 2.95 | 7.0 | 11.3 [11.1–11.7] | failed 5/5 (14.0 s) | 28.9 |
| 4 | `triple_cascade_oom_lr_grad` | exhausted 5/5 | 0.49 | 24.6 | 25.1 [24.5–25.6] | failed 5/5 (2.9 s) | 29.2 |
| 5 | `catastrophic_label_noise` | escalated 5/5 | 17.82 | 25.5 | 43.2 [42.9–44.0] | silent success 5/5 (12.1 s, acc 0.10–0.16) | 47.3 |
| 6 | `architecture_tensor_mismatch` | escalated 5/5 | 3.17 | 25.6 | 28.5 [27.1–29.2] | failed 5/5 (6.0 s) | 32.4 |
| 7 | `thrashing_bimodal_trap` | exhausted 5/5 | 0.51 | 26.3 | 26.8 [26.3–27.7] | failed 5/5 (3.0 s) | 30.9 |
| – | `clean_run` | passed 5/5 | – | – | – | success 5/5 (11.8 s) | 17.7 |

Key observations:
- **Worker loss**: the surviving ranks' all-reduce fails immediately (`Connection closed by peer`); rank 0's liveness check takes 2.0–2.5 s; the in-pod restart loop brings a replacement process up 1.6–2.4 s after the kill; Claude chose `restart_workers` in 5/5 runs. The replacement is started by the pod's restart loop, not by the agent.
- **LLM reasoning dominates MTTR**: 62% for worker loss, 90–98% for every other exception-type fault; actuation is sub-millisecond for config changes.
- **The cluster idles while agents reason**: incident handling is 51% of agentic wall time, during which all four ranks use ~0 CPU cores.
- Level 4 fails because its repair needs four remediations (five attempts) but the budget allows four. Escalation does not yet halt the retry loop (Level 5 retrains for ~26 s after the first escalation). There is no rule-based remediation baseline yet.

Earlier single-process runs (one laptop, one EKS pod) remain in `benchmarks/reports/` for reference but are not used in the paper.

### Reproduce on EKS

Requires `eksctl`, `kubectl`, the AWS CLI, and `.env` with the Claude Platform on AWS workspace key:

```bash
eksctl create cluster -f eks/cluster.yaml          # 4 on-demand c6i.xlarge nodes (node group dist-cpu), no NAT gateway
WORLD_SIZE=4 eks/launch.sh                         # uploads `git archive HEAD`, creates secret, applies Indexed Job + headless Service
python -m benchmarks.eks_listener --run-tag <RUN_TAG> --job <JOB>   # waits, fetches rank-0 bundle + per-rank metrics, analyzes
python -m benchmarks.plot_analytical_graphs --run-dir benchmarks/reports/eks/<RUN_TAG>
eksctl delete cluster -f eks/cluster.yaml --wait   # stop all charges
```

---

## 🔄 Reversion & Rollback Manifest: Full Auditability & Rapid Rollback

When an autonomous remediation agent intervenes in a training pipeline, engineers must have complete visibility and immediate reversion capability. AutoPARK automatically generates a **Reversion and Rollback Manifest** (`reversion_manifest.json` and human-readable `reversion_manifest.md`) in each run directory.

### The 4-Pillar Incident & Intervention Audit

Every manifest articulates four key questions:
1. 🔴 **Issues Detected**: Complete incident records including the execution stage (`training`, `data_loading`), anomaly classification (e.g. `memory_pressure`), and raw hardware/runtime error signatures.
2. 🛠️ **What Was Changed**: Complete table of mutations performed by the Remediator agent, including exact parameter diffs (e.g. `batch_size: 512 → 256`, `accum_steps: 1 → 2`) and any generated AST patches.
3. 🧠 **Why It Was Changed**: The raw cognitive root-cause diagnosis emitted by Claude Haiku 4.5 on Claude Platform on AWS explaining why the failure occurred and why the remediation was chosen.
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
├── run_benchmark.py           # Benchmark CLI — progressive 7-level fault injection suite
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
├── tests/                     # Unit Tests & Mathematical Parity Testbench (32 tests)
│   ├── conftest.py            # Pytest configuration & environment hooks
│   ├── test_abstractions.py   # Unit tests for DataModule, Model Registry, and AutoParkRunner SDK
│   ├── test_ast_patcher.py    # Unit tests & <50ms AST synthesis performance benchmark
│   ├── test_accumulation_parity.py # Exact mathematical parity proofs for Accumulation Guard
│   ├── test_remediation_safeguards.py # Parameter envelope bounds, retry budgets & safeguard invariants
│   ├── test_reversion_manifest.py  # Unit tests for 4-pillar Reversion & Rollback Manifests
│   └── test_benchmark_timing.py    # MTTR decomposition & outcome classification tests
│
├── benchmarks/                # Fault Injection Benchmark Framework
│   ├── scenarios.py           # Catalog of 14 preset scenarios across Levels 1–7
│   ├── harness.py             # BenchmarkHarness — paired Baseline vs. AutoPARK evaluation
│   ├── report.py              # Markdown & JSON telemetry report generator
│   ├── plot_analytical_graphs.py # Figures rendered only from measured results JSON + run traces
│   ├── plot_architecture_diagrams.py # Architecture diagram (paper Fig. 1)
│   ├── figures/               # High-resolution benchmark vector charts and PDF diagrams
│   └── reports/               # Persisted benchmark reports and execution runs
│
├── paper/                     # USENIX ATC Paper Submission & Artifacts
│   ├── usenix.tex             # LaTeX paper manuscript (7-level testbench, MTTR evaluation)
│   ├── usenix.pdf             # Compiled camera-ready PDF document
│   ├── sample.bib             # Curated bibliographic references and citations
│   └── figures/               # Publication architecture diagrams and analytical plots
│
├── models/                    # Neural Network Backbones
│   ├── resnet.py              # ResNet18 adapted for CIFAR-10 (~11M params)
│   └── tiny_cnn.py            # TinyCNN for rapid smoke testing (~57K params)
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

### ⚡ Cluster Lifecycle: Spin-Down & Spin-Up (Zero Ongoing AWS Costs)

When active experiments are concluded, the Amazon EKS cluster should be spun down to prevent control plane ($0.10/hr) and VPC NAT Gateway (~$0.045/hr) charges from accruing.

#### 🛑 Spin Down (Delete Cluster to Stop All Incurred Costs)
```bash
# 1. Clean up active Ray workloads
kubectl delete raycluster park-ray-cluster -n ray --ignore-not-found
kubectl delete namespace ray --ignore-not-found

# 2. Spin down EKS cluster, node groups, and VPC NAT Gateway
eksctl delete cluster -f aws/eks-cluster.yaml --wait --disable-nodegroup-eviction
```

#### 🚀 Spin Up (Restore Cluster from Scratch)
```bash
# Ensure AWS CLI, eksctl, kubectl, and Helm are installed
export AWS_REGION="us-east-1"
export CLUSTER_NAME="park-ml-cluster"

# 1. Spin up EKS control plane, Spot/GPU node groups, and VPC network (~12-15 min)
eksctl create cluster -f aws/eks-cluster.yaml

# 2. Update local kubeconfig context
aws eks update-kubeconfig --region "${AWS_REGION}" --name "${CLUSTER_NAME}"

# 3. Install NVIDIA GPU device plugin (for GPU worker nodes)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml

# 4. Install KubeRay Operator via Helm
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm upgrade -i kuberay-operator kuberay/kuberay-operator \
    --version 1.1.0 \
    --namespace ray-system \
    --create-namespace

# 5. Deploy Ray Cluster (uses pre-built ECR image)
kubectl apply -f kubernetes/ray-cluster.yaml

# 6. Verify cluster nodes and Ray pods
kubectl get nodes -L nvidia.com/gpu,role
kubectl get pods -n ray
```

*For complete step-by-step instructions on setting up EKS, IAM IRSA roles, and S3 datasets, see [`aws/README.md`](aws/README.md).*

