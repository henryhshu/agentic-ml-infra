# Agentic ML Training Platform — PARK Stack

A self-healing ML training infrastructure built on the **PARK stack** (PyTorch · Arrow · Ray · Kubernetes), supervised by a fleet of **Claude-powered agents** served through the **Anthropic AWS SDK** (`AnthropicAWS`) on **Amazon Bedrock**.

The agent fleet launches alongside every training job, watches the pipeline end-to-end (metrics, data integrity, cluster health), triages anomalies using Claude Haiku 4.5 in-the-loop, and applies scoped remediations — so minor breakages self-heal without paging a human.

---

## 🚀 Quick Start

```bash
# 1. Activate the Conda environment
conda activate agentic-ml-infra

# 2. Run a quick pipeline smoke test (non-agentic, tiny model, small subset)
python run_pipeline.py --model tiny --epochs 2 --subset 500 --no-tune

# 3. Run with agentic monitoring (requires AWS credentials — see Configuration)
python run_pipeline.py --model tiny --epochs 5 --subset 500 --agentic --no-tune
```

---

## 🛠️ Prerequisites

| Tool | Purpose |
|------|---------|
| [Conda](https://docs.conda.io/) (Miniforge/Miniconda) | Python environment management |
| Python ≥ 3.10 | Runtime |
| AWS credentials (for agentic mode) | `ANTHROPIC_AWS_API_KEY` + `ANTHROPIC_AWS_WORKSPACE_ID` |

### Install Dependencies

```bash
conda create -n agentic-ml-infra python=3.10 -y
conda activate agentic-ml-infra
pip install -r requirements.txt
```

**`requirements.txt`:**
```
torch
torchvision
pyarrow
huggingface_hub
pillow
ray[train,tune]
anthropic[aws]
python-dotenv
boto3
```

---

## ☁️ Claude on AWS — LLM Configuration

All agent LLM calls are routed through the **Anthropic AWS SDK** (`from anthropic import AnthropicAWS`). The client is initialized centrally in [`agents/base.py`](agents/base.py) and shared by every agent via inheritance.

### Authentication Priority

The `AgentBase._init_anthropic_client()` method selects a client in this order:

| Priority | Env Variable | Client |
|----------|-------------|--------|
| 1 | `ANTHROPIC_AWS_API_KEY` | `AnthropicAWS(api_key=..., workspace_id=..., aws_region=...)` |
| 2 | `AWS_ACCESS_KEY_ID` or `CLAUDE_CODE_USE_BEDROCK=1` | `AnthropicAWS(aws_region=...)` (uses IAM/STS) |
| 3 | `ANTHROPIC_API_KEY` | `anthropic.Anthropic(api_key=...)` (direct API fallback) |

### Required Environment Variables

Create a `.env` file (gitignored) in the project root:

```bash
ANTHROPIC_AWS_API_KEY=aws-external-anthropic-api-key-...
ANTHROPIC_AWS_WORKSPACE_ID=wrkspc_...
AWS_REGION=us-east-1
HF_TOKEN=hf_...
```

### Default Model

All agents default to **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) for cost-efficient in-the-loop analysis. Override with `ANTHROPIC_MODEL` or `CLAUDE_MODEL` env vars.

### Graceful Degradation

If no LLM credentials are configured, agents automatically fall back to **rule-based analysis** — the pipeline runs identically, but agent diagnoses use heuristic rules instead of Claude.

---

## 📂 Project Structure

```
agentic-ml-infra/
├── run_pipeline.py            # 🟢 Primary CLI — full 5-stage pipeline
├── run_benchmark.py           # 🧪 Benchmark CLI — fault injection & efficacy suite
├── main.py                    # Standalone single-node training (no pipeline stages)
│
├── pipeline/                  # Pipeline stage implementations
│   ├── config.py              # PipelineConfig dataclass (all hyperparams & flags)
│   ├── runner.py              # PipelineRunner — orchestrates stages + retry logic
│   ├── data_validation.py     # Stage 1: Parquet schema/row/decode validation
│   ├── data_loading.py        # Stage 2: Downloads data, builds DataLoaders
│   ├── training.py            # Stage 3: Training loop with anomaly detection
│   ├── tuning.py              # Stage 4: Ray Tune ASHA hyperparameter search
│   └── evaluation.py          # Stage 5: Final eval (per-class acc, confusion matrix)
│
├── agents/                    # Claude-powered monitoring agent fleet
│   ├── base.py                # AgentBase ABC — AnthropicAWS client, _call_llm()
│   ├── supervisor.py          # SupervisorAgent — orchestrates fleet, routes events
│   ├── training_watch.py      # TrainingWatchAgent — loss/acc/gradient anomalies
│   ├── cluster_health.py      # ClusterHealthAgent — Ray cluster & pod status
│   ├── data_quality.py        # DataQualityAgent — validates data pipeline
│   ├── remediator.py          # RemediatorAgent — gated config mutations
│   └── checkpoints.py         # AgentCheckpoint schema + CheckpointEmitter
│
├── benchmarks/                # Fault injection benchmarking framework
│   ├── scenarios.py           # 6 preset fault scenarios (see Experiments section)
│   ├── harness.py             # BenchmarkHarness — runs baseline vs agentic
│   └── report.py              # Terminal/Markdown/JSON report generation
│
├── models/                    # Model architectures
│   ├── resnet.py              # ResNet18 with CIFAR-friendly stem (~11M params)
│   └── tiny_cnn.py            # TinyCNN for fast smoke tests (~40K params)
│
├── datasets/
│   └── cifar10.py             # CIFAR10Parquet Dataset + HF Hub parquet downloader
│
├── metrics/                   # Experiment tracking & structured schemas
│   ├── experiment.py          # ExperimentTracker (per-run JSON persistence)
│   └── schemas.py             # StageMetrics, TrainingMetrics, EvaluationMetrics, etc.
│
├── aws/                       # AWS deployment manifests & guides
│   ├── README.md              # Full EKS/Bedrock deployment walkthrough
│   ├── eks-cluster.yaml       # eksctl cluster config
│   ├── iam_roles.yaml         # CloudFormation IRSA template
│   ├── s3_config.py           # S3Config + S3MetricsStore helpers
│   └── ecr_push.sh            # ECR build & push script
│
├── docker/                    # Container images
│   ├── Dockerfile.pytorch     # PyTorch base image
│   ├── Dockerfile.cifar10     # Application image
│   └── docker-compose.yml     # Multi-container orchestration
│
├── kubernetes/
│   └── park-cluster.yaml      # Namespace + ServiceAccount
│
├── ray/
│   ├── cifar10-train.py       # Ray Train distributed training script
│   └── ray-cluster.yaml       # KubeRay RayCluster CRD
│
├── experiments/               # Auto-generated per-run results (JSON)
├── data/                      # Local parquet cache
├── .env                       # Credentials (gitignored)
└── requirements.txt
```

---

## ⚙️ Execution Pipeline

The platform uses a **5-stage pipeline** orchestrated by [`PipelineRunner`](pipeline/runner.py). Each stage has automatic retry logic (default: 3 retries) and, when agentic mode is enabled, Claude-powered anomaly detection and remediation between retries.

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PipelineRunner.run()                        │
│                                                                    │
│   Stage 1         Stage 2         Stage 3         Stage 4          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│  │   Data   │──▶│   Data   │──▶│ Training │──▶│  Tuning  │──┐    │
│  │Validation│   │ Loading  │   │          │   │(optional)│  │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │    │
│                                                              ▼    │
│                                                     Stage 5       │
│                                                   ┌──────────┐    │
│                                                   │Evaluation│    │
│                                                   └──────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

| Stage | Module | What It Does |
|-------|--------|-------------|
| **1. Data Validation** | `pipeline/data_validation.py` | Downloads CIFAR-10 parquet shards from HF Hub; validates Arrow schema (`img`, `label` columns), row counts (50K train / 10K test), and sample image decoding. Computes a data quality score. |
| **2. Data Loading** | `pipeline/data_loading.py` | Builds train/eval transforms (RandomCrop, HorizontalFlip, Normalize); wraps parquet data in `CIFAR10Parquet` PyTorch `Dataset`; creates `DataLoader` instances with configured batch size. |
| **3. Training** | `pipeline/training.py` | Trains the selected model (ResNet18 or TinyCNN) with SGD + CosineAnnealingLR. Per-epoch: logs train loss, test accuracy, gradient norms. Detects NaN/Inf loss, gradient explosions, and supports fault injection. Routes epoch metrics to the agent fleet. |
| **4. Tuning** *(optional)* | `pipeline/tuning.py` | Runs Ray Tune hyperparameter search over `lr`, `batch_size`, `weight_decay` using an ASHA scheduler for early stopping. Enabled via `--tune` flag. |
| **5. Evaluation** | `pipeline/evaluation.py` | Final model assessment: overall accuracy, per-class accuracy breakdown, confusion matrix, inference throughput (samples/sec). Routes results to agent fleet. |

### Entry Points

| Script | Purpose | Example |
|--------|---------|---------|
| `run_pipeline.py` | **Primary CLI** — runs the full 5-stage pipeline | `python run_pipeline.py --model tiny --epochs 5 --agentic` |
| `run_benchmark.py` | **Benchmark CLI** — fault injection & efficacy comparison | `python run_benchmark.py --all --subset 500` |
| `main.py` | **Standalone training** — minimal loop, no pipeline stages | `python main.py --model resnet18 --epochs 20` |

### Key CLI Flags (`run_pipeline.py`)

```bash
python run_pipeline.py \
    --model tiny|resnet18       # Model architecture (default: tiny)
    --epochs 5                  # Training epochs (default: 5)
    --batch-size 128            # Batch size (default: 128)
    --lr 0.1                    # Learning rate (default: 0.1)
    --device cpu|mps|cuda       # Compute device (default: mps)
    --agentic / --no-agentic    # Toggle agent monitoring (default: off)
    --tune / --no-tune          # Toggle HP tuning stage (default: off)
    --tune-samples 4            # Number of Ray Tune trials (default: 4)
    --subset 500                # Use N samples for fast testing
    --max-retries 3             # Auto-retries per stage (default: 3)
```

### Default Device: MPS (Apple Silicon)

The platform defaults to **MPS** (Metal Performance Shaders) for GPU-accelerated training on Apple Silicon Macs. Override with `--device cpu` or `--device cuda` for other hardware.

---

## 🤖 Agentic Monitoring Layer

When `--agentic` is enabled, `PipelineRunner` initializes a fleet of 5 Claude-powered agents that run in-process alongside the training job.

### Agent Fleet Architecture

```
                    ┌─────────────────────────┐
                    │    SupervisorAgent       │
                    │  (orchestrates fleet,    │
                    │   routes events,         │
                    │   approves remediations) │
                    └────┬──────┬──────┬───────┘
                         │      │      │
              ┌──────────┘      │      └──────────┐
              ▼                 ▼                  ▼
    ┌─────────────────┐ ┌──────────────┐ ┌────────────────┐
    │TrainingWatchAgent│ │ClusterHealth │ │DataQualityAgent│
    │ loss/acc/grads   │ │ pods/workers │ │ schema/decode  │
    └────────┬────────┘ └──────┬───────┘ └───────┬────────┘
             │                 │                  │
             └─────────┐      │      ┌────────────┘
                       ▼      ▼      ▼
                 ┌─────────────────────────┐
                 │    RemediatorAgent       │
                 │  (gated config mutations │
                 │   + audit logging)       │
                 └─────────────────────────┘
```

### Agent Roles

| Agent | Watches | Capabilities |
|-------|---------|-------------|
| **SupervisorAgent** | All sub-agents | Routes events to specialized agents; decides auto-remediate vs. human escalation; persists incident log (`incidents.json`). |
| **TrainingWatchAgent** | Per-epoch loss, accuracy, gradient norms | Detects NaN/Inf loss, loss plateaus (5-epoch window), accuracy regressions (>5% drop), gradient explosions (norm > 100). Provides **Claude in-the-loop epoch analysis** via `analyze_epoch()`. |
| **ClusterHealthAgent** | Cluster status events | Monitors Ray worker availability, pod restarts, task failures, memory pressure. |
| **DataQualityAgent** | Data pipeline events | Flags schema mismatches, corrupt images, label imbalance, missing columns. |
| **RemediatorAgent** | Supervisor-approved actions only | Executes scoped config mutations: `reduce_learning_rate` (÷10), `reduce_batch_size` (÷2), `enable_gradient_clipping` (norm=1.0), `restart_workers`, `retry_stage`. Every action is audit-logged to `action_log.json`. |

### How Remediation Works

1. A pipeline stage throws an exception (e.g., NaN loss, OOM).
2. `PipelineRunner._run_stage_with_retry()` catches the error and calls `SupervisorAgent.handle_anomaly()`.
3. The Supervisor normalizes the error string to canonical anomaly types (e.g., `"nan_loss_detected"`, `"memory_pressure"`).
4. Sub-agents produce rule-based + Claude-powered diagnoses.
5. If the anomaly is in `SAFE_REMEDIATIONS`, the Supervisor approves auto-remediation → the RemediatorAgent **mutates the live `PipelineConfig`** (e.g., halves `batch_size`).
6. The pipeline retries the stage with the updated config.

### Safe Auto-Remediation Policies

| Anomaly | Remediation Action | Config Change |
|---------|--------------------|---------------|
| `nan_loss_detected` | `reduce_learning_rate` | `lr *= 0.1` |
| `gradient_explosion` | `enable_gradient_clipping` | `max_grad_norm = 1.0` |
| `memory_pressure` | `reduce_batch_size` | `batch_size //= 2` |
| `loss_plateau` | `reduce_learning_rate` | `lr *= 0.1` |
| `worker_shortage` | `restart_workers` | (logged, no local restart) |

Anomalies **not** in the safe list (e.g., `corrupt_data`, `schema_error`) are escalated — the agent fleet provides a structured diagnosis but does not auto-remediate.

---

## 🧪 Experiment & Benchmark Setups

The benchmarking framework compares **Standard (baseline)** vs **Agentic (Claude in-the-loop)** pipeline runs across controlled fault injection scenarios. Each scenario runs the pipeline twice — once without agents, once with — to measure self-healing efficacy.

### Implemented Scenarios

All 6 scenarios are defined in [`benchmarks/scenarios.py`](benchmarks/scenarios.py):

| # | ID | Title | Category | Fault Injection | Recoverable? |
|---|-----|-------|----------|----------------|-------------|
| 1 | `nan_divergence` | Learning Rate Divergence (NaN Loss) | `training_instability` | Sets `lr=5.0` causing NaN on step 1 | ✅ Yes — agent reduces LR by 10× |
| 2 | `gradient_explosion` | Unclipped Gradient Shock | `training_instability` | Simulates grad_norm=999.9 without clipping | ✅ Yes — agent enables `max_grad_norm=1.0` |
| 3 | `batch_oom` | Memory Pressure (Simulated OOM) | `resource_pressure` | Sets `batch_size=512` triggering OOM error | ✅ Yes — agent halves batch size iteratively |
| 4 | `corrupt_data` | Data Partition Corruption | `data_quality` | Simulates schema mismatch in data validation | ❌ No — agent diagnoses, pipeline safely halts |
| 5 | `loss_plateau` | Optimizer Stagnation | `optimizer_stagnation` | Sets `lr=1e-8` causing zero convergence | ✅ Yes — agent flags stagnant trajectory |
| 6 | `clean_run` | Reference Baseline | `baseline` | None — healthy run with optimal HPs | ✅ Passes — confirms agentic overhead is minimal |

### Expected Outcomes

| Scenario | Baseline (no agents) | Agentic (Claude in-the-loop) |
|----------|---------------------|-----------------------------|
| **NaN Divergence** | ❌ Crashes with NaN loss across all retries | ✅ Self-heals: detects NaN, reduces LR, retry succeeds |
| **Gradient Explosion** | ❌ Crashes with gradient explosion error | ✅ Self-heals: enables gradient clipping, retry succeeds |
| **Batch OOM** | ❌ Crashes with OOM on every attempt | ✅ Self-heals: halves batch size (512→256→128), retry succeeds |
| **Corrupt Data** | ❌ Data validation crashes, retries fail blindly | 🔍 Early diagnosis: identifies schema mismatch, safely halts |
| **Loss Plateau** | ⚠️ Silent failure: completes but ~10% accuracy | 🔍 In-the-loop: flags stagnant loss in real-time |
| **Clean Run** | ✅ Completes successfully | ✅ Completes with Claude telemetry confirming healthy convergence |

### Running Benchmarks

```bash
# List all available scenarios
python run_benchmark.py --list

# Run all 6 scenarios
python run_benchmark.py --all --subset 500 --epochs 2

# Run a specific scenario
python run_benchmark.py --scenario nan_divergence

# Run with custom settings and save report
python run_benchmark.py --all --subset 300 --epochs 2 --output report.md --json-output results.json
```

Benchmark results are saved to `experiments/benchmarks/` and a summary report is generated at `benchmarks/LATEST_BENCHMARK.md`.

### Efficacy Metrics Collected

- **Self-healing rate**: % of recoverable scenarios where agentic mode succeeded after baseline failed.
- **Compute saved**: Wasted baseline retry time that agentic mode avoided.
- **Remediations applied**: Count and type of autonomous config mutations.
- **Claude diagnoses**: LLM-generated root-cause analyses.

---

## 🧩 Stack Components

| Component | Role |
|-----------|------|
| **PyTorch** | Model definition (ResNet18, TinyCNN), training loop, SGD + CosineAnnealingLR |
| **Apache Arrow / PyArrow** | Columnar in-memory format; reads CIFAR-10 parquet shards from Hugging Face Hub |
| **Ray Train & Tune** | Distributed training (planned for multi-node); ASHA-based hyperparameter search |
| **Kubernetes + KubeRay** | Container orchestration and `RayCluster` CRD management (production path) |
| **Anthropic AWS SDK** | `AnthropicAWS` client for Claude on Bedrock; all agent LLM calls |
| **ExperimentTracker** | Structured JSON persistence of per-run metrics, events, and stage results |

---

## 📊 Experiment Output

Each pipeline run produces a structured results directory under `experiments/<run-id>/`:

```
experiments/<run-id>/
├── summary.json           # Overall run status, duration, config
├── data_validation.json   # Quality score, schema checks, row counts
├── data_loading.json      # Load time, sample counts
├── training.json          # Per-epoch loss/acc/grad norms, best accuracy
├── tuning.json            # Trial configs + results (if --tune enabled)
├── evaluation.json        # Final accuracy, per-class breakdown, confusion matrix
├── incidents.json         # Agent anomaly/remediation audit trail (if --agentic)
├── action_log.json        # Remediator action history (if --agentic)
└── checkpoints/           # Agent checkpoint JSONL stream (if --agentic)
```

---

## ☁️ Local → Cloud Path

The same pipeline code runs locally and on AWS. Only the surrounding infrastructure changes.

| Concern | Local (Mac) | Production (AWS) |
|---------|-------------|------------------|
| Compute | MPS (Apple Silicon) or CPU | EKS + NVIDIA GPU node groups (g4dn, g5, p4d) |
| Kubernetes | `kind` / `minikube` | Amazon EKS |
| KubeRay | Helm install on local kube | Helm install on EKS |
| Dataset storage | Local parquet cache (`data/cifar10/`) | S3 bucket via `pyarrow.fs.S3FileSystem` |
| Agent LLM calls | `AnthropicAWS` (API key in `.env`) | `AnthropicAWS` via IAM/IRSA (no static keys) |
| Container registry | Docker Desktop | Amazon ECR |
| Secrets | `.env` file (gitignored) | AWS Secrets Manager + IRSA |
| Observability | stdout + JSON experiment logs | CloudWatch Logs + Container Insights |

For the full AWS deployment guide (EKS, ECR, IRSA, Bedrock, CloudWatch), see [`aws/README.md`](aws/README.md).

---

## 📚 Documentation & References

-   [Anthropic AWS SDK (`AnthropicAWS`)](https://docs.anthropic.com/en/api/claude-on-amazon-bedrock)
-   [Claude on Amazon Bedrock](https://aws.amazon.com/bedrock/claude/)
-   [KubeRay Operator](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
-   [Ray Train (TorchTrainer)](https://docs.ray.io/en/latest/train/train.html)
-   [Ray Tune (ASHA Scheduler)](https://docs.ray.io/en/latest/tune/index.html)
-   [PyTorch Documentation](https://pytorch.org/docs/)
-   [Apache Arrow / PyArrow](https://arrow.apache.org/docs/)
-   [Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/what-is-eks.html)
