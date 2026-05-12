# Agentic ML with the PARK Stack

This capstone project aims to create a self-reliant infrastructure for distributed model training on the **PARK stack**. This stack leverages PyTorch for model development, Arrow for data interoperability, Ray for distributed execution, and Kubernetes for container orchestration.

Wrapped around the stack is a fleet of Claude-powered agents — built with the [Claude Agent SDK](https://docs.anthropic.com/en/api/agent-sdk/overview) — that launch alongside every training job, watch the cluster end-to-end (logs, metrics, Ray actors, Kubernetes pods, parquet shards), triage anomalies, and apply scoped remediations so minor breakages resolve without paging a human.

The same code path runs locally on a MacBook (Docker Desktop + `kind`/`minikube`) for development and on a managed cloud cluster (GKE on GCP) for production training.

## 🚀 Quick Start

1.  **Activate the environment**:
    ```bash
    conda activate agentic-ml-infra
    ```

2.  **Run the demo**:
    ```bash
    python main.py
    ```

## 🛠️ Prerequisites

-   [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) (Miniconda/Anaconda)
-   [Docker Desktop](https://www.docker.com/products/docker-desktop/)

## 📂 Project Structure

```
agentic-ml-infra/
├── docker/
│   ├── Dockerfile.cifar10     # CIFAR-10 trainer image
│   ├── Dockerfile.pytorch     # Generic PyTorch base image
│   └── docker-compose.yml     # Multi-container orchestration
├── kubernetes/
│   └── park-cluster.yaml      # Namespace + service account for Ray
├── ray/
│   ├── cifar10-train.py       # Ray Train distributed training script
│   └── ray-cluster.yaml       # KubeRay RayCluster CRD spec
├── models/
│   └── resnet.py              # ResNet18 with a CIFAR-friendly stem
├── datasets/
│   └── cifar10.py             # pyarrow loader for HF CIFAR-10 parquet shards
├── data/
│   └── cifar10/               # CIFAR-10 parquet cache (downloaded by script)
├── agents/                    # (planned) Claude Agent SDK monitoring fleet
│   ├── supervisor.py          # Top-level orchestrator launched with each job
│   ├── training_watch.py      # Tails loss/acc stream; catches NaN, divergence, stalls
│   ├── cluster_health.py      # Watches KubeRay CRDs, pod restarts, OOMs
│   ├── data_quality.py        # Validates parquet/Arrow schemas + row counts
│   ├── remediator.py          # Gated write actions (restart, rescale, rerun)
│   └── tools/                 # Tool wrappers exposed to agents (kubectl, Ray, GCS, parquet)
├── main.py                    # Single-node training entry point
├── requirements.txt           # Python dependencies
└── README.md
```

## 🧩 Stack Components

### 🔴 PyTorch
Deep learning framework for model definition and training.

### 🟡 Arrow
In-memory columnar format for efficient data exchange.

### 🟢 Ray
Distributed computing framework for scaling ML workloads.

### 🔵 Kubernetes
Container orchestration for cluster management.

### 🟣 Claude Agent SDK
Defines the monitoring agents — their system prompts, tool surfaces, and decision loops. Agents run as Python processes locally and as Kubernetes `Job`s in production. See [Agentic Monitoring Layer](#-agentic-monitoring-layer) below.

## 🤖 Agentic Monitoring Layer

The PARK stack is observed and partially self-healed by a fleet of Claude agents. Each training job spawns a **Supervisor** agent, which in turn launches narrowly-scoped sub-agents. Agents have read access broadly, but every write action goes through a single gated tool.

### Agent roles

| Agent | Surface it watches | Responsibility |
|---|---|---|
| **Supervisor** | the job manifest, other agents | Orchestrates the fleet for one training job. Routes anomalies to sub-agents, decides between auto-remediation and human escalation, persists the incident log. |
| **Training Watcher** | Ray Train `report()` stream, trainer stdout | Detects NaN/Inf losses, gradient explosions, accuracy divergence, stalled steps, dataloader starvation. |
| **Cluster Health** | KubeRay CRDs, `kubectl`, Ray dashboard API | Watches pod restarts, node pressure, Ray actor failures, OOMKilled events, GPU unavailability. |
| **Data Quality** | parquet shards (local FS or GCS) | Validates Arrow schema, row counts, and image-decoding integrity before training; spot-checks shards during long runs. |
| **Remediator** | gated write surface | Executes scoped corrective actions (rolling-restart a worker group, scale replicas, rerun a failed Ray task). Every call requires Supervisor approval and is logged. |

### How agents launch with training

The trainer (`main.py` or `ray/cifar10-train.py`) emits a structured "job started" event. The Supervisor agent is started in parallel and is responsible for spawning the rest:

```bash
# Local development — sidecar process
python -m agents.supervisor --job-id $(uuidgen) &
python main.py
```

In production, the Supervisor is itself a Kubernetes `Job` deployed alongside the `RayCluster`, with the job ID injected via env var.

The Supervisor's startup sequence:

1.  Reads the job manifest (model, dataset, scaling config, expected runtime).
2.  Instantiates the four sub-agents via the Claude Agent SDK, each with a **narrow** tool list — e.g., Cluster Health gets `kubectl get/describe/logs` for the `ray` namespace only, never `kubectl delete`.
3.  Subscribes to Ray Train's report stream and a Kubernetes watch channel.
4.  On anomaly: routes the event to the relevant sub-agent for diagnosis → asks for a remediation proposal → either approves and calls the Remediator, or escalates to a human via Slack/PagerDuty.

### Tooling exposed to agents

Each agent gets a focused tool list, not blanket shell access. Tools live under `agents/tools/`:

| Tool | Purpose | Permission |
|---|---|---|
| `kubectl_read.py` | wraps `kubectl get/describe/logs` for the `ray` namespace | read-only |
| `ray_api.py` | calls Ray dashboard REST endpoints (`/api/cluster_status`, `/api/jobs`) | read-only |
| `parquet_validate.py` | opens a shard with `pyarrow`, checks schema + row counts | read-only |
| `gcs_read.py` *(cloud only)* | reads manifests/logs from a GCS bucket | read-only |
| `remediate.py` | the only write surface: scale, restart, kill, rerun | gated by Supervisor approval, audit-logged |

The Claude Agent SDK's MCP server support lets us expose these tools uniformly across local and cloud — the same Python tool definitions, just pointed at a different cluster context.

## ☁️ Local → Cloud Path

The same agents, manifests, and trainer code run in both environments. Only the surrounding cluster and IO backends change.

| Concern | Local (MacBook) | Production (GCP) |
|---|---|---|
| Kubernetes | `kind` or `minikube` | GKE (Autopilot or Standard) |
| KubeRay operator | Helm install on local kube | Helm install on GKE |
| Container registry | Docker Desktop local images | Artifact Registry (`us-central1-docker.pkg.dev/...`) |
| Dataset storage | local parquet cache under `data/cifar10/` | GCS bucket, read via `pyarrow.fs.GcsFileSystem` |
| Secrets (Anthropic API key, HF token) | `.env` file (gitignored) | Secret Manager + Workload Identity |
| Agent runtime | Python process beside the trainer | Kubernetes `Job` in the `ray` namespace |
| Observability backend | stdout, `kubectl logs` | Cloud Logging + Cloud Monitoring; agents query via the GCP client libraries |
| GPUs | none / MPS for sanity checks | GKE node pool with A100/L4/T4 |

### Migration stages

1.  **Stage 0 — Local prototype** *(current)*. Single-node training via `main.py`; Supervisor runs as a local process; `kubectl` points at a `kind` cluster.
2.  **Stage 1 — Local KubeRay**. Apply `kubernetes/park-cluster.yaml` then `ray/ray-cluster.yaml` against `kind`. Trainer becomes `ray/cifar10-train.py --address ray://localhost:10001`. Agents still run locally.
3.  **Stage 2 — GKE lift**. Push images to Artifact Registry; reapply the same manifests to a GKE cluster. Parquet shards move to GCS. Supervisor becomes a `Job` in the cluster.
4.  **Stage 3 — GCP-native observability**. Agents swap stdout/log scraping for Cloud Logging queries; the Remediator's actions are scoped by IAM via Workload Identity.

## 🧰 How It All Fits Together

| Layer | Component | Role in the pipeline |
|---|---|---|
| **Data — at rest** | Parquet on HF Hub / GCS | Cold storage of CIFAR-10 shards (or any image dataset). |
| **Data — in motion** | Apache Arrow / `pyarrow` | Columnar in-memory format and parquet reader; zero-copy into PyTorch tensors. |
| **Modeling** | PyTorch + torchvision | ResNet18 definition, training loop, optimizer/scheduler. |
| **Scaling** | Ray Train (`TorchTrainer`) | Wraps the PyTorch loop, handles process group setup, distributes batches across workers. |
| **Cluster** | Kubernetes | Schedules pods, manages lifecycle, exposes services. |
| **Cluster glue** | KubeRay operator | Reconciles the `RayCluster` CRD into head + worker pods and services. |
| **Packaging** | Docker | Reproducible images for trainer and Ray nodes. |
| **Agents — brains** | Claude Agent SDK | Defines each agent's system prompt, tools, and message loop. |
| **Agents — hands** | Tool wrappers under `agents/tools/` | Bridge between Claude and the cluster (`kubectl`, Ray API, parquet, GCS). |
| **Cloud substrate** | GKE / GCS / Artifact Registry / Secret Manager / Cloud Logging | Production hosting; agents adopt GCP-native APIs without changing their decision logic. |

End-to-end flow of a single training run:

```
HF Hub / GCS  ──parquet──►  pyarrow reader  ──tensors──►  PyTorch model
                                                              │
                                                              ▼
                                                          Ray Train
                                                              │
                                                              ▼
                                                  KubeRay-managed pods (K8s)
                                                              │
                                          ┌───────────────────┼────────────────────┐
                                          ▼                   ▼                    ▼
                                  Training Watcher    Cluster Health         Data Quality
                                          │                   │                    │
                                          └────────► Supervisor (Claude Agent SDK) ◄┘
                                                              │
                                                  ┌───────────┴────────────┐
                                                  ▼                        ▼
                                               Remediator             Human escalation
                                          (gated write tools)        (Slack / PagerDuty)
```

## 🏃 Detailed Workflow

### 1. Build Docker Images

```bash
# Build the generic PyTorch base
docker build -t park-pytorch:v1 -f docker/Dockerfile.pytorch .

# Build the CIFAR-10 trainer image (depends on park-pytorch:v1)
docker build -t park-cifar10:v1 -f docker/Dockerfile.cifar10 .
```

### 2. Deploy Ray Cluster on Kubernetes
```bash
# Install Kubernetes (if not already running)
# On macOS, use Minikube or kind
brew install minikube
minikube start

# Create Kubernetes namespace
kubectl create namespace ray

# Apply Ray cluster configuration
kubectl apply -f ray/ray-cluster.yaml
```

### 3. Run Distributed Training
```bash
# Run the Ray Train script (point at the head node if running on a cluster)
python ray/cifar10-train.py --address ray://<head-svc>:10001 --num-workers 2
```

This script will:
1.  Download the CIFAR-10 parquet shards from `uoft-cs/cifar10` on Hugging Face
2.  Submit a TorchTrainer job to the Ray cluster
3.  Train ResNet18 across the configured workers
4.  Report per-epoch test accuracy back to the driver

## 📊 Demo Results

When you run `python main.py`, you should see output similar to:

```
Downloading CIFAR-10 parquet shards from uoft-cs/cifar10... OK
epoch 01/20  loss=1.7234  test_acc=0.4821
...
epoch 20/20  loss=0.2103  test_acc=0.9134

🚀 Success! ResNet18 trained on CIFAR-10 via the PARK stack.
```

## 🧪 Testing

### Verify Kubernetes Cluster
```bash
kubectl get pods -n ray
# Should show all Ray pods running
```

### Verify Ray Cluster
```bash
kubectl exec -n ray -it <ray-head-pod> -- ray status
# Should show cluster status with all nodes
```

### Verify Training
```bash
kubectl logs -n ray -l component=ray-worker --tail=50
# Should show training logs
```

## 🧹 Cleanup

```bash
# Delete Ray cluster
kubectl delete -f ray/ray-cluster.yaml

# Delete Docker images
docker rmi park-cifar10:v1
docker rmi park-pytorch:v1
```

## 📚 Documentation

-   [Claude Agent SDK](https://docs.anthropic.com/en/api/agent-sdk/overview)
-   [KubeRay Operator](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
-   [Ray Train (TorchTrainer)](https://docs.ray.io/en/latest/train/train.html)
-   [Docker Compose Documentation](https://docs.docker.com/compose/)
-   [PyTorch Documentation](https://pytorch.org/docs/)
-   [Apache Arrow Documentation](https://arrow.apache.org/docs/)
-   [GKE + KubeRay quickstart](https://cloud.google.com/kubernetes-engine/docs/tutorials/distributed-tensorflow-ray)
