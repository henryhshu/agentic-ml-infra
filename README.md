# Agentic ML with the PARK Stack

This capstone project aims to create a self-reliant infrastructure for distributed model training on the **PARK stack**. This stack leverages PyTorch for model development, Arrow for data interoperability, Ray for distributed execution, and Kubernetes for container orchestration.

We aim to use agents to help us ensure that minor breakages along the pipeline are resolved with minimal human intervention.

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

-   [Ray on Kubernetes Documentation](https://docs.ray.io/en/latest/cluster/kubernetes.html)
-   [Docker Compose Documentation](https://docs.docker.com/compose/)
-   [PyTorch Documentation](https://pytorch.org/docs/)
-   [Apache Arrow Documentation](https://arrow.apache.org/docs/)
