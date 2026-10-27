#!/usr/bin/env bash
# Launch the AutoPARK benchmark suite as a Kubernetes Job on the autopark-eval EKS cluster.
#
#   eks/launch.sh [benchmark args...]     (default: --suite paper --repeats 5 --epochs 3 --subset 5000 --device cpu)
#   WORLD_SIZE=4 eks/launch.sh [...]      multi-node DDP run (eks/dist-job.yaml), one rank per node
#
# Uploads `git archive HEAD` to S3 so the pod runs exactly the committed code, creates the
# Claude Platform on AWS secret from .env (API key + workspace ID only), and applies the Job.
# Prints the run tag; pass it to `python -m benchmarks.eks_listener --run-tag <tag>`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BUCKET="${BUCKET:-park-ml-experiments-515966515749-us-east-1}"
BENCH_ARGS="${*:---suite paper --repeats 5 --epochs 3 --subset 5000 --device cpu}"
WORLD_SIZE="${WORLD_SIZE:-1}"
if [ "${WORLD_SIZE}" -gt 1 ]; then MANIFEST=eks/dist-job.yaml; PREFIX=dist; else MANIFEST=eks/eval-job.yaml; PREFIX=bench; fi

if ! git diff --quiet HEAD -- . ':!autopark/agentic-ml-infra.code-workspace'; then
    echo "Refusing to launch: tracked files differ from HEAD. Commit first so the run is reproducible." >&2
    exit 1
fi

CODE_SHA="$(git rev-parse HEAD)"
RUN_TAG="eks-$(date -u +%Y%m%dT%H%M%SZ)"
CODE_KEY="autopark-eval/code/${CODE_SHA}.tar.gz"
RESULTS_PREFIX="autopark-eval/runs/${RUN_TAG}"
JOB_NAME="autopark-${PREFIX}-$(echo "${RUN_TAG}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9-')"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
git archive --format=tar.gz -o "${tmp}/code.tar.gz" HEAD
aws s3 cp --only-show-errors "${tmp}/code.tar.gz" "s3://${BUCKET}/${CODE_KEY}"

# Secret: read values from .env without echoing them
api_key="$(grep -E '^ANTHROPIC_AWS_API_KEY=' .env | cut -d= -f2-)"
workspace_id="$(grep -E '^ANTHROPIC_AWS_WORKSPACE_ID=' .env | cut -d= -f2-)"
kubectl -n autopark create secret generic claude-platform-aws \
    --from-literal=api-key="${api_key}" --from-literal=workspace-id="${workspace_id}" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null

JOB_NAME="${JOB_NAME}" RUN_TAG="${RUN_TAG}" BUCKET="${BUCKET}" CODE_KEY="${CODE_KEY}" CODE_SHA="${CODE_SHA}" \
RESULTS_PREFIX="${RESULTS_PREFIX}" BENCH_ARGS="${BENCH_ARGS}" WORLD_SIZE="${WORLD_SIZE}" \
python3 -c 'import os,re,sys; t=open(sys.argv[1]).read(); sys.stdout.write(re.sub(r"\$\{(\w+)\}", lambda m: os.environ[m.group(1)], t))' \
    "${MANIFEST}" | kubectl apply -f - >/dev/null

echo "RUN_TAG=${RUN_TAG}"
echo "JOB=${JOB_NAME}"
echo "CODE_SHA=${CODE_SHA}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "RESULTS=s3://${BUCKET}/${RESULTS_PREFIX}/"
