#!/usr/bin/env bash
# ==============================================================================
# ecr_push.sh - Build and push container images to Amazon ECR
#
# Repositories:
#   - park-pytorch (Base image with PyTorch CUDA runtime & requirements)
#   - park-cifar10 (Application image with CIFAR-10 training & agent platform code)
# ==============================================================================

set -euo pipefail

# Text formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Determine project root directory (parent directory of /aws)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source .env if present
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Default configurations
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
REPO_PYTORCH="park-pytorch"
REPO_CIFAR10="park-cifar10"
NO_CACHE="${NO_CACHE:-false}"

# Display help
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Builds Docker images for the ML training platform and pushes them to Amazon ECR.

Options:
  -r, --region REGION       AWS region (default: \$AWS_REGION or us-west-2)
  -a, --account-id ID       AWS Account ID (defaults to sts get-caller-identity)
  -t, --tag TAG             Image tag to apply (default: v1)
  --no-cache                Build Docker images without cache
  -h, --help                Show this help message and exit

Environment Variables:
  AWS_REGION                AWS Region for ECR registry
  AWS_ACCOUNT_ID            AWS 12-digit Account ID
  IMAGE_TAG                 Image tag (e.g. v1, latest, git commit SHA)

Examples:
  ./aws/ecr_push.sh
  ./aws/ecr_push.sh --region us-west-2 --tag v2
  IMAGE_TAG=latest ./aws/ecr_push.sh
EOF
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -a|--account-id)
            AWS_ACCOUNT_ID="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown argument '$1'${NC}" >&2
            usage
            exit 1
            ;;
    esac
done

echo -e "${BOLD}${BLUE}======================================================${NC}"
echo -e "${BOLD}${BLUE}   Agentic ML Platform - Amazon ECR Build & Push      ${NC}"
echo -e "${BOLD}${BLUE}======================================================${NC}"

# Check prerequisites
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: 'aws' CLI is not installed or not in PATH.${NC}" >&2
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: 'docker' CLI is not installed or not in PATH.${NC}" >&2
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running.${NC}" >&2
    exit 1
fi

# Auto-detect AWS Account ID if not set
if [[ -z "${AWS_ACCOUNT_ID}" ]]; then
    echo -e "${BLUE}==>${NC} Resolving AWS Account ID via STS..."
    if ! AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text 2>/dev/null); then
        echo -e "${RED}Error: Unable to determine AWS Account ID. Please configure AWS credentials or set AWS_ACCOUNT_ID.${NC}" >&2
        exit 1
    fi
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
URI_PYTORCH="${ECR_REGISTRY}/${REPO_PYTORCH}"
URI_CIFAR10="${ECR_REGISTRY}/${REPO_CIFAR10}"

echo -e "${GREEN}Configuration:${NC}"
echo -e "  AWS Region:        ${BOLD}${AWS_REGION}${NC}"
echo -e "  AWS Account ID:    ${BOLD}${AWS_ACCOUNT_ID}${NC}"
echo -e "  ECR Registry:      ${BOLD}${ECR_REGISTRY}${NC}"
echo -e "  Image Tag:         ${BOLD}${IMAGE_TAG}${NC}"
echo -e "  Project Root:      ${PROJECT_ROOT}"
echo ""

# Step 1: Authenticate Docker with ECR
echo -e "${BLUE}==>${NC} Authenticating Docker to Amazon ECR (${AWS_REGION})..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"
echo -e "${GREEN}✓ Docker authentication successful.${NC}"
echo ""

# Step 2: Ensure ECR Repositories Exist
create_repo_if_missing() {
    local repo_name="$1"
    echo -e "${BLUE}==>${NC} Checking ECR repository: ${BOLD}${repo_name}${NC}..."
    if aws ecr describe-repositories --repository-names "${repo_name}" --region "${AWS_REGION}" &> /dev/null; then
        echo -e "    Repository '${repo_name}' already exists."
    else
        echo -e "${YELLOW}    Repository '${repo_name}' not found. Creating...${NC}"
        aws ecr create-repository \
            --repository-name "${repo_name}" \
            --region "${AWS_REGION}" \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256 \
            --tags Key=Project,Value=agentic-ml-platform Key=ManagedBy,Value=ecr_push.sh \
            > /dev/null
        echo -e "${GREEN}✓ Created ECR repository '${repo_name}'.${NC}"
    fi
}

create_repo_if_missing "${REPO_PYTORCH}"
create_repo_if_missing "${REPO_CIFAR10}"
echo ""

# Step 3: Build Base PyTorch Image
BUILD_ARGS=""
if [[ "${NO_CACHE}" == "true" ]]; then
    BUILD_ARGS="--no-cache"
fi

echo -e "${BLUE}==>${NC} Building base PyTorch image: ${BOLD}${REPO_PYTORCH}${NC} (linux/amd64)..."
docker build --platform linux/amd64 ${BUILD_ARGS} \
    -f "${PROJECT_ROOT}/docker/Dockerfile.pytorch" \
    -t "${REPO_PYTORCH}:v1" \
    -t "${REPO_PYTORCH}:latest" \
    -t "${URI_PYTORCH}:${IMAGE_TAG}" \
    -t "${URI_PYTORCH}:latest" \
    "${PROJECT_ROOT}"
echo -e "${GREEN}✓ Built ${REPO_PYTORCH}.${NC}"
echo ""

# Step 4: Build CIFAR-10 Application Image
echo -e "${BLUE}==>${NC} Building CIFAR-10 training image: ${BOLD}${REPO_CIFAR10}${NC} (linux/amd64)..."
docker build --platform linux/amd64 ${BUILD_ARGS} \
    -f "${PROJECT_ROOT}/docker/Dockerfile.cifar10" \
    -t "${REPO_CIFAR10}:v1" \
    -t "${REPO_CIFAR10}:latest" \
    -t "${URI_CIFAR10}:${IMAGE_TAG}" \
    -t "${URI_CIFAR10}:latest" \
    "${PROJECT_ROOT}"
echo -e "${GREEN}✓ Built ${REPO_CIFAR10}.${NC}"
echo ""

# Step 5: Push Images to Amazon ECR
echo -e "${BLUE}==>${NC} Pushing ${BOLD}${REPO_PYTORCH}${NC} to ECR..."
docker push "${URI_PYTORCH}:${IMAGE_TAG}"
docker push "${URI_PYTORCH}:latest"
echo -e "${GREEN}✓ Pushed ${URI_PYTORCH}:${IMAGE_TAG} and ${URI_PYTORCH}:latest${NC}"
echo ""

echo -e "${BLUE}==>${NC} Pushing ${BOLD}${REPO_CIFAR10}${NC} to ECR..."
docker push "${URI_CIFAR10}:${IMAGE_TAG}"
docker push "${URI_CIFAR10}:latest"
echo -e "${GREEN}✓ Pushed ${URI_CIFAR10}:${IMAGE_TAG} and ${URI_CIFAR10}:latest${NC}"
echo ""

# Step 6: Summary
echo -e "${BOLD}${GREEN}======================================================${NC}"
echo -e "${BOLD}${GREEN}   ECR Push Completed Successfully!                   ${NC}"
echo -e "${BOLD}${GREEN}======================================================${NC}"
echo -e "Images available in Amazon ECR:"
echo -e "  Base PyTorch:"
echo -e "    - ${BOLD}${URI_PYTORCH}:${IMAGE_TAG}${NC}"
echo -e "    - ${BOLD}${URI_PYTORCH}:latest${NC}"
echo -e "  CIFAR-10 / Agent Platform:"
echo -e "    - ${BOLD}${URI_CIFAR10}:${IMAGE_TAG}${NC}"
echo -e "    - ${BOLD}${URI_CIFAR10}:latest${NC}"
echo ""
echo -e "Update your Kubernetes/Ray manifests with:"
echo -e "  ${BOLD}image: ${URI_CIFAR10}:${IMAGE_TAG}${NC}"
