#!/usr/bin/env bash
# Recreate the benchmark environment. Prints commands unless --apply is supplied.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ACTION="${1:-plan}"
APPLY="${2:-}"
REGION=us-east-1
ACCOUNT=515966515749
BUCKET="park-ml-experiments-${ACCOUNT}-${REGION}"
export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"
export KUBECONFIG="${KUBECONFIG:-$ROOT/.local/aws-recovery/kubeconfig}"

case "$ACTION" in plan|up|down|images|restore-results) ;; *) echo "Usage: bash eks/recover.sh {plan|up|down|images|restore-results} [--apply]" >&2; exit 2;; esac
if [[ -n "$APPLY" && "$APPLY" != --apply ]]; then echo "Expected --apply" >&2; exit 2; fi
if [[ "$ACTION" == plan && -n "$APPLY" ]]; then echo "Choose an explicit action for --apply" >&2; exit 2; fi

run() {
    printf '%q ' "$@"; printf '\n'
    if [[ "$APPLY" == --apply ]]; then "$@"; fi
}
if [[ "$APPLY" == --apply ]]; then
    command -v aws >/dev/null
    [[ "$(aws sts get-caller-identity --query Account --output text)" == "$ACCOUNT" ]] || {
        echo "Wrong AWS account; this recovery recipe targets $ACCOUNT" >&2; exit 1;
    }
fi

storage() {
    # A successful list distinguishes a missing bucket from an authentication error.
    if [[ "$APPLY" == --apply ]]; then
        names="$(aws s3api list-buckets --query 'Buckets[].Name' --output text)"
        if ! printf '%s\n' "$names" | tr '\t' '\n' | grep -Fxq "$BUCKET"; then
            run aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
        fi
    else
        run aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
    fi
    run aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    run aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
        '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
}

case "$ACTION" in
plan|up)
    echo "Recreates billable EKS control plane, four nodes, disks, and S3 storage. Does not launch training."
    if [[ "$APPLY" == --apply ]]; then
        command -v eksctl >/dev/null; command -v kubectl >/dev/null
        mkdir -p "$(dirname "$KUBECONFIG")"
    fi
    storage
    run eksctl create cluster --config-file eks/cluster.yaml --write-kubeconfig=false
    run aws eks update-kubeconfig --name autopark-eval --region "$REGION" --kubeconfig "$KUBECONFIG"
    run kubectl -n autopark get serviceaccount autopark-eval
    run kubectl get nodes
    echo "To launch later: KUBECONFIG='$KUBECONFIG' WORLD_SIZE=4 bash eks/launch.sh"
    ;;
down)
    run eksctl delete cluster --config-file eks/cluster.yaml --wait
    echo "S3 results remain. Archive them locally and delete the bucket separately to end storage charges."
    ;;
images)
    echo "Optional legacy image rebuild; current benchmark jobs use public Python images."
    run bash aws/ecr_push.sh --region "$REGION" --account-id "$ACCOUNT"
    ;;
restore-results)
    BACKUP="${BACKUP_DIR:-$ROOT/.local/aws-recovery/2026-11-03/s3}"
    if [[ "$APPLY" == --apply && ! -d "$BACKUP" ]]; then echo "Missing backup: $BACKUP" >&2; exit 1; fi
    storage
    run aws s3 sync "$BACKUP/" "s3://$BUCKET/" --only-show-errors
    ;;
esac
