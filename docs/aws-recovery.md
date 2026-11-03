# Recover the benchmark cluster

The four-node benchmark environment is defined in `eks/cluster.yaml`.
`eks/recover.sh` recreates its S3 bucket, EKS cluster, node group, OIDC provider,
and IRSA service account, then checks node and service-account visibility.
It targets account `515966515749`, region `us-east-1`, cluster `autopark-eval`.

## Start again

Install AWS CLI, eksctl, kubectl, and Python 3; authenticate with `aws login`.
Run from the repository root:

```sh
# Inspect the commands without making AWS calls or creating resources.
bash eks/recover.sh plan

# Creates billable infrastructure. Run only when ready to resume experiments.
bash eks/recover.sh up --apply

export KUBECONFIG="$PWD/.local/aws-recovery/kubeconfig"
# Requires .env with ANTHROPIC_AWS_API_KEY and ANTHROPIC_AWS_WORKSPACE_ID.
# Credentials are not stored in Git or the recovery archive.
WORLD_SIZE=4 bash eks/launch.sh
```

`up` is for an absent cluster; it fails rather than replacing an existing one.
After a partial provisioning failure, inspect eksctl/CloudFormation state before
retrying. These commands have been syntax-checked and exercised with mocked CLI
tools; a live cluster was not created to test recovery because that incurs costs.

## Restore saved results

Before S3 deletion on 2026-11-03, all 16 objects (7,754,838 bytes) were downloaded
to `.local/aws-recovery/2026-11-03/s3/`. Each object's size and MD5 were checked
against the remote inventory. SHA-256 hashes are recorded in the adjacent
`checksums.json`, with resource metadata alongside it. This archive is local,
ignored by Git, and must be copied elsewhere locally if this checkout is removed.
It is not available from a fresh Git clone. Recorded benchmark traces already
tracked under `benchmarks/reports/` remain available in Git independently.

```sh
# Recreates the bucket if needed and uploads the locally saved results.
bash eks/recover.sh restore-results --apply
# To use an archive moved elsewhere:
BACKUP_DIR=/absolute/path/to/s3 bash eks/recover.sh restore-results --apply
```

The deleted bucket name may become unavailable to recreate. If so, choose a new
bucket and update `eks/recover.sh`, the S3 resource ARNs in `eks/cluster.yaml`,
and `BUCKET` when invoking `eks/launch.sh`.

## Optional legacy containers

Current benchmark Jobs bootstrap from a public Python image and committed code
uploaded by `eks/launch.sh`; they do not need the deleted private ECR images.
For the older Ray pipeline, install/start Docker and run:

```sh
bash eks/recover.sh images --apply
```

This uses `docker/Dockerfile.pytorch`, `docker/Dockerfile.cifar10`, and
`aws/ecr_push.sh` to recreate and repopulate both repositories. This is a source
rebuild, not byte-for-byte recovery of the old image digests; dependencies and
upstream images can change. The historical CloudWatch log contents were not
archived and cannot be recovered. Neither is required to spin up the cluster.
The `.dockerignore` excludes local backups and credentials from image builds.

## Shut down afterward

```sh
bash eks/recover.sh down --apply
```

This waits for cluster teardown. S3 results and any rebuilt ECR images remain
billable until archived locally and deleted. Check for orphaned volumes,
snapshots, public IPs, load balancers, NAT gateways, and log groups after teardown.
Do not treat cluster deletion alone as proof of zero AWS charges.
