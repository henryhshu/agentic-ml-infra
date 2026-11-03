# AWS shutdown audit

Checked at 2026-11-03T04:40:57+00:00 using the authenticated AWS CLI session.

## Status

No active compute, database, load balancer, NAT gateway, allocated Elastic IP,
VPC endpoint, EBS volume, EBS snapshot, or backup vault was found by the checks
below. No infrastructure needed to be stopped or terminated.

On the owner's subsequent authorization, the retained storage below was deleted
on 2026-11-03 after preparing and testing the [recovery mechanism](aws-recovery.md).

| Deleted resource | Location | Recovery |
|---|---|---|
| ECR `park-pytorch` | us-east-1 | Rebuild using `eks/recover.sh images --apply` |
| ECR `park-cifar10` | us-east-1 | Rebuild using `eks/recover.sh images --apply` |
| CloudWatch `/aws/eks/park-ml-cluster/cluster` | us-east-1 | Historical log contents permanently deleted |
| CloudWatch `RDSOSMetrics` | us-east-1 | Historical log contents permanently deleted |
| S3 `park-ml-experiments-515966515749-us-east-1` | us-east-1 | Local verified archive; `eks/recover.sh restore-results --apply` |

All 16 S3 objects (7,754,838 bytes) were downloaded locally and verified by size
and MD5 against S3 ETags before deletion. SHA-256 checksums were also recorded
and checked after deletion. The archive and metadata are under
`.local/aws-recovery/2026-11-03/`, excluded from Git and Docker build contexts.
Private container layers were not archived: recovery rebuilds from source and
may produce different image digests. Current benchmark Jobs use public images.
The recovery flow passed five local tests without creating billable resources.

## Post-cleanup verification

At 2026-11-03T04:49:20+00:00, all 510 regional checks across
17 enabled regions completed without API errors. Every checked resource
category was empty except the four previously identified AWS-managed KMS keys.
Global S3, Route 53, CloudFront, and Savings Plans checks were also empty.
Separate us-east-1 queries confirmed empty ECR repositories and log groups.
No billable resource was found in this audited scope after cleanup.

## Scope and evidence

The audit completed 510 regional API checks across 17 enabled regions with no API
errors. Resource categories checked:

addresses, autoscaling, backup, classic_elb, dynamodb, ecr, ecs, efs, eks, elasticache, elasticbeanstalk, endpoints, fsx, instances, kms, lambda, loadbalancers, logs, nat, notebooks, opensearch, rds, rds_cluster_snapshots, rds_clusters, rds_snapshots, redshift, sagemaker, secrets, snapshots, volumes.

Regions: ap-northeast-1, ap-northeast-2, ap-northeast-3, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-north-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1, us-east-1, us-east-2, us-west-1, us-west-2.

The initial global checks found one S3 bucket, no Route 53 hosted zones, no CloudFront
distributions, and no Savings Plans. Additional us-east-1 checks found no
CloudWatch alarms or dashboards, CloudTrail trails, active EC2 reservations,
or active RDS reservations. The remaining CloudFormation stack
`park-ml-iam-roles` contains IAM resources. Four KMS keys were found and
individually described; all are AWS-managed.

This is an inventory of the named services at the recorded time, not a universal
check of every AWS product or a billing forecast. Historical usage charges can
still post after shutdown. No Cost Explorer usage query or pricing estimate was
used to infer a zero balance. Raw CLI evidence is retained locally in
`/private/tmp/autopark-cost-audit/` and is not committed.
