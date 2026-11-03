# AWS shutdown audit

Checked at 2026-11-03T04:40:57+00:00 using the authenticated AWS CLI session.

## Status

No active compute, database, load balancer, NAT gateway, allocated Elastic IP,
VPC endpoint, EBS volume, EBS snapshot, or backup vault was found by the checks
below. No infrastructure needed to be stopped or terminated.

**This is not a zero-cost attestation.** Retained storage remains and can incur
charges. Permanent deletion awaits the owner's answer to the stored-data
cleanup question; no cloud resources or data were deleted during this audit.

| Retained resource | Location | Finding |
|---|---|---|
| ECR `park-pytorch` | us-east-1 | Container images remain |
| ECR `park-cifar10` | us-east-1 | Container images remain |
| CloudWatch `/aws/eks/park-ml-cluster/cluster` | us-east-1 | 949,763,122 stored bytes reported |
| CloudWatch `RDSOSMetrics` | us-east-1 | 9,310 stored bytes reported |
| S3 `park-ml-experiments-515966515749-us-east-1` | us-east-1 | 16 objects/versions, 7,754,838 bytes; `autopark-eval/` prefix |

The S3 bucket reports no versioning configuration and no incomplete multipart
uploads. Counts and byte totals were calculated from the API results.

## Scope and evidence

The audit completed 510 regional API checks across 17 enabled regions with no API
errors. Resource categories checked:

addresses, autoscaling, backup, classic_elb, dynamodb, ecr, ecs, efs, eks, elasticache, elasticbeanstalk, endpoints, fsx, instances, kms, lambda, loadbalancers, logs, nat, notebooks, opensearch, rds, rds_cluster_snapshots, rds_clusters, rds_snapshots, redshift, sagemaker, secrets, snapshots, volumes.

Regions: ap-northeast-1, ap-northeast-2, ap-northeast-3, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-north-1, eu-west-1, eu-west-2, eu-west-3, sa-east-1, us-east-1, us-east-2, us-west-1, us-west-2.

Global checks found one S3 bucket, no Route 53 hosted zones, no CloudFront
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
