# EKS replication analysis — `eks-20261027T105238Z`

Code: `2d31d5387796fb9f9ccf0975935b7b3a540e0908`. All numbers below are computed from recorded data by `benchmarks/eks_analysis.py`.

## Environment

| | Local reference | EKS |
|---|---|---|
| Host | macOS-26.6.2-arm64-arm-64bit | c6i.xlarge (ON_DEMAND, us-east-1d), Amazon Linux 2023.12.20261021 |
| CPU | Apple Silicon (arm64) | Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz (4 vCPU) |
| Python / torch | 3.11.15 / 2.11.0 | 3.11.15 / 2.11.0+cpu (threads=2) |
| Kubernetes | – | kubelet v1.34.12-eks-39aca56, containerd://2.3.5+unknown |
| Workload | tiny, 3 ep, subset 5000, 5 repeats | same |

## Pod lifecycle (seconds)

| Pending→scheduled | Scheduled→container start (image pull) | Bootstrap (code + pip install) | Benchmark | Bundle + upload | Pod total |
|---|---|---|---|---|---|
| 0.0 | 2.0 | 43.6 | 2505.7 | 1.7 | 2553.0 |

## Container resources (cgroup v2, 2 s samples)

- Samples: 1253; CPU cores used: median 1.07, p90 1.69, max 1.74
- CPU cores while a stage attempt runs (agentic): median 1.62; while agents handle an incident (waiting on Claude): median 0.00
- Memory: median 3876 MiB, max sampled 5477 MiB, cgroup peak 5513 MiB; CPU throttled 0.0 s

## Outcomes and MTTR, local vs. EKS (median s, [min–max])

| Scenario | Expected | Local outcomes | EKS outcomes | Local MTTR | EKS MTTR | Local T_reason | EKS T_reason | Local / EKS baseline wall | Local / EKS agentic wall |
|---|---|---|---|---|---|---|---|---|---|
| `nan_divergence` | healed | healed 5 | healed 5 | 15.1 [14.3–15.8] | 11.8 [11.5–13.0] | 14.8 | 11.5 | 2.0 / 3.1 | 39.1 / 39.9 |
| `batch_oom` | healed | healed 5 | healed 5 | 15.9 [14.9–17.1] | 12.8 [12.6–13.9] | 15.7 | 12.7 | 1.8 / 2.6 | 40.1 / 42.2 |
| `compound_oom_and_explosion` | healed | healed 5 | healed 5 | 23.6 [22.0–23.7] | 18.7 [18.1–19.7] | 23.4 | 18.4 | 1.7 / 2.7 | 47.0 / 47.7 |
| `cascaded_lr_and_memory` | healed | healed 3, exhausted 2 | healed 4, exhausted 1 | 23.5 [22.2–31.0] | 18.8 [18.4–23.6] | 23.3 | 18.5 | 1.7 / 2.7 | 45.6 / 45.8 |
| `delayed_stochastic_explosion` | healed | healed 5 | healed 5 | 8.3 [7.6–8.5] | 6.6 [6.4–7.1] | 8.0 | 6.2 | 2.6 / 3.6 | 31.7 / 35.5 |
| `spot_worker_churn` | healed | healed 5 | healed 5 | 7.9 [7.0–9.4] | 6.4 [6.2–12.6] | 7.8 | 6.2 | 1.7 / 2.7 | 32.1 / 35.4 |
| `triple_cascade_oom_lr_grad` | healed | exhausted 5 | exhausted 5 | 29.5 [29.0–31.9] | 24.5 [23.4–25.2] | 29.2 | 24.2 | 1.7 / 2.8 | 31.2 / 27.1 |
| `catastrophic_label_noise` | escalated | escalated 5 | escalated 5 | 64.6 [64.2–78.2] | 65.9 [63.0–76.9] | 30.0 | 25.2 | 19.5 / 25.9 | 66.4 / 68.6 |
| `architecture_tensor_mismatch` | escalated | escalated 5 | escalated 5 | 29.3 [28.0–29.7] | 24.6 [24.5–25.2] | 29.2 | 24.6 | 1.8 / 2.7 | 30.9 / 27.9 |
| `thrashing_bimodal_trap` | exhausted | exhausted 5 | exhausted 5 | 31.5 [30.6–36.8] | 25.8 [24.8–26.5] | 31.2 | 25.5 | 1.9 / 3.1 | 33.1 / 28.6 |
| `clean_run` | passed | passed 5 | passed 5 | – | – | – | – | 19.8 / 24.3 | 24.3 / 27.4 |

Levels 1–3 healed: local 28/30, EKS 29/30. Scenarios with identical outcome counts: 10/11.

## Claude call latency (all agentic calls)

| | n | median s | p90 s | mean s | max s |
|---|---|---|---|---|---|
| Local | 796 | 1.45 | 2.42 | 1.59 | 6.97 |
| EKS (us-east-1) | 798 | 1.17 | 2.15 | 1.29 | 2.65 |

| Caller | Local median s | EKS median s |
|---|---|---|
| `analyze_epoch` | 1.41 | 1.13 |
| `diagnose` | 1.44 | 1.16 |
| `handle_anomaly` | 2.41 | 2.16 |
| `recommend_action` | 0.73 | 0.46 |

## Training speed and monitoring overhead

- Median healthy step time (clean_run baseline): local 0.096 s, EKS 0.105 s
- Fault-free overhead of agentic monitoring: local 19.8 → 24.3 s (+23%), EKS 24.3 → 27.4 s (+13%); per-epoch Claude time median local 4.2 s, EKS 3.1 s
- Total Claude usage: local 796 calls / 284,852 in / 70,402 out tokens; EKS 798 calls / 284,644 in / 70,086 out tokens

## EKS remediation sequences per run

- `nan_divergence`: reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate
- `batch_oom`: reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size
- `compound_oom_and_explosion`: reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping
- `cascaded_lr_and_memory`: reduce_batch_size → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → enable_gradient_clipping → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → reduce_learning_rate → reduce_learning_rate
- `delayed_stochastic_explosion`: enable_gradient_clipping; enable_gradient_clipping; enable_gradient_clipping; enable_gradient_clipping; enable_gradient_clipping
- `spot_worker_churn`: restart_workers; restart_workers; retry_stage → restart_workers; restart_workers; restart_workers
- `triple_cascade_oom_lr_grad`: reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping
- `catastrophic_label_noise`: (none); (none); (none); (none); (none)
- `architecture_tensor_mismatch`: (none); (none); (none); (none); (none)
- `thrashing_bimodal_trap`: reduce_learning_rate → reduce_learning_rate → reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → enable_gradient_clipping → reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → enable_gradient_clipping → reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping → reduce_learning_rate
- `clean_run`: (none); (none); (none); (none); (none)
