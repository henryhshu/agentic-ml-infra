# Multi-node EKS analysis — `eks-20261027T131911Z`

Code `b5138f3b4f4c1db141e7d450c3dd58664e84d700`; world size 4 (PyTorch DDP, gloo over TCP, one rank per node); torch 2.11.0+cpu; workload tiny, 3 epochs, subset 5000, 5 repeats. Benchmark wall time 34.4 min. All numbers are computed from recorded data by `benchmarks/dist_analysis.py`.

## Ranks

| Rank | Node | Type | Zone | Scheduled→start s | Bootstrap s | Proc. restarts | Cores (attempts) | Cores (incident handling) | Mem median MiB | Mem peak MiB | Throttled s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ip-192-168-33-211.ec2.internal | c6i.xlarge (ON_DEMAND) | us-east-1d | 0.0 | 40.4 | – | 0.99 | 0.01 | 4175 | 4453 | 0.0 |
| 1 | ip-192-168-60-250.ec2.internal | c6i.xlarge (ON_DEMAND) | us-east-1d | 1.0 | 40.1 | 0 | 0.66 | 0.00 | 2383 | 2387 | 0.0 |
| 2 | ip-192-168-33-100.ec2.internal | c6i.xlarge (ON_DEMAND) | us-east-1d | 0.0 | 39.7 | 0 | 0.63 | 0.00 | 2380 | 2383 | 0.0 |
| 3 | ip-192-168-41-244.ec2.internal | c6i.xlarge (ON_DEMAND) | us-east-1d | 1.0 | 41.5 | 25 | 0.76 | 0.00 | 2353 | 2396 | 0.0 |

## Outcomes and MTTR (median s [min–max])

| Scenario | Expected | Outcomes | MTTR | T_det | T_reason | T_act | T_recover | Baseline wall | Agentic wall |
|---|---|---|---|---|---|---|---|---|---|
| `nan_divergence` | healed | healed 5 | 12.5 [12.0–14.0] | 0.2 | 12.2 | 0.0 | 0.1 | 3.1 | 29.8 |
| `batch_oom` | healed | healed 5 | 13.2 [13.0–13.7] | 0.2 | 12.9 | 0.0 | 0.1 | 2.9 | 31.3 |
| `compound_oom_and_explosion` | healed | healed 5 | 19.2 [18.3–19.4] | 0.3 | 18.7 | 0.0 | 0.1 | 2.7 | 36.7 |
| `cascaded_lr_and_memory` | healed | healed 5 | 19.3 [18.3–19.9] | 0.4 | 18.8 | 0.0 | 0.1 | 2.8 | 37.1 |
| `delayed_stochastic_explosion` | healed | healed 5 | 6.4 [6.2–6.9] | 0.2 | 6.1 | 0.0 | 0.1 | 3.1 | 24.4 |
| `spot_worker_churn` | healed | healed 5 | 11.3 [11.1–11.7] | 2.9 | 7.0 | 0.5 | 0.9 | 14.0 | 28.9 |
| `triple_cascade_oom_lr_grad` | healed | exhausted 5 | 25.1 [24.5–25.6] | 0.5 | 24.6 | 0.0 | 0.0 | 2.9 | 29.2 |
| `catastrophic_label_noise` | escalated | escalated 5 | 43.2 [42.9–44.0] | 17.8 | 25.5 | 0.0 | 0.0 | 12.1 | 47.3 |
| `architecture_tensor_mismatch` | escalated | escalated 5 | 28.5 [27.1–29.2] | 3.2 | 25.6 | 0.0 | 0.0 | 6.0 | 32.4 |
| `thrashing_bimodal_trap` | exhausted | exhausted 5 | 26.8 [26.3–27.7] | 0.5 | 26.3 | 0.0 | 0.0 | 3.0 | 30.9 |
| `clean_run` | passed | passed 5 | – [–––] | – | – | – | – | 11.8 | 17.7 |

Levels 1–3 healed: 30/30. Runs matching the expected outcome: 50/55.

## Worker loss (`spot_worker_churn`)

| Repeat | Mode | Status | Attempts | Killed rank | Kill→collective error s | Liveness check s | Kill→attempt failed s | Kill→replacement connected s | restart_workers wait s | Actions |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | failed | 4 | 3 | 0.031, 0.031, 0.031, 0.031 | 2.25, 2.25, 2.25, 2.25 | 2.29, 2.29, 2.29, 2.29 | 1.89, 1.88, 1.86, 1.85 |  | – |
| 0 | agentic | success | 2 | 3 | 0.031 | 2.25 | 2.29 | 1.82 | 0.5 | restart_workers |
| 1 | baseline | failed | 4 | 3 | 0.031, 0.031, 0.031, 0.031 | 2.00, 2.00, 2.00, 2.00 | 2.04, 2.04, 2.04, 2.04 | 1.58, 1.60, 1.59, 1.57 |  | – |
| 1 | agentic | success | 2 | 3 | 0.031 | 2.00 | 2.04 | 1.58 | 0.5 | restart_workers |
| 2 | baseline | failed | 4 | 3 | 0.031, 0.031, 0.031, 0.031 | 2.00, 2.00, 2.00, 2.00 | 2.04, 2.04, 2.04, 2.04 | 1.68, 1.65, 1.61, 1.66 |  | – |
| 2 | agentic | success | 2 | 3 | 0.031 | 2.00 | 2.04 | 1.70 | 0.5 | restart_workers |
| 3 | baseline | failed | 4 | 3 | 0.031, 0.031, 0.031, 0.031 | 2.00, 2.00, 2.00, 2.00 | 2.04, 2.04, 2.04, 2.04 | 1.77, 1.67, 1.66, 1.69 |  | – |
| 3 | agentic | success | 2 | 3 | 0.031 | 2.00 | 2.04 | 1.75 | 0.5 | restart_workers |
| 4 | baseline | failed | 4 | 3 | 0.031, 0.031, 0.031, 0.031 | 2.25, 2.25, 2.50, 2.50 | 2.29, 2.29, 2.54, 2.54 | 1.80, 1.83, 2.08, 2.35 |  | – |
| 4 | agentic | success | 2 | 3 | 0.031 | 2.00 | 2.04 | 1.76 | 0.5 | restart_workers |

Example error seen by the agents: `Distributed data-parallel training failed on rank 0 (RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:547] Connection closed by peer [192.168.44.80]:7894). Worker liveness after the failure: 2/3 worker processes heartbeating; lost rank 3: process exited (restarted with a new proce`

## Claude call latency

| Caller | n | median s | p90 s | max s |
|---|---|---|---|---|
| all | 790 | 1.19 | 2.18 | 2.77 |
| `analyze_epoch` | 130 | 1.15 | 1.33 | 1.50 |
| `diagnose` | 420 | 1.17 | 1.71 | 2.37 |
| `handle_anomaly` | 140 | 2.18 | 2.42 | 2.77 |
| `recommend_action` | 100 | 0.47 | 0.54 | 0.74 |

## Training and monitoring overhead

- DDP step time (clean baseline, global batch over 4 ranks): median 0.041 s, p90 0.048 s
- Fault-free run: baseline 11.8 s, agentic 17.7 s (+50%); Claude time inside attempts median 3.3 s
- Claude usage: 790 calls, 282,962 input / 70,199 output tokens
