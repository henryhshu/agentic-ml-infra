# AutoPARK Benchmark Report (measured)

- **Generated at**: `2026-10-27T13:54:19.230170+00:00`
- **Environment**: Linux-6.12.111-136.204.amzn2023.x86_64-x86_64-with-glibc2.41 · Python 3.11.15 · torch 2.11.0+cpu · device `cpu`
- **Workload**: model `tiny`, 3 epochs, CIFAR-10 subset 5000, max_retries=3
- **Agent LLM**: `claude-haiku-4-5-20251001`
- **Repeats per scenario**: 5

All faults are injected in software by `pipeline/training.py` (see `benchmarks/scenarios.py`).
Timing components are measured from wall-clock timestamps; definitions are in `benchmarks/harness.py`.

## Outcomes

| Scenario | Expected | Matches | Observed outcomes | Baseline (success/failed) | Baseline wall s (median) | Agentic wall s (median) |
|---|---|---|---|---|---|---|
| `nan_divergence` | healed | 5/5 | healed×5 | 0/5 | 3.1 | 29.8 |
| `batch_oom` | healed | 5/5 | healed×5 | 0/5 | 2.9 | 31.3 |
| `compound_oom_and_explosion` | healed | 5/5 | healed×5 | 0/5 | 2.7 | 36.7 |
| `cascaded_lr_and_memory` | healed | 5/5 | healed×5 | 0/5 | 2.8 | 37.1 |
| `delayed_stochastic_explosion` | healed | 5/5 | healed×5 | 0/5 | 3.1 | 24.4 |
| `spot_worker_churn` | healed | 5/5 | healed×5 | 0/5 | 14.0 | 28.9 |
| `triple_cascade_oom_lr_grad` | healed | 0/5 | exhausted×5 | 0/5 | 2.9 | 29.2 |
| `catastrophic_label_noise` | escalated | 5/5 | escalated×5 | 5/0 | 12.1 | 47.3 |
| `architecture_tensor_mismatch` | escalated | 5/5 | escalated×5 | 0/5 | 6.0 | 32.4 |
| `thrashing_bimodal_trap` | exhausted | 5/5 | exhausted×5 | 0/5 | 3.0 | 30.9 |
| `clean_run` | passed | 5/5 | passed×5 | 5/0 | 11.8 | 17.7 |

## MTTR decomposition (median seconds over runs with an incident)

| Scenario | T_det | T_reason | T_act | T_handle_other | T_recover | residual | MTTR median [min–max] | LLM calls | in/out tokens |
|---|---|---|---|---|---|---|---|---|---|
| `nan_divergence` | 0.24 | 12.16 | 0.0000 | 0.00 | 0.13 | 0.00 | 12.5 [12.0–14.0] | 13 | 4354/1020 |
| `batch_oom` | 0.21 | 12.90 | 0.0000 | 0.00 | 0.15 | 0.00 | 13.2 [13.0–13.7] | 13 | 4295/1033 |
| `compound_oom_and_explosion` | 0.35 | 18.68 | 0.0000 | 0.00 | 0.14 | 0.00 | 19.2 [18.3–19.4] | 18 | 6354/1527 |
| `cascaded_lr_and_memory` | 0.35 | 18.79 | 0.0000 | 0.00 | 0.15 | 0.00 | 19.3 [18.3–19.9] | 18 | 6385/1460 |
| `delayed_stochastic_explosion` | 0.18 | 6.10 | 0.0000 | 0.00 | 0.14 | 0.00 | 6.4 [6.2–6.9] | 8 | 2363/618 |
| `spot_worker_churn` | 2.95 | 7.02 | 0.5011 | 0.00 | 0.85 | 0.00 | 11.3 [11.1–11.7] | 8 | 2622/616 |
| `triple_cascade_oom_lr_grad` | 0.49 | 24.63 | 0.0000 | 0.00 | 0.00 | 0.00 | 25.1 [24.5–25.6] | 20 | 8120/1753 |
| `catastrophic_label_noise` | 17.82 | 25.50 | 0.0000 | 0.00 | 0.00 | 0.00 | 43.2 [42.9–44.0] | 21 | 6726/2164 |
| `architecture_tensor_mismatch` | 3.17 | 25.62 | 0.0000 | 0.00 | 0.00 | 0.00 | 28.5 [27.1–29.2] | 16 | 6615/1803 |
| `thrashing_bimodal_trap` | 0.51 | 26.29 | 0.0000 | 0.00 | 0.00 | 0.00 | 26.8 [26.3–27.7] | 20 | 8486/1861 |
| `clean_run` | – | – | – | – | – | – | – | 3 | 316/168 |

## Remediation sequences per run

- `nan_divergence`: reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate
- `batch_oom`: reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size
- `compound_oom_and_explosion`: reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping
- `cascaded_lr_and_memory`: reduce_batch_size → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → reduce_learning_rate → reduce_learning_rate
- `delayed_stochastic_explosion`: enable_gradient_clipping; enable_gradient_clipping; enable_gradient_clipping; enable_gradient_clipping; enable_gradient_clipping
- `spot_worker_churn`: restart_workers; restart_workers; restart_workers; restart_workers; restart_workers
- `triple_cascade_oom_lr_grad`: reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping
- `catastrophic_label_noise`: (none); (none); (none); (none); (none)
- `architecture_tensor_mismatch`: (none); (none); (none); (none); (none)
- `thrashing_bimodal_trap`: reduce_learning_rate → reduce_learning_rate → reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate → reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate → reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping → reduce_learning_rate; enable_gradient_clipping → reduce_learning_rate → reduce_learning_rate → reduce_learning_rate
- `clean_run`: (none); (none); (none); (none); (none)
