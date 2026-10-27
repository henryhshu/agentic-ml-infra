# AutoPARK Benchmark Report (measured)

- **Generated at**: `2026-10-27T10:12:25.325632+00:00`
- **Environment**: macOS-26.6.2-arm64-arm-64bit · Python 3.11.15 · torch 2.11.0 · device `cpu`
- **Workload**: model `tiny`, 3 epochs, CIFAR-10 subset 5000, max_retries=3
- **Agent LLM**: `claude-haiku-4-5-20251001`
- **Repeats per scenario**: 5

All faults are injected in software by `pipeline/training.py` (see `benchmarks/scenarios.py`).
Timing components are measured from wall-clock timestamps; definitions are in `benchmarks/harness.py`.

## Outcomes

| Scenario | Expected | Matches | Observed outcomes | Baseline (success/failed) | Baseline wall s (median) | Agentic wall s (median) |
|---|---|---|---|---|---|---|
| `nan_divergence` | healed | 5/5 | healed×5 | 0/5 | 2.0 | 39.1 |
| `batch_oom` | healed | 5/5 | healed×5 | 0/5 | 1.8 | 40.1 |
| `compound_oom_and_explosion` | healed | 5/5 | healed×5 | 0/5 | 1.7 | 47.0 |
| `cascaded_lr_and_memory` | healed | 3/5 | exhausted×2, healed×3 | 0/5 | 1.7 | 45.6 |
| `delayed_stochastic_explosion` | healed | 5/5 | healed×5 | 0/5 | 2.6 | 31.7 |
| `spot_worker_churn` | healed | 5/5 | healed×5 | 0/5 | 1.7 | 32.1 |
| `triple_cascade_oom_lr_grad` | healed | 0/5 | exhausted×5 | 0/5 | 1.7 | 31.2 |
| `catastrophic_label_noise` | escalated | 5/5 | escalated×5 | 5/0 | 19.5 | 66.4 |
| `architecture_tensor_mismatch` | escalated | 5/5 | escalated×5 | 0/5 | 1.8 | 30.9 |
| `thrashing_bimodal_trap` | exhausted | 5/5 | exhausted×5 | 0/5 | 1.9 | 33.1 |
| `clean_run` | passed | 5/5 | passed×5 | 5/0 | 19.8 | 24.3 |

## MTTR decomposition (median seconds over runs with an incident)

| Scenario | T_det | T_reason | T_act | T_handle_other | T_recover | residual | MTTR median [min–max] | LLM calls | in/out tokens |
|---|---|---|---|---|---|---|---|---|---|
| `nan_divergence` | 0.13 | 14.80 | 0.0000 | 0.00 | 0.13 | 0.00 | 15.1 [14.3–15.8] | 13 | 4335/1008 |
| `batch_oom` | 0.01 | 15.73 | 0.0000 | 0.00 | 0.11 | 0.00 | 15.9 [14.9–17.1] | 13 | 4264/1016 |
| `compound_oom_and_explosion` | 0.14 | 23.38 | 0.0001 | 0.00 | 0.12 | 0.00 | 23.6 [22.0–23.7] | 18 | 6312/1475 |
| `cascaded_lr_and_memory` | 0.15 | 23.27 | 0.0001 | 0.00 | 0.11 | 0.00 | 23.5 [22.2–31.0] | 18 | 6407/1495 |
| `delayed_stochastic_explosion` | 0.19 | 8.01 | 0.0000 | 0.00 | 0.12 | 0.00 | 8.3 [7.6–8.5] | 8 | 2363/619 |
| `spot_worker_churn` | 0.00 | 7.80 | 0.0000 | 0.00 | 0.13 | 0.00 | 7.9 [7.0–9.4] | 8 | 2293/589 |
| `triple_cascade_oom_lr_grad` | 0.28 | 29.23 | 0.0001 | 0.00 | 0.00 | 0.00 | 29.5 [29.0–31.9] | 20 | 8140/1772 |
| `catastrophic_label_noise` | 35.91 | 29.98 | 0.0000 | 0.00 | 0.00 | 0.00 | 64.6 [64.2–78.2] | 21 | 6737/2182 |
| `architecture_tensor_mismatch` | 0.02 | 29.21 | 0.0000 | 0.00 | 0.00 | 0.00 | 29.3 [28.0–29.7] | 16 | 6578/1766 |
| `thrashing_bimodal_trap` | 0.26 | 31.20 | 0.0001 | 0.00 | 0.00 | 0.00 | 31.5 [30.6–36.8] | 20 | 8502/1869 |
| `clean_run` | – | – | – | – | – | – | – | 3 | 316/178 |

## Remediation sequences per run

- `nan_divergence`: reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate
- `batch_oom`: reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size; reduce_batch_size → reduce_batch_size
- `compound_oom_and_explosion`: reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping; reduce_batch_size → reduce_batch_size → enable_gradient_clipping
- `cascaded_lr_and_memory`: reduce_batch_size → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → enable_gradient_clipping → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → enable_gradient_clipping → reduce_learning_rate → reduce_learning_rate
- `delayed_stochastic_explosion`: enable_gradient_clipping; enable_gradient_clipping; enable_gradient_clipping; enable_gradient_clipping; enable_gradient_clipping
- `spot_worker_churn`: restart_workers; restart_workers; restart_workers; restart_workers; restart_workers
- `triple_cascade_oom_lr_grad`: reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → reduce_learning_rate; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping; reduce_batch_size → reduce_learning_rate → reduce_learning_rate → enable_gradient_clipping
- `catastrophic_label_noise`: (none); (none); (none); (none); (none)
- `architecture_tensor_mismatch`: (none); (none); (none); (none); (none)
- `thrashing_bimodal_trap`: reduce_learning_rate → reduce_learning_rate → reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → enable_gradient_clipping → reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → reduce_learning_rate → reduce_learning_rate → reduce_learning_rate; enable_gradient_clipping → reduce_learning_rate → reduce_learning_rate → reduce_learning_rate; reduce_learning_rate → enable_gradient_clipping → reduce_learning_rate → reduce_learning_rate
- `clean_run`: (none); (none); (none); (none); (none)
