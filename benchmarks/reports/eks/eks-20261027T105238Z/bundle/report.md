# AutoPARK Benchmark Report (measured)

- **Generated at**: `2026-10-27T11:35:13.297620+00:00`
- **Environment**: Linux-6.12.111-136.204.amzn2023.x86_64-x86_64-with-glibc2.41 · Python 3.11.15 · torch 2.11.0+cpu · device `cpu`
- **Workload**: model `tiny`, 3 epochs, CIFAR-10 subset 5000, max_retries=3
- **Agent LLM**: `claude-haiku-4-5-20251001`
- **Repeats per scenario**: 5

All faults are injected in software by `pipeline/training.py` (see `benchmarks/scenarios.py`).
Timing components are measured from wall-clock timestamps; definitions are in `benchmarks/harness.py`.

## Outcomes

| Scenario | Expected | Matches | Observed outcomes | Baseline (success/failed) | Baseline wall s (median) | Agentic wall s (median) |
|---|---|---|---|---|---|---|
| `nan_divergence` | healed | 5/5 | healed×5 | 0/5 | 3.1 | 39.9 |
| `batch_oom` | healed | 5/5 | healed×5 | 0/5 | 2.6 | 42.2 |
| `compound_oom_and_explosion` | healed | 5/5 | healed×5 | 0/5 | 2.7 | 47.7 |
| `cascaded_lr_and_memory` | healed | 4/5 | exhausted×1, healed×4 | 0/5 | 2.7 | 45.8 |
| `delayed_stochastic_explosion` | healed | 5/5 | healed×5 | 0/5 | 3.6 | 35.5 |
| `spot_worker_churn` | healed | 5/5 | healed×5 | 0/5 | 2.7 | 35.4 |
| `triple_cascade_oom_lr_grad` | healed | 0/5 | exhausted×5 | 0/5 | 2.8 | 27.1 |
| `catastrophic_label_noise` | escalated | 5/5 | escalated×5 | 5/0 | 25.9 | 68.6 |
| `architecture_tensor_mismatch` | escalated | 5/5 | escalated×5 | 0/5 | 2.7 | 27.9 |
| `thrashing_bimodal_trap` | exhausted | 5/5 | exhausted×5 | 0/5 | 3.1 | 28.6 |
| `clean_run` | passed | 5/5 | passed×5 | 5/0 | 24.3 | 27.4 |

## MTTR decomposition (median seconds over runs with an incident)

| Scenario | T_det | T_reason | T_act | T_handle_other | T_recover | residual | MTTR median [min–max] | LLM calls | in/out tokens |
|---|---|---|---|---|---|---|---|---|---|
| `nan_divergence` | 0.16 | 11.52 | 0.0000 | 0.00 | 0.12 | 0.00 | 11.8 [11.5–13.0] | 13 | 4318/1011 |
| `batch_oom` | 0.01 | 12.68 | 0.0000 | 0.00 | 0.12 | 0.00 | 12.8 [12.6–13.9] | 13 | 4281/1043 |
| `compound_oom_and_explosion` | 0.13 | 18.44 | 0.0000 | 0.00 | 0.12 | 0.00 | 18.7 [18.1–19.7] | 18 | 6320/1478 |
| `cascaded_lr_and_memory` | 0.17 | 18.51 | 0.0000 | 0.00 | 0.11 | 0.00 | 18.8 [18.4–23.6] | 18 | 6401/1487 |
| `delayed_stochastic_explosion` | 0.24 | 6.21 | 0.0000 | 0.00 | 0.12 | 0.00 | 6.6 [6.4–7.1] | 8 | 2366/633 |
| `spot_worker_churn` | 0.00 | 6.22 | 0.0000 | 0.00 | 0.12 | 0.00 | 6.4 [6.2–12.6] | 8 | 2306/591 |
| `triple_cascade_oom_lr_grad` | 0.28 | 24.18 | 0.0000 | 0.00 | 0.00 | 0.00 | 24.5 [23.4–25.2] | 20 | 8123/1756 |
| `catastrophic_label_noise` | 41.13 | 25.19 | 0.0000 | 0.00 | 0.00 | 0.00 | 65.9 [63.0–76.9] | 21 | 6715/2132 |
| `architecture_tensor_mismatch` | 0.01 | 24.55 | 0.0000 | 0.00 | 0.00 | 0.00 | 24.6 [24.5–25.2] | 16 | 6529/1717 |
| `thrashing_bimodal_trap` | 0.34 | 25.48 | 0.0000 | 0.00 | 0.00 | 0.00 | 25.8 [24.8–26.5] | 20 | 8489/1853 |
| `clean_run` | – | – | – | – | – | – | – | 3 | 316/175 |

## Remediation sequences per run

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
