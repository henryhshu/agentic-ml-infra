# Benchmark data provenance

Published performance figures use schema-v2 result files and their recorded run
traces. The current four-node EKS dataset is
[`eks/eks-20261027T131911Z/`](eks/eks-20261027T131911Z/): `bundle/results.json`,
`bundle/traces/`, and per-rank `cluster_metrics.json` contain the source records.
The repository README identifies the workload and limitations of that run.

Regenerate its figures with:

```sh
python -m benchmarks.plot_analytical_graphs --run-dir benchmarks/reports/eks/eks-20261027T131911Z
```

Plots read losses, timestamps, outcomes, and resource samples from those files.
Missing resource samples remain missing; they are not reported as zero usage.
Architecture diagrams describe the implementation and contain no measured series.

Legacy schema-v1 result files and their Markdown reports were removed because
they lacked the timing traces required by the current methodology and included
unsupported derived claims such as “compute saved.” They remain in Git history,
but must not be used as evidence for performance claims.

Fault injection is a controlled test workload. Simulated OOM/NaN/gradient
exceptions and their signatures are explicitly injected, while plotted telemetry
comes from actual computation. Synthetic datasets and mocked test fixtures are
for testing; they are not evidence of production performance. Archived LLM
diagnoses are model output, not independently verified descriptions of events.
