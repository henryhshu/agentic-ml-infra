"""Report generator for the Agentic ML Benchmarking Suite.

Every number in the generated reports is read from measured run data in
`BenchmarkSuiteResult`; see benchmarks/harness.py for the timing definitions.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.harness import BenchmarkSuiteResult


def _med(stat: dict[str, float] | None, fmt: str = "{:.2f}") -> str:
    return fmt.format(stat["median"]) if stat else "–"


def _range(stat: dict[str, float] | None) -> str:
    return f"{stat['median']:.1f} [{stat['min']:.1f}–{stat['max']:.1f}]" if stat else "–"


def _outcomes(row: dict[str, Any]) -> str:
    return ", ".join(f"{k}×{v}" for k, v in sorted(row["outcomes"].items()))


def format_terminal_summary(suite: BenchmarkSuiteResult) -> str:
    """Format a terminal summary table of benchmark results."""
    w = 110
    lines = ["\n" + "=" * w, "  AGENTIC ML BENCHMARK RESULTS (measured)", "=" * w]
    lines.append(f"  {'Scenario':<30} {'Expected':<10} {'Match':<6} {'Outcomes':<26} {'MTTR med [min–max] s':<22} {'Baseline s':<10}")
    lines.append("-" * w)
    for row in suite.aggregates:
        lines.append(
            f"  {row['id'][:29]:<30} {row['expected_outcome']:<10} {row['matches_expected']}/{row['n']:<4} "
            f"{_outcomes(row)[:25]:<26} {_range(row['mttr_s']):<22} {_med(row['baseline_duration_s'], '{:.1f}'):<10}"
        )
    lines.append("=" * w + "\n")
    return "\n".join(lines)


def generate_markdown_report(
    suite: BenchmarkSuiteResult,
    output_path: Path | str | None = None,
) -> Path:
    """Generate a Markdown benchmark report from measured results."""
    if output_path is None:
        reports_dir = Path("benchmarks/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"benchmark_report_{suite.timestamp.replace(':', '-').replace('+', '_')}.md"
        out_file = reports_dir / filename
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    env = suite.environment
    lines: list[str] = [
        "# AutoPARK Benchmark Report (measured)",
        "",
        f"- **Generated at**: `{suite.timestamp}`",
        f"- **Environment**: {env['platform']} · Python {env['python']} · torch {env['torch']} · device `{env['device']}`",
        f"- **Workload**: model `{env['model']}`, {env['epochs']} epochs, CIFAR-10 subset {env['subset']}, max_retries={env['max_retries']}",
        f"- **Agent LLM**: `{env['llm_model']}`",
        f"- **Repeats per scenario**: {env['repeats']}",
        "",
        "All faults are injected in software by `pipeline/training.py` (see `benchmarks/scenarios.py`).",
        "Timing components are measured from wall-clock timestamps; definitions are in `benchmarks/harness.py`.",
        "",
        "## Outcomes",
        "",
        "| Scenario | Expected | Matches | Observed outcomes | Baseline (success/failed) | Baseline wall s (median) | Agentic wall s (median) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in suite.aggregates:
        bs = row["baseline_status"]
        lines.append(
            f"| `{row['id']}` | {row['expected_outcome']} | {row['matches_expected']}/{row['n']} | {_outcomes(row)} "
            f"| {bs['success']}/{bs['failed']} | {_med(row['baseline_duration_s'], '{:.1f}')} | {_med(row['agentic_duration_s'], '{:.1f}')} |"
        )

    lines += [
        "",
        "## MTTR decomposition (median seconds over runs with an incident)",
        "",
        "| Scenario | T_det | T_reason | T_act | T_handle_other | T_recover | residual | MTTR median [min–max] | LLM calls | in/out tokens |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in suite.aggregates:
        lines.append(
            f"| `{row['id']}` | {_med(row['t_det'])} | {_med(row['t_reason'])} | {_med(row['t_act'], '{:.4f}')} "
            f"| {_med(row['t_handle_other'])} | {_med(row['t_recover'])} | {_med(row['residual'])} | {_range(row['mttr_s'])} "
            f"| {_med(row['llm_calls'], '{:.0f}')} | {_med(row['input_tokens'], '{:.0f}')}/{_med(row['output_tokens'], '{:.0f}')} |"
        )

    lines += ["", "## Remediation sequences per run", ""]
    for row in suite.aggregates:
        lines.append(f"- `{row['id']}`: " + "; ".join(" → ".join(a) or "(none)" for a in row["actions_per_run"]))

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_file


def save_json_report(
    suite: BenchmarkSuiteResult,
    output_path: Path | str | None = None,
) -> Path:
    """Save raw per-run results and aggregates as structured JSON."""
    if output_path is None:
        reports_dir = Path("benchmarks/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"benchmark_results_{suite.timestamp.replace(':', '-').replace('+', '_')}.json"
        out_file = reports_dir / filename
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "schema_version": 2,
        "timestamp": suite.timestamp,
        "environment": suite.environment,
        "aggregates": suite.aggregates,
        "runs": [
            {
                "id": r.scenario.id,
                "repeat": r.repeat,
                "expected_outcome": r.scenario.expected_outcome,
                "outcome": r.outcome,
                "outcome_matches_expected": r.outcome_matches_expected,
                "baseline": asdict(r.baseline),
                "agentic": asdict(r.agentic),
            }
            for r in suite.scenario_results
        ],
    }
    out_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out_file
