"""AutoPARK Analytical Figure Generator.

Renders paper/report figures exclusively from measured benchmark output:
  - a schema-v2 results JSON written by run_benchmark.py (aggregates + per-run records)
  - per-run trace files (timing.json, step_telemetry.json, action_log.json) from each
    run directory, or from the archived copy under benchmarks/reports/traces/<run_id>/

Figures:
  Fig 1: Measured training-loss traces around injected faults (4 agentic runs)
  Fig 2: Measured incident timeline (stage attempts, Claude calls, actions) for one run
  Fig 4: Median MTTR decomposition per scenario with min-max range, vs. baseline wall time
  Fig 5: Observed outcome counts per scenario across repeats

Usage:
    python -m benchmarks.plot_analytical_graphs [results.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#8a8a85",
    "grid.linewidth": 0.4,
    "grid.color": "#d8d8d3",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Categorical slots in fixed order (validated: scripts/validate_palette.js, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
INK = "#2b2b29"
MUTED = "#6b6b66"
FAIL_SPAN = "#f6d5d4"
HANDLE_SPAN = "#ddd8f6"

TRACES_ROOT = Path("benchmarks/reports/traces")
SCENARIO_LABELS = {
    "nan_divergence": "L1 nan_divergence",
    "batch_oom": "L1 batch_oom",
    "compound_oom_and_explosion": "L2 compound_oom_exp",
    "cascaded_lr_and_memory": "L2 cascaded_lr_mem",
    "delayed_stochastic_explosion": "L3 delayed_stoch_exp",
    "spot_worker_churn": "L3 spot_worker_churn",
    "triple_cascade_oom_lr_grad": "L4 triple_cascade",
    "catastrophic_label_noise": "L5 label_noise",
    "architecture_tensor_mismatch": "L6 tensor_mismatch",
    "thrashing_bimodal_trap": "L7 bimodal_trap",
    "clean_run": "clean_run (no fault)",
}


def latest_results() -> Path:
    for path in sorted(Path("benchmarks/reports").glob("benchmark_results_*.json"), reverse=True):
        if json.loads(path.read_text()).get("schema_version") == 2:
            return path
    raise FileNotFoundError("No schema-v2 benchmark results found; run run_benchmark.py first.")


def load_trace(run_dir: str) -> dict[str, Any]:
    """Load measured trace files for one run (live run dir first, then the archive)."""
    candidates = [Path(run_dir), TRACES_ROOT / Path(run_dir).name]
    for d in candidates:
        if (d / "timing.json").exists():
            def read(name: str, default: Any) -> Any:
                p = d / name
                return json.loads(p.read_text()) if p.exists() else default
            timing = read("timing.json", {})
            return {
                "timeline": timing.get("timeline", []),
                "llm_calls": timing.get("llm_calls", []),
                "steps": read("step_telemetry.json", []),
                "actions": read("action_log.json", []),
            }
    raise FileNotFoundError(f"No trace files for run {run_dir}")


def pick_run(results: dict, scenario_id: str, repeat: int = 0) -> dict:
    return next(r for r in results["runs"] if r["id"] == scenario_id and r["repeat"] == repeat)


def _save(fig, name: str, out_pdf_dir: Path, out_png_dir: Path) -> None:
    out_pdf_dir.mkdir(parents=True, exist_ok=True)
    out_png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_png_dir / f"{name}.png", bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"✅ {name}")


def plot_fig1_telemetry_series(results: dict, out_pdf_dir: Path, out_png_dir: Path) -> None:
    """Measured per-step training loss around injected faults for four agentic runs."""
    panels = ["nan_divergence", "batch_oom", "delayed_stochastic_explosion", "catastrophic_label_noise"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharey=False)
    for ax, sid in zip(axes.flat, panels):
        run = pick_run(results, sid)
        tr = load_trace(run["agentic"]["run_dir"])
        attempts = [e for e in tr["timeline"] if e["type"] == "attempt" and e["stage"] == "training"]
        t0 = attempts[0]["start"] if attempts else 0.0
        for e in tr["timeline"]:
            if e["stage"] != "training":
                continue
            if e["type"] == "attempt" and e["status"] == "failed":
                ax.axvspan(e["start"] - t0, e["end"] - t0, color=FAIL_SPAN, lw=0)
            elif e["type"] == "handle":
                ax.axvspan(e["start"] - t0, e["end"] - t0, color=HANDLE_SPAN, lw=0)
        healthy = [p for p in tr["steps"] if p.get("event") == "step" and p.get("loss") is not None]
        if healthy:
            ax.plot([p["timestamp"] - t0 for p in healthy], [p["loss"] for p in healthy],
                    color=SERIES[0], marker="o", ms=1.6, lw=1.0)
        for a in tr["actions"]:
            if a.get("start") is not None and a.get("status") in ("success", "retry_requested"):
                ax.axvline(a["start"] - t0, color=SERIES[2], lw=1.0)
        injected = [p for p in tr["steps"] if str(p.get("event", "")).startswith("injected")]
        for p in injected:
            ax.axvline(p["timestamp"] - t0, color=SERIES[1], lw=0.8, ls=":")
        ax.set_title(f"{SCENARIO_LABELS[sid]} — outcome: {run['outcome']}", color=INK)
        ax.set_xlabel("Seconds since first training attempt")
        ax.set_ylabel("Training loss (measured)")
        ax.grid(True, axis="y")
    handles = [
        plt.Line2D([], [], color=SERIES[0], marker="o", ms=3, lw=1.0, label="Healthy-step loss"),
        plt.Rectangle((0, 0), 1, 1, color=FAIL_SPAN, label="Failed attempt"),
        plt.Rectangle((0, 0), 1, 1, color=HANDLE_SPAN, label="Incident handling (Claude)"),
        plt.Line2D([], [], color=SERIES[1], lw=0.8, ls=":", label="Injected fault fires"),
        plt.Line2D([], [], color=SERIES[2], lw=1.0, label="Remediation actuated"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig1_telemetry_series", out_pdf_dir, out_png_dir)


def plot_fig2_agent_waterfall(results: dict, out_pdf_dir: Path, out_png_dir: Path,
                              scenario_id: str = "triple_cascade_oom_lr_grad") -> None:
    """Measured incident timeline: stage attempts, each Claude call by caller, and actions."""
    run = pick_run(results, scenario_id)
    tr = load_trace(run["agentic"]["run_dir"])
    lanes = ["Training attempts", "Claude: analyze_epoch", "Claude: diagnose",
             "Claude: synthesis", "Claude: recommend_action", "Remediator action"]
    caller_lane = {"analyze_epoch": 1, "diagnose": 2, "handle_anomaly": 3, "recommend_action": 4}
    train = [e for e in tr["timeline"] if e["type"] == "attempt" and e["stage"] == "training"]
    t0 = train[0]["start"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for e in train:
        color = SERIES[1] if e["status"] == "failed" else SERIES[2]
        ax.broken_barh([(e["start"] - t0, max(e["duration_s"], 0.05))], (0 - 0.3, 0.6), color=color)
    for c in tr["llm_calls"]:
        lane = caller_lane.get(c["caller"])
        if lane is not None and c["end"] >= t0:
            ax.broken_barh([(c["start"] - t0, c["duration_s"])], (lane - 0.3, 0.6), color=SERIES[0])
    for a in tr["actions"]:
        if a.get("start") is not None:
            ax.plot(a["start"] - t0, 5, marker="D", ms=5, color=SERIES[4])
            ax.annotate(a["action"], (a["start"] - t0, 5), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=6, color=MUTED)
    t = run["agentic"]["timing"]
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes)
    ax.invert_yaxis()
    ax.set_ylim(len(lanes) - 0.4, -0.8)
    ax.set_xlabel("Seconds since first training attempt (measured)")
    ax.grid(True, axis="x")
    ax.set_title(
        f"{SCENARIO_LABELS[scenario_id]} (repeat 0): MTTR {t['mttr_s']:.1f}s = "
        f"T_det {t['t_det']:.2f} + T_reason {t['t_reason']:.1f} + T_act {t['t_act']:.3f} + "
        f"T_recover {t['t_recover']:.2f} + other {t['t_handle_other'] + t['residual']:.2f}",
        color=INK, fontsize=8,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES[1], label="Failed attempt"),
        plt.Rectangle((0, 0), 1, 1, color=SERIES[2], label="Successful attempt"),
        plt.Rectangle((0, 0), 1, 1, color=SERIES[0], label="Claude call"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, ncol=3)
    fig.tight_layout()
    _save(fig, "fig2_agent_waterfall", out_pdf_dir, out_png_dir)


def plot_fig4_mttr_breakdown(results: dict, out_pdf_dir: Path, out_png_dir: Path) -> None:
    """Median MTTR components per scenario, min-max MTTR range, and baseline wall time."""
    rows = [r for r in results["aggregates"] if r["mttr_s"]]
    labels = [SCENARIO_LABELS.get(r["id"], r["id"]) for r in rows]
    comps = [
        ("T_det", lambda r: r["t_det"]["median"]),
        ("T_reason (Claude)", lambda r: r["t_reason"]["median"]),
        ("T_act", lambda r: r["t_act"]["median"]),
        ("T_recover", lambda r: r["t_recover"]["median"]),
        ("Other", lambda r: r["t_handle_other"]["median"] + r["residual"]["median"]),
    ]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(rows) + 1.0))
    left = np.zeros(len(rows))
    for i, (name, fn) in enumerate(comps):
        vals = np.array([max(fn(r), 0.0) for r in rows])
        ax.barh(y, vals, left=left, height=0.55, color=SERIES[i], label=name,
                edgecolor="white", linewidth=0.8)
        left += vals
    med = np.array([r["mttr_s"]["median"] for r in rows])
    lo = np.array([r["mttr_s"]["min"] for r in rows])
    hi = np.array([r["mttr_s"]["max"] for r in rows])
    ax.errorbar(med, y, xerr=[med - lo, hi - med], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5,
                label="MTTR min–max")
    base = [r["baseline_duration_s"]["median"] for r in rows]
    ax.scatter(base, y, marker="|", s=90, color=INK, linewidths=1.6, zorder=5,
               label="Baseline wall time (median)")
    for yi, m, r in zip(y, hi, rows):
        n_inc = r["mttr_s"]["n"]
        ax.text(m + 1.0, yi, f"{r['mttr_s']['median']:.1f}s (n={n_inc})", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds (measured; stacked bars are per-component medians)")
    ax.grid(True, axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 if len(rows) > 6 else -0.3), ncol=4, frameon=False)
    fig.tight_layout()
    _save(fig, "fig4_mttr_breakdown", out_pdf_dir, out_png_dir)


def plot_fig5_frontier_map(results: dict, out_pdf_dir: Path, out_png_dir: Path) -> None:
    """Observed agentic outcome counts per scenario across repeats."""
    order = ["healed", "passed", "escalated", "exhausted", "recovered", "failed"]
    rows = results["aggregates"]
    present = [o for o in order if any(o in r["outcomes"] for r in rows)]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(rows) + 1.0))
    left = np.zeros(len(rows))
    for i, o in enumerate(present):
        vals = np.array([r["outcomes"].get(o, 0) for r in rows])
        ax.barh(y, vals, left=left, height=0.55, color=(SERIES + ["#8a8a85"])[i], label=o,
                edgecolor="white", linewidth=0.8)
        left += vals
    for yi, r in zip(y, rows):
        ax.text(r["n"] + 0.1, yi, f"expected: {r['expected_outcome']} ({r['matches_expected']}/{r['n']})",
                va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([SCENARIO_LABELS.get(r["id"], r["id"]) for r in rows])
    ax.invert_yaxis()
    n_max = max(r["n"] for r in rows)
    ax.set_xlim(0, n_max * 1.9)
    ax.set_xticks(range(n_max + 1))
    ax.set_xlabel("Runs")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, "fig5_frontier_map", out_pdf_dir, out_png_dir)


def plot_fig6_env_comparison(local: dict, eks: dict, out_pdf_dir: Path, out_png_dir: Path) -> None:
    """Median MTTR [min-max] per scenario: local reference run vs. EKS replication."""
    erows = {r["id"]: r for r in eks["aggregates"]}
    rows = [(r, erows[r["id"]]) for r in local["aggregates"] if r["mttr_s"] and erows.get(r["id"], {}).get("mttr_s")]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(rows) + 1.0))
    for offset, idx, name, color, marker in ((-0.15, 0, "Local (Apple Silicon)", SERIES[0], "o"),
                                             (0.15, 1, "EKS (c6i.xlarge, us-east-1)", SERIES[1], "s")):
        med = np.array([pair[idx]["mttr_s"]["median"] for pair in rows])
        lo = np.array([pair[idx]["mttr_s"]["min"] for pair in rows])
        hi = np.array([pair[idx]["mttr_s"]["max"] for pair in rows])
        ax.errorbar(med, y + offset, xerr=[med - lo, hi - med], fmt=marker, ms=5, color=color,
                    ecolor=color, elinewidth=1.0, capsize=2, label=name)
    ax.set_yticks(y)
    ax.set_yticklabels([SCENARIO_LABELS.get(pair[0]["id"], pair[0]["id"]) for pair in rows])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("MTTR / time to termination, seconds (median, min–max over 5 runs)")
    ax.grid(True, axis="x")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, "fig6_env_comparison", out_pdf_dir, out_png_dir)


def generate_all_figures(results_path: Path | None = None,
                         out_pdf_dir: Path = Path("paper/figures"),
                         out_png_dir: Path = Path("benchmarks/figures"),
                         compare_path: Path | None = None) -> None:
    results_path = results_path or latest_results()
    results = json.loads(Path(results_path).read_text())
    print(f"🎨 Rendering figures from {results_path}")
    plot_fig1_telemetry_series(results, out_pdf_dir, out_png_dir)
    plot_fig2_agent_waterfall(results, out_pdf_dir, out_png_dir)
    plot_fig4_mttr_breakdown(results, out_pdf_dir, out_png_dir)
    plot_fig5_frontier_map(results, out_pdf_dir, out_png_dir)
    if compare_path is not None:
        plot_fig6_env_comparison(results, json.loads(Path(compare_path).read_text()), out_pdf_dir, out_png_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None, help="EKS results JSON to compare against")
    a = ap.parse_args()
    generate_all_figures(a.results, compare_path=a.compare)
