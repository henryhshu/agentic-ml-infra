"""AutoPARK Analytical Figure Generator.

Renders paper/report figures exclusively from measured benchmark output:
  - a schema-v2 results JSON written by run_benchmark.py (aggregates + per-run records)
  - per-run trace files (timing.json, step_telemetry.json, action_log.json) from each
    run directory, or from the archived copy under benchmarks/reports/traces/<run_id>/

Figures:
  Fig 1: Measured training-loss traces around injected faults (4 agentic runs)
  Fig 2: Measured incident timeline (stage attempts, Claude calls, actions) for one run;
         for a multi-node run, the worker-loss incident with the worker kill and liveness check
  Fig 3: Measured incident timeline of the Level 4 triple cascade
  Fig 4: Median MTTR decomposition per scenario with min-max range, vs. baseline wall time
  Fig 5: Observed outcome counts per scenario across repeats
  Fig 6: Per-rank CPU use (cgroup samples) of a multi-node run

Usage:
    python -m benchmarks.plot_analytical_graphs [results.json]
    python -m benchmarks.plot_analytical_graphs --run-dir benchmarks/reports/eks/<run_tag>   # multi-node
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
RANKS: dict[str, Any] = {}  # multi-node run: {"traces": Path, "metrics": {rank: cluster_metrics}}
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
    if RANKS:
        candidates.insert(0, RANKS["traces"] / Path(run_dir).name)
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
                              scenario_id: str = "triple_cascade_oom_lr_grad",
                              name: str = "fig2_agent_waterfall") -> None:
    """Measured incident timeline: stage attempts, each Claude call by caller, and actions.

    For a multi-node run, adds the worker-process kill (from the worker rank's process exits)
    and rank 0's liveness check after the failed collective."""
    run = pick_run(results, scenario_id)
    tr = load_trace(run["agentic"]["run_dir"])
    lanes = ["Training attempts", "Claude: analyze_epoch", "Claude: diagnose",
             "Claude: synthesis", "Claude: recommend_action", "Remediator action"]
    caller_lane = {"analyze_epoch": 1, "diagnose": 2, "handle_anomaly": 3, "recommend_action": 4}
    train = [e for e in tr["timeline"] if e["type"] == "attempt" and e["stage"] == "training"]
    t0 = train[0]["start"]
    t_end = max(e["end"] for e in tr["timeline"])
    lost = [e for e in tr["timeline"] if e["type"] == "worker_lost"]
    kills = [(r, x["exited"]) for r, cm in RANKS.get("metrics", {}).items()
             for x in cm.get("process_exits", []) if x["exit_code"] != 0 and t0 <= x["exited"] <= t_end]
    if lost or kills:
        lanes = ["Worker process"] + lanes
    off = 1 if (lost or kills) else 0
    fig, ax = plt.subplots(figsize=(7.2, 2.6 + 0.3 * off))
    for e in train:
        color = SERIES[1] if e["status"] == "failed" else SERIES[2]
        ax.broken_barh([(e["start"] - t0, max(e["duration_s"], 0.05))], (off - 0.3, 0.6), color=color)
    for e in lost:
        ax.broken_barh([(e["start"] - t0, e["duration_s"])], (off - 0.3, 0.6), color=SERIES[3])
    for r, t in kills:
        ax.plot(t - t0, 0, marker="X", ms=6, color=SERIES[1])
        ax.annotate(f"rank {r} SIGKILL", (t - t0, 0), xytext=(4, -3), textcoords="offset points",
                    ha="left", fontsize=6, color=MUTED)
    for c in tr["llm_calls"]:
        lane = caller_lane.get(c["caller"])
        if lane is not None and c["end"] >= t0:
            ax.broken_barh([(c["start"] - t0, c["duration_s"])], (lane + off - 0.3, 0.6), color=SERIES[0])
    for a in tr["actions"]:
        if a.get("start") is not None:
            ax.plot(a["start"] - t0, 5 + off, marker="D", ms=5, color=SERIES[4])
            ax.annotate(a["action"], (a["start"] - t0, 5 + off), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=6, color=MUTED)
    t = run["agentic"]["timing"]
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes)
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
    if lost:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=SERIES[3], label="Rank-0 liveness check"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, ncol=len(handles))
    fig.tight_layout()
    _save(fig, name, out_pdf_dir, out_png_dir)


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
    for yi, m, bw, r in zip(y, hi, base, rows):
        n_inc = r["mttr_s"]["n"]
        ax.text(max(m, bw) + 1.0, yi, f"{r['mttr_s']['median']:.1f}s (n={n_inc})", va="center", fontsize=6.5, color=MUTED)
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


def plot_fig6_rank_utilization(results: dict, out_pdf_dir: Path, out_png_dir: Path,
                               scenario_id: str = "spot_worker_churn") -> None:
    """Per-rank CPU cores (cgroup v2, 2 s samples): one worker-loss incident, and whole-run medians."""
    metrics = RANKS["metrics"]

    def series(cm):
        s_ = cm["resource_samples"]
        return [((a["t"] + b["t"]) / 2, (b["cpu_usage_usec"] - a["cpu_usage_usec"]) / 1e6 / (b["t"] - a["t"]))
                for a, b in zip(s_, s_[1:]) if b["t"] > a["t"]]

    run = pick_run(results, scenario_id)
    tr = load_trace(run["agentic"]["run_dir"])
    t0 = min(e["start"] for e in tr["timeline"]) - 4
    t1 = max(e["end"] for e in tr["timeline"]) + 4
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [2.2, 1]})
    for e in tr["timeline"]:
        if e["stage"] == "training" and e["type"] == "attempt" and e["status"] == "failed":
            ax.axvspan(e["start"] - t0, e["end"] - t0, color=FAIL_SPAN, lw=0)
        elif e["type"] == "handle":
            ax.axvspan(e["start"] - t0, e["end"] - t0, color=HANDLE_SPAN, lw=0)
    for r in sorted(metrics):
        pts = [(t - t0, c) for t, c in series(metrics[r]) if t0 <= t <= t1]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[r % len(SERIES)], lw=1.2,
                marker="o", ms=2, label=f"rank {r}")
        for x in metrics[r].get("process_exits", []):
            if x["exit_code"] != 0 and t0 + 4 <= x["exited"] <= t1 - 4:
                ax.axvline(x["exited"] - t0, color=SERIES[r % len(SERIES)], lw=0.8, ls=":")
    ax.set_xlabel("Seconds (agentic worker-loss run, repeat 0)")
    ax.set_ylabel("CPU cores used")
    ax.grid(True, axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=4, fontsize=6.5)

    attempt, handle = [], []
    for run_ in results["runs"]:
        for e in load_trace(run_["agentic"]["run_dir"])["timeline"]:
            if e["type"] in ("attempt", "handle"):
                (attempt if e["type"] == "attempt" else handle).append((e["start"], e["end"]))
    ranks = sorted(metrics)
    for j, (ivals, label, color) in enumerate(((attempt, "stage attempts", SERIES[0]),
                                               (handle, "incident handling", SERIES[4]))):
        meds = []
        for r in ranks:
            vals = [c for t, c in series(metrics[r]) if any(s_ <= t <= e_ for s_, e_ in ivals)]
            meds.append(float(np.median(vals)) if vals else 0.0)
        bx.bar(np.arange(len(ranks)) + (j - 0.5) * 0.38, meds, width=0.36, color=color, label=label)
    bx.set_xticks(range(len(ranks)))
    bx.set_xticklabels([f"rank {r}" for r in ranks])
    bx.set_ylabel("Median CPU cores")
    bx.grid(True, axis="y")
    bx.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=1, fontsize=6.5)
    fig.tight_layout()
    _save(fig, "fig6_rank_utilization", out_pdf_dir, out_png_dir)


def generate_all_figures(results_path: Path | None = None,
                         out_pdf_dir: Path = Path("paper/figures"),
                         out_png_dir: Path = Path("benchmarks/figures"),
                         run_dir: Path | None = None) -> None:
    if run_dir is not None:
        # multi-node run fetched by benchmarks/eks_listener.py
        results_path = run_dir / "bundle" / "results.json"
        RANKS["traces"] = run_dir / "bundle" / "traces"
        RANKS["metrics"] = {0: json.loads((run_dir / "bundle" / "cluster_metrics.json").read_text())}
        for d in sorted((run_dir / "ranks").glob("rank*")):
            RANKS["metrics"][int(d.name[4:])] = json.loads((d / "cluster_metrics.json").read_text())
    results_path = results_path or latest_results()
    results = json.loads(Path(results_path).read_text())
    print(f"🎨 Rendering figures from {results_path}")
    plot_fig1_telemetry_series(results, out_pdf_dir, out_png_dir)
    if RANKS:
        plot_fig2_agent_waterfall(results, out_pdf_dir, out_png_dir, "spot_worker_churn")
        plot_fig2_agent_waterfall(results, out_pdf_dir, out_png_dir, name="fig3_cascade_waterfall")
        plot_fig6_rank_utilization(results, out_pdf_dir, out_png_dir)
    else:
        plot_fig2_agent_waterfall(results, out_pdf_dir, out_png_dir)
    plot_fig4_mttr_breakdown(results, out_pdf_dir, out_png_dir)
    plot_fig5_frontier_map(results, out_pdf_dir, out_png_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", type=Path, default=None)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="multi-node run directory fetched by benchmarks/eks_listener.py")
    a = ap.parse_args()
    generate_all_figures(a.results, run_dir=a.run_dir)
