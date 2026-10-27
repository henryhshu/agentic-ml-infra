"""Consolidated analysis of an EKS benchmark run against the local reference run.

Reads the bundle fetched by benchmarks/eks_listener.py plus the Kubernetes analytics, and the
local results JSON + archived traces, and writes analysis.json and analysis.md covering:
  - environment (host/node facts) and pod lifecycle (scheduling, image pull, bootstrap, run)
  - container resource usage from cgroup samples, split by training vs. incident handling
  - per-scenario outcomes and MTTR components, local vs. EKS
  - Claude call latency distribution, overall and per caller
  - training step time and fault-free monitoring overhead
Every number is computed from recorded data; nothing is estimated.
"""
from __future__ import annotations

import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Any

LOCAL_TRACES = Path("benchmarks/reports/traces")


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dist(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values), "p90": pct(values, 0.9),
            "mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def iso(ts: str | None) -> float | None:
    if not ts:
        return None
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def load_traces(results: dict, traces_root: Path) -> dict[str, dict]:
    """run_id -> {'timeline','llm_calls','steps'} for every run in a results file."""
    out = {}
    for run in results["runs"]:
        for mode in ("baseline", "agentic"):
            rid = Path(run[mode]["run_dir"]).name
            d = traces_root / rid
            if not (d / "timing.json").exists():
                continue
            timing = json.loads((d / "timing.json").read_text())
            steps = json.loads((d / "step_telemetry.json").read_text()) if (d / "step_telemetry.json").exists() else []
            out[rid] = {"id": run["id"], "mode": mode, "timeline": timing["timeline"],
                        "llm_calls": timing["llm_calls"], "steps": steps}
    return out


def llm_latency(traces: dict[str, dict]) -> dict[str, Any]:
    calls = [c for t in traces.values() if t["mode"] == "agentic" for c in t["llm_calls"]]
    by_caller: dict[str, list[float]] = {}
    for c in calls:
        by_caller.setdefault(c["caller"], []).append(c["duration_s"])
    return {"all": dist([c["duration_s"] for c in calls]),
            "by_caller": {k: dist(v) for k, v in sorted(by_caller.items())}}


def step_time(traces: dict[str, dict]) -> dict[str, Any] | None:
    """Median wall time between consecutive healthy steps in fault-free baseline runs."""
    gaps = []
    for t in traces.values():
        if t["id"] != "clean_run" or t["mode"] != "baseline":
            continue
        steps = [p for p in t["steps"] if p.get("event") == "step"]
        for a, b in zip(steps, steps[1:]):
            if a["epoch"] == b["epoch"] and b["step"] == a["step"] + 1:
                gaps.append(b["timestamp"] - a["timestamp"])
    return dist(gaps)


def cpu_series(samples: list[dict]) -> list[tuple[float, float, float]]:
    """(t_mid, cores_used, memory_bytes) from consecutive cumulative cgroup samples."""
    out = []
    for a, b in zip(samples, samples[1:]):
        dt_s = b["t"] - a["t"]
        if dt_s > 0:
            out.append(((a["t"] + b["t"]) / 2, (b["cpu_usage_usec"] - a["cpu_usage_usec"]) / 1e6 / dt_s, b["memory_bytes"]))
    return out


def resource_split(series: list[tuple[float, float, float]], traces: dict[str, dict]) -> dict[str, Any]:
    """CPU use while a stage attempt executes vs. while agents handle an incident (Claude waits)."""
    attempt, handle = [], []
    for t in traces.values():
        if t["mode"] != "agentic":
            continue
        for e in t["timeline"]:
            if e["type"] in ("attempt", "handle"):
                (attempt if e["type"] == "attempt" else handle).append((e["start"], e["end"]))

    def within(ivals):
        return [c for (tm, c, _) in series if any(s <= tm <= e for s, e in ivals)]

    return {"cores_during_agentic_attempts": dist(within(attempt)),
            "cores_during_incident_handling": dist(within(handle))}


def scenario_rows(local: dict, eks: dict) -> list[dict]:
    lrows = {r["id"]: r for r in local["aggregates"]}
    rows = []
    for e in eks["aggregates"]:
        l = lrows.get(e["id"], {})

        def med(row, key):
            v = row.get(key)
            return v["median"] if v else None

        def rng(row, key):
            v = row.get(key)
            return [v["min"], v["max"]] if v else None

        rows.append({
            "id": e["id"],
            "expected": e["expected_outcome"],
            "local_outcomes": l.get("outcomes"), "eks_outcomes": e["outcomes"],
            "local_match": l.get("matches_expected"), "eks_match": e["matches_expected"], "n": e["n"],
            "local_mttr": med(l, "mttr_s"), "eks_mttr": med(e, "mttr_s"),
            "local_mttr_range": rng(l, "mttr_s"), "eks_mttr_range": rng(e, "mttr_s"),
            "local_t_reason": med(l, "t_reason"), "eks_t_reason": med(e, "t_reason"),
            "local_t_det": med(l, "t_det"), "eks_t_det": med(e, "t_det"),
            "local_baseline_s": med(l, "baseline_duration_s"), "eks_baseline_s": med(e, "baseline_duration_s"),
            "local_agentic_s": med(l, "agentic_duration_s"), "eks_agentic_s": med(e, "agentic_duration_s"),
            "eks_actions": e["actions_per_run"],
        })
    return rows


def analyze(run_dir: Path, local_results_path: Path) -> Path:
    bundle = run_dir / "bundle"
    eks = json.loads((bundle / "results.json").read_text())
    local = json.loads(local_results_path.read_text())
    k8s = json.loads((run_dir / "k8s_analytics.json").read_text())
    cm = json.loads((bundle / "cluster_metrics.json").read_text())

    eks_traces = load_traces(eks, bundle / "traces")
    local_traces = load_traces(local, LOCAL_TRACES)

    pod = k8s.get("pod", {})
    t_created, t_sched = iso(pod.get("created")), iso(pod.get("scheduled"))
    t_cstart, t_cend = iso(pod.get("container_started")), iso(pod.get("container_finished"))
    lifecycle = {
        "pending_to_scheduled_s": (t_sched - t_created) if t_created and t_sched else None,
        "scheduled_to_container_start_s": (t_cstart - t_sched) if t_sched and t_cstart else None,
        "bootstrap_s": (cm["entrypoint_started_at"] - t_cstart) if t_cstart else None,
        "benchmark_s": cm["benchmark_finished_at"] - cm["benchmark_started_at"],
        "bundle_and_upload_s": (t_cend - cm["benchmark_finished_at"]) if t_cend else None,
        "pod_total_s": (t_cend - t_created) if t_created and t_cend else None,
        "image_pull_events": [e for e in k8s.get("events", []) if e.get("reason") in ("Pulling", "Pulled")],
    }

    series = cpu_series(cm["resource_samples"])
    mem = [m for (_, _, m) in series]
    resources = {
        "samples": len(cm["resource_samples"]),
        "cores_used": dist([c for (_, c, _) in series]),
        "memory_mib": dist([m / 2**20 for m in mem]),
        "memory_peak_mib": (int(cm["host"]["memory_peak_bytes"]) / 2**20) if (cm["host"].get("memory_peak_bytes") or "").isdigit() else None,
        "cpu_throttled_s": (cm["resource_samples"][-1]["cpu_throttled_usec"] - cm["resource_samples"][0]["cpu_throttled_usec"]) / 1e6 if cm["resource_samples"] else None,
        **resource_split(series, eks_traces),
    }

    def clean_overhead(res):
        row = next(r for r in res["aggregates"] if r["id"] == "clean_run")
        b, a = row["baseline_duration_s"]["median"], row["agentic_duration_s"]["median"]
        return {"baseline_s": b, "agentic_s": a, "overhead_pct": 100 * (a - b) / b,
                "monitor_llm_s": statistics.median(r["agentic"]["timing"]["monitor_llm_s"] for r in res["runs"] if r["id"] == "clean_run")}

    def healed(res, levels):
        rs = [r for r in res["runs"] if r["id"] in levels]
        return sum(r["outcome"] == "healed" for r in rs), len(rs)

    l13 = ["nan_divergence", "batch_oom", "compound_oom_and_explosion", "cascaded_lr_and_memory",
           "delayed_stochastic_explosion", "spot_worker_churn"]
    analysis = {
        "run_tag": k8s.get("run_tag"),
        "code_sha": cm["host"].get("code_sha"),
        "environment": {"local": local["environment"], "eks": {**eks["environment"], **cm["host"]}, "node": k8s.get("node")},
        "lifecycle": lifecycle,
        "resources": resources,
        "scenarios": scenario_rows(local, eks),
        "levels_1_3_healed": {"local": healed(local, l13), "eks": healed(eks, l13)},
        "outcome_agreement": sum(
            1 for r in scenario_rows(local, eks) if r["local_outcomes"] == r["eks_outcomes"]),
        "llm_latency": {"local": llm_latency(local_traces), "eks": llm_latency(eks_traces)},
        "step_time_s": {"local": step_time(local_traces), "eks": step_time(eks_traces)},
        "clean_overhead": {"local": clean_overhead(local), "eks": clean_overhead(eks)},
        "tokens_total": {
            name: {"input": sum(r["agentic"]["timing"].get("input_tokens", 0) for r in res["runs"]),
                   "output": sum(r["agentic"]["timing"].get("output_tokens", 0) for r in res["runs"]),
                   "calls": sum(r["agentic"]["timing"].get("llm_calls", 0) for r in res["runs"])}
            for name, res in (("local", local), ("eks", eks))
        },
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    md = run_dir / "analysis.md"
    md.write_text(render_markdown(analysis), encoding="utf-8")
    return md


def _f(v, fmt="{:.1f}"):
    return fmt.format(v) if isinstance(v, (int, float)) else "–"


def _oc(o):
    return ", ".join(f"{k} {v}" for k, v in sorted((o or {}).items(), key=lambda x: -x[1])) or "–"


def render_markdown(a: dict) -> str:
    env_l, env_e, node = a["environment"]["local"], a["environment"]["eks"], a["environment"]["node"] or {}
    lc, rs = a["lifecycle"], a["resources"]
    lat_l, lat_e = a["llm_latency"]["local"]["all"], a["llm_latency"]["eks"]["all"]
    st_l, st_e = a["step_time_s"]["local"], a["step_time_s"]["eks"]
    co_l, co_e = a["clean_overhead"]["local"], a["clean_overhead"]["eks"]
    h_l, h_e = a["levels_1_3_healed"]["local"], a["levels_1_3_healed"]["eks"]
    L = [
        f"# EKS replication analysis — `{a['run_tag']}`",
        "",
        f"Code: `{a['code_sha']}`. All numbers below are computed from recorded data by `benchmarks/eks_analysis.py`.",
        "",
        "## Environment",
        "",
        "| | Local reference | EKS |",
        "|---|---|---|",
        f"| Host | {env_l['platform']} | {node.get('instance_type')} ({node.get('capacity_type')}, {node.get('zone')}), {node.get('os_image')} |",
        f"| CPU | Apple Silicon ({env_l['machine']}) | {env_e.get('cpu_model')} ({env_e.get('os_cpu_count')} vCPU) |",
        f"| Python / torch | {env_l['python']} / {env_l['torch']} | {env_e['python']} / {env_e['torch']} (threads={env_e.get('torch_threads')}) |",
        f"| Kubernetes | – | kubelet {node.get('kubelet_version')}, {node.get('container_runtime')} |",
        f"| Workload | {env_l['model']}, {env_l['epochs']} ep, subset {env_l['subset']}, {env_l['repeats']} repeats | same |",
        "",
        "## Pod lifecycle (seconds)",
        "",
        "| Pending→scheduled | Scheduled→container start (image pull) | Bootstrap (code + pip install) | Benchmark | Bundle + upload | Pod total |",
        "|---|---|---|---|---|---|",
        f"| {_f(lc['pending_to_scheduled_s'])} | {_f(lc['scheduled_to_container_start_s'])} | {_f(lc['bootstrap_s'])} | {_f(lc['benchmark_s'])} | {_f(lc['bundle_and_upload_s'])} | {_f(lc['pod_total_s'])} |",
        "",
        "## Container resources (cgroup v2, 2 s samples)",
        "",
        f"- Samples: {rs['samples']}; CPU cores used: median {_f((rs['cores_used'] or {}).get('median'), '{:.2f}')}, p90 {_f((rs['cores_used'] or {}).get('p90'), '{:.2f}')}, max {_f((rs['cores_used'] or {}).get('max'), '{:.2f}')}",
        f"- CPU cores while a stage attempt runs (agentic): median {_f((rs['cores_during_agentic_attempts'] or {}).get('median'), '{:.2f}')}; while agents handle an incident (waiting on Claude): median {_f((rs['cores_during_incident_handling'] or {}).get('median'), '{:.2f}')}",
        f"- Memory: median {_f((rs['memory_mib'] or {}).get('median'), '{:.0f}')} MiB, max sampled {_f((rs['memory_mib'] or {}).get('max'), '{:.0f}')} MiB, cgroup peak {_f(rs['memory_peak_mib'], '{:.0f}')} MiB; CPU throttled {_f(rs['cpu_throttled_s'], '{:.1f}')} s",
        "",
        "## Outcomes and MTTR, local vs. EKS (median s, [min–max])",
        "",
        "| Scenario | Expected | Local outcomes | EKS outcomes | Local MTTR | EKS MTTR | Local T_reason | EKS T_reason | Local / EKS baseline wall | Local / EKS agentic wall |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in a["scenarios"]:
        lr = f"{_f(r['local_mttr'])} [{_f((r['local_mttr_range'] or [None])[0])}–{_f((r['local_mttr_range'] or [None, None])[1])}]" if r["local_mttr"] is not None else "–"
        er = f"{_f(r['eks_mttr'])} [{_f((r['eks_mttr_range'] or [None])[0])}–{_f((r['eks_mttr_range'] or [None, None])[1])}]" if r["eks_mttr"] is not None else "–"
        L.append(f"| `{r['id']}` | {r['expected']} | {_oc(r['local_outcomes'])} | {_oc(r['eks_outcomes'])} | {lr} | {er} | "
                 f"{_f(r['local_t_reason'])} | {_f(r['eks_t_reason'])} | {_f(r['local_baseline_s'])} / {_f(r['eks_baseline_s'])} | {_f(r['local_agentic_s'])} / {_f(r['eks_agentic_s'])} |")
    L += [
        "",
        f"Levels 1–3 healed: local {h_l[0]}/{h_l[1]}, EKS {h_e[0]}/{h_e[1]}. Scenarios with identical outcome counts: {a['outcome_agreement']}/{len(a['scenarios'])}.",
        "",
        "## Claude call latency (all agentic calls)",
        "",
        "| | n | median s | p90 s | mean s | max s |",
        "|---|---|---|---|---|---|",
        f"| Local | {lat_l['n']} | {lat_l['median']:.2f} | {lat_l['p90']:.2f} | {lat_l['mean']:.2f} | {lat_l['max']:.2f} |",
        f"| EKS (us-east-1) | {lat_e['n']} | {lat_e['median']:.2f} | {lat_e['p90']:.2f} | {lat_e['mean']:.2f} | {lat_e['max']:.2f} |",
        "",
        "| Caller | Local median s | EKS median s |",
        "|---|---|---|",
    ]
    for caller, d in a["llm_latency"]["eks"]["by_caller"].items():
        dl = a["llm_latency"]["local"]["by_caller"].get(caller) or {}
        L.append(f"| `{caller}` | {_f(dl.get('median'), '{:.2f}')} | {_f(d['median'], '{:.2f}')} |")
    L += [
        "",
        "## Training speed and monitoring overhead",
        "",
        f"- Median healthy step time (clean_run baseline): local {_f((st_l or {}).get('median'), '{:.3f}')} s, EKS {_f((st_e or {}).get('median'), '{:.3f}')} s",
        f"- Fault-free overhead of agentic monitoring: local {co_l['baseline_s']:.1f} → {co_l['agentic_s']:.1f} s (+{co_l['overhead_pct']:.0f}%), "
        f"EKS {co_e['baseline_s']:.1f} → {co_e['agentic_s']:.1f} s (+{co_e['overhead_pct']:.0f}%); per-epoch Claude time median local {co_l['monitor_llm_s']:.1f} s, EKS {co_e['monitor_llm_s']:.1f} s",
        f"- Total Claude usage: local {a['tokens_total']['local']['calls']} calls / {a['tokens_total']['local']['input']:,} in / {a['tokens_total']['local']['output']:,} out tokens; "
        f"EKS {a['tokens_total']['eks']['calls']} calls / {a['tokens_total']['eks']['input']:,} in / {a['tokens_total']['eks']['output']:,} out tokens",
        "",
        "## EKS remediation sequences per run",
        "",
    ]
    for r in a["scenarios"]:
        L.append(f"- `{r['id']}`: " + "; ".join(" → ".join(x) or "(none)" for x in r["eks_actions"]))
    return "\n".join(L) + "\n"
