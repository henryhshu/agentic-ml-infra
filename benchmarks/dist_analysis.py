"""Consolidated analysis of a multi-node (DDP) EKS benchmark run.

Reads what benchmarks/eks_listener.py fetched for a WORLD_SIZE > 1 run:

    bundle/            rank 0: results.json, traces/, cluster_metrics.json, benchmark.log
    ranks/rank<N>/     worker ranks: cluster_metrics.json (cgroup samples, process exits), worker.log
    k8s_analytics.json Job, per-rank pod lifecycle and node facts, Kubernetes events

and writes analysis.json / analysis.md covering
  - cluster topology (nodes, zones, instance types) and per-rank pod lifecycle
  - per-rank cgroup CPU and memory, split into training attempts vs. agent incident handling
  - per-scenario outcomes and MTTR decomposition
  - worker-loss runs: process kill -> failure detected on rank 0 -> worker restarted -> retry
  - Claude call latency, DDP step time, fault-free monitoring overhead, token totals
Every number is computed from recorded data; nothing is estimated.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from benchmarks.eks_analysis import cpu_series, dist, iso, llm_latency, load_traces, step_time

LEVELS_1_3 = ["nan_divergence", "batch_oom", "compound_oom_and_explosion", "cascaded_lr_and_memory",
              "delayed_stochastic_explosion", "spot_worker_churn"]


def _windows(traces: dict[str, dict]) -> tuple[list, list]:
    attempt, handle = [], []
    for t in traces.values():
        if t["mode"] != "agentic":
            continue
        for e in t["timeline"]:
            if e["type"] in ("attempt", "handle"):
                (attempt if e["type"] == "attempt" else handle).append((e["start"], e["end"]))
    return attempt, handle


def rank_resources(cm: dict, attempt: list, handle: list) -> dict[str, Any]:
    samples = cm["resource_samples"]
    series = cpu_series(samples)

    def within(ivals):
        return [c for (tm, c, _) in series if any(s <= tm <= e for s, e in ivals)]

    peak = cm["host"].get("memory_peak_bytes") or ""
    return {
        "samples": len(samples),
        "cores_used": dist([c for (_, c, _) in series]),
        "cores_during_agentic_attempts": dist(within(attempt)),
        "cores_during_incident_handling": dist(within(handle)),
        "memory_mib": dist([m / 2**20 for (_, _, m) in series]),
        "memory_peak_mib": int(peak) / 2**20 if peak.isdigit() else None,
        "cpu_throttled_s": (samples[-1]["cpu_throttled_usec"] - samples[0]["cpu_throttled_usec"]) / 1e6 if samples else None,
        "torch_threads": cm["host"].get("torch_threads"),
        "node": cm["host"].get("node_name"),
    }


def worker_loss(results: dict, traces_root: Path, rank_metrics: dict[int, dict]) -> list[dict[str, Any]]:
    """Per spot_worker_churn run: each failed attempt matched to the worker process kill behind it."""
    kills = sorted(
        (e["exited"], r, e) for r, cm in rank_metrics.items() for e in cm.get("process_exits", []) if e["exit_code"] != 0
    )
    # kill -> replacement process connected to the coordination store
    restarts = {
        (r, e["exited"]): nxt.get("connected") or nxt["started"]
        for r, cm in rank_metrics.items()
        for e, nxt in zip(cm.get("process_exits", []), cm.get("process_exits", [])[1:])
    }
    rows = []
    for run in results["runs"]:
        if run["id"] != "spot_worker_churn":
            continue
        for mode in ("baseline", "agentic"):
            d = traces_root / Path(run[mode]["run_dir"]).name
            if not (d / "timing.json").exists():
                continue
            timing = json.loads((d / "timing.json").read_text())
            actions = json.loads((d / "action_log.json").read_text()) if (d / "action_log.json").exists() else []
            attempts = [e for e in timing["timeline"] if e["type"] == "attempt" and e["stage"] == "training"]
            lost_events = [e for e in timing["timeline"] if e["type"] == "worker_lost"]
            failures = []
            for a in attempts:
                if a["status"] != "failed":
                    continue
                k = next(((t, r) for (t, r, _) in kills if a["start"] <= t <= a["end"]), None)
                wl = next((e for e in lost_events if a["start"] <= e["start"] <= a["end"]), None)
                failures.append({
                    "attempt": a["attempt"],
                    "attempt_s": a["duration_s"],
                    "killed_rank": k[1] if k else None,
                    "kill_to_collective_error_s": (wl["start"] - k[0]) if k and wl else None,
                    "liveness_check_s": wl["duration_s"] if wl else None,
                    "kill_to_detect_s": (a["end"] - k[0]) if k else None,
                    "worker_restart_s": (restarts.get((k[1], k[0])) - k[0]) if k and restarts.get((k[1], k[0])) else None,
                    "error": (a.get("error") or "")[:300],
                })
            waits = [float(m.group(1)) for x in actions if x.get("action") == "restart_workers"
                     for m in [re.search(r"Waited ([0-9.]+)s", x.get("details") or "")] if m]
            rows.append({
                "repeat": run["repeat"], "mode": mode, "status": run[mode]["status"],
                "outcome": run["outcome"] if mode == "agentic" else None,
                "attempts": len(attempts), "failures": failures,
                "actions": [x.get("action") for x in actions],
                "restart_workers_wait_s": waits,
                "mttr_s": run[mode]["timing"].get("mttr_s") if mode == "agentic" else None,
            })
    return rows


def analyze(run_dir: Path) -> Path:
    bundle = run_dir / "bundle"
    res = json.loads((bundle / "results.json").read_text())
    k8s = json.loads((run_dir / "k8s_analytics.json").read_text())
    cm0 = json.loads((bundle / "cluster_metrics.json").read_text())
    rank_metrics = {0: cm0}
    for d in sorted((run_dir / "ranks").glob("rank*")):
        rank_metrics[int(d.name[4:])] = json.loads((d / "cluster_metrics.json").read_text())

    traces = load_traces(res, bundle / "traces")
    attempt, handle = _windows(traces)

    ranks = []
    for r in k8s.get("ranks", []):
        pod, node = r["pod"], r["node"]
        cm = rank_metrics.get(r["rank"], {})
        started = cm.get("entrypoint_started_at") or cm.get("worker_started_at")
        boot = cm.get("host", {}).get("bootstrap_started_at")
        t_created, t_sched, t_cstart = iso(pod.get("created")), iso(pod.get("scheduled")), iso(pod.get("container_started"))
        ranks.append({
            "rank": r["rank"],
            "node": node.get("name"), "instance_type": node.get("instance_type"), "zone": node.get("zone"),
            "capacity_type": node.get("capacity_type"),
            "pending_to_scheduled_s": (t_sched - t_created) if t_created and t_sched else None,
            "scheduled_to_container_start_s": (t_cstart - t_sched) if t_sched and t_cstart else None,
            "bootstrap_s": (started - boot) if started and boot else None,
            "process_restarts": max(0, len(cm.get("process_exits", [])) - 1) if r["rank"] else None,
            "exit_code": pod.get("exit_code"),
            "resources": rank_resources(cm, attempt, handle) if cm else None,
        })

    def clean_overhead():
        row = next((r for r in res["aggregates"] if r["id"] == "clean_run"), None)
        if row is None:
            return None
        b, a = row["baseline_duration_s"]["median"], row["agentic_duration_s"]["median"]
        return {"baseline_s": b, "agentic_s": a, "overhead_pct": 100 * (a - b) / b,
                "monitor_llm_s": statistics.median(r["agentic"]["timing"]["monitor_llm_s"]
                                                   for r in res["runs"] if r["id"] == "clean_run")}

    l13 = [r for r in res["runs"] if r["id"] in LEVELS_1_3]
    analysis = {
        "run_tag": k8s.get("run_tag"),
        "code_sha": cm0["host"].get("code_sha"),
        "environment": {**res["environment"], **cm0["host"]},
        "world_size": res["environment"].get("world_size"),
        "benchmark_s": cm0["benchmark_finished_at"] - cm0["benchmark_started_at"],
        "ranks": ranks,
        "aggregates": res["aggregates"],
        "levels_1_3_healed": [sum(r["outcome"] == "healed" for r in l13), len(l13)],
        "matches_expected": [sum(r["outcome_matches_expected"] for r in res["runs"]), len(res["runs"])],
        "worker_loss": worker_loss(res, bundle / "traces", rank_metrics),
        "llm_latency": llm_latency(traces),
        "step_time_s": step_time(traces),
        "clean_overhead": clean_overhead(),
        "tokens_total": {"input": sum(r["agentic"]["timing"].get("input_tokens", 0) for r in res["runs"]),
                         "output": sum(r["agentic"]["timing"].get("output_tokens", 0) for r in res["runs"]),
                         "calls": sum(r["agentic"]["timing"].get("llm_calls", 0) for r in res["runs"])},
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    md = run_dir / "analysis.md"
    md.write_text(render_markdown(analysis), encoding="utf-8")
    return md


def _f(v, fmt="{:.1f}"):
    return "–" if v is None else fmt.format(v)


def _m(d, key="median", fmt="{:.1f}"):
    return "–" if not d else fmt.format(d[key])


def render_markdown(a: dict) -> str:
    env = a["environment"]
    L = [f"# Multi-node EKS analysis — `{a['run_tag']}`", "",
         f"Code `{a['code_sha']}`; world size {a['world_size']} (PyTorch DDP, gloo over TCP, one rank per node); "
         f"torch {env.get('torch')}; workload {env.get('model')}, {env.get('epochs')} epochs, subset {env.get('subset')}, "
         f"{env.get('repeats')} repeats. Benchmark wall time {a['benchmark_s'] / 60:.1f} min. "
         "All numbers are computed from recorded data by `benchmarks/dist_analysis.py`.", "",
         "## Ranks", "",
         "| Rank | Node | Type | Zone | Scheduled→start s | Bootstrap s | Proc. restarts | Cores (attempts) | Cores (incident handling) | Mem median MiB | Mem peak MiB | Throttled s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in a["ranks"]:
        res = r["resources"] or {}
        L.append(f"| {r['rank']} | {r['node']} | {r['instance_type']} ({r['capacity_type']}) | {r['zone']} | "
                 f"{_f(r['scheduled_to_container_start_s'])} | {_f(r['bootstrap_s'])} | {_f(r['process_restarts'], '{}')} | "
                 f"{_m(res.get('cores_during_agentic_attempts'), fmt='{:.2f}')} | {_m(res.get('cores_during_incident_handling'), fmt='{:.2f}')} | "
                 f"{_m(res.get('memory_mib'), fmt='{:.0f}')} | {_f(res.get('memory_peak_mib'), '{:.0f}')} | {_f(res.get('cpu_throttled_s'))} |")
    L += ["", "## Outcomes and MTTR (median s [min–max])", "",
          "| Scenario | Expected | Outcomes | MTTR | T_det | T_reason | T_act | T_recover | Baseline wall | Agentic wall |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in a["aggregates"]:
        oc = ", ".join(f"{k} {v}" for k, v in r["outcomes"].items())
        mt = r.get("mttr_s")
        L.append(f"| `{r['id']}` | {r['expected_outcome']} | {oc} | "
                 f"{_m(mt)} [{_m(mt, 'min')}–{_m(mt, 'max')}] | {_m(r.get('t_det'))} | {_m(r.get('t_reason'))} | "
                 f"{_m(r.get('t_act'))} | {_m(r.get('t_recover'))} | {_m(r.get('baseline_duration_s'))} | {_m(r.get('agentic_duration_s'))} |")
    h, n = a["levels_1_3_healed"]
    m, nm = a["matches_expected"]
    L += ["", f"Levels 1–3 healed: {h}/{n}. Runs matching the expected outcome: {m}/{nm}.", "",
          "## Worker loss (`spot_worker_churn`)", "",
          "| Repeat | Mode | Status | Attempts | Killed rank | Kill→collective error s | Liveness check s | Kill→attempt failed s | Kill→replacement connected s | restart_workers wait s | Actions |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in a["worker_loss"]:
        f0 = w["failures"][0] if w["failures"] else {}
        L.append(f"| {w['repeat']} | {w['mode']} | {w['status']} | {w['attempts']} | {f0.get('killed_rank', '–')} | "
                 + ", ".join(_f(f["kill_to_collective_error_s"], "{:.3f}") for f in w["failures"]) + " | "
                 + ", ".join(_f(f["liveness_check_s"], "{:.2f}") for f in w["failures"]) + " | "
                 + ", ".join(_f(f["kill_to_detect_s"], "{:.2f}") for f in w["failures"]) + " | "
                 + ", ".join(_f(f["worker_restart_s"], "{:.2f}") for f in w["failures"]) + " | "
                 + ", ".join(f"{x:.1f}" for x in w["restart_workers_wait_s"]) + f" | {' → '.join(w['actions']) or '–'} |")
    errs = [f["error"] for w in a["worker_loss"] for f in w["failures"] if f["error"]]
    if errs:
        L += ["", f"Example error seen by the agents: `{errs[0]}`"]
    lat = a["llm_latency"]
    L += ["", "## Claude call latency", "", "| Caller | n | median s | p90 s | max s |", "|---|---|---|---|---|",
          f"| all | {lat['all']['n']} | {lat['all']['median']:.2f} | {lat['all']['p90']:.2f} | {lat['all']['max']:.2f} |"]
    for k, v in lat["by_caller"].items():
        L.append(f"| `{k}` | {v['n']} | {v['median']:.2f} | {v['p90']:.2f} | {v['max']:.2f} |")
    co, st, tk = a["clean_overhead"], a["step_time_s"], a["tokens_total"]
    L += ["", "## Training and monitoring overhead", "",
          f"- DDP step time (clean baseline, global batch over {a['world_size']} ranks): median {_m(st, fmt='{:.3f}')} s, p90 {_m(st, 'p90', '{:.3f}')} s"]
    if co:
        L.append(f"- Fault-free run: baseline {co['baseline_s']:.1f} s, agentic {co['agentic_s']:.1f} s (+{co['overhead_pct']:.0f}%); "
                 f"Claude time inside attempts median {co['monitor_llm_s']:.1f} s")
    L += [f"- Claude usage: {tk['calls']} calls, {tk['input']:,} input / {tk['output']:,} output tokens"]
    return "\n".join(L) + "\n"
