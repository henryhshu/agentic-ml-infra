"""Listener for an AutoPARK benchmark Job running on EKS.

Polls the Job until it finishes (or its S3 completion marker appears), snapshotting
Kubernetes analytics while they still exist (events expire after ~1h), then downloads
the results bundle from S3, consolidates everything into one directory, and runs the
local-vs-EKS analysis:

    benchmarks/reports/eks/<run_tag>/
        bundle/                 results.json, report.md, traces/, cluster_metrics.json, benchmark.log
        k8s_analytics.json      node facts, pod lifecycle, Job status, Kubernetes events
        pod.log                 full container log (bootstrap + benchmark)
        analysis.json / .md     consolidated analysis (see benchmarks/eks_analysis.py)

Multi-node runs (Indexed Job, one pod per rank) additionally get, per rank:
        ranks/rank<N>/          worker cluster_metrics.json + worker.log (from S3)
        pod-rank<N>.log         container log
and the analysis is benchmarks/dist_analysis.py.

Usage:
    python -m benchmarks.eks_listener --run-tag eks-20261027T120000Z --job autopark-bench-...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

NAMESPACE = "autopark"


def kubectl_json(*args: str) -> dict | None:
    r = subprocess.run(["kubectl", "-n", NAMESPACE, *args, "-o", "json"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def s3_read(uri: str) -> str | None:
    """Read an S3 object via the AWS CLI (which refreshes `aws login` sessions itself)."""
    r = subprocess.run(["aws", "s3", "cp", "--only-show-errors", uri, "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def marker(bucket: str, prefix: str) -> tuple[str, dict] | None:
    for name in ("_SUCCESS", "_FAILED"):
        body = s3_read(f"s3://{bucket}/{prefix}/{name}")
        if body:
            return name, json.loads(body)
    return None


def job_finished(job: dict | None) -> str | None:
    if not job:
        return None
    for c in job.get("status", {}).get("conditions", []) or []:
        if c.get("type") in ("Complete", "Failed") and c.get("status") == "True":
            return c["type"]
    return None


def pod_index(pod: dict) -> int:
    ann = pod["metadata"].get("annotations", {})
    return int(ann.get("batch.kubernetes.io/job-completion-index", 0))


def snapshot(job_name: str, events: dict[str, dict]) -> tuple[dict | None, list[dict]]:
    job = kubectl_json("get", "job", job_name)
    pods = kubectl_json("get", "pods", "-l", f"job-name={job_name}")
    ev = kubectl_json("get", "events")
    pod_names = {p["metadata"]["name"] for p in (pods or {}).get("items", [])}
    for e in (ev or {}).get("items", []):
        obj = e.get("involvedObject", {})
        if obj.get("name") in pod_names | {job_name}:
            events[e["metadata"]["uid"]] = {
                "object": f"{obj.get('kind')}/{obj.get('name')}",
                "reason": e.get("reason"),
                "message": e.get("message"),
                "type": e.get("type"),
                "first": e.get("firstTimestamp") or e.get("eventTime"),
                "last": e.get("lastTimestamp") or e.get("eventTime"),
                "count": e.get("count"),
            }
    return job, sorted((pods or {}).get("items", []), key=pod_index)


def node_facts(node_name: str | None) -> dict:
    if not node_name:
        return {}
    r = subprocess.run(["kubectl", "get", "node", node_name, "-o", "json"], capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    n = json.loads(r.stdout)
    labels = n["metadata"].get("labels", {})
    info = n["status"].get("nodeInfo", {})
    return {
        "name": node_name,
        "instance_type": labels.get("node.kubernetes.io/instance-type"),
        "zone": labels.get("topology.kubernetes.io/zone"),
        "capacity_type": labels.get("eks.amazonaws.com/capacityType"),
        "nodegroup": labels.get("eks.amazonaws.com/nodegroup"),
        "kubelet_version": info.get("kubeletVersion"),
        "os_image": info.get("osImage"),
        "kernel": info.get("kernelVersion"),
        "container_runtime": info.get("containerRuntimeVersion"),
        "architecture": info.get("architecture"),
        "capacity": n["status"].get("capacity", {}),
        "allocatable": n["status"].get("allocatable", {}),
    }


def pod_lifecycle(pod: dict | None) -> dict:
    if not pod:
        return {}
    st = pod.get("status", {})
    conds = {c["type"]: c.get("lastTransitionTime") for c in st.get("conditions", []) or []}
    cs = (st.get("containerStatuses") or [{}])[0]
    state = cs.get("state", {})
    term = state.get("terminated", {})
    running = state.get("running", {})
    return {
        "name": pod["metadata"]["name"],
        "node": pod["spec"].get("nodeName"),
        "created": pod["metadata"].get("creationTimestamp"),
        "scheduled": conds.get("PodScheduled"),
        "initialized": conds.get("Initialized"),
        "ready": conds.get("Ready"),
        "container_started": term.get("startedAt") or running.get("startedAt"),
        "container_finished": term.get("finishedAt"),
        "exit_code": term.get("exitCode"),
        "termination_reason": term.get("reason"),
        "restart_count": cs.get("restartCount"),
        "image": cs.get("image"),
        "image_id": cs.get("imageID"),
        "qos_class": st.get("qosClass"),
        "resources": pod["spec"]["containers"][0].get("resources", {}),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--bucket", default="park-ml-experiments-515966515749-us-east-1")
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--timeout", type=float, default=5 * 3600)
    p.add_argument("--local-results",
                   default="benchmarks/reports/benchmark_results_2026-10-27T10-12-25.325632_00-00.json")
    args = p.parse_args()

    prefix = f"autopark-eval/runs/{args.run_tag}"
    out = Path("benchmarks/reports/eks") / args.run_tag
    out.mkdir(parents=True, exist_ok=True)
    events: dict[str, dict] = {}
    started = time.time()
    job = None
    pods: list[dict] = []
    done = None

    print(f"[listener] watching job {args.job}; results at s3://{args.bucket}/{prefix}/", flush=True)
    while time.time() - started < args.timeout:
        job, pod_snap = snapshot(args.job, events)
        pods = pod_snap or pods
        m = marker(args.bucket, prefix)
        state = job_finished(job)
        phases = ",".join(p.get("status", {}).get("phase", "?") for p in pods) or None
        print(f"[listener] {time.strftime('%H:%M:%S')} pods={phases} job={state or 'running'} marker={m[0] if m else '-'}", flush=True)
        if m or state == "Failed":
            done = m
            break
        time.sleep(args.poll)
    else:
        print("[listener] timed out", file=sys.stderr)

    # Multi-node: rank 0 writes the marker first; wait for the worker ranks to finish uploading
    if done and len(pods) > 1:
        t_wait = time.time()
        while job_finished(job) is None and time.time() - t_wait < 600:
            time.sleep(10)
            job, pod_snap = snapshot(args.job, events)
            pods = pod_snap or pods

    # Final snapshot and logs while the pods still exist
    job, pod_snap = snapshot(args.job, events)
    pods = pod_snap or pods
    for p in pods:
        r = subprocess.run(["kubectl", "-n", NAMESPACE, "logs", p["metadata"]["name"]], capture_output=True, text=True)
        name = "pod.log" if len(pods) == 1 else f"pod-rank{pod_index(p)}.log"
        (out / name).write_text(r.stdout, encoding="utf-8")
    pod = pods[0] if pods else None

    analytics = {
        "run_tag": args.run_tag,
        "job": {"name": args.job, "status": (job or {}).get("status")},
        "pod": pod_lifecycle(pod),
        "node": node_facts((pod or {}).get("spec", {}).get("nodeName")),
        "ranks": [{"rank": pod_index(p), "pod": pod_lifecycle(p), "node": node_facts(p["spec"].get("nodeName"))}
                  for p in pods],
        "events": sorted(events.values(), key=lambda e: e.get("first") or ""),
        "marker": done[1] if done else None,
        "marker_type": done[0] if done else None,
        "listener_wait_s": time.time() - started,
    }
    (out / "k8s_analytics.json").write_text(json.dumps(analytics, indent=2), encoding="utf-8")

    if not done:
        print("[listener] no completion marker; results bundle not available", file=sys.stderr)
        return 1

    bundle = out / "bundle.tar.gz"
    subprocess.run(["aws", "s3", "cp", "--only-show-errors", f"s3://{args.bucket}/{prefix}/bundle.tar.gz", str(bundle)],
                   check=True)
    with tarfile.open(bundle) as tar:
        tar.extractall(out, filter="data")
    bundle.unlink()
    print(f"[listener] results extracted to {out}/bundle", flush=True)

    if len(pods) > 1:
        for rank in range(1, len(pods)):
            tgz = out / f"rank{rank}.tar.gz"
            r = subprocess.run(["aws", "s3", "cp", "--only-show-errors",
                                f"s3://{args.bucket}/{prefix}/ranks/rank{rank}.tar.gz", str(tgz)])
            if r.returncode == 0:
                (out / "ranks").mkdir(exist_ok=True)
                with tarfile.open(tgz) as tar:
                    tar.extractall(out / "ranks", filter="data")
                tgz.unlink()
            else:
                print(f"[listener] rank {rank} metrics missing", file=sys.stderr)
        from benchmarks.dist_analysis import analyze as analyze_dist

        md = analyze_dist(out)
        print(f"[listener] analysis written to {md}", flush=True)
        return 0 if done[0] == "_SUCCESS" else 1

    from benchmarks.eks_analysis import analyze

    md = analyze(out, Path(args.local_results))
    print(f"[listener] analysis written to {md}", flush=True)
    return 0 if done[0] == "_SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
