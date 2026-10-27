"""In-cluster entrypoint for running the AutoPARK benchmark suite as a Kubernetes Job.

Runs run_benchmark.py unchanged, samples container resource usage from cgroup v2 while it
runs, records host/node facts, then uploads a results bundle and a completion marker to S3:

  s3://<bucket>/<prefix>/bundle.tar.gz   results JSON + report + per-run traces + metrics + log
  s3://<bucket>/<prefix>/_SUCCESS|_FAILED  small JSON marker the listener waits for

With --worker (multi-node runs, ranks >= 1) it instead runs `python -m pipeline.distributed`
under a restart loop (a worker process that dies is restarted and rejoins the next training
attempt), samples the same cgroup counters, and uploads them with the worker log:

  s3://<bucket>/<prefix>/ranks/rank<N>.tar.gz   cluster_metrics.json + worker.log + restarts

Usage (inside the pod):
    python benchmarks/eks_entrypoint.py --bucket B --prefix P -- --suite paper --repeats 5 ...
    python benchmarks/eks_entrypoint.py --bucket B --prefix P --worker
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")


def read_cgroup() -> dict | None:
    """One cgroup v2 sample of cumulative CPU time and current memory for this container."""
    try:
        cpu = dict(line.split() for line in (CGROUP / "cpu.stat").read_text().splitlines())
        sample = {
            "t": time.time(),
            "cpu_usage_usec": int(cpu["usage_usec"]),
            "cpu_throttled_usec": int(cpu.get("throttled_usec", 0)),
            "memory_bytes": int((CGROUP / "memory.current").read_text()),
        }
        return sample
    except (OSError, KeyError, ValueError):
        return None


class ResourceSampler(threading.Thread):
    """Samples cgroup counters every `interval` seconds until stopped."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            s = read_cgroup()
            if s is not None:
                self.samples.append(s)
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)


def host_facts() -> dict:
    def read(path: str) -> str | None:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    cpu_model = None
    cpuinfo = read("/proc/cpuinfo") or ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    import torch

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu_model": cpu_model,
        "os_cpu_count": os.cpu_count(),
        "sched_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": read("/sys/fs/cgroup/memory.max"),
        "memory_peak_bytes": read("/sys/fs/cgroup/memory.peak"),
        "node_name": os.environ.get("NODE_NAME"),
        "pod_name": os.environ.get("POD_NAME"),
        "pod_namespace": os.environ.get("POD_NAMESPACE"),
        "code_sha": os.environ.get("CODE_SHA"),
        "bootstrap_started_at": float(os.environ["BOOTSTRAP_STARTED_AT"]) if os.environ.get("BOOTSTRAP_STARTED_AT") else None,
    }


def upload(bucket: str, key: str, path: Path | None = None, body: bytes | None = None) -> None:
    import boto3

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    if path is not None:
        s3.upload_file(str(path), bucket, key)
    else:
        s3.put_object(Bucket=bucket, Key=key, Body=body or b"")


def worker(args: argparse.Namespace) -> int:
    """Run one DDP worker rank under a restart loop; upload its resource metrics and log."""
    rank = int(os.environ["AUTOPARK_RANK"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sampler = ResourceSampler()
    sampler.start()
    t0 = time.time()
    exits: list[dict] = []
    with open(out / "worker.log", "w", encoding="utf-8") as log:
        while True:
            started = time.time()
            proc = subprocess.Popen([sys.executable, "-m", "pipeline.distributed"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            connected = None
            for line in proc.stdout:
                if connected is None and line.startswith(f"[rank {rank}] connected"):
                    connected = time.time()  # store client up, heartbeat thread started
                sys.stdout.write(line)
                log.write(line)
            rc = proc.wait()
            exits.append({"started": started, "connected": connected, "exited": time.time(), "exit_code": rc})
            if rc == 0:
                break
            msg = f"[eks_entrypoint] rank {rank} worker exited with {rc}; restarting\n"
            sys.stdout.write(msg)
            log.write(msg)
    t1 = time.time()
    sampler.stop()
    metrics = {
        "rank": rank,
        "host": host_facts(),
        "worker_started_at": t0,
        "worker_finished_at": t1,
        "process_exits": exits,
        "sample_interval_s": sampler.interval,
        "resource_samples": sampler.samples,
    }
    (out / "cluster_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    bundle = Path(f"rank{rank}.tar.gz")
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(out, arcname=f"rank{rank}")
    upload(args.bucket, f"{args.prefix.rstrip('/')}/ranks/rank{rank}.tar.gz", path=bundle)
    print(f"[eks_entrypoint] rank {rank} uploaded metrics ({len(exits) - 1} restarts)", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--out", default="eks_out")
    p.add_argument("--worker", action="store_true", help="run a DDP worker rank (multi-node runs)")
    p.add_argument("bench_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.worker:
        return worker(args)
    bench_args = [a for a in args.bench_args if a != "--"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entry_started = time.time()

    sampler = ResourceSampler()
    sampler.start()
    cmd = [sys.executable, "run_benchmark.py", *bench_args,
           "--json-output", str(out / "results.json"), "--output", str(out / "report.md")]
    print(f"[eks_entrypoint] running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(out / "benchmark.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    t1 = time.time()
    sampler.stop()

    # Archive the per-run trace files referenced by the results JSON
    n_traces = 0
    if (out / "results.json").exists():
        results = json.loads((out / "results.json").read_text())
        for run in results.get("runs", []):
            for mode in ("baseline", "agentic"):
                src = Path(run[mode]["run_dir"])
                dst = out / "traces" / src.name
                dst.mkdir(parents=True, exist_ok=True)
                for name in ("timing.json", "step_telemetry.json", "action_log.json", "incidents.json", "config.json"):
                    if (src / name).exists():
                        shutil.copy2(src / name, dst / name)
                        n_traces += 1

    metrics = {
        "host": host_facts(),
        "entrypoint_started_at": entry_started,
        "benchmark_started_at": t0,
        "benchmark_finished_at": t1,
        "benchmark_exit_code": rc,
        "sample_interval_s": sampler.interval,
        "resource_samples": sampler.samples,
    }
    (out / "cluster_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    bundle = Path("bundle.tar.gz")
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(out, arcname="bundle")
    prefix = args.prefix.rstrip("/")
    upload(args.bucket, f"{prefix}/bundle.tar.gz", path=bundle)
    marker = {"exit_code": rc, "finished_at": t1, "duration_s": t1 - t0, "trace_files": n_traces,
              "pod": os.environ.get("POD_NAME"), "node": os.environ.get("NODE_NAME")}
    upload(args.bucket, f"{prefix}/{'_SUCCESS' if rc == 0 else '_FAILED'}", body=json.dumps(marker).encode())
    print(f"[eks_entrypoint] uploaded s3://{args.bucket}/{prefix}/bundle.tar.gz (exit {rc})", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
