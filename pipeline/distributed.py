"""Multi-node data-parallel execution of the training stage (PyTorch DDP over gloo).

Rank 0 is the coordinator: it runs the whole pipeline, the benchmark harness and the
agent fleet, and also trains as one DDP rank. Ranks 1..W-1 run `worker_main()`, which
waits for training attempts published by rank 0 and joins each one as a DDP rank.

Coordination uses one persistent `TCPStore` hosted by rank 0:

    spec/<n>      JSON config of training attempt n (published by rank 0)
    seq           id of the latest published attempt (-1 = shut down)
    done/<n>      set by rank 0 when attempt n has ended
    pg/<n>/...    process-group rendezvous of attempt n (fresh per attempt)
    att/<n>/<r>   process id of rank r when it joined attempt n
    hb/<r>        "<process id>:<counter>" heartbeat of worker rank r (every 0.5 s)

Each attempt forms a new gloo process group, so a worker process that died and was
restarted (the container restarts it) can join the next attempt. When a collective fails
on rank 0, the heartbeats show which worker processes were lost; that measured liveness
is what the agents see in the error.

Environment: AUTOPARK_WORLD_SIZE, AUTOPARK_RANK, AUTOPARK_MASTER_ADDR, AUTOPARK_STORE_PORT.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
import uuid
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

STORE_TIMEOUT = timedelta(seconds=900)
PG_TIMEOUT = timedelta(seconds=600)
HEARTBEAT_S = 0.5
LIVENESS_WINDOW_S = 2.5

_CTX: DistContext | None = None


def get_context() -> DistContext | None:
    return _CTX


def init_from_env() -> DistContext | None:
    """Create the process-wide context when AUTOPARK_WORLD_SIZE > 1."""
    global _CTX
    world_size = int(os.environ.get("AUTOPARK_WORLD_SIZE", "1"))
    if world_size <= 1:
        return None
    _CTX = DistContext(
        rank=int(os.environ.get("AUTOPARK_RANK", "0")),
        world_size=world_size,
        master_addr=os.environ.get("AUTOPARK_MASTER_ADDR", "127.0.0.1"),
        port=int(os.environ.get("AUTOPARK_STORE_PORT", "29400")),
    )
    return _CTX


def _connect(master_addr: str, port: int, is_master: bool, deadline_s: float = 3600) -> dist.TCPStore:
    t0 = time.time()
    while True:
        try:
            return dist.TCPStore(master_addr, port, is_master=is_master,
                                 timeout=STORE_TIMEOUT, wait_for_workers=False)
        except Exception:
            if is_master or time.time() - t0 > deadline_s:
                raise
            time.sleep(2.0)


class WorkerLostError(RuntimeError):
    """A DDP collective failed on rank 0 and at least one worker process is gone."""


class DistContext:
    def __init__(self, rank: int, world_size: int, master_addr: str, port: int):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.port = port
        self.proc_id = f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.store = _connect(master_addr, port, is_master=(rank == 0))
        self.seq = 0  # attempt this process is in (rank 0: last published)
        self.pending_seq: int | None = None  # set by the worker loop before execute()
        self._hb_thread: threading.Thread | None = None

    # ── attempt lifecycle (all ranks) ──

    def open_attempt(self, spec: dict[str, Any] | None = None) -> int:
        """Rank 0 publishes `spec`; every rank then joins the attempt's process group."""
        if self.rank == 0:
            self.seq += 1
            self.store.set(f"spec/{self.seq}", json.dumps(spec, default=str))
            self.store.set("seq", str(self.seq))
        else:
            self.seq = self.pending_seq
        dist.init_process_group(
            "gloo",
            store=dist.PrefixStore(f"pg/{self.seq}", self.store),
            rank=self.rank,
            world_size=self.world_size,
            timeout=PG_TIMEOUT,
        )
        self.store.set(f"att/{self.seq}/{self.rank}", self.proc_id)
        return self.seq

    def close_attempt(self) -> None:
        if self.rank == 0:
            self.store.set(f"done/{self.seq}", "1")
        if dist.is_initialized():
            dist.destroy_process_group()

    # ── collectives used by the training loop ──

    def all_reduce_sum(self, values: list[float]) -> list[float]:
        t = torch.tensor(values, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.tolist()

    def broadcast(self, obj: Any) -> Any:
        box = [obj]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    # ── liveness (rank 0) ──

    def _heartbeats(self) -> dict[int, tuple[str, int] | None]:
        out: dict[int, tuple[str, int] | None] = {}
        for r in range(1, self.world_size):
            key = f"hb/{r}"
            if self.store.check([key]):
                pid, counter = self.store.get(key).decode().rsplit(":", 1)
                out[r] = (pid, int(counter))
            else:
                out[r] = None
        return out

    def lost_workers(self) -> list[dict[str, Any]]:
        """Workers whose attempt process exited or stopped heartbeating (measured over <= 2.5 s)."""
        joined = {}
        for r in range(1, self.world_size):
            key = f"att/{self.seq}/{r}"
            joined[r] = self.store.get(key).decode() if self.store.check([key]) else None
        first = self._heartbeats()
        t0 = time.time()
        pending = set(range(1, self.world_size))
        lost: dict[int, dict[str, Any]] = {}
        while pending and time.time() - t0 < LIVENESS_WINDOW_S:
            now = self._heartbeats()
            for r in list(pending):
                hb = now[r]
                if hb is None or (joined[r] and hb[0] != joined[r]):
                    lost[r] = {"rank": r, "reason": "process exited (restarted with a new process id)"
                               if hb else "never heartbeated", "observed_after_s": round(time.time() - t0, 2)}
                    pending.discard(r)
                elif first[r] and hb[1] > first[r][1]:
                    pending.discard(r)  # alive: counter advanced
            time.sleep(0.25)
        for r in pending:
            lost[r] = {"rank": r, "reason": f"no heartbeat for {LIVENESS_WINDOW_S:.1f} s",
                       "observed_after_s": LIVENESS_WINDOW_S}
        return [lost[r] for r in sorted(lost)]

    def worker_lost_error(self, exc: BaseException) -> WorkerLostError | None:
        lost = self.lost_workers()
        if not lost:
            return None
        first_line = str(exc).strip().splitlines()[0][:240] if str(exc).strip() else type(exc).__name__
        detail = "; ".join(f"rank {w['rank']}: {w['reason']}" for w in lost)
        alive = self.world_size - 1 - len(lost)
        return WorkerLostError(
            f"Distributed data-parallel training failed on rank 0 ({type(exc).__name__}: {first_line}). "
            f"Worker liveness after the failure: {alive}/{self.world_size - 1} worker processes heartbeating; "
            f"lost {detail}."
        )

    def await_workers(self, timeout_s: float = 300.0) -> dict[str, Any]:
        """Block until every worker rank heartbeats from a live process; return measured readiness."""
        t0 = time.time()
        first = self._heartbeats()
        while time.time() - t0 < timeout_s:
            time.sleep(0.25)
            now = self._heartbeats()
            ready = [r for r in now if now[r] and first[r] and now[r][0] == first[r][0] and now[r][1] > first[r][1]]
            if len(ready) == self.world_size - 1:
                return {"ready": True, "wait_s": time.time() - t0, "workers": {r: now[r][0] for r in now}}
            first = {r: (first[r] if first[r] and now[r] and first[r][0] == now[r][0] else now[r]) for r in now}
        return {"ready": False, "wait_s": time.time() - t0}

    def live_worker_count(self) -> int:
        return self.world_size - 1 - len(self.lost_workers())

    def shutdown(self) -> None:
        if self.rank == 0:
            self.store.set("seq", "-1")

    # ── worker side ──

    def start_heartbeat(self) -> None:
        def beat():
            store = _connect(self.master_addr, self.port, is_master=False)
            n = 0
            while True:
                n += 1
                store.set(f"hb/{self.rank}", f"{self.proc_id}:{n}")
                time.sleep(HEARTBEAT_S)

        self._hb_thread = threading.Thread(target=beat, name="autopark-heartbeat", daemon=True)
        self._hb_thread.start()

    def crash(self) -> None:
        """Hard-kill this worker process (SIGKILL), as a node or process loss would."""
        print(f"[rank {self.rank}] injected worker crash: SIGKILL pid {os.getpid()}", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)


def worker_main() -> int:
    """Rank >= 1: join every training attempt published by rank 0 until shutdown."""
    from metrics.experiment import ExperimentTracker
    from pipeline.config import PipelineConfig
    from pipeline.data_loading import DataLoadingStage
    from pipeline.training import TrainingStage

    ctx = init_from_env()
    if ctx is None or ctx.rank == 0:
        raise SystemExit("worker_main requires AUTOPARK_WORLD_SIZE > 1 and AUTOPARK_RANK >= 1")
    ctx.start_heartbeat()
    print(f"[rank {ctx.rank}] connected to {ctx.master_addr}:{ctx.port} as {ctx.proc_id}", flush=True)

    fields = set(PipelineConfig.__dataclass_fields__)
    datasets: dict[tuple, Any] = {}
    last = 0
    while True:
        if not ctx.store.check(["seq"]):
            time.sleep(0.1)
            continue
        seq = int(ctx.store.get("seq"))
        if seq < 0:
            print(f"[rank {ctx.rank}] shutdown", flush=True)
            return 0
        if seq == last or ctx.store.check([f"done/{seq}"]):
            time.sleep(0.1)
            continue
        last = seq
        spec = json.loads(ctx.store.get(f"spec/{seq}"))
        config = PipelineConfig(**{k: v for k, v in spec.items() if k in fields and k not in ("data_module", "model_fn")})
        config.workers_restarted = spec.get("workers_restarted", False)
        config.run_id = f"worker{ctx.rank}_attempt{seq}"

        tracker = ExperimentTracker(run_id=config.run_id, base_dir="experiments/workers")
        key = (config.dataset, config.subset, config.cache_dir)
        if key not in datasets:
            loading = DataLoadingStage(config, tracker, None)
            loading.execute()
            datasets[key] = (loading.train_set, loading.test_set, loading.train_loader, loading.test_loader)
        train_set, _, train_loader, test_loader = datasets[key]

        ctx.pending_seq = seq
        stage = TrainingStage(config, tracker, train_loader, test_loader, len(train_set))
        t0 = time.time()
        try:
            stage.execute()
            print(f"[rank {ctx.rank}] attempt {seq} ok ({time.time() - t0:.1f}s)", flush=True)
        except Exception as exc:
            print(f"[rank {ctx.rank}] attempt {seq} ended: {type(exc).__name__}: {str(exc)[:200]}", flush=True)


if __name__ == "__main__":
    # import the package module so get_context() in other modules sees the same _CTX
    from pipeline import distributed

    raise SystemExit(distributed.worker_main())
