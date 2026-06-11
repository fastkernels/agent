"""Section 3: robust candidate/baseline latency measurement.

Clocks CANNOT be locked on this box (no sudo) so absolute milliseconds are not
reproducible run-to-run (DVFS idles at 345 MHz, boosts to 1980 MHz under load).
The ONLY meaningful perf signal is the RATIO of baseline to candidate latency,
measured in the SAME warmed process with the two kernels timed INTERLEAVED so
both observe the identical clock state.

Recipe (mirrors .research/baseline-integration.md section D.2):
  * warm config.PERF['warmup_iters'] iters (both kernels) to reach steady boost
    clocks before any timing;
  * between every timed sample, flush the L2 by zeroing a >= config.PERF['l2_flush_mb']
    MiB scratch tensor so a tiny decode kernel cannot be served entirely from L2;
  * collect config.PERF['timed_iters'] CUDA-event samples per trial, across
    config.PERF['num_trials'] trials;
  * report median + p90 (ms) per kernel; ratio = base_median / cand_median.

A candidate that raises is reported with ratio=0.0 and a terse note (never a
traceback dump) — context discipline: only bounded, distilled output leaves here.
"""
from __future__ import annotations

import statistics
from typing import Optional

import torch

from .. import config as C
from .. import common
from .. import shapes


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _percentile(samples: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0,100]); robust for small sample counts."""
    if not samples:
        return 0.0
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _l2_flush_tensor(device: torch.device) -> torch.Tensor:
    nbytes = int(C.PERF["l2_flush_mb"]) * 1024 * 1024
    return torch.empty(nbytes, dtype=torch.int8, device=device)


def _time_kernel(
    namespace: str,
    inp: shapes.Inputs,
    flush: torch.Tensor,
    start: torch.cuda.Event,
    end: torch.cuda.Event,
) -> list[float]:
    """Collect timed_iters x num_trials CUDA-event samples (ms) for one kernel.

    L2 is flushed before every timed launch so each sample reads cold-in-L2 data,
    matching how both kernels are treated for a fair ratio.
    """
    timed = int(C.PERF["timed_iters"])
    trials = int(C.PERF["num_trials"])
    samples: list[float] = []
    for _ in range(trials):
        for _ in range(timed):
            flush.zero_()                      # evict q/k/v/out from L2
            torch.cuda.synchronize()
            start.record()
            shapes.call_fwd(namespace, inp)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))  # ms
    return samples


# --------------------------------------------------------------------------------------
# public API (interfaces.py: perf/measure.py)
# --------------------------------------------------------------------------------------
def measure(
    namespace: str,
    cfg: C.Config,
    gpu_id: int,
    baseline_ns: str = "_vllm_fa3_C",
) -> common.ConfigResult:
    """Time baseline (`baseline_ns`) and candidate (`namespace`) INTERLEAVED in
    this process and return a ConfigResult with the perf fields populated.

    ratio = base_median_ms / cand_median_ms  (>= 1.0 means candidate matches or
    exceeds the baseline).  Robust to a candidate that errors: ratio=0.0 + note.
    `correct` is a placeholder True here — the orchestrator gates correctness via
    verify/correctness.py separately.
    """
    res = common.ConfigResult(config=cfg.name, correct=True)

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    try:
        inp = shapes.make_inputs(cfg, seed=0, device=f"cuda:{gpu_id}")
    except Exception as e:  # input construction failure is non-recoverable for this cfg
        res.ratio = 0.0
        res.note = f"make_inputs error: {type(e).__name__}: {e}".replace("\n", " ")[:200]
        return res

    flush = _l2_flush_tensor(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # --- sanity: does the candidate even run once? (distilled terse note on failure) ---
    cand_ok = True
    cand_err = ""
    try:
        shapes.call_fwd(namespace, inp)
    except Exception as e:
        cand_ok = False
        cand_err = f"{type(e).__name__}: {e}".replace("\n", " ")[:200]

    # --- warmup: spin BOTH kernels to reach steady boost clocks before timing ---
    warm = int(C.PERF["warmup_iters"])
    try:
        for _ in range(warm):
            shapes.call_fwd(baseline_ns, inp)
            if cand_ok:
                shapes.call_fwd(namespace, inp)
        torch.cuda.synchronize()
    except Exception as e:
        # baseline must be runnable for any ratio to be meaningful.
        if cand_ok:
            cand_ok = False
            cand_err = f"warmup error: {type(e).__name__}: {e}".replace("\n", " ")[:200]

    # --- timed, interleaved: baseline then candidate so both see same clock state ---
    base_samples = _time_kernel(baseline_ns, inp, flush, start, end)
    base_median = statistics.median(base_samples) if base_samples else 0.0
    base_p90 = _percentile(base_samples, 90.0)
    res.base_latency_ms = base_median

    if not cand_ok:
        res.cand_latency_ms = 0.0
        res.ratio = 0.0
        res.note = f"candidate failed: {cand_err}"
        return res

    try:
        cand_samples = _time_kernel(namespace, inp, flush, start, end)
    except Exception as e:
        res.cand_latency_ms = 0.0
        res.ratio = 0.0
        res.note = f"candidate timing error: {type(e).__name__}: {e}".replace("\n", " ")[:200]
        return res

    cand_median = statistics.median(cand_samples) if cand_samples else 0.0
    cand_p90 = _percentile(cand_samples, 90.0)
    res.cand_latency_ms = cand_median

    if cand_median > 0.0 and base_median > 0.0:
        res.ratio = base_median / cand_median
    else:
        res.ratio = 0.0

    res.note = (
        f"base p50/p90={base_median:.4f}/{base_p90:.4f}ms "
        f"cand p50/p90={cand_median:.4f}/{cand_p90:.4f}ms "
        f"(warmup={warm}, timed={C.PERF['timed_iters']}x{C.PERF['num_trials']})"
    )
    return res


def measure_all(namespace, cfgs, pool) -> list[common.ConfigResult]:
    """Measure perf across `cfgs` using one exclusive GPU lease per config so
    that ONLY ONE kernel runs per GPU at a time (perf/profiling discipline).

    NOTE: this runs the measurements in THREADS (one per GPU) in the caller's process. Because
    the GIL serializes host-side launch dispatch, that asymmetrically inflates the small-kernel
    CUDA-event timing and the baseline/candidate ratio (~2-3x). Prefer measure_all_subprocess for
    the loop's perf gate; this threaded variant is kept only for simple single-call smoke tests.

    Returns a list of ConfigResult aligned with `cfgs`.
    """
    cfgs = list(cfgs)

    def _run(cfg, gpu_id):
        return measure(namespace, cfg, gpu_id)

    return pool.map_exclusive(_run, cfgs)


def measure_all_subprocess(namespace, cfgs, so_path, gpu_ids=None, baseline_ns="_vllm_fa3_C",
                           on_result=None, timeout=2400) -> list[common.ConfigResult]:
    """Measure perf across `cfgs`, each config in its OWN process pinned to one GPU via
    CUDA_VISIBLE_DEVICES, with up to len(gpu_ids) processes running concurrently.

    This is the CORRECT way to use multiple GPUs for perf: separate processes do NOT share the
    GIL, so host-launch dispatch isn't serialized and the CUDA-event timing is clean (verified:
    separate-process concurrent ratios == isolated ratios; in-process threaded concurrent is a
    ~2-3x artifact). Configs are sharded round-robin across GPUs; each GPU runs ONE process that
    measures its shard sequentially (one kernel per GPU at a time). `on_result(r, gpu)` is called
    live as each config completes (for progress logging). Returns ConfigResults aligned to `cfgs`.
    """
    import os
    import sys
    import json
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor

    if gpu_ids is None:
        gpu_ids = C.GPU_IDS
    cfgs = list(cfgs)
    shards = {g: [] for g in gpu_ids}
    for i, cfg in enumerate(cfgs):
        shards[gpu_ids[i % len(gpu_ids)]].append(cfg)

    results: dict = {}
    lock = threading.Lock()

    def run_shard(gpu):
        shard = shards[gpu]
        if not shard:
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTHONPATH=C.REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        cmd = [sys.executable, "-m", "fa3agent.perf._measure_main", so_path, namespace] + \
              [c.name for c in shard]
        try:
            p = subprocess.Popen(cmd, cwd=C.REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
            for line in p.stdout:
                line = line.strip()
                if line.startswith("MEASURE_JSON:"):
                    d = json.loads(line[len("MEASURE_JSON:"):])
                    r = common.ConfigResult(**{k: v for k, v in d.items()
                                               if k in common.ConfigResult.__dataclass_fields__})
                    with lock:
                        results[r.config] = r
                    if on_result:
                        try:
                            on_result(r, gpu)
                        except Exception:
                            pass
            p.wait(timeout=timeout)
        except Exception as e:  # noqa
            for c in shard:
                with lock:
                    results.setdefault(c.name, common.ConfigResult(
                        config=c.name, correct=True, ratio=0.0, note=f"perf shard err: {e!r}"[:200]))

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        list(ex.map(run_shard, gpu_ids))

    return [results.get(c.name, common.ConfigResult(config=c.name, correct=True, ratio=0.0,
                                                    note="no perf result")) for c in cfgs]

