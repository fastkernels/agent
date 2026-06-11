"""Subprocess perf runner: measure a SHARD of configs on ONE pinned GPU (device 0 in this
process's masked view; the parent sets CUDA_VISIBLE_DEVICES).

WHY a process and not a thread: measuring multiple GPUs from threads in one Python process
shares the GIL, and these attention kernels are tiny (tens of us). measure()'s CUDA-event window
spans host-side dispatch between launches; under GIL contention that host work serializes and
inflates the event-measured time ASYMMETRICALLY (the baseline does more host dispatch than the
candidate), which falsely inflates the baseline/candidate ratio ~2-3x. Verified: separate-process
concurrent timing == isolated timing; in-process threaded concurrent is the artifact. Each config
is therefore measured in its own process; the parent runs up to one process per GPU concurrently.

Usage: python -m fa3agent.perf._measure_main <so_path> <namespace> <cfg_name> [<cfg_name> ...]
Emits one line per config:  MEASURE_JSON:{...asdict(ConfigResult)...}
"""
import json
import sys
from dataclasses import asdict

import torch
import vllm.vllm_flash_attn._vllm_fa3_C  # noqa: F401  (registers the baseline op)

from fa3agent import config as C
from fa3agent.common import ConfigResult
from fa3agent.perf.measure import measure


def _cfg(name):
    for c in list(C.FULL_CONFIGS) + list(C.DEV_CONFIGS):
        if c.name == name:
            return c
    raise SystemExit(f"unknown config {name}")


if __name__ == "__main__":
    so_path, ns = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    try:
        torch.ops.load_library(so_path)
    except Exception as e:  # noqa
        for nm in names:
            print("MEASURE_JSON:" + json.dumps(asdict(ConfigResult(
                config=nm, correct=True, ratio=0.0, note=f"load_library failed: {type(e).__name__}: {e}"))), flush=True)
        sys.exit(0)
    for nm in names:
        try:
            r = measure(ns, _cfg(nm), 0)  # device 0 == the pinned GPU
        except Exception as e:  # noqa
            r = ConfigResult(config=nm, correct=True, ratio=0.0,
                             note=f"measure err: {type(e).__name__}: {e}"[:200])
        print("MEASURE_JSON:" + json.dumps(asdict(r)), flush=True)
