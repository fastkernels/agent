"""Smoke test for perf/measure.py (section 3 latency ratio).

Loads the prebuilt SEED candidate .so (slow on purpose), runs measure() on the
first DEV config, and asserts we get back finite, positive baseline + candidate
latencies and a sane (finite, > 0) ratio.  The seed is intentionally slow so a
ratio << 1 is EXPECTED and perfectly fine.

Run from /data-fast/agent (so `import fa3agent...` resolves):
    CUDA_VISIBLE_DEVICES=3 python fa3agent/tests/test_perf.py
"""
from __future__ import annotations

import math
import os
import sys

import torch

from fa3agent import config as C
from fa3agent.perf import measure

SEED_SO = os.path.join(C.WORK_ROOT, "_seedtest", "build", "_distill_fa3_C.so")


def main() -> int:
    # baseline op must be registered by importing the vLLM extension.
    import vllm.vllm_flash_attn._vllm_fa3_C  # noqa: F401  (registers _vllm_fa3_C ops)

    torch.ops.load_library(SEED_SO)  # registers torch.ops._distill_fa3_C.fwd

    cfg = C.DEV_CONFIGS[0]
    print(f"[smoke] measuring seed candidate on cfg={cfg.name} ...")
    res = measure.measure("_distill_fa3_C", cfg, gpu_id=0)

    print(f"[smoke] config           = {res.config}")
    print(f"[smoke] base_latency_ms  = {res.base_latency_ms:.5f}")
    print(f"[smoke] cand_latency_ms  = {res.cand_latency_ms:.5f}")
    print(f"[smoke] ratio (base/cand)= {res.ratio:.5f}")
    print(f"[smoke] note             = {res.note}")

    ok = True
    checks = [
        ("base_latency_ms finite>0", math.isfinite(res.base_latency_ms) and res.base_latency_ms > 0),
        ("cand_latency_ms finite>0", math.isfinite(res.cand_latency_ms) and res.cand_latency_ms > 0),
        ("ratio finite>0", math.isfinite(res.ratio) and res.ratio > 0),
        ("correct placeholder True", res.correct is True),
    ]
    for label, passed in checks:
        print(f"[smoke]   {'PASS' if passed else 'FAIL'}: {label}")
        ok = ok and passed

    print("[smoke] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
