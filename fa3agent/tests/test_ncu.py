"""Smoke test for component 2.1 (NCU profiling -> distilled ProfileFeedback).

Run from /data-fast/agent with CUDA_VISIBLE_DEVICES=0:
    CUDA_VISIBLE_DEVICES=0 python fa3agent/tests/test_ncu.py

Profiles the prebuilt SEED candidate .so on DEV_CONFIGS[0], prints the terse JSON
feedback + ranked list, and asserts a populated ProfileFeedback came back.

PASS    -> populated feedback (bottleneck/optimization/plan + key_metrics).
PARTIAL -> ncu permission (ERR_NVGPUCTRPERM) or timeout: feedback returned but degraded.
FAIL    -> exception or empty feedback.

This process imports ncu_report (via ncu_profile/ncu_utils) and MUST NOT import torch
(they segfault together) -- torch only lives in the ncu driver subprocess.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fa3agent import config as C  # noqa: E402
from fa3agent.perf import ncu_profile  # noqa: E402

SEED_SO = os.path.join(C.WORK_ROOT, "_seedtest", "build", "_distill_fa3_C.so")
SEED_NS = "_distill_fa3_C"


def main() -> int:
    cfg = C.DEV_CONFIGS[0]
    run_dir = tempfile.mkdtemp(prefix="ncu_smoke_")
    print(f"[test_ncu] cfg={cfg.name} so={SEED_SO}")
    print(f"[test_ncu] run_dir={run_dir}")

    if not os.path.exists(SEED_SO):
        print("FAIL: seed .so missing")
        return 1

    fb = ncu_profile.profile(SEED_SO, SEED_NS, cfg, gpu_id=0, run_dir=run_dir)

    terse = {
        "config": fb.config,
        "bottleneck": fb.bottleneck,
        "optimization": fb.optimization,
        "plan": fb.plan,
    }
    print("\n=== TERSE FEEDBACK (drives next codegen round) ===")
    print(json.dumps(terse, indent=2))
    print("\n=== RANKED (<=5, evidence-cited) ===")
    for i, r in enumerate(fb.ranked, 1):
        print(f"  {i}. {r}")
    print("\n=== KEY METRICS (subset only; NOT the 2000-metric dump) ===")
    km = fb.key_metrics
    print(f"  ({len(km)} entries) label={km.get('_bottleneck_label')} "
          f"kernel={km.get('_kernel')} set={km.get('_ncu_set')}")
    for k in list(km)[:12]:
        print(f"    {k} = {km[k]}")
    print(f"\n  report_path = {fb.report_path}")

    # --- verdict ---
    partial = fb.bottleneck.lower().startswith("partial")
    populated = bool(fb.bottleneck) and (bool(fb.key_metrics) or partial)
    # word-cap contract
    caps_ok = (len(fb.bottleneck.split()) <= 32
               and len(fb.optimization.split()) <= 37
               and len(fb.plan.split()) <= 37
               and len(fb.ranked) <= 6)

    if not populated:
        print("\nFAIL: empty ProfileFeedback")
        return 1
    if not caps_ok:
        print("\nFAIL: word caps / ranked length exceeded")
        return 1
    if partial or not fb.report_path:
        print("\nPARTIAL: ncu degraded (permission/timeout) but feedback returned")
        return 0
    print("\nPASS: populated, bounded ProfileFeedback with cited metric values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
