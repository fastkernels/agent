"""Smoke test for the verify core (golden cache + correctness checker).

Run from /data-fast/agent:
    CUDA_VISIBLE_DEVICES=1 python -m fa3agent.tests.test_verify_core
or
    CUDA_VISIBLE_DEVICES=1 python fa3agent/tests/test_verify_core.py

Loads the prebuilt SEED candidate .so (registers torch.ops._distill_fa3_C.fwd)
and asserts check_correctness on the DEV configs reports ALL correct=True,
printing per-config err_ratio / lse_err for visibility.
"""
import os
import sys

import torch

from fa3agent import config as C
from fa3agent.verify import correctness, golden

SEED_SO = os.path.join(C.WORK_ROOT, "_seedtest", "build", "_distill_fa3_C.so")
NS = "_distill_fa3_C"


def main() -> int:
    assert os.path.exists(SEED_SO), f"seed .so missing: {SEED_SO}"
    torch.ops.load_library(SEED_SO)
    assert hasattr(torch.ops, NS) and hasattr(getattr(torch.ops, NS), "fwd"), \
        f"loading {SEED_SO} did not register torch.ops.{NS}.fwd"

    cfgs = C.DEV_CONFIGS

    # 1) GoldenCache: first get computes, second get is served from memo/disk.
    gc = golden.GoldenCache(device="cuda")
    for cfg in cfgs:
        d = gc.get(cfg)
        assert isinstance(d, dict) and "out" in d and "lse" in d and "baseline_err" in d
        assert os.path.exists(gc._disk_path(cfg, 0)), "disk cache not written"
        be = gc.golden_baseline_error(cfg)
        assert be == d["baseline_err"]
        # module-level convenience must agree
        assert abs(golden.golden_baseline_error(cfg) - be) < 1e-9

    # 2) check_correctness on the seed candidate vs baseline.
    results = correctness.check_correctness(NS, cfgs, seed=0, gpu_id=0)
    assert len(results) == len(cfgs)

    print(f"{'config':45s} {'correct':8s} {'err_ratio':>10s} {'out_err':>10s} "
          f"{'base_err':>10s} {'lse_err':>10s}  note")
    all_ok = True
    for r in results:
        all_ok = all_ok and r.correct
        print(f"{r.config:45s} {str(r.correct):8s} {r.err_ratio:10.4f} {r.out_err:10.4e} "
              f"{r.baseline_err:10.4e} {r.lse_err:10.4e}  {r.note}")

    # perf fields must be untouched (left 0) by the correctness checker.
    for r in results:
        assert r.ratio == 0.0 and r.cand_latency_ms == 0.0 and r.base_latency_ms == 0.0, \
            "perf fields should be left at 0 by check_correctness"

    assert all_ok, "seed candidate must pass correctness on ALL dev configs"
    print("\nPASS: all dev configs correct=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
