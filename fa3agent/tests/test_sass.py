"""Smoke test for sec 2.3 (sass/analyze.py) + sec 2.6 (sass/cutlass_oracle.py).

Run from /data-fast/agent so `import fa3agent...` resolves:
    CUDA_VISIBLE_DEVICES=1 python fa3agent/tests/test_sass.py

Exercises:
  1. analyze(seed.so, '_distill_fa3_C', baseline_so=BASELINE_SO)
     -> seed has FFMA/no-HGMMA + spills; baseline has HGMMA/UTMALDG; diff says so.
  2. build_oracle(DEV_CONFIGS[0], tmpdir) -> prints which tier + hints (or degraded note),
     then feeds the oracle cubin back into analyze() for an oracle diff.

Keeps work bounded (per-function SASS dumps + a tiny inline-PTX oracle), so it runs in a
few seconds rather than the ~2.5 min a naive full FA3 .so dump would take.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

# Allow running both as a module and as a bare script from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fa3agent import config              # noqa: E402
from fa3agent.sass import analyze        # noqa: E402
from fa3agent.sass import cutlass_oracle  # noqa: E402

SEED_SO = os.path.join(config.WORK_ROOT, "_seedtest", "build", "_distill_fa3_C.so")


def _check(cond: bool, msg: str, fails: list):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        fails.append(msg)


def test_analyze(fails: list) -> dict:
    print("== analyze(seed, baseline_so=BASELINE_SO) ==")
    if not os.path.exists(SEED_SO):
        _check(False, f"seed .so present at {SEED_SO}", fails)
        return {}
    t = time.time()
    r = analyze.analyze(SEED_SO, config.DISTILL_NAMESPACE, baseline_so=config.BASELINE_SO)
    dt = time.time() - t
    print(f"  (analyze took {dt:.1f}s; {r.get('num_kernels_analyzed')} of "
          f"{r.get('num_kernels_total')} kernels disassembled)")

    _check(r.get("ok") is True, "analyze returned ok=True", fails)
    sm = r.get("summary", {})
    print(f"  seed summary:     {sm}")
    _check(not sm.get("uses_wgmma", True), "seed has NO HGMMA (FFMA-based)", fails)
    _check(sm.get("ffma_count", 0) > 0, "seed has FFMA instructions", fails)

    bsm = r.get("baseline_summary", {})
    print(f"  baseline summary: {bsm}")
    _check(bsm.get("uses_wgmma", False), "baseline HAS HGMMA (wgmma)", fails)
    _check(bsm.get("uses_tma", False), "baseline HAS UTMALDG (TMA)", fails)

    diff = r.get("baseline_diff", [])
    print("  baseline diff:")
    for ln in diff:
        print(f"     - {ln}")
    _check(any("wgmma" in ln.lower() for ln in diff),
           "diff reports candidate-vs-baseline wgmma gap", fails)

    # bound the time (must be far under the naive-full-dump cost)
    _check(dt < 60, f"analyze stayed bounded ({dt:.1f}s < 60s)", fails)

    # per-kernel shape
    ks = r.get("kernels", [])
    _check(len(ks) >= 1 and all(
        set(["name", "instr_mix", "regs", "smem", "spills"]) <= set(k) for k in ks),
        "each kernel has name/instr_mix/regs/smem/spills", fails)
    for k in ks:
        print(f"     kernel {k['name'][:46]!r} regs={k['regs']} smem={k['smem']} "
              f"spills={k['spills']} mix={k['instr_mix']}")
    return r


def test_oracle(fails: list):
    print("\n== build_oracle(DEV_CONFIGS[0], tmpdir) ==")
    wd = tempfile.mkdtemp(prefix="fa3_oracle_")
    t = time.time()
    orc = cutlass_oracle.build_oracle(config.DEV_CONFIGS[0], wd)
    dt = time.time() - t
    print(f"  (build_oracle took {dt:.1f}s)")
    print(f"  oracle tier: {orc.get('oracle')}")
    print(f"  cubin_path:  {orc.get('cubin_path')}")
    print(f"  instr_mix:   {orc.get('instr_mix')}")
    print(f"  note:        {orc.get('note')}")
    print(f"  tier_log:    {orc.get('tier_log')}")
    print("  hints:")
    for h in orc.get("hints", []):
        print(f"     - {h}")

    _check(orc.get("oracle") in ("cutedsl", "wgmma_nvcc", "baseline"),
           "oracle built at some tier (not 'none')", fails)
    _check(len(orc.get("hints", [])) >= 1, "oracle returned hints", fails)
    _check(dt < 240, f"build_oracle stayed bounded ({dt:.1f}s < 240s)", fails)

    # If the oracle has a real wgmma reference, feed it back into analyze() for a diff.
    cubin = orc.get("cubin_path")
    if cubin and os.path.exists(cubin) and os.path.exists(SEED_SO):
        r = analyze.analyze(SEED_SO, config.DISTILL_NAMESPACE, oracle_cubin=cubin)
        odiff = r.get("oracle_diff", [])
        print("  seed-vs-oracle diff:")
        for ln in odiff:
            print(f"     - {ln}")
        osm = r.get("oracle_summary", {})
        if osm.get("uses_wgmma"):
            _check(any("wgmma" in ln.lower() for ln in odiff),
                   "oracle diff reports candidate-vs-oracle wgmma gap", fails)


def main() -> int:
    fails: list = []
    test_analyze(fails)
    test_oracle(fails)
    print("\n" + ("ALL SMOKE CHECKS PASSED" if not fails
                   else f"{len(fails)} CHECK(S) FAILED: {fails}"))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
