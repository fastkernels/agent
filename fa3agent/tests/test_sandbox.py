"""Smoke test for the sandbox build + anti-cheat components (2.5 + build).

Run from /data-fast/agent:
    CUDA_VISIBLE_DEVICES=0 python -m fa3agent.tests.test_sandbox

Covers:
  (1) build_candidate(seed_kernel.cu, '_distill_fa3_C', tmp) -> ok and schema matches.
  (2) anticheat.check(seed_kernel.cu, built_so) -> ok (clean source + clean binary).
  (3) a tiny BAD .cu (#include <cutlass/cutlass.h> + a cute:: call) -> anticheat REJECTS
      it via the source scan.
Keeps everything under a writable tmp dir in the checkpoint root.
"""
from __future__ import annotations

import os
import sys

from fa3agent.sandbox import build, anticheat
from fa3agent import config as C

SEED_CU = os.path.join(C.PKG_DIR, "seed_kernel.cu")
SMOKE_ROOT = os.path.join(C.CHECKPOINT_ROOT, "_smoke", "sandbox")


def _ok(msg: str) -> None:
    print(f"  PASS: {msg}")


def _fail(msg: str) -> int:
    print(f"  FAIL: {msg}")
    return 1


def main() -> int:
    os.makedirs(SMOKE_ROOT, exist_ok=True)
    failures = 0

    # ---- (1) build the seed kernel ---------------------------------------------------
    print("[1] build_candidate(seed_kernel.cu, '_distill_fa3_C')")
    work_dir = os.path.join(SMOKE_ROOT, "build")
    os.makedirs(work_dir, exist_ok=True)
    br = build.build_candidate(SEED_CU, C.DISTILL_NAMESPACE, work_dir)
    print(f"    ok={br.ok} env_fatal={br.env_fatal} dur={br.duration_s:.1f}s")
    if br.error:
        print(f"    error/warn: {br.error}")
    if not br.ok:
        failures += _fail(f"build failed: {br.error}")
        print("    --- log tail ---")
        print("\n".join(br.log.splitlines()[-15:]))
        return failures  # nothing else works without a build
    if not br.so_path or not os.path.isfile(br.so_path):
        failures += _fail("build ok but so_path missing")
        return failures
    _ok(f"built {os.path.basename(br.so_path)}")
    if br.error.startswith("WARN:"):
        failures += _fail(f"schema mismatch warning present: {br.error}")
    else:
        _ok("fwd schema matches config.FWD_SCHEMA (no mismatch warning)")

    # ---- (2) anti-cheat on the (clean) seed source + binary --------------------------
    print("[2] anticheat.check(seed_kernel.cu, built_so)")
    ac = anticheat.check(SEED_CU, br.so_path)
    print(f"    ok={ac.ok} scanned_source={ac.scanned_source} "
          f"scanned_binary={ac.scanned_binary} violations={ac.violations}")
    if not ac.scanned_source or not ac.scanned_binary:
        failures += _fail("expected both source and binary scans to run")
    if ac.ok:
        _ok("clean seed kernel accepted (no violations)")
    else:
        failures += _fail(f"clean seed kernel REJECTED: {ac.violations}")

    # ---- (3) anti-cheat must REJECT a cheating .cu (source scan) ----------------------
    print("[3] anticheat.check(bad.cu) -> must REJECT")
    bad_cu = os.path.join(SMOKE_ROOT, "bad_cheat.cu")
    with open(bad_cu, "w") as fh:
        fh.write(
            "// a deliberately cheating kernel\n"
            "#include <torch/library.h>\n"
            "#include <cutlass/cutlass.h>   // forbidden include\n"
            "#include <cute/tensor.hpp>     // also forbidden\n"
            "void f() {\n"
            "  /* the line below uses a forbidden namespace */\n"
            "  auto x = cute::make_tensor();\n"
            "  cutlass::gemm::device::Gemm g;\n"
            "}\n"
        )
    ac_bad = anticheat.check(bad_cu)
    print(f"    ok={ac_bad.ok} violations={ac_bad.violations}")
    if ac_bad.ok:
        failures += _fail("cheating .cu was ACCEPTED (should be rejected)")
    else:
        joined = " ".join(ac_bad.violations).lower()
        if "include" in joined and "cute::" in joined:
            _ok("cheating .cu rejected (forbidden include + cute:: token detected)")
        else:
            _ok(f"cheating .cu rejected ({len(ac_bad.violations)} violations)")

    # ---- (3b) comment-only mention must NOT trip the scanner -------------------------
    print("[3b] anticheat.check(comment-only mention) -> must ACCEPT")
    clean_cu = os.path.join(SMOKE_ROOT, "clean_comment.cu")
    with open(clean_cu, "w") as fh:
        fh.write(
            "// This kernel does NOT use cutlass:: or cute:: -- mentioned only here.\n"
            "/* #include <cutlass/cutlass.h> is intentionally commented out */\n"
            "#include <torch/library.h>\n"
            "int g() { return 0; }\n"
        )
    ac_clean = anticheat.check(clean_cu)
    print(f"    ok={ac_clean.ok} violations={ac_clean.violations}")
    if ac_clean.ok:
        _ok("comment-only mentions correctly ignored")
    else:
        failures += _fail(f"comment-only mentions wrongly flagged: {ac_clean.violations}")

    print()
    if failures == 0:
        print("SMOKE: PASS (all checks)")
    else:
        print(f"SMOKE: FAIL ({failures} failing checks)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
