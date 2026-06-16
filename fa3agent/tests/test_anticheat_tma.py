"""Regression test for A1 (anti-cheat TMA false-positive) + A2 (host TMA linkable).

Run from /data-fast/agent:
    CUDA_VISIBLE_DEVICES=0 python -m fa3agent.tests.test_anticheat_tma

Two layers:
  (A1, deterministic) Symbol-aware binary matching: a `cuTensorMapEncodeTiled` driver import
     and an ordinary 'execute' identifier are NOT flagged as the CuTe namespace, while a real
     `cute::` / mangled `_ZN4cute...` symbol and a `cutlass::` symbol STILL fail.  Driven by a
     synthetic `nm -D` / `ldd` output (no GPU/compiler needed) so it always runs.
  (A1+A2, real build, guarded) Compile a tiny .cu that calls the driver TMA encoder
     (cuTensorMapEncodeTiled) in host code: it must BUILD + LOAD (A2: libcuda linked) and then
     PASS anti-cheat (A1: the real TMA-importing .so is accepted).  Degrades to PARTIAL when the
     toolkit/torch/GPU is unavailable.
"""
from __future__ import annotations

import os
import sys
import tempfile

from fa3agent.sandbox import anticheat, build
from fa3agent import config as C

SMOKE_ROOT = os.path.join(C.WORK_ROOT, "_smoke", "anticheat_tma")

# A synthetic `nm -D` dump exercising the false-positive and the true-positive cases.
_FAKE_NM = """\
0000000000000000 A SOMEVERSION
                 U cuTensorMapEncodeTiled
                 U cuTensorMapReplaceAddress
0000000000001000 T execute_attention_kernel
0000000000002000 T _ZN4cute6detail4doitEv
0000000000003000 T _ZN7cutlass4gemm3runEv
0000000000004000 T _ZN3fa36mha_fwdEv
"""
# A synthetic `ldd` dump: only libcute / libcublas are real forbidden deps; a directory named
# '.../execute/...' and the allowed torch/cuda libs must NOT be flagged.
_FAKE_LDD = """\
\tlibtorch.so => /opt/torch/lib/libtorch.so (0x00007f0000000000)
\tlibcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12 (0x00007f0000001000)
\tlibcuda.so.1 => /usr/lib/x86_64-linux-gnu/libcuda.so.1 (0x00007f0000002000)
\tlibmyexecutekit.so => /opt/execute/lib/libmyexecutekit.so (0x00007f0000003000)
"""


def _ok(msg: str) -> None:
    print(f"  PASS: {msg}")


def _fail(msg: str) -> int:
    print(f"  FAIL: {msg}")
    return 1


def _test_symbol_matcher() -> int:
    """Deterministic unit-test of the symbol-aware matcher (no GPU/compiler)."""
    print("[1] symbol-aware binary matching (synthetic nm/ldd)")
    failures = 0
    m = anticheat._forbidden_symbol_match
    # cute false-positives that must NOT match.
    for sym in ("cuTensorMapEncodeTiled", "cuTensorMapReplaceAddress",
                "execute_attention_kernel", "_ZN3fa36mha_fwdEv"):
        if m(sym, "cute"):
            failures += _fail(f"{sym!r} wrongly flagged as 'cute'")
    # real CuTe / cutlass symbols that MUST match.
    if not m("_ZN4cute6detail4doitEv", "cute"):
        failures += _fail("mangled _ZN4cute... not flagged as 'cute'")
    if not m("_ZN7cutlass4gemm3runEv", "cutlass"):
        failures += _fail("mangled cutlass symbol not flagged")
    if not m("cute::foo", "cute"):
        failures += _fail("demangled 'cute::foo' not flagged")
    if failures == 0:
        _ok("matcher: TMA/execute pass; cute::/4cute/cutlass fail")

    # End-to-end through _scan_binary with monkeypatched nm/ldd output.
    saved = anticheat._run
    try:
        def fake_run(cmd):
            if cmd[0] == "nm":
                return True, _FAKE_NM
            if cmd[0] == "ldd":
                return True, _FAKE_LDD
            return False, ""
        anticheat._run = fake_run  # type: ignore[assignment]
        probe = os.path.join(SMOKE_ROOT, "probe.so")
        os.makedirs(SMOKE_ROOT, exist_ok=True)
        open(probe, "wb").close()  # _scan_binary requires the path to exist
        viol: list[str] = []
        anticheat._scan_binary(probe, viol)
        joined = " | ".join(viol)
        # cuTensorMap import + execute lib must not appear; cute + cutlass must.
        if "cuTensorMap" in joined:
            failures += _fail(f"scan flagged a cuTensorMap driver symbol: {joined}")
        if "execute" in joined.lower() and "libcute" not in joined.lower():
            failures += _fail(f"scan flagged the 'execute' dir/symbol: {joined}")
        has_cute = any("'cute'" in v for v in viol)
        has_cutlass = any("'cutlass'" in v for v in viol)
        if not (has_cute and has_cutlass):
            failures += _fail(f"scan missed the real cute/cutlass symbols: {joined}")
        if failures == 0:
            _ok("_scan_binary: real cute+cutlass caught, TMA/execute clean")
    finally:
        anticheat._run = saved  # type: ignore[assignment]
    return failures


_TMA_SMOKE_CU = """\
#include <torch/library.h>
#include <cuda.h>

#ifndef FA3_LIB_NAME
#define FA3_LIB_NAME _vllm_fa3_C
#endif

// Host-side reference to the CUDA driver TMA encoder: forces the .so to link libcuda (A2) and
// to export a 'cuTensorMapEncodeTiled' dynamic import (the A1 anti-cheat false-positive case).
static int64_t fwd() {
  CUtensorMap map;
  cuuint64_t gdim[2] = {64, 64};
  cuuint64_t gstr[1] = {128};
  cuuint32_t bdim[2] = {32, 32};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)0, gdim, gstr,
      bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int64_t)r;
}

TORCH_LIBRARY(FA3_LIB_NAME, m) {
  m.def("fwd() -> int", &fwd);
}
"""


def _test_real_tma_build() -> int:
    """Real build: a TMA-using .cu must build+load (A2) and pass anti-cheat (A1)."""
    print("[2] real TMA smoke build + load + anti-cheat")
    os.makedirs(SMOKE_ROOT, exist_ok=True)
    cu = os.path.join(SMOKE_ROOT, "tma_smoke.cu")
    with open(cu, "w") as fh:
        fh.write(_TMA_SMOKE_CU)
    work_dir = os.path.join(SMOKE_ROOT, "build")
    os.makedirs(work_dir, exist_ok=True)
    ns = "_distill_fa3_C_tma_smoke"
    br = build.build_candidate(cu, ns, work_dir)
    print(f"    ok={br.ok} env_fatal={br.env_fatal} dur={br.duration_s:.1f}s")
    if br.error:
        print(f"    error/warn: {br.error[:300]}")
    if not br.ok:
        # A genuine TMA-link failure is an A2 regression; an unavailable toolkit is PARTIAL.
        if "cuTensorMapEncodeTiled" in (br.error or "") or "undefined symbol" in (br.error or "").lower():
            return _fail(f"TMA host symbol did not link/load (A2 regression): {br.error[:200]}")
        print("    PARTIAL: build env unavailable (no nvcc/torch/GPU) -> skipping real-build check")
        return 0
    _ok("TMA smoke kernel built + loaded (libcuda linked, cuTensorMapEncodeTiled resolved)")

    ac = anticheat.check(cu, br.so_path)
    print(f"    anticheat ok={ac.ok} violations={ac.violations}")
    if ac.ok:
        _ok("real TMA-importing .so accepted by anti-cheat (no 'cute' false-positive)")
        return 0
    return _fail(f"TMA-importing .so wrongly rejected by anti-cheat: {ac.violations}")


def main() -> int:
    failures = 0
    failures += _test_symbol_matcher()
    failures += _test_real_tma_build()
    print()
    if failures == 0:
        print("SMOKE: PASS (anti-cheat TMA + host-link)")
    else:
        print(f"SMOKE: FAIL ({failures} failing checks)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
