"""Hopper reference "oracle" builder (sec 2.6).

Build a *minimal* Hopper bf16 reference kernel OUT OF BAND so the agent can inspect
how an ideal kernel issues `wgmma` (HGMMA) and `cp.async.bulk.tensor` (TMA / UTMALDG),
then distill that into a few actionable hints + an instruction histogram.

CRITICAL anti-cheat note: CUTLASS/CuTe is NEVER #included or vendored into the agent's
output `.cu`.  The oracle is a SEPARATE build artifact we only read SASS/PTX from.  The
inline-PTX snippet we compile here uses ONLY raw PTX ISA (`wgmma.mma_async...`,
`cp.async.bulk.tensor...`, `mbarrier.*`) + the CUDA driver `cuTensorMapEncodeTiled` path,
all of which are free to re-derive in our own kernel.

Tier strategy (degrade gracefully, report which tier produced the oracle):
  1. "cutedsl"     - CuTeDSL (nvidia-cutlass-dsl 4.5.2) cute.compile[KeepPTX,KeepCUBIN]
                     of a tiny bf16 GEMM, run in a subprocess (DSL needs a real file).
  2. "wgmma_nvcc"  - nvcc -arch=sm_90a -cubin on an inline wgmma + TMA snippet (RELIABLY
                     emits HGMMA + UTMALDG; this is the workhorse).
  3. "baseline"    - last resort: read the prebuilt baseline `_vllm_fa3_C` .so SASS.

Public API (interfaces.py):
    build_oracle(cfg, work_dir) -> dict
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Optional

from fa3agent import config
from fa3agent.sass import analyze as sass_analyze

GENCODE_ARCH = config.GPU_ARCH  # "sm_90a"
NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"

# Bounds: building the oracle is auxiliary work; never let it dominate a step.
CUTEDSL_TIMEOUT_S = 180
NVCC_TIMEOUT_S = 120


# --------------------------------------------------------------------------------------
# Tier 2 source: inline wgmma (HGMMA) + TMA (UTMALDG) snippet.  Parameterised by the
# wgmma N tile and the accumulator element count so we can roughly track the target cfg's
# head_dim, but its only purpose is to emit the *primitives* for SASS/PTX inspection.
# --------------------------------------------------------------------------------------
def _wgmma_snippet(n_tile: int = 128) -> str:
    nacc = n_tile // 2  # m64nNk16 f32 accum => N/2 f32 regs per thread
    accs = "\n".join(
        f"  float a{i}=0.f;" for i in range(nacc))
    acc_constraints = ",".join(f'"+f"(a{i})' for i in range(nacc))
    acc_store = "\n".join(
        f"  out[threadIdx.x*{nacc}+{i}]=a{i};" for i in range(nacc))
    acc_ops = "{" + ",".join(f"%{i}" for i in range(nacc)) + "}"
    desc_a = f"%{nacc}"
    desc_b = f"%{nacc + 1}"
    return textwrap.dedent(f"""\
        #include <cuda.h>
        #include <cuda_bf16.h>
        #include <cstdint>
        // Minimal Hopper oracle: TMA bulk-tensor load (UTMALDG) + warpgroup MMA (HGMMA).
        // Pure PTX ISA only -- NO cutlass/cute. For SASS/PTX inspection in sec 2.6.
        extern "C" __global__ void __launch_bounds__(128)
        oracle_wgmma_n{n_tile}(const __grid_constant__ CUtensorMap tmap,
                               float* out, uint64_t descA, uint64_t descB) {{
          __shared__ __align__(1024) uint8_t smem[16384];
          __shared__ __align__(8) uint64_t mbar;
        {accs}
          const CUtensorMap* tmap_ptr = &tmap;
          if (threadIdx.x == 0) {{
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\\n" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&mbar)));
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&mbar)), "r"(16384));
            asm volatile(
              "cp.async.bulk.tensor.2d.shared::cta.global"
              ".mbarrier::complete_tx::bytes [%0], [%1, {{%2, %3}}], [%4];\\n" ::
              "r"((uint32_t)__cvta_generic_to_shared(smem)),
              "l"(tmap_ptr), "r"(0), "r"(0),
              "r"((uint32_t)__cvta_generic_to_shared(&mbar)) : "memory");
          }}
          asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");
          asm volatile(
            "wgmma.mma_async.sync.aligned.m64n{n_tile}k16.f32.bf16.bf16 "
            "{acc_ops}, {desc_a}, {desc_b}, 1, 1, 1, 0, 0;\\n"
            : {acc_constraints}
            : "l"(descA), "l"(descB) : "memory");
          asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");
          asm volatile("wgmma.wait_group.sync.aligned 0;\\n" ::: "memory");
        {acc_store}
        }}
        """)


# --------------------------------------------------------------------------------------
# Tier 1 source: CuTeDSL driver written to a file (the DSL cannot JIT from a -c string;
# it reads source via inspect.getsourcelines).  KeepPTX/KeepCUBIN dump the artifacts into
# the process CWD, so we run it with cwd=work_dir and glob for cutlass_launch_*.cubin.
# --------------------------------------------------------------------------------------
def _cutedsl_driver(m: int, n: int, k: int) -> str:
    return textwrap.dedent(f"""\
        import os, glob, sys
        try:
            import torch
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack
            from cutlass.cute import KeepPTX, KeepCUBIN
        except Exception as e:
            print("CUTEDSL_IMPORT_FAIL", type(e).__name__, str(e)[:160]); sys.exit(3)

        @cute.kernel
        def gemm_k(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            i = bidx * 64 + tidx
            mm = cute.size(gC, mode=[0]); nn = cute.size(gC, mode=[1])
            kk = cute.size(gA, mode=[1])
            if i < mm:
                for j in cutlass.range(nn):
                    acc = cutlass.Float32(0.0)
                    for p in cutlass.range(kk):
                        acc += gA[i, p].to(cutlass.Float32) * gB[p, j].to(cutlass.Float32)
                    gC[i, j] = acc

        @cute.jit
        def launch(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
            gemm_k(mA, mB, mC).launch(grid=[{(m + 63) // 64}, 1, 1], block=[64, 1, 1])

        def main():
            M, N, K = {m}, {n}, {k}
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
            c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
            ta, tb, tc = (from_dlpack(x) for x in (a, b, c))
            try:
                cute.compile[KeepPTX, KeepCUBIN](launch, ta, tb, tc)
            except Exception as e:
                print("CUTEDSL_COMPILE_FAIL", type(e).__name__, str(e)[:160]); sys.exit(4)
            cubins = glob.glob(os.path.join(os.getcwd(), "*.cubin"))
            ptxs = glob.glob(os.path.join(os.getcwd(), "*.ptx"))
            if not cubins:
                print("CUTEDSL_NO_CUBIN"); sys.exit(5)
            print("CUTEDSL_OK", cubins[0], (ptxs[0] if ptxs else ""))

        if __name__ == "__main__":
            main()
        """)


def _run(cmd, timeout, cwd=None, env=None):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout, cwd=cwd, env=env)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return 124, "<timeout>"
    except Exception as e:  # noqa: BLE001
        return 1, f"<error: {e}>"


def _instr_mix_from_cubin(cubin_path: str) -> dict:
    """Run the sec-2.3 analyzer on the oracle cubin -> aggregate instr histogram."""
    res = sass_analyze.analyze(cubin_path, namespace="oracle")
    agg: dict = {}
    for k in res.get("kernels", []):
        for cls, v in k.get("instr_mix", {}).items():
            agg[cls] = agg.get(cls, 0) + v
    return agg


def _hints_from_mix(mix: dict, tier: str, cfg) -> list[str]:
    """Distilled, bounded hints derived from the oracle instruction mix + known FA3 facts."""
    hints: list[str] = []
    wgmma = mix.get("HGMMA", 0) + mix.get("QGMMA", 0) + mix.get("IGMMA", 0)
    tma = mix.get("UTMALDG", 0) + mix.get("UBLKCP", 0)
    if wgmma:
        hints.append(
            f"oracle issues {wgmma} warpgroup-MMA (HGMMA) -> the distilled kernel MUST use "
            "wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 (N>=64 for near-peak), "
            "not FFMA, for both Q@K^T and P@V.")
    if tma:
        hints.append(
            f"oracle issues {tma} TMA bulk loads (UTMALDG) -> load K/V tiles via "
            "cp.async.bulk.tensor + mbarrier expect_tx/try_wait, not LDG, to bypass L1 "
            "and overlap copy with compute.")
    if mix.get("LDSM_STSM"):
        hints.append("oracle uses LDSM/STSM (ldmatrix/stmatrix) for smem<->reg matrix moves "
                     "in the epilogue / RS-operand path.")
    # FA3 algorithmic hints that are always true for the Hopper target (cutlass-cute.md sec3/6).
    hints.append("warp-specialize: 1 producer WG (reg_dealloc, issues TMA) + N consumer WGs "
                 "(reg_alloc, wgmma loop); order math WGs with a NamedBarrier.")
    hints.append("online softmax in log2 domain: P=exp2(scale*S - scale*rowmax); "
                 "corr=exp2((m_prev-m)*scale); rescale O and rowsum by corr; fold log2(e) "
                 "into softmax_scale (exp2 via MUFU, not expf).")
    hints.append("pin wgmma accumulators with wgmma.fence + empty-asm operand touch and "
                 "bracket each group with commit_group/wait_group, or the compiler spills "
                 "(watch LDL/STL).")
    if tier == "baseline":
        hints.append("NOTE: oracle is the prebuilt vLLM FA3 baseline .so (degraded tier); "
                     "its SASS is the production target but mixes many template shapes.")
    elif tier == "wgmma_nvcc":
        hints.append("NOTE: oracle is a minimal inline-PTX wgmma+TMA probe (one MMA, one TMA); "
                     "use it to confirm the SASS forms, not the pipeline cadence.")
    return hints[:7]


def _try_cutedsl(cfg, work_dir: str) -> Optional[dict]:
    # head_dim-ish small GEMM; keep tiny for fast compile.
    hd = getattr(cfg, "head_dim", 128)
    m, n, k = 128, max(64, min(hd, 128)), 32
    drv = os.path.join(work_dir, "cutedsl_oracle_driver.py")
    with open(drv, "w") as f:
        f.write(_cutedsl_driver(m, n, k))
    env = dict(os.environ)
    env["CUTE_DSL_KEEP_PTX"] = "1"
    env["CUTE_DSL_CACHE_DIR"] = os.path.join(work_dir, "cute_cache")
    # Use one GPU only; CuTeDSL needs a device to compile/run.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    rc, log = _run([sys.executable, drv], CUTEDSL_TIMEOUT_S, cwd=work_dir, env=env)
    if rc != 0 or "CUTEDSL_OK" not in log:
        return {"_failed": True, "_log": log.strip().splitlines()[-3:]}
    cubin = None
    for line in log.splitlines():
        if line.startswith("CUTEDSL_OK"):
            parts = line.split()
            if len(parts) >= 2:
                cubin = parts[1]
    if not cubin or not os.path.exists(cubin):
        cubins = glob.glob(os.path.join(work_dir, "*.cubin"))
        cubin = cubins[0] if cubins else None
    if not cubin:
        return {"_failed": True, "_log": ["no cubin produced"]}
    return {"cubin_path": cubin}


def _try_nvcc_wgmma(cfg, work_dir: str) -> Optional[dict]:
    if not (os.path.exists(NVCC) or shutil.which("nvcc")):
        return {"_failed": True, "_log": ["nvcc not found"]}
    hd = getattr(cfg, "head_dim", 128)
    n_tile = 128 if hd >= 128 else (64 if hd >= 64 else 64)
    cu = os.path.join(work_dir, "wgmma_oracle.cu")
    cubin = os.path.join(work_dir, "wgmma_oracle.cubin")
    with open(cu, "w") as f:
        f.write(_wgmma_snippet(n_tile))
    rc, log = _run([NVCC, "-arch=" + GENCODE_ARCH, "-O3", "-cubin", "-o", cubin, cu],
                   NVCC_TIMEOUT_S)
    if rc != 0 or not os.path.exists(cubin):
        return {"_failed": True, "_log": log.strip().splitlines()[-4:]}
    return {"cubin_path": cubin}


def build_oracle(cfg, work_dir: str) -> dict:
    """Build a minimal Hopper reference out of band and distill its SASS/PTX.

    Returns a bounded dict::

        {
          "oracle": "cutedsl" | "wgmma_nvcc" | "baseline" | "none",
          "cubin_path": str | None,         # the artifact to feed analyze(oracle_cubin=...)
          "instr_mix": {HGMMA: n, UTMALDG: m, ...},   # aggregate, key classes only
          "hints": [ "...", ... ],          # <= 7 short actionable lines
          "note": str,                      # one-line status (incl. degraded reasons)
          "tier_log": {cutedsl: ..., wgmma_nvcc: ...},  # why earlier tiers were skipped
        }

    The oracle artifact is SEPARATE from the agent's output .cu and is only read for
    SASS/PTX ideas (anti-cheat: never compiled into the deliverable).
    """
    os.makedirs(work_dir, exist_ok=True)
    tier_log: dict = {}
    t0 = time.time()

    # ---- Tier 1: CuTeDSL (preferred) -------------------------------------------------
    res = _try_cutedsl(cfg, work_dir)
    if res and not res.get("_failed"):
        cubin = res["cubin_path"]
        mix = _instr_mix_from_cubin(cubin)
        # The scalar CuTeDSL GEMM may not emit wgmma; if so it's still a valid PTX/SASS
        # reference, but a real wgmma oracle is more useful -> fall through when absent.
        if mix.get("HGMMA", 0) + mix.get("QGMMA", 0) + mix.get("IGMMA", 0) > 0:
            return _finish("cutedsl", cubin, mix, cfg, tier_log,
                           "CuTeDSL bf16 GEMM compiled; wgmma present.")
        tier_log["cutedsl"] = "compiled but no wgmma in SASS (scalar GEMM); fell through"
        # keep the cubin around as a PTX reference but prefer the wgmma tier
        cutedsl_fallback = (cubin, mix)
    else:
        tier_log["cutedsl"] = "failed: " + "; ".join(res.get("_log", [])) if res else "failed"
        cutedsl_fallback = None

    # ---- Tier 2: nvcc inline wgmma + TMA (reliable) ----------------------------------
    res = _try_nvcc_wgmma(cfg, work_dir)
    if res and not res.get("_failed"):
        cubin = res["cubin_path"]
        mix = _instr_mix_from_cubin(cubin)
        if mix.get("HGMMA", 0) + mix.get("QGMMA", 0) > 0:
            return _finish("wgmma_nvcc", cubin, mix, cfg, tier_log,
                           "inline wgmma+TMA probe compiled; HGMMA/UTMALDG present.")
        tier_log["wgmma_nvcc"] = "compiled but no HGMMA found in SASS"
    else:
        tier_log["wgmma_nvcc"] = "failed: " + "; ".join(res.get("_log", [])) if res else "failed"

    # If CuTeDSL produced *something* (scalar), use it as a PTX reference before baseline.
    if cutedsl_fallback is not None:
        cubin, mix = cutedsl_fallback
        return _finish("cutedsl", cubin, mix, cfg, tier_log,
                       "CuTeDSL GEMM (scalar, no wgmma) used as PTX reference.")

    # ---- Tier 3: baseline .so (last resort) ------------------------------------------
    base = config.BASELINE_SO
    if os.path.exists(base):
        mix = _instr_mix_from_cubin(base)
        return _finish("baseline", base, mix, cfg, tier_log,
                       "degraded: using prebuilt baseline _vllm_fa3_C .so as oracle.")

    return {
        "oracle": "none",
        "cubin_path": None,
        "instr_mix": {},
        "hints": ["oracle build failed at every tier; no Hopper reference available"],
        "note": f"all tiers failed in {time.time() - t0:.1f}s",
        "tier_log": tier_log,
    }


def _finish(tier: str, cubin: str, mix: dict, cfg, tier_log: dict, note: str) -> dict:
    key = ["HGMMA", "QGMMA", "IGMMA", "UTMALDG", "UBLKCP", "LDGSTS", "LDSM_STSM",
           "HMMA", "FFMA", "MUFU", "LDG", "STG", "LDL_SPILL", "STL_SPILL"]
    slim = {k: mix[k] for k in key if mix.get(k)}
    return {
        "oracle": tier,
        "cubin_path": cubin,
        "instr_mix": slim,
        "hints": _hints_from_mix(mix, tier, cfg),
        "note": note,
        "tier_log": tier_log,
    }
