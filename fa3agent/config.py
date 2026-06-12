"""Single source of truth for the FA3-distillation agent.

Everything keys off this module: paths, the EXACT vLLM op-schema contract the
distilled `.cu` must register, the config matrix (correctness + perf coverage),
tolerances, the anti-cheat denylist, the NCU metric subset, and build flags.

Hardware/software facts (this box, verified this session):
  4x NVIDIA H200 (sm_90a / compute 9.0), CUDA 13.0, ncu 2025.3, torch 2.11+cu130,
  vllm 0.22.0.  GPU clocks CANNOT be locked (no sudo) -> perf is always reported as
  a candidate/baseline RATIO measured in the same warmed process.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
# Every root below is env-overridable so the agent runs on ANY box.  Defaults are derived
# from this file's location (so the repo can live anywhere) or are this-cluster conventions
# the spec lets us keep as defaults.

# This package dir (<repo>/fa3agent) and the repo root that contains it, derived from
# __file__ so they are correct wherever the repo is checked out.  REPO_ROOT (used as the
# cwd for `python -m fa3agent...` subprocesses) can still be overridden with FA3_REPO_ROOT;
# PKG_DIR always points at the real package so its bundled resources resolve regardless.
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get("FA3_REPO_ROOT") or os.path.dirname(PKG_DIR)
RESEARCH_DIR = os.path.join(REPO_ROOT, ".research")
REFERENCE_DIR = os.path.join(REPO_ROOT, "reference")

# Checkpoints live OUTSIDE the repo (spec requirement).  The default is this-cluster
# convention; override with FA3_CHECKPOINT_ROOT on another machine.
CHECKPOINT_ROOT = os.environ.get("FA3_CHECKPOINT_ROOT") or "/code/users/goliaro/cuda-agent-checkpoints"

# Scratch / working artifacts (gitignored via .gitignore: build/, *.so). Kept inside the
# checkpoint root so a run is fully self-contained and resumable.
def run_dir(run_id: str) -> str:
    return os.path.join(CHECKPOINT_ROOT, run_id)

# HuggingFace cache root for the vLLM-based gates.  Honors FA3_HF_HOME / HF_HOME from the
# environment; when neither is set we leave it empty so HF falls back to its own default
# (~/.cache/huggingface) instead of a hard-coded box-specific path.
HF_HOME = os.environ.get("FA3_HF_HOME") or os.environ.get("HF_HOME") or ""


def _find_vllm_pkg() -> str:
    """Locate the installed vLLM package dir WITHOUT importing it (no torch side effects).

    Order: FA3_VLLM_PKG override -> importlib spec for the installed `vllm`.  Returns ""
    if vLLM isn't installed (the vLLM-dependent gates then report unavailable).
    """
    override = os.environ.get("FA3_VLLM_PKG")
    if override:
        return override
    try:
        spec = importlib.util.find_spec("vllm")
    except Exception:  # broken finder / partial install -> treat as absent
        return ""
    if spec is None:
        return ""
    locs = list(getattr(spec, "submodule_search_locations", None) or [])
    if locs:
        return locs[0]
    if spec.origin:
        return os.path.dirname(spec.origin)
    return ""


# Prebuilt baseline ("the oracle") shipped inside the installed vLLM.  Located via the
# installed package so it works in any env; override with FA3_VLLM_PKG.
VLLM_PKG = _find_vllm_pkg()
BASELINE_SO = os.path.join(VLLM_PKG, "vllm_flash_attn", "_vllm_fa3_C.abi3.so") if VLLM_PKG else ""
VLLM_FA_WRAPPER = os.path.join(VLLM_PKG, "vllm_flash_attn", "flash_attn_interface.py") if VLLM_PKG else ""

# NCU python module (auto-probed; first existing wins).  ncu_utils also glob-probes the
# usual install roots, so these are fast-path defaults for this box; FA3_NCU_PYTHON, if set,
# is tried first.
NCU_PYTHON_CANDIDATES = [
    p for p in (
        os.environ.get("FA3_NCU_PYTHON"),
        "/opt/nvidia/nsight-compute/2025.3.1/extras/python",
        "/opt/nvidia/nsight-compute/2025.2.1/extras/python",
        "/usr/local/cuda/nsight-compute-2025.3.1/extras/python",
    ) if p
]

# --------------------------------------------------------------------------------------
# Target / toolchain
# --------------------------------------------------------------------------------------
GPU_ARCH = "sm_90a"
GPU_GENCODE = "arch=compute_90a,code=sm_90a"
NCU_CHIP = "gh200"  # for `ncu --query-metrics --chip gh200`
NUM_GPUS = 4
GPU_IDS = list(range(NUM_GPUS))

# Candidate .cu registers ops under this namespace token. The token is build-overridable
# via `-DFA3_LIB_NAME=...` so a test build can coexist in-process with the baseline
# (`_vllm_fa3_C`) for A/B. The FINAL deliverable compiles with the default `_vllm_fa3_C`
# for the real vLLM module swap.
DISTILL_NAMESPACE = "_distill_fa3_C"   # used during development / in-process A/B
DELIVERABLE_NAMESPACE = "_vllm_fa3_C"  # used for the final swap into vLLM

# --------------------------------------------------------------------------------------
# Claude Code headless brain (codegen / optimize).
# --------------------------------------------------------------------------------------
# Use the FULL model id, not the alias.  (settings.json also sets ANTHROPIC_MODEL=claude-opus-4-8.)
HEADLESS_MODEL = "claude-opus-4-8"
# Effort level passed to `claude --effort <level>` (low|medium|high|xhigh|ultracode).
# Default xhigh: deepest single-agent reasoning for hard kernel codegen, while the Python
# orchestrator (NOT the brain) owns the loop/eval/multi-GPU/checkpoint and the "one candidate
# per step" contract.  Use ultracode only when you want the brain to do in-step multi-variant
# search (costlier, the brain spawns its own workflows); use high to economize on long runs.
HEADLESS_EFFORT = "xhigh"
# Carry the Claude session across refine steps via `claude --resume`?  DEFAULT FALSE.
# `--resume` replays the ENTIRE prior conversation as the new request; on the Snowflake-Cortex
# relay that oversized payload fails with retryable "unknown" errors (observed: step 0 fresh
# session OK, step 1 --resume stuck in an api_retry loop).  Each step instead runs a FRESH
# session -- the refine prompt carries the distilled context (last eval, profiler bottleneck,
# knowledge hints) and the champion .cu is on disk, so continuity is preserved without the
# unbounded-growth resume payload.
HEADLESS_RESUME_SESSION = False


# Toolkit-only nvcc flags (NO cutlass/cute include path -> anti-cheat by construction).
NVCC_BASE_FLAGS = [
    "-O3", "-std=c++17", "--use_fast_math",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-gencode", GPU_GENCODE,
    "-lineinfo",            # source<->SASS attribution for NCU/SASS components (NOT -G)
]
# Semantic compile defs that match the vLLM FA3 build surface (forward / varlen / packgqa only).
SEMANTIC_DEFS = [
    "FLASHATTENTION_DISABLE_BACKWARD", "FLASHATTENTION_DISABLE_DROPOUT",
    "FLASHATTENTION_DISABLE_APPENDKV", "FLASHATTENTION_DISABLE_SM8x",
    "FLASHATTENTION_DISABLE_CLUSTER", "FLASHATTENTION_DISABLE_UNEVEN_K",
    "FLASHATTENTION_VARLEN_ONLY", "FLASHATTENTION_PACKGQA_ONLY",
]

# --------------------------------------------------------------------------------------
# EXACT op-schema contract (verbatim, confirmed against the live `.so` this session).
# The distilled .cu MUST register byte-identical schemas (positional call from vLLM).
# --------------------------------------------------------------------------------------
FWD_SCHEMA = (
    "fwd(Tensor! q, Tensor k, Tensor v, Tensor? k_new, Tensor? v_new, Tensor? q_v, "
    "Tensor!? out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new, "
    "Tensor? seqused_q, Tensor? seqused_k, int? max_seqlen_q, int? max_seqlen_k, "
    "Tensor? page_table, Tensor? kv_batch_idx, Tensor? leftpad_k, Tensor? rotary_cos, "
    "Tensor? rotary_sin, Tensor? seqlens_rotary, Tensor? q_descale, Tensor? k_descale, "
    "Tensor? v_descale, float softmax_scale, bool is_causal, int window_size_left, "
    "int window_size_right, float softcap, bool is_rotary_interleaved, "
    "Tensor? scheduler_metadata, int num_splits, bool? pack_gqa, int sm_margin, "
    "Tensor? s_aux, int cp_world_size, int cp_rank, Tensor? cp_tot_seqused_k) -> Tensor[]"
)
GET_SCHED_SCHEMA = (
    "get_scheduler_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, "
    "int num_heads, int num_heads_k, int headdim, int headdim_v, ScalarType qkv_dtype, "
    "Tensor seqused_k, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new, "
    "Tensor? seqused_q, Tensor? leftpad_k, int? page_size, int max_seqlen_k_new, "
    "bool is_causal, int window_size_left, int window_size_right, bool has_softcap, "
    "int num_splits, bool? pack_gqa, int sm_margin) -> Tensor"
)
# `fwd` returns a 4-element list: [out, softmax_lse, out_accum, softmax_lse_accum].
# vLLM unpacks `out, softmax_lse, _, _ = ...`; elements 2,3 may be empty when num_splits==1.
FWD_RETURN_LEN = 4

# --------------------------------------------------------------------------------------
# Anti-cheating denylist (§2.5).
# --------------------------------------------------------------------------------------
# Forbidden include substrings (source scan) -- any occurrence rejects the candidate.
FORBIDDEN_INCLUDE_SUBSTR = [
    "cutlass", "cute/", "cute.h", "cublas", "cublasLt", "cudnn", "flashinfer",
    "thunderkittens", "kittens.cuh", "triton",
]
# Forbidden source tokens (namespaces / API) outside of comments.
FORBIDDEN_SOURCE_TOKENS = [
    "cute::", "cutlass::", "cublas", "cudnn", "flashinfer", "tvm::", "tvm_ffi",
    "at::scaled_dot_product_attention", "at::native::", "_vllm_fa3_C",  # must not link the original
]
# Forbidden dynamic symbols / shared-object names (nm -D / ldd scan of the built .so).
FORBIDDEN_DYNAMIC_SUBSTR = [
    "cutlass", "cute", "cublas", "cudnn", "flashinfer", "triton", "nvrtc",
]
# torch MUST appear only in host code; it is allowed to link libtorch (the boundary).
ALLOWED_LINK_SUBSTR = ["torch", "c10", "cuda", "cudart", "stdc++", "gcc_s", "c.so", "m.so",
                       "pthread", "dl.so", "rt.so", "nvToolsExt", "gomp"]

# --------------------------------------------------------------------------------------
# Config matrix.  A Config fully specifies an attention problem instance; shapes.py turns
# it into identical input tensors for BOTH baseline and candidate.
# --------------------------------------------------------------------------------------
DTYPES = ("bf16", "fp16", "fp8e4m3")
TORCH_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp8e4m3": "float8_e4m3fn"}


@dataclass(frozen=True)
class Config:
    name: str
    phase: str               # "prefill" | "decode"
    dtype: str               # one of DTYPES
    head_dim: int
    num_heads_q: int
    num_heads_kv: int
    causal: bool = True
    paged: bool = True        # paged KV cache (decode) vs contiguous varlen (prefill)
    block_size: int = 16
    num_splits: int = 0       # 0 = FA3 auto-heuristic; 1 = forced single; >1 forced split
    softcap: float = 0.0
    window_left: int = -1     # -1 = no sliding window
    window_right: int = -1
    batch: int = 1
    q_len: int = 1            # per-sequence query length (decode=1; prefill=prompt len)
    k_len: int = 256          # per-sequence kv/context length
    head_dim_v: Optional[int] = None  # defaults to head_dim
    weight: float = 1.0       # perf weight (hot configs weigh more in the geomean)

    @property
    def hdv(self) -> int:
        return self.head_dim_v if self.head_dim_v is not None else self.head_dim

    def to_dict(self) -> dict:
        return asdict(self)


# Hot configs (the two production-critical shapes) weigh most.
_HOT = 4.0

# --- DEV set: small + fast, exercises the machinery on a real run (§4) ---------------
DEV_CONFIGS = [
    Config("dev_decode_hd128_bf16_gqa_paged", "decode", "bf16", 128, 8, 2,
           batch=4, q_len=1, k_len=512, weight=_HOT),
    Config("dev_prefill_hd128_bf16_causal_paged", "prefill", "bf16", 128, 8, 2,
           paged=True, batch=2, q_len=512, k_len=512, weight=_HOT),
    Config("dev_decode_hd64_bf16", "decode", "bf16", 64, 8, 8,
           batch=2, q_len=1, k_len=256, weight=1.0),
]

# --- FULL set: complete correctness + perf coverage of what vLLM exercises -----------
def _full_configs() -> list[Config]:
    cfgs: list[Config] = []
    # head-dim sweep, bf16, decode + prefill, paged, GQA 8:2, causal
    for hd in (64, 96, 128, 192, 256):
        w = _HOT if hd == 128 else 1.0
        cfgs.append(Config(f"decode_hd{hd}_bf16_gqa", "decode", "bf16", hd, 8, 2,
                           batch=8, q_len=1, k_len=1024, weight=w))
        cfgs.append(Config(f"prefill_hd{hd}_bf16_causal", "prefill", "bf16", hd, 8, 2,
                           paged=True, batch=2, q_len=1024, k_len=1024, weight=w))
    # fp16 + fp8 dtype coverage at hd128
    cfgs.append(Config("decode_hd128_fp16_gqa", "decode", "fp16", 128, 8, 2,
                       batch=8, q_len=1, k_len=1024))
    cfgs.append(Config("decode_hd128_fp8_gqa", "decode", "fp8e4m3", 128, 8, 2,
                       batch=8, q_len=1, k_len=1024))
    cfgs.append(Config("prefill_hd128_fp8_causal", "prefill", "fp8e4m3", 128, 8, 2,
                       paged=True, batch=2, q_len=1024, k_len=1024))
    # softcap (Gemma), sliding window, forced split-KV, MQA, larger GQA group, long context
    cfgs.append(Config("decode_hd128_bf16_softcap", "decode", "bf16", 128, 8, 2,
                       batch=8, q_len=1, k_len=1024, softcap=30.0))
    cfgs.append(Config("decode_hd128_bf16_swa", "decode", "bf16", 128, 8, 2,
                       batch=8, q_len=1, k_len=2048, window_left=1024, window_right=0))
    cfgs.append(Config("decode_hd128_bf16_split", "decode", "bf16", 128, 8, 2,
                       batch=2, q_len=1, k_len=8192, num_splits=4, weight=_HOT))
    cfgs.append(Config("decode_hd128_bf16_mqa", "decode", "bf16", 128, 16, 1,
                       batch=8, q_len=1, k_len=1024))
    cfgs.append(Config("decode_hd128_bf16_gqa16", "decode", "bf16", 128, 16, 2,
                       batch=8, q_len=1, k_len=1024))
    cfgs.append(Config("prefill_hd128_bf16_noncausal", "prefill", "bf16", 128, 8, 2,
                       paged=True, causal=False, batch=2, q_len=512, k_len=512))
    return cfgs

FULL_CONFIGS = _full_configs()


def get_configs(which: str) -> list[Config]:
    if which == "dev":
        return DEV_CONFIGS
    if which == "full":
        return FULL_CONFIGS
    raise ValueError(f"unknown config set {which!r} (use 'dev' or 'full')")

# --------------------------------------------------------------------------------------
# Tolerances.  Primary correctness criterion is the "fp32-reference 2x-error" method
# (candidate error vs fp32 ref must be <= ERR_MULT * baseline error vs fp32 ref).  The
# allclose tolerances below are a coarse secondary gate.  Final promotion uses the
# semantic 100-token greedy-decode equality check (decode_equiv.py).
# --------------------------------------------------------------------------------------
ERR_MULT = 2.0
# Calibration study (Experiment A) measured the baseline's OWN self-nondeterminism envelope
# (split-K reduction order): bf16 out max-abs<=3.9e-3 / mean-abs<=3.7e-5, fp16<=1.2e-4,
# lse(fp32)<=2e-6.  IMPORTANT: that is how much the baseline differs from ITSELF; a candidate
# is a DIFFERENT algorithm and legitimately differs from the baseline by more even when correct
# (e.g. the known-correct seed has bf16 bias ~1.4e-4 vs baseline self-bias 3.7e-5).  So these
# SYNTHETIC tolerances are a loose NECESSARY pre-filter (reject NaN/shape/order-of-magnitude
# drift) -- NOT a decode-equivalence proof.  Faithfulness comes from verify/insitu.py (live
# tensors) + decode_equiv (greedy tokens).  Tightened ~2.5x vs the original loose defaults,
# with headroom over a known-correct candidate.  atol-dominated (rtol useless near zero).
ALLCLOSE_TOL = {  # (atol, rtol) per dtype, for out
    "bf16": (8e-3, 1e-2),
    "fp16": (1e-3, 1e-2),
    "fp8e4m3": (1e-1, 5e-2),   # NOT calibrated (fp8 excluded from the study) -> kept loose
}
# Mean-abs (bias) gate on out: catches systematic synthetic drift while admitting correct
# alternative algorithms (seed bias ~1.4e-4 < these).  Live-layout drift is caught by insitu.
MEAN_ABS_TOL = {"bf16": 5e-4, "fp16": 1e-4, "fp8e4m3": 1e-2}
LSE_TOL = (5e-3, 1e-2)  # softmax_lse is fp32 and NOT used for token selection -> secondary gate

# Perf gate: candidate must be <= PERF_TARGET_RATIO x baseline latency on ALL configs,
# i.e. min over configs of (baseline_latency / candidate_latency) >= 1/PERF_TARGET_RATIO.
PERF_TARGET_RATIO = 1.0  # "match or exceed" (spec: at least 1.0x)
# A small measurement-noise band: treat <= this as a tie when promoting.
PERF_NOISE_BAND = 0.02

# --------------------------------------------------------------------------------------
# Performance-measurement knobs (CUDA events; clocks unlockable -> warm + ratio).
# --------------------------------------------------------------------------------------
PERF = dict(
    warmup_iters=300,      # warm to steady boost clocks
    timed_iters=100,
    num_trials=5,
    l2_flush_mb=128,       # > 60MiB L2 on H200
    report=("median", "p90"),
)

# --------------------------------------------------------------------------------------
# NCU metric subset (Hopper).  ~24 CUDAForge-style + FA3-specific additions.  Names are
# verified at runtime via ncu_report safe()/metric_or_none() (B200/CUDAForge names drift).
# --------------------------------------------------------------------------------------
NCU_METRIC_SUBSET = [
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks", "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem", "launch__registers_per_thread",
    "launch__waves_per_multiprocessor", "launch__shared_mem_per_block",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",  # wgmma activity (key)
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    # register spill (VERY BAD if > 0)
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
]

# Env-fatal error signatures: classify as unfixable-by-agent -> abort early, don't burn
# Claude rounds (syfi `is_env_fatal`).
ENV_FATAL_SIGNATURES = [
    "no CUDA-capable device", "CUDA driver version is insufficient",
    "ABI", "undefined symbol", "version `GLIBCXX", "out of memory",
    "ERR_NVGPUCTRPERM", "could not select device driver",
]
