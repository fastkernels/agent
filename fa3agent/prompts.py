"""Prompt builders for the Claude Code headless brain.

The brain runs as `claude -p` in cwd=/data-fast/agent with full tools, so it can read
reference/, .research/, the seed, and CALL the agent's components (knowledge.query,
docs.retrieve, reflib.hints, sass tools) itself.  We hand it DISTILLED feedback (not dumps)
and tell it to WRITE the candidate .cu to an exact path; the DRIVER runs the eval.
"""
from __future__ import annotations

from . import config as C

CONTRACT = f"""\
HARD CONTRACT (a candidate that violates any of these is rejected by the driver):
- Output: ONE self-contained CUDA .cu file. NO external libraries — forbidden: CUTLASS/CuTe,
  cuBLAS/cuBLASLt, cuDNN, FlashInfer, ThunderKittens headers, Triton, the original _vllm_fa3_C,
  and any at::scaled_dot_product_attention / at::native attention path. You may re-derive raw
  PTX (wgmma.mma_async, cp.async.bulk.tensor/TMA, mbarrier) yourself — that is allowed and is the
  intended path to FA3-class performance.
- torch may appear ONLY in the host mha_fwd function (tensor unpack, output alloc) and the
  TORCH_LIBRARY registration. NEVER inside any __global__ kernel.
- Register ops under namespace FA3_LIB_NAME (a -D macro the build sets). Use:
    #ifndef FA3_LIB_NAME
    #define FA3_LIB_NAME _vllm_fa3_C
    #endif
    TORCH_LIBRARY(FA3_LIB_NAME, m) {{ ... }}
- Register BOTH ops with these EXACT schemas (positional call from vLLM):
    fwd:  {C.FWD_SCHEMA}
    get_scheduler_metadata: {C.GET_SCHED_SCHEMA}
- fwd returns a 4-element Tensor[]: [out, softmax_lse, out_accum, softmax_lse_accum]
  (last two may be empty when num_splits==1). softmax_lse is fp32, shape (num_heads_q, total_q).
  out dtype = input dtype, except fp8-e4m3 input -> bf16 out. num_splits and scheduler_metadata
  are perf/advisory; a single-pass kernel that ignores them is always correct.
- Target sm_90a (Hopper H200). Build is toolkit-only (no cutlass include path).
"""

TOOLS = """\
TOOLS YOU SHOULD USE (run from /data-fast/agent; they return DISTILLED info, not dumps):
- Knowledge base (typed: ks=kernel, pf=profiling, alg=algorithmic):
    python -c "from fa3agent.knowledge.store import KnowledgeStore as K; s=K(); s.seed(); [print(r) for r in s.query('<your query>')]"
- Reference-lib primitive hints (pure-PTX models for wgmma/TMA/mbarrier/softmax):
    python -c "from fa3agent.reflib.hints import get_hints; print(get_hints('wgmma bf16'))"
- NVIDIA docs retrieval:
    python -c "from fa3agent.docs.retrieve import retrieve; print(retrieve('cp.async.bulk.tensor mbarrier', 3))"
- A correct but slow REFERENCE implementation is at fa3agent/seed_kernel.cu (one warp per
  (query,head), online softmax, fp32 accumulate). Read it to understand the host/registration
  boilerplate and the exact tensor layouts/strides.
- SASS oracle / analysis (after a build): fa3agent.sass.analyze.analyze, fa3agent.sass.cutlass_oracle.build_oracle.
Do NOT run the build or the eval yourself — the driver compiles, anti-cheat-scans, checks
CORRECTNESS FIRST, then profiles, and feeds you distilled results next round.
"""


def bootstrap_prompt(candidate_path: str, config_names: list[str]) -> str:
    return f"""\
You are optimizing a Hopper FlashAttention-3 kernel by distilling vLLM's FA3 into a single
self-contained .cu. This is STEP 0 (bootstrap). CORRECTNESS COMES FIRST.

GOAL THIS STEP: produce a CORRECT candidate that builds and matches the baseline
(torch.ops._vllm_fa3_C.fwd) within tolerance on these configs: {", ".join(config_names)}.
The simplest path: start from the known-correct reference fa3agent/seed_kernel.cu (you may copy
it to the output path and confirm it is complete), OR write your own simple correct kernel. Do
NOT chase performance yet — a correct slow kernel is the required foundation.

{CONTRACT}

{TOOLS}

WRITE your candidate to EXACTLY this path (overwrite if present):
  {candidate_path}
Then STOP. The driver will build + anti-cheat-scan + verify correctness and report back.
"""


def refine_prompt(candidate_path: str, best_path: str, last_eval_summary: str,
                  profile_feedback: str, knowledge_hints: str, focus: str,
                  supervisor_note: str = "") -> str:
    return f"""\
You are optimizing a Hopper FlashAttention-3 kernel (single self-contained .cu). Continue from
the current CHAMPION at {best_path} (read it). Make TARGETED edits unless you have a concrete
reason for a rewrite.

PRIORITY: if the last candidate was INCORRECT, fix correctness first (speed is secondary). If it
was correct, raise the MINIMUM speedup ratio across ALL configs (not just the mean) while keeping
every config correct. The perf gate is candidate latency <= baseline on every config.

LAST EVAL: {last_eval_summary}

PROFILER FEEDBACK (distilled, NCU): {profile_feedback or "(none yet)"}

KNOWLEDGE / HINTS (from the typed knowledge base + reference libs): {knowledge_hints or "(query the knowledge base yourself)"}

SUGGESTED FOCUS: {focus}
{("SUPERVISOR: " + supervisor_note) if supervisor_note else ""}

{CONTRACT}

{TOOLS}

WRITE the improved candidate to EXACTLY this path (overwrite):
  {candidate_path}
Then STOP. The driver will build + anti-cheat + verify correctness + profile and report back.
"""


# Focus levers cycled by the supervisor when stuck (from AVO inflection techniques + Hopper levers)
FOCUS_LEVERS = [
    "Use wgmma.mma_async (m64nNk16 bf16, N>=64) for the QK^T and P*V GEMMs instead of scalar FFMA; "
    "load Q/K/V tiles to shared memory with cp.async.bulk.tensor (TMA) + mbarrier producer/consumer.",
    "Warp-specialize: a producer warpgroup issues TMA loads, consumer warpgroups run wgmma; "
    "use setmaxnreg to rebalance registers; named barriers to order the two math warpgroups.",
    "Online softmax in log2 domain (exp2, fold log2(e) into the scale); branchless predicated "
    "rescaling on fully-unmasked K-blocks; overlap the correction with the P*V GEMM pipeline.",
    "For decode (small M), pack heads/tokens into the wgmma M dimension or use split-KV across the "
    "KV sequence with an LSE-weighted combine to raise occupancy.",
    "Tune tile sizes (block_m/block_n), pipeline stages (2-4), and SMEM swizzle for the head dim; "
    "request >48KB dynamic SMEM via cudaFuncSetAttribute.",
]
