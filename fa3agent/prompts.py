"""Prompt builders for the Claude Code headless brain.

The brain runs as `claude -p` in cwd=<repo root> (config.REPO_ROOT) with full tools, so it
can read reference/, .research/, the seed, and CALL the agent's components (knowledge.query,
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

GROUNDING = """\
GROUND YOUR DESIGN IN THE DOCS, NOT MEMORY. The reference docs are authoritative for THIS GPU
(sm_90a / Hopper H200); your recollection is not. Before you rely on any PTX instruction, wgmma
shape, TMA / cp.async.bulk.tensor mode, shared-memory matrix-descriptor or swizzle encoding, or
occupancy / shared-memory limit, CONFIRM it against the sm_90a docs — do NOT assume a feature,
shape, or layout is valid here just because it exists on another architecture or you recall it.
When changing the kernel, FIRST read the relevant sections: docs.retrieve('<keywords>') to find
where a topic lives, then docs.get_toc('<file>.md') for the exact heading and docs.get_section(
'<file>.md', '<heading>') to read a dense table (e.g. the wgmma descriptor / swizzle layout) in
full. If a design choice hinges on a hardware detail, name the doc section it came from.
"""

TOOLS = """\
TOOLS YOU SHOULD USE (run from your current working directory, the repo root; they return
DISTILLED info, not dumps):
- Knowledge base (typed: ks=kernel, pf=profiling, alg=algorithmic):
    python -c "from fa3agent.knowledge.store import KnowledgeStore as K; s=K(); s.seed(); [print(r) for r in s.query('<your query>')]"
- Reference-lib primitive hints (pure-PTX models for wgmma/TMA/mbarrier/softmax):
    python -c "from fa3agent.reflib.hints import get_hints; print(get_hints('wgmma bf16'))"
- NVIDIA docs (reference/{cuda-docs,design-guides,architecture-specs}; incl. the FULL PTX ISA
  parallel-thread-execution.md, the CUDA C programming guide, ncu/nsys, cutlass, the gemm /
  kernel-patterns design guides, and the Hopper-H200 spec). THREE ways to read them, all DISTILLED:
    retrieve('<keywords>', k=5, scope='<file>.md'?)  -> top ranked section snippets (find WHERE a topic lives)
    get_toc('<file>.md')                             -> that doc's headings + line numbers (see its structure)
    get_section('<file>.md', '<exact heading>')      -> ONE whole section incl. subsections, UNTRUNCATED
  Use it for exact PTX syntax/layout (wgmma matrix descriptors & swizzle, mma shapes,
  cp.async.bulk.tensor/TMA), occupancy / dynamic-SMEM rules, and NCU metric meanings. Typical flow:
  retrieve to locate -> get_toc to read off the exact heading -> get_section to read a dense table in full.
    python -c "from fa3agent.docs.retrieve import retrieve, get_toc, get_section; print(get_section('parallel-thread-execution.md', 'Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions'))"
- A correct but slow REFERENCE implementation is at fa3agent/seed_kernel.cu (one warp per
  (query,head), online softmax, fp32 accumulate). Read it to understand the host/registration
  boilerplate and the exact tensor layouts/strides.
- SASS oracle / analysis (after a build): fa3agent.sass.analyze.analyze, fa3agent.sass.cutlass_oracle.build_oracle.
- Sanitizer (after you build a .so to test): a BOUNDED compute-sanitizer wrapper — prefer it over raw
  compute-sanitizer (returns {clean, num_errors, errors[<=5]}, not the raw dump):
    python -c "from fa3agent.verify.sanitizer import run_sanitizer; from fa3agent import config as C; print(run_sanitizer('<built.so>', '<namespace>', C.DEV_CONFIGS[2], tool='memcheck'))"
"""


def workflow_block(candidate_path: str, bootstrap: bool = False) -> str:
    """The closing instructions: either the in-loop TRIAL workflow (C.TRIAL_IN_LOOP, default) or the
    legacy write-one-blind-candidate-then-STOP contract.  In-loop, the brain compiles + tests +
    benchmarks the candidate ITSELF and iterates before committing -- the AVO/K-Search local-
    refinement step -- which is what lets it land a risky structural rewrite (build it, see it
    regress one config, fix it) instead of fearing a blind commit that could lose the champion."""
    if not C.TRIAL_IN_LOOP:
        legacy_stop = ("Then STOP. The driver will build + anti-cheat + verify correctness "
                       + ("" if bootstrap else "+ profile ") + "and report back.")
        return (f"WRITE the candidate to EXACTLY this path (overwrite):\n  {candidate_path}\n"
                f"{legacy_stop}\n"
                "Do NOT run the build or the eval yourself — the driver does it next round.")
    k = C.TRIAL_STAGNATION_LIMIT
    goal = ("BUILDS and is CORRECT on every config (this is the bootstrap goal — do not chase speed yet)"
            if bootstrap else
            "BUILDS, is CORRECT on every config, AND raises the MINIMUM speedup ratio over the champion")
    return f"""\
WORKFLOW — ITERATE, do NOT write blind. You have the build + evaluator as a tool this round; use it.
1. Edit the candidate at {candidate_path}.
2. TRIAL it yourself (the SAME build/correctness/perf pipeline the driver uses):
     python -m fa3agent.trial {candidate_path}
   It prints a one-line summary + a `TRIAL_JSON: {{...}}` verdict: whether it built (else the compile
   error), whether every config is correct (else which configs failed + why), the per-config + min /
   geomean speedup ratios vs the FA3 baseline, and the SASS instruction mix (wgmma/TMA present?
   register spills?). Flags: `--no-perf` (build+correctness only — fast, use while fixing a
   compile/correctness bug), `--profile` (also attach the NCU bottleneck for the worst config),
   `--configs a,b` (restrict to a subset for faster iteration). Each call is isolated (its own build
   + namespace); run it as many times as useful.
3. READ the verdict and DIAGNOSE. If it did not build, is incorrect on any config, or did not improve
   the min ratio, go back to 1 and fix the specific problem the verdict points to (use the docs/SASS
   tools above to ground the fix).
4. Repeat until the candidate {goal} — or until ~{k} consecutive trials yield no improvement, in
   which case keep your BEST correct version and stop (don't burn the whole budget on a dead end;
   a correct kernel that ties the champion beats one that is broken or doesn't build). For a risky
   structural rewrite, trial early and often so a regression on one config is something you FIX, not
   a reason to abandon the rewrite.

The file you leave at {candidate_path} is what gets committed; the driver then re-runs the
AUTHORITATIVE eval (correctness + perf + the faithful vLLM gates) on it. STOP once the candidate is
correct and you've improved it as far as this round's trials show (or you've stagnated)."""


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

{GROUNDING}

{TOOLS}

{workflow_block(candidate_path, bootstrap=True)}
"""


def refine_prompt(candidate_path: str, best_path: str, last_eval_summary: str,
                  profile_feedback: str, knowledge_hints: str, focus: str,
                  supervisor_note: str = "", doc_hints: str = "",
                  sass_feedback: str = "", knowledge_notes: str = "",
                  focus_docs: str = "", profile_digest: str = "",
                  memory_digest: str = "", structural_carveout: str = "",
                  branch_note: str = "") -> str:
    return f"""\
You are optimizing a Hopper FlashAttention-3 kernel (single self-contained .cu). Continue from
the current CHAMPION at {best_path} (read it). Make TARGETED edits by default — BUT a structural
rewrite is the RIGHT move when the profiler bottleneck calls for one (e.g. an overlap-bound /
off-tensor-core / latency-bound verdict justifies warp-specialization + TMA over a one-line tweak).
{structural_carveout or ""}

PRIORITY: if the last candidate was INCORRECT, fix correctness first (speed is secondary). If it
was correct, raise the MINIMUM speedup ratio across ALL configs (not just the mean) while keeping
every config correct. The perf gate is candidate latency <= baseline on every config.

RUN MEMORY (cross-step continuity — this session is FRESH and has no chat history; this is what
earlier steps learned. Honor the dead-ends: do NOT re-attempt something listed as already-tried-
and-failed the same way. Open {best_path}'s sibling MEMORY.md for the full log):
{memory_digest or "(empty — first steps of the run)"}
{("REWRITE-BRANCH JOURNAL (the half-built rewrite from earlier branch rungs — CONTINUE it, do not restart):" + chr(10) + branch_note) if branch_note else ""}

LAST EVAL: {last_eval_summary}

PROFILER FEEDBACK (distilled, NCU): {profile_feedback or "(none yet)"}

NCU DIAGNOSIS (full distilled signal — the classifier's bottleneck class, the recommended
optimization + plan, and the evidence-cited ranked priorities incl. per-kernel time split). Treat
this as the primary STEER for what to change this step:
{profile_digest or "(none yet — rely on the one-line PROFILER FEEDBACK above)"}

SASS (your kernel's instruction mix — wgmma/TMA presence, register spills): {sass_feedback or "(none)"}

CORE TECHNIQUE REFERENCE — authoritative sm_90a docs for THIS step's focus (read these first; they
are curated to the suggested focus below):
{focus_docs or "(none mapped for this focus — use docs.get_toc/get_section on kernel-patterns.md or hopper-h200.md)"}

RELEVANT NVIDIA DOCS (auto-retrieved per scope for the above bottleneck; query docs.retrieve for more, or open the file for a full section):
{doc_hints or "(none — query docs.retrieve yourself for exact PTX syntax/layout, e.g. with scope='parallel-thread-execution.md')"}

REFERENCE-LIB HINTS (pure-PTX models for wgmma/TMA/mbarrier/softmax): {knowledge_hints or "(query reflib.get_hints yourself)"}

LEARNED NOTES (distilled from THIS run's knowledge base; query it for more):
{knowledge_notes or "(none yet — query the knowledge base yourself)"}

SUGGESTED FOCUS: {focus}
{("SUPERVISOR: " + supervisor_note) if supervisor_note else ""}

{CONTRACT}

{GROUNDING}

{TOOLS}

{workflow_block(candidate_path, bootstrap=False)}
"""


def completion_followup(candidate_path: str, focus: str, technique_label: str,
                        gaps: list, ptx_ref: str = "") -> str:
    """In-step COMPLETION GATE follow-up: the brain just wrote `candidate_path` to INTRODUCE a new
    functionality, but the built kernel did not fully/correctly land it.  Hand it the precise gaps
    and tell it to FINISH the same file in place (not restart, not revert).  Used as a FRESH
    session by default, so it must RE-READ the candidate -- the file on disk is the continuity.

    `ptx_ref` (optional): a literal Hopper PTX sequence from the out-of-band oracle (CUTLASS / inline
    wgmma+TMA reference) for the exact instruction class that is still missing -- injected only when
    the gate proved the class is absent, so the brain sees the concrete forms it must emit."""
    gap_block = "\n".join(f"  - {g}" for g in (gaps or [])) or "  - (unspecified)"
    ref_block = ""
    if ptx_ref:
        ref_block = f"""
REFERENCE — a WORKING Hopper kernel issuing {technique_label}, compiled out-of-band and disassembled
to PTX. This is the exact instruction SEQUENCE (and ORDER) an ideal kernel emits. RE-DERIVE it in
YOUR kernel (you may write this PTX via inline asm or the matching CUDA intrinsics) — do NOT #include
cutlass/cute, and do NOT copy it verbatim into a dead path; the REAL, taken code path must issue it:
```ptx
{ptx_ref}
```
Mirror these forms: the mbarrier.init + mbarrier.arrive.expect_tx, the cp.async.bulk.tensor TMA load
keyed off a CUtensorMap, the wgmma.fence.sync.aligned, the wgmma.mma_async.sync.aligned.m64nNk16
MMA over shared-memory descriptors, then wgmma.commit_group + wgmma.wait_group. After you finish,
SASS must show the instruction class present.
"""
    return f"""\
You just wrote a Hopper FlashAttention-3 candidate to {candidate_path} for this step's goal:
  {focus}

The driver built and checked it, and the change is INCOMPLETE — it did not fully/correctly land
the intended functionality. Concrete gaps found in the BUILT kernel:
{gap_block}
{ref_block}
FINISH IT IN PLACE — do NOT start over and do NOT revert to the previous approach:
1. READ the current {candidate_path} (it is the file you just wrote) so you continue YOUR work.
2. Complete the implementation so that {technique_label} is ACTUALLY present in the compiled
   kernel — not stubbed, not guarded out behind a disabled #if/branch, not left on the scalar
   fallback — AND every config stays correct. If the build failed, fix the compile error; if a
   config is numerically wrong, fix it; if the instruction class is absent from SASS, make the
   REAL (taken) code path use it.
3. {("TRIAL it yourself before you finish: `python -m fa3agent.trial " + candidate_path + "` — confirm it BUILDS, is CORRECT on every config, and that the instruction class now shows up in the SASS verdict. Iterate until it does." ) if C.TRIAL_IN_LOOP else "You may build the .so and inspect SASS yourself (fa3agent.sass.analyze.analyze) to CONFIRM the instruction class is present before you finish — optional but encouraged."}

{CONTRACT}

{GROUNDING}

{TOOLS}

{workflow_block(candidate_path, bootstrap=False)}
"""


# D1 rung-0: a behavior-preserving refactor (no new features, no perf chase) that turns the
# encrusted seed into a clean, extensible substrate for the structural rungs that follow.
BRANCH_REFACTOR_FOCUS = (
    "RUNG 0 — REFACTOR ONLY (behavior-preserving). Do NOT add features or chase speed this rung. "
    "Restructure the kernel into ONE clean, extensible code path: collapse duplicated/special-case "
    "branches, name the tiling/pipeline parameters, and separate the load / GEMM / softmax / "
    "epilogue stages into clearly delineated sections so the next rungs can warp-specialize and add "
    "a TMA pipeline without fighting the layout. Every config MUST stay correct (this is the gate). "
    "If you find the layout fundamentally blocks separating producer/consumer warpgroups without "
    "spilling O (the known register-pressure wall), SAY SO explicitly in your summary so the driver "
    "can escalate to a from-scratch seed.")

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

# Curated doc sections per FOCUS_LEVER (SAME index): the authoritative sm_90a reference for the
# exact technique that step is pushing.  The orchestrator get_section()s these and injects them so
# the brain always has the on-target pattern in front of it -- independent of whether the lexical
# retriever happened to surface it.  Headings are matched leniently (section numbers optional), so
# these must stay in sync with reference/design-guides/kernel-patterns.md +
# reference/architecture-specs/hopper-h200.md (the problem->technique mapping mirrors KernelWiki's
# queries/by-technique taxonomy).  Keep <=2 sections per lever to bound prompt size.
FOCUS_DOC_SECTIONS = [
    [("kernel-patterns.md", "Async execution model"),
     ("hopper-h200.md", "Tensor Cores (4th Generation)")],
    [("kernel-patterns.md", "Warp specialization"),
     ("kernel-patterns.md", "Software pipelining (double buffering)")],
    [("kernel-patterns.md", "Reduction & online softmax")],
    [("kernel-patterns.md", "Occupancy tuning"),
     ("kernel-patterns.md", "Persistent kernels & scheduling")],
    [("kernel-patterns.md", "Shared memory swizzling"),
     ("kernel-patterns.md", "GEMM tiling")],
]
