# AVO — Agentic Variation Operators (NVIDIA, NeurIPS 2026 preprint) — Distilled Notes

Source: `/data-fast/agent/reference/related-works/nvidia-avo/` (main.tex, sections/01-08, appendix-01, references.bib, figures/).
Authored by NVIDIA (Terry Chen, Zhifan Ye, Bing Xu, Zihao Ye, Tianqi Chen, et al.). Target HW = **Blackwell B200**, CUDA 13.1, PyTorch 2.10.0. We are on **Hopper H200 / sm_90a**, CUDA 13.0, torch 2.11 — architecture differs (no 2-SM tcgen05, different register-file specifics) but the *agent-loop methodology* transfers directly.

================================================================
## 1. Headline result & what they built
================================================================
- AVO = replace classical evolutionary-search variation operators (mutation/crossover/heuristics) with a **self-directed coding agent** that *is* the variation operator. The agent subsumes Sample + Generate + Evaluate into ONE autonomous loop.
- Applied to forward MHA attention kernels on B200. Over **7 days continuous, no human intervention**, agent explored **>500 optimization directions**, committed **40 kernel versions**, reached **1668 TFLOPS BF16**, beating cuDNN by up to +3.5% and FlashAttention-4 (FA4) by up to +10.5% (causal). Final committed v40 geomean = 1520 TFLOPS (causal) vs cuDNN 1488, FA4 1426.
- Discovered MHA optimizations **transferred to GQA in ~30 min** of additional autonomous agent effort: +7.0% over cuDNN, +9.3% over FA4 (causal GQA).
- Key thesis: prior LLM-in-the-loop (FunSearch, AlphaEvolve, LoongFlow, TTT-Discover) confine the LLM to a single-turn `Generate` inside a *fixed pipeline*; the framework owns sampling/eval/population. AVO **elevates the agent to own the whole loop** — it decides what to consult, what to edit, when to test, and how to revise.

================================================================
## 2. Formal framework (maps directly to our agent design)
================================================================
Evolutionary search: population `P = {(x_i, f(x_i))}`, iterate `P_{t+1} = Update(P_t, (x_{t+1}, f(x_{t+1})))`, `x_{t+1} = Vary(P_t)`.

- Classical / prior LLM: `Vary(P_t) = Generate(Sample(P_t))` — LLM only in Generate; Sample is a fixed algorithm (AlphaEvolve = island MAP-Elites + fitness/diversity prompt sampler; LoongFlow = MAP-Elites + Boltzmann selection + fixed Plan-Execute-Summarize; TTT-Discover = PUCT selection + test-time gradient updates to the policy).
- **AVO**: `Vary(P_t) = Agent(P_t, K, f)` where:
  - `P_t` = the **full lineage** of solutions + scores (each x_i = CUDA source w/ inline PTX).
  - `K` = **domain-specific knowledge base**: CUDA programming guides, **PTX ISA docs**, **Blackwell architecture specs**, and **existing kernel implementations including FA4 source**.
  - `f` = scoring function = **vector** `f(x) = (f_1,...,f_n)`, one `f_j` per test configuration. Each `f_j` = throughput (TFLOPS) on target HW. **Correctness gate: if a candidate fails correctness vs reference, ALL `f_j = 0` regardless of speed.** (Correctness FIRST, then perf — exactly our constraint.)
- Population structure is orthogonal; this paper deliberately uses a **single-lineage** continuous run (seed x_0 -> x_1 -> ... -> x_t) "to isolate the effect of the operator itself." Archive/island/branching left to future work.

REUSABLE: Our scoring vector should mirror this — one score per vLLM config (prefill/decode shapes, causal/non-causal, seq lengths, batch). Correctness failure => zero across the board. Track geomean of per-config TFLOPS as the running-best selection metric.

================================================================
## 3. Anatomy of a single variation step (the inner agent loop)
================================================================
A single step = an autonomous agent loop producing x_{t+1} from P_t, may involve MANY internal actions:
1. **Examine multiple prior implementations in P_t** within one step — compare their *profiling characteristics* to identify bottlenecks/opportunities.
2. **Consult docs in K** to understand the relevant hardware constraints BEFORE implementing.
3. Implement a candidate optimization.
4. **Invoke f to test** (correctness + throughput).
5. On failure (correctness OR no improvement on the benchmark suite): **diagnose -> revise -> repeat** the edit-evaluate-diagnose cycle.
6. **Commit x_{t+1} only when it passes correctness AND matches-or-improves the best committed score so far.** Unsuccessful intermediate attempts stay in the agent's internal trajectory but are NOT added to committed lineage.

Strategy shift over time (emergent, not scripted):
- Early steps: structural changes informed by reference implementations in K.
- Late steps: micro-architectural tuning guided by profiling feedback from f + patterns across accumulated lineage P_t.

Agent characteristics (Section 4 setup):
- "Internally-developed general-purpose coding agent powered by frontier LLMs." **No task-specific modifications** — same agent used for general SWE. Only K and f are injected.
- Tools: autonomous code editing, shell command execution, filesystem navigation, documentation retrieval.
- **Persistent memory = its conversation history**, accumulating full context of prior edits, compiler outputs, profiling results, and reasoning across the evolutionary process.

REUSABLE: We replicate this with Claude Code headless sub-agents. Our richer toolset (NCU/Nsys parser 2.1, compute-sanitizer 2.2, SASS diff 2.3, docs nav 2.4, anti-cheat 2.5, CUTLASS/CuTe inspiration 2.6, ref-lib hints 2.7) is exactly the "standard SWE tools + doc retrieval" they describe, specialized. Key: keep the agent free to decide *when* to invoke each, do not force a rigid pipeline.

================================================================
## 4. Continuous evolution + self-supervision (anti-stall) — IMPORTANT
================================================================
- Each committed x_i persisted as a **git commit annotated with its score** -> full state continuity across the entire run. (This is our checkpoint+resume mechanism, for free.)
- Two failure modes in long-horizon autonomy:
  1. **Stall** — agent exhausts its current line of exploration.
  2. **Unproductive cycles** — repeated edits that keep failing to improve scores.
- **Self-supervision mechanism** (a supervisor): detects these scenarios and intervenes. When triggered, it **reviews the overall evolutionary trajectory** and **steers toward several candidate optimization directions** — a conditional intervention that redirects exploration with fresh perspective when the current strategy plateaus.
- In the 7-day / 40-version run: the main agent autonomously decided when to try new opts, when to revisit earlier P_t approaches, and when to shift strategy; the supervisor maintained forward progress by intervening during stagnation.

REUSABLE (directly maps to our "learn" stage + in-loop distillation):
- Implement a **supervisor/watchdog** that monitors the commit stream: if N consecutive candidates fail to beat best, or M wall-clock with no commit, trigger a "strategy review" that summarizes the trajectory and proposes 3-5 fresh directions. This is our anti-stall + knowledge-injection hook.
- Use git commits (one per committed version, with score in commit msg/notes) as the canonical checkpoint format -> trivially resumable.

================================================================
## 5. Benchmark / scoring setup (how they measure perf reliably)
================================================================
- Forward **prefilling** throughput, head dim **128**, **BF16**, seq lengths {4096, 8192, 16384, 32768}.
- **Fixed total tokens = 32768**, achieved by adjusting batch size per seq len (bs=8 @ 4096, ..., bs=1 @ 32768). Keeps total FLOPs ~constant across configs.
- MHA = 16 heads, both causal and non-causal.
- GQA configs from Qwen3: 32 Q heads / 4 KV (group 8, Qwen3-30B-A3B); 32 Q / 8 KV (group 4, Qwen3-8B).
- **Timing**: reused FA4 repo's `benchmarks/benchmark_attn.py`, same warmup/repeat rounds as FA4 paper, PLUS **ran each experiment 10x to report mean +/- std**. Same setup used both for agent evolution AND final benchmarking — no train/test mismatch in the harness.
- Appendix: also compared vs cuDNN/FA4 numbers *as published in FA4 paper* to control for driver/thermal/clock variance on their box (acknowledges absolute TFLOPS is system-sensitive). On those reported baselines: non-causal +1.4% to +3.4% over cuDNN, +2.3% to +3.9% over FA4; causal +3.6% to +7.5% over cuDNN, +3.7% to +8.8% over FA4 (largest at short seq bs=8 seq=4096).

REUSABLE:
- Our latency measurement (CUDA events) should likewise do warmup + many repeats + report mean/std; run a config sweep that matches vLLM's actual prefill+decode shapes. Use geomean across configs as the single fitness scalar for "running best."
- Be aware absolute TFLOPS/latency is sensitive to clocks/thermals -> lock clocks if possible, and/or always re-measure baseline on the SAME box in the SAME run (we already do: baseline = prebuilt torch.ops._vllm_fa3_C.fwd). Compare candidate vs freshly-measured baseline, not a cached number.

================================================================
## 6. Evolution plot — exact construction (to replicate our evolution plot)
================================================================
From `figures/avo_mha_causal_neurips.pdf` (and the non-causal twin `avo_mha_noncausal_neurips.pdf`). Title: "Evolution Trajectory of AVO on Causal BF16 Multi-head Attention".
- **X axis** = "Kernel Version" — discrete v1..v40 (the *committed* lineage only, NOT the >500 internal attempts).
- **Y axis** = "TFLOPS", range ~1100-1600 (ticks at 1100,1200,...,1600).
- **Solid green line** = **running-best geomean throughput** across all 4 configs (legend: "Running best (Geomean)").
- **Green circles** = versions that set a NEW best (legend: "New best (Geomean)").
- **Dashed colored lines** = per-configuration throughput, one per config: seq_len = 4k, 8k, 16k, 32k.
- **Horizontal dashed reference lines** = geomean throughput of **cuDNN (1488)** and **FA4 (1426)** baselines.
- Annotated final point: "AVO v40 (1520 TFLOPs)".

Narrative the plot must show (from Section 4 trajectory analysis):
- **Discrete jumps, not gradual** — big steps separated by plateaus of micro-refinement.
- Five largest gains = architectural inflection points (label these on the plot):
  - **v8**: QK-PV interleaving with **bitmask causal masking**.
  - **v13**: restructured **single-pass softmax**.
  - **v20**: **branchless accumulator rescaling** + lighter memory fence for unmasked iterations.
  - **v30**: **correction/MMA pipeline overlap**.
  - **v33**: **register rebalancing across warp groups**.
- **Diminishing returns**: v1-v20 = largest absolute gains (naive->competitive); v21-v40 = smaller compounding gains (cycle-level scheduling, resource allocation).

REUSABLE: Replicate exactly — X=committed version index, Y=TFLOPS (or latency, but they use TFLOPS), solid line = running-best geomean with markers on new-best, dashed per-config lines, horizontal dashed baseline line(s) (for us: the prebuilt _vllm_fa3_C baseline geomean). Annotate inflection commits with the optimization name. Track per step: version id, commit hash, per-config score vector, geomean, is_new_best, the optimization description, and the bottleneck it addressed.

================================================================
## 7. The 3 deep-dive optimizations (concrete attention/Hopper-relevant techniques)
================================================================
Table (geomean TFLOPS improvement, ablation v_before -> v_after):
- Branchless accumulator rescaling | v19->v20 | non-causal +8.1% | causal +1.6%
- Correction/MMA pipeline overlap  | v29->v30 | non-causal +1.1% | causal +0.4%
- Register rebalancing across warp groups | v32->v33 | non-causal +2.1% | causal ~0%

### 7.1 Branchless accumulator rescaling (largest single win)
- Bottleneck (v19): online-softmax running row-max may change per K-block => output accumulator O must be rescaled. v19 used a **conditional branch** (check if any thread in warp needs rescale, skip if max unchanged). The branch adds **warp-sync overhead every K-block iteration**, and the conditional control flow **blocks use of lighter memory fences** in the correction path.
- Fix (v20): **branchless speculative path** — always compute rescale factor, use a **predicated select substituting 1.0** when no rescale needed (an unnecessary multiply-by-1 is cheap vs sync overhead). Eliminating the branch removes warp divergence => enables swapping a **blocking memory fence** (stalls until all pending writes complete) for a **lighter non-blocking fence** (just enforces ordering). Safe because branchless => all warp threads share control flow => guaranteed reconvergence before next sync point.
- Asymmetry (+8.1% non-causal vs +1.6% causal): branchless path applies only to **fully-unmasked** K-blocks; non-causal processes all blocks unmasked, causal keeps original branched logic for masked blocks.

### 7.2 Correction/MMA pipeline overlap
- Bottleneck (v29): dual Q-stage pipeline (2 Q-tiles concurrent). Each needs PV GEMM -> output normalization by correction warp. v29 **serialized at MMA->correction boundary**: correction warp waited for BOTH PV GEMMs before normalizing either => idle during 2nd GEMM.
- Fix (v30): let correction warp **start normalizing stage-1 output as soon as its PV GEMM finishes**, overlapping with stage-2 PV GEMM. Converts sequential dependency -> pipelined execution, cuts correction-warp idle.

### 7.3 Register rebalancing across warp groups
- HW fact (Blackwell): fixed budget **2048 warp-registers per SM** partitioned across warp groups. (Hopper note: sm_90a also exposes `setmaxnreg` per-warpgroup register reallocation — same lever applies.)
- Bottleneck (v32): followed FA4 allocation = **192 regs / 8 softmax warps, 80 / 4 correction warps, 48 / remaining 4 warps**. Profiling showed **correction group spilling to local memory** (80 too few) while softmax had headroom.
- Fix (v33): move 8 regs from softmax to each other group -> **184 / 88 / 56**. Viable because AVO's softmax processes scores in **small fragments with packed arithmetic** => low peak register usage, headroom even at 184. Correction benefits because after 7.2 it is now concurrent with 2nd PV GEMM => on the critical path; 88 vs 80 regs => fewer spills => fewer stalls.

Discussion: each opt requires **jointly reasoning about multiple HW subsystems** (sync + memory ordering; pipeline scheduling; register allocation) — not isolated single-param tuning. This is the evidence of "genuine hardware-level reasoning."

================================================================
## 8. Attention/warp-specialization background (FA4-style design they evolved from)
================================================================
On Blackwell, SOTA attention (FA4) uses **warp specialization** — distinct warp-group roles in the pipeline:
- **MMA warps**: the two core GEMMs via tensor-core instructions — QK GEMM (scores S = QK^T) and PV GEMM (O += P*V where P = softmax(S)).
- **Softmax warps**: compute P from S via online softmax with running row-max.
- **Correction warps**: rescale the O accumulator when the running max changes across K-block iterations.
- **Load + epilogue warps**: data movement via **TMA** (Tensor Memory Accelerator).
- These groups run **concurrently across two Q-tiles (dual Q-stage)** with **barrier-based signaling** for handoffs.
- Causal: some K-blocks fully masked, others fully unmasked => different execution paths in the same kernel (key to the branchless-rescale asymmetry).

REUSABLE for our MHA/causal work: The five inflection optimizations are a concrete checklist of high-value moves to seed our knowledge base / suggest to the agent:
1. QK-PV interleaving + **bitmask** causal masking.
2. Single-pass softmax restructuring.
3. Branchless predicated accumulator rescaling + non-blocking fence on unmasked path.
4. Correction/MMA (PV-GEMM) pipeline overlap in dual-Q-stage designs.
5. Per-warp-group register rebalancing guided by spill profiling.
Note our baseline is **FA3 (vLLM fork)** on Hopper, not FA4 on Blackwell — Hopper uses wgmma (warpgroup MMA) + setmaxnreg + TMA + the same warp-specialized softmax/correction split, so all five concepts apply, but exact register budgets and 2-SM features differ.

================================================================
## 9. Concrete things to adopt in OUR agent (synthesis)
================================================================
- Single-lineage continuous loop, git-commit-per-version with score -> checkpoint/resume = git log replay. Start from a seed (our distilled FA3 .cu) = x_0.
- Fitness = vector of per-config TFLOPS (or 1/latency); correctness failure => all zeros; running-best = geomean; only commit on pass-correctness AND >= best-so-far.
- Inner loop = autonomous: examine-prior-versions -> consult-K-docs -> edit -> run f -> diagnose -> repeat -> commit. Do not over-script; give tools, let agent choose.
- Knowledge base K = CUDA/PTX ISA/Hopper(sm_90a) arch docs + reference kernel sources (FA3 vllm fork, CUTLASS/CuTe for inspiration). Our 2.4 (docs nav) + 2.6 (CUTLASS inspiration) + 2.7 (ref-lib hints) implement K.
- Profiling feedback (our 2.1 NCU/Nsys parser -> concise per-op feedback) is the f-side signal that drives late-stage micro-opt. The paper shows profiling directly drove v33 (spill detection) — so our parser MUST surface: register spills/local-mem traffic, warp-group occupancy/headroom, stall reasons (sync/membar), pipeline idle of specific warp roles.
- Compute-sanitizer (our 2.2) = the diagnose step for correctness failures (races/uninit/sync) before the candidate can ever score nonzero.
- Supervisor/anti-stall: detect "stall" (exhausted line) and "unproductive cycles" (repeated non-improving edits); on trigger, summarize trajectory + propose 3-5 fresh directions. This is also where in-loop distillation lands.
- Evolution plot built exactly as Section 6 above.
- Measure: warmup + many repeats + 10x mean/std; re-measure baseline each run; lock clocks; geomean over the vLLM-relevant config sweep.

================================================================
## 10. Gotchas / caveats for our port
================================================================
- HW MISMATCH: paper = B200 (Blackwell, tcgen05/2-SM, 2048 regs/SM budget described), CUDA 13.1, torch 2.10; we = H200 (Hopper sm_90a, wgmma, CUDA 13.0, torch 2.11). Register budgets, MMA instr, 2-Q-stage specifics differ. Use the *concepts*, re-derive the *numbers* via profiling on H200.
- Baseline differs: they beat cuDNN/FA4; we must MATCH-OR-EXCEED the prebuilt vLLM FA3 baseline AND preserve greedy decode equivalence (<=100 tokens). Their correctness gate is "numerical correctness vs reference" — ours is stricter (argmax-stable decode), so define f's correctness check to assert greedy-token equivalence, not just rtol/atol.
- Absolute TFLOPS noisy (they explicitly add the FA4-reported-baseline appendix to hedge). Always compare candidate vs baseline measured in the SAME session; never trust a stored absolute.
- They committed only on >= best; intermediate fails discarded from lineage but kept in agent context. Do not lose the *learning* from failures — that is why persistent conversation memory matters; we should distill failed-attempt lessons into typed knowledge (kernel-specific/profiling/algorithmic) rather than discard.
- Single-lineage was a deliberate simplification to "isolate the operator"; real gains may come faster with islands/branching — but start single-lineage to de-risk.
- The >500 vs 40 ratio (~12:1 explore:commit) sets expectations: most candidate directions fail/regress. Budget compute accordingly; multi-GPU parallelism (our 1 kernel/GPU for perf) is how we afford the 12:1 search.
- "No task-specific agent mods" — they used a stock SWE agent + K + f. Resist over-engineering the agent; invest in K quality, f reliability, and the profiling-feedback parser.

================================================================
## 11. Figure inventory (what each PDF shows)
================================================================
- `teaser.pdf`: EVO vs AVO. Left = prior fixed pipeline (LLM = single-turn Generate; framework owns Sample/Evaluate/population). Right = AVO autonomous agent that iteratively plans/implements/tests/debugs across long sessions with access to previous solutions, evaluation utilities, tools, persistent memory. (Custom-font, labels not text-extractable but described in intro caption.)
- `overview.pdf`: the AVO operator diagram (Section 3). Shows Vary(P_t)=Agent(P_t, K, f): agent reads lineage P_t + knowledge base K, edits/tests/diagnoses, commits scored versions.
- `mha_performance.pdf`: MHA throughput bars on B200 (h=128, 16 heads, BF16), causal+non-causal, AVO vs cuDNN vs FA4.
- `gqa_performance.pdf`: GQA throughput, 32 Q heads, group sizes 8 and 4, causal+non-causal.
- `mha_performance_official.pdf`: AVO (measured) vs FA4-paper-reported cuDNN/FA4 numbers (appendix hedge).
- `avo_mha_causal_neurips.pdf` / `avo_mha_noncausal_neurips.pdf`: the evolution trajectory plots (Section 6 above — this is the one to replicate).
