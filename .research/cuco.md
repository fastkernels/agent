# CUCO (CuFuse) — Dense Distillation for Our FA3 Kernel Agent

UT Austin "CUCo/CuFuse": agentic framework that turns a host-driven CUDA+NCCL program into a
device-initiated (GIN/LSA) fused compute-communication kernel via a two-agent loop:
**fast-path (correctness-first) → slow-path (island-based LLM-driven evolutionary search)**.
Networking-focused, but the *harness, evolutionary engine, knowledge/memory management, candidate
scheduling, and infra* transfer almost 1:1 to our FA3-distillation agent. ~18K LoC Python.
Code: `/data-fast/agent/reference/related-works/cuco-code/`.

Their domain differs (they ADD device comm; we DISTILL FA3 to one .cu). The reusable core is:
treat the LLM as a **bounded variation operator over a structured design space**, persist EVERY
candidate (incl. failures) in a SQLite DB w/ embeddings + LLM feedback, drive mutation with a
phase-scheduled explore→exploit policy, maintain diversity with islands + MAP-Elites archive +
novelty filtering, and inject a closed-loop meta-summarizer recommendation signal. Correctness is
gated FIRST (cascade L1 compile → L2 verify → L3 benchmark); fitness = `10000/(1+t_ms)`.

================================================================================
## 1. END-TO-END ARCHITECTURE (docs/architecture.md, docs/slow-path-agent.md)
================================================================================
```
Host-driven seed → FAST-PATH (analyze→transform→annotate EVOLVE-BLOCK) → verified seed (gen0)
                 → SLOW-PATH (island evolutionary search w/ LLM mutation + cascade eval) → best
```
Module map (cuco/):
- core/runner.py        EvolutionConfig, EvolutionRunner — MAIN LOOP (1922 lines, the orchestrator)
- core/sampler.py       PromptSampler — assembles system+user prompt (diff/full/cross)
- core/summarizer.py    MetaSummarizer — 3-step cross-generation meta-learning
- core/novelty_judge.py NoveltyJudge — embedding + LLM novelty filter (anti-collapse)
- core/wrap_eval.py     generic eval wrapper (multi-run, validate_fn, aggregate, metrics.json)
- transform/cuda_analyzer.py  CUDAAnalyzer — REGEX static analyzer → AnalysisReport (comm graph)
- transform/transformer.py    HostToDeviceTransformer — LLM build/verify loop; insert_evolve_markers
- transform/pipeline.py       PreTransformPipeline — ordered conditional steps
- database/dbase.py     ProgramDatabase (SQLite), Program dataclass (THE central object)
- database/parents.py   PowerLaw / Weighted / BeamSearch / BestOfN parent selection
- database/inspirations.py  Archive + Top-K context selectors (MAP-Elites flavored)
- database/islands.py   assignment + Elitist/Directional migration, multi-seed islands
- database/complexity.py  radon/custom C++ complexity metrics (behavioral descriptor candidate)
- llm/{client,llm,query,embedding,dynamic_sampling}.py  provider routing, cost, UCB bandit
- llm/models/claude_cli.py  Claude Code CLI backend (`claude -p`) — DIRECTLY relevant to us
- prompts/{base,diff,full,cross,init,meta,novelty}.py  prompt templates
- edit/{apply_diff,apply_full,async_apply,summary}.py  EVOLVE-BLOCK-aware patch application
- launch/{scheduler,local,slurm}.py  job execution backends (async submit/poll)
- plots/  lineage tree, improvement-over-gen, pareto, similarity heatmap, code-evolution video
- webui/  HTTP server + D3 single-page viz of lineage

================================================================================
## 2. THE EVOLUTION LOOP (core/runner.py — most reusable file)
================================================================================
`EvolutionRunner.run()`:
1. (optional) pre-transform on init_program (fast path). Skipped if `completed_generations>0`.
2. **gen 0 runs SEQUENTIALLY** to populate the DB from the seed before parallelism begins
   (`_run_generation_0` or `_run_generation_0_multi_seed`).
3. Main loop: while not done OR jobs running:
   - `_check_completed_jobs()` poll → `_process_completed_job()` for each → add to DB
   - `_update_completed_generations()` — a gen is "complete" only if ALL gens 0..g have >=1
     program (contiguous, no gaps) → robust resume semantics
   - if `len(running_jobs) < max_parallel_jobs` and more to submit → `_submit_new_job()`
   - `time.sleep(2)` between polls
4. final meta summary + save meta_memory + db.print_summary.

`_submit_new_job()` (the per-candidate pipeline, lines ~1210-1355):
- get current meta-recs/summary/scratchpad from MetaSummarizer
- NESTED loop: `for nov_attempt in max_novelty_attempts: for resample in max_patch_resamples:`
  - `db.sample()` → (parent, archive_inspirations, top_k_inspirations) [island-constrained]
  - `run_patch(...)` → LLM generates patch, applied with up to `max_patch_attempts` retries
    (feeds compiler/patch errors back into the SAME LLM conversation each retry)
  - compute code embedding (`get_code_embedding`, redacts immutable regions first, cap 28k chars)
  - `novelty_judge.should_check_novelty` → `assess_novelty_with_rejection_sampling`; reject near-dup
- `scheduler.submit_async(exec_fname, results_dir)` → append RunningJob to queue (ASYNC, non-blocking)
- Note: embedding+novelty computed at SUBMIT time, reused at completion time (cached on RunningJob).

KEY: candidate code lives on disk at `results/gen_{N}/main.cu`; eval writes
`results/gen_{N}/results/{metrics.json, correct.json, build.log, run.log}`.

### CHECKPOINT + RESUME (built-in, no extra machinery)
- Resume detected purely by existence of the SQLite DB at `results_dir/db_path`.
- `self.completed_generations = db.last_iteration + 1`; `_update_best_solution()`,
  `_restore_meta_memory()` from `meta_memory.json`.
- ALL state is in SQLite + meta_memory.json + per-gen dirs. No separate checkpoint format.
- meta_memory.json saved ATOMICALLY (write .tmp then `replace`) after every job completion.
- Best program dir copied to `results_dir/best/` on every improvement.
- TAKEAWAY FOR US: persist the candidate DB to SQLite, write the best .cu + its NCU/score to a
  `best/` dir, and atomically checkpoint meta-memory after each step. Resume = "DB exists".

### MULTI-GPU PARALLELISM
- `max_parallel_jobs` + `JobScheduler.submit_async` + `check_job_status` + `get_job_results`.
- Their runner doesn't pin GPUs itself (one node, `CUDA_VISIBLE_DEVICES` set in env / eval).
- OUR NEED (perf=1 kernel/GPU, correctness can share): implement a GPU-id pool in the scheduler;
  hand each async perf job an exclusive GPU via `CUDA_VISIBLE_DEVICES`; correctness checks can
  oversubscribe. The async submit/poll loop here is the right skeleton.

================================================================================
## 3. CANDIDATE DATABASE & THE Program OBJECT (database/dbase.py)
================================================================================
`Program` dataclass fields (the schema we should mirror):
- identity: id (uuid), code (full source text), language, generation, island_idx, timestamp
- lineage: parent_id, archive_inspiration_ids[], top_k_inspiration_ids[], children_count,
           migration_history (list of {generation,from,to,timestamp})
- metrics: combined_score (REAL), correct (0/1), public_metrics (JSON timing), private_metrics,
           text_feedback (LLM diagnostic string)
- embeddings: embedding (BLOB), embedding_pca_2d/3d, embedding_cluster_id
- meta: complexity, code_diff (mutation diff from parent), in_archive (0/1), metadata (JSON blob:
        compute_time, api_costs, patch_type/name/description, stdout/stderr_log, directive, etc.)

SQLite-backed, persistent. Roles: (1) candidate pool for parent+inspiration sampling,
(2) knowledge base for meta-summarizer. **Failures (compile/verify) are stored with score 0** and
their diagnostics — used as NEGATIVE examples in later prompts (explicit negative selection).

`db.sample()` (line 820): if not all islands initialized → return initial program w/o inspirations;
else pick random initialized island → `CombinedParentSelector.sample_parent(island_idx)` →
`CombinedContextSelector.sample_context(parent, num_archive, num_topk)`.

`_update_archive()` (line 1315): MAP-Elites style. Archive holds only correct programs, capacity
`archive_size`; when full, new program replaces the WORST archived program if strictly better.
Archive is the exploitation pool (parent power-law sampling draws from it w/ prob `exploitation_ratio`).

================================================================================
## 4. PARENT SELECTION (database/parents.py)
================================================================================
4 strategies behind `parent_selection_strategy`:
- **power_law** (default): w/ prob `exploitation_ratio` sample from ARCHIVE (correct elites),
  else from all correct programs; both ranked best-first then `sample_with_powerlaw(items, alpha)`
  where prob ∝ (rank+1)^(-alpha). alpha=0 uniform/explore, alpha=1 strong exploit.
- **weighted** (used in FA/MoE run): per-archive-program weight w_i = sigmoid(λ·(score_i-median)/MAD)
  · (1/(1+children_i)). The 1/(1+children) is an explicit NOVELTY/over-exploration penalty.
  MAD-normalized → scale-robust across workloads. λ (parent_selection_lambda)=8.0 in their run.
- **beam_search**: lock onto a parent for `num_beams` children then switch to current best.
- **best_of_n**: always return the gen-0 seed (ablation baseline).
REUSE: weighted sampling with children-count penalty is a clean, scale-robust default for us.

================================================================================
## 5. INSPIRATIONS / CONTEXT (database/inspirations.py) — MAP-Elites in spirit
================================================================================
Each mutation prompt gets two context sets drawn from the ARCHIVE:
- **Archive inspirations** (`num_archive_inspirations`): best program + island elites
  (`elite_selection_ratio` of slots) + random correct, optionally island-separated
  (`enforce_island_separation`). Cross-pollination beyond current lineage.
- **Top-K inspirations** (`num_top_k_inspirations`): globally highest-scoring, as performance
  targets. Excludes parent + already-chosen archive insp.
Their FA/MoE config: num_archive=3, num_topk=3, elite_selection_ratio=0.3.

================================================================================
## 6. ISLANDS & MIGRATION (database/islands.py)
================================================================================
- Each program on exactly one island; children inherit parent's island; gen-0 seeds distributed
  one per island (or multi-seed: each island gets a distinct seed via different directive configs).
- `island_api_types` maps island→API tag (e.g. island0=LSA, island1=GIN) stored in metadata.
- Migration every `migration_interval` gens: Elitist (copy `migration_rate` fraction, protect the
  island's single best, only correct & gen>0 programs migrate) or Directional (follow a
  `migration_graph` Dict[src→[dsts]], e.g. {0:[2],1:[2],2:[]} = two specialist islands feed a hybrid).
- FA/MoE config: num_islands=2, migration_interval=8, migration_rate=0.15, island_elitism=True.
- FOR US: islands = distinct algorithmic strategies for FA3 (e.g. different tiling/pipelining/
  warp-specialization seeds, or one per head_dim/seqlen config class). Migration cross-pollinates.

================================================================================
## 7. PROMPT ASSEMBLY & MUTATION FORMS (core/sampler.py, prompts/, run_evo.py)
================================================================================
`PromptSampler.sample(parent, archive_insp, top_k_insp, meta_recs, island_idx)`:
- system msg = task_sys_msg (per-island override possible) + format block for sampled patch_type
  + (optional) DIRECTIVE_PROMPT + appended meta-recommendations.
- user msg = eval-history of inspirations (code+metrics+feedback) + parent code + parent perf +
  parent text_feedback section. For cross: append the cross components (2-3 archive programs).
Mutation forms (`patch_types`, probs sampled per generation):
- **diff**: SEARCH/REPLACE localized edits inside EVOLVE-BLOCK. Best for param tuning / sync changes.
- **full**: rewrite all EVOLVE-BLOCK content (5 prompt VARIANTS sampled randomly). Architectural.
- **cross**: crossover synthesis from 2-3 archive programs.
- LLM also emits `<NAME>`/`<DESCRIPTION>` tags (extracted with `extract_between`).

### PHASE-SCHEDULED EXPLORE→EXPLOIT (run_evo.py `_PHASE_CONFIGS`)
- explore (first `explore_fraction`=0.4 of gens): probs [diff .15, full .70, cross .15],
  temperatures [0.2,0.5,0.8] → diverse but compilable, discover architectures.
- exploit (rest): probs [diff .25, full .60, cross .15], temperatures [0.0,0.2,0.5] → refine.
- Implemented as TWO sequential EvolutionRunner.run() calls sharing the SAME results_dir/DB
  (phase 2 resumes from phase 1's DB). Elegant — no special phase state needed.
- Ablation: two-phase peaks 83.95 vs exploit-only 80.36 (4.3% lower); explore phase finds the
  diverse strategies (barrier-free overlap, split put/wait, signal-shadow) that exploit refines.

### EVOLVE-BLOCK markers (edit/apply_diff.py)
- `// EVOLVE-BLOCK-START` ... `// EVOLVE-BLOCK-END` delimit mutable regions; everything else FROZEN.
- Patches that touch frozen regions are REJECTED with a detailed error (lists editable regions) fed
  back to the LLM. `insert_evolve_markers()` annotates a fresh seed (LLM + regex fallback).
- `redact_immutable(code, no_state)` collapses frozen regions to placeholders — used before
  embedding so similarity reflects only the MUTABLE code. CRUCIAL for our case: FA3 has a large
  frozen torch host wrapper / registration; mark only the kernel body as evolvable, embed only that.

================================================================================
## 8. CASCADE EVALUATION & SCORING (paper slow-path; workloads/.../evaluate.py)
================================================================================
3-level fast-fail cascade per candidate:
- **L1 compile** (nvcc): fail → store score 0, feed compiler errors to next mutation.
- **L2 run+verify**: substring `"Verification: PASS"` in stdout; fail → score 0 + runtime errors.
- **L3 benchmark**: parse `Time: X.XXXX ms`, run NUM_RUNS=2, take the BEST (min) run.
- fitness `combined_score = 10000/(1+time_ms)`; failures = 0.
evaluate.py structure (the contract we mirror; it is what each gen_N job runs):
- writes metrics.json {combined_score, public{time_ms,per-rank,build_duration,all_run_times},
  private{stdout_excerpt}, text_feedback} and correct.json {correct, error}.
- text_feedback is RICH and ACTIONABLE: time, score, per-rank breakdown, run stability (spread%),
  NEW BEST / REGRESSION vs best-so-far (computed by scanning sibling gen_*/results/metrics.json),
  per-phase timings, and an LLM optimization suggestion (<=200 words, 1-3 concrete improvements).
- **Per-phase profiling via LLM instrumentation** (`_try_instrument`): an LLM inserts
  cudaEventRecord/ElapsedTime around phases, prints `Phase: name = X ms`; build the instrumented
  variant, FALL BACK to original if it fails to compile. Phase lines parsed back into feedback.
  -> This is their lightweight stand-in for NCU. WE replace this with real NCU parsing.
- **Hardware context injected dynamically** (`get_hardware_context`): arch->(SMs,HBM BW,L2,smem)
  lookup table, CUDA/NCCL versions from paths, topology from mpirun flags. Portability via injected
  context, NOT retraining. WE inject H200/sm_90a facts (132+ SMs, 228KB smem, etc.) the same way.
- **Anti-cheating already present in their prompt**: hard constraint "do NOT add external libs
  (cuBLAS/cuDNN/CUB/Thrust); build links ONLY cuda+nccl+mpi; any extra #include fails compile;
  compute kernel internals are FIXED." WE need a STRONGER enforced version: toolkit-only include
  path + nm/ldd scan of the .so + symbol diff vs forbidden libs (CUTLASS/CuTe/cuBLAS). CUCo relies
  only on the prompt + the fact that the build won't link extra libs; we must ENFORCE.

================================================================================
## 9. STATIC ANALYZER (transform/cuda_analyzer.py) — pattern for concise actionable feedback
================================================================================
Pure-regex CUDA analyzer (no clang). Extracts: includes, macros, fn defs, __global__ kernel defs
(name, line span, params), for-loops (with body ranges), NCCL collectives (+ positional arg parse
per signature table), nccl setup calls, mem allocs (cudaMalloc vs ncclMemAlloc -> gin_compatible
flag), kernel launches (`name<<<grid,block,smem,stream>>>(args)`), sync points, MPI calls.
Builds a **communication dependency graph** (CommunicationNode): per collective -> parsed args,
buffer roles (send/recv) w/ allocation provenance, producer kernels (writes before the call) and
consumer kernels (reads after), pointer-alias resolution (`T* x = base + off`), memcpy/memset
producer/consumer detection. `format_for_llm()` emits a COMPACT graph + per-function execution-order
timeline (`compute -> communicate -> ...`) + a one-line structural summary (counts only, NOT a dump).
TAKEAWAY: the "regex extract structured facts -> emit COMPACT graph + timeline, never dump code"
recipe is exactly the shape our (2.1) NCU parser and (2.3) SASS analyzer should follow: parse the
raw tool output, return only the per-op actionable deltas (occupancy, smem, regs, stall reasons,
mem throughput), with file:line anchoring. Helpers (`_extract_balanced_parens`,
`_split_toplevel_commas`, `_find_matching_brace`) are reusable to parse kernel launch configs out of
a .cu without a real C++ parser.

================================================================================
## 10. META-SUMMARIZER — closed-loop in-loop distillation (core/summarizer.py, prompts_meta.py)
================================================================================
Every `meta_rec_interval` programs (NOT generations — counts unprocessed programs, robust to
parallel jobs), runs a 3-step LLM pipeline:
- **Step1 Individual summaries** (BATCH query, one per program incl. failures): format
  "Program Name / Implementation / Performance / Feedback" + identifier (gen, patch_name, correct).
- **Step2 Global insights scratchpad**: merges new summaries + PREVIOUS insights + CURRENT BEST
  program into 4 sections: Successful Algorithmic Patterns / Ineffective Approaches /
  Implementation Insights / Performance Analysis. "Do not recommend; ONLY analyze."
- **Step3 Recommendations**: produce `max_recommendations`=5 specific/actionable/evidence-based/
  diverse recs, prioritizing extensions of the current best program. "ONLY algorithmic recs;
  do NOT change the eval code."
State (meta_summary, meta_scratch_pad, meta_recommendations, history, unprocessed programs)
persisted in meta_memory.json (atomic) and human-readable meta_N.txt. Recs injected into every
subsequent mutation system prompt. This is the framework's KNOWLEDGE-TYPING / in-loop distillation:
the scratchpad is a persistent typed memory of what worked / failed / why.
FOR US: this maps directly onto the required typed knowledge (kernel-specific / profiling /
algorithmic). Keep three scratchpads (or three sections) updated by a periodic meta-summarizer
fed by NCU deltas + correctness outcomes + SASS diffs.

================================================================================
## 11. NOVELTY FILTERING (core/novelty_judge.py)
================================================================================
- embed candidate code (immutable-redacted) -> cosine sim vs all DB entries; if > threshold
  (`code_embed_sim_threshold`, FA/MoE used 0.995) -> reject, resample (up to `max_novelty_attempts`=5).
- Optional second-stage LLM novelty judge for structural difference. Only checks once all islands
  initialized. Prevents population collapse to lexical variants of the best.

================================================================================
## 12. LLM BACKEND & MODEL SELECTION (llm/, dynamic_sampling.py)
================================================================================
- Multi-provider routing (Anthropic/Bedrock/OpenAI/Gemini/DeepSeek/Claude-CLI). FA/MoE run used
  Claude Opus for mutation+meta, Titan embeddings; paper headline runs used Sonnet 4.5, <$10/workload.
- **claude_cli.py**: invokes `claude -p --model <alias> --output-format text
  --dangerously-skip-permissions --no-session-persistence --tools ""` (tools DISABLED -> pure
  text-in/out) for patch/feedback generation; long prompts via STDIN to dodge arg-length limits;
  msg_history flattened into one user message; 600s timeout. cost=0.
- **Agent (autonomous) mode for fast-path** (runner `_run_pre_transform_agent`): `claude -p` WITH
  `--allowedTools Bash,Read,Write,Edit --add-dir <workdir> --add-dir <srcdir> --system-prompt ...`,
  1800s timeout, full filesystem autonomy to iterate build/run/fix. This is the headless sub-agent
  pattern our build calls for (claude CLI 2.1.170, --dangerously-skip-permissions).
- **AsymmetricUCB bandit** (dynamic_sampling.py): UCB1 + epsilon-greedy over LLM models as arms;
  reward = child combined_score shifted by parent/baseline (asymmetric: only positive improvement
  counts), optional exponential (log-space) scaling, auto-decay 0.95. `update_submitted` on dispatch,
  `update(arm, reward, baseline)` on completion. Lets the loop learn which model/temperature works
  best for this workload. Reusable if we run multiple models or temperature arms.

================================================================================
## 13. STRUCTURED DESIGN SPACE (paper 3.1, cufuse.tex, directive.py) — concept to adapt
================================================================================
CuFuse forces the LLM to emit an `<DIRECTIVE>` block (8 dims: backend, sync mechanism, placement,
sync scope, issuer granularity, chunk size, ordering, contexts) BEFORE writing code. Concrete dims
map to fixed API identifiers (validated); intent dims are free-text. Stored in metadata, makes
decisions inspectable & traceable. Fast-path emits a fixed CONSERVATIVE directive; slow-path treats
the directive as the SEARCH VARIABLE. Expert systems (DeepEP, FLUX) map onto points in this space.
FOR US (FA3 is single-GPU compute, no comm): define an ANALOGOUS structured directive for the FA3
kernel design space — e.g. {tile_M, tile_N, tile_K, num_warps, pipeline_stages, smem_layout
(swizzle), mma_shape, q/k/v load policy (cp.async/TMA), softmax_online vs two-pass, splitK/splitKV,
register vs smem accumulation, causal-masking strategy}. Make the LLM emit the chosen point first,
validate against sm_90a limits, store in metadata, use as MAP-Elites behavioral descriptor +
constrain the search to valid configs (prevents free-form hallucinated code).

================================================================================
## 14. ABLATION EVIDENCE (eval.tex) — why each component is load-bearing
================================================================================
- FAST-PATH necessity: w/ fast-path a correct kernel in 2 LLM iters then slow-path 83.95 by gen3;
  WITHOUT it the first 5 gens all FAIL (compile/deadlock), wasting 25% of budget, peaks 81.81
  (2.5% lower). -> ALWAYS gate correctness first / seed from a known-correct baseline.
- TWO-PHASE schedule: 83.95 vs exploit-only 80.36 (4.3% lower). Explore-phase diversity matters.
- POPULATION search vs naive iterative prompting: 17% improvement by gen3 vs naive plateau at 1.1%.
- Search cost (Sonnet 4.5): Flash Attention = 3 fast-path iters + 20 slow-path gens, ~1.8 hr,
  $10.96. Latency gains 5.3-36.2% over host baseline; up to 1.57x.
- FA case study: evolved kernel = single cooperative grid (B*n_h+1 = 81 blocks), block0 = comm
  block running ring protocol w/ atomic-counter sync, others compute; per-tile pipelining via
  atomicAdd(tile_ready) + __nanosleep(100) polling. Host baseline suffers GPU-side priority
  inversion (comm kernel can't schedule while attention saturates SMs). (FA here = ring-attention
  context-parallel comm workload, NOT the FA3 single-kernel we distill — but the kernel-design
  vocabulary (cooperative grid, atomic counters, polling, per-tile pipeline) is relevant.)

================================================================================
## 15. CONCRETE THINGS TO ADOPT IN OUR AGENT
================================================================================
1. Mirror runner.py's structure: gen0 sequential seed -> async submit/poll loop -> SQLite DB of EVERY
   candidate (incl. failures w/ score 0 + diagnostics) -> resume = "DB exists". Atomic meta-memory.
2. Cascade eval, correctness FIRST: L1 nvcc compile -> L2 run+greedy-decode-equivalence vs baseline
   torch.ops._vllm_fa3_C.fwd -> L3 CUDA-event latency (best of N). fitness from latency only when
   correct. Our L2 must check the actual requirement: argmax-token-equivalence over up to 100 tokens
   (not a PASS string). Encode that as the validate_fn in a wrap_eval-style harness.
3. evaluate.py-style RICH text_feedback per candidate: latency vs baseline + best-so-far,
   NEW BEST/REGRESSION flags, per-config breakdown (all vLLM shapes), and concise NCU-derived
   bottleneck note. Keep it ACTIONABLE & <200 words (their explicit cap).
4. EVOLVE-BLOCK markers around ONLY the kernel/device code; freeze the torch mha_fwd host wrapper +
   TORCH_LIBRARY_EXPAND registration. Reject patches touching frozen regions w/ a helpful error.
   redact_immutable before embedding so novelty/similarity reflect only the kernel.
5. Phase-scheduled explore->exploit as two sequential runner passes over the same DB
   (explore: full-rewrites high temp; exploit: diffs low temp). Use 0.4 explore fraction default.
6. Islands = distinct FA3 strategies (different tiling/pipeline/warp-spec seeds, or per vLLM config
   class); elitist or directional migration every ~8 gens; MAP-Elites archive keyed on the FA3
   design-space directive. Novelty filter (cosine on redacted code, thresh ~0.99) to avoid collapse.
7. Structured FA3 directive (section 13) emitted before code, validated vs sm_90a, stored as metadata
   + archive descriptor — bounds the LLM as a variation operator, not a free coder.
8. 3-step meta-summarizer for our TYPED knowledge (kernel-specific/profiling/algorithmic): persistent
   scratchpad of what improved latency, what broke correctness, NCU/SASS-derived insights, injected
   into subsequent mutation prompts. Count unprocessed candidates (not gens) to trigger.
9. NCU/SASS parsers follow cuda_analyzer's recipe: parse raw tool output -> emit COMPACT per-op
   deltas w/ file:line, never dump. Optionally an LLM-instrumentation fallback (their _try_instrument)
   when NCU is too slow/unavailable, with compile-and-fallback safety.
10. Headless sub-agents via `claude -p` (text-only for mutation/feedback; Bash/Read/Write/Edit +
    --add-dir for autonomous fast-path/debug agents; STDIN for long prompts; per-call timeouts).
11. ENFORCE anti-cheating beyond the prompt: toolkit-only -I include path, nm/ldd scan of compiled
    .so for forbidden symbols (CUTLASS/CuTe/cuBLAS/cuDNN/_vllm_fa3_C), reject candidate if present.
    (CUCo only relied on prompt + link-time failure — we need active enforcement, component 2.5.)
12. Evolution plot + lineage tree come free from the DB (plots/plot_improvement.py best-score-over-gen,
    plot_lineage_tree.py NetworkX from parent_id). Build score-over-step plot from combined_score.

================================================================================
## 16. GOTCHAS / DIVERGENCES
================================================================================
- CUCo is networking/multi-GPU (NCCL GIN/LSA, mpirun, ring attention). Their "Flash Attention"
  workload is CONTEXT-PARALLEL RING ATTENTION (a comm workload), NOT the single-GPU FA3 fwd kernel
  we distill. The *algorithm/harness* transfers; the *domain knowledge* (GIN/LSA rules, comm graph)
  does NOT — replace with FA3/Hopper knowledge (TMA, wgmma, async pipelines, smem swizzle, warp spec).
- Their correctness check is a stdout substring "Verification: PASS". Ours is far stricter:
  numerically-close-enough outputs that keep greedy decoding identical over 100 tokens. Must be a
  real numerical comparator vs the prebuilt baseline op, not a string match.
- Their "profiling" is LLM-inserted cudaEvent phase timers, not NCU. We must build the real NCU/Nsys
  parser (component 2.1) and compute-sanitizer integration (2.2); CUCo gives the FEEDBACK SHAPE, not
  the tool integration.
- Anti-cheat is prompt-only + link-time; insufficient for our toolkit-only / nm-ldd enforcement need.
- claude_cli backend reports cost=0 and estimates tokens as len//4 — cost tracking is unreliable
  when using the CLI; don't trust its cost numbers for budgeting.
- max_parallel_jobs in their runner does NOT pin GPUs; we must add a GPU pool for perf=1-kernel/GPU.
- Regex analyzer is brittle on macros / templated code; fine for extracting launch configs but not a
  substitute for SASS/NCU ground truth.
