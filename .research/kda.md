# Kernel Design Agents (KDA) — Distillation for our FA3→single-.cu agent

Source: `/data-fast/agent/reference/related-works/kernel-design-agents/`
KDA = MIT HAN-lab's agent-centric workflow for using Claude Code to research/implement/verify/iterate
on perf-sensitive CUDA kernels. Won #1–3 on MLSys 2026 FlashInfer contest tracks. The *repo itself* is
deliberately tiny/task-agnostic (a flow + prompt template + 2 skills). All the value is in:
(a) the workflow loop discipline, (b) two installable Claude Code **skills** (ncu-report-skill, KernelWiki),
(c) a Blackwell-programming companion. We are on Hopper (sm_90a), so the *mechanics* transfer directly but
the metric names / hardware advice must be retargeted from sm_100→sm_90a.

================================================================================
## 1. HARNESS / REPO STRUCTURE (how the whole thing is laid out)
================================================================================

Top-level repo (the "flow reference", stays small & generic):
- `README.md`         — install, minimal flow, recommended workspace layout
- `CLAUDE.md`         — repo-facing agent rules (keep generic; put task stuff elsewhere)
- `docs/agent-flow.md`— the minimal loop + task contract + evidence records + promotion rule
- `prompts/README.md`, `prompts/basic-flow.md` — fill-in-the-blank starter prompt
- `skills/ncu-report-skill/` — profiling skill (SKILL.md + reference/*.md + helpers/*.py+.cu+.h)
- `skills/KernelWiki/`        — knowledge-base skill (SKILL.md + scripts/ + wiki/ + sources/ + data/)

KEY ARCHITECTURAL DECISION: **separate the reusable workflow from the task workspace.**
The repo explains the flow; a *separate* "implementation workspace" owns code, tests, datasets,
benchmark scripts, private acceptance thresholds, and generated artifacts. This maps onto us cleanly:
our agent harness code = the "flow"; the per-candidate .cu + profiling/bench artifacts = the "workspace".

Recommended workspace layout (we should mirror this for our run dirs):
```
task-workspace/
  docs/{draft.md, plan.md}          # draft.md = first plan; plan.md = executable plan
  runs/      outputs/      profile/  # generated artifacts (gitignored)
  benchmark.csv                      # tabular perf log
  candidates.jsonl                   # candidate name, parent link, status  <-- evolution tree
```

================================================================================
## 2. THE LOOP (drive mechanism) — `docs/agent-flow.md` + `prompts/basic-flow.md`
================================================================================

Minimal loop (their wording, condensed):
1. Define the **task contract**.
2. Agent inspects local workspace (read existing impl/tests/docs FIRST).
3. Agent writes `docs/draft.md` (plan draft) — **DO NOT start coding until draft exists**.
4. Convert draft → executable `docs/plan.md` (manually or with a planning tool, "Humanize" plugin).
5. Implement ONE candidate at a time.
6. **Validate correctness.**
7. Measure target metric (when applicable).
8. Record evidence; decide keep / revise / reject.
9. Repeat until promotion criteria met OR remaining blockers are explicit.

**Task Contract** fields (each task must state):
- Objective; Inputs & outputs; Correctness requirements (tolerances/invariants);
- Constraints (language, deps, APIs, deployment);
- Validation command; Evaluation command (if different); Promotion criteria.

**Promotion rule:** promote a candidate ONLY if it satisfies the contract AND has evidence it
improves-or-preserves the target metric. If rejected, **record the reason** (never silently discard).

**Evidence records** (simple files, consistency > format):
- `docs/draft.md`, `docs/plan.md`, `benchmark.csv`, `candidates.jsonl`, `profile/`, `runs/`/`outputs/`.
Goal: a future reader can reconstruct what was tried, what passed, and why a candidate was promoted.

`prompts/basic-flow.md` is a TEMPLATE with `<fill in>` placeholders for every contract field, then a
fixed 9-step Workflow section, then "Plan Draft Requirements" (the draft must contain: baseline + how
validated; risks/unknowns; candidate directions ranked by expected value AND risk; first concrete steps;
exact validation/eval commands; evidence required to promote/revise/reject). **"Do not start
implementation until the draft exists."** — explicit gate.

NOTE: KDA's published loop is mostly *manual/human-in-the-loop* (you paste the prompt, the agent drafts,
a human or Humanize converts to plan). It is NOT a fully autonomous closed loop like ours. Our agent must
supply the automation KDA leaves to the human: candidate generation, the verify-then-profile gate,
checkpoint/resume, multi-GPU scheduling. KDA gives us the *content/structure* of each step, not the driver.

================================================================================
## 3. SKILL FILE FORMAT & CONVENTIONS (we should mirror this exactly)
================================================================================

A Claude Code skill = a directory with `SKILL.md` at its root containing YAML frontmatter:
```
---
name: ncu-report-skill
description: <ONE dense sentence with explicit TRIGGER phrases — this is what the model matches on
             to auto-invoke. ncu-report-skill even lists Chinese trigger phrases.>
argument-hint: "[natural-language-question] | [--tag foo --type kernel] | [page-id]"   # optional
allowed-tools: "Bash Read Grep Glob"                                                   # optional, restricts tools
---
# Skill: <title>
**When to use:** <repeat triggers>
... body: a workflow with numbered phases, a file index table, "critical lessons", related skills ...
```
Install paths (Claude Code auto-discovers):
- `~/.claude/skills/<name>/SKILL.md`  (user-level, all projects)
- `<repo>/.claude/skills/<name>/SKILL.md` (project-level)
KDA installs via symlink from a clone, or git clone/submodule directly into `~/.claude/skills/`.

Skill body conventions worth copying:
- **"Golden rule" up top** in one line ("Profile → Diagnose → Plan, in that order. Never guess.").
- **Quickstart numbered 0..N** that names exact commands + which reference doc to read for detail.
- **File index table** (`| File | Purpose |`) pointing at `reference/NN-topic.md` docs — keeps SKILL.md
  short while letting the agent pull detail on demand (progressive disclosure).
- **"Critical lessons (don't skip)"** — hard-won gotchas (e.g. "metric names differ on B200",
  "always -lineinfo", "don't delegate understanding — cite specific metric values").
- Reference docs numbered `00-..09-` so reading order is obvious.
- `helpers/` holds reusable code (python parsers, a .cu harness template, a header-only loader).

Skills are SELF-CONTAINED and INVOCABLE two ways: manually `/skill-name`, or auto by the model when the
conversation matches the `description` triggers. For our headless sub-agents, we can ship skills the same
way and let the orchestrator name them, OR just inline the reference docs into sub-agent prompts.

================================================================================
## 4. ncu-report-skill — PROFILING (maps to our component 2.1) — DEEP NOTES
================================================================================

Directory: SKILL.md + reference/00..09 + helpers/. Target HW in the original = B200/sm_100; WE retarget
to H200/sm_90a (CUDA 13.0, ncu 2025.3). The *workflow* is HW-agnostic; only metric NAMES change.

### 4.1 Workflow phases (reference/01-workflow.md)
- **Phase 0**: `mkdir -p profile/<descriptive_run>/{harness,reports,analysis}`. ONE run = ONE dir,
  NEVER reuse. New kernel version OR new workload = new run. Everything written under `$PROFILE_RUN_DIR`.
- **Phase 0.5 (frame before tools)**: which kernel(s) (exact demangled name/regex — templated kernels need
  the full `void k<(int)8,(int)256>(...)` string), which workload/shape (pick REAL representative shapes,
  not synthetic), which dispatch path (profile each ACTIVE path separately — averaging hides patterns),
  what question, what baseline.
- **Phase 1 (env check)**: `ncu --version`, `nvidia-smi`, `nvcc --version`, locate `ncu_report.py` under
  `.../nsight-compute-*/extras/python`, export PYTHONPATH, set writable `$HOME` (silences section-file warn).
- **Phase 2 (build target)**: prefer a **standalone harness** (compiles in <5s vs minutes; lets you add
  `-lineinfo`; precise input control). Skip only if existing build already has `-lineinfo` and is fast.
- **Phase 3 (collect)**: two runs per (kernel, path, workload): `--set full` + PmSampling for overview,
  `--set source --section SourceCounters` for per-line stalls. `-c 1` profiles first match only.
- **Phase 4 (extract structured)**: DO NOT eyeball CLI. Parse with `ncu_report` python. Produce
  metrics_key, metrics_all.json, compare, stall_hotspots, pm_timeline, details (`--page details`).
- **Phase 5 (diagnose)**: walk **six dimensions** (below), cite specific metric VALUES.
- **Phase 6 (report)**: REPORT.md, top-3 findings visible in 30s, ranked by NCU `Est. Speedup`.

### 4.2 The CONCISE-feedback machinery (this is exactly our 2.1 deliverable)
The skill's whole point: turn a 2000-metric ncu dump into a SHORT, ranked, evidence-backed plan.
Mechanisms we should copy verbatim:
- **NCU rule engine is half the work**: `ncu --import full.ncu-rep --page details` emits
  `OPT Est. Speedup: X%` bullets. Read FIRST. Accessible programmatically via
  `action.rule_results_as_dicts()` → dicts with `type` (OPT/INF/WRN), `rule_name`,
  `message_for_display`, `estimated_speedup_pct`. Sort desc by est_speedup to rank fixes.
  Caveat: sum of Est.Speedups usually >100% (rules overlap) — don't add; use only to RANK.
- **Ranking template** for the plan: `(expected speedup) × (effort ratio)`, ≤3–5 priorities, each:
  Evidence (metric values) / NCU Est.Speedup / Effort / Why-now. >5 priorities = padding.
- **"Specificity is the deliverable"**: never "it's memory-bound"; instead name 2–3 metric values
  (e.g. "dram pct_of_peak <10% AND long_scoreboard dominates pcsamp → latency-bound on L1, not BW-bound").

### 4.3 Six analysis dimensions (reference/05) — the diagnostic lenses
1. **SM occupancy & launch geometry**: launch__{grid_size,block_size,waves_per_multiprocessor,
   registers_per_thread,shared_mem_per_block,occupancy_limit_{blocks,registers,shared_mem,warps}};
   device__attribute_multiprocessor_count (132 on H200 vs 148 B200);
   sm__maximum_warps_per_active_cycle_pct (theoretical occ), sm__warps_active.avg.pct_of_peak_sustained_active
   (achieved). Waves<1 = grid too small (Est.Speedup 50-90%); waves in [1,2) = tail wave;
   theoretical 100% but achieved<< = stall-bound → dim 3. Includes wave-math snippet.
2. **Thread-block balance / tail effect**: no single metric — combine per-SM active-cycle spread
   (details page "max X% above / min Y% below avg") + PM-sampling timeline SHAPE. Flat-high→clean-drop=ideal;
   flat-high→gradual-tail=tail effect (var-length inputs); flat-low=grid too small; sawtooth=pipeline bubble.
   ALWAYS inspect input distribution; max/avg or max/min work ratio >5x ⇒ tail risk.
3. **Stall reasons + per-line hotspots**: aggregate ratios
   `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` and per-PC
   `smsp__pcsamp_warps_issue_stalled_<reason>` (needs --set source + -lineinfo). Reason table:
   long_scoreboard=global-mem latency; short_scoreboard=shared/local/compute chain; wait=fixed-latency pipe;
   barrier=__syncthreads; mio/lg/tex_throttle=LSU pipe saturated; not_selected/selected=GOOD (parallelism);
   math_pipe_throttle=legit compute-bound. Rules of thumb: long_scoreboard>40%=mem-latency-bound;
   short>30%=dep chains; barrier>20%=too much sync; selected<10%=whole kernel stall-bound.
4. **Tensor Core util**: sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_{active,elapsed};
   subpipe hmma (BF16/FP16) / imma / dmma. 0% on a matmul kernel = missed optimization. (FA3 IS this kernel
   → expect high HMMA; on Hopper that's wgmma not tcgen05.)
5. **SM utilization timeline** (PM sampling, time-series): same shapes as dim 2. PM interval ~2µs; kernels
   <20µs give too few samples — interpret with care. (RELEVANT: our decode kernels are tiny.)
6. **Memory access pattern**: dram__bytes_read.sum.pct_of_peak_sustained_elapsed (≈80-100%=BW-bound;
   <10%+slow=latency-bound); l1tex/lts hit rates; sectors/request (ideal 4 for 128B coalesced; >8=bad);
   smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio (<16 = sparse writes);
   smsp__sass_inst_executed_op_local_{ld,st}.sum >0 = REGISTER SPILL (very bad).

### 4.4 Diagnosis playbook (reference/06) — pattern→cause→fix (each: Signals/Why/First-fix/Deeper/Exceptions)
A Small grid / SM idle • B Tail effect (var-len) • C Uncoalesced loads • D Sparse writes •
E Latency-bound (long_scoreboard) • F Compute-bound but not on TC • G Atomics contention •
H Shared-mem bank conflicts • I Sync overhead • J Achieved<<theoretical occ • K Register spill •
L FP64 used unintentionally (literal `1.0` not `1.0f`) • M Pipeline bubbles (no compute/mem overlap) •
N Warp divergence. Exceptions list tells you when a "violation" is EXPECTED (LLM decode is fundamentally
small-grid → split-K over KV; FlashAttention accepts some spill for fusion wins). EXTREMELY relevant to us:
FA3 forward = attention; decode (q_len=1) hits Pattern A/B by design.

### 4.5 Helper scripts (helpers/) — REUSE THESE near-verbatim (just swap metric list to sm_90a)
- `ncu_utils.py` — the core. `_locate_ncu_report()` probes common CUDA paths so `import ncu_report` works;
  `load_report/load_action`; `safe(action,name,default)` (wrap EVERY metric read — names differ per GPU gen);
  `metric_or_none(*candidates)` (try gen-specific names in order); `per_instance_values`,
  `metric_value_at` (value-kind robust: UINT64/DOUBLE/FLOAT/STRING); `dump_all_metrics`→JSON archive;
  `per_pc_values`+`pc_to_source_line` (PC→file:line via `action.source_info(pc)`, needs -lineinfo);
  `rule_results`/`rule_speedups` (sorted Est.Speedup). `B200_KEY_METRICS` = ~90 curated metric names —
  WE MUST PORT THIS to an `H200_KEY_METRICS` / sm_90a list (most names same; a handful differ).
- `analyze_reports.py --run-dir --report --tag` (repeatable) → metrics_all_<tag>.json, metrics_key_<tag>.{txt,json},
  and compare_<a>_vs_<b>.txt when ≥2 reports. This is our A/B comparison & key-metric extractor.
- `extract_stall_hotspots.py` → aggregates per-PC stall samples to per-(file,line), ranks top-30 + per-stall
  top-10. Directly names the offending LDG/BAR.SYNC/compute line. Needs --set source + -lineinfo.
- `plot_timeline.py` (ASCII PM timeline — reveals tail/sawtooth), `list_flashinfer_workloads.py`,
  `safetensors_loader.h` (header-only, no deps — load real tensors into the harness),
  `harness_template.cu` (standalone driver: CUDA_CHECK macro, fill_bf16/f32_random, paste-kernel section,
  explicit template instantiations, --workload safetensors mode OR synthetic shape mode).

### 4.6 Collection recipes (reference/03) — exact ncu commands
- Full: `ncu --set full --section PmSampling --section PmSampling_WarpStates -k "regex:K" -c 1 -o reports/full_<tag> ./harness`
  (45-50 replay passes, slow). Source: `ncu --set source --section SourceCounters -k regex -c 1 -o reports/source_<tag>` (~5 passes).
- Details (no recollect): `ncu --import full.ncu-rep --page details > analysis/details.txt`.
- Targeted re-check after a fix: `ncu --metrics <comma,list> -k regex -c 1 ./harness` (1-2 passes, prints table).
- `-s N` skip first N matches, `-c N` limit to N (skip warmup / pick steady-state iteration).
- `--set` ladder: basic(~3-5)/detailed(~15)/full(~45)/source(~50). Always start full.
- Find kernel name: `cuobjdump --dump-function-names ./harness` (or on the .so).

### 4.7 Critical lessons (verbatim-worth)
- Stock metric names DON'T all work on newer GPUs → enumerate with `action.metric_names()`; keep a
  curated, gen-specific KEY list. (For us: build/verify the list on the actual H200+ncu2025.3 once.)
- ALWAYS compile with `-lineinfo` (else source view blank, no per-line stalls). If build system can't add
  it (torch inline/JIT) → build a standalone harness. (Our .cu is already standalone-friendly.)
- PM sampling is the ONLY way to see tail effects (static metrics average over whole kernel).
- Don't delegate understanding — run profiles yourself, cite specific values.
- ERR_NVGPUCTRPERM → sudo, or `NVreg_RestrictProfilingToAdminUsers=0` in /etc/modprobe.d. (env)
- Reproducibility: lock clocks `nvidia-smi -lgc`, persistent mode `-pm 1`, pin stream. Don't use `-G` (100× slow).

================================================================================
## 5. KernelWiki — KNOWLEDGE BASE skill (maps to our 2.4/2.6/2.7 + knowledge sources)
================================================================================

A structured, queryable KB of Blackwell/Hopper kernel optimization: 2179 merged PR summaries
(cutlass32/sglang645/vllm833/flashinfer583/pytorch85/DeepGEMM1) + 48 synthesized wiki pages + 20 blogs +
11 doc summaries + 7 contest pages. Packaged AS a Claude Code skill (repo root = skill dir). Inspired by
Karpathy's "LLM Wiki" 3-layer pattern.

### 5.1 Three-layer architecture (good model for OUR knowledge store / in-loop distillation)
1. `sources/` — RAW, immutable summaries: `prs/{repo}/PR-{N}.md`, `contests/`, `docs/`, `blogs/`. Each has
   YAML frontmatter with a unique `id`.
2. `wiki/` — SYNTHESIZED, cross-referenced pages: hardware/ techniques/ kernels/ patterns/ languages/ migration/.
   Cross-linked by `id` via `sources:`/`related:`/`prerequisites:` fields.
3. `queries/` — AUTO-GENERATED indices (by-problem/technique/hardware-feature/repo/kernel-type/language).
   Regenerated by `scripts/generate-indices.py` from frontmatter — never hand-edited.

This 3-layer split is exactly the typed-knowledge model our spec wants (kernel-specific / profiling /
algorithmic): raw evidence → synthesized typed pages → auto indices for retrieval.

### 5.2 Page schema (frontmatter is the retrieval substrate — copy the discipline)
Typed `id` prefixes: pr-/doc-/blog-/contest-/hw-/technique-/kernel-/pattern-/lang-/migration-.
Required fields per type in `data/schemas.yaml`. Cross-cutting controlled vocabulary in `data/tags.yaml`
(80+ tags) + canonical-alias map `data/aliases.yaml` (UMMA→tcgen05, B200→sm100, etc.). A `validate.py`
enforces schema, link integrity, and quality gates (0 errors over 2265 files).
Key quality contracts (we should adopt analogous ones):
- `confidence`: verified (needs official-doc + upstream-code evidence) > source-reported > inferred > experimental.
- `reproducibility`: concept < pseudocode < snippet < runnable < benchmarked. technique/kernel/language pages
  MUST be ≥ snippet (compilable fragment, validator-checked).
- `performance_claims` MUST carry 6 fields: gpu, dtype, shape, metric, value, source_id (+utilization).
- version-sensitive claims pin tool versions (Triton 3.6, CUTLASS 4.5, CUDA) via a central registry.

### 5.3 Query tooling (3 navigation paths + indices + primer)
- `scripts/query.py "<nl>"` (alias-aware keyword + filters --type/--tag/--repo/--language/--symptom/--compact),
- `scripts/get_page.py <id|path> [--follow-sources|--body-only|--frontmatter-only|--include-code]`,
- `scripts/grep_wiki.py "<regex>" [--only wiki|sources]`. Scripts auto-resolve wiki root (`_wiki_root.py`),
  no env var. `references/primer.md` = topic→canonical-page-id lookup table (read first when question broad);
  `references/examples.md` = 10 worked question→command-sequence→synthesis patterns;
  `references/schema.md` = condensed schema.
- `artifacts/` = verbatim/extracted/derived code bundles pinned to upstream SHAs via PROVENANCE.yaml
  (byte-verified SHA-256). This is how they keep "reference code" without depending on the live repo.

### 5.4 Output contract when answering from the KB (good template for our knowledge sub-agent)
Cite page paths + ids; follow `sources:` to PRs/blogs/docs; respect confidence (flag inferred/experimental);
include the validated code snippet; report perf with all 6 fields. Anti-patterns: recommending techniques
without `sources:`; quoting perf without 6 fields; conflating sm90 vs sm100 (always check `architectures:`).

NOTE: KernelWiki is Blackwell-FIRST and content cutoff 2026-04-27. It explicitly covers FlashAttention-4
(sm100), DeepGEMM, FlashMLA, NVFP4. For OUR Hopper FA3 task it's useful for: (a) attention algorithmic
structure (ping-pong scheduling, software-exp, conditional rescale — see wiki/kernels/flash-attention-4.md),
(b) Hopper migration pages (wgmma, register accumulators — sm90 is what we target), (c) vLLM/FlashInfer PR
references for FA implementations, (d) the pattern→technique diagnosis indices. Treat sm100-specific HW
(tcgen05/TMEM/CLC/2CTA) as NOT-applicable to sm_90a (we use wgmma + register/smem accumulators).

================================================================================
## 6. Blackwell-CUDA-programming companion (skills/ncu-report-skill/blackwell-cuda-programming.md)
================================================================================
A Chinese-language "kernel programming checklist" — 16 numbered 准则 (guidelines), each tied to its
governing NCU metric + listing which kernel types naturally violate it (so the agent doesn't "optimize"
an inherent property). This is a strong template for an AGENT SELF-CHECK / pre-coding checklist:
- Each guideline = {principle, quantitative target, NCU diagnosis metrics, fix directions, EXCEPTIONS}.
- 16 generic guidelines: parallelism/occupancy; coalescing; shared-mem reuse; bank conflicts; warp divergence;
  register pressure; hide mem latency (ILP/TLP); precision & fast-math (`1.0f` suffix!); host-device sync;
  tensor-core use; grid/block config; atomic contention; vectorized loads; read-only path (`const __restrict__`,
  __ldg); pipelining/double-buffer; reduce sync.
- A "special kernel types" cheat sheet: LLM decode attention NATURALLY violates parallelism/grid/pipeline →
  fix via split-K over KV, persistent kernel, CUDA-graph (NOT by forcing more parallelism). FlashAttention/
  fused kernels accept register pressure for fusion savings. DIRECTLY relevant to FA3 fwd+decode.
- Stated motivation: "I know the principles but forget to apply them when actually coding" → forces a
  systematic checklist BEFORE writing, and a metric→guideline lookup AFTER profiling. We should bake an
  equivalent Hopper checklist into our generator/optimizer sub-agent prompts.
WARNING: the HW table is B200 (148 SM, TMEM, FP4). Retarget to H200/sm_90a: H200 has 132 SM,
no TMEM, wgmma not tcgen05, HBM3e ~4.8TB/s. Use only the *generic* guidelines + the metric mappings.

================================================================================
## 7. WHAT TO REUSE IN OUR AGENT (concrete)
================================================================================
1. Ship the profiling logic as `ncu_utils.py` + `analyze_reports.py` + `extract_stall_hotspots.py`
   near-verbatim; replace `B200_KEY_METRICS` with a verified sm_90a list (most names identical, confirm on
   the box once). Use `action.rule_results_as_dicts()`→sorted Est.Speedup as the CONCISE per-op feedback (2.1).
2. Adopt the "one run = one dir under profile/<name>/{harness,reports,analysis}; never reuse" rule, plus
   REPORT.md template (headline table → 6-dim analysis → ≤5 ranked priorities → confidence/caveats → repro).
3. Adopt the **verify-correctness-BEFORE-profile** gate and the **task contract + promotion rule + record
   the rejection reason** discipline. Our `candidates.jsonl` (name, parent, status, reason) = their evolution
   ledger → also feeds our evolution plot.
4. Adopt the **three-layer typed knowledge store** (raw sources → synthesized typed pages with frontmatter
   {id,confidence,reproducibility,sources,related} → auto-generated indices) for in-loop distillation, with
   the confidence/reproducibility/perf-claim quality gates and a validate.py-style checker.
5. Adopt the **SKILL.md format** (frontmatter name/description-with-explicit-triggers/allowed-tools; body =
   golden-rule + numbered quickstart + file-index table + critical-lessons) for our sub-agents (profiling,
   sanitizer, SASS, docs-nav, anti-cheat, cutlass-inspiration). Progressive disclosure: short SKILL.md +
   numbered reference/NN-*.md pulled on demand.
6. Adopt the **diagnosis playbook** pattern→cause→fix-with-EXCEPTIONS (esp. decode = inherently small grid;
   FA accepts spill) so the optimizer doesn't chase non-bugs.
7. Adopt a **Hopper programming checklist** (port the 16-guideline companion to sm_90a) injected into the
   generator/optimizer prompt — principle + target + NCU metric + fix + exceptions.
8. Harness pattern: standalone `.cu` with `-lineinfo`, explicit template instantiations, real-tensor loading
   via header-only safetensors loader; sanity-run WITHOUT ncu first (ncu errors are cryptic).

================================================================================
## 8. GOTCHAS / MISMATCHES (don't blindly copy)
================================================================================
- KDA targets B200/sm_100/CUDA 13.2/ncu 2026.1. WE target H200/sm_90a/CUDA 13.0/ncu 2025.3. Metric names
  and the entire Blackwell-feature layer (tcgen05/TMEM/CLC/2CTA/NVFP4) do NOT apply. Verify metric names on
  the actual box; keep `safe()`/`metric_or_none()` everywhere.
- KDA's loop is human-in-the-loop (draft→human/Humanize→plan→implement). It does NOT provide: autonomous
  candidate generation, checkpoint/resume, multi-GPU scheduling, CUDA-event latency harness, anti-cheating
  scans, compute-sanitizer integration, SASS comparison. We build all of those; KDA only informs structure.
- KernelWiki content is a SNAPSHOT (cutoff 2026-04-27) and Blackwell-first; FA3-on-Hopper specifics are thin.
  Use it for algorithmic structure + PR pointers, not as ground truth for our exact baseline.
- The repo's CLAUDE.md insists the flow repo stay generic — i.e. they intentionally DON'T ship the winning
  task harness. So there's no turnkey FA pipeline to lift; we reconstruct the automation ourselves.
- ncu `--set full` is ~45 replay passes; for tiny decode kernels this is fine but for prefill it's slow —
  use `--metrics <list>` for fast iteration re-checks, full only for the deep dive.
- PM sampling needs ≥~20µs kernel for meaningful timeline; many decode configs are sub-20µs → rely on
  static metrics + stall ratios there, not the timeline shape.
- Skill auto-invocation depends on the `description` trigger text; for headless sub-agents prefer explicitly
  naming/inlining the skill rather than relying on auto-match.
