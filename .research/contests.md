# Contest Works Distillation (for FA3→single-.cu autonomous agent)

Three studied works:
1. **syfi full-agent-pipeline** (`reference/related-works/mlsys-contest-syfi-fully-agent/full-agent-pipeline/`) — a fully autonomous driver (`run_pipeline.py`) that loops Claude Code headless + flashinfer-bench eval + an ncu MCP server. The most directly reusable artifact for us.
2. **mlsys2026-flashinfer-contest** (HAN Lab Kernel Mafia) — prompt/workflow release: 3-phase prompts, `verify.py`, `KernelWiki` + `ncu-report-skill` skills, Humanize plugin. 1st MoE, 2nd DSA, 3rd GDN in the MLSys26 FlashInfer NVIDIA track.
3. **mlsys2026-flashinfer-contest-solution** — final kernel snapshots + `verify.py` (pack-and-verify harness), per-submission `config.toml` + `scripts/pack_solution.py`.

All built on **flashinfer-bench** (`TraceSet`, `Benchmark`, `BenchmarkConfig`, `Solution`, `BuildSpec`, `pack_solution_from_files`). Target was B200/sm_100, CUDA 13.2. Our target is H200/sm_90a, CUDA 13.0, but the harness/agent shapes transfer almost verbatim.

---

## 1. AGENT PIPELINE SHAPE (syfi run_pipeline.py — the blueprint)

Two phases per track, each N refinement rounds gated by a real benchmark:
- Phase 1: PyTorch reference → **Triton** (N rounds)
- Phase 2: best Triton source → **C++/CUDA** (N rounds; CUDA initial prompt seeded with best Triton source text as a hint via `dir_to_prompt_hint`, truncated to 20k chars/file).

Round loop (`run_phase`):
1. `dump_definition_context()` writes `DEFINITION.md` (human IO schema), `definition.json` (full schema), `reference.py` (PyTorch ref) into the per-phase workdir for the agent to read.
2. Build prompt: round 0 = `initial_prompt`; round>0 = `refine_prompt` (carries last EvalResult + best EvalResult).
3. `run_claude()` spawns `claude -p <prompt> --output-format json --permission-mode bypassPermissions --model ... --mcp-config .mcp.json --strict-mcp-config [--resume <session_id>]`, cwd = phase_dir, timeout 5400s. Parses the JSON envelope; carries `session_id` forward so each round RESUMES the same Claude session (conversation continuity across rounds).
4. Agent writes only into `solution/<lang>/`. Driver, NOT the agent, runs the benchmark ("Do NOT run the benchmark yourself — the driver does that and feeds results back next round").
5. `evaluate_solution()`: packs via `pack_solution_from_files`, runs flashinfer-bench `Benchmark.run_all()`, aggregates per-workload into `EvalResult`.
6. Archive source to `round_NN_src/`; promote to `best_src/` if correct AND mean_speedup beats current best.
7. Append to `history.json`. Abort phase early if `is_env_fatal(error)` (driver/GPU/dataset failures the agent can't fix — saves API spend).

Key dataclass `EvalResult`: `correct, num_workloads, num_passed, mean/median/min_speedup, max_abs_err, max_rel_err, per_workload[], error`.

**Prompt engineering details worth copying:**
- `initial_prompt` lists files to read, a `_contract_block` (language, source dir, DPS yes/no, entry_point file::fn, binding), and explicitly says read DEFINITION.md/definition.json for exact param names/order/dtypes ("Mismatched arity will be rejected by the builder's signature validator").
- `refine_prompt` branches on `last.correct`: if INCORRECT → "Favor correctness this round; speed is secondary." If CORRECT → "increase the **minimum** speedup (not just mean) while keeping every workload passing" + a `focus` block of CUDA tuning levers (wgmma/tcgen05, TMA, cp.async, smem swizzle, mbarrier producer/consumer, register-tiling + launch bounds, persistent blocks). Shows first 8 per-workload lines (status/speedup/abs_err/log[:180]).
- If best exists: "Make targeted edits rather than a full rewrite unless you have a concrete reason to expect a big gain."
- A `_skill_block` lists available Skill names + the ncu MCP tool flow.

**Resume/checkpoint**: `--resume` flag — for each round whose `round_NN_src/` archive already exists, skip the agent and just RE-EVALUATE the archive. Lets a crashed/rebooted phase pick up without re-spending on Claude. (Coarse-grained: per-round, not per-step.)

**Multi-GPU / parallelism**: `--gpus` sets `CUDA_VISIBLE_DEVICES` for the whole pipeline (agent subprocess + bench runner + ncu). Modal path (`_benchmark_modal`) fans out workloads into N shards via `.starmap()` across N containers, merges per-workload client-side. Note: flashinfer-bench `Benchmark` parallelizes across *solutions* (always 1 here), so the only way to scale a single solution's eval is to **split the workload list** — directly relevant to our "perf = 1 kernel/GPU, correctness can share" plan.

**Reference-latency cache** (`runs/_ref_cache/<def>.json`, keyed by workload uuid → ref_latency_ms): when every workload is cached, sets `BenchmarkConfig(profile_baseline=False)` and skips the expensive reference timing loop, only timing the solution + correctness. `seed_ref_cache.py` pre-populates it by directly timing `BuilderRegistry.build_reference(definition)` with torch.cuda.synchronize + perf_counter (warmup then `iters`, mean ms). For us: cache the baseline `_vllm_fa3_C.fwd` latency per config once, then every candidate round only times the candidate. Big wall-time win.

HAN Lab's variant (prompts/README): a **3-phase** human-in-the-loop workflow (not fully auto): Phase 1 = research + first correct kernel; Phase 2 = profiling-guided bottleneck optimization (repeatable with progressively higher target speedup); Phase 3 = workload-shape analysis + specialize/dispatch per shape group. Uses Humanize plugin `/humanize:gen-plan` (draft.md → plan) and `/humanize:start-rlcr-loop` (plan→execute→review). Ablation: Humanize (plan-execute-verify structure) was the dominant contributor; KernelWiki + ncu-report-skill secondary.

---

## 2. CORRECTNESS VERIFICATION METHODOLOGY (verify.py + BenchmarkConfig)

flashinfer-bench is the judge. Pattern (identical across all three `verify.py` + `run_local.py`):
```python
solution = Solution.model_validate_json(path.read_text())   # packed solution.json
ts = TraceSet.from_path(dataset)
bench_ts = TraceSet(root=ts.root,
    definitions={d.name: d}, solutions={d.name: [solution]},
    workloads={d.name: workloads}, traces={d.name: []})
result_ts = Benchmark(bench_ts, BenchmarkConfig(...)).run_all(dump_traces=...)
for tr in result_ts.traces[d.name]:
    ev = tr.evaluation
    ev.status.value          # "PASSED"/... — correctness gate
    ev.correctness.max_absolute_error / .max_relative_error
    ev.performance.latency_ms / .reference_latency_ms / .speedup_factor
```
A solution PASSES a workload iff `ev.status.value.lower() == "passed"`; overall correct iff all workloads passed. The framework internally generates inputs (`gen_inputs`), runs the PyTorch reference, runs the candidate, and compares within tolerance — the agent never writes its own comparator.

**Tolerances (the actual numbers used):**
- Default: `rtol=1e-2, atol=1e-2` (syfi `DEFAULT_BENCH_CONFIG_KWARGS`).
- MoE-FP8 special-cased everywhere: `atol=1.0, rtol=0.3, required_matched_ratio=0.9` — i.e. quantized/low-precision ops loosen tolerance AND only require a *fraction of elements* to match (`required_matched_ratio`). Important: tolerance is op-specific; we should pick FA3 tolerances that map to greedy-decode invariance, not a fixed 1e-2.

**Bench config knobs (`BenchmarkConfig`):** `warmup_runs`, `iterations`, `num_trials`, `rtol`, `atol`, `required_matched_ratio`, `timeout_seconds`, `log_dir`, `use_isolated_runner`, `profile_baseline`.
- syfi default: warmup=3, iterations=50, num_trials=3, timeout=600.
- submission `run_local.py`: warmup=3, iterations=100, num_trials=5 (heavier, for final numbers).
- `--fast`: warmup=1, iterations=5, num_trials=1, limit=2 (smoke).
- `use_isolated_runner` / "isolated runner" = run each candidate in a separate process (spawn-based) — protects the driver from a candidate that segfaults / leaks. `manual_eval.py` notes the runner re-imports `__main__` in workers, so eval entrypoints must be real importable scripts, not heredocs.

**Representative-workload strategy** (phase prompts): during development, use ~8 hand-picked representative workload UUIDs (spanning the variable axes, e.g. num_tokens ∈ {1,2,6,7,8}), then run the FULL set (e.g. 23) only after a major improvement / for promotion. We should mirror this: a small "dev set" of vLLM configs for fast iteration, full sweep before promoting a candidate. `verify.py` supports `--workload-uuid` (repeatable) and `--limit N`.

For OUR contest: correctness target is greedy(argmax) decode unchanged for ≤100 sequential tokens (prefill+decode). That is stricter and more semantic than per-element rtol — we should add our own end-to-end check (run vLLM with swapped kernel, compare token streams) ON TOP of an element-wise tolerance gate, because matching argmax can tolerate larger per-element error than 1e-2 in non-decisive positions but must be exact at the argmax. Verify correctness FIRST, always, before profiling (matches every prompt's "correctness is the priority for this phase").

---

## 3. PERFORMANCE SCORING / MEASUREMENT

- Metric = **speedup_factor = reference_latency_ms / solution_latency_ms**, computed per workload by the framework. Aggregations the agent optimizes: mean, median, and **min** speedup across all workloads. Phase-2/refine prompts explicitly push to raise the *minimum* (worst-case) speedup, not just mean — avoids a kernel that's great on some shapes and a regression on others. Directly maps to our "match or exceed baseline across ALL configs."
- Latency measured inside flashinfer-bench (warmup_runs then `iterations` timed runs × `num_trials`, presumably min/median over trials). `seed_ref_cache.py` shows the canonical low-level recipe: warmup loop → `torch.cuda.synchronize()` → `perf_counter` around `iters` calls → `torch.cuda.synchronize()` → divide. For our robust CUDA-event measurement we should prefer `torch.cuda.Event(enable_timing=True)` start/stop around the kernel + synchronize, take median of trials, with adequate warmup (≥3) and enough iters that the timer resolution is negligible.
- The ncu MCP `compare_kernels` tool diffs two reports on `gpu__time_duration.sum` (and throughput metrics) with delta + %change — used to validate an optimization actually helped at the kernel level (separate from end-to-end latency).

---

## 4. SUBMISSION / CANDIDATE STRUCTURE

**Packed `solution.json`** (pydantic `Solution`): top-level `name`, `definition`, `author`, `spec` (= BuildSpec: `language`, `target_hardware:["cuda"]`, `entry_point:"file::fn"`, `dependencies:[]`, `destination_passing_style:bool`, `binding: "tvm-ffi"|"torch"|null`), and `sources: [{path, content}, ...]` — the source files are inlined as strings. So a "single self-contained .cu" maps cleanly to one `sources` entry. (Confirmed by reading `gdn_decode/solution.json`.)

**Per-submission dir layout** (solution repo):
```
submissions/<track>/
  config.toml            # [solution] name/definition/author  [build] language/entry_point/destination_passing_style
  scripts/pack_solution.py   # reads config.toml -> BuildSpec -> pack_solution_from_files -> solution.json
  scripts/run_local.py       # pack + Benchmark.run_all locally
  scripts/run_modal.py       # remote eval
  solution/cuda/kernel.cu    # (or solution/python/*.cu+*.py for python-language hybrids)
```
`pack_solution_from_files(path=source_dir, spec, name, definition, author)` walks the source dir and inlines all files into `Solution.sources`. `config.toml` is the single source of truth (name/def/author/lang/entry/dps).

**Candidate provenance** (HAN Lab CLAUDE.md / phase prompts mandate):
- `benchmark.csv` — every performance-related commit/result.
- `solutions.jsonl` — every candidate, with **parent links maintained as a DAG** (search tree of candidates). This is exactly the evolution/lineage structure we want for our evolution plot.
- Keep NCU profiling records per major optimization direction.
- syfi's equivalent: `history.json` per phase (round, eval dict, session_id, archive path) + `best_src/` + per-round `logs_round_NN/` (bench logs + packed solution.json). Lighter-weight but resumable.

**Entry-point / binding contract:** `entry_point="kernel.cu::run"`; DPS=true means the harness pre-allocates outputs and passes (inputs..., outputs...) and the kernel returns void; DPS=false means kernel allocates+returns. Binding "torch" → `torch::Tensor` params + `PYBIND11_MODULE(TORCH_EXTENSION_NAME,...)`; "tvm-ffi" → `tvm::ffi::Tensor` + `TVM_FFI_DLL_EXPORT_TYPED_FUNC`. **For us:** our contract is different (vLLM `torch.ops._vllm_fa3_C.fwd` signature; torch allowed only in `mha_fwd` host fn + `TORCH_LIBRARY_EXPAND`), but the principle — the agent MUST match the exact registered signature/arity or the build/registration is rejected — is the same. Bake the exact signature into the initial prompt and validate arity before eval.

---

## 5. REUSABLE FLASHINFER-BENCH / TOOLING HARNESS

- **flashinfer-bench API** (drop-in): `TraceSet.from_path`, build a single-solution `TraceSet`, `Benchmark(ts, BenchmarkConfig(...)).run_all()`, read `trace.evaluation.{status,correctness,performance}`. If our baseline isn't in a TraceSet we can replicate this shape with our own driver around `torch.ops._vllm_fa3_C.fwd` as the "reference."
- **ncu-mcp server** (`mcps/ncu-mcp/`) — THE most reusable component for our 2.1 (profiling parser). MCP tools: `list_sections_and_sets`, `list_rules`, `query_metrics`, `profile(binary, report_name, set/sections/metrics, kernel_filter, launch_count, launch_skip, device, apply_rules)`, `read_report_details(sections, include_rules, max_chars=8000)`, `read_report_raw`, `read_report_source`, `read_report_session`, `list_reports`, `compare_kernels`. It shells `ncu --csv --page details/raw/source/session`, parses CSV (`parsers.py`), and FORMATS for an LLM (`formatters.py`).
  - **Concise-feedback design (copy this for 2.1):** `read_report_details` returns, in order: (1) **Summary** of 5 key metrics (Duration, Compute SM Throughput, Memory Throughput, Waves/SM, Achieved Occupancy); (2) **Optimization Opportunities** = all NCU rule results deduped + **sorted by `Estimated Speedup` descending** as a markdown table (rule, est-speedup%, advice[:120]); (3) section metrics sorted by a priority list (SOL→Memory→Compute→Occupancy→…). Auto-filters noise: hides array-valued occupancy sweeps (`;` in value), hides 0%-pipe metrics, dedups rules, truncates to max_chars keeping summary+rules+top sections, emits "OUTPUT TRUNCATED, use sections= filter". Occupancy section auto-computes tightest limiter (min over registers/warps/smem/blocks/barriers). Source view dedups by (line,sass), puts USER code before library code, shows top-3 stall reasons as percentages, warns on 0 stall samples ("kernel too fast — use launch_count=10").
  - The whole point: turn raw ncu (thousands of metrics) into a ranked, evidence-backed, <8k-char actionable report. Profiling strategy embedded in docstrings: start `set="basic"` (~4 passes, ~5s) → identify bottleneck → re-profile targeted sections, use `set="full"` (~40 passes) only when needed.
- **`.mcp.json`**: minimal stdio MCP registration (`uv --directory ./mcps/ncu-mcp run ncu-mcp-server`). We register our profiling MCP the same way.
- **ncu-report-skill** (`skills/ncu-report-skill/`) — a Claude *skill* (not MCP) that uses the `ncu_report` Python module directly. Highly reusable references:
  - `reference/06-diagnosis-playbook.md` — 14 patterns (A–N) each = Signals (named metrics+thresholds) → Why → First-line fix → Deeper fixes → Exceptions. e.g. A small-grid/SM-idle (waves<0.5, grid<SM count → split-K/persistent), B tail-effect on variable-len (sort/split/chunkwise), C uncoalesced loads (sectors/request>5), E latency-bound (long_scoreboard>40% → unroll/cp.async/TMA), F compute-bound-not-on-tensor-cores (fma pipe>50%, tensor=0% → wgmma/tcgen05), H smem bank conflicts (pad [32][33]), K register spill (local ld/st>0, regs>128 → `__launch_bounds__`), L unintended FP64 (add `f` suffixes), M pipeline bubbles (double-buffer), N divergence. This is a ready-made "profiling → typed action" knowledge base — adapt thresholds for sm_90a and drop straight into our profiling-knowledge type.
  - `reference/04-python-api.md` — exact `ncu_report` module usage: `load_report → range_by_idx(0).action_by_idx(0)`, `action[metric].value()/unit()/kind()`, `action.rule_results_as_dicts()` (rule_name, message, estimated_speedup_pct — sort desc), `action.source_info(pc)` for per-PC→source-line stall mapping (needs `-lineinfo`), per-instance/PM-sampling via `num_instances()`/`as_double(i)`. PYTHONPATH = `<cuda>/nsight-compute-*/extras/python`.
  - `reference/08-b200-metric-names.md` — metric-name drift across GPU gens (e.g. `smsp__sass_inst_executed_op_global_ld.sum`, `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`, `dram__bytes = read+write`). **Caution: this file is sm_100; we need sm_90a names — enumerate via `action.metric_names()` and build our own.** Lesson: always wrap metric reads in try/except; never hardcode names blindly.
  - helpers: `harness_template.cu` (standalone single-kernel harness — compile in seconds with `-lineinfo`, enabling clean SASS↔source), `safetensors_loader.h` (header-only, load real workload tensors), `extract_stall_hotspots.py`, `plot_timeline.py` (ASCII PM-sampling timeline → tail effect), `list_flashinfer_workloads.py`, `ncu_utils.py`. SKILL golden rule: "Profile → Diagnose → Plan, never guess"; mandates one `profile/<run_name>/` dir per run; cite actual metric values, don't say "it's memory-bound" without the numbers.
- **KernelWiki skill** (`skills/KernelWiki/`) — offline structured KB: 2179 merged PRs + 48 wiki pages + contests/blogs/docs for Blackwell/Hopper kernels (FlashAttention-4, DeepGEMM, FlashMLA, tcgen05/TMEM/TMA/wgmma, warp specialization, pipelining). Query via `scripts/query.py "<nl>" [--tag --type --repo --symptom]`, `get_page.py <id> [--follow-sources]`, `grep_wiki.py`. Pre-built indices `queries/by-problem|technique|hardware-feature|kernel-type|language|repo.md`. Confidence ladder verified>source-reported>inferred>experimental; reproducibility concept<pseudocode<snippet<runnable<benchmarked. **Has Hopper/SM90 pages with `blackwell_relevance` — directly useful for our FA3 Hopper kernel** (wgmma, TMA, mbarrier, FlashAttention-3 patterns). This is the model for our "knowledge sources (KernelWiki, flashinfer-bench solutions)" component.

---

## 6. CROSS-CUTTING LESSONS / GOTCHAS FOR OUR AGENT

- **Verify correctness FIRST, then profile** — every prompt and the driver enforce it; refine prompt says favor correctness when not all pass, speed only when all pass. Our loop must gate profiling on a passing correctness check.
- **Optimize the MINIMUM speedup across configs**, not the mean — matches our "match or exceed across ALL configs" requirement exactly.
- **Per-round session resume** keeps the agent's reasoning continuous across refinement rounds (cheaper, more coherent than fresh sessions). Combine with archive-based checkpoint/resume for crash recovery.
- **Reference-latency caching** is a major wall-time lever when the baseline is slow to time repeatedly — cache baseline `_vllm_fa3_C.fwd` latency per config once.
- **Isolated/spawn runner** to survive segfaulting candidates; eval entrypoints must be importable real scripts (no heredocs).
- **Env-fatal detection**: classify driver/CUDA/GPU/dataset errors as unfixable and abort early instead of burning agent rounds. We'll want analogous classification (e.g. "ABI mismatch with torch", "no sm_90a device", "include path violation").
- **Tolerance is op-specific** — don't hardcode 1e-2; derive a tolerance that guarantees argmax invariance for FA3.
- **Self-contained CUDA is normal here**: gdn-decode/dsa kernel.cu are 200–550 lines, single .cu, only `#include <cuda_*.h>`/`<math.h>` + the binding header (tvm/ffi or torch). Maps cleanly to our no-external-libs constraint.
- **Real-world env hack worth remembering**: syfi's `gdn_decode/solution.json` works around a libcudart-vs-driver version mismatch by JIT-compiling the kernel via **NVRTC + CUDA Driver API** (dlopen libcuda.so.1 + libnvrtc, `nvrtcCompileProgram`→PTX→`cuModuleLoadData`→`cuLaunchKernel`), with a host-side fallback if any step fails. Demonstrates: (a) the agent can be pushed to robust fallbacks, (b) NVRTC/driver path sidesteps runtime-vs-driver checks — relevant if we hit CUDA 13.0 toolkit vs driver friction on H200.
- **Anti-cheating** (HAN Lab CLAUDE.md / README "Release Boundary"): explicit rule that the agent MUST NOT clone/inspect/copy the final-submission repo while solving; final kernels kept out of the agent workspace. For us this maps to: forbid the agent from reading/linking the prebuilt `_vllm_fa3_C` or any CUTLASS/CuTe/cuBLAS source into the output; enforce toolkit-only include path + nm/ldd scan of the built .so (no symbols from forbidden libs). The contest relied on policy + human boundary; we will enforce mechanically.

## Key file pointers (absolute)
- Driver: `…/full-agent-pipeline/run_pipeline.py` (EvalResult, run_phase, prompts, ref-cache, modal shards, resume, env-fatal).
- ncu MCP server: `…/full-agent-pipeline/mcps/ncu-mcp/src/ncu_mcp/{server,parsers,formatters,ncu_runner}.py`.
- MCP reg: `…/full-agent-pipeline/.mcp.json`.
- Ref-cache seeding: `…/full-agent-pipeline/scripts/seed_ref_cache.py`; manual eval: `scripts/manual_eval.py`.
- Verify harnesses: `…/mlsys2026-flashinfer-contest/verify.py` (single solution) and `…/mlsys2026-flashinfer-contest-solution/verify.py` (pack+verify all, MoE tolerance override).
- Phase prompts: `…/mlsys2026-flashinfer-contest/prompts/<track>/phase{1,2,3}.md`; prompt philosophy `prompts/README.md`.
- Diagnosis playbook: `…/skills/ncu-report-skill/reference/06-diagnosis-playbook.md`; ncu_report API: `reference/04-python-api.md`; metric drift: `reference/08-b200-metric-names.md`; SKILL.md golden rules.
- KernelWiki: `…/skills/KernelWiki/SKILL.md` + `CLAUDE.md` + `queries/`.
- Submission shape: `…/mlsys2026-flashinfer-contest-solution/submissions/<track>/{config.toml,scripts/pack_solution.py,solution/cuda/kernel.cu}`; packed schema example `…/mlsys-contest-syfi-fully-agent/gdn_decode/solution.json`.
