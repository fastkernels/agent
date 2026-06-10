# NCU report parsing → concise actionable feedback (for §2.1 of our agent)

Two reference bodies studied:
- **ncu-report-skill** (`reference/related-works/kernel-design-agents/skills/ncu-report-skill/`) — a human-grade Claude Code skill: build harness → run ncu → parse with the `ncu_report` Python module → 6 analysis dimensions → diagnosis playbook → ranked report.
- **CUDAForge** (`reference/related-works/cudaforge/`) — an autonomous Coder/Judge agent paper. Its key contribution for us: how to pick a SMALL (24-metric) subset of NCU metrics and feed *only* that to the LLM, why the full set hurts, and the exact Judge prompt that produces one-bottleneck-one-fix JSON.

The two are complementary: ncu-report-skill = the *mechanics* (commands, Python API, metric meanings, pattern→fix). CUDAForge = the *distillation discipline* (metric selection + ultra-terse structured feedback + one-fix-per-round loop). Our agent's §2.1 parser should combine both: parse the full report with the Python API, then emit a CUDAForge-style terse, ranked, evidence-cited feedback blob.

NOTE on metric names: ncu-report-skill targets B200/sm_100 (ncu 2026.1); CUDAForge uses generic/older names (`dram__bytes.sum`, `smsp__warp_issue_stalled_*_per_warp_active.pct`). OUR target is H200 / sm_90a / CUDA 13.0 / ncu 2025.3 — so we must verify metric names via `ncu --query-metrics --chip gh200` or `action.metric_names()`. Several B200 names differ on Hopper; several CUDAForge names also differ. Treat both metric lists as *starting points*, not gospel.

---

## 1. HOW THEY COLLECT (commands)

### ncu-report-skill: two-pass collection, write .ncu-rep, parse later
Recipe 1 — full overview (mandatory first pass, ~45-50 replay passes):
```
ncu --set full \
    --section PmSampling --section PmSampling_WarpStates \
    -k "regex:KERNEL_REGEX" -c 1 \
    -o $RUN/reports/full_<tag> ./harness [args]
```
Recipe 2 — source-level per-PC stalls (fast ~5 passes; NEEDS `-lineinfo` at compile):
```
ncu --set source --section SourceCounters \
    -k "regex:KERNEL_REGEX" -c 1 \
    -o $RUN/reports/source_<tag> ./harness [args]
```
Recipe 3 — details page (rule engine, no re-collection, just import):
```
ncu --import $RUN/reports/full_<tag>.ncu-rep --page details > analysis/details_<tag>.txt
```
Recipe 4 — CSV/raw export for grep/awk: `ncu --import ... --page raw --csv > raw.csv`; `--page source` for SASS-source.
Recipe 5 — targeted metrics only (1-2 passes, prints table to stdout, no -o): `ncu --metrics m1,m2,... -k regex -c 1 ./harness`. Use for fast re-check after a code change.
Recipe 6 — A/B: profile v1 and v2 separately, compare durations.

`--set` mapping (B200, ncu2026): basic(SOL+Launch+Occupancy ~3-5 passes) < detailed(~15) < full(everything except Source ~45) < source(full+SourceCounters ~50). PmSampling/PmSampling_WarpStates/SourceCounters are NOT in `--set full`; add explicitly. NVLink: `--section Nvlink_Topology --section Nvlink_Tables`.
Useful flags: `-k "regex:..."` (match DEMANGLED name), `-c N` (limit launches, avoids loop dupes), `-s N` (skip first N matches — skip warmup), `-o` (.ncu-rep path).

### CUDAForge: profile a curated subset, feed text verbatim to LLM
They call `NCU_Profile(kernel)` per round but DO NOT pass the full metric set to the model. Instead they pre-select 24 metrics OFFLINE (see §3) and only feed those (as raw text in `{NCU_METRICS}`) to the Judge. The Judge itself then captures only 3-4 of those 24 per round. They run on RTX 6000 / various GPUs; profiling the subset (vs full) cuts time from ~40 min to ~26.5 min per kernel and API cost from ~$1 to ~$0.3 (fewer input tokens to the Judge).

---

## 2. HOW THEY PARSE — `ncu_report` Python module (NOT eyeballing CLI, NOT CSV)

The ncu-report-skill's central rule: **parse with the `ncu_report` Python module**, not by reading CLI tables. Module ships with ncu under `<cuda>/nsight-compute-<ver>/extras/python/ncu_report.py` — must be on PYTHONPATH. `ncu_utils.py::_locate_ncu_report()` auto-probes `/usr/local/cuda-*/nsight-compute-*/extras/python` and `/opt/nvidia/...` so the helpers work without manual PYTHONPATH. (For OUR env: CUDA 13.0 + ncu 2025.3, the path is something like `/usr/local/cuda-13.0/nsight-compute-2025.3.*/extras/python`.)

Core API (from `04-python-api.md` and `ncu_utils.py`):
```python
report = ncu_report.load_report(path)
rng    = report.range_by_idx(0)          # with -c 1: exactly one range
action = rng.action_by_idx(0)            # one kernel launch
action.name()                            # demangled name
action.metric_names()                    # all 2000+ metric names — ENUMERATE to discover real names
action[name].value()                     # aggregate value (wrap in safe())
action[name].unit()                      # "%", "cycle", "byte", "ns"...
action.source_info(pc) -> .file_name(),.line()   # PC→source (needs -lineinfo)
action.sass_by_pc()                      # {pc: SASS string}
action.ptx_by_pc()                       # {pc: PTX} (needs --keep)
action.rule_results_as_dicts()           # NCU rule-engine output as dicts
m.num_instances(), m.has_correlation_ids(), m.correlation_ids()   # per-instance / per-PC
m.kind()  m.as_uint64(i)  m.as_double(i)  m.as_string(i)          # value-kind-aware accessors
```
Robustness idioms (steal verbatim):
- `safe(action,name,default=None)`: try/except around `action[name].value()` — metric names vary by GPU gen, KeyError is expected. **Always** wrap.
- `metric_or_none(action, *candidates)`: try a list of names, return first that works — for GPU-gen name differences (e.g. B200 vs H200).
- `metric_value_at(m,i)`: dispatch on `m.kind()` (UINT64/DOUBLE/FLOAT/STRING) for per-instance reads.
- `dump_all_metrics(action, outfile)`: archive ALL metrics+units to JSON once, so re-analysis never reopens the .ncu-rep.

PC→source-line stall mapping (the highest-value piece for actionable feedback):
```python
m = action[stall_metric]; cor = m.correlation_ids()
for i in range(m.num_instances()):
    pc = cor.as_uint64(i); val = metric_value_at(m,i)
    file,line = action.source_info(pc)   # aggregate by (file,line), sort desc
```

CUDAForge by contrast keeps it crude: it forwards the NCU text verbatim for the 24 metrics; the *LLM* does the interpretation. No Python parsing layer described. The lesson is the OPPOSITE direction is wrong (full dump), not that parsing is unnecessary.

---

## 3. HOW THEY IDENTIFY BOTTLENECK CLASS (memory / latency / compute-bound)

### ncu-report-skill: 6 analysis dimensions + diagnosis playbook (pattern→cause→fix)
Six dimensions (walk all six; only 1-2 dominate any kernel):
1. **Occupancy & launch geometry**: `launch__waves_per_multiprocessor` (<1 grid too small → 50-90% Est.Speedup; in [1,2) tail wave; >4 fine), `launch__occupancy_limit_{blocks,registers,shared_mem,warps}` (which is tightest), theoretical (`sm__maximum_warps_per_active_cycle_pct`) vs achieved (`sm__warps_active.avg.pct_of_peak_sustained_active`). Wave math: blocks_per_sm=min(limits); wave=blocks_per_sm*num_sms; last-wave util = remainder/wave.
2. **Thread-block balance / tail effect**: no single metric — per-SM active-cycle spread (details page "max X% above, min Y% below"), PM-sampling timeline shape, input-length distribution (max/avg>3 → tail). Critical for variable-seqlen attention (our case: prefill+decode mixed).
3. **Stall reason breakdown + per-line hotspots**: aggregate `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` + per-PC `smsp__pcsamp_warps_issue_stalled_<reason>`. Normalize pcsamp counts by `smsp__pcsamp_sample_count` to get "% of samples on X".
4. **Tensor-core util**: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` (0% on matmul-ish kernel = missed; <50% underutilized; >50% good). For FA3 on Hopper this is the #1 metric — wgmma must be active.
5. **SM utilization timeline** (PM sampling): flat-high+clean-drop=ideal; flat-high+tail=tail effect; flat-low=tiny grid or stall-bound; sawtooth=no compute/mem overlap (missing double-buffer/pipeline).
6. **Memory access pattern**: DRAM %peak (≈80-100% = BW-bound; <<10% = NOT BW-bound→latency/compute), L1/L2 hit rate, sectors/request (=sectors.sum/requests.sum; ideal 4, >8 = bad coalescing), store efficiency `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio` (<16 = sparse writes), register spill (`smsp__sass_inst_executed_op_local_{ld,st}.sum > 0` = VERY BAD).

**Decision logic for bottleneck class** (cross-dimension synthesis):
- DRAM %peak high (~80-100%) → **DRAM-bandwidth-bound** → cut bytes / cache in smem.
- DRAM %peak <10% AND long_scoreboard stalls dominate (>40% samples) → **latency-bound on L1/global** → coalesce, add ILP, raise occupancy, async copy/TMA.
- FMA pipe >50% & tensor 0% on matmul → **compute-bound but off tensor cores** → use wgmma/MMA.
- math_pipe_throttle high → legit **compute-bound** (you're doing well).
- barrier >20% → **sync overhead / divergence**.
- The "one-line read" is the deliverable: "runs at X% peak SM, Y% peak DRAM; stalls dominated by <reason> (Z% samples) on <N> source lines whose access is <coalesced/not>; timeline is <flat/tail/sawtooth>; tensor cores <used/0%>."

Diagnosis playbook = 14 patterns A–N, each: Signals (exact metric thresholds) → Why → First-line fix → Deeper fixes → Exceptions → Blackwell cross-ref. Patterns most relevant to FA3-on-Hopper:
- A small grid/SM idle (decode batch=1 → split-K over KV is the standard fix).
- B tail effect on variable seqlen (split long seqs across CTAs; chunkwise).
- E latency-bound long_scoreboard (unroll loads, more warps, cp.async/TMA, software pipelining).
- F compute-bound off-tensor-cores (wgmma).
- K register spill (`__launch_bounds__`; FA accepts some spill).
- M pipeline bubbles (double-buffer; multi-stage TMA pipeline).
Ranking: use NCU rule-engine `Est. Speedup: X%` to rank; "fix the 50% problem before the 5% problem"; 3-5 priorities max.

### CUDAForge: LLM Judge does it, fed only the 24-metric subset + GPU spec
Judge cross-references the 24 NCU metrics against static GPU spec and labels ONE dominant bottleneck (memory-bound / compute-bound / occupancy-limited). Example heuristic stated in paper: "memory-bound when memory throughput high but compute utilization low → pick memory-related metrics this round." The Judge is forced to pick only 3-4 of the 24 metrics per round to justify its single chosen bottleneck.

---

## 4. WHAT THE CONDENSED FEEDBACK LOOKS LIKE (the actual artifact)

### CUDAForge — the gold standard for "concise actionable per-op feedback"
Judge output is STRICT JSON, three fields, hard word caps:
```json
{ "bottleneck": "<max 30 words>",
  "optimisation method": "<max 35 words>",
  "modification plan": "<max 35 words>" }
```
Prompt rules: "identify EXACTLY ONE highest-impact bottleneck by 3-4 most important metrics, propose EXACTLY ONE optimisation, be surgical and metrics-driven. Return one and only one method — the largest expected speedup. Prefer changes that address measured bottlenecks (occupancy limits, coalescing, smem bank conflicts, register pressure, long/short scoreboard stalls, tensor-core underutilisation). Keep fields brief; avoid lists of alternatives, disclaimers, generic advice."
Real subset-fed output (correct): `{"bottleneck":"Kernel is DRAM-bound (102.9% peak); 57% long-scoreboard stalls from second pass reading original_x", "optimisation method":"Fuse the second loop: cache each thread's original_x in registers during diff accumulation, then add gelu_val and store, eliminating extra B×F global read", "modification plan":"Replace two for-loops with one... ~4MB traffic per batch, >30% speedup"}` — cites the actual metric VALUES (102.9%, 57%), names the exact code construct (second loop over original_x), gives one concrete fix with quantified payoff.

### ncu-report-skill — longer human report, but same "evidence-cited, ranked" discipline
`07-report-template.md`: headline table (duration, SM%, DRAM%, L1/L2 hit, tensor%, regs, occupancy, waves/SM), one-line read (the punchline), per-dimension analysis (each finding cites metric+value), ranked optimization priorities:
```
Priority N — <name>
  Evidence: <metric value(s)> + <NCU rule + Est.Speedup X%>
  Expected impact: <X% end-to-end>
  Effort: <low/med/high>
```
Style rules: cite specific metric VALUES for every claim ("SM throughput 14%" not "low"); name file:line; use NCU's own Est.Speedup to rank; ≤5 priorities.

**For our agent**: emit BOTH — a machine-parseable terse JSON (CUDAForge style, for the loop/checkpoint) AND a short evidence-cited ranked list (skill style). The JSON drives the next code-gen round; the list is the human-readable distilled knowledge entry.

---

## 5. THE METRIC-SUBSET DISCIPLINE (critical, validated by ablation)

CUDAForge's central finding: **feeding the FULL NCU metric set to the LLM hurts.** Ablation (Table main_results): full-metrics variant = 1.280× median / 80% Fast@1; 24-subset = better median+perf, and ~26.5 min vs ~40 min per kernel, $0.3 vs $1 API cost. Case study (Level-2 task 51): full-metrics Judge MISIDENTIFIED the bottleneck (blamed gemm_out reload, recommended a hard-to-realize monolithic CUTLASS epilogue); subset Judge correctly found a DRAM-bound second-pass read and proposed a lightweight one-pass register-caching rewrite (>30% speedup). Reason: too many partially-redundant signals overwhelm the model → unstable, wrong diagnoses.

How the subset was chosen (offline, before the loop):
- Step 1: for each representative task (Conv2D, MatMul, SpMM...), run 100 self-refine cycles with one SOTA model, collect correct kernels, pick the 10 with largest fastest-vs-slowest runtime spread.
- Step 2: profile all 10, remove aliases/collinear metrics, compute Pearson r(metric, runtime), keep Top-20 by |r| per task.
- Step 3: across tasks keep metrics that recur, keep same sign, and have global score (mean |r| across tasks) > 75th percentile. → final 24 metrics.

**The 24-metric subset (CUDAForge, verify names for Hopper/ncu2025.3):**
1 sm__cycles_active.avg
2 sm__warps_active.avg.pct_of_peak_sustained_active
3 launch__occupancy_limit_blocks
4 launch__occupancy_limit_registers
5 launch__occupancy_limit_shared_mem
6 launch__registers_per_thread
7 sm__inst_executed.sum
8 sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active
9 sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active   ← tensor-core util (key for FA3)
10 dram__bytes_read.sum
11 dram__bytes_write.sum
12 dram__throughput.avg.pct_of_peak_sustained_elapsed
13 dram__bytes.sum.per_second
14 gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
15 l1tex__t_sector_hit_rate.pct
16 l1tex__throughput.avg.pct_of_peak_sustained_active
17 lts__t_sector_hit_rate.pct
18 lts__throughput.avg.pct_of_peak_sustained_active
19 smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct
20 smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
21 smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
22 smsp__warp_issue_stalled_barrier_per_warp_active.pct
23 smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct
24 smsp__sass_average_branch_targets_threads_uniform.pct

Top-20 example (Conv2D) most-correlated metrics included: sm__cycles_active.avg, gpc__cycles_elapsed.max, launch__occupancy_limit_shared_mem, dram__bytes.sum.per_second (neg corr — higher BW = faster), smsp__inst_executed.{avg,sum}, lts__t_sector_hit_rate.pct, launch__waves_per_multiprocessor, launch__shared_mem_per_block_static.

**ncu-report-skill's curated set** is bigger (~90 in `B200_KEY_METRICS`) but organized by category (launch geometry, SOL, occupancy, IPC, compute pipes, tensor, DRAM, caches, mem-inst counts, sectors/requests, stall ratios, per-PC pcsamp). It's a *parse/archive* set, not what's fed to an LLM. For our agent: parse the ~90, but DISTILL down to a CUDAForge-style ~24 (+ a few FA3-specific: tensor pipe, wgmma, TMA/cp.async, smem-per-block, register spill) before feeding the model.

---

## 6. REUSABLE HELPER SCRIPTS (directly adaptable)

All under `reference/related-works/kernel-design-agents/skills/ncu-report-skill/helpers/`:
- **`ncu_utils.py`** — THE most reusable. ncu_report locator+import, `safe`, `safe_many`, `metric_or_none`, `metric_value_at`, `per_instance_values`, `dump_all_metrics`, `per_pc_values`, `pc_to_source_line`, the ~90-metric `B200_KEY_METRICS` list, and `rule_results`/`rule_speedups` (sorted `(est_speedup_pct, rule_name, message)` from `action.rule_results_as_dicts()`). Adapt metric names to Hopper.
- **`analyze_reports.py`** — loads report(s), writes `metrics_all_<tag>.json` (archive), `metrics_key_<tag>.{json,txt}` (curated), and `compare_<t1>_vs_<t2>.txt` side-by-side. CLI: `--run-dir --report (repeatable) --tag (repeatable)`. Perfect for our baseline-vs-candidate A/B and for the evolution plot.
- **`extract_stall_hotspots.py`** — aggregates per-PC pcsamp stalls → per-(file,line), ranks top-N by total stall samples with per-stall-type breakdown, plus per-stall top-10 lines. Output `stall_hotspots_<tag>.txt`. This is the per-line actionable layer. Needs `--set source` + `-lineinfo`.
- **`plot_timeline.py`** — ASCII PM-sampling timelines (default = stall-reason series, which populate more reliably than sm/dram throughput pmsampling). Reveals tail effect / sawtooth visually.
- **`harness_template.cu`** + **`safetensors_loader.h`** — standalone, dep-free harness pattern (load real tensors from safetensors, run kernel in isolation, compile with `-lineinfo` in seconds). Aligns with our "single .cu, no external libs" goal — we can profile our distilled kernel standalone.
- **`list_flashinfer_workloads.py`** — browse flashinfer-trace dataset shape histograms to pick representative shapes (relevant: we need representative vLLM attention shapes).

---

## 7. CONCRETE RECOMMENDATIONS FOR OUR §2.1 PARSER

1. Collect with the two-pass recipe: `--set full --section PmSampling --section PmSampling_WarpStates` then `--set source --section SourceCounters`. Always compile our distilled .cu with `-lineinfo` so per-line stall attribution works (and `--page details` for the rule engine).
2. Parse via `ncu_report` Python API (reuse `ncu_utils.py`), NOT CSV/eyeball. Archive `metrics_all.json` once per candidate for checkpoint/resume.
3. Auto-discover Hopper metric names: `ncu --query-metrics --chip gh200` once, and verify each name with `safe()`/`metric_or_none()`. Don't trust B200 or CUDAForge names blindly.
4. Bottleneck classifier: implement the cross-dimension decision tree (DRAM%peak high→BW-bound; DRAM%peak<10% + long_scoreboard>40%→latency-bound; FMA>50% & tensor=0%→off-tensor-core; tensor>50%→good). Pull `rule_speedups()` and rank by Est.Speedup.
5. Emit feedback in CUDAForge's terse contract: ONE bottleneck (with 3-4 cited metric VALUES) + ONE fix + plan, AND a ranked ≤5-priority list with file:line evidence. Feed only a ~24-30 metric subset to the LLM, never the full 2000.
6. For FA3 specifically, add to the subset: `sm__pipe_tensor_cycles_active.*.pct_of_peak_sustained_elapsed` (wgmma activity — must be high), TMA/`cp.async` related (mio/lg throttle, async copy stalls), `launch__shared_mem_per_block`, register spill (`smsp__sass_inst_executed_op_local_{ld,st}.sum`), and the tail-effect signals for mixed prefill+decode variable seqlen.
7. Per-line hotspots (`extract_stall_hotspots.py` adapted) give the load-bearing actionable detail: "line L of kernel.cu is N% of long_scoreboard samples" → maps NCU signal to an exact code location for the next code-gen round.
8. Lessons to bake in: full-metric dumps cause misdiagnosis (CUDAForge ablation); the NCU rule engine's Est.Speedup is "shockingly accurate" — read `--page details` first; don't add Est.Speedups (rules overlap); _elapsed normalization is more honest than _active for under-utilized kernels; short kernels (<20µs) give noisy PM sampling.

## 8. GOTCHAS (env-specific)
- ERR_NVGPUCTRPERM → sudo ncu, or set NVreg_RestrictProfilingToAdminUsers=0. Likely already 0 on our dedicated H200 box.
- Set writable `HOME` to silence "Could not deploy stock section files".
- `-k "regex:..."` must match the DEMANGLED name; check with `cuobjdump --dump-function-names`. Our kernel name will be FA3-derived templated → escape carefully.
- Metric returns None: wrong name for this GPU/ncu, or section not collected. Always `safe()`.
- PM sampling can return empty arrays for sm/dram throughput series; stall-reason pmsampling series are the reliable proxy — check `m.num_instances()>0`.
- Don't compile with `-G` (debug) for perf profiling — ~100× regression. Use `-lineinfo` only.
- `--set full` is slow (45+ replays); for fast re-checks after a code change use Recipe 5 targeted `--metrics`.
