# NCU + Nsys Profiling Cookbook (for the FA3-distill agent's profiling component 2.1)

Environment-verified on this box (run these, do not guess):
- `ncu` = `/usr/local/bin/ncu` -> `/opt/nvidia/nsight-compute/2025.3.1/ncu`, **Version 2025.3.1.0**
- `nsys` = `/usr/local/cuda/bin/nsys`, **Nsight Systems 2025.5.1**
- `compute-sanitizer`, `cuobjdump`, `nvdisasm`, `nvcc` all in `/usr/local/cuda/bin`
- Python report module: `/opt/nvidia/nsight-compute/2025.3.1/extras/python/` (contains `ncu_report.py` + `_ncu_report.so`, also `ncu_occupancy.py`, `ForumSearchTool.py`). Add this dir to `PYTHONPATH`.
- Section files: `/opt/nvidia/nsight-compute/2025.3.1/sections/*.section`. Default search dir is `sections/`.
- Target GPU: H200, sm_90a. Profiling needs GPU perf-counter permission (ERR_NVGPUCTRPERM) — run as root or enable non-admin profiling; this box already had ncu working.

==========================================================================
## 0. The two tools, in one sentence each
- **ncu (Nsight Compute)**: deep, per-kernel HW-counter microarchitecture profiler. Replays the kernel many times. Gives SOL / memory / scheduler / occupancy / roofline / SASS-line counters. THIS is the workhorse for kernel optimization. High overhead (10x-1000x), serializes kernels.
- **nsys (Nsight Systems)**: low-overhead, whole-timeline tracer. Tells you kernel durations, launch overhead, gaps, API time, which kernel dominates, host/device overlap. NO HW counters by default. Use FIRST to find the hot kernel and to measure end-to-end latency; use ncu to dig into the hot kernel.

==========================================================================
## 1. AUTHORITATIVE sets & sections on THIS box (`ncu --list-sets`, `--list-sections`)

### Sets (`--set <id>`), with estimated #metrics (≈ replay-pass proxy):
| set        | sections included                                                                 | est metrics |
|------------|-----------------------------------------------------------------------------------|-------------|
| basic      | LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution                         | 213  (default; no `--set` => basic) |
| detailed   | + ComputeWorkloadAnalysis, MemoryWorkloadAnalysis(+Chart), SourceCounters, Roofline | 906 |
| full       | everything incl. InstructionStats, MemoryTables, SchedulerStats, WarpStateStats, PmSampling, all roofline | 7794 |
| roofline   | SpeedOfLight + all Hierarchical roofline charts + WorkloadDistribution             | 6649 |
| pmsampling | PmSampling, PmSampling_WarpStates                                                  | 411 |
| nvlink     | Nvlink, Nvlink_Tables, Nvlink_Topology                                            | 122 |

### Section identifiers (`--section <id>`), enabled-by-default marked *:
- `LaunchStats`* (Launch Statistics) — grid/block/regs/smem, waves/partial wave
- `Occupancy`* (Occupancy) — theoretical vs achieved occupancy, limiters
- `SpeedOfLight`* (GPU Speed Of Light Throughput) — top-level Compute% vs Memory% of peak
- `WorkloadDistribution`* (GPU and Memory Workload Distribution)
- `ComputeWorkloadAnalysis` — IPC, pipe utilization (FMA/ALU/Tensor/LSU...)
- `MemoryWorkloadAnalysis` (+ `_Chart`, `_Tables`) — DRAM/L2/L1 throughput, hit rates, bank conflicts, sectors/req
- `SchedulerStats` (Scheduler Statistics) — issued/active/eligible warps per scheduler, issue slot utilization
- `WarpStateStats` (Warp State Statistics) — avg cycles per instr in each stall state (latency-hiding view)
- `SourceCounters` (Source Counters) — per-SASS-line warp-stall sampling + branch efficiency (needs `--page source` to view per-line)
- `InstructionStats` (Instruction Statistics) — SASS instruction mix per opcode
- `SpeedOfLight_RooflineChart`, `SpeedOfLight_Hierarchical{Double,Half,Single,Tensor}RooflineChart`
- `PmSampling`, `PmSampling_WarpStates` — timeline sampling over kernel runtime
- `Nvlink*`, `NumaAffinity`, `C2CLink` — irrelevant for single-GPU FA3.

NOTE: section *identifiers* (e.g. `SchedulerStats`) differ from filenames (`SchedulerStatistics.section`). Use the identifier with `--section`.

==========================================================================
## 2. ncu: (a) profile ONE kernel with LOW overhead

Three independent levers to bound overhead/time:
1. **Filter to one kernel** (most important): `-k`, `--kernel-id`, `-s/-c`.
2. **Collect fewer metrics**: pick a small `--set`/`--section`/`--metrics`.
3. **Cut replay count / passes** and serialization knobs.

### Kernel name filter (`-k` / `--kernel-name`)
- `-k flash_fwd` matches by regex on (mangled by default) kernel name.
- Force base name space: `--kernel-name-base demangled` (or `mangled`/`function`). For C++ FA3 kernels, demangled regex is friendlier: `--kernel-name-base demangled -k "regex:.*flash.*fwd.*"`.
- `--kernel-id "ctx:strm:name:invocation"` filters by context/stream/name/invocation index — use to grab exactly one launch.

### Instance count limiting (huge overhead win)
- `-c N` / `--launch-count N`: profile only first N matching launches.
- `-s M` / `--launch-skip M`: skip first M matching launches (e.g. skip warmup).
- `--launch-skip-before-match`: skip applies before name match.
- Canonical "one kernel, one instance":
```
ncu -k "regex:flash.*fwd" -s 2 -c 1 -o prof_one python repro.py
```
(skip 2 warmups, capture the 3rd launch only.)

### Replay mode (`--replay-mode kernel|application|range`)
- `kernel` (default): replays just the kernel; saves/restores device memory between passes. Best for an isolated kernel benchmark. If only 1 pass is needed, NO save/restore is done (lowest overhead).
- `application`: reruns whole app once per pass (good if kernel has host interdependencies / hangs under kernel replay; lets you disable cache control). Cost = app_runtime x passes.
- `range`: captures a CUDA API range (cu(da)ProfilerStart/Stop or NVTX) and replays it — needed when kernels MUST run concurrently. Driver-API only, restricted API set.
- Compatibility: plain CUDA kernels support kernel/application/range. CUDA-graph-launched kernels support **kernel replay only** and **no instruction-level source** metrics.

### Overhead drivers (from §2.2.6) — minimize these:
- # and TYPE of metrics. SW-patched (source/instruction) metrics = highest overhead, collected in separate passes. HW counters = low. Launch/occupancy attrs = ~free, no replay.
- More metrics => more passes => more memory save/restore. There is a one-time per-context setup cost for the first profiled kernel.
- So: a `basic`/SpeedOfLight-only profile of 1 kernel instance is the cheapest meaningful profile.

### Reproducibility / variance control (keep results stable AND comparable):
- `--clock-control base` (default): lock clocks to base. For matching baseline-vs-candidate latency you want IDENTICAL clock policy. Use `--clock-control none` only if an external tool locks clocks.
- `--cache-control all` (default): flush L1/L2 before each replay pass (deterministic counters). `--cache-control none` to keep caches warm (realistic in-app behavior).
- ncu **serializes** kernel launches (within and across processes) via lock files in `$TMPDIR/nvidia/nsight_compute/`. Only ONE process can profile a given device at a time — relevant for the agent's multi-GPU scheduler: assign one ncu run per GPU, set `CUDA_VISIBLE_DEVICES` per worker; lock is per-device so different GPUs profile concurrently.
- Enable persistence mode for stable startup behavior.

==========================================================================
## 3. ncu: (b) MACHINE-PARSEABLE output

### Store report, then parse offline (RECOMMENDED for the agent):
```
ncu -f -o /tmp/prof_cand --set detailed -k "regex:flash.*fwd" -s 2 -c 1 python repro.py
# -f = force overwrite; produces /tmp/prof_cand.ncu-rep
```
Re-read without re-profiling using `--import`:
```
ncu --import /tmp/prof_cand.ncu-rep --page raw --csv --log-file /tmp/prof_cand.csv
```

### CSV to a file (best for a parser):
- `--csv` makes ALL console output comma-separated.
- `--log-file <path>` writes console output (the page) to a file. Combine with `--csv`.
- `--page details|raw|source` chooses what to print:
  - `details`: per-section 3-col tables + RULE results (the green/yellow advice). Add `--details-all` to print every metric, not just header metrics.
  - `raw`: every collected metric as name/unit/value (incl. device attrs, launch info). Best for a stable key=value parser; no unresolved regex:/group:/breakdown: names.
  - `source`: per-SASS/PTX/CUDA-C line counters (needs SourceCounters section).
- IMPORTANT: if `--export`/`-o` is used and `--page` is NOT given, NOTHING prints to console. To both save and print: pass `--page` explicitly.

### Canonical agent recipes:
```
# A) Save full report (parse later via Python API or --import)
ncu -f -o REP --set detailed -k "regex:KERN" -s S -c 1 APP

# B) Get a flat CSV of every metric for one kernel, no UI needed
ncu --csv --page raw --log-file metrics.csv -f -o REP --set detailed -k "regex:KERN" -c 1 APP

# C) Get the human/rule advice as CSV (the bottleneck verdicts)
ncu --csv --page details --log-file details.csv -f -o REP --set detailed -k "regex:KERN" -c 1 APP
```

### Pull specific metrics only (fewest passes):
```
ncu --csv --log-file m.csv \
  --metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem \
  -k "regex:KERN" -c 1 APP
```
- `--query-metrics` lists all metrics for the chip; `--query-metrics-mode suffix --metrics <base>` shows the valid suffixes.
- `--list-metrics` lists metrics that WILL be collected for current section selection.

### Metric naming convention (so the parser knows what suffix to ask for):
- Structure: `<unit>__<counter>.<rollup>.<submetric>`, e.g. `sm__throughput.avg.pct_of_peak_sustained_elapsed`.
- Counter rollups: `.sum .avg .min .max`. Then sub-metrics: `.per_second`, `.per_cycle_active/elapsed`, `.pct_of_peak_sustained_active`, `.pct_of_peak_sustained_elapsed`, `.peak_sustained...`.
- Ratios: `.pct .ratio .max_rate`. Throughputs: `.pct_of_peak_sustained_active/elapsed` (+ `breakdown:<throughput>` to expand contributors).
- `group:<name>` and `breakdown:<throughput>` are macro selectors usable with `--metrics`.

### Python Report Interface (the cleanest parser path — no CSV scraping):
```python
import sys; sys.path.append("/opt/nvidia/nsight-compute/2025.3.1/extras/python")
import ncu_report
ctx = ncu_report.load_report("/tmp/prof_cand.ncu-rep")
for ri in range(ctx.num_ranges()):
    rng = ctx.range_by_idx(ri)
    for ai in range(rng.num_actions()):
        a = rng.action_by_idx(ai)
        name = a.name()
        dur  = a["gpu__time_duration.sum"].value()          # ns
        sol_c= a["sm__throughput.avg.pct_of_peak_sustained_elapsed"].value()
        sol_m= a["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"].value()
        # a.metric_names() -> tuple of all metric names; a[name].value() auto-types
```
- `IAction.metric_by_name` / `a[name]` -> `IMetric`; `.value()` auto-types, `.as_double/.as_uint64/.as_string`, `.unit()`, `.description()`.
- `IMetric.metric_type()` => COUNTER/RATIO/THROUGHPUT/OTHER; `metric_subtype()`, `rollup_operation()` (SUM/AVG/MIN/MAX) for filtering.
- Works on any Python 3.7+. This is how the agent should build CONCISE feedback: load the .ncu-rep once, pull a handful of named metrics + the rule results, emit a few-line verdict.

==========================================================================
## 4. ncu: (c) The specific analyses the agent needs (exact `--section`/metrics)

### Speed-of-Light (top-level bottleneck) — section `SpeedOfLight`
Key metrics:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`  (Compute SOL %)
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`  (Memory SOL %)
- `gpu__time_duration.sum`  (kernel duration, ns) — also `gpu__time_active`
- `sm__cycles_elapsed.avg`, `gpc__cycles_elapsed.max`
Verdict logic: high Compute% + low Mem% => compute-bound; reverse => memory-bound; both low => latency/occupancy-bound (look at scheduler/warp-state).

### Memory Workload Analysis — section `MemoryWorkloadAnalysis` (+ `_Tables` for bank conflicts)
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`  (HBM BW %)
- `dram__bytes.sum`, `dram__bytes_read.sum`, `dram__bytes_write.sum`
- `lts__t_sector_hit_rate.pct`  (L2 hit rate); `l1tex__t_sector_hit_rate.pct` (L1)
- `l1tex__data_bank_conflicts_pipe_lsu.sum`  (shared-mem bank conflicts — critical for FA3 smem tiling)
- `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` (coalescing: sectors/request, lower=better)
- `--metrics group:memory__chart` reproduces the Memory Chart numbers.

### Scheduler / Warp-state stall sampling — sections `SchedulerStats` + `WarpStateStats`
- `smsp__issue_active.avg.pct_of_peak_sustained_active` (issue-slot utilization; <~50% => poor latency hiding)
- `smsp__warps_active.avg`, `smsp__warps_eligible.avg.per_cycle_active`, fraction of cycles with NO eligible warp.
- WarpStateStats: avg cycles/instr per stall reason, e.g.
  - `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` (global/L2 memory latency)
  - `..._short_scoreboard...` (shared-mem / MIO latency)
  - `..._barrier...` (`__syncthreads`/named-barrier waits — common in FA3 producer/consumer)
  - `..._mio_throttle...`, `..._lg_throttle...`, `..._wait...`, `..._not_selected...`, `..._math_pipe_throttle...`
  Doc rule: only act on stalls if schedulers fail to issue every cycle.

### Occupancy — section `Occupancy`
- `sm__warps_active.avg.pct_of_peak_sustained_active` (ACHIEVED occupancy %)
- `launch__occupancy_limit_registers`, `launch__occupancy_limit_shared_mem`, `launch__occupancy_limit_blocks`, `launch__occupancy_limit_warps` (which resource caps occupancy)
- `launch__registers_per_thread`, `launch__shared_mem_per_block_static/dynamic`, `launch__waves_per_multiprocessor`
- Big theoretical-vs-achieved gap => imbalanced workload / tail effect.

### Roofline — set `roofline` or sections `SpeedOfLight_RooflineChart` (+ Hierarchical{Half,Tensor}RooflineChart for FP16/Tensor)
- Vertical=FLOP/s, horizontal=arithmetic intensity (FLOP/byte). For FA3 (FP16/BF16 + tensor cores) use the **Half** and **Tensor** hierarchical roofline sections.
- Use ncu **baselines** to overlay candidate-vs-original on the same roofline (track optimization progress). Roofline values come from `--set full` / `--section ...RooflineChart` / `--metrics group:memory__chart` plus the SM FLOP metrics.

### SOURCE / SASS per-line counters — section `SourceCounters`, view with `--page source`
```
ncu -f -o REP --section SourceCounters --set detailed \
    -k "regex:KERN" -c 1 APP
ncu --import REP.ncu-rep --page source --csv --log-file source.csv
```
- Per-line metrics: `inst_executed`, `thread_inst_executed`, predication, branch efficiency, and **warp-stall sampling** correlated to SASS lines (which line stalls and why).
- Compile target with `-lineinfo` (nvcc) so SASS lines map back to CUDA-C; otherwise you get SASS/PTX only. (For FA3 distillation, `-lineinfo` lets ncu point at the exact line of the inner mma/copy loop that stalls.)
- Source/instruction-level metrics are SW-patched (highest overhead) and NOT available for graph-launched kernels.

==========================================================================
## 5. ncu: filtering, replay, and bounding overhead — quick reference
- Kernel filter: `-k <regex>` (+ `--kernel-name-base demangled`), `--kernel-id`, `-s SKIP`, `-c COUNT`.
- NVTX filter: `--nvtx --nvtx-include "Domain@Range/"` to profile only kernels inside an NVTX range (wrap the FA3 call in vLLM under an NVTX push/pop and target it). Push/Pop uses trailing `/`; Start/End has no `/`.
- Replay: `--replay-mode kernel` (default, lowest if 1 pass). Use `range` only for mandatory-concurrent kernels.
- Reduce passes: smaller `--set`, explicit `--metrics`, single section. Each section roughly adds passes; SourceCounters/InstructionStats add the most.
- `--profile-from-start off` + `cudaProfilerStart/Stop` (or `--nvtx-include`) to skip everything before the region of interest.
- `--config-file` (`config.ncu-cfg`, INI per mode) to keep invariant flags out of the command line.
- `--kill yes` to terminate the process after capturing the requested launches (don't run the rest of the app).
- Multi-GPU agent: per-GPU `CUDA_VISIBLE_DEVICES=k`, per-GPU lock is automatic; set distinct `-o` per worker. Set `TMPDIR` per worker if lock-file contention/permission issues arise.

==========================================================================
## 6. nsys: timeline + cuda-gpu-trace/summary (overhead-bounded)

### Capture a timeline (low overhead, this is the latency/where-is-time tool):
```
nsys profile -t cuda,nvtx -s none --cpuctxsw=none \
   -o /tmp/tl_cand -f true python repro.py
```
- `-t/--trace cuda,nvtx` (drop osrt to cut overhead). Add `cublas,cudnn` only if needed — but our .cu has NO libs so `cuda,nvtx` is enough.
- `-s none` (no CPU IP sampling) and `--cpuctxsw=none` cut overhead significantly.
- `-d 20` (`--duration`) bounds wall time; `-y/--delay N` to skip warmup; `-c cudaProfilerApi` or `-c nvtx -p MSG@DOMAIN` to start/stop on a marker (profile only the FA3 region).
- `-x true` / `--stop-on-exit`, `--kill none` to keep app alive. Keep runs < ~5 min (nsys is not meant for longer).
- `--gpu-metrics-devices=0` adds sampled GPU HW metrics to the timeline (SM/DRAM utilization over time) — heavier; optional.
- On Blackwell+ there's hardware-based CUDA trace (lower overhead for many short kernels) — N/A on H200 (Hopper), so instrumented trace is used.

### Get machine-parseable stats (no GUI). Two ways:
1. During capture: `nsys profile --stats=true ...` prints default summary tables.
2. Post-process a `.nsys-rep` (preferred; repeatable):
```
# Per-launch GPU kernel TRACE (every kernel instance, start/dur/grid/stream)
nsys stats --report cuda_gpu_trace --format csv --output . /tmp/tl_cand.nsys-rep
#   -> /tmp/tl_cand_cuda_gpu_trace.csv

# Kernel SUMMARY (per-kernel total/avg/%), API summary, mem-op summary
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
   --format csv --output . /tmp/tl_cand.nsys-rep
```
- `--report` names: `cuda_gpu_trace` (per-instance), `cuda_gpu_kern_sum` (per-kernel summary), `cuda_api_sum` (host API time = launch overhead), `cuda_gpu_mem_time_sum`/`cuda_gpu_mem_size_sum`. `nsys stats --help-reports` lists all; default report set runs many `_sum` scripts.
- `--format csv,column`, `--output .,-` lets you fan reports to file+console; lists are zip-expanded (1st report->1st format->1st output, then last repeats).
- `--timeunit usec` to normalize units. `--force-export` to refresh the SQLite. `--sqlite f.sqlite` to control the export name.
- nsys auto-creates a sibling `.sqlite`; you can also `nsys export --type sqlite ...` and query directly (schema documented). Good for the agent to compute "kernel duration delta vs baseline" without parsing CSV.
- `nsys analyze report.nsys-rep` runs expert-system rules (`gpu_time_util`, `gpu_gaps`, `cuda_api_sync`, `cuda_memcpy_async/sync`...) -> concise CSV/console verdicts; `--rule <name>` to pick.

### Agent usage pattern:
1. `nsys profile -t cuda,nvtx -s none -o tl APP` then `nsys stats --report cuda_gpu_kern_sum --format csv -o . tl.nsys-rep` to find the dominant kernel + its total/avg time, and `cuda_api_sum` to see if launch overhead dominates (decode!).
2. Feed the kernel name into `ncu -k <name> -c 1` for microarchitecture detail.
3. For our latency requirement, nsys `cuda_gpu_trace`/`cuda_gpu_kern_sum` gives device-side per-kernel time; but for the official latency gate use CUDA events around the op (nsys/ncu both add overhead and serialize). Use nsys to *attribute* time, not to certify the final latency number.

==========================================================================
## 7. Gotchas / pitfalls the agent must encode
- ncu replays kernels => measured ncu "duration" is fine for relative analysis but ncu/nsys both perturb timing; CERTIFY latency with CUDA events on an un-profiled run, with locked clocks, warmups discarded.
- `--export -o` WITHOUT `--page` prints nothing to console (silent). Always add `--page` if you want stdout.
- Section IDENTIFIER (`SchedulerStats`) != filename (`SchedulerStatistics.section`).
- CUDA-graph-launched kernels: kernel-replay only, NO source/instruction-level metrics. vLLM uses CUDA graphs for decode — may need `--replay-mode kernel`, and source-line stalls may be unavailable for graph-captured launches; profile prefill (eager) for SASS line data, or disable graphs in the repro.
- `n/a` metric => typo, wrong suffix, or metric not on this chip. Validate via `--query-metrics`.
- Permission: ERR_NVGPUCTRPERM => need perf-counter access (root / enable non-admin). Instruction-level source metrics via CLI do NOT need device profiling permission.
- Serialization lock is per-device in `$TMPDIR/nvidia/nsight_compute/` and is NOT auto-deleted; cross-user permission can fail — set per-worker `TMPDIR`. Only one ncu per device at a time (matters for the multi-GPU perf scheduler: 1 perf kernel/GPU).
- DCGM or another profiler holding the perf monitor => "driver resource unavailable"; `dcgmi profile --pause`.
- Keep nsys runs < 5 min and bound with `-d`; system-wide sampling explodes symbol-resolution time.
- For comparing candidate vs baseline: use IDENTICAL `--clock-control`, `--cache-control`, set, kernel filter, and instance index, and ncu **baselines** (UI overlay) or just diff named metrics from two `.ncu-rep` via the Python API.

==========================================================================
## 8. Copy-paste command bank (final)
```
# --- discover ---
ncu --list-sets ; ncu --list-sections ; ncu --query-metrics | less
nsys stats --help-reports

# --- nsys: find hot kernel + launch overhead (cheap) ---
nsys profile -t cuda,nvtx -s none --cpuctxsw=none -f true -o tl python repro.py
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_trace \
  --format csv --output . tl.nsys-rep

# --- ncu: cheap top-level SOL on exactly one launch ---
ncu -f -o sol --set basic -k "regex:flash.*fwd" --kernel-name-base demangled \
  -s 2 -c 1 python repro.py
ncu --import sol.ncu-rep --page details --csv --log-file sol_details.csv

# --- ncu: detailed (SOL+Mem+Compute+Occupancy+Source+Roofline), 1 launch ---
ncu -f -o det --set detailed -k "regex:flash.*fwd" -s 2 -c 1 python repro.py

# --- ncu: stalls (scheduler + warp state) ---
ncu -f -o stall --section SchedulerStats --section WarpStateStats \
  -k "regex:flash.*fwd" -c 1 python repro.py

# --- ncu: per-SASS-line stalls (compile with -lineinfo) ---
ncu -f -o src --section SourceCounters --set detailed -k "regex:flash.*fwd" -c 1 python repro.py
ncu --import src.ncu-rep --page source --csv --log-file src_lines.csv

# --- ncu: just the numbers I parse, minimal passes ---
ncu --csv --log-file core.csv -k "regex:flash.*fwd" -c 1 \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu.sum,lts__t_sector_hit_rate.pct \
  python repro.py

# --- parse a report in Python (concise verdict) ---
PYTHONPATH=/opt/nvidia/nsight-compute/2025.3.1/extras/python python parse.py det.ncu-rep
```
