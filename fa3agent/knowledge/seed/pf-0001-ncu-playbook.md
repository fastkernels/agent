---
id: pf-0001
type: pf
title: NCU diagnosis playbook A-N + bottleneck decision tree + 24-metric discipline
tags: [ncu, profiling, bottleneck, decision-tree, metric-subset, stalls, tensor-core, spill]
confidence: source-reported
source: .research/ncu-skill.md (ncu-report-skill + CUDAForge)
---
Parse with the ncu_report Python module (NOT CSV/eyeball). Always wrap reads in
safe()/metric_or_none() - Hopper/ncu2025.3 names drift from B200/CUDAForge. Feed the
LLM only a ~24-30 metric SUBSET; the full 2000-metric dump CAUSES MISDIAGNOSIS
(CUDAForge ablation). Compile the .cu with -lineinfo (NOT -G) for per-line attribution.

BOTTLENECK DECISION TREE (cross-dimension synthesis):
- DRAM %peak ~80-100% -> DRAM-bandwidth-bound -> cut bytes / cache in smem.
- DRAM %peak <10% AND long_scoreboard stalls >40% samples -> latency-bound on
  L1/global -> coalesce, add ILP, raise occupancy, async copy/TMA.
- FMA pipe >50% AND tensor pipe ~0% on a matmul kernel -> compute-bound but OFF
  tensor cores -> use wgmma. (For FA3 the #1 metric is
  sm__pipe_tensor_cycles_active.*.pct_of_peak_sustained_elapsed - wgmma MUST be high.)
- math_pipe_throttle high -> legitimately compute-bound (doing well).
- barrier stall >20% -> sync overhead / divergence.
- register spill: smsp__sass_inst_executed_op_local_{ld,st}.sum > 0 = VERY BAD
  (drove AVO opt #5 rebalance, see [[alg-0005]]).
Rank fixes by NCU rule-engine Est.Speedup (shockingly accurate; read --page details
first; do NOT sum overlapping Est.Speedups). Fix the 50% problem before the 5%; <=5 priorities.

14-PATTERN PLAYBOOK A-N (each: signals->cause->fix). Most relevant to FA3-on-Hopper:
- A small grid / SM idle (decode batch=1) -> split-K over KV (FlashDecoding).
- B tail effect on variable seqlen (mixed prefill+decode) -> split long seqs / chunkwise.
- E latency-bound long_scoreboard -> unroll loads, more warps, cp.async/TMA, sw pipeline.
- F compute-bound off-tensor-cores -> wgmma.
- K register spill -> __launch_bounds__ / setmaxnreg ([[ks-0004]]); FA accepts some spill.
- M pipeline bubbles (sawtooth timeline) -> double-buffer / multi-stage TMA pipeline.

EMIT BOTH: a terse machine-parseable JSON (CUDAForge: one bottleneck <=30 words with
3-4 cited metric VALUES, one optimization <=35 words, one plan <=35 words) to drive the
next code-gen round, AND a ranked <=5-priority list with file:line evidence as the
human-readable distilled note. Cite VALUES (e.g. 'SM 14%', 'DRAM 102.9%'), not adjectives.

Gotchas: ERR_NVGPUCTRPERM -> profiling perm; set writable HOME; -k regex matches the
DEMANGLED name; short kernels (<20us) give noisy PM sampling; stall-reason pmsampling
series are more reliable than sm/dram throughput series.
