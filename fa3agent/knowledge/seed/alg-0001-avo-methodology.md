---
id: alg-0001
type: alg
title: AVO agent-loop methodology - correctness-first, optimize-the-minimum, single-lineage
tags: [methodology, evolution, agent-loop, correctness-first, fitness, geomean]
confidence: source-reported
source: AVO (NVIDIA, NeurIPS 2026 preprint), .research/avo.md
---
AVO makes a coding agent THE variation operator: one autonomous loop owns
examine-lineage -> consult-knowledge -> edit -> test(f) -> diagnose -> repeat -> commit.

Rules that transfer directly to this agent:
- CORRECTNESS FIRST. Scoring is a vector f(x)=(f_1..f_n), one per config. If a
  candidate fails correctness vs reference, ALL f_j=0 regardless of speed. Only
  then does perf matter. (Our gate is stricter: greedy-decode token equivalence.)
- COMMIT ONLY ON >= best. A new version joins the committed lineage only if it
  passes correctness AND matches-or-improves the best running score. Failed
  intermediates stay in agent context (learn from them) but not in the lineage.
- RUNNING-BEST METRIC = geomean of per-config scores. We use the baseline/candidate
  latency RATIO since clocks are unlockable; re-measure baseline same session.
- OPTIMIZE THE MINIMUM: the worst config drags the geomean; spend effort there.
- Single-lineage continuous run de-risks; islands/branching are future work.
- ~12:1 explore:commit ratio expected (>500 attempts -> 40 commits over 7 days).
- Strategy emerges, not scripted: early = structural changes from reference
  kernels; late = micro-arch tuning driven by profiling feedback.

See [[alg-0002]] for the inner-step anatomy and anti-stall supervisor.
See [[alg-0005]] for the five concrete inflection optimizations AVO found.
HW MISMATCH: AVO=B200/Blackwell; we=H200/Hopper sm_90a. Use the concepts,
re-derive numbers (register budgets, tile sizes) via H200 profiling.
