---
id: alg-0002
type: alg
title: AVO inner variation step + anti-stall supervisor (self-supervision)
tags: [methodology, agent-loop, supervisor, anti-stall, checkpoint, profiling-feedback]
confidence: source-reported
source: AVO sections 3-4, .research/avo.md
---
A single variation step is an autonomous agent loop producing x_{t+1} from the
lineage P_t. One step may do many internal actions:
1. Examine multiple prior implementations in P_t; compare their PROFILING
   characteristics to find bottlenecks/opportunities.
2. Consult knowledge K to understand HW constraints BEFORE implementing.
3. Implement one candidate optimization.
4. Invoke f (correctness then throughput).
5. On failure (incorrect OR no improvement): diagnose -> revise -> repeat the
   edit-evaluate-diagnose cycle within the same step.
6. Commit x_{t+1} only on pass-correctness AND >= best (see [[alg-0001]]).

Persistent memory = the agent's conversation history (compiler output, profiling
results, prior reasoning). Each committed version = a git commit annotated with
its score => checkpoint/resume is git-log replay.

Two long-horizon failure modes and the SUPERVISOR that fixes them:
- STALL: the agent exhausts its current line of exploration.
- UNPRODUCTIVE CYCLES: repeated edits that keep failing to improve.
A self-supervision watchdog detects these (N consecutive non-improving candidates,
or M wall-clock with no commit), reviews the whole trajectory, and steers toward
3-5 fresh candidate directions. This is also where in-loop distillation lands:
turn failed-attempt lessons into typed knowledge (ks/pf/alg) rather than discard.

Resist over-engineering the agent: AVO used a stock SWE agent + only K + f.
Invest in K quality and f reliability. See [[pf-0001]] for the profiling-feedback
signal that drives late-stage micro-opt.
