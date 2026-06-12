---
id: ks-0007
type: ks
title: Register budgeting & spill mitigation on Hopper (no TMEM escape valve)
tags: [register-budgeting, register-spill, occupancy, launch_bounds, maxrregcount, hopper, sm90a]
confidence: source-reported
source: reference/related-works/.../KernelWiki register-budgeting/register-pressure (sm90 subset), distilled
---
SM occupancy is inversely proportional to registers/thread, and NCU "register spilling to local
memory" (LDL/STL, long-scoreboard stalls on local loads) means the compiler ran out. Levers on
sm_90a:
- `__launch_bounds__(threads, minBlocksPerSM)` or `-maxrregcount=N` caps the budget and raises
  occupancy; pick N from the occupancy you need (e.g. 2-3 CTAs/SM).
- IMPORTANT sm_90a caveat: Hopper has NO TMEM. wgmma accumulators live in REGISTERS (a 64xN f32
  tile costs ~tile/4 regs/thread per warpgroup), so you cannot move them off-register the way
  Blackwell tcgen05/TMEM does. Reduce accumulator pressure by shrinking block_m / the wgmma N tile,
  cutting live variables across branches, and recompute-vs-store for cheap values.
- Memory-bound DECODE (small M): spills can hide under memory latency, so aggressively budgeting
  for higher occupancy often still wins.
- Compute-bound PREFILL GEMM: do NOT force a budget so tight it spills hot inner-loop values -
  those spills serialize and cost more than the occupancy gained.
Diagnose first: if NCU shows LDL/STL > 0 and high long-scoreboard %, attack registers; if occupancy
is already high and tensor cores idle, the problem is the pipeline, not registers.
Pairs with [[ks-0004]] (setmaxnreg to rebalance producer vs math warpgroups) and [[pf-0001]].
