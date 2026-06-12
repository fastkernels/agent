---
id: ks-0008
type: ks
title: Warp-specialization roles on Hopper (4-warp warpgroup, accumulators in registers)
tags: [warp-specialization, producer-consumer, wgmma, tma, mbarrier, named-barrier, hopper, sm90a]
confidence: source-reported
source: reference/related-works/.../KernelWiki warp-specialization (sm90 subset), distilled
---
Warp specialization assigns distinct roles to warps in the CTA so each focuses on one pipeline
stage, cutting per-warp register need and role-switching stalls. The sm_90a structure (DIFFERENT
from Blackwell -- do not copy the 16-warp / tcgen05 / TMEM model):
- wgmma.mma_async is a 4-warp WARPGROUP (128 threads) collective instruction, and its accumulators
  live in REGISTERS shared across the warpgroup (no TMEM on Hopper; see [[ks-0007]]).
- Typical layout: a small PRODUCER warpgroup issues cp.async.bulk.tensor (TMA) loads and signals
  an mbarrier; one or two CONSUMER warpgroups run wgmma + online softmax; optional epilogue.
- Coordinate producer<->consumer with mbarrier arrive/wait + phase parity (per [[pf-0002]]); order
  the two math warpgroups with NAMED barriers (bar.sync <id>), not a blanket __syncthreads.
- Use setmaxnreg to shift registers from the cheap producer (needs few) to the expensive math
  warpgroups (per [[ks-0004]]).
- Double/triple-buffer the SMEM tiles (>=3 stages) so loads overlap math; mind the 228KB dynamic
  SMEM budget (request >48KB via cudaFuncSetAttribute, see [[ks-0005]]).
When NOT to bother: tiny decode kernels that are memory-bound and occupancy-limited -- there the
win is split-KV + occupancy, not a producer/consumer split.
