---
id: pf-0002
type: pf
title: Pipeline-stall / mbarrier diagnosis on Hopper (TMA <-> wgmma overlap)
tags: [pipeline-stalls, mbarrier, tma, wgmma, long-scoreboard, warp-stall, num-stages, sm90a]
confidence: source-reported
source: reference/related-works/.../KernelWiki pipeline-stalls (sm90 subset), distilled
---
Symptom: NCU shows tensor cores / TMA idle on a nominally compute-bound kernel, or a high
long-scoreboard / barrier stall %, with warps blocked on mbarrier.try_wait. The TMA load and the
wgmma math are not overlapping. Likely causes on sm_90a, in order to check:
1. Pipeline too shallow: 2 SMEM stages cannot hide TMA latency -- use 3-4 stages (double/triple
   buffer) so the producer fills stage s+1 while the consumer runs wgmma on stage s.
2. Wrong mbarrier phase-parity: each wait must flip the expected parity bit; a stale phase makes
   the consumer wait forever or read a stale buffer.
3. Missing fence between "TMA complete" and the wgmma read of that SMEM (the async proxy must be
   made visible: fence.proxy.async / the wgmma.fence before issuing).
4. Single-tile serialization: all warps stuck on one tile's softmax/epilogue -> split work so the
   next tile's loads start early.
5. Producer over-arrive: do NOT manually mbarrier.arrive after issuing cp.async.bulk.tensor (TMA);
   use mbarrier.arrive.expect_tx with the byte count and let the TMA hardware signal completion --
   a manual arrive races the async transfer and releases the consumer onto stale data.
Checklist: confirm tensor-core active cycles; inspect mbarrier wait stalls in the warp-state
breakdown; verify parity flips each wait; verify arrive.expect_tx (not manual arrive); try +1
NUM_STAGES. Watch the 228KB dynamic-SMEM budget -- too many stages stops fitting.
Profile FIRST: pipelining is wasted effort on a memory-bound kernel (see [[pf-0001]], [[ks-0002]]).
