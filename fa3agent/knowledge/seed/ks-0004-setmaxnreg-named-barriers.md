---
id: ks-0004
type: ks
title: setmaxnreg per-warpgroup register reallocation + named barriers
tags: [setmaxnreg, register, warp-specialization, named-barrier, hopper, sm90a, occupancy]
confidence: verified
source: PTX 9.7.20.5 + hopper-hw.md
---
Warp-specialization splits the CTA into a producer WG (TMA only) + consumer WG(s)
(wgmma + softmax). setmaxnreg moves registers from the cheap producer to the
expensive consumers - the lever AVO opt #5 tuned (see [[alg-0005]]).

setmaxnreg (PTX):
  setmaxnreg.dec.sync.aligned.u32 24;    // producer: release regs to CTA pool
  setmaxnreg.inc.sync.aligned.u32 240;   // consumer: blocks until pool has enough
- imm-reg-count in [24,256], MULTIPLE OF 8. ALL 4 warps of the WG execute the SAME
  setmaxnreg (uniform); sync after.
- Requires a valid max-reg via __launch_bounds__ / -maxrregcount, else it is a no-op.
- Typical FA3 split: 1 producer @ 24-32, N consumers @ 232-240. hd128 2-MMA-WG:
  load 24/40, mma 240/232. Profile-driven rebalance: if a consumer group spills to
  local mem (see [[pf-0001]]) while another has headroom, shift 8 regs.

Named barriers (intra-CTA sync beyond barrier 0):
  bar.sync id, threadcount;   /   bar.arrive id, count;     // id in [0,15], 16 HW barriers
- Used for ping-pong consumer scheduling (each consumer WG waits its own id, signals
  the next) and to separate producer/consumer phases without a full __syncthreads().

Required fences (common deadlock/corruption sources):
- fence.mbarrier_init.release.cluster after mbarrier.init, before first use.
- fence.proxy.async.shared::cta after STSM/TMA smem write, BEFORE wgmma reads it.
- Missing wgmma.fence before reading the accumulator => reorder => wrong results.
- Releasing the EMPTY barrier before wgmma.wait_group<0> => producer overwrites
  smem still read by tensor cores => corruption.
- Producer must drain (producer_tail) before exit, else the consumer hangs.

See [[ks-0005]] for the 228KB smem opt-in that pairs with this, [[alg-0003]] for the
overall warp-specialized structure.
