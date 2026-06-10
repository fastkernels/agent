---
id: ks-0002
type: ks
title: TMA cp.async.bulk.tensor + mbarrier expect-tx producer/consumer cycle
tags: [tma, mbarrier, cp-async-bulk, hopper, sm90a, pipeline, expect-tx, phase]
confidence: verified
source: PTX ISA 9.7.9.26 / 9.7.14.16 + hopper-hw.md
---
TMA = one thread issues, HW copies the whole 1D-5D tile gmem<->smem, no in-kernel
address math, no register traffic. The FA3 producer instruction.

Load (gmem -> smem), mbarrier-based completion:
  cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.tile
      [smemDst], [tensorMap, {coordX, coordY}], [mbar];
- tensorMap = the 128-byte CUtensorMap (built host-side, see [[ks-0005]]), passed
  as const __grid_constant__ by value.
- On completion HW does complete-tx (decrements mbar tx-count by bytes copied).
- L2 hints: Q EVICT_FIRST (read once), K/V EVICT_LAST (streamed/reused).

mbarrier protocol (the producer-consumer cycle, PTX 9.7.14.16):
  mbarrier.init.shared::cta.b64 [mbar], 1;     // arrival count
  fence.mbarrier_init.release.cluster;          // REQUIRED after init, before use
  // producer (ONE leader thread): set expected bytes, then issue TMA:
  mbarrier.arrive.expect_tx.shared::cta.b64 _, [mbar], TX_BYTES;
  cp.async.bulk.tensor... [smem],[tmap,{..}],[mbar];
  // consumer: spin on phase parity:
  LAB: mbarrier.try_wait.parity.shared::cta.b64 P, [mbar], phase;  @!P bra LAB;

GOTCHAS (deadlock/corruption sources):
- TX_BYTES must EXACTLY equal bytes copied (= tile_elems * sizeof(elem)). Too many
  => deadlock; too few => early consume / corruption.
- EXACTLY ONE leader thread sets expect_tx (multiple => premature phase flip).
- Producer phase starts at 1, consumer at 0 (buffers start empty). Phase flips only
  when the ring index wraps to 0: phase = (total_iters/Stages) % 2.
- Two-barrier-per-stage: FULL (producer->consumer data-ready, TMA auto-signals) +
  EMPTY (consumer->producer buffer-free). EMPTY init count = #consumer warps.
- fence.proxy.async.shared::cta after a smem write, BEFORE wgmma reads it.
- smem int addr for inline PTX: __cvta_generic_to_shared(ptr).

Store (epilogue): cp.async.bulk.tensor...global.shared::cta.tile.bulk_group +
cp.async.bulk.commit_group + cp.async.bulk.wait_group.read 0.

See [[ks-0005]] for host tensormap creation, [[ks-0001]] for the wgmma it feeds,
[[ks-0006]] for copy-able ThunderKittens PTX.
