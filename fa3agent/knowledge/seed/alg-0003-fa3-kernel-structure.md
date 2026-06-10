---
id: alg-0003
type: alg
title: FA3 SM90 forward kernel structure - warp-specialized producer/consumer
tags: [fa3, warp-specialization, producer-consumer, pipeline, persistent-kernel, hopper]
confidence: source-reported
source: vLLM FA3 fork (flash_fwd_kernel_sm90.h), .research/fa3-kernel-algo.md
---
The distilled .cu is ONE persistent warp-specialized kernel:
- 1 producer (load) warpgroup + 1..3 consumer (MMA) warpgroups. WG = 128 threads
  = 4 warps (HW-fixed; no 2/6 splits). MaxThreadsPerBlock = NumMmaThreads + 128.
- Producer issues TMA loads; only warp 0 issues when NumProducerThreads==32
  (SingleProducerWarp, the common bf16 path) - other producer warps early-return
  or you over/under-arrive barriers.
- Consumers do wgmma (QK then PV) + online softmax.
- Communicate only via SMEM + transaction mbarriers, no block-wide barrier between
  producer and consumer. Total time ~= max(load, compute) not load+compute.
- Dynamic smem ~227KB (H200 max). smem_o ALIASES smem_v - producer must wait
  barrier_O before overwriting; get the (work_idx+1)%2 phase right or deadlock.

Register reallocation via setmaxnreg (see [[ks-0004]]): producer reg_dealloc
(24/32/40/56), consumer reg_alloc (160/232/240/256). hd128 common case:
2 MMA WG -> load 24/40, mma 240/232.

Phase 1 (correctness, bf16 only): non-persistent SingleTileScheduler, cluster=1,
kStages=2, IntraWGOverlap on, hd128 tile {128,128/160}, 2 MMA WG + 1 load WG.
Implement TMA Q/K/V + mbarrier pipeline; wgmma QK^T (SS); online softmax;
wgmma PV (RS, P in regs); causal+seqlenk mask split into masked/unmasked iters;
epilogue STSM->smem->TMA store O + LSE. VALIDATE vs _vllm_fa3_C.fwd BEFORE optimizing.

See [[alg-0004]] for the online-softmax math, [[ks-0001]] for wgmma,
[[ks-0002]] for TMA/mbarrier, [[ks-0006]] for reference-library file pointers.
