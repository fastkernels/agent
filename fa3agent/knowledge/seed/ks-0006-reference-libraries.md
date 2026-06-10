---
id: ks-0006
type: ks
title: Reference-library file pointers - ThunderKittens (pure PTX) + DeepGEMM skeleton
tags: [reference, thunderkittens, deepgemm, flashinfer, ptx, copy-able, anti-cheat]
confidence: source-reported
source: .research/ref-libs.md
---
Mine these for READABLE pure-PTX models to transcribe into the .cu - NONE are linked
(verify nm/ldd has no cutlass/cute symbols). Usefulness ranking for our pure-PTX kernel:

ThunderKittens (BEST, pure inline-PTX, copy-able). Root reference/libraries/ThunderKittens/:
- include/ops/thread/memory/tile/tma.cuh: load_async (cp.async.bulk.tensor 5d), store_async.
- include/ops/thread/util/sync.cuh + util/tma.cuh: init_semaphore (mbarrier.init), arrive,
  wait (try_wait.parity spin), expect_bytes (mbarrier.arrive.expect_tx).
- include/ops/group/mma/base/64x{N}.impl + warpgroup.cuh: the canonical bf16
  wgmma.mma_async.m64nNk16.f32.bf16.bf16, mma_fence/commit/wait. RS (A in regs) for PV.
- include/types/shared/descriptor.cuh: matrix_descriptor_encode/raw + chunk_descriptor
  (CRITICAL, pure - copy this, see [[ks-0003]]).
- include/types/global/tma.cuh: create_tensor_map -> cuTensorMapEncodeTiled host recipe ([[ks-0005]]).
- kernels/attention/mha_h100/mha_h100.cu (full warp-specialized FA3) and especially
  mha_h100_lcf.cu (294 lines, the single best end-to-end readable model - read first;
  online-softmax math L64-83, see [[alg-0004]]).

DeepGEMM (architecture model; MMA layer uses cute, do not copy that). Root reference/libraries/DeepGEMM/:
- impls/sm90_bf16_gemm.cuh (432 ln): best STRUCTURAL skeleton - producer/consumer,
  full/empty barriers, warpgroup_reg_dealloc/alloc, persistent scheduler, STSM epilogue.
- impls/sm90_fp8_mqa_logits.cuh: closest to attention (Q@K^T + per-row reduce, dual pipeline).
- ptx/tma.cuh: pure-PTX TMA/mbarrier (strip the cutlass barrier types).

flashinfer (two slices): mma.cuh/cp_async.cuh = self-contained Ampere mma.sync m16n8k16
(non-WGMMA fallback / small-M decode); attention/prefill.cuh::update_mdo_states = clean
fragment-layout online softmax. Hopper path is 100% CUTLASS (reference only).
FlashMLA / TransformerEngine / cccl = low priority for our kernel.

WGMMA is m64 in M - for decode (M small) prefer mma.sync m16n8 or pack heads into M.
See [[alg-0003]] for how these assemble into the kernel.
