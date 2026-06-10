---
id: ks-0003
type: ks
title: WGMMA shared-memory matrix descriptor + SMEM swizzling
tags: [wgmma, descriptor, swizzle, smem, hopper, sm90a, layout]
confidence: verified
source: PTX 9.7.16.5 + hopper-hw.md + ref-libs.md (TK descriptor.cuh)
---
WGMMA reads SMEM through a 64-bit matrix descriptor; HW applies an XOR swizzle.
The descriptor and swizzle math can be hand-written with ZERO CUTLASS.

64-bit descriptor fields (TK matrix_descriptor_raw, cross-checked vs DeepGEMM):
  enc(x) = (x & 0x3FFFF) >> 4            // 16B granularity
  desc = enc(smem_addr) | enc(LBO)<<16 | enc(SBO)<<32 | base_offset<<49 | swizzle<<62
- bits[13:0] start addr (smem 16B-aligned); [29:16] leading-dim byte offset (LBO);
  [45:32] stride-dim byte offset (SBO); [51:49] base-offset; [63:62] swizzle mode.
- Swizzle mode codes: 0=none, 1=128B, 2=64B, 3=32B (CUTLASS enum; note TK/DeepGEMM
  encode 128B as 1). For attention bf16 hd128 (256B/row) use 128B swizzle.
- K-major: LBO ~ don't-care (16); SBO = 1024/512/256 for 128/64/32B swizzle.
  MN-major: LBO = 2048*rows/16, SBO = 1024 (128B).
- chunk_descriptor(k) advances per K-step: 128B-swizzle K-major formula
  enc((k%4)*32 + (k/4)*(rows/16)*2048), K=16 elems/chunk for bf16.

Swizzle (PTX 5.5.7): WGMMA requires CuTe-equivalent Swizzle<B,4,3> (M=4,S=3 fixed),
only B varies: B=3 -> 128B, B=2 -> 64B, B=1 -> 32B, B=0 -> none. Pick the largest B
that divides the contiguous tile dim. hd128 bf16 -> 128B.
- 128B swizzle forces inner contiguous dim = 64 bf16 elems and 1024B smem alignment;
  hd128 spans two 64-elem swizzle atoms (the chunk math handles it).
- The tensormap CU_TENSOR_MAP_SWIZZLE_* (see [[ks-0005]]) MUST match this descriptor swizzle.
- Padding ([32][32+1]) does NOT compose with WGMMA descriptor access - use XOR
  swizzle for tensor-core tiles, padding only for non-MMA scratch.

See [[ks-0001]] for how the descriptor is consumed, [[ks-0006]] for copy-able encode code.
