---
id: ks-0005
type: ks
title: Host-side tensormap creation + 228KB SMEM opt-in (toolkit-only host code)
tags: [tma, tensormap, cutensormap, smem, opt-in, host, sm90a, anti-cheat]
confidence: verified
source: hopper-hw.md sections 3-4 + ref-libs.md (TK types/global/tma.cuh)
---
All of this lives in the HOST mha_fwd (NOT __global__), uses only the CUDA driver
(-lcuda, toolkit-shipped) - no CUTLASS. This is the one allowed host dependency.

Tensormap (128-byte opaque, describes base/shape/strides/box/swizzle/OOB-fill):
  CUtensorMap tma_map;   // 128 bytes, 64B-aligned
  cuTensorMapEncodeTiled(&tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
     rank, d_ptr, globalDim[], globalStrides[] /*BYTE strides, dim0 implicit*/,
     boxDim[] /*smem tile extent*/, elementStrides[],
     CU_TENSOR_MAP_INTERLEAVE_NONE,
     CU_TENSOR_MAP_SWIZZLE_128B,   // MUST match the WGMMA descriptor swizzle (ks-0003)
     CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);  // OOB tile reads zero-filled
- Pass to kernel as const __grid_constant__ CUtensorMap by value (FA3/DeepGEMM
  convention) OR via a device pointer (TK convention; cudaMalloc+memcpy the map).
- OOB ZERO-FILL handles ragged seqlen edge tiles with NO separate edge kernel and
  NO masking in the load - a Hopper-only luxury. V uses descriptor shape (hdv,seqlen)
  for transposed-by-layout load; bf16 needs no real transpose.
- Constraints: base 16B-aligned; strides %16==0; box dim <= 256; for 128B swizzle the
  inner contiguous tile dim must be 64 bf16 elems.

228KB SMEM opt-in (static __shared__ is capped at 48KB; above that is dynamic):
  if (smem_bytes >= 48*1024)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); // <=228KB
  kernel<<<grid, block, smem_bytes, stream>>>(...);
- CHECK the return and fall back gracefully (non-H200 may lack 228KB).
- At 228KB occupancy is usually 1 CTA/SM for big FA3 tiles - that is NORMAL and fine
  (FA3 is tensor-core-bound, runs near peak at ~1 block/SM).
- smem budget: ((BLK_M*BLK_K + BLK_K*BLK_N)*sizeof(elem))*num_stages + mbarriers*8
  + epilogue_staging <= carveout. 228KB fits ~2-4 stages of bf16 hd128 tiles.

See [[ks-0002]] for the device TMA that uses this map, [[ks-0003]] for the swizzle
it must match. Anti-cheat: cuTensorMapEncodeTiled is driver API, allowed.
