---
id: ks-0001
type: ks
title: WGMMA wgmma.mma_async - shapes, SS vs RS, fence/commit/wait protocol
tags: [wgmma, tensor-core, hopper, sm90a, gemm, descriptor, ss-rs]
confidence: verified
configs: [bf16, fp16, fp8e4m3]
source: PTX ISA 9.7.16 + hopper-hw.md
---
The GEMM workhorse on Hopper. Warpgroup = exactly 4 contiguous warps = 128 threads;
all 128 issue the same .sync.aligned instruction (uniform control flow).

Shapes (M always 64; N multiple of 8; K fixed by dtype):
- bf16/fp16: .m64nNk16, K=16.   fp8 e4m3/e5m2: .m64nNk32, K=32.   tf32: .m64nNk8.
- Use f32 accumulator for FA3 (bf16/fp16 in). N drives register pressure: each
  thread holds N/2 f32 accumulator regs (N=256 => 64 f32/thread). Pick N to fit budget.

SS vs RS (B operand is ALWAYS in shared memory, a 64-bit matrix descriptor):
- A can be SS (shared, a-desc) or RS (registers, a vector).
- FA3 trick: QK^T is SS (Q,K tiles in smem via TMA). PV is RS - P (softmax probs)
  is already in registers from QK's accumulator, feed directly as A-in-regs,
  avoiding a smem round-trip. RS form drops imm-trans-a.
- scale-d predicate: false => D = A*B (first k-block, zero-init); true => D = A*B + D.
- imm-trans-b=1 gives K^T for the QK GEMM (only valid for f16/bf16 SS operands).

MANDATORY ordering protocol (PTX 9.7.16.7):
  wgmma.fence.sync.aligned;            // order prior reg/smem writes vs wgmma
  fence.proxy.async.shared::cta;       // make generic-proxy smem writes visible
  wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 {d..}, a-desc, b-desc, p,1,1,0,0;
  wgmma.commit_group.sync.aligned;     // batch uncommitted wgmma into one group
  wgmma.wait_group.sync.aligned N;     // block until <= N groups pending (0 = all)
  // ONLY after wait_group may you read accumulator D or reuse A-in-reg regs.
Defer waits (wait_group<1>) to overlap QK and PV GEMMs (the FA3 pipeline trick).
Missing wgmma.fence before reading accumulator => compiler reorders => wrong results.

See [[ks-0003]] for the 64-bit descriptor + swizzle, [[ks-0006]] for copy-able PTX,
[[alg-0004]] for the softmax between the two GEMMs.
