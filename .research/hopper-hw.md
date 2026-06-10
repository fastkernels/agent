# Hopper (sm_90a) Hand-Written FA3 Cheat Sheet

Target: H200 SXM, GH100 die, CC 9.0 / **compile with `-arch=sm_90a`** (wgmma + advanced TMA need
the `a` variant; plain sm_90 will NOT compile wgmma/setmaxnreg). CUDA 13.0, torch 2.11+cu130.

Sources distilled: reference/architecture-specs/hopper-h200.md; design-guides/{gemm-design-space,
gemm-extended-design-space,kernel-patterns}.md; cuda-docs/parallel-thread-execution.md
(§5.5.8 tensor-map, §9.7.9.26 cp.async/bulk, §9.7.14.16 mbarrier, §9.7.16 wgmma, §9.7.20.5 setmaxnreg).

---

## 0. H200 hardware numbers that constrain the kernel

- 132 SMs. Per SM: 128 FP32 cores, 4 4th-gen tensor cores, 4 warp schedulers.
- 64 warps/SM, 2048 threads/SM, 32 blocks/SM max.
- Register file 256 KB = 64K x 32-bit regs/SM; **max 255 regs/thread**.
- SMEM+L1 unified 256 KB/SM. **Configurable SMEM up to 228 KB** (carveouts: 0,8,16,32,64,100,132,164,196,228 KB).
  Static SMEM cap 48 KB; >48 KB is dynamic and requires opt-in (see §6).
- L2 50 MB. HBM3e 141 GB @ 4.8 TB/s. FP16/BF16 dense 989 TFLOPS; FP8 1979 TFLOPS.
- Peak compute identical to H100 SXM; H200 wins only on memory-bound regimes (bigger KV cache, fewer stalls).
  Roofline ridge shifts: more attention configs become compute-bound on H200.
- FP8 = E4M3 (max 448, fwd) / E5M2 (max 57344). **No FP4 on Hopper** (Blackwell only). No TMEM (Blackwell only).

---

## 1. WGMMA — `wgmma.mma_async` (the GEMM workhorse) [PTX §9.7.16]

Warpgroup = **exactly 4 contiguous warps = 128 threads**, first warp's warp-rank a multiple of 4.
warp-rank = `(tid.x + tid.y*ntid.x + tid.z*ntid.x*ntid.y)/32`. The 4-warp boundary is HW-fixed;
asymmetric "2 producer/6 consumer" warp splits are impossible (must be multiples of 4 warps).

### 1.1 Shapes (M is always 64; N varies in multiples of 8; K fixed by dtype)
- **f16/bf16**: `.m64nNk16`, N ∈ {8,16,24,...,256}.  K=16.
- **tf32**: `.m64nNk8`, same N set.  K=8.
- **fp8 e4m3/e5m2**: `.m64nNk32`, same N set.  K=32.  (e4m3/e5m2 can be MIXED for A vs B.)
- s8/u8: `.m64nNk32` (restricted N set); b1: `.m64nNk256`.
- Accumulator D: f16 in → {f16,f32}; bf16 in → f32; tf32 in → f32; fp8 in → {f16,f32}; int → s32.
  For FA3 use **f32 accumulator** (bf16/fp16 Q·K and P·V).

### 1.2 SS vs RS mode
- **B operand MUST be in shared memory** (always a 64-bit matrix descriptor `b-desc`).
- **A operand can be SS (shared, `a-desc`) or RS (registers, `a` vector)**.
  - SS form: `wgmma.mma_async.sync.aligned.SHAPE.DTYPE.f16.f16  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;`
  - RS form (A in regs): drops `imm-trans-a`: `... d, a, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-b;`
  - RS is the FA3 trick for the **second GEMM (P·V)**: P (softmax probs) is already in registers from the
    first GEMM's accumulator → feed directly as A-in-registers, avoiding a SMEM round-trip. First GEMM (Q·Kᵀ)
    is typically SS (both Q,K tiles in SMEM via TMA).
- Operand modifiers:
  - `scale-d` (predicate): false ⇒ `D = A*B` (ignore old D); true ⇒ `D = A*B + D` (accumulate). FA3 K-loop sets
    scale-d=false on the first k-block, true thereafter.
  - `imm-scale-a`, `imm-scale-b` ∈ {-1,+1}: negate operand (cheap sign flip, useful for some algos).
  - `imm-trans-a`, `imm-trans-b` ∈ {0,1}: 0=K-major, 1=M-major(A)/N-major(B). **Transpose only valid for
    f16/bf16 SS operands** (not fp8/tf32, not RS).

### 1.3 Register-fragment layouts (accumulator D in registers, distributed over 128 threads)
- Threads hold fragments; 4 threads share each output row (FA3 reductions use `Allreduce<4>` butterfly).
- **m64nNk16 (f16/bf16)**: A-in-regs = 4× `.f16x2` registers (8 elems a0..a7). D(f32) = N/2 f32 regs per
  thread (e.g. n=8 → {f32d0..d3}; n=256 → 64 f32 regs). D(f16) = N/4 `.f16x2` regs.
- **m64nNk32 (fp8/int8)**: A-in-regs = 4× `.b32` regs (4 elems each = 16 elems). D layout same as k16 by dtype.
- **m64nNk8 (tf32)**: A-in-regs = 4× `.b32` (4 tf32). D = N/2 f32 regs.
- Each thread's accumulator regs grow linearly with N → N drives register pressure (N=256 ⇒ 64 f32/thread for
  the accumulator alone, pushing toward 240 reg/thread consumer allocation). Pick N to fit register budget.

### 1.4 Async proxy + fence/commit/wait protocol (MANDATORY ordering) [§9.7.16.7]
wgmma runs in the **async proxy**; results invisible until committed+waited. Required sequence:
```
wgmma.fence.sync.aligned;          // order prior reg/smem writes (accumulator + A-in-reg) vs wgmma
fence.proxy.async.shared::cta;     // make generic-proxy SMEM writes (TMA/STS) visible to async proxy
wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16  {d0,..}, a-desc, b-desc, p, 1, 1, 0, 0;
wgmma.mma_async... (more, same group)
wgmma.commit_group.sync.aligned;   // batch all prior uncommitted wgmma into one group
wgmma.wait_group.sync.aligned N;   // block until <= N groups pending (N=0 = wait all)
// only AFTER wait_group may you read accumulator D or reuse A-in-reg regs
```
- `.sync.aligned` ⇒ all 128 threads of the warpgroup must execute the identical instruction (uniform control flow).
- **Defer waits** to overlap independent GEMMs: issue QK group + PV group, then `wgmma.wait_group<1>`/`<0>`
  so the two GEMMs and softmax run concurrently (FA3 / FA4 pattern, kStages=2 with separate K/V pipelines).
- CuTe helpers map 1:1: `warpgroup_fence_operand` / `warpgroup_arrive` (=wgmma.fence) /
  `cute::gemm` / `warpgroup_commit_batch` / `warpgroup_wait<N>`.

### 1.5 Shared-memory matrix descriptor (`a-desc`/`b-desc`, 64-bit register) [§9.7.16.5.1.2.2]
The PTX doc shows the bit layout as an image, but it is the standard CUTLASS GMMA descriptor
(`cute/arch/mma_sm90_desc.hpp`). 64-bit fields:
- bits [13:0]   start address of matrix in SMEM, encoded `enc(x)=(x & 0x3FFFF)>>4` (i.e. >>4, 16B granularity).
- bits [29:16]  leading-dimension byte offset (LBO), encoded (>>4).
- bits [45:32]  stride-dimension byte offset (SBO), encoded (>>4).
- bits [51:49]  base-offset (matrix base offset; 0 if pattern starts on swizzle-aligned boundary, see below).
- bits [63:62]  swizzle mode: 0=none, 1=128B, 2=64B, 3=32B (CUTLASS enum).
- `matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4`. SMEM matrix start MUST be 16B aligned.
- Base-offset = 0 when SMEM addr aligned to: 1024B (128B-swz), 512B (64B), 256B (32B); else
  `base_offset = (pattern_start_addr >> 7) & 0x7`.
- LBO/SBO encodings for canonical swizzles (T=128/elem_bits): see §1.6 table.

### 1.6 SMEM swizzling for WGMMA [§5.5.7 tensor swizzle, §9.7.16.5.1.2, kernel-patterns §3]
WGMMA reads SMEM through the descriptor; HW applies XOR swizzle. **WGMMA requires CuTe `Swizzle<B,4,3>`
(M=4 base, S=3 shift fixed); only B varies**:
- `Swizzle<3,4,3>` = 128B swizzle (B=3)
- `Swizzle<2,4,3>` = 64B swizzle (B=2)
- `Swizzle<1,4,3>` = 32B swizzle (B=1)
- `Swizzle<0,4,3>` = none / interleaved (B=0)
Rule: pick the **largest B that divides the contiguous tile dimension** (CUTLASS `ss_smem_selector`).
For head_dim=128 bf16 (256B/row), 128B swizzle is the natural choice. Swizzle-atom (128b-element units):
128B-swz K-major = 8x8 atom; 64B = 8x4; 32B = 8x2; none = 8x1 (multiply leading dim by 128/elem_bits for
sub-128b types: bf16 ⇒ x8). Canonical layouts + descriptor LBO/SBO encodings (from §9.7.16.5.1.2.1.3):
| Major | Swizzle | Canonical CuTe layout |
|---|---|---|
| K | none | ((8,m),(T,2k)):((1T,SBO),(1,LBO)), Swizzle<0,4,3> |
| K | 128B | ((8,m),(T,2k)):((8T,SBO),(1,T)), Swizzle<3,4,3> |
| MN | 128B | ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO)), Swizzle<3,4,3> |
T = 128/sizeof_bits(elem). Padding (`[32][32+1]`) is the cheap alternative but does NOT compose with WGMMA
descriptor access — use XOR swizzle for tensor-core tiles, padding only for non-MMA scratch.

---

## 2. TMA — `cp.async.bulk.tensor` (async tile copy GMEM↔SMEM) [§9.7.9.26.5.2]

One thread issues; HW copies the whole 1D–5D tile; no in-kernel address math; no register traffic.

### 2.1 Load (GMEM → SMEM), the FA3 producer instruction
```
// to shared::cta (single CTA, no cluster):
cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.tile
    [smemDst], [tensorMap, {coordX, coordY}], [mbar];
// to shared::cluster with multicast (multi-CTA cluster):
@p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint
    [smemDst], [tensorMap, {x,y}], [mbar], ctaMask, cachePolicy;
```
- `.dim` = .1d…​.5d. `.load_mode` default `.tile` (preserves layout); `.im2col*` for conv (not attention).
- **Completion is mbarrier-based**: `.mbarrier::complete_tx::bytes` — on copy completion HW performs a
  complete-tx (decrement tx-count by the bytes copied) on `[mbar]`. Generic-proxy access to mbar.
- `tensorMap` = generic address of the 128-byte tensor-map (in `.param`/`.const`/`.global`). `tensorCoords`
  are `.s32` element offsets per dim, one per `.dim`.
- `.multicast::cluster` + `ctaMask` (16-bit, bit i = ctarank i) broadcasts one GMEM read into multiple CTAs'
  SMEM — use when several CTAs in a cluster share the same K/V tile. **Only fast on sm_90a.**
- `.L2::cache_hint` + 64-bit `cache_policy`: FA3 gives K/V `EVICT_LAST` (streamed/reused) and Q `EVICT_FIRST`
  (read once). CacheHintSm90 enum: EVICT_NORMAL=0x1000…, EVICT_FIRST=0x12F0…, EVICT_LAST=0x14F0… (cute copy_sm90_desc.hpp).

### 2.2 Store (SMEM → GMEM), the epilogue write
```
cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group  [tensorMap, {x,y}], [smemSrc];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group.read 0;     // wait for stores (read-complete) before reusing SMEM
```
Store uses `.bulk_group` (async-group) completion, NOT mbarrier. For FA3 you may instead write O directly
with vectorized STG from the accumulator if SMEM staging isn't worth it.

### 2.3 mbarrier protocol for a TMA load (the producer-consumer cycle) [§9.7.14.16]
mbarrier object: opaque `.b64`, **8-byte aligned, in `.shared`**. Max 16 mbarriers/CTA recommended.
Tracks BOTH pending-arrivals AND **tx-count** (transaction bytes) — phase flips only when both hit 0.
```
// init (once, by one thread), arrival count = #threads that will arrive (often 1):
mbarrier.init.shared::cta.b64 [mbar], 1;
fence.mbarrier_init.release.cluster;          // REQUIRED after init, before first use
// producer: set expected bytes for this stage, then issue TMA:
mbarrier.arrive.expect_tx.shared::cta.b64 _, [mbar], TX_BYTES;   // one leader thread only
cp.async.bulk.tensor... [smem],[tmap,{..}],[mbar];               // HW signals complete-tx on done
// consumer: spin-wait on phase parity:
.reg .pred P;
LAB: mbarrier.try_wait.parity.shared::cta.b64 P, [mbar], phase;  // blocking-ish, may sleep
@!P bra LAB;
// ... read SMEM, compute wgmma ...
```
- `mbarrier.arrive.expect_tx [mbar], tx` = expect-tx(tx) THEN arrive(1). **Exactly ONE leader thread**
  must set expect_tx (multiple ⇒ premature flip). `TX_BYTES` MUST equal `bytes copied` exactly
  (= tile_elems * sizeof(elem)); too many ⇒ deadlock, too few ⇒ corruption / early consume.
- Alternatively split: `mbarrier.expect_tx [mbar], tx;` (separate) + plain `mbarrier.arrive`.
- `mbarrier.try_wait.parity` (sm_90+): potentially-blocking, thread may **sleep** until phase completes or a
  system/`timeHint`-ns limit; loop until predicate True. `test_wait` is the non-blocking variant.
- **Phase tracking**: each barrier has a phase bit; parity 0/1. Producer/consumer track index+phase in a
  `PipelineState`; flip phase on buffer wraparound. **Producer starts phase=1, consumer phase=0** (buffers
  start empty). Wrong phase ⇒ instant return on stale data or deadlock. `phase = (total_iters/Stages) % 2`.
- Two-barrier-per-stage pattern (CUTLASS PipelineTmaAsync): **Full** barrier (TransactionBarrier, producer
  signals via TMA HW, consumer waits = "data ready") + **Empty** barrier (ClusterBarrier, consumer signals,
  producer waits = "buffer free"). TMA auto-signals Full on completion (producer_commit is a NOP).

### 2.4 SMEM pointer for PTX
```
uint32_t smem_int = (uint32_t)__cvta_generic_to_shared(smem_ptr);  // or cute::cast_smem_ptr_to_uint
```
mbarrier addr and TMA smem dst both need this 32-bit shared address in inline PTX.

---

## 3. Tensormap creation — HOST side (allowed: this is host code, not __global__)

The tensor-map is a **128-byte opaque object** describing tensor base/shape/strides/box/swizzle/OOB-fill.
Created on host with the CUDA driver API (NOT inside the kernel):
```cpp
#include <cuda.h>   // driver API, cuTensorMapEncodeTiled
CUtensorMap tma_map;                                  // 128 bytes, 64B-aligned
CUtensorMapDataType dt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; // or _FLOAT16 / _UINT8 for fp8
cuTensorMapEncodeTiled(
   &tma_map, dt,
   /*tensorRank*/ rank,                 // 1..5; attention K/V often 2D-3D (seqlen, headdim[, head])
   /*globalAddress*/ d_ptr,             // base GMEM pointer (16B aligned)
   /*globalDim[]*/ {d, seqlen, ...},    // size per dim, inner dim first
   /*globalStrides[]*/ {stride1B, ...}, // BYTE strides for dims 1..rank-1 (dim0 stride implicit=elem size)
   /*boxDim[]*/ {BLK_D, BLK_N, ...},    // SMEM tile (box) extent per dim
   /*elementStrides[]*/ {1,1,...},
   CU_TENSOR_MAP_INTERLEAVE_NONE,
   CU_TENSOR_MAP_SWIZZLE_128B,          // MUST match the WGMMA descriptor swizzle (Swizzle<3,4,3>)
   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);  // OOB tile reads zero-filled (handles ragged seqlen edges!)
```
- Pass to kernel as a **`const __grid_constant__ CUtensorMap` by value** parameter (recommended) OR via
  `__constant__`/global pointer. Grid-constant param lives in `.param` space, directly usable by TMA.
- **OOB zero-fill** is the clean way to handle partial tiles at the end of a non-tile-divisible seqlen —
  no separate edge kernel, no masking in the load. This is a Hopper-only luxury (Ampere needs masks/pad).
- If the descriptor must change mid-kernel, use `tensormap.replace` (device) + `fence.proxy.tensormap::generic`,
  and `prefetch.tensormap [ptr]` to warm the const cache — but for FA3 a per-launch host-built map is enough.
- Restrictions: inner (dim0) box extent * elem size must respect swizzle (128B-swz ⇒ 128B inner chunk);
  global base + per-dim base must be 16B aligned; box dim ≤ 256 per dim.

---

## 4. SMEM capacity & the 228 KB opt-in (host side, allowed)

- Static `__shared__` ≤ 48 KB. Above 48 KB you MUST request dynamic SMEM and opt in:
```cpp
int smem_bytes = /* e.g. 200*1024 */;
if (smem_bytes >= 48*1024)
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);  // up to 228 KB
// optional carveout preference:
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, /*0..100 percent or KB*/);
kernel<<<grid, block, smem_bytes, stream>>>(...);
```
- **Check return** and fall back gracefully (a non-H200 device may not provide 228 KB).
- SMEM budget formula per CTA (with double/multi-buffering):
  `((BLK_M*BLK_K + BLK_K*BLK_N)*sizeof(elem))*num_stages + mbarriers*8 + epilogue_staging ≤ carveout`.
  At 228 KB you can fit ~3-4 stages of bf16 128x128 Q + K/V tiles for typical head dims.
- 228 KB / SM usually limits occupancy to **1 CTA/SM** for big FA3 tiles — that is normal and fine
  (FA3 is tensor-core-bound, runs near peak at ~1 block/SM / ~25% occupancy).

---

## 5. Warp specialization + setmaxnreg [§9.7.20.5, kernel-patterns §7]

Split the CTA into **producer warpgroups (issue TMA)** and **consumer warpgroups (issue WGMMA + softmax)**.
Communicate only through SMEM + transaction mbarriers — no block-wide barrier between them. Total time ≈
max(load, compute) instead of load+compute.

### 5.1 Register redistribution
```cpp
if (warpgroup_id == 0) { cutlass::arch::warpgroup_reg_dealloc<24>();  /* producer: TMA only, ~24-40 regs */ }
else                   { cutlass::arch::warpgroup_reg_alloc<240>();   /* consumer: WGMMA accum, up to 240 */ }
```
Raw PTX: `setmaxnreg.dec.sync.aligned.u32 24;` / `setmaxnreg.inc.sync.aligned.u32 240;`
- imm-reg-count ∈ [24,256], **multiple of 8**. `.dec` releases regs to CTA pool; `.inc` blocks until pool has
  enough. ALL 4 warps of the warpgroup must execute the SAME setmaxnreg (uniform); sync after.
- Requires the kernel launched with a valid max-reg (set via `__launch_bounds__` / `-maxrregcount`); else no-op.
- Typical FA3 split: 1 producer WG @ 24-32 / N consumer WGs @ 232-240. Pingpong = 2 consumer WGs alternating
  via **named barriers** so tensor cores stay ~100% fed while one WG does softmax and the other does WGMMA.
- Valid configs (multiples of 4 warps): 1prod+1cons = 8 warps (24-56/232-256); 1prod+2cons(pingpong)=12 warps
  (24-40/232-240); 1prod+3cons=16 warps (32/160).

### 5.2 Named barriers (intra-CTA, beyond barrier 0)
- `bar.sync id, threadcount;` / `bar.arrive id, count;` — `id` ∈ [0,15] (16 hardware named barriers per CTA).
- Used for pingpong consumer scheduling (each consumer WG waits its own barrier id, signals the next) and to
  separate producer/consumer phases without a full `__syncthreads()`.
- `barrier.cluster.arrive`/`.wait` for cluster-scope sync (only if using clusters/DSMEM).

### 5.3 Required fences (common deadlock/corruption sources) [kernel-patterns §6]
- `fence.mbarrier_init.release.cluster;` after `mbarrier.init`, before first use. (missing ⇒ races)
- `fence.proxy.async.shared::cta;` after STSM/TMA write into SMEM, **before** the WGMMA reads it.
- `fence.proxy.async.global;` before a TMA store reads from generic-proxy-written SMEM.
- Missing `wgmma.fence` before reading accumulator ⇒ compiler reorders ⇒ wrong results.
- Releasing the Empty barrier (consumer_release) before `wgmma.wait_group<0>` ⇒ producer overwrites
  SMEM still being read by tensor cores ⇒ corruption.
- Producer must run `producer_tail()` (drain) before exit, else consumer hangs.

---

## 6. Design-space knobs for attention (what to sweep) [design-space.md, kernel-patterns §4-5]

Tile / pipeline / cluster choices relevant to FA3 forward (prefill + decode):
- **block_m (Q rows/CTA)**: 64, 128, 192. 128 typical for prefill; decode often M tiny (1 query) → GEMV-class.
- **block_n (KV cols/iter)**: 64, 128, 256. Larger N = bigger WGMMA accumulator = more regs.
- **head_dim (K dim)**: fixed by model (64, 128, 256). Drives WGMMA k-step count and SMEM row bytes.
- **num_stages**: SMEM pipeline depth for K/V tiles. 2-4 on Hopper (FA3 uses kStages=2 w/ separate K & V
  pipelines; ThunderKittens uses 4). Deeper = more SMEM, more mbarriers, more warmup/drain.
- **pipeline_type**: producer-consumer (warp-specialized) is the Hopper default; pingpong (2 consumer WGs)
  for max tensor-core occupancy.
- **wgmma N choice**: trades accumulator regs vs instruction count; keep tile dims multiples of WGMMA shape
  (M=64 fixed, N mult of 8, K=16 bf16 / 32 fp8).
- **cluster_dim**: 1 (no cluster) is simplest and usually right for attention. Clusters (≤8 portable, ≤16
  non-portable) only help if multiple CTAs share the same K/V tile and you want TMA `multicast` + DSMEM —
  marginal for FA3 unless doing very large context with split-KV. Launch via `cudaLaunchKernelEx`.
- **grid_strategy**: **persistent kernel** (grid = 132 SMs, loop tiles from a scheduler) is the Hopper default
  — warm L2, no relaunch/drain between tiles. Combine with grid/tile swizzling for L2 locality.
- **split_k / split-KV (FlashDecoding)**: for decode with long context + few queries, split the KV sequence
  across CTAs, each does partial online-softmax, then a small reduction kernel combines (log-sum-exp merge).
  Raises occupancy when M is tiny. `split_k_reduction` via separate kernel or atomics/DSMEM.
- **online softmax** [kernel-patterns §10]: incremental over K-tiles, never materialize N×N. Per tile:
  update running max → rescale O accumulator → `exp2f` (base-2, fused scale+max-subtract) → accumulate row sum.
  4-thread butterfly `Allreduce<4>` matches the WGMMA row layout (4 threads share a row).
- **epilogue**: FP32 accum → bf16 output; TMA store via SMEM staging OR direct vectorized STG.

---

## 7. Anti-cheating / compile notes for the generated .cu
- Build line: `nvcc -arch=sm_90a -O3 ...`. wgmma, setmaxnreg, advanced TMA, multicast all REQUIRE sm_90a.
- Everything in §3 (cuTensorMapEncodeTiled), §4 (cudaFuncSetAttribute), §5.1 launch-bounds, output alloc may
  live in the host `mha_fwd`. NONE of CUTLASS/CuTe is needed: the descriptor (§1.5), swizzle math (§1.6),
  mbarrier protocol (§2.3), and PTX (§1.4/2.1/5.1) can all be written as raw inline PTX / hand structs.
- Inline-PTX gotchas [kernel-patterns §11]: errors surface at ptxas not C++; brace multi-instr blocks;
  match constraints (r=u32, l=u64, f=f32, h=u16, n=compile-time imm); outputs numbered before inputs;
  use `asm volatile(... ::: "memory")` for barriers/stores/sync; convert C++ bool↔PTX pred via `setp`/`selp`.

---

## 8. Minimal FA3 mainloop skeleton (per consumer warpgroup, one K/V tile iter)
```
// stage s buffer ready? (Full barrier, producer filled via TMA)
mbarrier.try_wait.parity [full[s]], phase_full[s];
// GEMM1: S = Q @ Kᵀ   (Q SS or RS, K SS), f32 accum
wgmma.fence; fence.proxy.async.shared::cta;
wgmma.mma_async.m64nNk16.f32.bf16.bf16  Sacc, qDesc, kDesc, scaleD0, 1,1,0,1;  // imm-trans-b=1 → Kᵀ
wgmma.commit_group; wgmma.wait_group 0;
mbarrier.arrive [empty[s]];                  // free K buffer for producer
// online softmax on Sacc (in regs): rowmax, rescale O, exp2, rowsum  (Allreduce<4>)
// GEMM2: O += P @ V    (P = softmax(S) in REGISTERS → RS mode A-in-regs; V SS)
wgmma.fence;
wgmma.mma_async.m64nNk16.f32.bf16.bf16  Oacc, pRegs, vDesc, scaleD_accum, 1,1,1;  // RS form
wgmma.commit_group; wgmma.wait_group 0;
mbarrier.arrive [empty_v[s]];
// advance stage/phase; on wrap flip phase bits
```
Producer warpgroup loop: `try_wait(empty[s]); arrive.expect_tx(full[s], tile_bytes); cp.async.bulk.tensor(...)`.

---

## 9. Quick correctness/perf gotchas summary
1. TX_BYTES on expect_tx must EXACTLY equal copied bytes (deadlock/corruption otherwise); single leader thread.
2. Producer phase starts at 1, consumer at 0.
3. WGMMA needs `Swizzle<B,4,3>` (M=4,S=3 fixed); tensormap CU_TENSOR_MAP_SWIZZLE_* must match the descriptor.
4. Read accumulator only after `wgmma.wait_group`; fence operand before/after.
5. >48 KB SMEM ⇒ cudaFuncSetAttribute opt-in (≤228 KB); likely 1 CTA/SM, that's OK for tensor-core-bound FA3.
6. setmaxnreg imm must be multiple of 8 in [24,256]; all 4 warps of the WG execute it identically.
7. f16/bf16 accumulate to f32 for FA3 numerics; fp8 path uses k32 and may need per-row/col scaling (no HW
   block-scale on Hopper).
8. Use OOB zero-fill in the tensor-map for ragged seqlen edge tiles instead of masking in the load.
9. wgmma B is ALWAYS in SMEM; A can be SMEM(SS) or regs(RS) — use RS for P·V to keep softmax probs in place.
