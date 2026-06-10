<!-- originally from https://github.com/xinhaoc/ferret/tree/main/docs/patterns -->
# CUDA Kernel Optimization Patterns

Consolidated reference for GEMM and attention kernel patterns on NVIDIA Ampere (SM80),
Hopper (SM90), and Blackwell (SM100). Combines what were previously separate pattern
files: memory access, GEMM tiling, swizzling, software pipelining, async execution,
warp specialization, persistent kernels, occupancy tuning, reductions, and inline PTX.

Code excerpts cite the upstream source they were taken from (CUTLASS, FlashAttention,
DeepGemm, ThunderKittens, Triton, FlashInfer).

## Contents

1. [Architecture cheat-sheet](#1-architecture-cheat-sheet)
2. [Memory access patterns](#2-memory-access-patterns)
3. [Shared memory swizzling](#3-shared-memory-swizzling)
4. [GEMM tiling](#4-gemm-tiling)
5. [Software pipelining (double buffering)](#5-software-pipelining-double-buffering)
6. [Async execution model](#6-async-execution-model)
7. [Warp specialization](#7-warp-specialization)
8. [Persistent kernels & scheduling](#8-persistent-kernels--scheduling)
9. [Occupancy tuning](#9-occupancy-tuning)
10. [Reduction & online softmax](#10-reduction--online-softmax)
11. [Inline PTX reference](#11-inline-ptx-reference)

---

## 1. Architecture cheat-sheet

The single most important axis across every pattern below is the SM80 → SM90 → SM100
progression in how data moves and how MMA is issued.

| Aspect | Ampere SM80 | Hopper SM90 | Blackwell SM100 |
|---|---|---|---|
| Global→shared load | `cp.async` (thread-issued, commit/wait groups) | TMA (one thread issues, hardware copies tiles) | TMA (enhanced) |
| Load completion signal | `cp.async.wait_group<N>` | transaction barrier (mbarrier tracks bytes) | transaction barrier + `tcgen05` pipeline |
| MMA instruction | `mma.sync` / `wmma` | `wgmma.mma_async` (warpgroup, 128 threads) | `tcgen05.mma` (single elected thread) |
| Accumulator location | registers | registers (distributed over 128 threads) | TMEM (256 KB on-chip, separate from SMEM) |
| Specialization unit | none (all threads load+compute) | warpgroup (4 warps = 128 threads) | warp (1 warp = 32 threads) |
| Clusters / DSMEM / multicast | no | yes | yes (clusters up to 16, enhanced) |
| Register redistribution | no | `setmaxnreg` | `setmaxnreg` |
| Max SMEM/CTA | 164 KB | 228 KB | 228 KB |

Consumer Blackwell (SM120, e.g. RTX 5090) is **not** SM100: no TMEM, no `tcgen05`, uses
`mma.sync` like Ampere/Ada. Treat it like the SM80 column for MMA and accumulator
placement even though it is branded "Blackwell".

---

## 2. Memory access patterns

### Global memory coalescing

A warp issues 32 memory requests together; hardware coalesces them into as few 128-byte
transactions as possible.

```cpp
float v = input[blockIdx.x*blockDim.x + threadIdx.x];   // coalesced: 1 txn / 32 floats
float v = input[2*(blockIdx.x*blockDim.x + threadIdx.x)]; // stride-2: 2x transactions
float v = input[indices[threadIdx.x]];                    // random: up to 32 transactions
```

Structure-of-Arrays beats Array-of-Structures on GPUs: `float x_arr[N]` gives coalesced
access where `struct{float x,y,z,w;} arr[N]; arr[tid].x` is stride-4.

**Production (FlashAttention SM80,** `mainloop_fwd_sm80.hpp`**)**: stride-1 on the head
dim makes elements contiguous; the thread layout maps consecutive threads to consecutive
chunks so a warp touches one contiguous 128-byte region in a single transaction.

```cpp
using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;   // (seqlen, d, head, batch)
static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t)/sizeof(Element); // 8 for fp16
using GmemLayoutAtom = Layout<Shape<Int<NumMmaThreads/kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                              Stride<Int<kGmemThreadsPerRow>, _1>>;
using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                                  GmemLayoutAtom{}, Layout<Shape<_1,_8>>{}));  // 8 vals/thread
```

### Vectorized loads

Wider loads cut instruction count and improve bandwidth. CUTLASS issues predicated
128-bit (`uint4`) loads, with a doubled 256-bit variant and an `L2::128B` prefetch hint
on Ampere+ (`cutlass-4.4.2/include/cutlass/arch/memory.h`):

```cpp
// 128-bit predicated load: @p guards OOB threads without branching
"  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
// 256-bit (two back-to-back) with L2 prefetch hint:
"  @p ld.global.L2::128B.v4.u32 {%0,%1,%2,%3}, [%8];\n"
"  @p ld.global.L2::128B.v4.u32 {%4,%5,%6,%7}, [%18];\n"   // +16 bytes
```

### Shared memory bank conflicts

32 banks, 4 bytes each; bank = `(addr/4) % 32`. Same-bank/different-address within a warp
serializes; same-address is a free broadcast.

```cpp
float v = smem[threadIdx.x];     // no conflict
float v = smem[2*threadIdx.x];   // 2-way conflict (thread 0 and 16 → bank 0)
float v = smem[0];               // 32-way "conflict" → broadcast, actually free
```

Two fixes: padding (`__shared__ float tile[32][32+1]`, simple but wastes SMEM and doesn't
compose with tensor-core access) or XOR swizzle (§3, zero waste, matches ldmatrix/WGMMA).

### L2 cache control

Persistent L2 window (Ampere+): pin hot data via stream attributes.

```cpp
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = d_data;
attr.accessPolicyWindow.num_bytes = min(size, l2_size/2);
attr.accessPolicyWindow.hitRatio  = 1.0f;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

PTX cache hints: `ld.global.cs` (streaming, don't pollute), `ld.global.ca` (cache all
levels, default), `ld.global.lu` (last-use, evict after). On SM90+, TMA takes a 64-bit
cache hint — FlashAttention gives K/V `EVICT_LAST` (streamed repeatedly) and Q
`EVICT_FIRST` (read once):

```cpp
// cute/arch/copy_sm90_desc.hpp
enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST  = 0x12F0000000000000,
  EVICT_LAST   = 0x14F0000000000000,
};
```

### Global→shared async copy

**cp.async (SM80)** moves global→shared without going through registers. CUTLASS
templates it on size (4/8/16 B) with instruction-level predication, plus a zero-fill
variant for boundary tiles (`memory_sm80.h`):

```cpp
"  @p cp.async.ca.shared.global [%1], [%2], %3;\n"          // predicated copy
"cp.async.ca.shared.global [%0], [%1], %2, %3;\n"           // zfill: src_in_bytes=0 → zeros
"cp.async.commit_group;"  /  "cp.async.wait_group N;"       // batch + wait
```

**TMA (SM90+)** copies an entire 2D–5D tile with one instruction, no address arithmetic
in-kernel, optional multicast to a cluster (`copy_sm90_tma.hpp`):

```cpp
"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
" [%0], [%1, {%3, %4}], [%2], %5;"   // [smem], [desc, coords], [mbar], hint
```

TMA advantages: no register address math, hardware multi-dim tiling, multicast, lower
register pressure, per-transfer L2 hints. Available on Hopper, enhanced on Blackwell.
**TMEM** (Blackwell only) is a separate on-chip space for tensor-core operands/accumulators,
accessed via `tcgen05`; see §7 and the Blackwell architecture doc.

---

## 3. Shared memory swizzling

Naive row-major SMEM layouts make tensor-core tile accesses alias to the same banks (up
to 32-way conflict → 32 cycles for a 1-cycle access). XOR swizzling permutes bank
assignment per row with zero wasted memory and matches ldmatrix/WGMMA patterns exactly.

### `Swizzle<B,M,S>` (cute/swizzle.hpp)

Operates on the bits of a SMEM offset: `result = offset XOR ((offset & Y_mask) >> S)`,
mapping Y bits (row) into Z bits (bank). XOR is an involution, so it's its own inverse.

| Param | Meaning | Common values |
|---|---|---|
| B (BBits) | number of XOR bits → swizzle width | 0 none, 1 (32B), 2 (64B), 3 (128B) |
| M (MBase) | low bits left unchanged → max vector access = 2^M bytes | 3 (8B), 4 (16B), 5 (32B) |
| S (SShift) | distance between Y and Z fields; needs `abs(S) >= B` | 2, 3 |

`Swizzle<3,3,3>` example: column 0 across 8 rows hits banks {0..7} instead of all bank 0
→ zero conflicts.

### Per-architecture swizzle

| Arch | Instruction | Swizzle | Notes |
|---|---|---|---|
| SM80 | ldmatrix | `Swizzle<3,3,3>` | half precision, 128-bit smem→reg |
| SM90 | WGMMA SS | `Swizzle<3,4,3>` SW128 / `<2,4,3>` SW64 / `<1,4,3>` SW32 / `<0,4,3>` none | **WGMMA requires M=4, S=3**; only B varies |
| SM100 | UMMA | `Swizzle<2,5,2>` | 128B base, 32B granularity |

On SM90 the GMMA descriptor encodes the layout type (INTERLEAVE/B32/B64/B128) and the
hardware applies the swizzle when reading SMEM. Rule of thumb: pick the largest B that
divides the tile dimension (CUTLASS `ss_smem_selector` does this automatically).
`ComposedLayout<Swizzle<B,M,S>, Offset, LayoutB>` maps `coord → Swizzle(Offset+LayoutB(coord))`;
slicing a swizzled layout often decays to a plain strided layout for per-thread views.

---

## 4. GEMM tiling

Tiling loads reusable blocks into SMEM so DRAM traffic drops proportionally to tile size.
The hierarchy: grid blocks tile output C (CTA tile, e.g. 128×256) → warps tile the CTA
tile (warp tile, e.g. 64×64) → MMA instructions tile the warp tile (e.g. 16×8×16).

### CuTe SM80 GEMM (`examples/cute/tutorial/sgemm_sm80.cu`)

```cpp
auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<64>{});   // BLK_M, BLK_N, BLK_K
auto bP = Int<3>{};                                                // 3 pipeline stages
auto swizzle_atom = composition(Swizzle<3,3,3>{}, /* ... */);      // bank-conflict-free
TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                               Layout<Shape<_2,_2>>{}, Tile<_32,_32,_16>{});
// data flow: Global -(cp.async)-> Shared (swizzled) -(ldmatrix)-> Registers -(mma)-> Accum
```

### ThunderKittens H100 GEMM (`kernels/gemm/bf16_h100`)

CTA 128×256 from 64×64 bf16 base tiles, 4 pipeline stages, producer warpgroup TMA-loads,
consumer warpgroups WGMMA from SMEM into register accumulators `rt_fl<16,256>`, SUPER_M=12
for L2 grid swizzling.

### DeepGemm FP8 Hopper (`impls/sm90_fp8_gemm_1d1d.cuh`)

WGMMA 64×N×32, BLOCK_K=128 (4 WGMMAs/k-block for per-128-channel FP8 scaling). Split
TMA/math warps; after each k-block the raw accumulator is promoted with per-row/per-col
FP8 scale factors: `final += scale_a * scale_b * accum`.

### Choosing tile sizes

| Constraint | Rule |
|---|---|
| SMEM | `(TILE_M·TILE_K + TILE_K·TILE_N)·bytes·stages ≤ smem limit` |
| Occupancy | bigger tiles → more SMEM → fewer blocks/SM |
| Registers | accumulator = TM×TN floats per thread |
| Tensor cores | tile dims multiple of MMA tile (16 on SM80, 64 for WGMMA) |
| Wave quantization | grid should divide evenly into SM count |

Typical: SM80 128×128×64 / 16×8×16 / 3 stages; SM90 128×256×64 / WGMMA 64×N×16 / 4 stages;
SM90 FP8 BLOCK_M×BLOCK_N×128 / 64×N×32.

---

## 5. Software pipelining (double buffering)

Without pipelining, compute and memory serialize. Pipelining overlaps them: while
computing stage N, load stage N+1.

### Ampere cp.async (3-stage, `sgemm_sm80.cu`)

Two nested levels: (1) SMEM pipeline of 3 stages via `cp_async_fence`/`wait_group<N>`,
(2) register pipeline within a stage where ldmatrix for the next sub-tile overlaps MMA on
the current one. Prefill stages 0–1 before the loop; in the mainloop, on the last k_block
advance the read pointer and `cp_async_wait<K_PIPE_MAX-2>()`, on k_block 0 issue the next
tile's cp.async.

### Hopper TMA + mbarrier

Unlike cp.async's thread-local ordered groups, TMA uses **per-stage mbarrier pairs** with
separate producer/consumer warp groups; TMA hardware auto-signals the full barrier on
completion (`producer_commit` is a NOP). See §6 for the barrier protocol.

### FlashAttention FA4 overlapped QK/PV (2-stage)

`kStages=2`, separate K and V pipelines. One `fwd_step` overlaps three things: iteration
N's QK GEMM, iteration N-1's PV GEMM, and iteration N's softmax — by deferring waits
(`wg_wait=-1`) so the GEMMs stay in flight while softmax runs.

### Comparison

| | Ampere cp.async | Hopper TMA | FA4 | ThunderKittens |
|---|---|---|---|---|
| Stages | 3 | configurable | 2 | 4 |
| Sync | commit/wait groups | per-stage mbarrier pairs | CUTLASS PipelineTmaAsync | kittens semaphores |
| Producer/consumer | same threads | separate warpgroups | WG0 / WG1-3 | last WG / rest |
| Commit | `cp_async_fence` | NOP (TMA auto-signals) | NOP | `tma::expect`+`load_async` |
| Reg budget | shared | prod 24-48 / cons 224-240 | prod 24 / cons 240 | prod 40 / cons 232 |

---

## 6. Async execution model

The producer-consumer machinery underlying Hopper/Blackwell pipelines.

### Two barriers per stage

| Barrier | Type | Signals | Waits | Meaning |
|---|---|---|---|---|
| Full | `ClusterTransactionBarrier` | producer (via TMA HW) | consumer | "data ready" |
| Empty | `ClusterBarrier` | consumer | producer | "buffer free" |

The cycle: producer waits on empty[stage], calls `arrive_and_expect_tx(bytes)` on
full[stage], issues TMA → smem[stage] (TMA auto-signals full on completion). Consumer
waits on full[stage], computes, then arrives on empty[stage] to free the buffer.

### Phase tracking

Each barrier has a phase bit flipped each cycle; `PipelineState` tracks index + phase,
flipping phase on wrap. **Producer starts at phase=1** (opposite the consumer's phase=0)
because buffers start empty. Wrong phase → instant return on stale data, or deadlock.

### Transaction barriers

`ClusterTransactionBarrier` flips phase only when **both** all thread arrivals **and** all
expected transaction bytes complete. Bytes must be exact:
`size(smem_tile)·sizeof(element)`. Too many → deadlock; too few → corruption. Only **one**
leader thread should call `arrive_and_expect_tx`.

### WGMMA fence/commit/wait (SM90)

WGMMA is async — issued, committed, waited separately, and bracketed by a compiler fence:

```cpp
warpgroup_fence_operand(accum);   // compiler: don't move accum reads before here
warpgroup_arrive();               // wgmma.fence.sync.aligned — smem visible
cute::gemm(tiled_mma, A, B, accum);
warpgroup_commit_batch();         // wgmma.commit_group.sync.aligned
warpgroup_wait<0>();              // wgmma.wait_group.sync.aligned — wait until ≤0 groups
warpgroup_fence_operand(accum);   // compiler: accum now safe
```

Defer waits (`warpgroup_wait<1>`) to overlap independent GEMMs (FlashAttention runs QK and
PV concurrently).

### Blackwell extensions

`tcgen05.commit.mbarrier::arrive` lets the MMA hardware signal barriers directly,
eliminating `wgmma.wait_group` + explicit arrive. 2-SM (`cta_group::2`) MMA needs
cross-SM barrier coordination — always signal on the even-numbered leader SM.

### Fences

| Fence | PTX | Use |
|---|---|---|
| barrier init | `fence.mbarrier_init.release.cluster` | after `mbarrier.init`, before use |
| async shared | `fence.proxy.async.shared::cta` | after STSM/TMA write, before reading |
| async global | `fence.proxy.async.global` | before TMA store |

### CUTLASS pipeline classes

| Class | Full | Empty | Use |
|---|---|---|---|
| `PipelineTmaAsync` | TransactionBarrier | ClusterBarrier | TMA loads (Hopper) |
| `PipelineAsync` | ClusterBarrier | ClusterBarrier | cp.async loads |
| `PipelineTransactionAsync` | TransactionBarrier | ClusterBarrier | non-TMA with tx tracking |
| `PipelineTmaStore` | — | — | TMA stores (commit/wait groups) |
| `PipelineUmmaAsync` | TransactionBarrier | ClusterBarrier | tcgen05 MMA (Blackwell) |

### Common pitfalls

1. Wrong transaction bytes → deadlock or corruption.
2. Missing `fence.mbarrier_init` after init.
3. Missing `fence.proxy.async.shared` after STSM/TMA before GMMA reads.
4. Multiple threads calling `arrive_and_expect_tx` → premature flip.
5. `consumer_release` before `warpgroup_wait<0>` → producer overwrites SMEM still being read.
6. Missing `warpgroup_fence_operand` → compiler reads accumulator before WGMMA completes.
7. Phase-bit confusion (`phase = (total_iters/Stages) % 2`).
8. Producer exits before consumer finishes — must call `producer_tail()`.

---

## 7. Warp specialization

Assign warps to roles so producer warps load stage N+1 while consumer warps compute on
stage N, with no barrier between them: total time ≈ max(load, compute) instead of
load + compute. **Pingpong** (2+ consumers alternating) keeps tensor cores ~100% busy.
**Register redistribution** (`setmaxnreg`) lets producers drop to ~24 regs and hand the
freed registers to consumers (up to 240) for larger accumulator tiles.

### Hopper: warpgroup is the unit

A warpgroup is **always** 4 warps / 128 threads (`NumThreadsPerWarpGroup=128`) — a
hardware requirement of `wgmma.mma_async.sync.aligned`. You cannot break the 4-warp
boundary; asymmetric "2 producer / 6 consumer" splits are impossible. Accumulators are
distributed across all 128 threads' registers (160–256 regs/thread).

| Config | Warps | Producer regs | Consumer regs |
|---|---|---|---|
| 1 prod + 1 cons | 8 (2 WG) | 24–56 | 232–256 |
| 1 prod + 2 cons (pingpong) | 12 (3 WG) | 24–40 | 232–240 |
| 1 prod + 3 cons | 16 (4 WG) | 32 | 160 |

Redistribution in practice (FlashAttention / DeepGemm):

```cpp
if (warp_group_idx == 0) { cutlass::arch::warpgroup_reg_dealloc<24>();  /* TMA loads */ }
else                     { cutlass::arch::warpgroup_reg_alloc<240>();   /* WGMMA    */ }
```

Pingpong uses named barriers so each consumer WG waits on its own barrier and signals the
next, keeping tensor cores fed.

### Blackwell: warp is the unit

`tcgen05.mma` is issued by a **single elected thread** (`if (elect_one_sync())`), not a
cooperative warpgroup. The accumulator goes to **TMEM**, not registers
(`CRegisters = uint32_t[1]` holds a TMEM *address*, not data; TMEM = 128×512×32-bit =
256 KB/SM). This removes the 224+ reg/thread accumulator pressure that dominates Hopper.

CUTLASS SM100 assigns roles per **warp** (32 threads): MMA, Sched, MainloopLoad,
EpilogueLoad, Epilogue(×N). The MMA warp uses 1 thread to issue; the other 31 are free.
`cta_group::1` (1 SM, M=64/128) vs `cta_group::2` (2 SMs cooperate, M=128/256, leader
issues, B read from both CTAs via DSMEM).

### The shift in one table

| Aspect | Hopper SM90 | Blackwell SM100 |
|---|---|---|
| Specialization unit | warpgroup (128 threads) | warp (32 threads) |
| MMA thread requirement | all 128 synchronously | 1 elected thread |
| Accumulator | 128-thread registers | TMEM |
| Consumer reg pressure | high (224–256) | low (TMEM holds accum) |
| Asymmetric allocation | no (multiples of 4 warps) | yes (any warp count/role) |
| MMA commit | `wgmma.commit_group` | `tcgen05.commit` (hardware) |

---

## 8. Persistent kernels & scheduling

Launch exactly `NUM_SMs` CTAs (one per SM); each loops pulling tiles from a scheduler.
Benefits: zero per-tile relaunch overhead, warm L2 across tiles, no pipeline drain/refill
between tiles. The default for SM90+ in all modern reference implementations.

```
launch NUM_SMs CTAs; each CTA: while (scheduler.get_next_tile()) { load; compute; store; }
```

- **DeepGemm** (`common/scheduler.cuh`): CTA i processes tiles i, i+NUM_SMs, i+2·NUM_SMs…;
  both TMA and math warpgroups run the same `get_next_block` loop.
- **Triton** (`09-persistent-matmul.py`): `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)`;
  `flatten=True` overlaps a tile's epilogue stores with the next tile's prologue loads.
- **ThunderKittens LCF** (`prototype/lcf/lcf.cuh`): grid = 132 (H100 SMs), `task_iter` loop,
  `common_setup` maps `task_iter*gridDim.x + blockIdx.x` → tile coords, `num_iters<0` stops.
- **CUTLASS** `StaticPersistentTileScheduler`: linear index strided by total grid size,
  with grid swizzle for L2 locality; grid capped at SM count.

Use persistent kernels always on Hopper/Blackwell, especially with many tiles or deep
pipelines, and combine with **grid/tile swizzling** for L2 reuse. Note: grid swizzling
(reordering which CTA computes which output tile, to improve L2 hit rate) is distinct from
the shared-memory XOR swizzling in §3 — they share a name but solve different problems.

---

## 9. Occupancy tuning

Occupancy = active warps / max warps per SM. Higher occupancy hides latency, but **max is
not always optimal** — fewer warps with more registers can win on compute-bound kernels.

Limiters: registers/thread (H100: 32 regs → 64 warps/100%; 128 → 16 warps/25%; 255 →
8 warps/12.5%), SMEM/block, and block size. Exact per-arch limits live in the
architecture docs.

### `__launch_bounds__` in production

```cpp
__launch_bounds__(threads, 1)        // DeepGemm/flashMLA: exactly 1 block/SM, max registers
__launch_bounds__(NUM_THREADS, 2)    // ThunderKittens light kernel: 2 blocks/SM
__launch_bounds__(maxThreads, minBlocks, maxBlocks)  // CUDA 12+ three-arg form
```

### Register redistribution (SM90+)

Producer 24 / consumer 240 (FlashAttention), or 48 / 224 (DeepGemm), via
`warpgroup_reg_dealloc` / `warpgroup_reg_alloc`. See §7.

### Shared memory carveout

Above 48 KB requires opt-in; production code falls back gracefully if the GPU can't
provide it:

```cpp
if (smem_size >= 48*1024)
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
```

### Programmatic occupancy

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` is used both to **reject** infeasible GEMM
configs (returns 0) and to **size** persistent-kernel grids
(`num_blocks_per_sm * num_sms`). Low occupancy is fine when compute-bound and near peak
(GEMM at 1 block/SM hitting 90%+ TFLOPS; FlashAttention at ~25% occupancy fully utilizing
tensor cores). In ncu, high `stall_not_selected` means too many warps — trade some for
registers.

---

## 10. Reduction & online softmax

### Warp-level (butterfly allreduce)

```cpp
template<int THREADS> struct Allreduce {  // THREADS ∈ {4,8,16,32}
  template<typename T, typename Op> static __device__ T run(T x, Op &op) {
    constexpr int OFFSET = THREADS/2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};
```

FlashAttention uses `Allreduce<4>` because in the WGMMA layout 4 threads share each row.

### Block-level (cross-warp)

Thread-local accumulate → warp shuffle → one float/warp into SMEM → warp 0 reduces →
broadcast via SMEM (FlashInfer RMSNorm, `norm.cuh`).

### Online softmax (FlashAttention `softmax.h`)

Computes softmax incrementally across K-tiles without materializing the N×N matrix. Per
tile: `max_get_scale` (update running max, compute rescale factor) → `rescale_o` (adjust
output accumulator for the new max) → `online_softmax` (apply `exp2`, accumulate row sum).
Uses base-2 (`exp2f`) with fused scale + max subtraction; a layout conversion reshapes the
MMA accumulator fragment so row-wise max/sum map to the thread layout. The full attention
matrix never leaves registers.

### For comparison

Triton fused softmax (`02-fused-softmax.py`): one load, `row - max`, `exp`, `sum`, divide,
one store; whole row must fit in SRAM. ~4x over naive PyTorch (2MN vs 8MN DRAM traffic).
FlashInfer RMSNorm: two-pass (sum-of-squares, then normalize+scale) with vectorized
`vec_t<T,VEC_SIZE>` loads; Hopper variant adds programmatic dependent launch.

---

## 11. Inline PTX reference

### Syntax

```cuda
asm [volatile] ( "template" : outputs : inputs : clobbers );
```

`%0` is the first operand (outputs numbered first, then inputs); `%%` escapes a literal
`%` for special registers. All colons are positional.

### Constraints

| Constraint | PTX | C++ | Notes |
|---|---|---|---|
| `r` | `.u32` | `uint32_t`/`int` | most common |
| `l` | `.u64` | `uint64_t`/pointer | |
| `f` | `.f32` | `float` | |
| `d` | `.f64` | `double` | |
| `h` | `.u16` | `uint16_t` | e.g. multicast mask |
| `n` | — | compile-time int | immediate; template params / `sizeof` only |

Output modifiers: `=` write-only, `+` read-write. C++ type must match constraint size;
one constraint per operand; scalars only.

### volatile and "memory"

Use `asm volatile` whenever the instruction has side effects (stores, barriers,
sync, reading hardware state) so it isn't deleted/reordered. The `"memory"` clobber
prevents reordering across the asm. Empty-asm compiler fence: `asm volatile("" : "+r"(reg) :: "memory")`.

### Scoping

Always wrap multi-instruction blocks in `{ }` to avoid `.reg` name collisions when
inlined. Declare temporaries with `.reg .pred p; .reg .b32 tmp;` etc.

### Predicates

C++ bools can't cross into PTX directly — convert with `setp`:

```cuda
"  .reg .pred p;\n  setp.ne.b32 p, %5, 0;\n  @p ld.global.v4.u32 {...}, [%4];\n"  // guard
"  .reg .pred P1;\n  mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n  selp.b32 %0, 1, 0, P1;\n"  // extract
```

### Instruction quick-reference

```cuda
// cp.async (SM80)
"cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
"cp.async.commit_group;"  /  "cp.async.wait_group N;"

// TMA load/store (SM90)
"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%3,%4}],[%2],%5;"
"cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"   // store
"cp.async.bulk.commit_group;"  /  "cp.async.bulk.wait_group.read %0;"

// WGMMA (SM90)
"wgmma.fence.sync.aligned;"
"wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 {%0,%1},%2,%3,p,...;"
"wgmma.commit_group.sync.aligned;"  /  "wgmma.wait_group.sync.aligned N;"

// mma.sync (SM80)
"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"

// mbarrier
"mbarrier.init.shared::cta.b64 [%1], %0;"
"mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;"
"mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;"   // in a spin loop

// register reconfig (SM90)
"setmaxnreg.inc.sync.aligned.u32 %0;"  /  "setmaxnreg.dec.sync.aligned.u32 %0;"

// fences
"fence.proxy.async.shared::cta;"  /  "fence.proxy.async.global;"  /  "fence.mbarrier_init.release.cluster;"
```

### SMEM pointer conversion

```cuda
uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
// or cute::cast_smem_ptr_to_uint(smem_ptr)
```

### Pitfalls

1. Errors surface at **ptxas**, not C++ compile — the template string isn't parsed earlier.
2. Memory space must be explicit (`ld.global`/`ld.shared`/`ld.local`).
3. Always brace multi-instruction blocks.
4. Match constraint sizes (`r`/u32, `l`/u64, `f`/float) — mismatches are silent.
5. `n` operands must be compile-time constant.
6. Outputs numbered before inputs.
