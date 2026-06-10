# §2.7 Reference-Library Hints for a Hand-Written (non-CUTLASS) Hopper Attention Kernel

Goal: distill a single self-contained `.cu` FA3-equivalent for sm_90a (H200), bf16/fp16, NO external libs
(no CUTLASS/CuTe). torch only in host `mha_fwd`. The libraries below are mined for *readable models* of raw
TMA / WGMMA / mbarrier / online-softmax that we can transcribe into pure inline-PTX + `cuTensorMapEncodeTiled`.

## TL;DR usefulness ranking (for OUR pure-PTX kernel)

1. **ThunderKittens (TK)** — BEST. Pure inline-PTX wrappers for *every* Hopper primitive we need
   (TMA load/store, mbarrier init/arrive/wait, WGMMA mma_async + fence/commit/wait, the 64-bit WGMMA
   shared-mem matrix descriptor encoding, and the HOST-side `cuTensorMapEncodeTiled` recipe). Plus a full
   warp-specialized FA3 kernel (`mha_h100.cu`) and a clean pedagogical one (`mha_h100_lcf.cu`). Almost
   everything we write can be lifted from TK with zero CUTLASS.
2. **DeepGEMM** — VERY USEFUL for kernel *structure* (warp-specialized producer/consumer, register
   reconfig, multi-stage pipeline, persistent scheduler) and for some pure-PTX TMA/mbarrier. BUT its WGMMA
   path and descriptor *bitfield* lean on `cute::` (cutlass) types — use as architecture model, not copy-paste
   for the MMA layer. `sm90_fp8_mqa_logits.cuh` is the closest thing to attention (Q@Kᵀ + per-row reduce).
3. **flashinfer** — USEFUL in two slices: (a) self-contained Ampere-style PTX (`mma.cuh` ldmatrix +
   `mma.sync.m16n8k16`, `cp_async.cuh`, `permuted_smem.cuh`) — perfect for the online-softmax math
   (`prefill.cuh::update_mdo_states`) and a non-WGMMA fallback; (b) its Hopper path is 100% CUTLASS/CuTe
   (collective/pipeline) → architecture reference only.
4. **FlashMLA** — LIMITED. Real sm90 kernel (`splitkv_mla.cuh`) is CUTLASS/CuTe + MLA-specific
   (head_dim 576, decode). `kerutils` has some self-contained sm80 PTX. Mostly inspiration for split-KV /
   combine epilogue, not core attention.
5. **TransformerEngine** — LOW. `fused_attn_*.cu` are cuDNN-graph wrappers (no device kernel of our kind).
   Skip for kernel code; only its docs are tangentially relevant.
6. **cccl (libcu++/cub/thrust docs)** — LOW for the kernel. `cuda::pipeline`/`cuda::barrier` abstract
   LDGSTS/TMA but we want raw PTX. Possibly handy for host-side utilities only.

---

## 1. ThunderKittens — concrete file pointers (PURE PTX, copy-able)

Root: `/data-fast/agent/reference/libraries/ThunderKittens/`

### 1a. TMA load/store PTX — `include/ops/thread/memory/tile/tma.cuh`
- `load_async(ST&dst, GL&src, coord, semaphore&bar)` (L673): the core async copy:
  `cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{coords}],[%2];`
  (5d for swizzled tiles, 4d for non-swizzled; axis==2 only for non-swizzled).
- `store_async` (L104): `fence.proxy.async.shared::cta;` then
  `cp.async.bulk.tensor.{4,5}d.global.shared::cta.tile.bulk_group [%0,{coords}],[%1];` then `store_commit_group()`.
- `prefetch` (L36): `cp.async.bulk.prefetch.tensor...`.
- `store_{add,min,max}_async`: `cp.reduce.async.bulk.tensor...add/min/max` (we likely only need plain store;
  note O output is the only store for fwd attention).
- swizzle coords helper `detail::tma_coords` (L16): for swizzled tiles col index is divided by
  `swizzle_bytes/sizeof(dtype)`.

### 1b. TMA / mbarrier sync PTX — `include/ops/thread/util/tma.cuh` + `.../util/sync.cuh`
- `init_semaphore` (sync.cuh L37): `mbarrier.init.shared::cta.b64 [%0], %1;` with count=thread+txn.
- `invalidate_semaphore`: `mbarrier.inval.shared::cta.b64`.
- `arrive(sem)` (L67): `mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];` (and a `,count` variant L89).
- `wait(sem, phase)` (L106): spin loop `mbarrier.try_wait.parity.shared::cta.b64 P1,[%0],%1; @P1 bra DONE;`.
- `expect_bytes(sem, bytes)` (tma.cuh L30): `mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;` — set
  expected transaction bytes BEFORE issuing the TMA load.
- `store_commit_group` = `cp.async.bulk.commit_group;`; `store_async_wait<N>` = `cp.async.bulk.wait_group N;`;
  `store_async_read_wait<N>` = `cp.async.bulk.wait_group.read N;` (use read-wait before reusing the O smem tile).
- The phase-bit pattern: each pipeline slot has a phase that flips 0/1 each time the slot is reused; `wait`
  spins on `try_wait.parity`. This is THE Hopper double/N-buffer idiom we must replicate.

### 1c. WGMMA PTX — `include/ops/group/mma/base/64x{N}.impl` (+ `base.cuh`, `warpgroup.cuh`)
- `base.cuh` includes one `.impl` per N (16..256). `64x64.impl` L24 shows the canonical bf16 instr:
  `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {32 f32 accum regs}, {4 a_regs}, b_desc, p,1,trans_b,scale_b;`
  - A from registers (`rt_st`): A passed as 4 `uint32_t` regs (RS form), B from shared via 64-bit descriptor.
  - `st_st` variant: both A and B from shared (SS form) — what GEMMs use; for attention S=Q@Kᵀ we want
    Q in regs or smem, second matmul O=P@V uses P (just-computed, in regs → RS form).
  - `setp.ne.b32 p, scale_d, 0` makes accumulate-vs-overwrite a runtime predicate. scale_b ∈ {1,-1}.
- `warpgroup.cuh`:
  - `mma_fence(D&)` (L22): touches every accumulator reg with empty `"+f"` asm + `wgmma.fence.sync.aligned;`
    + `fence.proxy.async.shared::cta;`. MUST bracket WGMMA so the compiler treats accum as live.
  - `mma_commit_group()` (L86) = `wgmma.commit_group.sync.aligned;`.
  - `mma_async_wait<N>()` (L98) = `wgmma.wait_group.sync.aligned N;`.
  - Sequence per K-step batch: fence → (issue WGMMAs) → commit_group → wait_group<0> (or pipeline by leaving
    N>0 groups in flight, as DeepGEMM does).

### 1d. WGMMA shared-memory matrix descriptor — `include/types/shared/descriptor.cuh` (CRITICAL, pure)
- `matrix_descriptor_encode(x) = ((x & 0x3FFFF) >> 4)` (L24).
- `matrix_descriptor_raw(addr, LBO, SBO, swizzle_mode)` (L27): 64-bit field =
  `enc(addr) | enc(LBO)<<16 | enc(SBO)<<32 | swizzle_mode<<62`.
- Swizzle mode codes: 128B→1, 64B→2, 32B→3 (in bits 62-63). For attention bf16 head_dim=128 we want 128B
  swizzle → mode 1, SBO=1024, LBO depends on MN/K-major.
- K-major (M×K for A, N×K for B): LBO "does not matter" (16), SBO = 1024/512/256 for 128/64/32B swizzle.
- MN-major: LBO = 2048*rows/16, SBO = 1024 (for 128B).
- `chunk_descriptor(k)` (L77): advances the descriptor by `enc(k*2048)` (MN) or the 128B-swizzle K-major
  formula `enc((k%4)*32 + (k/4)*(rows/16)*2048)` per K-chunk (K=16 elems for bf16). This is the per-WGMMA
  K-iteration address math we must reproduce.
- This is exactly the DeepGEMM `make_smem_desc`/`make_gmma_desc` logic but WITHOUT cute types — copy THIS.

### 1e. HOST-side TMA descriptor creation — `include/types/global/tma.cuh` (toolkit-only, no CUTLASS)
- `create_tensor_map<ST,axis>(CUtensorMap*, src, b,d,rows,cols)` (L25): builds gmem_shape/stride[],
  smem_shape/stride[], picks `CU_TENSOR_MAP_DATA_TYPE_BFLOAT16`, swizzle `CU_TENSOR_MAP_SWIZZLE_128B`,
  then calls driver API `cuTensorMapEncodeTiled(...)` (L166). Swizzled tile ⇒ tma_dim=5; non-swizzled⇒4.
- `allocate_and_create_tensor_map` (L209): cudaMalloc a `CUtensorMap`, encode on host stack, cudaMemcpy H2D,
  pass the device pointer to the kernel. (FA3/baseline instead pass `const __grid_constant__ CUtensorMap` by
  value — both work; grid_constant is the FA3/DeepGEMM convention.)
- Key constraints baked into asserts: gmem addr 16B-aligned; gmem strides %16==0; smem_shape dims ≤256;
  `smem_shape[0]*sizeof(dtype) ≤ swizzle_bytes`; for 128B swizzle the inner (contiguous) tile dim must be
  64 bf16 elems. THIS dictates how we tile head_dim into smem.
- NOTE: `cuTensorMapEncodeTiled` is in the CUDA driver (`-lcuda`), allowed under toolkit-only rules.

### 1f. Full warp-specialized FA3 kernel — `kernels/attention/mha_h100/mha_h100.cu` (1119 lines)
The closest analog to vLLM FA3. Structure to mirror:
- `__launch_bounds__(NUM_WORKERS*32,1)`, `CONSUMER_WARPGROUPS=3`, `PRODUCER_WARPGROUPS=1` (4 WG = 512 thr).
- Tile dims `fwd_attend_ker_tile_dims<D>`: qo_height=64, kv_height=128, tile_width=D; **stages=4 for D=64,
  stages=2 for D=128** (smem-bound). KV double/quad-buffered.
- Producer WG: `warpgroup::decrease_registers<32>()`, loops issuing `tma::load_async` of K/V tiles into
  ring buffer slots, signaling per-stage `*_smem_arrived` mbarriers (`expect_bytes` then `load_async`).
- Consumer WGs: `warpgroup::increase_registers<160>()`, hold `o_reg`, `max_vec`, `norm_vec` in registers,
  loop over KV stages: `mm_ABt(att, Q, K)` → wait → online softmax → `mma_AB(o_reg, P_bf16, V)` → arrive
  `compute_done`. Online-softmax math at L152-204 (see 1g).
- Epilogue: divide o_reg by norm_vec, store to smem, `tma::store_async(O,...)`, also store `l` (logsumexp).
- Causal handling via `is_causal` template + masking the att_block.

### 1g. Online-softmax math (FA3 style, log2 trick) — `mha_h100_lcf.cu::compute` L64-83
Cleanest, exact transcription target (D=128 → scale = 0.08838834764 * 1.44269504089 = 1/sqrt(128) * log2(e)):
```
att = Q @ Kᵀ                         (WGMMA, f32 accum)
max_last_scaled = max_vec * SCALE
right_fill(att, OOB_cols, -inf)      (causal / seqlen tail masking)
max_vec = rowmax(att, max_vec)       (running max)
max_scaled = max_vec * SCALE
att = exp2(att*SCALE - max_scaled)   (note exp2, not exp; SCALE folds in log2(e))
corr = exp2(max_last_scaled - max_scaled)
norm_vec = norm_vec*corr + rowsum(att)
o_reg = o_reg * corr                 (rescale running output)
P_bf16 = (bf16) att
o_reg += P_bf16 @ V                  (WGMMA)
... finish: o_reg /= norm_vec; store
```
The `mha_h100_lcf.cu` whole file (294 lines) is the single best end-to-end readable model — read it first.

### 1h. Other TK refs
- `kernels/attention/bf16_b300_*` are Blackwell (tcgen05) — ignore for Hopper.
- `prototype/lcf/` is the pipeline template that `mha_h100_lcf.cu` uses (load-compute-finish) — useful for
  understanding the producer/consumer skeleton, but we'll hand-roll, not template-ify.
- `include/ops/group/util/tma_cluster.cuh`, multicast loads — only needed if we use TMA multicast (probably
  not for single-CTA-per-(b,h) attention).

---

## 2. DeepGEMM — file pointers (architecture model; MMA layer uses cute)

Root: `/data-fast/agent/reference/libraries/DeepGEMM/`. Device kernels in `deep_gemm/include/deep_gemm/`.

### 2a. Pure-PTX TMA/mbarrier — `deep_gemm/include/deep_gemm/ptx/tma.cuh`
- `tma_load_1d`, `tma_store_1d` (raw `cp.async.bulk...L2::cache_hint`), `tma_store_wait<N>`,
  `mbarrier_arrive`, `mbarrier_arrive_and_set_tx`, `mbarrier_wait_and_flip_phase` (try_wait spin w/ 0x989680
  timeout const, then `phase ^= 1`). BUT the mbarrier arg type is `cutlass::arch::ClusterTransactionBarrier*`
  and TMA hints are `cute::TMA::CacheHintSm90` — strip those types; the asm itself is clean.
- `tensormap.replace.*` instrs for in-smem descriptor mutation (only if we mutate descriptors at runtime;
  not needed for static-shape attention).

### 2b. WGMMA selector + descriptor — `deep_gemm/include/deep_gemm/mma/sm90.cuh` (USES cute, model only)
- `BF16MMASelector<N>` (L104) maps N→`cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS`. The `.wgmma()` wrapper
  (L85) just calls `MMA::fma(desc_a, desc_b, d..., ScaleOut)`. We can't use cute, but this shows the same
  m64nNk16 bf16 SS instruction family TK exposes directly.
- `make_smem_desc` (L196): writes `cute::GmmaDescriptor` bitfield: `start_address_ = ptr>>4`,
  `layout_type_`, `leading_byte_offset_ = LBO>>4`, `stride_byte_offset_ = SBO>>4`, `base_offset_=0`. Same
  encoding as TK's `matrix_descriptor_raw` — cross-check the two when transcribing.
- `make_gmma_desc<Major,BLOCK_MN,BLOCK_K,Swizzle>` (L245) + `advance_gmma_desc_lo` (L239): runtime K/MN
  address advance. K-major requires 128B swizzle & BLOCK_K=128. Good doc of LBO/SBO meaning (atom = 8×swizzle
  bytes; SBO/LBO are byte strides between atoms on K/MN).

### 2c. Warp-specialized kernel skeleton — `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh` (432 ln)
Best STRUCTURAL model (read even though it's GEMM):
- TMA descriptors passed as `const __grid_constant__ cute::TmaDescriptor` (replace with `CUtensorMap`).
- smem layout: `extern __shared__ __align__(1024) uint8_t smem_buffer[]` then carve D/A/B/barriers by offset.
  **128B-swizzle requires 1024-byte smem alignment** (L94).
- Barriers: `full_barriers[stage]` (init count 1, producer arrives w/ expect_tx), `empty_barriers[stage]`
  (init count = #consumers; consumers arrive to release slot). `fence_barrier_init()` after init, then sync.
- Register reconfig: TMA warpgroup `warpgroup_reg_dealloc<48>()`, math warpgroups `warpgroup_reg_alloc<248>()`.
- Producer: `while scheduler.get_next_block` → for each K-block: `empty_barriers[s]->wait(phase^1)` →
  issue TMAs → `full_barrier.arrive_and_expect_tx(bytes)`. Phase/stage advance via `advance_pipeline`
  (`stage = (stage+1)%kNumStages; phase ^= (stage==0)`).
- Consumer: `full_barriers[s]->wait(phase)` → `warpgroup_arrive()` → loop K issuing WGMMA on advancing
  descriptors → `warpgroup_commit_batch()` → `warpgroup_wait<0>()` → `empty_barrier_arrive(s)`.
- Epilogue: STSM (`SM90_U32x2_STSM_N`) writes f32 accum→bf16 into swizzled smem, then `cute::SM90_TMA_STORE_2D`.
  The swizzle-address math (L307-331) is intricate but exactly what an O-store epilogue needs.
- `cudaGridDependencySynchronize()` (PDL) — programmatic dependent launch; optional perf knob.

### 2d. Attention-shaped kernel — `deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh` (330 ln)
Closest DeepGEMM to attention: computes `[BLOCK_Q*H, D] @ [BLOCK_KV, D] → [BLOCK_Q, BLOCK_KV]` then a
per-row reduce. Shows: separate Q-pipeline and KV-pipeline (different stage counts kNumQStages/kNumKVStages),
WGMMA from two smem descriptors (L257-263), then `__shfl_xor_sync` warp reduction over head dim (L298-304)
and predicated global store. Read for: dual-pipeline structure + how WGMMA f32 accum is consumed by
warp-level reductions (analogous to our rowmax/rowsum).

### 2e. Heuristics (host) — `csrc/jit_kernels/heuristics/sm90.hpp`
Tile-size/stage/multicast selection logic. Useful as a model for how OUR agent should pick BLOCK_M/N/K,
num_stages, num_math_warpgroups per (head_dim, seqlen) config — but it's CUTLASS-typed host code.

---

## 3. flashinfer — file pointers

Root: `/data-fast/agent/reference/libraries/flashinfer/include/flashinfer/`

### 3a. Self-contained Ampere PTX (NO cutlass) — `mma.cuh`, `cp_async.cuh`, `permuted_smem.cuh`
- `mma.cuh`: `ldmatrix_m8n8x4` / `_trans` (`ldmatrix.sync.aligned[.trans].m8n8.x4.shared.b16`),
  `mma_sync_m16n16k16_row_col_f16f16f32` (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, two m16n8
  halves), `m16k16_rowsum_f16f16f32`. This is the pre-Hopper tensor-core path. Use it: (i) as a non-WGMMA
  *fallback / debugging oracle*, (ii) for the small-M decode case where WGMMA (m64) is wasteful.
- `cp_async.cuh`: `cp.async.cg/ca.shared.global[.L2::128B]`, `commit_group`/`wait_group`, predicated loads.
  Ampere async copy — fallback if TMA misbehaves.

### 3b. Online-softmax reference — `attention/prefill.cuh::update_mdo_states` (L1153)
Cleanest mma-fragment-layout online softmax with `math::ptx_exp2` and `sm_scale_log2`. Mirrors TK's math but
on m16n8 fragment layout. Read alongside TK 1g to understand per-thread accumulator indexing
(`s_frag[mma_q][mma_kv][...]`, `o_scale = exp2(m_prev*scale - m*scale)`). Good for verifying our fragment
reduction indexing when we hand-roll rowmax/rowsum on WGMMA f32 accumulators.

### 3c. Hopper path = CUTLASS — `attention/hopper/prefill_sm90.cuh` (+ mainloop.cuh, kernel_traits.cuh,
collective/, `cutlass/pipeline/pipeline.hpp`, `cute/tensor.hpp`). Architecture reference only; do NOT copy.
Useful to see how a production FA3 splits Q-load / KV-mainloop / softmax / correction / epilogue into
collective ops, and the `attention_updater.cuh` (running max/sum object).

---

## 4. FlashMLA — file pointers (limited)

Root: `/data-fast/agent/reference/libraries/FlashMLA/csrc/` (the `cutlass/` subdir is a vendored submodule —
IGNORE all 350k lines of it).
- `sm90/decode/dense/splitkv_mla.cuh` (1355 ln): real sm90 kernel but `#include <cutlass/cutlass.h>` and
  MLA-specific (head_dim 512+64 rope, decode, split-KV with separate combine kernel `smxx/decode/combine/combine.cu`).
  Take only the *split-KV + combine* idea if we ever need it for long-context decode; not core.
- `kerutils/include/kerutils/device/sm80/intrinsics.cuh`: self-contained `cp.async.cg.shared.global.L2::{64,128,256}B`,
  `%smid` read, `trap`. Minor.
- `sm100/*` is Blackwell (tcgen05) — ignore.
Net: low priority; only split-KV/combine architecture inspiration.

---

## 5. TransformerEngine — LOW

- `transformer_engine/common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu`: 51 cuDNN references, it's a
  cuDNN-frontend graph builder — NO hand-written attention device kernel. `fused_attn_fp8.cu` same.
- `transformer_engine/pytorch/attention/` is Python dispatch.
Skip for kernel code. Its docs may have FA background but nothing transcribable.

---

## 6. cccl (libcu++/cub/thrust) — LOW for kernel

`/data-fast/agent/reference/cuda-docs/cccl.md` (1MB). Headings: CUDA C++ Core Libraries, libcu++,
`cuda::memcpy_async` (abstracts LDGSTS/TMA), `cuda::pipeline`, `cuda::barrier`. These are HIGHER-level
abstractions over the exact PTX we are hand-writing. We deliberately bypass them (raw PTX gives control over
phase bits / register reconfig that `cuda::pipeline` hides). Possible host-side utility use only. Not a model
for the device kernel.

---

## 7. Concrete adoption plan for OUR kernel

Pure-PTX building blocks to transcribe (all from TK, cross-checked vs DeepGEMM):
1. Host: `cuTensorMapEncodeTiled` for Q,K,V,O (bf16, 128B swizzle, dim=5) — TK `types/global/tma.cuh`. Pass
   descriptors by `__grid_constant__` value (DeepGEMM convention) or device pointer (TK convention).
2. Device smem: `extern __shared__ __align__(1024) uint8_t[]`, carve Q tile + K/V ring (stages=2 for D=128,
   4 for D=64) + mbarriers.
3. mbarrier: `mbarrier.init` (sync.cuh), `expect_tx`+`cp.async.bulk.tensor.5d...` load, `try_wait.parity`
   wait with flipping phase bit per stage.
4. WGMMA: `wgmma.fence` → `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16` (S=Q@Kᵀ RS-form A-in-regs;
   O=P@V) → `wgmma.commit_group` → `wgmma.wait_group`. Build B-descriptor via TK
   `matrix_descriptor_raw`/`chunk_descriptor` (128B swizzle, mode 1).
5. Online softmax (exp2 + log2(e)*1/sqrt(d) scale, running max/sum, o_reg rescale) per TK 1g / flashinfer 3b.
6. Warp-specialization: 1 producer WG (`reg_dealloc<32-48>`) + N consumer WG (`reg_alloc<160-248>`), exactly
   the DeepGEMM `sm90_bf16_gemm.cuh` skeleton.
7. Epilogue: o_reg/=norm, store smem (STSM optional), `cp.async.bulk.tensor...store` of O; store logsumexp `l`.

Anti-cheating note: every PTX above compiles with toolkit + `-lcuda` only; the WGMMA descriptor + softmax can
be written with zero CUTLASS headers. The ONE host dependency is the CUDA *driver* `cuTensorMapEncodeTiled`
(toolkit-shipped, allowed). DeepGEMM/flashinfer Hopper paths pull cute — keep them as architecture references
only, and verify (nm/ldd) the final `.so` has no cutlass/cute symbols.

Gotchas captured:
- WGMMA is m64 in M — for decode (M small) prefer mma.sync m16n8 (flashinfer mma.cuh) or pad/batch heads into M.
- 128B swizzle forces inner contiguous dim = 64 bf16 elems and 1024B smem alignment; head_dim 128 spans two
  64-elem swizzle atoms (descriptor chunk math handles this).
- Must `fence.proxy.async.shared::cta` between smem writes and TMA store, and `wgmma.fence` + empty-asm touch
  of accumulators around WGMMA or the compiler reorders/spills.
- exp2 (not expf): fold `log2(e)` into the softmax scale; matches FA3 baseline numerics for greedy-decode parity.
- Stage/phase bookkeeping: phase flips only when stage wraps to 0; empty_barrier init count = #consumer warps.
