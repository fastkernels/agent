# CUTLASS / CuTe — distillation for §2.6 (inspiration + run-and-inspect-PTX/SASS)

Goal of §2.6: when our agent is building/optimizing a single self-contained FA3 `.cu`, give it the
ability to (a) READ the load-bearing Hopper primitives in CUTLASS/CuTe source for ideas, (b) BUILD a
small CUTLASS/CuTe FMHA or GEMM, run it, and dump its PTX/SASS as a "golden reference" of how a hand-tuned
kernel issues `wgmma`/`cp.async.bulk.tensor` (TMA), and (c) use CuTeDSL (pip `nvidia-cutlass-dsl` 4.5.2,
verified installed) to JIT-generate a reference and dump its PTX/CUBIN with one line.

IMPORTANT licensing/anti-cheat note: CUTLASS/CuTe must NOT be `#include`d or vendored into our output `.cu`
(constraint forbids CUTLASS/CuTe). We use it ONLY as: (1) source of ideas/algorithm, (2) a PTX/SASS
"oracle" to diff our emitted code against. The PTX patterns (the exact `wgmma.mma_async...`, `cp.async.bulk.tensor`,
`mbarrier`, `wgmma.fence/commit_group/wait_group` sequences) are PTX ISA, free to re-derive in our own inline asm.

Local roots (all absolute):
- Repo: `/data-fast/agent/reference/libraries/cutlass/`
- Combined docs (1.4MB, mostly CHANGELOG): `/data-fast/agent/reference/cuda-docs/cutlass.md`
- Toolchain verified present: nvcc 13.0 (`/usr/local/cuda/bin/nvcc`), cuobjdump, nvdisasm, ncu (`/usr/local/bin/ncu`), compute-sanitizer.
- CuTeDSL: `import cutlass` -> 4.5.2 at
  `/home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/`; `import cutlass.cute` works.

================================================================================
## 1. Where the load-bearing Hopper primitives live (read these for ideas)
================================================================================

### 1a. WGMMA (Hopper warpgroup MMA) — the PTX-emitting layer
- `include/cute/arch/mma_sm90_gmma.hpp` — RAW PTX for every `wgmma.mma_async.sync.aligned.m64nNk16...` shape.
  This is the single most useful file to mirror. Key lines:
  - L53  `asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");`  -> `warpgroup_fence()`
  - L67  `asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");` -> `warpgroup_wait<N>()`
  - L80  `asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");` -> `warpgroup_commit_batch()`
  - L88/L99 `warpgroup_fence_operand(reg)` -> empty `asm volatile("" : "+r"/"+f"(reg) :: "memory")` to pin
    accumulators in registers across the async MMA (CRITICAL correctness pattern — prevents the compiler
    from spilling/reordering acc regs around `wgmma`).
  - L154+ each shape e.g. `wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16`, ...m64n16k16..., n32, n64, n96,
    up to n256; both `_F16F16F16` (fp16 acc) and `_F32F16F16` (fp32 acc) families.
- `include/cute/arch/mma_sm90_gmma_ext.hpp` — extended/larger N shapes.
- `include/cute/atom/mma_traits_sm90_gmma.hpp` — the CuTe "traits" that map each `wgmma` to A/B/C layouts.
  - `_SS` = both operands from SMEM via 64-bit "GMMA descriptor" (`GMMA::smem_desc`); `_RS` = A operand from
    registers, B from SMEM. FMHA uses `_SS` for Q*K (both in smem) and `_RS` for P*V (P=softmax output already
    in registers). RS variants at L979 (`SM90_64x8x16_F32F16F16_RS`), L1038, L1097, etc.
  - `GMMA::Major` (MN vs K major), `GMMA::ScaleIn`, `GMMA::ScaleOut::{One,Zero}` (Zero = first MMA zeroes acc).
  - `GMMA::DescriptorIterator` / `make_gmma_desc<MajorMode>` (L362-370): the SMEM matrix descriptor encoding
    (base addr>>4, leading/stride byte offsets >>4, swizzle bits). We must replicate this descriptor packing
    by hand if we emit `wgmma` ourselves.
  - `GMMA::Layout_MN_*_Atom` / `Layout_K_*_Atom` (referenced L176/L191): canonical swizzled SMEM atom layouts
    that wgmma descriptors require (swizzle 128B/64B/32B). Getting SMEM swizzle right is the #1 correctness gotcha.

### 1b. TMA (Tensor Memory Accelerator) async bulk copy — the PTX-emitting layer
- `include/cute/arch/copy_sm90_tma.hpp` — RAW PTX for `cp.async.bulk.tensor.{1..5}d...`. Key:
  - `SM90_TMA_LOAD_2D::copy(...)` L116: `cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint`
    and L124 the `...shared::cluster...` multicast form.
  - PREFETCH variants `cp.async.bulk.prefetch.tensor.Nd.L2.global` (L89/145/202...).
  - 1D..5D structs (`SM90_TMA_LOAD_1D..5D`), plus store variants in the same file family.
- `include/cute/arch/copy_sm90_desc.hpp` — TMA descriptor (`CUtensorMap`) creation knobs.
- `include/cute/atom/copy_traits_sm90_tma.hpp` — `Copy_Traits<SM90_TMA_LOAD,...>` and `TMA_LOAD_Unpack`
  (L66): the `.with(tma_mbar)` pattern that binds an mbarrier to a TMA op (non-executable -> executable).
  `AuxTmaParams` (L52) holds gmem->tma-coord strides + swizzle. `prefetch_tma_descriptors` is the host-side
  step the kernel calls once at start (used in 88's kernel, L199-200).
- `include/cute/atom/copy_traits_sm90_tma_swizzle.hpp` — swizzle handling for TMA<->smem.

### 1c. Async pipeline (mbarrier producer/consumer state machine) — read for the warp-spec recipe
- `include/cutlass/pipeline/sm90_pipeline.hpp` (the canonical Hopper pipeline). Classes:
  - `PipelineTmaAsync<Stages>` (L271): producer = TMA loads, consumer = MMA warpgroups. API surface our
    own pipeline should imitate (even if we hand-roll mbarriers):
    - `producer_acquire(state)` L425 / `producer_acquire(state, token)` L430 — wait for buffer free.
    - `producer_commit(state, bytes)` L435 — arrive on the "full" mbarrier with expected-tx bytes (TMA fills it).
    - `consumer_wait(state)` L480 / `consumer_try_wait` (returns token) — wait for "full".
    - `consumer_release(state)` L490 — arrive on "empty" mbarrier to free the buffer.
  - `PipelineState<Stages>` (L171): {index, phase, count}; `++` advances stage & flips phase at wrap.
    `make_producer_start_state<Pipeline>()` (L254) seeds producer phase.
  - `PipelineTmaStore` (L655) for the epilogue TMA store.
  - `PipelineAsync<N>` (used in 88 as `MainloopPipelineReducer`) is the cp.async (non-TMA) variant.
- `include/cutlass/pipeline/pipeline.hpp` includes the right arch header. Also see
  `include/cutlass/arch/barrier.h` for `ClusterBarrier`/`NamedBarrier` and `cutlass::arch::NamedBarrier`
  used as the math-warpgroup ordering barrier (`math_wg_order_barrier`) in the FMHA mainloop.

### 1d. TiledMMA / CuTe partitioning helpers (algorithmic glue, do NOT vendor, but mirror the shapes)
- `include/cute/atom/mma_atom.hpp` (`make_tiled_mma`, `partition_fragment_C`, `get_thread_slice`,
  `partition_A/B/C`, `make_fragment_A/B`). `cute::gemm(tiled_mma, A, B, C)` is the call that expands to the
  wgmma loop. We won't use CuTe, but the THREAD-VALUE layout (which lane owns which acc element) is exactly
  the `CLayout_64xN` in the gmma traits — we must reproduce that ownership map for our softmax reduction.

================================================================================
## 2. The Hopper FMHA example (the actual FA3-style kernel to study)
================================================================================
Dir: `/data-fast/agent/reference/libraries/cutlass/examples/88_hopper_fmha/`
(Blackwell version: `examples/77_blackwell_fmha/` — sm100, not our target but similar structure.)

README facts: FMHA fwd = "GEMM - online softmax - GEMM" fusion. fwd supports headdim 32/64/128/256 for
fp16/bf16 (128/256 for fp8); all loads via TMA; headdim 128/256 use warp-specialized COOPERATIVE schedule.
"Forward kernels can come close to FA3." Causal mask fused via `before_softmax` hook in `fmha_fusion.hpp`.

Files (what to read for what):
- `88_hopper_fmha.cu` (1192 lines) — host runner + DISPATCH TABLE. The dispatch (L925-984) tells us exactly
  which tile shapes vLLM-relevant configs map to:
    headdim 32:  `Shape<_64,_128,_32>` KernelTma; `Shape<_128,_64,_32>` cooperative.
    headdim 64:  `Shape<_64,_128,_64>` Tma; `Shape<_128,_64,_64>` cooperative; `_128,_64,_64` ping-pong.
    headdim 128: `Shape<_128,_128,_128>` cooperative; `Shape<_128,_256,_128>` cooperative (fp16 acc AND fp32 acc
                 variants — note fp16-acc QK is a perf trick); `_128,_128,_128` ping-pong.
    headdim 256: `Shape<_128,_128,_256>` cooperative / ping-pong; `_128,_64,_256` cooperative.
  Tile = `Shape<M(qBlk), N(kvBlk), HeadDim>`. Three schedule kinds: KernelTma (simple), KernelCooperative
  (2 MMA WG share a tile), KernelPingpong (2 WG alternate tiles to hide softmax). LATENCY measured with
  cudaEvent_t pair (L443-491) — same event-timing pattern our perf harness needs.
- `collective/fmha_collective_tma_warpspecialized.hpp` (560 lines) — THE MAINLOOP. `compute()` at L340.
  This is the FA3 inner loop. Structure (study lines 393-540):
    * `TiledMmaQK` (SS, fp32 or fp16 acc), `TiledMmaPV = convert_to_gmma_rs(...)` (RS: P from regs).
    * Smem tensors sQ/sK/sV; `partition_A/B`, `make_fragment_A/B`.
    * 3-phase loop: (1) prologue first KV tile, (2) `while(k_tile_count>0)` unmasked body, (3) masked tail
      (for causal). Each body:
        - `pipeline.consumer_wait(smem_pipe_read)` (KV ready in smem)
        - `warpgroup_fence_operand(acc); warpgroup_arrive(); gemm_zero_acc(QK); warpgroup_commit_batch();`
        - `warpgroup_wait<0>(); warpgroup_fence_operand(acc);`  (drain QK)
        - `softmax.step(...)`  (online rescale + exp2 + rowsum, see §3)
        - `make_acc_into_op<Element>(acc_qk, LayoutA_TV)` — repack fp32 acc -> fp16 P in the RS A-fragment layout.
        - `warpgroup_arrive(); cute::gemm(tiled_mma_pv, P, V, acc_pv); warpgroup_commit_batch();`
        - `pipeline.consumer_release(...)` (free KV buffers — note 2 releases per body for double-buffered KV).
    * `math_wg_order_barrier.wait()/arrive()` (NamedBarrier) serializes the two math warpgroups in
      cooperative mode so their softmaxes don't collide — the warp-specialization sync skeleton.
    * Uses SEPARATE pipelines: `MainloopPipelineQ` (Q loaded once), `MainloopPipeline` (KV streamed),
      `MainloopPipelineReducer` (PipelineAsync<2>).
- `collective/fmha_collective_softmax.hpp` (305 lines) — see §3, the online-softmax math. HIGH VALUE.
- `collective/fmha_collective_load.hpp` / `fmha_collective_tma.hpp` — TMA producer warp (issues the loads,
  arrives on full mbarrier with byte counts `kInnerLoadBytes`/`kOuterLoadBytes`).
- `collective/fmha_epilogue.hpp` — O = acc/rowsum then TMA store; LSE write.
- `collective/fmha_fusion.hpp` — `before_softmax(acc, coords, problem)` mask/activation hook;
  `get_unmasked_trip_count` / `get_masked_trip_count` split the KV loop for causal.
- `collective/fmha_common.hpp` — `layout_acc_mn`, `reduction_target_n`, `convert_to_gmma_rs`,
  `make_acc_into_op`, `gemm_zero_acc`. These are the layout helpers that define WHICH lanes hold a row
  (needed to do the `__shfl_xor_sync` rowmax/rowsum correctly).
- `kernel/fmha_kernel_tma_warpspecialized.hpp` (422 lines) — top kernel `operator()` (L162): sets up warp
  ROLES (`WarpGroupRole::{Producer,Consumer0,Consumer1}`, `ProducerWarpRole::{LoadQ,LoadKV,Reducer,
  MainloopEpilogue}`), pipeline params (transaction_bytes, is_leader, num_consumers), then branches each warp
  into producer (TMA) vs consumer (MMA) code. The `if (warp_idx==0 && lane_predicate)
  prefetch_tma_descriptors(...)` (L199) one-time setup. This is the warp-specialization template to copy.
- `kernel/fmha_tile_scheduler.hpp` — CTA->(batch,head,qblock) mapping (persistent-ish scheduler).
- `reference/fmha_reference.hpp` + `reference/reference_abs_error.hpp` — a NON-fused reference + an abs/rel
  error comparator. Directly reusable idea for our correctness gate.

================================================================================
## 3. Online-softmax math (fmha_collective_softmax.hpp) — copy this exactly
================================================================================
`CollectiveSoftmax<ElementAccumulator, Fusion, Params>`. Uses log2 domain: `params.scale_softmax_log2`
(= softmax_scale * log2(e)). exp via `exp2f` (and a `ex2.approx.f16` inline-asm path for fp16 acc, L67).
- `init(acc_pv, tiled_mma_pv)` -> state = (s_max[rows], a_sum[rows]) fragments sized by row-count of the
  PV-acc MN layout.
- `step(...)` (fused QK+PV variant, L208-264) per KV tile, per row i:
    1. `s_max_prev = s_max`
    2. row max over local acc columns (linear), then cross-lane reduce: `__shfl_xor_sync` over
       `reduction_target_n(tiled_mma)` strides (this is the lanes-in-a-row reduction; rank may be >1).
    3. `scale = scale_softmax_log2; scale_max = scale*local_max(0 if -inf)`
       `P_ij = exp2(scale*S_ij - scale_max)`  (the numerically-stable exp2)
    4. correction `scale_pv = exp2((s_max_prev - s_max_cur)*scale)`; `a_sum *= scale_pv`;
       `acc_pv *= scale_pv` (rescale the running O accumulator) — THE flash rescale.
    5. `a_sum += rowsum(P_i)`.
- `step_interleave_begin`/`step_interleave_step` — variant that does the acc_pv rescale BEFORE the next
  QK to overlap better (ping-pong).
- `tail(state, acc_pv, tiled_mma_pv)` (L267): final cross-lane sum reduce of a_sum, then
  `inv_sum = 1/sum` (guard sum==0/nan -> 1), `O *= rp_dropout*inv_sum`, and
  `lse = s_max*scale_softmax + log(sum)` (guard -> +inf). Returns LSE.
Gotchas to mirror: -inf rows -> treat max as 0; sum==0/nan -> inv=1 & lse=+inf; the `reduction_target_n`
stride pattern is what makes the shuffle reduce correct for the wgmma C-layout (don't hardcode 4 — it depends
on tile N / atom).

================================================================================
## 4. Recipe A — BUILD + RUN + DUMP PTX/SASS of the C++ CUTLASS example (golden oracle)
================================================================================
Why: gives us a hand-tuned FA3-class `.cubin` whose SASS shows the IDEAL `HGMMA`/`UTMALDG` issue cadence,
register usage, and `BAR`/`mbarrier` patterns to diff our kernel against (feeds §2.3 SASS analysis).

Fastest path — compile a single example translation unit directly with nvcc (no full CMake), arch sm_90a:
```
NVCC=/usr/local/cuda/bin/nvcc
CUTLASS=/data-fast/agent/reference/libraries/cutlass
$NVCC -std=c++17 -arch=sm_90a -O3 --expt-relaxed-constexpr --use_fast_math \
  -I$CUTLASS/include -I$CUTLASS/tools/util/include \
  -I$CUTLASS/examples/88_hopper_fmha \
  --keep --keep-dir /tmp/fa3oracle \         # keeps .ptx
  -cubin -o /tmp/fa3oracle/fmha.cubin \
  $CUTLASS/examples/88_hopper_fmha/88_hopper_fmha.cu
# (-cubin to get device code only; or -c for object, or full link to run.)
```
Notes/gotchas:
- MUST be `sm_90a` (the "a" arch) or `wgmma`/TMA won't compile (CUTE_ARCH_MMA_SM90A_ENABLED guards).
- 88's CMake sets `COMPILE_FLAGS "--use_fast_math"` on the .cu — replicate.
- This TU is heavy/templated; compile is slow and RAM-hungry. To shrink, the agent can write a tiny
  driver that instantiates ONLY the one config it cares about (e.g. headdim128 `Shape<_128,_128,_128>`
  cooperative) instead of the full dispatch — copy the `Operation`/`run<>` instantiation from the .cu.
- Easier alternative: use upstream CMake to build just this target:
  `cmake -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_TESTS=OFF ...; make 88_hopper_fmha` then run the binary
  for latency, but for PTX/SASS the direct nvcc `--keep`/`-cubin` is simpler.

Dump PTX/SASS from the cubin/object (feeds SASS comparator §2.3):
```
cuobjdump -sass /tmp/fa3oracle/fmha.cubin            # human SASS
cuobjdump -ptx  /tmp/fa3oracle/fmha.cubin            # embedded PTX
nvdisasm -c -g  /tmp/fa3oracle/fmha.cubin            # control-flow + line info
cuobjdump -res-usage /tmp/fa3oracle/fmha.cubin       # regs/smem per kernel
# Count the load-bearing instructions to compare cadence vs our kernel:
cuobjdump -sass fmha.cubin | grep -cE 'HGMMA|QGMMA|GMMA'    # tensor-core MMA count
cuobjdump -sass fmha.cubin | grep -cE 'UTMALDG|UBLKCP'      # TMA bulk loads
cuobjdump -sass fmha.cubin | grep -cE 'BAR\.|ELECT|BMOV'    # barrier/mbarrier ops
```
The same commands run on OUR `.cu` cubin give a direct A/B diff: if the oracle issues N HGMMA back-to-back
between barriers and ours interleaves a BAR each time, that's the actionable hint.

Also can profile the oracle binary with NCU to get a target roofline (feeds §2.1):
`ncu --set full --kernel-name-base demangled -o oracle_prof ./88_hopper_fmha <args>`.

================================================================================
## 5. Recipe B — CuTeDSL (Python, pip 4.5.2) to JIT a reference and dump PTX/CUBIN
================================================================================
Why: zero C++ build; one decorator flips PTX/CUBIN dumping on. Great for quickly generating a "what does an
ideal wgmma/TMA GEMM look like" PTX without writing CMake. The CuTeDSL emits the SAME `wgmma`/`cp.async.bulk`
PTX as the C++ path.

Canonical one-liner (from `examples/python/CuTeDSL/cute/notebooks/hello_world.ipynb`):
```python
import cutlass.cute as cute
from cutlass.cute import KeepPTX, KeepCUBIN
compiled = cute.compile[KeepPTX, KeepCUBIN](my_jit_fn, *args)   # dumps .ptx/.cubin during compile
compiled(*args)                                                  # run
```
Env-var form used by the GEMM examples (grouped_gemm.py L2224):
`CUTE_DSL_KEEP_PTX=1 python ...` -> selects `cute.compile[cute.KeepPTX()]`.
Other useful env: `CUTE_DSL_CACHE_DIR=/tmp/cutecache` (where caches/dumps land, added in 4.3.2).

Ready-made Hopper kernels to compile & dump (in `examples/python/CuTeDSL/cute/hopper/kernel/`):
- `attention/fmha.py` (2611 lines) — `HopperFusedMultiHeadAttentionForward`, an FA-style fwd kernel with
  TMA + WGMMA, optimized mbarrier wait (`_use_optimized_mbarrier_wait`, L158). Closest CuTeDSL analog to our
  target. Builds TMA atoms via `cute.nvgpu.cpasync.make_tiled_tma_atom` / `CopyBulkTensorTileS2GOp` (store),
  two TiledMmas (qk, pv).
- `dense_gemm/dense_gemm.py` — clean Hopper WGMMA GEMM. Key building-block calls to learn the API:
    * `sm90_utils.make_trivial_tiled_mma(...)` (L319) using `a_layout.sm90_mma_major_mode()`.
    * `cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)` (L532), `cute.nvgpu.cpasync.tma_partition(...)` (L682).
    * `tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False/True)` then
      `cute.nvgpu.warpgroup.fence(); cute.gemm(...); commit_group(); wait_group(k_pipe_mmas)` (L790-860) —
      EXACTLY the fence/commit/wait skeleton from §1a, but in Python.
    * Epilogue `StMatrix8x8x16bOp` + TMA store. SMEM layouts via `sm90_utils.make_smem_layout_a/b/epi`.
  `import cutlass.utils.hopper_helpers as sm90_utils` is the helper module.
- `dense_gemm/dense_gemm_persistent.py`, `dense_gemm_fp8_2xacc.py` (fp8 2x-acc trick), `grouped_gemm/`.

Run skeleton (most examples have `if __name__=="__main__":` + argparse): build torch tensors,
`from cutlass.cute.runtime import from_dlpack`, `cute.compile(kernel, *cute_tensors, stream)`, then call.
For PTX: wrap with `cute.compile[KeepPTX, KeepCUBIN]`. The resulting `.ptx`/`.cubin` -> feed to cuobjdump/
nvdisasm exactly as in Recipe A.

Other CuTeDSL tutorials worth scraping (`examples/python/CuTeDSL/dsl_tutorials/`):
- `inline_ptx.py` — how CuTeDSL inlines raw PTX via `llvm.inline_asm` + `@dsl_user_op` (template for any
  instruction not exposed). Mirrors what WE do in our `.cu` with `asm volatile`.
- `print_latex.py` — `cute.print_latex(layout/tiled_mma/tiled_copy)` renders the thread-value layout as a
  diagram. Use to UNDERSTAND the wgmma C-layout (which lane owns which acc element) so our hand-written
  softmax shuffle reduction matches.
- `smem_allocator.py`, `dynamic_smem_size.py`, `cooperative_launch.py`, `programmatic_dependent_launch.py`.

================================================================================
## 6. Concrete things to adopt in OUR agent (§2.6 deliverable)
================================================================================
1. Ship a "cutlass_oracle" tool: given a target config (headdim, dtype, causal, qblk/kvblk), it (a) emits a
   minimal driver instantiating the matching 88_hopper_fmha `Operation`, (b) nvcc `-arch=sm_90a --keep -cubin`,
   (c) cuobjdump `-sass/-ptx/-res-usage`. Cache by config. Output = instruction histogram (HGMMA/UTMALDG/BAR
   counts) + reg/smem usage as the "target profile".
2. A SASS DIFF helper: run the same cuobjdump histogram on OUR kernel's cubin and report deltas (e.g.
   "oracle 0 spills, ours 12 STL/LDL"; "oracle 8 HGMMA/loop vs ours 8 but with extra BAR" ). Feeds §2.3.
3. A CuTeDSL "quick oracle": `CUTE_DSL_KEEP_PTX=1` + `cute.compile[KeepPTX,KeepCUBIN]` on
   `hopper/kernel/attention/fmha.py` or `dense_gemm.py` for a fast PTX reference when C++ build is too slow.
4. Lift the ONLINE-SOFTMAX recipe verbatim (log2 domain, exp2, prev-max correction `exp2((m_prev-m)*scale)`,
   acc rescale, tail inv_sum + lse) into our kernel — it's the exact FA3 math; matches numerics closely.
5. Lift the WARP-SPECIALIZATION skeleton: producer warp = TMA + arrive(full,bytes); consumer WGs = wgmma loop
   with consumer_wait/release; NamedBarrier to order the two math WGs. mbarrier phase/parity from PipelineState.
6. Lift the fence/commit/wait_group + `warpgroup_fence_operand` accumulator-pinning pattern around every
   `wgmma` (correctness-critical) and the `wgmma.fence` before the first MMA of a group.
7. Use `print_latex` on the relevant TiledMma to derive the lane->acc-element ownership so our cross-lane
   softmax reductions use the right `__shfl_xor_sync` strides (don't hardcode).

================================================================================
## 7. Gotchas / cautions
================================================================================
- Anti-cheat: NEVER `#include` cute/cutlass in the output `.cu`; the oracle stays a separate build artifact.
  The PTX strings (`wgmma.mma_async...`, `cp.async.bulk.tensor...`, `mbarrier.*`) are ISA, fine to re-emit.
- SMEM swizzle + GMMA descriptor packing is the top correctness risk. CuTe hides it in
  `Layout_*_Atom`/`make_gmma_desc`; if we hand-roll, replicate the 128B/64B/32B swizzle and the
  `desc = (addr>>4) | (lbo>>4)<<16 | (sbo>>4)<<32 | swizzle<<62`-style encoding exactly.
- 88's full .cu is slow/large to compile — instantiate one config for the oracle.
- TMA needs a `CUtensorMap` built on host (driver API) + `prefetch_tma_descriptors` once; mbarrier
  expected-tx bytes must equal the exact TMA transfer size or consumer_wait hangs/races (compute-sanitizer
  `--tool synccheck` catches this — ties to §2.2).
- fp16-acc QK variant (`Shape<_128,_256,_128>` "acc fp16") is a real perf lever in 88 but changes numerics;
  only use if greedy decode still matches.
- CuTeDSL major version moves fast (4.5.2 here); some `cute.arch` calls now take string literals not enums
  (4.4.0 CHANGELOG) — match the installed version's API when scripting.
- `cute.print_latex` / layout APIs were refactored in 4.1.0 ("Factor out print_latex"); behavior matches 4.5.x.
