<!-- originally from https://github.com/xinhaoc/ferret/blob/main/docs/design-space.md -->
# GEMM Kernel Design Space

Complete optimization dimensions for GEMM kernels on modern NVIDIA GPUs.
Each candidate is a JOINT configuration — all dimensions together define one kernel.

> **Architecture scope**: This reference is written around the Hopper/Blackwell
> tensor-core GEMM path. Many dimensions (TMA, TMEM, clusters, DSMEM, multicast,
> cluster-scope barriers, `wgmma`/`tcgen05`) **do not exist on Ampere** and have no
> effect there. See the [Architecture Applicability Matrix](#architecture-applicability-matrix)
> before applying a candidate to an SM80 target.

---


## 1. Problem Formulation

- **transpose_a**: whether A is transposed (N or T)
- **transpose_b**: whether B is transposed (N or T)
- **swap_ab**: compute C^T = B^T × A^T instead of C = A × B (swaps role of M and N)
- **split_k**: decompose K across multiple CTAs (K-parallel)
- **split_k_factor**: number of K-splits (1, 2, 4, ...)
- **split_k_reduction**: how partial results combine (separate reduce kernel, in-place atomics, cooperative reduction via SMEM/DSMEM)


## 2. Tiling

- **block_m**: tile M dimension per CTA
- **block_n**: tile N dimension per CTA
- **block_k**: tile K dimension per pipeline stage
- **partial_tile_handling**: edge tiles where problem doesn't divide evenly (TMA OOB zero-fill, separate edge kernel, pad inputs, mask in epilogue)
- **tile_shape_selection**: how tile shapes are chosen (fixed, runtime heuristic, autotuned)


## 3. Launch Configuration

- **grid_dim**: number of thread blocks [x, y, z]
- **block_dim**: threads per block [x, y, z]
- **cluster_dim**: CTAs per cluster [x, y, z]
- **launch_bounds**: compiler hint [max_threads_per_block, min_blocks_per_sm]
- **dynamic_smem_size**: shared memory requested at launch
- **launch_api**: <<<>>> vs cudaLaunchKernelEx (required for clusters)


## 4. Grid & Scheduling

- **grid_strategy**: non-persistent, persistent (grid = SMs), multi-tile (grid < tiles)
- **tile_ordering**: linear, swizzled/grouped for L2 locality (GROUP_SIZE_M in Triton)
- **tile_assignment**: static (blockIdx mapping) vs dynamic (atomic counter, CLC work-stealing)
- **tiles_per_cta**: fixed 1, fixed N, or dynamic
- **clc_scheduling**: cluster launch control for dynamic work distribution
- **target_occupancy**: desired resident CTAs per SM (1, 2, ...)
- **wave_quantization**: grid size chosen to minimize partial waves


## 5. Pipeline

- **num_stages**: pipeline depth (SMEM buffer count for overlapping loads + compute)
- **split_ab_pipeline**: independent stage counts for A and B operands
- **stages_a / stages_b**: per-operand depth (if split)
- **pipeline_type**: simple circular, ping-pong, producer-consumer
- **pipeline_overlap**: what operations overlap (TMA↔MMA, MMA↔epilogue, TMA↔MMA↔epilogue)
- **pipeline_warmup**: how many stages filled before MMA starts
- **pipeline_drain**: how tail iterations are handled


## 6. K-loop Structure

- **k_unroll**: unroll factor (1, 2, 4, 8)
- **k_block_grouping**: K-blocks processed per barrier cycle
- **k_loop_structure**: for-loop, while-loop with predicate, fully unrolled (small K)
- **software_pipelining_depth**: independent MMA operations in flight (distinct from SMEM stages)
- **inner_loop_unroll**: unroll within a single K-block (MMA instructions per block_k)


## 7. Threads & Warp Specialization

- **num_threads**: total threads per CTA
- **num_warps**: derived from threads (threads/32)
- **warp_specialization**: whether warps have dedicated roles vs all-participate
- **warp_roles**: role assignment (TMA producer, MMA compute, epilogue consumer, idle/helper)
- **num_tma_warps**: warps dedicated to TMA loads
- **num_mma_warps**: warps dedicated to MMA compute
- **num_epilogue_warps**: warps dedicated to epilogue stores
- **role_overlap**: whether warps handle multiple roles (e.g., MMA warp also does epilogue)
- **thread_mapping**: how threads map to output elements (1D, 2D, warp-tiled)


## 8. MMA (Matrix Multiply-Accumulate)

- **mma_instruction**: tcgen05.mma, wgmma.mma_async, mma.sync, wmma
- **mma_shape**: per-instruction [M, N, K] (e.g., 128×256×16)
- **mma_mode**: normal (operands from SMEM), .ws (A from TMEM, B from SMEM)
- **cta_group**: 1 (single CTA) or 2 (cooperative 2-SM MMA)
- **accumulator_type**: FP32, FP16, BF16
- **mma_issue_thread**: which thread(s) issue MMA (lane 0, all lanes predicated, elected)
- **mma_commit**: how MMA completion is signaled to barriers


## 9. TMA (Tensor Memory Accelerator)

- **tma_enabled**: use TMA vs manual loads (cp.async, LDG, LDGSTS)
- **tma_dim_a**: descriptor dimensionality for A (1D, 2D, 3D, 4D, 5D)
- **tma_dim_b**: descriptor dimensionality for B
- **tma_box_size_a**: TMA load box dimensions for A
- **tma_box_size_b**: TMA load box dimensions for B
- **tma_oob_fill**: out-of-bounds fill (zero, NaN)
- **tma_cache_hint_a**: L2 cache hint for A (EVICT_NORMAL, EVICT_FIRST, EVICT_LAST, STREAMING)
- **tma_cache_hint_b**: L2 cache hint for B
- **tma_prefetch**: prefetch descriptor (prefetch.tensormap)
- **tma_store**: use TMA for output stores (vs thread-driven STG)
- **tma_load_order**: order of issuing A vs B loads per stage (A-first, B-first, interleaved)


## 10. Shared Memory (SMEM)

- **smem_total**: total shared memory per CTA
- **smem_layout_a**: how A is laid out in SMEM
- **smem_layout_b**: how B is laid out in SMEM
- **smem_swizzle_a**: swizzle mode for A (NONE, 32B, 64B, 128B)
- **smem_swizzle_b**: swizzle mode for B
- **smem_padding**: padding to avoid bank conflicts
- **smem_store_buffer**: SMEM staging for TMA store epilogue (if used)
- **smem_store_stages**: epilogue store buffer count (if TMA store)
- **smem_allocation**: static (compile-time) vs dynamic (launch-time)


## 11. Tensor Memory (TMEM)

- **tmem_cols**: columns allocated for accumulator
- **tmem_buffering**: single, double, triple buffered
- **tmem_load_width**: columns per epilogue load instruction (x8, x16, x32)
- **tmem_release_timing**: when TMEM is signaled empty (after epilogue, before epilogue during async store)
- **tmem_layout**: mapping of accumulator elements to TMEM rows/cols
- **tmem_a_cols**: columns for A operand (if .ws mode)
- **tmem_scratch**: TMEM used for non-accumulator purposes (descriptors, small tables)


## 12. Epilogue

- **epilogue_type**: thread-driven STG, TMA store via SMEM staging, warp-cooperative
- **epilogue_threads**: how many threads participate in epilogue
- **epilogue_warps**: dedicated epilogue warps vs reusing other warps
- **epilogue_vectorization**: store width (32b, 64b, 128b)
- **epilogue_coalescing**: how stores are ordered for memory coalescing
- **epilogue_conversion**: accumulator→output type conversion (FP32→BF16, FP32→FP8, etc.)
- **epilogue_fusion**: fused post-ops (bias add, ReLU, GELU, residual add, quantization)
- **epilogue_overlap**: overlap with next tile (epilogue of tile N runs during TMA of tile N+1)
- **epilogue_batching**: TMEM load grouping (load all then convert, or load-convert-store interleaved)


## 13. Cluster & Multicast

- **cluster_size**: total CTAs (1, 2, 4, 8, 16)
- **cluster_layout**: [M_cluster, N_cluster] arrangement
- **multicast_a**: TMA multicast for A operand
- **multicast_b**: TMA multicast for B operand
- **multicast_type**: manual (multicast::cluster PTX), cta_group::2 cooperative TMA
- **dsmem_enabled**: distributed shared memory for cross-CTA data sharing
- **dsmem_usage**: what data travels via DSMEM (partial results, operand tiles, barriers)
- **cluster_sync_points**: where cluster-wide sync occurs (init, per-tile, exit)


## 14. Barrier & Synchronization

- **num_barriers**: mbarrier objects used (max 16 per CTA)
- **barrier_roles**: which barriers for which purpose (full per stage, empty per stage, TMEM full, TMEM empty, cluster sync, ...)
- **barrier_scope**: CTA-scope (.shared::cta) vs cluster-scope (.shared::cluster)
- **barrier_protocol**: arrive_expect_tx, arrive_one, arrive_and_expect_tx
- **barrier_init_count**: arrival count per barrier
- **barrier_wait_type**: try_wait spin, try_wait with nanosleep, blocking wait
- **fence_type**: proxy fence (.async), memory fence (.cta/.gpu/.sys)
- **fence_placement**: before MMA, after TMA, after epilogue store
- **commit_strategy**: how MMA completion signals barrier (per K-block, per unroll group, per tile)
- **sync_granularity**: per-stage, per-K-block, per-tile, per-cluster


## 15. Descriptor Management

- **smem_descriptor_encoding**: how SMEM descriptors are built (SBO, LBO, swizzle, version)
- **descriptor_computation**: per-stage arithmetic, shfl broadcast, precomputed in registers
- **instruction_descriptor**: compile-time constant vs runtime dynamic
- **tma_descriptor_storage**: **grid_constant**, global memory, SMEM cached


## 16. Memory Policies

- **l2_policy_a**: L2 cache promotion for A (NONE, 128B, 256B)
- **l2_policy_b**: L2 cache promotion for B
- **l2_persistence**: pin data in L2 via stream attributes (cudaStreamSetAttribute)
- **l2_access_window**: L2 persistence window size
- **prefetch_strategy**: TMA descriptor prefetch, software prefetch ahead of pipeline
- **memory_ordering**: relaxed, acquire/release — scope of memory operations


## 17. Registers & Compiler

- **max_registers**: per-thread register limit (auto, 32, 64, 128, 255)
- **fast_math**: --use_fast_math (faster but less precise intrinsics)
- **ptx_optimization**: -Xptxas -O{level} (ptxas optimization aggressiveness)
- **expensive_optimizations**: -Xptxas --allow-expensive-optimizations
- **inline_ptx**: hand-written PTX for critical sections
- **pragma_unroll**: explicit unroll hints on specific loops
- **restrict_pointers**: **restrict** for pointer aliasing hints
- **const_memory**: **constant** for read-only data
- **volatile**: where volatile is used/avoided


## 18. Data Layout

- **a_layout**: row-major, column-major, custom tiled
- **b_layout**: row-major, column-major, custom tiled
- **c_layout**: output layout
- **a_alignment**: memory alignment of A pointer
- **b_alignment**: memory alignment of B pointer
- **data_type_in**: input precision (FP16, BF16, FP8, TF32, FP32, INT8)
- **data_type_out**: output precision
- **data_type_accum**: accumulator precision


## 19. Numeric

- **accumulation_order**: sequential, pairwise, Kahan compensated
- **rounding_mode**: RN (nearest), RZ (toward zero), RU, RD
- **denormal_handling**: flush-to-zero vs IEEE compliant
- **mixed_precision**: input vs accumulator vs output precision combinations

---


## Architecture Applicability Matrix

Not every dimension exists on every architecture. The table below shows, per section,
whether a dimension is **supported as-is**, uses a **different mechanism**, or is
**N/A** (the hardware feature does not exist, so the dimension has no meaning).

Legend: ✅ supported · ⚠️ different mechanism / partial · ❌ N/A (no hardware support)

| Section / Dimension                  | Ampere (SM80)            | Hopper (SM90)             | Blackwell (SM100)         |
| ------------------------------------- | ------------------------ | ------------------------- | ------------------------- |
| **1. Problem formulation**            | ✅                       | ✅                        | ✅                        |
| split_k / split_k_reduction          | ✅ (SMEM/atomic reduce)  | ✅ (+DSMEM reduce)        | ✅ (+DSMEM reduce)        |
| **2. Tiling**                         | ✅                       | ✅                        | ✅                        |
| partial_tile (TMA OOB zero-fill)      | ❌ (mask/pad/edge kernel)| ✅                        | ✅                        |
| **3. Launch config**                  | ✅                       | ✅                        | ✅                        |
| cluster_dim                           | ❌ (always 1)            | ✅                        | ✅                        |
| launch_api (cudaLaunchKernelEx)       | ⚠️ (not for clusters)    | ✅                        | ✅                        |
| **4. Grid & scheduling**              | ✅                       | ✅                        | ✅                        |
| clc_scheduling                        | ❌                       | ⚠️ (limited)              | ✅ (CLC work-stealing)    |
| persistent / wave_quant / swizzle     | ✅                       | ✅                        | ✅                        |
| **5. Pipeline**                       | ✅ (cp.async stages)     | ✅                        | ✅                        |
| producer-consumer / ping-pong         | ⚠️ (no warp-spec HW)     | ✅                        | ✅                        |
| **6. K-loop structure**               | ✅                       | ✅                        | ✅                        |
| **7. Warp specialization**            | ❌ (all-participate)     | ✅ (setmaxnreg)           | ✅ (setmaxnreg)           |
| TMA-producer / MMA-consumer roles     | ❌                       | ✅                        | ✅                        |
| **8. MMA instruction**                | `mma.sync`, `wmma`       | `wgmma.mma_async` (+mma.sync)| `tcgen05.mma` (+wgmma)|
| mma_mode `.ws` (A from TMEM)          | ❌                       | ❌                        | ✅                        |
| cta_group::2 (2-SM MMA)               | ❌                       | ❌                        | ✅                        |
| accumulator in registers vs TMEM      | registers                | registers                 | TMEM                      |
| **9. TMA**                            | ❌ (use cp.async/LDGSTS) | ✅                        | ✅ (enhanced)             |
| tma_store / multicast / prefetch      | ❌                       | ✅                        | ✅                        |
| **10. SMEM**                          | ✅                       | ✅                        | ✅                        |
| smem_swizzle (128B etc.)              | ⚠️ (manual/XOR swizzle)  | ✅ (TMA-aware)            | ✅ (TMA-aware)            |
| max SMEM/CTA                          | 164 KB                   | 228 KB                    | 228 KB                    |
| **11. TMEM**                          | ❌                       | ❌                        | ✅ (256 KB/SM)            |
| all tmem_* dimensions                 | ❌                       | ❌                        | ✅                        |
| **12. Epilogue**                      | ✅ (thread-driven STG)   | ✅ (+TMA store)           | ✅ (+TMA store, TMEM load)|
| epilogue_fusion                       | ✅                       | ✅                        | ✅                        |
| epilogue from TMEM                    | ❌                       | ❌                        | ✅                        |
| **13. Cluster & multicast**           | ❌                       | ✅                        | ✅ (up to 16, enhanced)   |
| dsmem_enabled                         | ❌                       | ✅                        | ✅                        |
| **14. Barrier & sync**                | ⚠️ (async barriers only) | ✅ (transaction barriers) | ✅ (+TMEM barriers)       |
| barrier_scope cluster (.shared::cluster)| ❌                     | ✅                        | ✅                        |
| arrive_expect_tx (byte-tracking)      | ❌                       | ✅                        | ✅                        |
| **15. Descriptor mgmt**               | ❌ (no SMEM descriptors) | ✅ (wgmma descriptors)    | ✅ (+instruction desc.)   |
| tma_descriptor_storage                | ❌                       | ✅                        | ✅                        |
| **16. Memory policies**               | ✅                       | ✅                        | ✅                        |
| l2_persistence / access_window        | ✅                       | ✅                        | ✅                        |
| **17. Registers & compiler**          | ✅                       | ✅                        | ✅                        |
| max_registers (255 cap)               | ✅                       | ✅                        | ✅                        |
| **18. Data layout**                   | ✅                       | ✅                        | ✅                        |
| data_type FP8                         | ❌                       | ✅                        | ✅                        |
| data_type FP4 / FP6 / MX-scaled       | ❌                       | ❌                        | ✅                        |
| data_type TF32 / BF16 / INT8 / INT4   | ✅                       | ✅ (INT4 dropped on some) | ✅                        |
| **19. Numeric**                       | ✅                       | ✅                        | ✅                        |

### Per-architecture summary

- **Ampere (SM80)**: No TMA, no TMEM, no clusters/DSMEM, no warp specialization, no
  `wgmma`/`tcgen05`. The relevant load path is `cp.async`/`LDGSTS` into multi-stage SMEM
  pipelines; MMA is `mma.sync` (or legacy `wmma`) with register-resident accumulators.
  Sections 9, 11, 13, 15, and the warp-specialization parts of 7 are entirely N/A.
  Async barriers (split arrive/wait) exist but without transaction-byte tracking, so
  section 14's `arrive_expect_tx` does not apply. Max SMEM/CTA is 164 KB. No FP8/FP4.
- **Hopper (SM90)**: Adds TMA, thread-block clusters, DSMEM, cluster-scope and
  transaction-byte barriers, warp specialization (`setmaxnreg`), and `wgmma` (warpgroup
  MMA, operands from SMEM, accumulators in registers). FP8 via the Transformer Engine.
  No TMEM, no `.ws` mode, no `cta_group::2`, so all of section 11 and the TMEM-dependent
  rows of sections 8/12 are N/A. Max SMEM/CTA 228 KB.
- **Blackwell (SM100)**: Superset of Hopper. Adds TMEM (section 11 fully applies),
  `tcgen05.mma` with single-thread issue and TMEM-resident accumulators, `.ws` mode
  (A operand in TMEM), `cta_group::2` cooperative 2-SM MMA, enhanced TMA and clusters
  (up to 16 CTAs), and FP4/FP6/MX-scaled data types. Note that consumer Blackwell
  (SM120, e.g. RTX 5090) does **NOT** share this: no TMEM, no `tcgen05`, uses `mma.sync`
  like Ampere/Ada — treat SM120 closer to the Ampere column for sections 8, 11, 13.

> **Forward-compatibility note**: A kernel written for the Ampere column (cp.async +
> `mma.sync`, no clusters/TMA/TMEM) will run on Hopper and Blackwell but leaves their
> dedicated hardware idle. A kernel using Hopper-or-Blackwell-only features will **not**
> compile/run on Ampere. Candidates that toggle section 9/11/13 dimensions are therefore
> only valid within the architectures that support them.

---


## Dimension Interactions

These dimensions are NOT independent. Key interactions:

| Change                | Affects                                                               |
| --------------------- | --------------------------------------------------------------------- |
| Tiling (block\_m/n/k) | SMEM per stage, pipeline depth, tile count, TMEM cols, TMA box size   |
| Pipeline depth        | SMEM total, barrier count, warmup/drain overhead                      |
| Threads               | Warp count, registers per thread, epilogue parallelism                |
| K-unroll              | Register pressure, code size, barrier amortization, compiler behavior |
| Cluster               | Grid size, barrier scope, multicast options, launch API               |
| Occupancy             | SMEM budget per CTA, register budget, pipeline depth                  |
| MMA mode (.ws)        | TMEM layout (A in TMEM), SMEM budget (B only), pipeline balance       |
| Split-K               | Grid size, reduce kernel, workspace memory, synchronization           |
| Swap-AB               | ALL tiling, TMEM layout, pipeline balance, multicast roles            |
| Epilogue type         | SMEM budget (staging buffer), thread roles, overlap strategy          |
| Data type             | MMA instruction selection, TMEM layout, conversion cost               |

---


## How to Use This

This is a REFERENCE of what dimensions exist and how they interact.
Each candidate specifies only the dimensions that DIFFER from the current
baseline — everything else inherits. A candidate is a coherent joint
configuration, not an isolated change to one dimension.

Before proposing a candidate, check the
[Architecture Applicability Matrix](#architecture-applicability-matrix): a candidate that
sets a dimension marked ❌ for the target architecture is invalid (the feature does not
exist), and one that sets a ⚠️ dimension should specify the correct mechanism for that
architecture rather than assuming the Hopper/Blackwell default.
