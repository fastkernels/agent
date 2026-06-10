# GEMM Kernel Design Space — Extensions

Extends [`design-space.md`](./design-space.md) with three areas the base reference does
not cover: grouped/batched GEMM, scaling/quantization metadata flow, and fusion beyond
the epilogue. The same rules apply — each candidate is a JOINT configuration, and only
dimensions that DIFFER from the baseline are specified.

> **Architecture scope**: Block-scaled narrow types (MX, NVFP4) and their scale-factor
> machinery, plus single-instruction grouped GEMM via `tcgen05`, are **Blackwell SM100
> only**. Hopper (SM90) does grouped GEMM via pointer-array `wgmma` and supports FP8
> per-tensor/per-row scaling but not hardware block scaling. Ampere (SM80) has neither
> FP8 nor block scaling. Per-dimension applicability is called out inline and summarized
> in the matrix at the end.

---


## 20. Grouped & Batched GEMM

A "grouped" GEMM runs many independent GEMMs of *different* shapes in one launch (e.g.
MoE expert FFNs); a "batched" GEMM runs many of the *same* shape. Both differ from a
plain GEMM in how problems are enumerated, scheduled, and addressed.

- **batch_mode**: single, strided-batched (uniform shape, fixed stride), grouped (per-group shapes), ragged (variable M per group)
- **num_groups**: number of independent problems in the launch
- **group_shape_source**: compile-time constant, host array, device array (problem sizes read on device)
- **problem_shape_layout**: how per-group [M, N, K] are stored and fetched (host-passed, `grid_constant` array, global pointer array)
- **operand_addressing**: strided (base + i·stride) vs pointer-array (per-group A/B/C/D pointers)
- **pointer_array_storage**: `grid_constant`, global memory, SMEM-cached
- **group_to_tile_mapping**: how the global tile index decomposes into (group, tile-within-group); prefix-sum over per-group tile counts vs linear scan
- **group_scheduling**: static block→group assignment vs dynamic (atomic counter / CLC work-stealing across all groups' tiles)
- **load_balancing**: even tiles-per-CTA vs stream-K across the *flattened* tile space (critical when group shapes are skewed, e.g. uneven MoE token routing)
- **ragged_m_handling**: how variable token counts per expert are tiled (per-group M from device, padded-to-tile vs masked partial tiles)
- **empty_group_handling**: groups with zero rows (skip via scheduler, or launch-time prune)
- **tma_descriptor_per_group**: one TMA descriptor per group (rebuilt/swapped per group) vs descriptor recomputed on device per tile
- **batch_dim_in_tma**: encode batch as an extra TMA descriptor dimension (strided-batched) vs separate descriptors (grouped)
- **epilogue_per_group**: per-group output pointers, scales, and fused-op parameters (bias/activation differ per expert)
- **k_major_vs_group_major_loop**: iteration order when combining split-K with grouping

**Interactions**: grouped GEMM interacts strongly with the base **grid_strategy** (almost
always persistent + dynamic assignment), **tile_assignment** (CLC work-stealing is the
natural fit), and **split_k** (stream-K must flatten across groups). On Hopper, grouped
GEMM uses pointer-array `wgmma`; on Blackwell, `tcgen05` grouped GEMM with TMA is
supported for non-block-scaled types, and block-scaled grouped GEMM adds per-group
scale-factor tensors (see §21).

**Decode/thin-M note**: MoE decode is the common ragged case — many groups, each with
M = a few tokens. Here the per-group tiles are tiny and scheduler overhead dominates;
prefer a flattened stream-K / persistent scheduler and consider a GEMV-class mainloop
per group rather than full MMA tiles.


## 21. Scaling & Quantization Metadata

Covers how quantization scales (and zero-points) are represented, stored, loaded, and
applied. This is mostly orthogonal to the math tiling but adds its own load path,
SMEM/TMEM residency, and pipeline coupling.

### 21a. Quantization scheme

- **quant_granularity**: per-tensor, per-row (per-token / per-output-channel), per-column, per-block (group of K elements), per-2D-block
- **block_scale_size**: K-elements per scale (Blackwell: 16 for NVFP4 / `scale_vec::4X`, 32 for MX formats / `scale_vec::2X`)
- **scale_dtype**: FP32, BF16, FP8 (E4M3/E5M2), UE8M0 (the MX shared-exponent scale type)
- **format**: per-tensor FP8, MXFP8, MXFP6, MXFP4, NVFP4, INT8 (+zero-point), INT4 (+zero-point)
- **symmetric_vs_asymmetric**: symmetric (scale only) vs asymmetric (scale + zero-point)
- **zero_point_dtype**: where asymmetric (INT8/INT4 weight-only)
- **scale_static_vs_dynamic**: weights pre-quantized offline (static) vs activations quantized at runtime (dynamic, needs an amax pass)

### 21b. Scale-factor data flow

- **sf_layout_a / sf_layout_b**: physical layout of A/B scale tensors (the interleaved per-128 / per-32 atom layouts CUTLASS uses on SM100)
- **sf_load_path**: TMA (`tcgen05.cp` SMEM→TMEM) vs cp.async vs plain load
- **sf_residency**: where scales live during the K-loop (registers, SMEM, TMEM). On SM100 block-scaled MMA, SFA/SFB must be **replicated across all 32 TMEM lane partitions**.
- **sf_smem_staging**: whether scale tiles are double/circular-buffered (note: on SM100 the `tcgen05.cp`→`tcgen05.mma` implicit pipeline often makes SF buffering unnecessary)
- **sf_tile_to_math_tile_mapping**: how SF blocks index onto the math tile (e.g. a 64-wide N-tile sharing a per-128 SFB block needs `sfb_idx = n_tile >> 1` with an offset)
- **sf_pipeline_coupling**: SF load tied to operand load stage vs independent stage
- **amax_strategy**: for dynamic activation quant — fused amax in a prior epilogue, separate reduction kernel, or running amax in the producer

### 21c. Dequant / requant placement

- **dequant_point**: hardware block-scale inside MMA (SM100 `*.block_scale`), dequant-on-load (convert in producer before SMEM), dequant-in-epilogue
- **weight_only_dequant**: INT4/INT8 weights dequantized to FP16/BF16 on the load side while activations stay high-precision (common for decode)
- **output_requant**: epilogue produces a quantized output + generated scale (e.g. NVFP4-out GEMM with scale-factor generation)
- **scale_generation**: when output is quantized, how the output scale is computed (per-block amax in epilogue, fixed scale)

**Interactions**: `format` and `block_scale_size` select the **mma_instruction** kind
(`mxf8f6f4`, `mxf4`, `nvf4mxf4`) from base §8; scale residency competes with accumulator
columns for the **TMEM** budget (§11) — pushing two N-tiles into one CTA can overflow
`tcgen05.alloc`; `weight_only_dequant` shifts cost onto the **load path** (§9) and
register/SMEM budget rather than the tensor cores.


## 22. Fusion Beyond the Epilogue

Base §12 (`epilogue_fusion`) covers post-MMA ops on C. This section covers fusion on the
**input/load side**, **across the K-loop**, and **across multiple GEMMs**.

### 22a. Producer-side (pre-MMA) fusion

- **fused_dequant_load**: convert/dequantize operands during the load→SMEM step (INT4→BF16, FP8→BF16) instead of a separate kernel
- **fused_transpose_load**: transpose an operand in the load path (TMA layout / SMEM swizzle) to avoid a standalone transpose kernel
- **fused_layout_convert**: on-the-fly layout change (e.g. interleave, pack) during load
- **fused_norm_input**: fold a normalization (RMSNorm/LayerNorm) scale into the activation load when the upstream op is a norm
- **fused_bias_pre**: add bias to an operand before MMA (rare; usually epilogue)
- **fused_activation_quant**: quantize activations in the producer as they stream in (pairs with dynamic `amax_strategy` in §21)

### 22b. In-loop / cross-K fusion

- **fused_elementwise_b**: apply a per-element op to B tiles inside the K-loop (e.g. gating mask)
- **fused_partial_reduction**: combine split-K partials with an op other than plain add

### 22c. Cross-GEMM / multi-op fusion

- **gemm_chain_fusion**: keep an intermediate in SMEM/TMEM/registers between back-to-back GEMMs to skip a global round-trip (back-to-back GEMM, à la fused attention QK·V)
- **fused_gate_up**: MoE/MLP fusing gate and up projections (one GEMM with concatenated N, split in epilogue) — pairs with grouped GEMM §20
- **fused_silu_mul**: SwiGLU — fuse the SiLU(gate)·up elementwise product into the epilogue of the gate/up GEMM before the down projection
- **fused_residual_add**: add the residual stream in the epilogue of the final projection
- **producer_consumer_gemm**: overlap the epilogue of GEMM N with the mainloop of GEMM N+1 when chained
- **fusion_boundary**: what stays on-chip vs spills to global between fused stages (the key budget decision — SMEM/TMEM capacity caps chain depth)

**Interactions**: cross-GEMM fusion competes for **SMEM/TMEM** (§10/§11) against pipeline
depth and accumulator buffering — deeper fusion means shallower pipelines. `fused_gate_up`
and `fused_silu_mul` are the standard MoE/MLP fusions and only make sense layered on top
of grouped GEMM (§20) and the quant flow (§21). Producer-side dequant (§22a) is the main
lever for **decode** weight-only-quantized GEMV, where the kernel is memory-bound on
weights and the dequant is nearly free under the load.

---


## Architecture Applicability — Extension Sections

Legend: ✅ supported · ⚠️ different mechanism / partial · ❌ N/A (no hardware support)

| Dimension                                | Ampere (SM80)              | Hopper (SM90)              | Blackwell (SM100)          |
| ---------------------------------------- | -------------------------- | -------------------------- | -------------------------- |
| **20. Grouped / batched GEMM**           | ✅ (pointer-array, mma.sync)| ✅ (pointer-array wgmma)   | ✅ (tcgen05 + TMA)         |
| stream-K across groups                   | ⚠️ (manual)                | ✅                         | ✅ (CLC work-stealing)     |
| TMA descriptor per group                 | ❌ (no TMA)                | ✅                         | ✅                         |
| block-scaled grouped GEMM                | ❌                         | ❌                         | ✅                         |
| **21. Scaling / quantization**           |                            |                            |                            |
| per-tensor / per-row FP8 scaling         | ❌ (no FP8)                | ✅ (SW, Transformer Engine)| ✅                         |
| hardware block scaling (MX / NVFP4)      | ❌                         | ❌                         | ✅ (`*.block_scale` MMA)   |
| scale residency in TMEM                  | ❌ (no TMEM)               | ❌ (no TMEM)               | ✅ (replicated 32 lanes)   |
| INT8/INT4 weight-only dequant            | ✅ (SW, in load/epilogue)  | ✅ (SW)                    | ✅ (SW or HW narrow types) |
| UE8M0 / MX shared-exponent scale type    | ❌                         | ❌                         | ✅                         |
| **22. Fusion beyond epilogue**           |                            |                            |                            |
| producer-side dequant / convert          | ✅ (in cp.async path)      | ✅ (in TMA/producer)       | ✅                         |
| cross-GEMM on-chip via SMEM              | ✅                         | ✅                         | ✅                         |
| cross-GEMM on-chip via TMEM              | ❌                         | ❌                         | ✅                         |
| fused gate/up, SwiGLU, residual          | ✅                         | ✅                         | ✅                         |
| producer_consumer_gemm (warp-spec)       | ❌ (no warp-spec HW)       | ✅                         | ✅                         |

### Notes

- **Ampere**: Grouped/batched GEMM and INT8/INT4 weight-only quant work via software
  (pointer arrays, `mma.sync`, dequant folded into the cp.async load or the epilogue).
  No FP8, no hardware block scaling, no TMEM, so all of §21's block-scale rows and the
  TMEM-resident fusion paths are N/A. Producer-side dequant happens in the cp.async path.
- **Hopper**: Adds FP8 with software per-tensor/per-row scaling via the Transformer
  Engine, pointer-array `wgmma` grouped GEMM, and TMA-based producer fusion. Still no
  hardware block scaling and no TMEM — MX/NVFP4 block-scaled MMA and TMEM-resident scale
  factors are N/A.
- **Blackwell SM100**: Full block-scaled narrow-precision path: `tcgen05.mma` with
  `mxf8f6f4` / `mxf4` / `nvf4mxf4` kinds, scale factors copied SMEM→TMEM via `tcgen05.cp`
  and replicated across 32 TMEM lane partitions, single-instruction grouped GEMM with
  TMA, and CLC stream-K across groups. NVFP4 uses a 16-element K block, MX formats use 32.
- **Consumer Blackwell SM120 (RTX 5090)**: supports block-scaled NVFP4/MX GEMM and
  grouped GEMM at the CUTLASS-example level, but with **no TMEM and `mma.sync`-style**
  execution — treat scale residency and accumulator placement like the Ampere/Hopper
  columns (registers/SMEM), not the SM100 TMEM column.

---


## How to Use This

These sections layer onto the base 19. A grouped, block-scaled, fusion-heavy MoE kernel
is the cross product: §20 (ragged grouped + stream-K) × §21 (NVFP4 block scaling, scale
in TMEM) × §22 (fused gate/up + SwiGLU) on top of the base tiling/pipeline/MMA choices.
As before, a candidate that sets a dimension marked ❌ for the target architecture is
invalid, and a ⚠️ dimension must name the correct mechanism for that architecture.
