# FA4 — distillation reference

All paths are **relative to the repo root** `vllm_repo/vllm-flash-attention/`.

## ⚠️ Read this first: FA4 has no `.cu` file

FA4 is **not** hand-written CUDA. It is written in **CuteDSL (NVIDIA CUTLASS Python DSL)** and **JIT-compiled to PTX/CUBIN at runtime**. There is nothing to "extract" into a `.cu` — distilling FA4 into a dependency-free single CUDA file is a **ground-up reimplementation** using these Python kernels only as the algorithm reference. It also depends on a live `nvidia-cutlass-dsl` toolchain. Treat this as the hardest of the three and likely a separate project.

## Target

| | |
|---|---|
| Build target | **none** — pure Python package `flash_attn/cute/` (installed as `flash-attn-4`) |
| Arch | Ampere(SM80) / Hopper(SM90) / Blackwell(SM100) / consumer Blackwell(SM120), selected at runtime |
| Use in vLLM | Default dense backend is FA4 only as fallback (Hopper edge cases, Blackwell SM100 when FlashInfer absent) |
| Entry exposed | `_flash_attn_fwd` (imported by the vLLM wrapper's fa_version==4 branch) |
| Scope | Forward only for this purpose (backward exists but vLLM is inference-only) |

vLLM Python wrapper: `vllm_repo/vllm/vllm/vllm_flash_attn/flash_attn_interface.py` → `flash_attn_varlen_func` (fa_version==4 branch, line 367) does `from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd` and calls it. (`vllm/vllm_flash_attn/cute` is a symlink/copy of `flash_attn/cute`.)

---

## Functions to port — call tree (file:line)

### 1. Entry point + arch dispatch — `flash_attn/cute/interface.py`
- **`_flash_attn_fwd` — :295**  ← the function vLLM calls; builds config, dispatches by SM
- arch dispatch inside it instantiates:
  - `FlashAttentionForwardSm80(...)` — :818  (Ampere)
  - `FlashAttentionForwardSm90(...)` — :837  (Hopper)
  - `FlashAttentionForwardSm100(...)` — :936  (Blackwell SM100)
  - `FlashAttentionForwardSm120(...)` — :968  (consumer Blackwell SM120)
- `_flash_attn_fwd_combine` — :2393  (split-KV combine driver)
- `FwdConfig` — :111  (tile/config dataclass)
- *(public `flash_attn_func` :2205 / `flash_attn_varlen_func` :2255 / autograd `FlashAttnFunc` :1986 — wrappers around `_flash_attn_fwd`; vLLM bypasses them)*

### 2. Per-arch forward kernels (the CuteDSL kernel bodies = the algorithm to reimplement)
- `flash_attn/cute/flash_fwd.py` — `class FlashAttentionForwardBase` :39 (`__call__` :323) · `class FlashAttentionForwardSm80` :618 (`__call__` :668)  ← **Ampere**
- `flash_attn/cute/flash_fwd_sm90.py` — `class FlashAttentionForwardSm90` :51 (`__call__` :167)  ← **Hopper** (TMA + WGMMA + warp-spec)
- `flash_attn/cute/flash_fwd_sm100.py` — `class FlashAttentionForwardSm100` :113 (`__call__` :370)  ← **Blackwell SM100** (UMMA/2CTA, SplitKV, paged KV, persistent)
- `flash_attn/cute/flash_fwd_sm120.py` — `class FlashAttentionForwardSm120(FlashAttentionForwardSm80)` :14  ← **consumer Blackwell**
- `flash_attn/cute/flash_fwd_combine.py` — `class FlashAttentionForwardCombine` :21 (`__call__` :200)  ← merges SplitKV partials
- *(optional, MLA only) `flash_attn/cute/flash_fwd_mla_sm100.py` — `FlashAttentionMLAForwardSm100`; `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` — `BlackwellFusedMultiHeadAttentionForward` (hdim-256 2CTA)*

---

## Full list of files used by FA4 (`flash_attn/cute/`, 48 files)

### A. Forward kernels (the math to reimplement per arch)
```
flash_attn/cute/interface.py                      # entry + dispatch (_flash_attn_fwd)
flash_attn/cute/flash_fwd.py                       # Base + Sm80 (Ampere) forward
flash_attn/cute/flash_fwd_sm90.py                  # Hopper forward
flash_attn/cute/flash_fwd_sm100.py                 # Blackwell SM100 forward
flash_attn/cute/flash_fwd_sm120.py                 # consumer Blackwell forward
flash_attn/cute/flash_fwd_combine.py               # SplitKV combine
```

### B. Shared kernel components (definitions used by the forward path)
```
flash_attn/cute/softmax.py            # online softmax + score_mod
flash_attn/cute/mask.py               # causal / local / block-sparse / mask_mod
flash_attn/cute/block_info.py         # tile dims, n/m block ranges
flash_attn/cute/seqlen_info.py        # varlen seqlen/offset tracking
flash_attn/cute/pipeline.py           # pipelined load state
flash_attn/cute/tile_scheduler.py     # tile scheduling (single/varlen/persistent)
flash_attn/cute/copy_utils.py         # type-converting copies, TMA copy atoms
flash_attn/cute/named_barrier.py      # warp-sync named barriers
flash_attn/cute/barrier.py            # barrier primitives
flash_attn/cute/pack_gqa.py           # GQA head packing
flash_attn/cute/paged_kv.py           # paged KV manager (TMA)
flash_attn/cute/fast_math.py          # exp2 coeffs, softcap score_mod
flash_attn/cute/utils.py              # hashes, warp reductions, predicates
```

### C. Arch-specific GEMM/MMA helpers
```
flash_attn/cute/ampere_helpers.py     # SM80 mma
flash_attn/cute/hopper_helpers.py     # SM90 warp-group GEMM, smem layouts, fence/commit/wait
flash_attn/cute/blackwell_helpers.py  # SM100 UMMA GEMM, 2CTA, PTX paths
flash_attn/cute/mma_sm100_desc.py     # SM100 MMA descriptor enums
```

### D. JIT / compile / cache infrastructure (CuteDSL plumbing — likely dropped in a static rewrite)
```
flash_attn/cute/cache_utils.py        # JIT compile cache
flash_attn/cute/cute_dsl_utils.py     # patched cute.compile (+ optional SASS dump)
flash_attn/cute/cute_dsl_ptxas.py     # custom ptxas hook
flash_attn/cute/fa_logging.py
flash_attn/cute/__init__.py
```

### E. Block-sparse (optional; used only for block_sparse_tensors / score_mod)
```
flash_attn/cute/block_sparse_utils.py
flash_attn/cute/block_sparsity.py
flash_attn/cute/compute_block_sparsity.py
flash_attn/cute/topk_gather_kv.py
```

### F. MLA + hdim-256 2CTA forward (optional; MLA path, not non-MLA dense)
```
flash_attn/cute/flash_fwd_mla_sm100.py
flash_attn/cute/sm100_hd256_2cta_fmha_forward.py
```

### G. Backward (skip — inference doesn't need these)
```
flash_attn/cute/flash_bwd.py
flash_attn/cute/flash_bwd_preprocess.py
flash_attn/cute/flash_bwd_postprocess.py
flash_attn/cute/flash_bwd_sm90.py
flash_attn/cute/flash_bwd_sm100.py
flash_attn/cute/flash_bwd_sm120.py
flash_attn/cute/sm100_hd256_2cta_fmha_backward.py
flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py
flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py
```

### H. Test / benchmark / search (skip)
```
flash_attn/cute/testing.py
flash_attn/cute/bench_utils.py
flash_attn/cute/benchmark.py
flash_attn/cute/benchmark_flash_attention_fp8.py
flash_attn/cute/sm90_config_search.py
```

### I. External dependencies (the boundary)
`cutlass`, `cutlass.cute` (the CuteDSL itself — this *is* the framework FA4 is written in), `cuda.bindings.driver`, `torch`. A no-CUTLASS rewrite cannot reuse any of this; the Python kernels serve only as the reference algorithm.

---

## Feature set `_flash_attn_fwd` supports (kwargs the vLLM wrapper passes)
q/k/v, `cu_seqlens_q/k`, `seqused_k`, `max_seqlen_q/k`, `page_table` (paged KV), `softmax_scale`, `causal`, `softcap`, `window_size_left/right` (sliding window), `num_splits` (SplitKV), `return_lse`, `out`, `learnable_sink` (= `s_aux`, attention sinks). Layout `(batch, seqlen, nheads, headdim)`, last dim contiguous & 16-byte aligned. **No** ALiBi.
