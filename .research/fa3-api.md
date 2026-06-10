# FA3 Interface Contract (vLLM `_vllm_fa3_C`) — Distillation for Standalone .cu Agent

Goal recap: produce ONE self-contained `.cu` that registers `torch.ops._vllm_fa3_C.fwd` (and
`get_scheduler_metadata`) with byte-for-greedy-token-identical behavior to the prebuilt baseline,
matching/exceeding latency. `torch` allowed only in the host `mha_fwd` fn + `TORCH_LIBRARY` registration.
No CUTLASS/CuTe/cuBLAS/etc. in `__global__` code.

Source of truth: `reference/libraries/vllm-flash-attention/hopper/` and installed wrapper at
`/home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/vllm/vllm_flash_attn/flash_attn_interface.py`.

---

## (a) EXACT torch op schema strings (VERBATIM, confirmed against the live `.so`)

These were dumped from the installed `_vllm_fa3_C` via `torch.ops._vllm_fa3_C.fwd.default._schema`.
Our `.cu` MUST register **byte-identical** schemas (same names/order/types/optionality) or vLLM's
positional call breaks.

### `fwd`  (returns `Tensor[]`, a 4-element list: `[out, softmax_lse, out_accum, softmax_lse_accum]`)

Live normalized schema:
```
_vllm_fa3_C::fwd(Tensor($0! -> ) q, Tensor k, Tensor v, Tensor? k_new, Tensor? v_new,
  Tensor? q_v, Tensor($1! -> )? out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k,
  Tensor? cu_seqlens_k_new, Tensor? seqused_q, Tensor? seqused_k, int? max_seqlen_q,
  int? max_seqlen_k, Tensor? page_table, Tensor? kv_batch_idx, Tensor? leftpad_k,
  Tensor? rotary_cos, Tensor? rotary_sin, Tensor? seqlens_rotary, Tensor? q_descale,
  Tensor? k_descale, Tensor? v_descale, float softmax_scale, bool is_causal,
  int window_size_left, int window_size_right, float softcap, bool is_rotary_interleaved,
  Tensor? scheduler_metadata, int num_splits, bool? pack_gqa, int sm_margin, Tensor? s_aux,
  int cp_world_size, int cp_rank, Tensor? cp_tot_seqused_k) -> Tensor[]
```

The DEF string as written in `hopper/flash_api_torch_lib.cpp` (line 93-129) — note `Tensor!  q`
(mutable/aliased), `Tensor!? out` (mutable optional). Reproduce EXACTLY:
```
fwd(Tensor!  q, Tensor   k, Tensor   v, Tensor?  k_new, Tensor?  v_new, Tensor?  q_v,
    Tensor!? out, Tensor?  cu_seqlens_q, Tensor?  cu_seqlens_k, Tensor?  cu_seqlens_k_new,
    Tensor?  seqused_q, Tensor?  seqused_k, int?     max_seqlen_q, int?     max_seqlen_k,
    Tensor?  page_table, Tensor?  kv_batch_idx, Tensor?  leftpad_k, Tensor?  rotary_cos,
    Tensor?  rotary_sin, Tensor?  seqlens_rotary, Tensor?  q_descale, Tensor?  k_descale,
    Tensor?  v_descale, float    softmax_scale, bool     is_causal, int      window_size_left,
    int      window_size_right, float    softcap, bool     is_rotary_interleaved,
    Tensor?  scheduler_metadata, int      num_splits, bool?    pack_gqa, int      sm_margin,
    Tensor?  s_aux, int      cp_world_size, int      cp_rank, Tensor?  cp_tot_seqused_k) -> Tensor[]
```
(38 args total: q,k,v + 3 optional KV + out + ... ; positions matter — see wrapper dispatch below.)

### `get_scheduler_metadata` (returns single `Tensor`)
```
_vllm_fa3_C::get_scheduler_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k,
  int num_heads, int num_heads_k, int headdim, int headdim_v, ScalarType qkv_dtype,
  Tensor seqused_k, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new,
  Tensor? seqused_q, Tensor? leftpad_k, int? page_size, int max_seqlen_k_new, bool is_causal,
  int window_size_left, int window_size_right, bool has_softcap, int num_splits,
  bool? pack_gqa, int sm_margin) -> Tensor
```
(23 args. `ScalarType` is a first-class torch schema type — pass `at::ScalarType`.)

C++ signatures (`flash_api_torch_lib.cpp` lines 20-87): `std::vector<at::Tensor> mha_fwd(...)`,
`at::Tensor mha_fwd_get_scheduler_metadata(...)`.

---

## (b) `Flash_fwd_params` — full field list (`hopper/flash.h`, struct inherits `Qkv_params`)

`using index_t = int64_t`. We need a byte-compatible struct ONLY IF we reuse upstream kernel objects;
since we write our own kernel we can define our OWN params struct. But replicating field semantics is
the safest path. Fields:

From `Qkv_params` (base):
- `void* q_ptr, k_ptr, v_ptr`
- strides (index_t = int64_t, IN ELEMENTS not bytes): `q_batch_stride,k_batch_stride,v_batch_stride,
  q_row_stride,k_row_stride,v_row_stride, q_head_stride,k_head_stride,v_head_stride, v_dim_stride`
- `int h, h_k`  (num query heads, num kv heads)

`Flash_fwd_params` adds:
- Output: `void* o_ptr, oaccum_ptr`; `index_t o_batch_stride,o_row_stride,o_head_stride`
- LSE: `void* softmax_lse_ptr, softmax_lseaccum_ptr`
- FP8 descale: `float* q_descale_ptr,k_descale_ptr,v_descale_ptr`; `index_t q_descale_batch_stride,
  q_descale_head_stride,k_descale_batch_stride,k_descale_head_stride,v_descale_batch_stride,v_descale_head_stride`
- Dims (int): `b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded,
  rotary_dim; total_q, total_k, total_knew; b_k; dv, dv_rounded`
- Scales: `float scale_softmax; float softcap`
- Varlen: `int* cu_seqlens_q, cu_seqlens_k, cu_seqlens_knew, leftpad_k; int* seqused_q, seqused_k`
- Oaccum strides: `index_t oaccum_split_stride,oaccum_batch_stride,oaccum_row_stride,oaccum_head_stride`
- LSEaccum strides: `index_t lseaccum_split_stride,lseaccum_batch_stride,lseaccum_head_stride`
- New KV (appendKV, DISABLED in build): `void* knew_ptr,vnew_ptr`; strides
  `knew_batch_stride,vnew_batch_stride,knew_row_stride,vnew_row_stride,knew_head_stride,vnew_head_stride`
- Q-as-V: `void* qv_ptr; index_t qv_batch_stride,qv_row_stride,qv_head_stride`
- Rotary (DISABLED via appendKV): `void* rotary_cos_ptr,rotary_sin_ptr; int* seqlens_rotary`
- `int* kv_batch_idx`
- Paged KV: `int* page_table; index_t page_table_batch_stride; int page_size, num_pages; bool pagedkv_tma`
- Dropout (DISABLED): `float p_dropout; uint8_t p_dropout_in_uint8_t; float rp_dropout`
- Local window: `int window_size_left, window_size_right`
- `uint64_t* rng_state`
- dtype flags: `bool is_bf16, is_fp32, is_e4m3, is_causal, is_local, is_rotary_interleaved`
- Split-KV: `int num_splits; bool pack_gqa`
- Scheduler: `int* tile_count_semaphore, prepare_seqlen_q_ptr, num_splits_dynamic_ptr,
  varlen_batch_idx_ptr, num_nheads_in_l2_ptr; bool skip_scheduler_metadata_computation,
  varlen_sort_batches; int tile_count_semaphore_offset; bool head_swizzle, prepare_varlen_pdl`
- `int arch, num_sm`
- `void* s_aux_ptr` (attention sink, shape (num_heads), bf16)
- CP: `int cp_world_size, cp_rank; int* cp_tot_seqused_k`

`Flash_bwd_params` exists but BACKWARD IS DISABLED in the vLLM build
(`FLASHATTENTION_DISABLE_BACKWARD`) — we do NOT need it.

---

## (c) Standalone `.cu` registration: what shim/registration give us, and avoiding them

`csrc/common/registration.h`:
- `TORCH_LIBRARY_EXPAND(NAME, MODULE)` == just `TORCH_LIBRARY(NAME, MODULE)` with macro-expanded name.
  `TORCH_EXTENSION_NAME` is defined by the torch build system to the module name (`_vllm_fa3_C`).
- `REGISTER_EXTENSION(NAME)` defines `PyInit_<NAME>()` returning an empty PyModule — needed so Python
  `import _vllm_fa3_C` succeeds and triggers static `TORCH_LIBRARY` registration.

`csrc/common/pytorch_shim.h`:
- `make_pytorch_shim(&fn)` wraps a fn so torch-incompatible C++ types are bridged to torch schema types:
  - `int` -> `int64_t`, `float` -> `double`
  - `std::optional<T>&` -> `const std::optional<T>&`
  - `std::optional<const at::Tensor>&` -> `const std::optional<at::Tensor>&`
  - `std::optional<int>` -> `std::optional<int64_t>`
  It returns a lambda taking the torch-compatible types and converting back. Pure compile-time glue,
  no runtime cost, no external deps (only `<torch/library.h>`).

CAN WE AVOID THEM? YES — both are tiny, header-only, dependency-free (registration.h only needs
`<Python.h>`; pytorch_shim.h only `<torch/library.h>`). Two options for the single `.cu`:
1. **Write `mha_fwd` directly with torch-schema-native types** so no shim is needed: declare host fn as
   `std::vector<at::Tensor> mha_fwd(at::Tensor q, at::Tensor k, ..., int64_t window_size_left,
   double softmax_scale, ...)` and register with `ops.impl("fwd", torch::kCUDA, &mha_fwd);` directly.
   This is the CLEANEST for a single file: avoids both headers entirely.
2. OR inline the ~30 lines of `make_pytorch_shim` + `TORCH_LIBRARY`/`REGISTER_EXTENSION` macros into
   the `.cu` and keep upstream-style `int`/`float`/`std::optional<const at::Tensor>&` signatures.

Registration body (inline, no external headers):
```cpp
#include <torch/library.h>
#include <torch/extension.h>   // pulls PyInit machinery; or include <Python.h> + PyModule
TORCH_LIBRARY(_vllm_fa3_C, ops) {
    ops.def("fwd(...verbatim schema...) -> Tensor[]");
    ops.impl("fwd", torch::kCUDA, &mha_fwd);
    ops.def("get_scheduler_metadata(...) -> Tensor");
    ops.impl("get_scheduler_metadata", torch::kCUDA, &mha_fwd_get_scheduler_metadata);
}
```
IMPORTANT: the library namespace token must be `_vllm_fa3_C` so the wrapper's
`torch.ops._vllm_fa3_C.fwd` resolves. If we build as a `.so` named `_vllm_fa3_C` and monkeypatch the
package, `TORCH_EXTENSION_NAME` will already be `_vllm_fa3_C`; if we build a standalone test module we
hardcode the literal `_vllm_fa3_C`. Need `PyInit__vllm_fa3_C` (via REGISTER_EXTENSION or
`torch.utils.cpp_extension` which adds it) if loaded via `import`; if loaded via
`torch.ops.load_library(path)` the PyInit is not strictly required (torch dlopens and runs static
ctors). For swapping into vLLM, the simplest is `torch.ops.load_library("our.so")` which re-registers
the same op namespace (must NOT also import the original — name clash). See (e).

---

## (d) Building the `.cu` against torch + CUDA only (no CUTLASS)

vLLM's actual FA3 build (`CMakeLists.txt` ~180-308):
- Target `_vllm_fa3_C`, sources: `flash_api.cpp`, `flash_api_torch_lib.cpp`, `flash_fwd_combine.cu`,
  `flash_prepare_scheduler.cu`, + generated instantiations. Includes: `hopper`, `csrc/common`,
  `csrc/cutlass/include` (WE DROP cutlass include).
- GPU flags (`VLLM_FA_GPU_FLAGS`): `--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`
  `-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1` `--threads=N`; arch is per-file `9.0a` via
  `cuda_archs_loose_intersection(FA3_ARCHS "9.0a;" ...)` => `-gencode arch=compute_90a,code=sm_90a`.
- compile defs used: `FLASHATTENTION_DISABLE_BACKWARD, _DROPOUT, _UNEVEN_K, _PYBIND, _CLUSTER,
  _SM8x, _APPENDKV`, plus `FLASHATTENTION_VARLEN_ONLY, FLASHATTENTION_PACKGQA_ONLY`,
  `CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED, CUTLASS_ENABLE_GDC_FOR_SM90`. (These are cutlass/feature
  toggles; for OUR kernel only the semantic ones matter: varlen-only, packgqa-only, no
  appendkv/rotary/dropout/backward.)

For our single `.cu` (torch+CUDA only) the minimal nvcc invocation:
```
nvcc -O3 -std=c++17 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda \
  -gencode arch=compute_90a,code=sm_90a \
  $(python -c "import torch.utils.cpp_extension as c; print(' '.join('-I'+i for i in c.include_paths()))") \
  $(python -c "import torch,os;print('-I'+os.path.join(os.path.dirname(torch.__file__),'include'))") \
  -I$CONDA_PREFIX/include/python3.12 \
  --shared -Xcompiler -fPIC \
  -L$(python -c "import torch.utils.cpp_extension as c;print(c.library_paths()[0])") \
  -ltorch -ltorch_cuda -lc10 -lc10_cuda -ltorch_python \
  -D_GLIBCXX_USE_CXX11_ABI=$(python -c "import torch;print(int(torch.compiled_with_cxx11_abi()))") \
  our_fa3.cu -o _vllm_fa3_C.so
```
EASIER: use `torch.utils.cpp_extension.load(name="_vllm_fa3_C", sources=["our_fa3.cu"],
extra_cuda_cflags=[...], extra_cflags=["-DPy_LIMITED_API"...])` which handles ABI flags, include/lib
paths, and emits `PyInit__vllm_fa3_C`. Match torch's CXX11 ABI (torch 2.11 cu130). Watch the `USE_SABI 3`
(`Py_LIMITED_API`) in vLLM — not required for our test module.

---

## (e) Wrapper dispatch / the MONKEYPATCH swap point

Installed wrapper: `.../vllm/vllm_flash_attn/flash_attn_interface.py`.
- Module import (lines 22-29): `from . import _vllm_fa3_C` sets `FA3_AVAILABLE`. The compiled `.so`'s
  static `TORCH_LIBRARY` ctor registers `torch.ops._vllm_fa3_C.{fwd,get_scheduler_metadata}`.
- `flash_attn_varlen_func(...)` is the ONLY fwd entry vLLM uses (plus `get_scheduler_metadata`).
  For `fa_version == 3` (Hopper auto-selects 3) it calls (lines 328-366), POSITIONALLY:
  ```
  out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd(
      q, k, v, None, None,          # k_new, v_new
      q_v, out, cu_seqlens_q, cu_seqlens_k, None,  # cu_seqlens_k_new
      None, seqused_k,              # seqused_q, seqused_k
      max_seqlen_q, max_seqlen_k, block_table,
      None, None,                   # kv_batch_idx, leftpad_k
      None, None, None,             # rotary_cos, rotary_sin, seqlens_rotary
      q_descale, k_descale, v_descale,
      softmax_scale, causal,
      real_window_size[0], real_window_size[1],
      softcap, True,                # is_rotary_interleaved (hardcoded True!)
      scheduler_metadata, num_splits, None,  # pack_gqa=None
      0,                            # sm_margin
      s_aux, cp_world_size, cp_rank, cp_tot_seqused_k)
  ```
  So OUR kernel only ever sees: k_new/v_new/q_v/kv_batch_idx/leftpad_k/rotary*=None,
  pack_gqa=None (auto), sm_margin=0, dropout absent. The 4-tuple return is unpacked; we MUST return a
  4-element list (`[out, softmax_lse, out_accum_or_empty, lse_accum_or_empty]`). vLLM discards
  elements 2,3. Wrapper returns `out` (or `(out, softmax_lse)` if `return_softmax_lse`).
- `q,k,v` are made contiguous-last-dim via `maybe_contiguous` before the call.
- `get_scheduler_metadata` wrapper (lines 122-173) passes `cu_seqlens_k=None, seqused_q=None`; called
  from vLLM `flash_attn.py` `schedule()` (line ~454) only when AOT scheduling; returns an int32 tensor
  passed back as `scheduler_metadata`. We can return a correctly-sized zero int32 tensor (or implement
  the prepare-scheduler logic) — but if we set `num_splits=1` path and don't need persistent-scheduler
  semaphore, behavior must still match. SAFEST: implement get_scheduler_metadata to return the SAME
  shape/contents as baseline, OR ensure our fwd ignores scheduler_metadata and recomputes internally
  (then any tensor of right shape is fine since `skip_scheduler_metadata_computation` only matters to
  upstream kernel). For correctness-first, simplest is to recompute everything inside fwd and treat the
  passed scheduler_metadata as advisory.

### Swap points (how to put OUR .cu in):
1. **torch.ops re-registration**: build our `.so` registering namespace `_vllm_fa3_C`. Either (a) replace
   the file `site-packages/vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so`, or (b) at runtime
   `torch.ops.load_library("our.so")` BEFORE vLLM imports the original (avoid double registration —
   torch raises on duplicate op def). Cleanest test harness: monkeypatch by importing our `.so` and
   shadowing `vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_func` to call our op, OR
   directly replace the `.so` so `from . import _vllm_fa3_C` loads ours.
2. The wrapper `flash_attn_varlen_func` itself is the highest-level Python swap point if we want to
   bypass the op entirely (replace the function object in the two import sites:
   `vllm.vllm_flash_attn` and `vllm.v1.attention.backends.fa_utils`). But the task says register the
   SAME op, so prefer (1).

---

## Configs vLLM actually exercises (for correctness/perf coverage)

- dtype: bf16 / fp16 (and fp8_e4m3 if quantized KV — `kv_cache_dtype`). For greedy decode default is
  bf16/fp16.
- Always VARLEN path: `cu_seqlens_q` provided; KV via `seqused_k` + `page_table` (paged KV) for
  decode/prefill in the paged backend; non-paged prefill uses `cu_seqlens_k`.
- `causal=True` for decoder self-attn; `causal=False` for encoder. Sliding window via `window_size`.
- `num_splits=0` (FA3 heuristic) normally; `>1` only under full-CUDA-graph with capped size;
  `VLLM_BATCH_INVARIANT` forces `num_splits=1`.
- `pack_gqa=None` (auto). GQA/MQA common (num_heads_q % num_heads_kv == 0). headdim typically 64/128
  (also 96/192/256 supported). `softcap` for some models (Gemma). `s_aux` (attention sink) for
  gpt-oss-style. `cp_world_size=1` normally.
- Output dtype = input dtype, except FP8 input -> BF16 output. softmax_lse is fp32, shape
  `(num_heads, total_q)` for varlen, `(b, num_heads, seqlen_q)` otherwise.

## Host-side mha_fwd essentials our .cu must replicate (flash_api.cpp 711-1287)
- Validates dtypes/shapes (see CHECK_SHAPE blocks), contiguous last dim.
- Allocates `out` if not given: varlen `{total_q, num_heads, head_size_v}`, else
  `{b, seqlen_q, num_heads, head_size_v}`, dtype out_type.
- Allocates `softmax_lse`: varlen `{num_heads, total_q}` fp32, else `{b, num_heads, seqlen_q}`.
- strides set IN ELEMENTS from `.stride(-3/-2/-1)`; batch stride only when not varlen.
- `is_causal = window_size_left<0 && window_size_right==0`; local otherwise; seqlen_q==1 special-cases
  causal off (decode). seqlen rounding to multiples of 128; headdim rounded via round_up_headdim.
- num_splits: `<=0` -> `get_num_splits(params)` heuristic; else use given. Split>1 allocates oaccum/
  lseaccum and runs combine kernel after main kernel. For correctness-first single kernel we can FORCE
  num_splits=1 internally (still must match outputs — splitKV is just numerically-equivalent reduction).
- Empty-seq handling: if total_q>0 but seqlen_k==0 -> `out.zero_()`, `softmax_lse.fill_(+inf)`.
- Device guard from q's device. Stream = `at::cuda::getCurrentCUDAStream()`.
- Return `{out, softmax_lse, out_accum, softmax_lse_accum}` (last two empty when num_splits==1).

## Gotchas
- 4-element return list is mandatory; wrapper does `out, lse, _, _ = ...`.
- Schema arg order/types must match EXACTLY (positional call). `is_rotary_interleaved` is hardcoded
  True by the wrapper but rotary is None so irrelevant.
- `softmax_scale` is `float`/`double` in schema (shim converts), `window_size_*`/`num_splits`/
  `sm_margin`/`cp_*` are `int`/`int64_t`. `qkv_dtype` in get_scheduler_metadata is `ScalarType`.
- Build ABI: must match torch's `_GLIBCXX_USE_CXX11_ABI` and Python limited API expectations.
- Double-registration of the `_vllm_fa3_C` namespace raises; never load both ours and baseline.
- We can define our OWN params struct (don't need binary compat) since we write the kernel; only the
  op schema + tensor shapes/dtypes/strides are the true contract.
