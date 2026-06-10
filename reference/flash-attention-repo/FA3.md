# FA3 — distillation reference

All paths are **relative to the repo root** `vllm_repo/vllm-flash-attention/`.

## Target

| | |
|---|---|
| Compiled extension | `_vllm_fa3_C` (built by `CMakeLists.txt:180-313`) |
| Kernel arch | `9.0a` only — **Hopper / SM90** (FA3 is not built for Blackwell) |
| Use in vLLM | Default dense backend on Hopper; also FlashAttn-MLA prefill on Hopper |
| Ops exposed | `fwd` (required) · `get_scheduler_metadata` (required) |
| Scope | **Forward only** — backward/dropout/append-KV/sm8x/cluster all disabled by compile flags |

**Active compile flags** (`CMakeLists.txt:294-313`) that shrink the surface:
`FLASHATTENTION_DISABLE_BACKWARD`, `_DISABLE_DROPOUT`, `_DISABLE_APPENDKV`, `_DISABLE_SM8x`, `_DISABLE_CLUSTER`, `_DISABLE_UNEVEN_K`, `VARLEN_ONLY`, `PACKGQA_ONLY`. → Only the SM90 forward, varlen, pack-GQA path is live.

vLLM Python wrapper: `vllm_repo/vllm/vllm/vllm_flash_attn/flash_attn_interface.py` → `flash_attn_varlen_func` (fa_version==3 branch, line 326) calls `torch.ops._vllm_fa3_C.fwd`; `get_scheduler_metadata` (line 122) calls `torch.ops._vllm_fa3_C.get_scheduler_metadata`.

---

## Functions to port — call tree (file:line)

### 1. Op registration
- `hopper/flash_api_torch_lib.cpp:92` — `TORCH_LIBRARY_EXPAND` (binds `fwd` → `mha_fwd`, `get_scheduler_metadata` → `mha_fwd_get_scheduler_metadata`)

### 2. Host entries + parameter setup — `hopper/flash_api.cpp`
- **`mha_fwd` — :711**  ← top-level forward op (runs to ~1637)
- **`mha_fwd_get_scheduler_metadata` — :554**  ← second op (AOT tile-scheduler metadata)
- `set_params_fprop` — :70
- `run_mha_fwd` — :267  (dispatcher: arch × split × paged × packGQA × softcap × dtype × headdim)
- `run_mha_fwd_combine` — :406  (split-KV reduction launcher)
- `get_pagedkv_tma` — :434 · `get_pack_gqa` — :446  (heuristics)
- `prepare_varlen_num_blocks` — used at :696 (def in `flash_prepare_scheduler.cu`, below)

### 3. Launch template — `hopper/flash_fwd_launch_template.h`
- `run_mha_fwd_` — :209  (instantiates kernel by `<Arch, dtype, headdim, headdimV, Split, PagedKVNonTMA, Softcap, PackGQA>`)
- `run_flash_fwd` — :31  (builds CUTLASS kernel + epilogue + scheduler, configures TMA, launches)

### 4. Device kernel + collectives (the heart of FA3)
- `hopper/flash_fwd_kernel_sm90.h` — `class FlashAttnFwdSm90` :28 · kernel body `operator()` :183 (warp-specialized producer/consumer)
- **`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` — `struct CollectiveMainloopFwdSm90` :34**  ← TMA loads + WGMMA + softmax pipeline (main work)
- `hopper/epilogue_fwd.hpp` — `struct CollectiveEpilogueFwd` :25 (writes O + LSE)

### 5. Split-KV combine — `hopper/flash_fwd_combine_kernel.h` + launch template
- `hopper/flash_fwd_combine_launch_template.h` — `run_flash_fwd_combine` :21 · `run_mha_fwd_combine_` :64
- `hopper/flash_fwd_combine.cu` — instantiation TU
- `hopper/flash_fwd_combine_kernel.h` — combine kernel body

### 6. Tile scheduler — `hopper/flash_prepare_scheduler.cu` + `hopper/tile_scheduler.hpp`
- `prepare_varlen_num_blocks_kernel` — `flash_prepare_scheduler.cu:43` (`__global__`)
- `prepare_varlen_num_blocks` — `flash_prepare_scheduler.cu:214`
- `TileSchedulerArguments` / scheduler structs — `tile_scheduler.hpp:18`

### 7. Shared device helpers (definitions to inline / re-implement)
- `hopper/softmax.h` — `struct Softmax` :93 (online softmax)
- `hopper/mask.h` — `struct Mask` :18 (causal / sliding-window)
- `hopper/block.h` — `struct BlockMN` :10 (n/m block range incl. the vLLM `n_offset` local-attn patch)
- `hopper/seqlen.h` — `struct SeqlenInfoQK` :32 (varlen offsets; FA3 fork adds context-parallel `tot_seqlen_k`)
- `hopper/pack_gqa.h` — `struct PackGQAManager` :18 (GQA head packing — always on here)
- `hopper/paged_kv.h` — `struct PagedKVManager` :18 (paged KV with TMA gather)
- `hopper/utils.h`, `hopper/named_barrier.hpp`, `hopper/sm90_pipeline_no_cluster.hpp`, `hopper/copy_sm90_bulk_reduce.hpp` — WGMMA/TMA/barrier/pipeline primitives (CUTLASS-heavy)
- `hopper/tile_size.h`, `hopper/heuristics.h` — tile-size + split/packgqa heuristics
- `hopper/flash.h` — `Flash_fwd_params` struct (data contract; fork adds FP8 descale, `s_aux` sinks, `cp_*` params)
- `hopper/cuda_check.h`, `hopper/static_switch.h` — boilerplate / compile-time switches

---

## Full list of files used by FA3

### A. Core hand-written sources (real definitions — what you port). FORWARD path:
```
csrc/common/pytorch_shim.h
csrc/common/registration.h
hopper/flash_api.cpp
hopper/flash_api_torch_lib.cpp
hopper/flash.h
hopper/flash_fwd_kernel_sm90.h
hopper/flash_fwd_launch_template.h
hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
hopper/epilogue_fwd.hpp
hopper/flash_fwd_combine.cu
hopper/flash_fwd_combine_kernel.h
hopper/flash_fwd_combine_launch_template.h
hopper/flash_prepare_scheduler.cu
hopper/tile_scheduler.hpp
hopper/tile_size.h
hopper/heuristics.h
hopper/softmax.h
hopper/mask.h
hopper/block.h
hopper/seqlen.h
hopper/pack_gqa.h
hopper/paged_kv.h
hopper/utils.h
hopper/named_barrier.hpp
hopper/sm90_pipeline_no_cluster.hpp
hopper/copy_sm90_bulk_reduce.hpp
hopper/cuda_check.h
hopper/static_switch.h
```

### B. In the include graph but COMPILED OUT (backward + sm8x — disabled by flags, skip for a forward distill)
```
hopper/flash_fwd_kernel_sm80.h
hopper/mainloop_fwd_sm80.hpp
hopper/flash_bwd_kernel_sm80.h
hopper/flash_bwd_kernel_sm90.h
hopper/flash_bwd_launch_template.h
hopper/flash_bwd_preprocess_kernel.h
hopper/flash_bwd_postprocess_kernel.h
hopper/epilogue_bwd.hpp
hopper/mainloop_bwd_sm80.hpp
hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp
hopper/rotary.h                # fused rotary; vLLM passes rotary args as None
```

### C. Generated instantiation `.cu` stubs (auto-generated by `hopper/instantiations/generate_kernels.py`; each instantiates `run_mha_fwd_<90, dtype, hdim, hdimV, Split, PagedKVNonTMA, Softcap, PackGQA>` — reproduce the matrix, don't hand-port)

**190 files**, matrix = `hdim ∈ {64, 96, 128, 192, 256}` (+ head-dim-diff `64→256`, `64→512`, `192→128`) × dtype `{bf16, fp16, e4m3}` × variant `{packgqa, split, paged, paged_split}` × `{±softcap}`. CMake globs at `CMakeLists.txt:182-251`. Full list:

```
hopper/instantiations/flash_fwd_hdim128_bf16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim128_bf16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim128_bf16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim128_bf16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim128_bf16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim128_bf16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim128_bf16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim128_bf16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim128_e4m3_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim128_e4m3_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim128_e4m3_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim128_e4m3_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim128_e4m3_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim128_e4m3_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim128_e4m3_split_sm90.cu
hopper/instantiations/flash_fwd_hdim128_e4m3_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim128_fp16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim128_fp16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim128_fp16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim128_fp16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim128_fp16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim128_fp16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim128_fp16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim128_fp16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_bf16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_bf16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_bf16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_bf16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_bf16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_bf16_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_bf16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_bf16_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_bf16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_bf16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_e4m3_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_e4m3_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_e4m3_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_e4m3_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_e4m3_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_e4m3_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_e4m3_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_e4m3_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_e4m3_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_e4m3_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_fp16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_fp16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_fp16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_fp16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_fp16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_fp16_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_fp16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_fp16_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_fp16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_128_fp16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_bf16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_bf16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim192_bf16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_bf16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_bf16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_bf16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_bf16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_bf16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_e4m3_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_e4m3_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim192_e4m3_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_e4m3_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_e4m3_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_e4m3_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_e4m3_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_e4m3_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_fp16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_fp16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim192_fp16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_fp16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_fp16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim192_fp16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim192_fp16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim192_fp16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim256_bf16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim256_bf16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim256_bf16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim256_bf16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim256_bf16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim256_bf16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim256_bf16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim256_bf16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim256_e4m3_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim256_e4m3_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim256_e4m3_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim256_e4m3_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim256_e4m3_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim256_e4m3_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim256_e4m3_split_sm90.cu
hopper/instantiations/flash_fwd_hdim256_e4m3_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim256_fp16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim256_fp16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim256_fp16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim256_fp16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim256_fp16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim256_fp16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim256_fp16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim256_fp16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_bf16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_bf16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_bf16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_bf16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_bf16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_bf16_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_bf16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_bf16_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_bf16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_bf16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_fp16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_fp16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_fp16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_fp16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_fp16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_fp16_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_fp16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_fp16_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_fp16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_256_fp16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_bf16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_bf16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_bf16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_bf16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_bf16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_bf16_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_bf16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_bf16_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_bf16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_bf16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_fp16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_fp16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_fp16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_fp16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_fp16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_fp16_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_fp16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_fp16_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_fp16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_512_fp16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_bf16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_bf16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim64_bf16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_bf16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_bf16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_bf16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_bf16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_bf16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_e4m3_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_e4m3_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim64_e4m3_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_e4m3_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_e4m3_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_e4m3_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_e4m3_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_e4m3_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_fp16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_fp16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim64_fp16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_fp16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_fp16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim64_fp16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim64_fp16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim64_fp16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim96_bf16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim96_bf16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim96_bf16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim96_bf16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim96_bf16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim96_bf16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim96_bf16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim96_bf16_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim96_e4m3_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim96_e4m3_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim96_e4m3_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim96_e4m3_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim96_e4m3_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim96_e4m3_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim96_e4m3_split_sm90.cu
hopper/instantiations/flash_fwd_hdim96_e4m3_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim96_fp16_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim96_fp16_paged_sm90.cu
hopper/instantiations/flash_fwd_hdim96_fp16_paged_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim96_fp16_paged_split_sm90.cu
hopper/instantiations/flash_fwd_hdim96_fp16_paged_split_softcap_sm90.cu
hopper/instantiations/flash_fwd_hdim96_fp16_softcap_packgqa_sm90.cu
hopper/instantiations/flash_fwd_hdim96_fp16_split_sm90.cu
hopper/instantiations/flash_fwd_hdim96_fp16_split_softcap_sm90.cu
```

### D. External dependencies (the boundary to REMOVE for a no-CUTLASS build)
FA3 is **fundamentally CUTLASS/CuTe**: `cute/*`, `cutlass/pipeline/*`, `cutlass/gemm/collective/builders/sm90_common.inl`, `cutlass/arch/grid_dependency_control.h`, TMA + WGMMA + mbarrier. Removing CUTLASS here means hand-writing TMA descriptors (`cp.async.bulk.tensor`), `wgmma.mma_async`, and warp-group/mbarrier orchestration from PTX. **Budget this as a rewrite, not an extraction.**

---

## Feature set the forward op must honor (from the args vLLM passes)
fp16/bf16 **+ FP8 e4m3** (`q/k/v_descale`) · varlen · GQA/MQA (pack-GQA) · paged KV (`page_table`) · causal · sliding window · softcap · **attention sinks** (`s_aux`) · **split-KV** (`num_splits>1` + combine) · **context parallel** (`cp_world_size/rank/tot_seqused_k`) · AOT scheduler metadata · returns `softmax_lse`. **No** ALiBi. Fused rotary / KV-append (`k_new/v_new`) are passed `None` by vLLM today → deprioritize.
