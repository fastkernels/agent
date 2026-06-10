# FA3 SM90 Forward Kernel Algorithm — distillation for a from-scratch PTX rewrite

Source: `reference/libraries/vllm-flash-attention/hopper/` (vLLM fork of FlashAttention-3, sm_90a).
Goal of this note: describe the algorithm at a level sufficient to hand-write a single self-contained `.cu`
targeting H200 (bf16/fp16 primary; FP8 secondary), reproducing baseline `_vllm_fa3_C.fwd` outputs and latency.

For the vLLM decode/prefill path you almost certainly only need a SUBSET. The features that matter for vLLM
greedy decode of ≤100 tokens (prefill+decode, paged KV cache):
- bf16/fp16, Is_causal, Is_local (sliding window), Has_softcap, PackGQA, paged KV, Split-KV + combine, softmax sinks (S_aux).
- You can likely DISABLE/ignore at first: FP8 (Is_FP8 / Transpose_V / V_colmajor / two-level accum), AppendKV (rotary in-kernel KV-cache append), HasQv (MLA-style Qv), LargeHeadDimV (>256, used for hdim 512 e.g. DeepSeek MLA), ClusterShape>1 / multicast (vLLM uses cluster=1 ⇒ `PipelineTmaAsyncNoCluster`).

================================================================================
## 0. Top-level structure (what the .cu must contain)
================================================================================
A single persistent warp-specialized kernel `FlashAttnFwdSm90::operator()` (`flash_fwd_kernel_sm90.h:183`):
- 1 producer (load) warpgroup + 1..3 consumer (MMA) warpgroups. WG = 128 threads = 4 warps.
- `MaxThreadsPerBlock = NumMmaThreads + 128` (e.g. 2 MMA WG ⇒ 384 threads = 256 MMA + 128 load).
- Grid is persistent: `TileScheduler::get_grid_shape`. SingleTileScheduler grid = (num_blocks_M, num_splits*num_head, num_batch); persistent/varlen schedulers launch ~num_SM blocks and loop work tiles.
- Dynamic smem ~227KB (max for H200). Layout overlaps mainloop smem with epilogue smem_o (smem_o aliases smem_v start).

Register reallocation via `setmaxnreg` (CUTLASS `warpgroup_reg_dealloc/alloc`):
- Producer WG: `warpgroup_reg_dealloc<LoadRegisterRequirement>()` (e.g. 24/32/40/56 regs).
- Consumer WG: `warpgroup_reg_alloc<MmaRegisterRequirement>()` (e.g. 160/232/240/256 regs).
- PTX: `setmaxnreg.dec.sync.aligned.u32 N;` (dealloc) and `setmaxnreg.inc.sync.aligned.u32 N;` (alloc). N must be a multiple of 8, called by all 128 threads of the WG (aligned).
- Values (`flash_fwd_kernel_sm90.h:86`): 1 MMA WG ⇒ load 56 / mma 256. 2 MMA WG ⇒ load 24 (TMA-KV) or 40, mma 240/232. 3 MMA WG ⇒ load 32, mma 160.

================================================================================
## 1. Warp specialization & per-WG roles
================================================================================
`warp_group_idx = canonical_warp_group_idx()` (= threadIdx.x/128 + 1 essentially; CUTLASS uses 1-based via warp idx).
- `warp_group_idx == 0` ⇒ PRODUCER (load) WG.
  - Only warp 0 of the WG does TMA issues when `NumProducerThreads == 32` (`SingleProducerWarp`); other warps early-return.
  - `NumProducerThreads = 32` when not transposing V and using TMA for Q+KV (the common bf16 path); else 128 (cp.async path / PagedKVNonTMA).
- `warp_group_idx >= 1` ⇒ CONSUMER (MMA) WG(s). Thread offset into MMA region = `MmaThreadOffset = 128`.

Init phase (all threads):
1. Thread (warp0, elect_one) issues `prefetch_tma_descriptors` for Q,K,V (and O store, K_new/V_new if AppendKV).
2. Barrier inits: `barrier_Q`, `barrier_Qv`, `barrier_O` (ClusterTransactionBarrier or ClusterBarrier), and all pipeline mbarriers (`pipeline_k`, `pipeline_v`, `pipeline_vt`, `pipeline_k_new`, `pipeline_v_new`). `fence_barrier_init()` then `__syncthreads()` (cluster=1) or `cluster_arrive/wait`.
3. S_aux (softmax sinks, one float per head) copied gmem→smem by first `num_heads` threads if `ptr_S_aux != nullptr`.

================================================================================
## 2. The pipeline (TMA loads into SMEM via mbarrier, ring buffer of kStages)
================================================================================
Multi-stage circular buffer in smem. `kStages` (typ 2) per K and V. Each stage is one kBlockN tile of K and of V.
Pipeline = CUTLASS `PipelineTmaAsync` (cluster>1) or `PipelineTmaAsyncNoCluster<kStages>` (cluster=1, FA3 perf fix — only 1/128 threads signals empty barrier on consumer_release; see `sm90_pipeline_no_cluster.hpp`). For cp.async path (PagedKVNonTMA): `PipelineAsync`.

Each stage has 2 mbarriers: FULL (producer→consumer: data arrived) and EMPTY (consumer→producer: buffer free). `PipelineState{index, phase, count}` tracks position; phase flips each time index wraps.

Producer per stage (`load`, mainloop hpp:617):
- `pipeline_k.producer_acquire(state)` → waits EMPTY barrier of that stage (phase).
- Issue TMA: `cute::copy(tma_load_K.with(*producer_get_barrier(state), mcast_mask, EVICT_LAST), gK, sK)`.
  - PTX: `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [smem], [tma_desc, coords], [mbar];` (TMA tensor copy). The mbarrier is the FULL barrier; producer must first `arrive_and_expect_tx(transaction_bytes)` (encoded in pipeline.with).
- For V the same, but V is loaded TRANSPOSED-by-layout: TMA descriptor for V uses shape (headdim_v, seqlen) with `select<1,0,2,3>` strides, so V lands in smem as Vt (column = seqlen). For bf16 no actual transpose; for FP8 row-major a real LDSM.T+STSM transpose (`transpose_V`) is run.
- Q is loaded ONCE per work tile via a separate `barrier_Q` (ClusterTransactionBarrier): `barrier_Q.arrive_and_expect_tx(bytes); cute::copy(tma_load_Q.with(barrier_Q,...), gQ, sQ);` EVICT_FIRST for non-split. Producer first waits on `FwdNamedBarriers::QueryEmpty` (MMA WGs signal smem_q free).
- `barrier_O.wait((work_idx+1)%2)`: producer waits until MMA WGs finished reading V/wrote O of the PREVIOUS work tile before overwriting smem (since smem_o overlaps smem_v).

Consumer per stage: `consumer_try_wait`/`consumer_wait(state)` waits FULL barrier; after using data `consumer_release(state)` arrives EMPTY barrier.
`producer_commit` for TMA is implicit (the TMA HW arrives the mbarrier via complete_tx). For cp.async path, `producer_commit(state, cpasync_barrier_arrive)`.

mbarrier PTX primitives to reimplement:
- init: `mbarrier.init.shared.b64 [mbar], count;`
- expect_tx: `mbarrier.arrive.expect_tx.shared.b64 _, [mbar], txbytes;` (or `.expect_tx` variant)
- arrive: `mbarrier.arrive.shared.b64 _, [mbar];`
- wait (try): `mbarrier.try_wait.parity.shared.b64 p, [mbar], phase;` loop until p.
- fence for smem visibility before TMA reads written smem: `fence.proxy.async.shared::cta;` (`fence_view_async_shared`).

================================================================================
## 3. The two GEMMs (wgmma.mma_async) and the inner loop
================================================================================
Two GMMA (warpgroup-MMA) ops per KV block:
- GEMM-0 (QK^T): S = Q @ K^T. `TiledMmaQK` from `GMMA::ss_op_selector<Element,Element,float,TileShape_MNK>` (SS = both operands from smem) — A=Q (smem), B=K (smem). Accumulator `tSrS` in registers (fp32), shape ((2,2,V),MMA_M,MMA_N).
- GEMM-1 (P@V): O += P @ V. `TiledMmaPV` `ss_op_selector<...,TileShape_MNK_PV, Major::K, MmaMajorV>` (or rs — P in registers). For bf16, `MmaPV_is_RS` is usually true ⇒ P stays in registers; for SS path, P is written to smem (`smem_p`) first via STSM.

GMMA = `wgmma.mma_async.sync.aligned.mNnNk16.f32.bf16.bf16` (shape e.g. m64n(kBlockN)k16). All 128 threads of a WG cooperatively issue; operands are described by 64-bit smem descriptors (matrix descriptor: base addr>>4, leading dim byte offset, stride, swizzle mode). Sequence per `flash::gemm` (utils.h:237):
1. `warpgroup_fence_operand` on accumulator (and RS operand).
2. `warpgroup_arrive()` → PTX `wgmma.fence.sync.aligned;`
3. Loop k_block: `cute::gemm(tiled_mma, A(_,_,k), B(_,_,k), C)` → `wgmma.mma_async...` accumulate. First k sets `ScaleOut::Zero` (zero_init), rest `One`.
4. `warpgroup_commit_batch()` → `wgmma.commit_group.sync.aligned;`
5. `if wg_wait>=0: warpgroup_wait<wg_wait>()` → `wgmma.wait_group.sync.aligned N;` (wait until ≤N groups outstanding).
6. `warpgroup_fence_operand(C)`.

### IntraWGOverlap software pipeline (the key perf trick; mainloop hpp:1212)
The default high-perf path. Each `fwd_step` computes: GEMM-0 (QK) for iter n, GEMM-1 (PV) for iter n+1, softmax for iter n — overlapping the PV of the previous block with QK of the current. Structure:
- Prologue (first n_block, n=n_block_max-1): QK gemm with `wg_wait=-1`, wait, release K; softcap; mask (Seqlenk+causal/local); `softmax.max_get_scale<Is_first=true>`; `online_softmax<Is_first=true>`; convert P fp32→bf16 (`convert_type_out`); write P to smem if SS; clear(tOrO).
- Steps (`fwd_step`), 3 phases interleaved:
  - `consumer_wait(pipeline_k, ++read)`; `warp_scheduler_barrier_sync()`; QK gemm (wg_wait=-1).
  - optional `rescale_o(tOrO, scores_scale)` if RescaleOBeforeGemm (hdim>128).
  - `consumer_wait(pipeline_v, read_v)`; PV gemm (wg_wait=-1) using PREVIOUS block's P.
  - `warp_scheduler_barrier_arrive()`; `warpgroup_wait<1>()`; release K.
  - softcap; mask; `max_get_scale<Is_first=false>` → scores_scale; `online_softmax<false>`; convert P; write P to smem; `rescale_o(tOrO, scores_scale)`; arrive P-write barrier.
  - `warpgroup_wait<0>()`; release V.
- Epilogue of mainloop: final PV gemm; `finalize`/`finalize_aux` (compute inv row_sum, LSE, handle sinks); `rescale_o(tOrO, scores_scale)`; release V.

Loop ordering (n_block descends from max-1 to min):
1. Masked iters (causal/local): from n_block_max-1 down to `n_block_min_causal_local_mask` — apply full row+col mask.
2. No-mask iters: down to `n_block_min_before_local_mask` — no per-element mask (interior).
3. Left local-mask iters (local only): down to n_block_min.
Splitting iters by masking regime avoids paying mask cost on interior blocks (huge perf).

### WarpSchedulerBarrier (2-WG ping-pong, `UseSchedulerBarrier`)
For NumMmaWarpGroups>=2 (and hdim<=128 bf16). WG1/WG2 alternate issuing GMMA so the two warpgroups don't both hammer the tensor cores at the same instant — improves wgmma issue overlap. Implemented with named barriers `WarpSchedulerWG1/WG2(+WG3)` (`mma_init`, `warp_scheduler_barrier_sync/arrive`). Each WG signals the NEXT WG's barrier after issuing its QK gemm.

================================================================================
## 4. Online softmax math (softmax.h)
================================================================================
Per consumer thread holds row_max, row_sum (size `kNRows = 2*(2*kBlockM/NumMmaThreads)` for bf16; rows owned by this thread per GMMA C layout). `softmax_scale_log2 = softmax_scale * log2(e)` (and *= softcap pre-scaling if softcap, *= q_descale*k_descale if FP8).

Per KV block (after S = QK^T in fp32 registers, scoremod+mask applied):
1. `max_get_scale<Is_first>`: 
   - first: `row_max = rowwise max(S)` (thread reduce over cols then `quad_allreduce` = `__shfl_xor` over 4 lanes that share a row); scores_scale = 1.
   - else: save prev max; `row_max = max(prev, rowmax(S))`; `scores_scale = exp2((prev_max - cur_max)*scale_log2)`; `row_sum *= scores_scale`.
2. `online_softmax<Is_first>`: `scale_apply_exp2(S, row_max, scale_log2)` ⇒ `S(i,j)=exp2(S(i,j)*scale - row_max*scale)`. (exp2 not exp ⇒ uses `ex2.approx.f32`/`exp2f`; ffma fuses scale.) Then `reduce_sum` over cols into row_sum WITHOUT cross-thread reduce (deferred).
   - Check_inf: if row_max==-inf treat max_scaled=0 to avoid NaN.
3. `rescale_o(tOrO, scores_scale)`: multiply running O accumulator rows by scores_scale (correct for the new max).

`finalize(v_descale)` (after last block): `quad_allreduce(row_sum)` (cross-lane sum); `inv_sum = 1/sum` (0 if sum 0/NaN); `scores_scale = inv_sum * v_descale`; `row_sum = LSE = row_max*scale_log2*ln2 + ln(sum)` (= max + log sum-exp). FP8: divide sum by 2^Max_offset.

`finalize_aux(tSrSAux)` (softmax SINKS): adds `exp2(ln2*sink - max_scaled)` to the denominator sum before inverting — i.e. an extra virtual logit per head equal to the sink value `S_aux[head]`. Used by models with attention sinks (e.g. gpt-oss). Sink value loaded from `ptr_S_aux` (per-head) into smem; broadcast to all rows (or per-row head mapping if PackGQA).

`Max_offset` (FP8 only, =8): subtract 8 from max before exp2 so exp2 outputs in [0,256] to use more FP8 range; divide sum by 256 at finalize.

Reduce helpers (utils.h:78): `Allreduce<4>::run` = two `__shfl_xor_sync` (xor 2, xor 1) over the 4 threads in a GMMA row quad. Use `MaxOp`/`SumOp`.

================================================================================
## 5. Masking (mask.h) — causal / sliding-window / softcap / sinks
================================================================================
Applied to S (rowcol layout) before softmax. Templated on `Seqlenk_mask`, `Causal_mask`, `Local_mask` (mutually exclusive causal/local). Uses identity tensor `cS` partitioned by the MMA to get (row,col) of each element; uses thread0's col indices (compile-time) and subtracts thread_col_offset.
- Seqlenk only (non-causal, last block): mask col where `col_idx >= seqlen_k - n_block*kBlockN`.
- Causal: `col_limit_right = row_idx + (1 + seqlen_k - n_block*kBlockN - seqlen_q - thread_col_offset)`; mask col ≥ limit. Uses `__viaddmin_s32` to fuse add+min with seqlenk limit. row_idx = `m_block*kBlockM + row` (PackGQA: divmod by qhead_per_khead, shuffled within row quad).
- Local (sliding window): two limits — right = causal_row_offset+window_size_right; left = causal_row_offset-1-window_size_left; plus sink columns (`col_limit_sink = sink_token_length - n_block*kBlockN`) kept. Mask if `col≥right || (col<left && col≥sink)`.
- Context-parallel (cp_world_size>1): remaps to absolute k index `local_k_idx*cp_world_size + cp_rank` and compares to `row_idx + tot_seqlen_k - seqlen_q`. (vLLM CP patch; ignore unless CP used.)

Softcap (utils.h:641, before masking): `S = tanh(S * softcap_scaled)`; pre-scaling: params.softcap_val = softmax_scale/softcap; params.softmax_scale_log2 = softcap * log2(e). So effective: `tanh(S*softmax_scale/softcap)`, then exp2 with scale=softcap*log2e ⇒ `exp(tanh(...)*softcap*ln2/ln2)`. `fast_tanh` = `tanh.approx.f32` PTX.

n_offset / sliding window block skipping (block.h `BlockMN::get_n_block_min_max`): vLLM patch — for local attention, `n_offset = max(0, m_idx_min + seqlen_k - seqlen_q - window_size_left)`; KV gmem pointers are advanced by n_offset and seqlen_k reduced ⇒ skip KV blocks fully left of the window. This is the load-bearing vLLM local-attn change.

================================================================================
## 6. Block range computation (block.h)
================================================================================
`get_n_block_min_max(seqlen_info, m_block, ..., window_l, window_r, qhead_per_khead) -> {n_block_min, n_block_max, n_offset}`:
- n_block_max = ceil_div(seqlen_k, kBlockN). Causal/local: clamp by `m_idx_max + tot_seqlen_k - seqlen_q + window_size_right`.
- Split: divide [n_block_min,n_block_max) among num_splits; each split takes a contiguous chunk. Dynamic num_splits encoded in top 16 bits of split_idx (`split_idx>>16`); actual split index = low 16 bits.
- Returns early (no work) when `n_block_max <= n_block_min` ⇒ producer skips load, consumer writes zeros+(-inf LSE) via `epilogue.store_zero`.

================================================================================
## 7. PackGQA (pack_gqa.h)
================================================================================
Packs `qhead_per_khead` query heads into the M dimension so one KV head's block serves all its Q heads (good for decode where seqlen_q=1 but many Q heads share a KV head). Q reshaped to `((qhead_per_khead, seqlen_q), d, nheads_kv, batch)`.
- If `kBlockH>1` (PackGQA_TMA) Q can still use TMA; else Q loaded with cp.async (`load_Q`, `SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>` = `cp.async.cg.shared.global` 16B). Pointers per row computed via divmod(qhead_per_khead) and shared across the row's threads with `__shfl_sync` (expensive divmod amortized).
- mask & LSE/O store also use divmod to map packed M-row → (h_idx, m_idx). `store_O`, `store_LSE`, `store_O_direct` (partial/split). Row valid if `idx < seqlen_q * qhead_per_khead`.

================================================================================
## 8. Paged KV (paged_kv.h) + AppendKV
================================================================================
Two modes:
- TMA paged (`!PagedKVNonTMA`, page_size % kBlockN == 0, not local): page table maps block → physical KV page; `get_indices_for_K_TMA` returns (n_block_idx, bidb_kv_idx) used to index the TMA tensor's batch dim. Requires page_size multiple of kBlockN and no leftpad/local.
- cp.async paged (`PagedKVNonTMA`): per-row page-table lookup; threads in a row share the page entry; `load_K<Seqlenk_mask>`/`load_V` issue cp.async with per-row computed pointers, ZFILL OOB. Slower but flexible (local, leftpad, arbitrary page size).
AppendKV (decode KV-cache write): `load_kv_new`/`store_kv_new` — load new K/V (always TMA), optionally apply rotary, write into the cache in-place (TMA store or cp.async/paged store), with a NamedBarrier (`AppendKV`) + `fence.proxy.async.global` so the producer's later read sees the write. Synchronizes producer↔consumer across the AppendKV pipeline. For vLLM, KV-cache append usually handled outside; may be skippable.

================================================================================
## 9. Split-KV + combine (flash_fwd_combine_kernel.h)
================================================================================
For long KV with few CTAs: split the KV range into num_splits chunks (each CTA/split computes partial O + partial LSE). Output written to `ptr_O_partial` (fp32) and `ptr_LSE_partial`, indexed by split_idx in the last tensor dim. Epilogue `store` writes partials when `is_split` (no smem TMA; direct fp32 gmem write `store_O_direct`).
A separate combine kernel (`flash_fwd_combine.cu`) reduces partials: for each (row,head) load all splits' LSE, compute global max LSE, weight each partial O by `exp(lse_split - lse_max)`, sum, divide by sum of weights ⇒ final O + final LSE. Dynamic num_splits chosen by `flash_prepare_scheduler.cu` based on seqlen/SM count. For ≤100-token decode this is the standard decode acceleration; reproduce both kernels.

================================================================================
## 10. Tile sizes per head dim (tile_size.h, bf16 sm90)
================================================================================
`tile_size_fwd_sm90(headdim, headdim_v, is_causal, is_local, elem_size, ...) -> {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}`:
- hd<=64, hdv=512: {64,64,false,false}; hdv=256: {128,96,true,false}; else (use_one_mma_wg): {64,192,true,true}; else {192, causal?128 : local||pagedNonTMA?160 :192, causal||local, !local}.
- hd<=96: {192, local||pagedNonTMA?128:144, false, true}.
- hd<=128: one_mma_wg {64, causal||local||pagedNonTMA?128:176, true, true}; else {128, causal||local||pagedNonTMA?128:160, true, true}.  ← hdim128 is the common LLM case.
- hd<=192: {128, pagedNonTMA||local?96:(hdv<=128?128:96), true, true}.
- hd>192 (256): {128, local?64:80, true, true}.
kStages typically 2 (set in launch template). NumMmaWarpGroups derived from kBlockM/64 (AtomLayoutQK = kBlockM/64).
FP8 path: smaller tiles + two-level accumulation (separate fp32 accumulator). headdim128 fp8: {128,192,true,true}.

================================================================================
## 11. Epilogue (epilogue_fwd.hpp)
================================================================================
After mainloop produces `tOrO` (fp32 accumulator, already rescaled by 1/row_sum) and `lse`:
1. Convert tOrO fp32→Element (bf16) (`convert_type_out`); FP8 col permute if needed.
2. `EpilogueBarrier` named sync (ensure all WGs done reading V — since smem_o aliases smem_v).
3. Use_smem: rmem O → smem (STSM `SM90_U32x4_STSM_N`), then either TMA store (`Use_TMA_O`, non-varlen non-split) or vectorized gmem store. Use_TMA_O store: `cp.async.bulk.tensor...global.shared` (`tma_store_O`), `tma_store_arrive; tma_store_wait<0>;` then arrive `barrier_O` (lets producer reuse smem).
4. Write LSE rmem→gmem (one value per row by lane where col==0). PackGQA uses `store_LSE` with divmod.
5. Split path: direct fp32 partial write to `ptr_O_partial` (no smem).
`store_zero`: writes 0 to O and -inf to LSE for empty tiles.

================================================================================
## 12. Load-bearing CUTLASS/CuTe primitives → raw PTX/intrinsic equivalents
================================================================================
| CUTLASS/CuTe thing | What it is | Raw equivalent to hand-write |
|---|---|---|
| `make_tma_copy` + `cp.async.bulk.tensor` | TMA tensor descriptor + async bulk copy gmem↔smem | Build `CUtensorMap` host-side (cuTensorMapEncodeTiled); device `cp.async.bulk.tensor.{2,3}d.shared::cluster.global.mbarrier::complete_tx::bytes`; store: `cp.async.bulk.tensor...global.shared::cta.bulk_group` + `cp.async.bulk.commit_group` + `cp.async.bulk.wait_group`. |
| `cute::prefetch_tma_descriptor` | warm TMA descriptor | `prefetch.tensormap [desc];` |
| `wgmma` GMMA (`ss_op_selector`/`rs_op_selector`) | warpgroup MMA m64nNk16 | `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 {acc...}, descA, descB, scaleD, ...;`. Build 64-bit smem matrix descriptors (addr/16, LBO, SBO, swizzle). RS variant: A operand from registers. |
| `warpgroup_arrive/commit_batch/wait` | GMMA sync | `wgmma.fence.sync.aligned;`, `wgmma.commit_group.sync.aligned;`, `wgmma.wait_group.sync.aligned N;`. Also `wgmma.fence` on operands. |
| `cutlass::PipelineTmaAsync` / mbarrier | stage FULL/EMPTY barriers | `mbarrier.init`, `mbarrier.arrive[.expect_tx]`, `mbarrier.try_wait.parity` (poll), `mbarrier.arrive` for empty. Track {phase,index}. |
| `NamedBarrier::sync/arrive` (named_barrier.hpp) | inter-WG sync, arbitrary thread count | `bar.sync  barrier_id, num_threads;` / `bar.arrive barrier_id, num_threads;` (barrier ids 0..15; reserve high ids for user). |
| `ClusterTransactionBarrier barrier_Q/O` | transaction mbarrier for Q load / O reuse | mbarrier with expect_tx (Q TMA bytes); barrier_O is a plain mbarrier arrived by epilogue, waited by producer (phase = (work_idx+1)%2). |
| `warpgroup_reg_dealloc/alloc` | per-WG register budget | `setmaxnreg.dec/inc.sync.aligned.u32 N;` (all 128 threads, N%8==0). |
| `cute::copy` STSM / LDSM (`SM90_U32x4_STSM_N`, `SM75_U16x8_LDSM_T`) | shared↔reg matrix transpose copies | `stmatrix.sync.aligned.m8n8.x4.shared.b16`, `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16`. STSM for P→smem & O→smem; LDSM.T for FP8 V transpose. |
| `cp.async` (PackGQA Q / paged cp.async) | non-TMA async copy | `cp.async.cg.shared.global [smem],[gmem],16;` + `cp.async.commit_group` + `cp.async.wait_group`. ZFILL variant for OOB. |
| `exp2f`/`fast_tanh` | softmax exp / softcap | `ex2.approx.f32`, `tanh.approx.f32`. |
| `__shfl_xor_sync`/`__shfl_sync` | quad allreduce & pointer broadcast | `shfl.sync.bfly.b32` / `shfl.sync.idx.b32`. |
| `__viaddmin_s32` | fused add+min in mask | `vimin/viaddmin` or `add` then `min.s32`. |
| `fence_view_async_shared` / `fence.proxy.async` | async-proxy smem/gmem ordering | `fence.proxy.async.shared::cta;` / `fence.proxy.async.global;`. |
| `cute::elect_one_sync` | pick 1 lane | `elect.sync` (sm90) or `%laneid==0` w/ ballot. |
| `launch_dependent_grids`/`wait_on_dependent_grids` | PDL (programmatic dependent launch) | `griddepcontrol.launch_dependent_grids` / `.wait`. Used for split→combine overlap; optional. |
| `cutlass::FastDivmod` | fast int divmod (qhead_per_khead, page) | precompute magic/shift on host, mulhi+shift on device. |

================================================================================
## 13. Concrete reproduction checklist for the agent's .cu
================================================================================
- Phase 1 (correctness, bf16 only, no fancy features): non-persistent SingleTileScheduler, cluster=1, kStages=2, IntraWGOverlap on, hdim128 tile {128,128/160}, 2 MMA WG + 1 load WG. Implement: TMA Q/K/V load + mbarrier pipeline; wgmma QK^T (SS); online softmax (exp2, quad allreduce); wgmma PV (RS, P in regs); causal+seqlenk mask split into masked/unmasked iters; epilogue STSM→smem→TMA store O + LSE write. Validate vs `_vllm_fa3_C.fwd` on a few shapes BEFORE optimizing.
- Phase 2: add PackGQA (decode), paged KV (TMA mode first), sliding window + n_offset skip, softcap, softmax sinks (S_aux), Split-KV + combine kernel. These cover vLLM prefill+decode.
- Phase 3 (perf): WarpSchedulerBarrier ping-pong, register reallocation tuning, EVICT hints, descriptor prefetch, RescaleOBeforeGemm for hdim>128, kStages tuning.
- mha_fwd host fn: unpack torch tensors, alloc O/LSE, build TMA descriptors (CUtensorMap), pick tile via `tile_size_fwd_sm90`, set softmax_scale_log2, launch persistent grid; TORCH_LIBRARY registration. NO torch inside __global__.

## 14. Gotchas
- smem_o ALIASES smem_v: producer must `barrier_O.wait` before overwriting; epilogue arrives barrier_O after reading V / writing O. Get the (work_idx+1)%2 / work_idx%2 phase right or deadlock.
- V is loaded transposed-by-TMA-layout (descriptor shape (hdv,seqlen)); for bf16 no actual transpose, GMMA PV uses MmaMajorV=MN. FP8 row-major needs real LDSM.T/STSM transpose (`copy_Vt_to_V`).
- The `permute_Cregs_fp8`/`permute_Aregs_fp8` register shuffles are ONLY for FP8 layout fixups; not needed for bf16.
- `softmax_scale_log2` already folds log2(e) (and softcap scaling). exp2 used everywhere, not exp.
- row_sum cross-thread reduce is DEFERRED to `finalize` (only thread-local sum during loop) — must do `quad_allreduce` at the end.
- Masked vs unmasked iteration split is essential for both correctness (boundaries) and perf (don't mask interior).
- Producer SingleProducerWarp: only warp 0 issues TMA when NumProducerThreads==32; other producer warps early-return — replicate or you under/over-arrive barriers.
- Q EVICT_FIRST, K/V EVICT_LAST cache hints (L2) — matter for perf, not correctness.
