# Baseline Integration: correctness + performance grounding for the FA3-distillation agent

Environment confirmed live: torch 2.11.0+cu130, CUDA 13.0, vllm 0.22.0, device = NVIDIA H200,
capability (9,0) = sm_90a. L2 = 60 MiB (62,914,560 bytes), 132 SMs, max SM clock 1980 MHz,
mem clock 3201 MHz. **Clocks currently idle at 345 MHz (DVFS active); `nvidia-smi --lock-gpu-clocks`
is DENIED for our user (no sudo).** This dominates the latency-measurement design (see section D).

Baseline lib: `/home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so`
(153 MB). Wrapper: same dir `flash_attn_interface.py`. Backend driver:
`vllm/v1/attention/backends/flash_attn.py` (1223 lines) + `vllm/v1/attention/backends/fa_utils.py`.

---

## A. How to call the prebuilt baseline directly to get golden outputs

Two ops are registered by the `.so` (verified via `torch._C._jit_get_schemas_for_operator`):

### A.1 `torch.ops._vllm_fa3_C.fwd` — EXACT schema (38 args, returns `Tensor[]`)
```
_vllm_fa3_C::fwd(
  Tensor($0!->) q, Tensor k, Tensor v,
  Tensor? k_new, Tensor? v_new, Tensor? q_v,
  Tensor($1!->)? out,
  Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new,
  Tensor? seqused_q, Tensor? seqused_k,
  int? max_seqlen_q, int? max_seqlen_k,
  Tensor? page_table, Tensor? kv_batch_idx, Tensor? leftpad_k,
  Tensor? rotary_cos, Tensor? rotary_sin, Tensor? seqlens_rotary,
  Tensor? q_descale, Tensor? k_descale, Tensor? v_descale,
  float softmax_scale, bool is_causal,
  int window_size_left, int window_size_right,
  float softcap, bool is_rotary_interleaved,
  Tensor? scheduler_metadata, int num_splits, bool? pack_gqa, int sm_margin,
  Tensor? s_aux, int cp_world_size, int cp_rank, Tensor? cp_tot_seqused_k
) -> Tensor[]
```
**Returns a list of 4**: `[out, softmax_lse, out_accum, softmax_lse_accum]`. In normal
(non-split-returning) calls the last two are `None`. The Python wrapper unpacks
`out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd(...)`.

Note `q` arg0 and `out` arg6 are aliased/mutable (`$0!`, `$1!`): if `out` is passed it is
written in place. Our distilled `.cu` must keep this aliasing contract (out optional, written in place).

### A.2 `torch.ops._vllm_fa3_C.get_scheduler_metadata` — EXACT schema (returns int32 Tensor)
```
_vllm_fa3_C::get_scheduler_metadata(
  int batch_size, int max_seqlen_q, int max_seqlen_k,
  int num_heads, int num_heads_k, int headdim, int headdim_v,
  ScalarType qkv_dtype, Tensor seqused_k,
  Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new,
  Tensor? seqused_q, Tensor? leftpad_k,
  int? page_size, int max_seqlen_k_new,
  bool is_causal, int window_size_left, int window_size_right,
  bool has_softcap, int num_splits, bool? pack_gqa, int sm_margin
) -> Tensor
```
Returned tensor is small int32 (observed shape `(9,)` for bs=4; formula in vLLM code:
`1 + round_up(batch_size,4)*4` for CUDA-graph buffer sizing). It carries the AOT tile schedule
(which SM does which (batch,head,split) tile). **It is OPTIONAL** — passing `scheduler_metadata=None`
makes the kernel compute its own schedule at launch. For the distilled `.cu` the simplest correct path
is to NOT implement AOT scheduling and ignore `scheduler_metadata` (compute schedule on device);
but performance parity on CUDA-graph/decode paths may need it (see gotchas).

### A.3 Easiest golden path: call through the Python wrapper
`flash_attn_varlen_func(..., fa_version=3)` (signature in `flash_attn_interface.py`). It maps cleanly
onto `fwd`. **Verified working** with a standalone script (no vLLM engine needed) — see exact tensor
shapes/layouts in section B. The wrapper sets defaults: `is_rotary_interleaved=True`, `sm_margin=0`,
`softmax_scale = q.shape[-1]**-0.5` if None.

### A.4 Tensor layouts (CRITICAL — these are what vLLM passes)
- **q**: `(total_q, nheads, headdim)` packed varlen (total_q = sum of per-seq query lens).
  For decode, total_q = batch_size (1 query token/seq).
- **Paged KV** (the decode/most-common path): `k`,`v` are the cache tensors of shape
  `(num_blocks, block_size, num_kv_heads, headdim)`. vLLM allocates kv_cache as
  `(2, num_blocks, block_size, num_kv_heads, head_size)` then `kv_cache.unbind(0)` →
  separate k_cache, v_cache (see `get_kv_cache_shape` flash_attn.py:138). `block_size` must be
  a multiple of 16. `page_table`/`block_table` is `(batch, max_blocks_per_seq)` int32 mapping
  logical→physical blocks.
- **Non-paged prefill**: `k`,`v` are `(total_k, num_kv_heads, headdim)` packed varlen, `block_table=None`,
  use `cu_seqlens_k` (NOT seqused_k).
- **cu_seqlens_q / cu_seqlens_k**: `(batch+1,)` int32 cumulative offsets.
- **seqused_k**: `(batch,)` int32 actual K length per sequence — used for paged/decode instead of cu_seqlens_k.
  vLLM passes `seqused_k = attn_metadata.seq_lens`, `cu_seqlens_q = attn_metadata.query_start_loc`.
- **softmax_lse out**: shape `(nheads, total_q)` — note this is (heads, tokens), transposed vs out.
  Confirmed: decode bs=2,nh=8 → lse `(8,2)`.
- **fp8 descales**: `q_descale/k_descale/v_descale` shape `(batch, num_kv_heads)` fp32. vLLM builds them
  via `layer._q_scale.expand((cu_seqlens_q.shape[0]-1, num_kv_heads))`.

### A.5 Golden-harness recipe (validated, runs in <1s, NO vLLM engine)
```python
import torch, vllm.vllm_flash_attn._vllm_fa3_C
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func, get_scheduler_metadata
# build q/k/v per layouts above, then:
out, lse = flash_attn_varlen_func(q=q,k=kc,v=vc, max_seqlen_q=Mq, cu_seqlens_q=cu_q,
    max_seqlen_k=Mk, seqused_k=seqused_k, causal=True, block_table=bt, fa_version=3,
    num_splits=0, softcap=0.0, window_size=None, return_softmax_lse=True)
```
For the agent: golden = run the prebuilt op; candidate = run distilled `.cu` op (register under a
distinct namespace, e.g. `torch.ops._distill_fa3_C.fwd`) with identical inputs; compare. **Seed RNG and
reuse the SAME input tensors for both.** Generate per-config golden tensors once, cache to disk
(`.pt`), reuse across iterations to keep the loop fast.

---

## B. The exact set of CONFIGS vLLM exercises (perf + correctness must cover)

### B.1 Backend gating facts (from flash_attn.py / fa_utils.py)
- FA3 selected automatically on sm_90 when supported (`get_flash_attn_version`: major==9 → fa_version=3).
- `supports_head_size`: head_size % 8 == 0 AND head_size <= 256 (FA3). >256 only via FA4. So our `.cu`
  must handle head_dim multiples of 8 up to 256; the named targets {64,96,128,192,256} are the
  representative perf points.
- `supported_dtypes` (Q/out): float16, bfloat16. **fp8 is a KV-cache dtype**, not a Q dtype, gated by
  `supports_kv_cache_dtype`: "fp8"/"fp8_e4m3" allowed only on FA3 + sm_90. fp8 path uses
  q/k/v_descale tensors. (When kv-cache is fp8, key_cache/value_cache are `.view(fp8_dtype)`.)
- `alibi_slopes` is asserted None for FA3 (ALiBi falls back to FA2) — **we do NOT need ALiBi**.
- `s_aux` (attention sinks / "learnable sink") supported on FA3 — passed through (`self.sinks`).
- Sliding window: vLLM stores `self.sliding_window=(w-1,w-1)` (or `(w-1,0)` for left-only) and passes as
  `window_size`. `(-1,-1)` = full.
- softcap: `self.logits_soft_cap` passed as `softcap` (0.0 = off).
- `num_splits`: `attn_metadata.max_num_splits`. **0 = let FA3 heuristic choose** (the common non-CUDA-graph
  case, flash_attn.py:430). With full CUDA graphs, vLLM sets a fixed `max_num_splits`
  (`flash_attn_max_num_splits_for_cuda_graph`) for graph-capturability; batch-invariant mode forces
  `num_splits=1`.
- `pack_gqa`: passed `None` by vLLM (kernel auto-decides). GQA itself is implicit: nheads_q divisible by
  nheads_kv (e.g. 8/2, 16/2, 64/8).
- `scheduler_metadata`: present only with AOT schedule (`aot_schedule = fa_version==3`), used heavily for
  full-CUDA-graph + cascade/split paths. Plain eager path can pass None.
- Cascade attention (`use_cascade`, common_prefix_len>0): calls `fwd` TWICE (prefix + suffix) then merges
  via `merge_attn_states`. Asserts sliding_window==(-1,-1). **Out of scope for a first correct kernel**, but
  note vLLM may invoke it; the merge happens in vLLM, each `fwd` call is a normal attention. Disable by not
  exercising shared prefixes in the validation model.

### B.2 Representative shapes — PREFILL (varlen, causal, non-paged or paged-prefix)
Drive these from real model dims. Common vLLM models on H200:
- Llama-3-8B: nheads_q=32, nheads_kv=8 (GQA 4), head_dim=128.
- Llama-3-70B: nheads_q=64, nheads_kv=8, head_dim=128.
- Qwen2.5/Mistral: head_dim=128, GQA.
- Models with head_dim=64 (some small/MoE), 96 (phi-style), 256 (rare, e.g. some 2B).
Prefill representative set (head_dim ∈ {64,96,128,192,256}, dt ∈ {bf16,fp16}):
- batch=1, single long seq: seqlen_q = seqlen_k ∈ {512, 2048, 8192, 16384}, causal=True.
- varlen batch: lens=[100,50,...] mixed; max_seqlen_q = max len; cu_seqlens_q==cu_seqlens_k.
- chunked-prefill: seqlen_q < seqlen_k (e.g. q=512 new tokens, k=4096 context), causal bottom-right aligned,
  paged KV with block_table.

### B.3 Representative shapes — DECODE (1 query token/seq, paged KV)
- batch ∈ {1, 8, 32, 128, 256}; max_seqlen_q=1; cu_seqlens_q=arange(batch+1).
- context length seqused_k ∈ {128, 1024, 4096, 16384, 65536} (long-context decode stresses split-KV).
- head_dim ∈ {64,96,128,192,256}; GQA ratios 4 (32/8) and 8 (64/8), MQA (n/1).
- block_size = 16 (vLLM default min; multiple of 16). Also test 32, 64.
- split-KV: long context + small batch → FA3 heuristic picks num_splits>1. Test num_splits ∈ {0(auto),1,2,4,8,16}.
- fp8 KV cache variant (e4m3) with descales for at least head_dim=128.

### B.4 Feature cross-matrix to cover (each ✔ verified callable on the baseline live)
- head_dim {64✔,96✔,128✔,192✔,256✔}; dtype bf16✔/fp16✔/fp8-e4m3✔(kv+descale);
- causal✔ / non-causal; paged✔ / non-paged✔; split-KV✔ (num_splits=4 ran);
- GQA✔ (16/2 ran) / MQA; softcap✔ (30.0 ran); sliding-window✔ ([64,0] ran);
- s_aux (sinks) — present in API, lower priority; rotary — vLLM passes None (RoPE done outside attn), **skip**.

### B.5 PRIORITIZATION for "match or exceed latency on ALL configs vLLM uses"
The hot configs in real serving are: (1) **decode, head_dim=128, GQA, bf16, paged, causal, batch 1–256,
long context with auto split-KV** (dominates token latency); (2) **prefill/chunked-prefill head_dim=128 bf16
causal paged**. Get these two correct+fast first; head_dim {64,96,192,256}, fp16, fp8, softcap, sliding-window
are correctness-required but lower perf weight. Build the config list as a typed table the loop iterates.

---

## C. Smallest practical way to validate "100 greedy tokens unchanged"

Two complementary layers; use BOTH, cheap-first:

### C.1 PRIMARY (fast, per-iteration): pure-op numerical comparison
Compare distilled `.cu` op vs prebuilt op on the cached golden configs (section A.5/B). Metrics:
- **Output**: max abs err and a relative/cosine check. Reference tolerance from FA practice for bf16:
  the relevant criterion is "no worse than a high-precision (fp32 SDPA) reference by more than ~2x the
  baseline's own error". Concretely: compute `ref = fp32 math`, `e_base = (baseline-ref)`, `e_cand =
  (candidate-ref)`; require `max|e_cand| <= 2*max|e_base| + eps`. This avoids over-tight bit-exact demands
  while catching real bugs. Also a direct `allclose(candidate, baseline, atol=2e-2, rtol=1e-2)` for bf16 /
  `atol=1e-2` for fp16 as a quick gate.
- **softmax_lse**: compare too (catches scaling/normalization bugs invisible in out when one row dominates).
- This is the gate that runs every loop iteration; it does NOT require a model.

### C.2 The ACTUAL acceptance criterion: greedy argmax must match → use ARGMAX-EQUIVALENCE, not bit-match
"100 greedy tokens unchanged" reduces to: at every decode step the **argmax of the logits** is identical.
Attention output feeds through the rest of the network → logits. Tiny attention diffs almost never flip an
argmax unless logits are near-tied. So the right cheap proxy is:
- Build a synthetic logits-margin test: run a small real model once with the baseline op, capture per-step
  logit vectors, record the **margin** = top1−top2 logit gap. Then assert the candidate op keeps the
  attention-output error small enough that it cannot plausibly flip argmax. But this is indirect.

### C.3 DIRECT end-to-end validation (run periodically / before "accept"): monkeypatch into vLLM or HF
Cheapest robust method: a tiny model + greedy decode, run twice (baseline op vs distilled op), assert the
generated token-id sequences are identical for 100 tokens.
- **Option A (recommended): vLLM with monkeypatched op.** Use a small model (e.g. a 1B–1.5B Llama/Qwen,
  head_dim=128 GQA — exercises the hot path). `LLM(model=..., enforce_eager=True)` to avoid CUDA-graph
  capture complications initially. Monkeypatch by replacing `torch.ops._vllm_fa3_C.fwd` binding that
  `flash_attn_interface.py` calls — simplest is to register the distilled kernel as `_vllm_fa3_C.fwd` (same
  namespace/name) BEFORE importing vllm, OR patch `vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_func`
  to route fa_version=3 to our op. Then `llm.generate(prompts, SamplingParams(temperature=0, max_tokens=100))`.
  Compare `output.token_ids` to the baseline run's token_ids — must be element-wise equal.
- **Option B: HF transformers** with attention swapped — more wiring; prefer A since the target is vLLM and
  it guarantees the exact same calling convention (paged KV, varlen, scheduler metadata) is exercised.
- Run end-to-end validation on a few prompts spanning prefill lengths (short + long) so both prefill and
  decode code paths are hit; include one prompt that triggers long-context decode (>4k) to stress split-KV.
- Determinism caveats: set temperature=0 (greedy/argmax), fixed seed, `enforce_eager=True` first; only test
  with CUDA graphs after eager passes (graphs add the AOT scheduler_metadata path).

### C.4 Loop policy: gate on C.1 every iteration; promote to C.3 only when C.1 passes for all configs.
A candidate that passes C.1 on all configs but fails C.3 indicates an argmax flip from accumulated error or
a path not covered by C.1 (e.g. CUDA-graph/scheduler_metadata, cascade) → add that config to C.1's table.

---

## D. Robust CUDA-event latency measurement recipe (H200, this box)

### D.1 The killer caveat on THIS machine: we canNOT lock clocks (no sudo; lock denied).
GPU idles at 345 MHz and boosts to 1980 MHz under load. **Consequence:** absolute ms numbers are not
reproducible run-to-run, and a cold kernel measured during ramp-up looks slow. Mitigations:
- **Always warm up enough to reach steady boost clocks** before timing (spin the kernel ~200–1000 iters or
  ~0.5–1s of GPU work). Verify by sampling `nvidia-smi --query-gpu=clocks.sm` during the timed region; if it
  is not near 1980 MHz, extend warmup.
- **Measure baseline and candidate back-to-back in the SAME process, interleaved**, and report the RATIO
  (candidate/baseline) not absolute ms. The ratio is robust to DVFS because both see the same clock state.
  Acceptance = ratio <= 1.0 (within noise) on every config.
- Run on a **dedicated GPU** (perf=1 kernel per GPU per the design); pin one config to one GPU; no other load.
- Report median + p10/p90 across repeats; reject if variance is high (re-run after longer warmup).

### D.2 The CUDA-event recipe (per timed sample)
```python
flush = torch.empty(64*1024*1024, dtype=torch.int8, device='cuda')  # >60MiB L2 → full flush
start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
# WARMUP: run op N_warm times (>= a few hundred; until clocks boost) with torch.cuda.synchronize()
times = []
for _ in range(N_rep):           # N_rep ~ 50–200
    flush.zero_()                # L2 flush: evict KV/Q/out from L2 so each timed run is cold-in-L2
    torch.cuda.synchronize()
    start.record()
    op(...)                      # single launch (or fixed small batch of launches if too short)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))   # ms
median_ms = statistics.median(times)
```
- **L2 flush**: H200 L2 = 60 MiB → allocate >= 64 MiB scratch and write it (`zero_()`) between samples so
  the attention inputs aren't served from L2 (otherwise tiny decode kernels read entirely from L2 and you
  measure an unrealistic best case). For attention, the relevant question is parity with baseline; flushing
  identically for both keeps the comparison fair. Provide BOTH "L2-flushed" and "L2-resident" numbers; the
  L2-resident (no flush) better mirrors steady decode where KV is hot, the flushed mirrors cold context.
- **Single-launch timing for very short kernels**: a decode kernel may be ~5–20 µs; event overhead (~few µs)
  and DVFS noise dominate. For sub-20µs kernels, time a fixed inner loop of K launches (e.g. K=50) WITHOUT
  inter-launch sync, divide by K — but beware the FA3 op writes `out` in place; reuse distinct out buffers or
  accept identical writes (fine for timing). Cross-check single-launch vs looped to detect launch-overhead skew.
- **Repeats**: warmup >= ~300 iters or until clocks steady; N_rep = 50–200 timed; take median; also p90.
- **Do NOT use python `time.perf_counter` around async launches.** Use CUDA events with synchronize, OR
  `torch.cuda.synchronize()` bracket. Events measure GPU time excluding python overhead — preferred.
- **CUDA graphs option for stable measurement**: capture the single op into a CUDA graph and replay; this
  removes launch/python jitter and is closer to how vLLM runs decode (it uses full CUDA graphs). Recommended
  for the final perf gate on decode configs.
- **`torch.cuda.synchronize()` once before starting**; clear pending work. Set `torch.backends.cudnn` etc.
  irrelevant here. Disable TF32 concerns — attention is fp16/bf16.
- NCU profiling (component 2.1) runs separately and serializes/locks clocks itself via `--clock-control`;
  but `ncu` clock control may also require perms — fall back to `--clock-control none` + warmup, and rely on
  NCU's per-kernel metrics (which are largely clock-independent: occupancy, memory throughput %, instruction
  mix) rather than wall-clock from NCU.

### D.3 What to actually report per config (for the evolution plot / accept decision)
- candidate_median_ms, baseline_median_ms, ratio (= candidate/baseline), p90 ratio.
- Accept perf iff ratio <= 1.0 + noise_band (e.g. 1.02) on ALL configs; the design says "match or exceed".
- Tag each config row with the feature matrix (head_dim, dtype, causal, paged, splits, gqa, softcap, window).

---

## E. Concrete reusable artifacts for the agent

- Op namespace strategy: register distilled kernel under a SEPARATE namespace (e.g. `_distill_fa3_C::fwd`)
  with the SAME 38-arg schema + same `Tensor[]` (4-elem) return + same arg0/out aliasing. A thin adapter can
  expose it as `flash_attn_varlen_func(..., fa_version=3)` for the e2e test by monkeypatch.
- Cache golden tensors per config to `.pt` once; the verify step loads them (fast loop).
- The `mha_fwd` host fn in the `.cu` may use torch ONLY for: tensor unpack (sizes/strides/data_ptr/dtype),
  output allocation (`torch::empty_like`/`empty`), and `TORCH_LIBRARY` registration. No torch inside
  `__global__`. The schema above is exactly what to register.
- Standalone golden scripts already validated live at `/tmp/golden_test.py` and `/tmp/golden_test2.py`
  (decode paged, prefill varlen, scheduler_metadata, all head dims, fp16, fp8, softcap, sliding window,
  split-KV, GQA, raw 4-tensor return arity). Reuse these as the seed of the config table.

## F. Gotchas / traps
- LSE is `(nheads, total_q)` — transposed vs out `(total_q, nheads, hd)`. Easy to get wrong.
- `fwd` returns a LIST of 4; wrapper discards last two. fp8 path needs `(batch,num_kv_heads)` fp32 descales.
- `num_splits=0` means "auto heuristic" not "no split". For deterministic compare set explicit num_splits.
- scheduler_metadata is optional for correctness but is on the CUDA-graph/decode hot path in real vLLM; a
  None-schedule kernel can be correct yet exercise a different code path than production. Validate the
  CUDA-graph e2e path before declaring perf parity.
- Cascade attention calls fwd twice + merges outside; not a kernel feature but appears with shared prefixes.
- Clocks can't be locked here → never trust absolute ms; always ratio vs baseline, same process, warmed up.
- Block size multiple of 16 (paged). head_dim multiple of 8, <=256 for FA3.
- ALiBi and rotary are NOT in scope (vLLM routes ALiBi to FA2; RoPE applied before attention, passes None).
