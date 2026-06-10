// fa3agent seed kernel: a minimal, CORRECT, fully self-contained FlashAttention-3-shaped
// forward op for Hopper (sm_90a).  NO external libraries (no CUTLASS/CuTe/cuBLAS/...).
// torch appears ONLY in the host mha_fwd + TORCH_LIBRARY registration (the allowed boundary).
//
// This is the loop's bootstrap candidate x0 and the component test fixture.  It favors
// CORRECTNESS over speed: one warp computes one (query, head) output row via an online
// softmax with fp32 accumulation, streaming over the KV sequence.  It ignores num_splits
// and scheduler_metadata (both are perf-only / advisory -> single-pass is always correct).
//
// Features: bf16/fp16/fp8-e4m3 inputs, paged + non-paged KV, GQA/MQA, causal, sliding
// window, softcap, varlen (cu_seqlens_q + seqused_k or cu_seqlens_k).  Output dtype = input
// dtype, except fp8 -> bf16.  softmax_lse is fp32, shape (num_heads_q, total_q).
//
// Build with -DFA3_LIB_NAME=<namespace> to pick the torch op namespace
// (default _vllm_fa3_C for the real vLLM swap; _distill_fa3_C for in-process A/B).

#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <vector>
#include <cstdint>

#ifndef FA3_LIB_NAME
#define FA3_LIB_NAME _vllm_fa3_C
#endif

#define MAX_HD 256

// ---- scalar load/convert helpers -----------------------------------------------------
template <typename T> __device__ __forceinline__ float to_float(T x);
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float to_float<__nv_fp8_e4m3>(__nv_fp8_e4m3 x) { return float(x); }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Params: all strides are IN ELEMENTS.
template <typename TIn, typename TOut>
struct Params {
  const TIn* q; const TIn* k; const TIn* v;
  TOut* o; float* lse;
  const int* cu_q;          // (b+1) or null
  const int* cu_k;          // (b+1) or null (non-paged)
  const int* seqused_k;     // (b) or null (paged)
  const int* page_table;    // (b, max_blocks) or null
  const float* q_descale; const float* k_descale; const float* v_descale;  // (b, hkv) or null
  long q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head;
  int b, hq, hkv, hd, hdv, total_q, block_size, page_stride;
  float scale, softcap;
  int win_l, win_r; bool causal;
};

template <typename TIn, typename TOut>
__global__ void attn_kernel(Params<TIn, TOut> p) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const long total_pairs = (long)p.total_q * p.hq;
  if (warp >= total_pairs) return;
  const int q_idx = warp / p.hq;
  const int head  = warp % p.hq;
  const int hk    = head / (p.hq / p.hkv);

  // locate batch b s.t. cu_q[b] <= q_idx < cu_q[b+1]
  int b = 0;
  while (b + 1 <= p.b && p.cu_q[b + 1] <= q_idx) ++b;
  const int q_in_seq = q_idx - p.cu_q[b];
  const int sq = p.cu_q[b + 1] - p.cu_q[b];
  const int klen = p.seqused_k ? p.seqused_k[b] : (p.cu_k[b + 1] - p.cu_k[b]);
  const int q_abs = klen - sq + q_in_seq;   // causal alignment: last query <-> last key

  float qd = p.q_descale ? p.q_descale[b * p.hkv + hk] : 1.f;
  float kd = p.k_descale ? p.k_descale[b * p.hkv + hk] : 1.f;
  float vd = p.v_descale ? p.v_descale[b * p.hkv + hk] : 1.f;
  const float qk_scale = p.scale * qd * kd;

  // load q row into per-lane registers
  const int per = (p.hd + 31) / 32;
  float qreg[(MAX_HD + 31) / 32];
  const TIn* qp = p.q + (long)q_idx * p.q_row + (long)head * p.q_head;
  #pragma unroll
  for (int e = 0; e < per; ++e) { int d = lane + e * 32; qreg[e] = d < p.hd ? to_float<TIn>(qp[d]) : 0.f; }

  const int perv = (p.hdv + 31) / 32;
  float acc[(MAX_HD + 31) / 32];
  #pragma unroll
  for (int e = 0; e < perv; ++e) acc[e] = 0.f;
  float m = -INFINITY, l = 0.f;

  for (int j = 0; j < klen; ++j) {
    bool masked = false;
    if (p.causal && j > q_abs) masked = true;
    if (p.win_l >= 0 && j < q_abs - p.win_l) masked = true;
    if (p.win_r >= 0 && j > q_abs + p.win_r) masked = true;
    if (masked) continue;

    long krow;
    if (p.page_table) {
      int blk = p.page_table[(long)b * p.page_stride + j / p.block_size];
      krow = ((long)blk * p.block_size + (j % p.block_size));
    } else {
      krow = (long)p.cu_k[b] + j;
    }
    const TIn* kp = p.k + krow * p.k_row + (long)hk * p.k_head;
    float dot = 0.f;
    #pragma unroll
    for (int e = 0; e < per; ++e) { int d = lane + e * 32; if (d < p.hd) dot += qreg[e] * to_float<TIn>(kp[d]); }
    float s = warp_reduce_sum(dot) * qk_scale;
    if (p.softcap > 0.f) s = p.softcap * tanhf(s / p.softcap);

    float m_new = fmaxf(m, s);
    float corr = __expf(m - m_new);
    float pj = __expf(s - m_new);
    l = l * corr + pj;
    const TIn* vp = p.v + krow * p.v_row + (long)hk * p.v_head;
    #pragma unroll
    for (int e = 0; e < perv; ++e) { int d = lane + e * 32; float vv = d < p.hdv ? to_float<TIn>(vp[d]) * vd : 0.f; acc[e] = acc[e] * corr + pj * vv; }
    m = m_new;
  }

  TOut* op = p.o + (long)q_idx * p.o_row + (long)head * p.o_head;
  float inv = l > 0.f ? 1.f / l : 0.f;
  #pragma unroll
  for (int e = 0; e < perv; ++e) { int d = lane + e * 32; if (d < p.hdv) op[d] = from_float<TOut>(acc[e] * inv); }
  if (lane == 0) p.lse[(long)head * p.total_q + q_idx] = (l > 0.f) ? (m + logf(l)) : (-INFINITY);
}

// ---- host -----------------------------------------------------------------------------
static inline const void* dptr(const at::Tensor& t) { return t.data_ptr(); }

template <typename TIn, typename TOut>
static void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                   at::Tensor& out, at::Tensor& lse,
                   const at::Tensor& cu_q, const std::optional<at::Tensor>& cu_k,
                   const std::optional<at::Tensor>& seqused_k, const std::optional<at::Tensor>& page_table,
                   const std::optional<at::Tensor>& qd, const std::optional<at::Tensor>& kd, const std::optional<at::Tensor>& vd,
                   int b, int hq, int hkv, int hd, int hdv, int total_q,
                   double scale, double softcap, int win_l, int win_r, bool causal, cudaStream_t stream) {
  Params<TIn, TOut> p{};
  p.q = (const TIn*)dptr(q); p.k = (const TIn*)dptr(k); p.v = (const TIn*)dptr(v);
  p.o = (TOut*)out.data_ptr(); p.lse = (float*)lse.data_ptr();
  p.cu_q = (const int*)cu_q.data_ptr();
  p.cu_k = cu_k ? (const int*)cu_k->data_ptr() : nullptr;
  p.seqused_k = seqused_k ? (const int*)seqused_k->data_ptr() : nullptr;
  p.page_table = page_table ? (const int*)page_table->data_ptr() : nullptr;
  p.page_stride = page_table ? (int)page_table->stride(0) : 0;
  p.block_size = (page_table) ? (int)k.size(1) : 0;
  p.q_descale = qd ? (const float*)qd->data_ptr() : nullptr;
  p.k_descale = kd ? (const float*)kd->data_ptr() : nullptr;
  p.v_descale = vd ? (const float*)vd->data_ptr() : nullptr;
  p.q_row = q.stride(0); p.q_head = q.stride(1);
  // k/v paged (nblocks, block, hkv, hd): index krow over flattened (blk*block_size+off),
  // so row stride = stride(1), head stride = stride(2). Non-paged: row stride(0), head stride(1).
  if (page_table) { p.k_row = k.stride(1); p.k_head = k.stride(2); p.v_row = v.stride(1); p.v_head = v.stride(2); }
  else            { p.k_row = k.stride(0); p.k_head = k.stride(1); p.v_row = v.stride(0); p.v_head = v.stride(1); }
  p.o_row = out.stride(0); p.o_head = out.stride(1);
  p.b = b; p.hq = hq; p.hkv = hkv; p.hd = hd; p.hdv = hdv; p.total_q = total_q;
  p.scale = (float)scale; p.softcap = (float)softcap; p.win_l = win_l; p.win_r = win_r; p.causal = causal;
  long pairs = (long)total_q * hq;
  int threads = 128; long warps_per_block = threads / 32;
  long blocks = (pairs + warps_per_block - 1) / warps_per_block;
  attn_kernel<TIn, TOut><<<(unsigned)blocks, threads, 0, stream>>>(p);
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor q, at::Tensor k, at::Tensor v,
    std::optional<at::Tensor> k_new, std::optional<at::Tensor> v_new, std::optional<at::Tensor> q_v,
    std::optional<at::Tensor> out_,
    std::optional<at::Tensor> cu_seqlens_q, std::optional<at::Tensor> cu_seqlens_k, std::optional<at::Tensor> cu_seqlens_k_new,
    std::optional<at::Tensor> seqused_q, std::optional<at::Tensor> seqused_k,
    std::optional<int64_t> max_seqlen_q, std::optional<int64_t> max_seqlen_k,
    std::optional<at::Tensor> page_table, std::optional<at::Tensor> kv_batch_idx, std::optional<at::Tensor> leftpad_k,
    std::optional<at::Tensor> rotary_cos, std::optional<at::Tensor> rotary_sin, std::optional<at::Tensor> seqlens_rotary,
    std::optional<at::Tensor> q_descale, std::optional<at::Tensor> k_descale, std::optional<at::Tensor> v_descale,
    double softmax_scale, bool is_causal, int64_t window_size_left, int64_t window_size_right,
    double softcap, bool is_rotary_interleaved,
    std::optional<at::Tensor> scheduler_metadata, int64_t num_splits, std::optional<bool> pack_gqa, int64_t sm_margin,
    std::optional<at::Tensor> s_aux, int64_t cp_world_size, int64_t cp_rank, std::optional<at::Tensor> cp_tot_seqused_k) {
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  int total_q = (int)q.size(0);
  int hq = (int)q.size(1);
  int hd = (int)q.size(2);
  bool paged = page_table.has_value();
  int hkv = paged ? (int)k.size(2) : (int)k.size(1);
  int hdv = paged ? (int)v.size(3) : (int)v.size(2);
  int b = (int)cu_seqlens_q->size(0) - 1;

  bool is_fp8 = q.scalar_type() == at::kFloat8_e4m3fn;
  auto out_dtype = is_fp8 ? at::kBFloat16 : q.scalar_type();
  at::Tensor out = out_.has_value() ? *out_
                 : at::empty({total_q, hq, hdv}, q.options().dtype(out_dtype));
  at::Tensor lse = at::empty({hq, total_q}, q.options().dtype(at::kFloat));

  int win_l = (int)window_size_left, win_r = (int)window_size_right;
  bool causal = is_causal;

  if (total_q > 0) {
    if (q.scalar_type() == at::kBFloat16)
      launch<__nv_bfloat16, __nv_bfloat16>(q, k, v, out, lse, *cu_seqlens_q, cu_seqlens_k, seqused_k, page_table,
        q_descale, k_descale, v_descale, b, hq, hkv, hd, hdv, total_q, softmax_scale, softcap, win_l, win_r, causal, stream);
    else if (q.scalar_type() == at::kHalf)
      launch<__half, __half>(q, k, v, out, lse, *cu_seqlens_q, cu_seqlens_k, seqused_k, page_table,
        q_descale, k_descale, v_descale, b, hq, hkv, hd, hdv, total_q, softmax_scale, softcap, win_l, win_r, causal, stream);
    else if (is_fp8)
      launch<__nv_fp8_e4m3, __nv_bfloat16>(q, k, v, out, lse, *cu_seqlens_q, cu_seqlens_k, seqused_k, page_table,
        q_descale, k_descale, v_descale, b, hq, hkv, hd, hdv, total_q, softmax_scale, softcap, win_l, win_r, causal, stream);
    else
      TORCH_CHECK(false, "fa3agent seed: unsupported dtype");
  }
  // return [out, softmax_lse, out_accum(empty), softmax_lse_accum(empty)]
  at::Tensor empty = at::empty({0}, q.options().dtype(at::kFloat));
  return {out, lse, empty, empty};
}

at::Tensor mha_fwd_get_scheduler_metadata(
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k, int64_t num_heads, int64_t num_heads_k,
    int64_t headdim, int64_t headdim_v, at::ScalarType qkv_dtype, at::Tensor seqused_k,
    std::optional<at::Tensor> cu_seqlens_q, std::optional<at::Tensor> cu_seqlens_k, std::optional<at::Tensor> cu_seqlens_k_new,
    std::optional<at::Tensor> seqused_q, std::optional<at::Tensor> leftpad_k, std::optional<int64_t> page_size,
    int64_t max_seqlen_k_new, bool is_causal, int64_t window_size_left, int64_t window_size_right, bool has_softcap,
    int64_t num_splits, std::optional<bool> pack_gqa, int64_t sm_margin) {
  // Our fwd recomputes everything internally; return a benign tensor.
  return at::zeros({1}, seqused_k.options().dtype(at::kInt));
}

TORCH_LIBRARY(FA3_LIB_NAME, m) {
  m.def("fwd(Tensor! q, Tensor k, Tensor v, Tensor? k_new, Tensor? v_new, Tensor? q_v, "
        "Tensor!? out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new, "
        "Tensor? seqused_q, Tensor? seqused_k, int? max_seqlen_q, int? max_seqlen_k, "
        "Tensor? page_table, Tensor? kv_batch_idx, Tensor? leftpad_k, Tensor? rotary_cos, "
        "Tensor? rotary_sin, Tensor? seqlens_rotary, Tensor? q_descale, Tensor? k_descale, "
        "Tensor? v_descale, float softmax_scale, bool is_causal, int window_size_left, "
        "int window_size_right, float softcap, bool is_rotary_interleaved, "
        "Tensor? scheduler_metadata, int num_splits, bool? pack_gqa, int sm_margin, "
        "Tensor? s_aux, int cp_world_size, int cp_rank, Tensor? cp_tot_seqused_k) -> Tensor[]");
  m.impl("fwd", torch::kCUDA, &mha_fwd);
  m.def("get_scheduler_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, "
        "int num_heads, int num_heads_k, int headdim, int headdim_v, ScalarType qkv_dtype, "
        "Tensor seqused_k, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new, "
        "Tensor? seqused_q, Tensor? leftpad_k, int? page_size, int max_seqlen_k_new, "
        "bool is_causal, int window_size_left, int window_size_right, bool has_softcap, "
        "int num_splits, bool? pack_gqa, int sm_margin) -> Tensor");
  m.impl("get_scheduler_metadata", torch::kCUDA, &mha_fwd_get_scheduler_metadata);
}
