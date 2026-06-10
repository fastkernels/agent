---
id: alg-0004
type: alg
title: Online softmax (log2 trick) - the FA3 numerics for greedy-decode parity
tags: [softmax, online-softmax, exp2, log2e, rescale, lse, numerics, allreduce]
confidence: source-reported
source: FA3 softmax.h + ThunderKittens/flashinfer, .research/fa3-kernel-algo.md + ref-libs.md
---
Incremental softmax over K-tiles, never materialize N x N. Use exp2 (ex2.approx.f32),
NOT expf: fold log2(e) into the scale so it matches the FA3 baseline closely enough
for greedy argmax stability.

scale = softmax_scale * log2(e)   (e.g. hd128: 1/sqrt(128) * 1.44269504 = 0.1275)
Per KV block, after S = QK^T (f32 regs), scoremod + mask applied:
  max_prev = row_max
  row_max  = max(row_max, rowmax(S))          # running max, quad_allreduce<4>
  S        = exp2(S*scale - row_max*scale)     # ffma fuses; if max==-inf treat as 0
  corr     = exp2((max_prev - row_max)*scale)  # rescale factor
  row_sum  = row_sum*corr + rowsum(S)          # cross-thread sum DEFERRED to finalize
  O        = O * corr                          # rescale running output accumulator
  P        = (bf16) S;  O += P @ V             # wgmma PV (RS, P in regs)
finalize: quad_allreduce(row_sum); inv = 1/sum (0 if sum 0/NaN); O *= inv;
  LSE = row_max*scale*ln2 + ln(sum).

Key details:
- 4 threads share a GMMA output row -> Allreduce<4> = two __shfl_xor_sync (xor 2, xor 1).
- row_sum cross-thread reduce is deferred to finalize; only thread-local during loop.
- Softcap (Gemma): S = tanh(S * softmax_scale/softcap) with scale folded as
  softcap*log2(e); fast_tanh = tanh.approx.f32. Applied BEFORE masking.
- Softmax sinks (S_aux): add exp2(ln2*sink - max_scaled) to the denominator.
- FP8: subtract Max_offset=8 before exp2 (use FP8 range), divide sum by 256 at finalize.

See [[alg-0003]] for kernel structure, [[alg-0005]] technique #2 single-pass softmax,
[[ks-0001]] for the PV-RS wgmma form.
