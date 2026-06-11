# Threshold calibration study (can a static allclose replace the vLLM decode check?)

Empirical study on this box (H200, baseline `_vllm_fa3_C`, Llama-3.1-8B-Instruct). Verdict:
a static element-wise threshold is **necessary but not sufficient** — keep the cost-tiered design
(synthetic pre-filter → in-situ A/B on live tensors → full greedy decode-equality).

## A. Baseline self-nondeterminism envelope (no model)
- Baseline FA3 is run-to-run **bitwise-deterministic** at fixed num_splits (0 delta over repeats).
- Only legitimate variation = split-K reduction order across num_splits {1,2,4} (what vLLM auto picks).
- Envelope: bf16 out max-abs ≤ **3.9e-3** / mean-abs ≤ **3.7e-5**; fp16 ≤ 1.2e-4; lse(fp32) ≤ 2e-6.
- Pure rtol useless (372× on near-zero out) → atol-dominated.
- IMPORTANT: this is baseline-vs-ITSELF. A candidate is a *different algorithm* and legitimately
  differs by more even when correct (the known-correct seed has bf16 bias ~1.4e-4 vs self 3.7e-5).
  → synthetic gate must be a loose PRE-FILTER, not tightened to this floor (else false-negatives).

## B. Logit margins + decode error budget (Llama-3.1-8B)
- Min top1–top2 logit margin over ~300 decoded tokens = **0.0** (true ties at short pos 57/82/93).
- Divergence vs out-perturbation is **non-monotonic** (ε=1e-3 flips for some seeds; 3e-3,1e-2 clean;
  3e-2 flips) → no calibratable "budget".
- ⇒ no element-wise threshold can certify token equality; the full decode gate is irreducible.

## C. Live-input capture + seed diagnosis (the decisive finding)
- Captured the REAL arg tuples vLLM passes. `q/k/v_descale` = broadcast (b,hkv), stride **[0,0]**,
  one physical element, value 1.0 — present **even for bf16/fp16**.
- The seed read descale densely → **out-of-bounds** → garbage scales → live out norm 40 vs 1.33,
  0/100 tokens. A `clone()` densifies the broadcast → **masks the bug** (so offline-replay/captured-
  clone gates are UNFAITHFUL; the in-situ live A/B is required).
- Fix (seed_kernel.cu): apply descale only for fp8 + read it stride-aware. After fix: in-situ 128
  live calls max-diff **7.8e-3**, argmax-match all; decode 10/32 (residual = near-tie at token 10).

## Design consequences (implemented)
- `config.ALLCLOSE_TOL` tuned as a loose pre-filter (bf16 out atol 8e-3 / rtol 1e-2) + `MEAN_ABS_TOL`
  bias gate (bf16 5e-4) — admits known-correct candidates, rejects gross/systematic drift.
- `verify/insitu.py`: faithful in-situ A/B on live tensors (one short forward), run at promotion.
- `verify/decode_equiv.py`: full 100-token greedy equality, run only at the perf target.
