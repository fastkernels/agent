# fa3agent — autonomous distiller for vLLM FlashAttention-3 → one self-contained Hopper `.cu`

An autonomous optimization agent that distills vLLM's fork of FlashAttention-3 (a forward-only,
warp-specialized SM90 kernel that is fundamentally CUTLASS/CuTe: TMA + WGMMA + mbarrier) into a
**single self-contained CUDA file** with **no external libraries**, targeting Hopper (H200, sm_90a).

The loop: **generate → verify correctness FIRST → profile → optimize → learn**, driven by
**Claude Code headless (Opus)** as the codegen brain, until either a step budget is hit or BOTH
(a) the candidate is ≥ 1.0× the baseline on **every** config and (b) greedy(argmax) decoding of up
to 100 tokens is **unchanged** vs the prebuilt baseline.

## Quick start
```bash
cd /data-fast/agent
python -m fa3agent run  --configs dev  --max-steps 5          # bounded run on the dev config subset
python -m fa3agent run  --configs full                        # indefinite run (no budget) on the full matrix
python -m fa3agent run  --resume                              # resume the latest run from its last step
python -m fa3agent list                                       # list checkpointed runs
python -m fa3agent validate                                   # run every component smoke test
```
Flags: `--model opus` (headless brain, default), `--run-id NAME`, `--decode-tokens 100`,
`--no-decode-equiv` (skip the slow greedy gate), `--no-profile` (skip NCU).
Checkpoints (kernels, lineage, knowledge, plots, logs) go to
`/code/users/goliaro/cuda-agent-checkpoints/<run-id>/`.

## How it works
A deterministic Python **orchestrator** owns control flow (loop, step budget, GPU scheduling,
checkpoint/resume, the correctness gate). Each step it invokes Claude headless
(`claude -p … --dangerously-skip-permissions --model opus [--resume <sid>]`); the brain WRITES the
candidate `.cu` and the **driver** runs evaluation and feeds back DISTILLED results (syfi/AVO model).
Every component returns only concise, bounded output — never raw dumps (context discipline).

```
config.py     single source of truth: paths, EXACT op schemas, config matrix, tolerances,
              anti-cheat denylists, NCU metric subset, perf knobs
common.py     JSON-serializable dataclasses (BuildResult/AntiCheatResult/ConfigResult/EvalResult/
              ProfileFeedback/Candidate/StepRecord)
shapes.py     AUTHORITATIVE input builder: make_inputs / fwd_args (the 37-arg positional order in one
              place) / call_fwd(namespace,…) / ref_fp32 (pure-torch fp32 oracle). Drives baseline AND
              candidate on identical tensors.
interfaces.py the component contract
seed_kernel.cu the bootstrap candidate x0 + test fixture: a correct, self-contained, no-CUTLASS
              attention .cu (one warp per (query,head), online softmax, fp32 accumulate)
orchestrator.py the loop  •  llm.py Claude headless wrapper  •  checkpoint.py per-step state + resume
gpu_pool.py   lease scheduler (perf/NCU = 1 kernel/GPU; correctness may share)  •  prompts.py
sandbox/      build.py (toolkit-only nvcc → .so, schema check) + anticheat.py (2.5)
verify/       golden.py (cached baseline+fp32-ref) + correctness.py (verify FIRST) +
              decode_equiv.py (100-token greedy gate via monkeypatched vLLM) + sanitizer.py (2.2)
perf/         measure.py (CUDA-event interleaved ratio, §3) + ncu_profile.py/ncu_utils.py (2.1) +
              _profile_main.py (torch-free NCU subprocess runner)
sass/         analyze.py (2.3 instr-mix diff) + cutlass_oracle.py (2.6 CuTeDSL/nvcc PTX oracle)
docs/         retrieve.py (2.4)   reflib/ hints.py (2.7)   plot/ evolution.py (AVO-style)
knowledge/    store.py + query/get_page/grep + seed/*.md (typed in-loop distillation: ks/pf/alg)
verify/insitu.py  faithful in-situ A/B gate: compares baseline vs candidate on LIVE vLLM tensors
              during one short forward (no full decode) -- the cost-tier between synthetic and decode
```

## Key design decisions
- **Correctness before performance, verify FIRST, COST-TIERED.** Three gates of increasing cost/fidelity,
  each doing distinct work (empirically calibrated — see "Calibration findings"):
  1. *Synthetic element-wise* (every step, instant): out+lse vs cached golden `_vllm_fa3_C`, fp32-ref
     2×-error + atol/rtol + a mean-bias gate. A **loose necessary pre-filter** (rejects NaN/shape/gross
     drift) — NOT a decode proof.
  2. *In-situ A/B* (`verify/insitu.py`, at promotion): baseline vs candidate on the **live** vLLM
     tensors during one short forward. Faithful to the live paged-KV / broadcast-descale layout that a
     clone/offline-replay silently masks; far cheaper than a full decode.
  3. *Full greedy decode-equality* (`verify/decode_equiv.py`, at the perf target): 100-token argmax
     stream identical to baseline — the final acceptance gate for the residual near-tie risk.
  Any correctness failure ⇒ all perf scores 0.
- **Perf is a RATIO, optimize the MINIMUM across configs.** GPU clocks cannot be locked here (no
  sudo), so baseline and candidate are timed interleaved in the same warmed process; success =
  candidate ≤ baseline latency on every config.
- **Anti-cheat is mechanical (2.5).** Toolkit-only `-I` (no cutlass), source regex scan, and
  `nm -D`/`ldd` scan of the built `.so` for forbidden symbols/links → reject.
- **Hand-written kernel, CUTLASS only as an out-of-band oracle.** The kernel re-derives raw PTX
  (wgmma / cp.async.bulk.tensor / mbarrier); CUTLASS/CuTeDSL is used only to dump target PTX/SASS to
  diff against (2.6) and is never `#include`d in the output `.cu`.
- **Unique op namespace per step** (`_distill_fa3_C_sNNN`) so successive `.so` loads in the
  long-running driver don't double-register; the FINAL deliverable builds under `_vllm_fa3_C`.
- **NCU runs out-of-process**: `ncu_report` and torch segfault if co-imported, so profiling goes
  through `perf/_profile_main.py` (torch-free).

## Validation status (see HANDOFF notes)
All 11 components pass isolated smoke tests; the full loop, checkpoint/resume, evolution plot, and
multi-GPU scheduling are validated end-to-end.

### Calibration findings (why the gates are tiered, not a single threshold)
An empirical study (baseline self-variation + Llama-3.1-8B decode + live-input capture) established:
- The baseline FA3 is **run-to-run bitwise-deterministic** except for split-K reduction order; its
  whole self-envelope is bf16 out max-abs ≤3.9e-3 / mean ≤3.7e-5. But a *candidate is a different
  algorithm* and legitimately differs by more even when correct — so a synthetic threshold can't be
  both safe and admit-all-correct; it stays a loose pre-filter (`config.ALLCLOSE_TOL` tuned to this).
- **A static threshold cannot certify greedy-decode equality**: the minimum top1–top2 logit margin
  over ~300 decoded tokens is **0.0** (true ties), and divergence vs perturbation is non-monotonic —
  so the full decode gate is irreducible.
- **The decisive failure surface does not survive `clone()`/serialization**, so offline-replay is
  unfaithful; the in-situ A/B gate (live tensors) is required.

### seed_kernel.cu: a real bug the study found (and fixed)
The seed initially decoded 0/100 tokens on the live path. Root cause (pinned via live-input capture):
vLLM passes `q/k/v_descale` as a **broadcast (b,hkv) tensor, stride [0,0], one physical element**,
*even for bf16/fp16*; the seed read it densely → out-of-bounds → garbage scales (output norm 30× too
large). Fix: apply descale **only for fp8** and read it **stride-aware** (broadcast-safe). Post-fix
the in-situ gate is clean (128 live calls, max-diff 7.8e-3, argmax-match all). The residual token-10
decode divergence is the **near-tie fragility** above (a scalar-fp32 bootstrap can't match baseline's
wgmma reduction at a 0.125-nat tie), which the full decode gate correctly flags — exactly why it stays.
The seed is a bootstrap fixture, not the converged kernel.

