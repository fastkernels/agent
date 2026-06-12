"""Smoke test for the decode-equivalence promotion gate.

Exercises `check_decode_equiv` end-to-end and reports pass / skip / fail.

The gate has two outcomes that both prove it WORKS:
  * pass  -> the candidate produced a greedy token-id stream identical to baseline
             over a real vLLM decode (the gate ran AND the kernel is equivalent), OR
             the gate is wired correctly (self-validation: routing the patch to the
             baseline op itself yields an identical stream).
  * skip  -> the environment could not run a vLLM engine; the monkeypatch MECHANISM
             was validated instead (swap point routes + matches baseline argmax).
  * fail  -> the gate ran but is mis-wired (the candidate never routed, the
             comparison is broken, or the self-validation diverged).

We test against the prebuilt SEED `.so` (namespace `_distill_fa3_C`).  NOTE: the
seed kernel is only a starting fixture and is NOT expected to be byte-for-token
identical to baseline on the live vLLM paged-KV path; the seed verdict is reported
for information.  The smoke's pass/fail hinges on the SELF-VALIDATION (routing the
patch to the baseline `_vllm_fa3_C` op must reproduce baseline exactly), which is
what actually proves the harness is correct.

Run from /data-fast/agent:
    CUDA_VISIBLE_DEVICES=2 python -m fa3agent.tests.test_decode_equiv
"""
from __future__ import annotations

import os
import sys

from fa3agent import config as C

# Pin the smoke to GPU 2 (spreads load across the 4 H200s) unless already set.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
if C.HF_HOME:
    os.environ.setdefault("HF_HOME", C.HF_HOME)

from fa3agent.verify.decode_equiv import check_decode_equiv

SEED_SO = os.path.join(C.CHECKPOINT_ROOT, "_seedtest", "build", "_distill_fa3_C.so")
SEED_NS = "_distill_fa3_C"
BASELINE_NS = "_vllm_fa3_C"


def run() -> int:
    # (1) Exercise the gate on the SEED candidate (the task-requested call).
    seed_ident, seed_note = check_decode_equiv(SEED_SO, SEED_NS, num_tokens=16)
    seed_skipped = seed_note.startswith("[skip:")
    print(f"[test_decode_equiv] seed: identical={seed_ident} note={seed_note}")

    if seed_skipped:
        # vLLM cannot run here at all -> the whole gate degrades to mechanism-only.
        # That is a legitimate 'skip' for this environment.
        print("[test_decode_equiv] verdict=skip (vLLM engine unavailable; "
              "monkeypatch mechanism validated)")
        return 0

    if seed_ident:
        # Seed happened to be fully decode-equivalent -> strongest pass.
        print("[test_decode_equiv] verdict=pass (seed greedy stream identical)")
        return 0

    # (2) Seed diverged on the live path (expected for a starting fixture).  Prove
    #     the GATE itself is correct via self-validation: routing the monkeypatch to
    #     the baseline op MUST reproduce baseline exactly.
    self_ident, self_note = check_decode_equiv(SEED_SO, BASELINE_NS, num_tokens=16)
    print(f"[test_decode_equiv] self-validate(baseline-routed): "
          f"identical={self_ident} note={self_note}")

    if self_note.startswith("[skip:"):
        print("[test_decode_equiv] verdict=skip (vLLM engine unavailable)")
        return 0
    if self_ident:
        print("[test_decode_equiv] verdict=pass (gate correct: seed diverges as "
              "expected; baseline-routed stream is identical, routing confirmed)")
        return 0

    print("[test_decode_equiv] verdict=fail (gate self-validation diverged -- "
          "harness is mis-wired)")
    return 1


if __name__ == "__main__":
    sys.exit(run())
