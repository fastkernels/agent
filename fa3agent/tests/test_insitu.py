"""Smoke test for the in-situ A/B faithful gate (verify/insitu.py).

Runs check_insitu on the seed fixture: one short live vLLM forward comparing baseline vs
candidate per call.  On the FIXED seed this returns ok=True (max_diff within atol, argmax all
match); if vLLM cannot run in this env it returns skip=True (still a pass for the smoke).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fa3agent import config as C
from fa3agent.verify.insitu import check_insitu

SEED_SO = os.path.join(C.WORK_ROOT, "_seedtest", "build", "_distill_fa3_C.so")

if __name__ == "__main__":
    r = check_insitu(SEED_SO, "_distill_fa3_C", num_steps=2)
    print("insitu result:", r.get("note"))
    ok = bool(r.get("ok")) or bool(r.get("skip"))
    print("TEST", "PASS" if ok else "FAIL",
          f"(ok={r.get('ok')} skip={r.get('skip')} n_calls={r.get('n_calls')} "
          f"max_diff={r.get('max_diff')})")
    sys.exit(0 if ok else 1)
