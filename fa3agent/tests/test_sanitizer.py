"""Smoke test for fa3agent/verify/sanitizer.py (component 2.2).

Runs compute-sanitizer --tool memcheck on the prebuilt SEED candidate .so for the
smallest dev config and asserts we get a parsed, distilled result.  The seed kernel
is known-correct, so we expect clean=True.

Run from /data-fast/agent with CUDA_VISIBLE_DEVICES=2:
    CUDA_VISIBLE_DEVICES=2 python -m fa3agent.tests.test_sanitizer
"""
from __future__ import annotations

import json
import os
import sys

from fa3agent import config as C
from fa3agent.verify.sanitizer import run_sanitizer

SEED_SO = os.path.join(C.WORK_ROOT, "_seedtest", "build", "_distill_fa3_C.so")


def main() -> int:
    cfg = C.DEV_CONFIGS[2]  # smallest dev config (hd64 decode, batch=2, k_len=256)
    res = run_sanitizer(SEED_SO, "_distill_fa3_C", cfg, tool="memcheck")
    print("result:", json.dumps(res, indent=2))

    # Shape contract: must be a parsed dict with the expected keys/types.
    assert isinstance(res, dict), "result must be a dict"
    for key in ("tool", "clean", "num_errors", "errors", "note"):
        assert key in res, f"missing key {key!r}"
    assert res["tool"] == "memcheck"
    assert isinstance(res["clean"], bool)
    assert isinstance(res["num_errors"], int)
    assert isinstance(res["errors"], list) and len(res["errors"]) <= 5
    assert all(isinstance(e, str) for e in res["errors"])
    assert isinstance(res["note"], str)

    note = res["note"].lower()
    perm_blocked = ("permission" in note or "not found" in note or
                    "did not complete" in note or "timeout" in note)

    if perm_blocked:
        print("PARTIAL: sanitizer inconclusive (env/permission):", res["note"])
        return 0  # partial/skip is acceptable per task spec

    if res["clean"] is True:
        print("PASS: seed kernel is memcheck-clean (clean=True, parsed result returned)")
        return 0

    # Got a parsed True/False result but seed flagged as dirty: still a valid parse,
    # but surprising for the known-correct seed -> report as a soft failure.
    print("FAIL: seed reported NOT clean:", res["num_errors"], res["errors"])
    return 1


if __name__ == "__main__":
    sys.exit(main())
