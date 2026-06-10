"""Smoke test for fa3agent.reflib.hints (component 2.7).

Run from /data-fast/agent:
    python -m fa3agent.tests.test_reflib
or
    python fa3agent/tests/test_reflib.py
"""

import sys

from fa3agent.reflib.hints import get_hints


def _check(query, must_have_any, must_have_substr=("file:line", ":")):
    out = get_hints(query)
    assert isinstance(out, str) and out.strip(), f"{query!r}: empty output"
    # must reference reference/libraries/ paths with a :line pointer
    assert "reference/libraries/" in out, f"{query!r}: no ref path in output:\n{out}"
    # at least one of the expected source files must appear
    hit = any(h in out for h in must_have_any)
    assert hit, (
        f"{query!r}: expected one of {must_have_any} in output, got:\n{out}"
    )
    # bounded
    assert len(out) <= 8200, f"{query!r}: output not bounded ({len(out)} chars)"
    # must contain a file:line pointer (a digit right after a colon following a path)
    import re
    assert re.search(r"\.(cu|cuh|impl):\d+", out), (
        f"{query!r}: no file:line pointer found:\n{out}"
    )
    return out


def main():
    cases = [
        # get_hints('wgmma bf16') -> ThunderKittens/DeepGEMM snippet with file:line
        ("wgmma bf16", ("ThunderKittens", "DeepGEMM")),
        # get_hints('tma descriptor') -> TK descriptor.cuh / tma.cuh snippet
        ("tma descriptor", ("descriptor.cuh", "tma.cuh", "ThunderKittens")),
        # get_hints('online softmax') -> flashinfer / TK
        ("online softmax", ("flashinfer", "ThunderKittens")),
    ]
    for query, expect in cases:
        out = _check(query, expect)
        first_loc = next(
            (ln for ln in out.splitlines() if ln.startswith("=== ")), "<none>"
        )
        print(f"PASS  get_hints({query!r:22}) -> {first_loc}")

    # bonus topics exercised (non-fatal coverage)
    for q in ("warp specialization", "mbarrier phase", "split-kv", "pipeline"):
        out = get_hints(q)
        assert "reference/libraries/" in out, f"{q!r}: no hit"
        print(f"PASS  get_hints({q!r:22}) -> {len(out)} chars bounded")

    # unknown topic must degrade gracefully (no crash, helpful message)
    miss = get_hints("quantum teleportation")
    assert "no topic match" in miss, f"unknown topic not handled: {miss!r}"
    print("PASS  unknown topic handled gracefully")

    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
