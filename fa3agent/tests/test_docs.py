"""Smoke test for docs/retrieve.py (component 2.4).

Run from /data-fast/agent:  python -m fa3agent.tests.test_docs
"""

from __future__ import annotations

import sys

from fa3agent.docs.retrieve import retrieve


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' :: ' + detail) if detail else ''}")
    return bool(cond)


def main() -> int:
    ok = True

    # 1. wgmma query -> snippet from PTX or hopper specs.
    r1 = retrieve("wgmma m64n128k16 bf16 warpgroup", k=3)
    ok &= _check("wgmma query non-empty", bool(r1.strip()), f"{len(r1)}B")
    rel1 = ("wgmma" in r1.lower()) or ("warpgroup" in r1.lower())
    src1 = ("parallel-thread-execution.md" in r1) or ("hopper" in r1.lower())
    ok &= _check("wgmma query relevant term present", rel1)
    ok &= _check("wgmma query from PTX/hopper source", src1,
                 r1.splitlines()[0] if r1 else "")
    ok &= _check("wgmma output bounded <=6KB", len(r1.encode("utf-8")) <= 6 * 1024,
                 f"{len(r1.encode('utf-8'))}B")

    # 2. TMA descriptor query.
    r2 = retrieve("TMA cp.async.bulk.tensor descriptor", k=2)
    ok &= _check("TMA query non-empty", bool(r2.strip()), f"{len(r2)}B")
    rel2 = ("tma" in r2.lower()) or ("cp.async.bulk.tensor" in r2.lower()) \
        or ("descriptor" in r2.lower()) or ("tensor" in r2.lower())
    ok &= _check("TMA query relevant term present", rel2)
    ok &= _check("TMA output bounded <=6KB", len(r2.encode("utf-8")) <= 6 * 1024,
                 f"{len(r2.encode('utf-8'))}B")

    # 3. snippet line bound (<= ~40 lines body per snippet, +header/markers).
    longest = max((len(s.splitlines()) for s in r1.split("\n--------\n")), default=0)
    ok &= _check("snippet line-bound respected", longest <= 45, f"{longest} lines")

    # 4. scope filtering: restrict to architecture-specs.
    r3 = retrieve("TMA warpgroup tensor core", k=2, scope="architecture-specs")
    ok &= _check("scope=architecture-specs returns from that dir",
                 bool(r3.strip()) and "hopper" in r3.lower(),
                 r3.splitlines()[0] if r3 else "(empty)")

    # 5. scope filtering by filename.
    r4 = retrieve("wgmma async warpgroup matrix", k=2,
                  scope="parallel-thread-execution.md")
    ok &= _check("scope=filename returns only that file",
                 bool(r4.strip()) and all(
                     "parallel-thread-execution.md" in ln
                     for ln in r4.splitlines() if ln.startswith("source:")),
                 r4.splitlines()[0] if r4 else "(empty)")

    # 6. header format present.
    ok &= _check("header 'source:' + '#heading' present", r1.startswith("source:")
                 and "#" in r1.splitlines()[0])

    # 7. empty / nonsense query handled gracefully (no crash, returns str).
    r5 = retrieve("", k=3)
    r6 = retrieve("zzqqxx_nonexistent_token_99999", k=3)
    ok &= _check("empty query returns str", isinstance(r5, str))
    ok &= _check("no-match query returns str", isinstance(r6, str), repr(r6[:60]))

    print("\n=== SMOKE", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
