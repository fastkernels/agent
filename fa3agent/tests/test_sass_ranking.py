"""Regression tests for sass/analyze.py kernel SELECTION (torch-free, no GPU/cuobjdump).

Run from /data-fast/agent:
    python -m fa3agent.tests.test_sass_ranking

These pin the step041 trap: the SASS verdict (uses_wgmma/uses_tma) is computed from the SAMPLED
kernels, so a real wgmma kernel whose symbol does NOT contain "attn_kernel" (e.g.
`attn_prefill_wgmma`) must still be sampled -- otherwise scalar `attn_kernel`/`*_coop`
instantiations monopolise the budget and the verdict mis-reads "no wgmma" though HGMMA is present.

Covers:
  - _family_key parses the Itanium length-prefixed base identifier of a free function;
  - _rank_kernels guarantees per-family COVERAGE: the wgmma family is in the bounded sample even
    when many distinct scalar attn_kernel instantiations would otherwise fill it;
  - _select_targets matches by EXACT family (NCU-dominant-kernel targeting), so "attn_kernel" does
    not also grab "attn_kernel_coop"/"attn_prefill_wgmma", and template/paren/namespace suffixes
    on the NCU name are stripped.
"""
from __future__ import annotations

from collections import OrderedDict

from fa3agent.sass import analyze as A


def _ok(m):
    print(f"  PASS: {m}")


def _fail(m):
    print(f"  FAIL: {m}")
    return 1


def _fam(fn: str) -> str:
    return A._family_key(fn)


def _synthetic_res() -> "OrderedDict[str, dict]":
    """Mimic `cuobjdump --dump-resource-usage` parse output: mangled name -> resource dict.

    8 distinct scalar `attn_kernel` instantiations (is_main=1) that would monopolise the old top-N,
    plus a wgmma prefill kernel NOT named "attn_kernel", a coop kernel, and a combine kernel."""
    res: "OrderedDict[str, dict]" = OrderedDict()
    # 8 scalar attn_kernel instantiations, distinct fingerprints (regs 70..63)
    for i, regs in enumerate(range(70, 62, -1)):
        res[f"_Z11attn_kernelIfLi{128 + i}EE6ParamsIffE"] = dict(
            regs=regs, stack=0, shared=1024, local=0)
    # warpgroup-specialized scalar variant (still "attn_kernel" substring -> its own family)
    res["_Z16attn_kernel_coopIfEv6ParamsIffE"] = dict(regs=60, stack=0, shared=1024, local=0)
    # the genuine wgmma prefill kernel -- NO "attn_kernel" substring; 2 instantiations
    res["_Z18attn_prefill_wgmmaI6__halfS0_Li128ELi128EEv6ParamsIS_S_E"] = dict(
        regs=255, stack=176, shared=1024, local=0)
    res["_Z18attn_prefill_wgmmaI6__halfS0_Li64ELi64EEv6ParamsIS_S_E"] = dict(
        regs=245, stack=64, shared=1024, local=0)
    # tiny split-K combine kernel
    res["_Z14combine_kernelIfEv6ParamsIffE"] = dict(regs=32, stack=0, shared=512, local=0)
    return res


def test_family_key(failures: int) -> int:
    print("[1] _family_key parses the length-prefixed base identifier")
    cases = {
        "_Z18attn_prefill_wgmmaI13__nv_fp8_e4m313__nv_bfloat16Li256ELi256EEv6ParamsIT_T0_E":
            "attn_prefill_wgmma",
        "_Z11attn_kernelIfLi128EE6ParamsIffE": "attn_kernel",
        "_Z16attn_kernel_coopIfEv6ParamsIffE": "attn_kernel_coop",
        "_Z14combine_kernelIfEv6ParamsIffE": "combine_kernel",
    }
    for mangled, want in cases.items():
        got = _fam(mangled)
        if got == want:
            _ok(f"{mangled[:24]}... -> {got!r}")
        else:
            failures = _fail(f"_family_key({mangled[:24]}...) = {got!r}, want {want!r}")
    return failures


def test_family_cover(failures: int) -> int:
    print("[2] _rank_kernels samples the wgmma family even when scalar kernels would fill the budget")
    res = _synthetic_res()
    ranked = A._rank_kernels(res, None)
    sample = ranked[:A.MAX_KERNELS_SASS]
    fams = [_fam(fn) for fn in sample]

    if "attn_prefill_wgmma" in fams:
        _ok(f"wgmma family is in the bounded sample (first {A.MAX_KERNELS_SASS}): {fams}")
    else:
        failures = _fail(f"wgmma family MISSING from sample {fams} -- the step041 trap is back")

    # every distinct family must appear within the first len(families) entries (true family cover)
    all_fams = {_fam(fn) for fn in res}
    lead = {_fam(fn) for fn in ranked[:len(all_fams)]}
    if lead == all_fams:
        _ok(f"all {len(all_fams)} families covered by the leading slots")
    else:
        failures = _fail(f"family cover incomplete: leading={lead}, all={all_fams}")

    # sanity: a pure score sort (the OLD behavior) would have buried the wgmma kernel
    def _score(fn):
        r = res[fn]
        is_main = 1 if ("FlashAttnFwdSm90" in fn or "attn_kernel" in fn) else 0
        return (is_main, r.get("regs", 0))
    old_top = [_fam(fn) for fn in sorted(res, key=_score, reverse=True)[:A.MAX_KERNELS_SASS]]
    if "attn_prefill_wgmma" not in old_top:
        _ok(f"(control) pure-score top-{A.MAX_KERNELS_SASS} would have missed wgmma: {old_top}")
    else:
        _ok("(control) note: pure-score sort already included wgmma here")
    return failures


def test_select_targets(failures: int) -> int:
    print("[3] _select_targets matches the NCU-dominant kernel by EXACT family")
    res = _synthetic_res()

    sel = A._select_targets(res, ["attn_prefill_wgmma<__half, 128, 128>(Params)"])
    sel_fams = {_fam(fn) for fn in sel}
    if sel and sel_fams == {"attn_prefill_wgmma"}:
        _ok(f"target 'attn_prefill_wgmma<...>(...)' -> only wgmma kernels ({len(sel)})")
    else:
        failures = _fail(f"wgmma target selected families {sel_fams} (want just attn_prefill_wgmma)")

    # 'attn_kernel' must NOT also grab attn_kernel_coop / attn_prefill_wgmma (substring trap)
    sel2 = A._select_targets(res, ["attn_kernel"])
    sel2_fams = {_fam(fn) for fn in sel2}
    if sel2 and sel2_fams == {"attn_kernel"}:
        _ok(f"target 'attn_kernel' -> only the scalar family ({len(sel2)}), not coop/wgmma")
    else:
        failures = _fail(f"'attn_kernel' target leaked into families {sel2_fams}")

    # no match -> [] so analyze() falls back to the family-cover ranking
    if A._select_targets(res, ["does_not_exist"]) == [] and A._select_targets(res, []) == []:
        _ok("no-match / empty targets -> [] (caller falls back to family-cover ranking)")
    else:
        failures = _fail("expected [] for unmatched / empty targets")
    return failures


def main() -> int:
    failures = 0
    failures = test_family_key(failures)
    failures = test_family_cover(failures)
    failures = test_select_targets(failures)
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{failures} CHECK(S) FAILED"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
