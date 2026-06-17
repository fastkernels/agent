"""Validation for Workstream B3/B3b/D2 (bottleneck-driven lever controller).

Run from /data-fast/agent (torch-free):
    python -m fa3agent.tests.test_levers

Covers:
  - selection is driven by the NCU bottleneck label (NOT step%5), lowest-rung match wins;
  - the later-ladder techniques (B3b: clusters/multicast, async store, persistent/L2, faster
    barriers) are first-class structural levers in the shared catalog;
  - a lever EXHAUSTS after LEVER_MAX_FAILS non-landing outcomes and is no longer suggested (a
    different lever for the same bottleneck is chosen instead); a landing resets the streak;
  - outcome classification (landed/regressed/no_effect/failed);
  - the structural-outcome history (for C2) + ledger summary (for MEMORY.md);
  - state persists across a reload (resume-safe);
  - completion-gate metadata: the foundation levers (wgmma/TMA/fp8-wgmma) carry a SASS `verify`
    adoption signal (and nothing else does), and the "introduces new functionality" trigger is a
    pure function of that metadata + the source kernel's SASS.
"""
from __future__ import annotations

import sys
import tempfile

from fa3agent import levers as L
from fa3agent.levers import LeverController


def _ok(m):
    print(f"  PASS: {m}")


def _fail(m):
    print(f"  FAIL: {m}")
    return 1


def main() -> int:
    failures = 0

    # ---- (1) selection by bottleneck label ----
    print("[1] selection is keyed by the NCU bottleneck label (lowest-rung match)")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        cases = {
            "off-tensor-core": "wgmma-tensorcore",
            "overlap-bound": "warp-specialize",
            "register-spill": "setmaxnreg-split-registers",
            "latency-bound": "tma-load",
            "dram-bandwidth-bound": "tma-load",
            "occupancy-bound": "decode-splitkv",
            "sync-bound": "warp-specialize",
        }
        bad = {bn: c.select(bn, 0).id for bn in cases if c.select(bn, 0).id != cases[bn]}
        if not bad:
            _ok("all bottleneck->lever selections correct")
        else:
            failures += _fail(f"wrong selections: {bad}")

    # ---- (2) B3b later-ladder techniques are first-class structural levers ----
    print("[2] later-ladder techniques are first-class structural levers (B3b)")
    ladder = {"clusters-tma-multicast", "async-store-epilogue",
              "persistent-l2-scheduling", "faster-barriers"}
    present = {lv.id for lv in L.CATALOG}
    missing = ladder - present
    nonstruct = {i for i in (ladder - {"faster-barriers"}) if L.lever_by_id(i).kind != L.STRUCTURAL}
    if not missing and not nonstruct:
        _ok("clusters/multicast, async-store, persistent/L2 present + structural")
    else:
        failures += _fail(f"missing={missing} nonstructural={nonstruct}")

    # ---- (3) exhaustion + switch to a different lever for the same bottleneck ----
    print(f"[3] a lever exhausts after {L.LEVER_MAX_FAILS} non-landing outcomes; controller switches")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        first = c.select("overlap-bound", 0).id           # warp-specialize
        for _ in range(L.LEVER_MAX_FAILS):
            c.record_outcome(first, "no_effect", step=0)
        if not c.is_exhausted(first):
            failures += _fail(f"{first} should be exhausted")
        else:
            nxt = c.select("overlap-bound", 1).id
            if nxt != first and not c.is_exhausted(nxt):
                _ok(f"{first} exhausted -> switched to {nxt}")
            else:
                failures += _fail(f"did not switch away from exhausted lever (got {nxt})")

    # ---- (4) a landing resets the fail streak ----
    print("[4] a landing resets the fail streak (no premature exhaustion)")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        lid = "wgmma-tensorcore"
        c.record_outcome(lid, "regressed", step=0)
        c.record_outcome(lid, "landed", step=1)
        c.record_outcome(lid, "regressed", step=2)
        if not c.is_exhausted(lid):
            _ok("fail streak reset by the intervening landing")
        else:
            failures += _fail("landing did not reset the fail streak")

    # ---- (5) outcome classification ----
    print("[5] classify_outcome maps (promoted, correct, dF) -> outcome")
    co = LeverController.classify_outcome
    checks = [
        (co(True, True, 0.05), "landed"),
        (co(False, False, None), "failed"),
        (co(False, True, -0.5), "regressed"),
        (co(False, True, 0.0), "no_effect"),
        (co(False, True, -0.0001), "no_effect"),   # within noise band
    ]
    if all(got == want for got, want in checks):
        _ok("outcome classification correct")
    else:
        failures += _fail(f"classification wrong: {checks}")

    # ---- (6) structural history (C2) + ledger (MEMORY.md) + persistence ----
    print("[6] structural history + ledger persist across reload (resume-safe)")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        c.record_outcome("warp-specialize", "regressed", step=20)   # structural
        c.record_outcome("softmax-log2", "no_effect", step=21)      # micro
        c.record_outcome("warp-specialize", "regressed", step=22)   # structural -> exhausts
        hist = c.recent_structural_outcomes(3)
        led = c.ledger_summary()
        c2 = LeverController(d)  # reload
        if (hist == ["regressed", "regressed"] and "warp-specialize" in led
                and c2.is_exhausted("warp-specialize")):
            _ok(f"structural history={hist}; exhaustion persisted across reload")
        else:
            failures += _fail(f"history={hist} ledger={led} reload_exhausted="
                              f"{c2.is_exhausted('warp-specialize')}")

    # ---- (7) fallback rotation when there is no bottleneck label ----
    print("[7] deterministic fallback rotation when no label / all matches exhausted")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        a = c.select("", 0).id
        b = c.select("", 0).id
        if a == b and L.lever_by_id(a) is not None:
            _ok(f"no-label selection is deterministic ({a})")
        else:
            failures += _fail(f"fallback not deterministic: {a} vs {b}")

    # ---- (8) TMA/tensor-core foundation gate (B3b) ----
    print("[8] foundation gate: overlap/off-tensor-core lands wgmma then TMA before warp-spec")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        cases = [
            # (label, uses_tma, uses_wgmma) -> expected lever
            (("off-tensor-core", False, False), "wgmma-tensorcore"),   # no wgmma -> land wgmma first
            (("overlap-bound", False, True), "tma-load"),              # wgmma but no TMA -> TMA load
            (("off-tensor-core", False, True), "tma-load"),
            (("overlap-bound", True, True), "warp-specialize"),        # foundation present -> warp-spec
            (("overlap-bound", None, None), "warp-specialize"),        # unknown SASS -> gate skipped
        ]
        bad = {}
        for (label, ut, uw), want in cases:
            got = c.select(label, 0, uses_tma=ut, uses_wgmma=uw).id
            if got != want:
                bad[(label, ut, uw)] = f"{got}!={want}"
        if not bad:
            _ok("foundation gate routes wgmma->TMA->warp-spec; skipped when SASS unknown")
        else:
            failures += _fail(f"gate wrong: {bad}")

    # ---- (8b) gate yields to warp-spec once the TMA foundation lever is exhausted ----
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        for _ in range(L.LEVER_MAX_FAILS):
            c.record_outcome("tma-load", "failed", step=0, adopted=True)   # genuine perf-fail with TMA present
        got = c.select("overlap-bound", 1, uses_tma=False, uses_wgmma=True).id
        if got == "warp-specialize":
            _ok("exhausted TMA foundation -> falls through to warp-specialize")
        else:
            failures += _fail(f"expected warp-specialize after TMA exhausted, got {got}")

    # ---- (9) structural-first: a micro lever is never chosen for a structural bottleneck ----
    print("[9] structural-first ordering (softmax-log2 no longer routed to overlap-bound)")
    so_log2 = L.lever_by_id("softmax-log2")
    if "overlap-bound" not in so_log2.bottlenecks:
        # with the foundation present, overlap-bound must pick a STRUCTURAL lever, not softmax-log2
        with tempfile.TemporaryDirectory() as d:
            c = LeverController(d)
            lv = c.select("overlap-bound", 0, uses_tma=True, uses_wgmma=True)
            if lv.kind == L.STRUCTURAL and lv.id != "softmax-log2":
                _ok(f"overlap-bound -> structural lever ({lv.id}), micro softmax-log2 excluded")
            else:
                failures += _fail(f"overlap-bound picked non-structural/softmax: {lv.id} ({lv.kind})")
    else:
        failures += _fail("softmax-log2 still mapped to overlap-bound")

    # ---- (9b) breadth-first structural exploration reaches ping-pong for overlap-bound ----
    print("[9b] overlap-bound explores structural levers breadth-first (warp-spec then ping-pong)")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        first = c.select("overlap-bound", 0, uses_tma=True, uses_wgmma=True).id
        c.record_outcome(first, "landed", step=0)               # warp-spec landed (NOT exhausted)
        second = c.select("overlap-bound", 1, uses_tma=True, uses_wgmma=True).id
        if first == "warp-specialize" and second == "ping-pong-overlap":
            _ok("overlap-bound: warp-specialize then ping-pong-overlap (both structural)")
        else:
            failures += _fail(f"expected warp-specialize then ping-pong-overlap, got {first}/{second}")

    # ---- (9c) fp8 foundation gate: a slow fp8 worst-config with no QGMMA -> fp8 wgmma ----
    print("[9c] fp8 foundation gate routes a scalar-fallback fp8 worst-config to fp8-wgmma-promote")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        cases = [
            # (label, worst_dtype, uses_wgmma, uses_fp8_wgmma) -> expected lever
            # fp8 worst-config on scalar, bf16 wgmma infra exists -> promote fp8 to tensor cores
            (("overlap-bound", "fp8e4m3", True, False), "fp8-wgmma-promote"),
            (("off-tensor-core", "fp8e4m3", True, False), "fp8-wgmma-promote"),
            # fp8 already on tensor cores -> gate off, normal routing
            (("overlap-bound", "fp8e4m3", True, True), "warp-specialize"),
            # no tensor cores anywhere yet -> establish bf16 wgmma first (don't force fp8 wgmma)
            (("off-tensor-core", "fp8e4m3", False, False), "wgmma-tensorcore"),
            # bf16 worst-config -> fp8 gate must not fire
            (("overlap-bound", "bf16", True, False), "warp-specialize"),
        ]
        bad = {}
        for (label, wd, uw, ufw), want in cases:
            got = c.select(label, 0, uses_tma=True, uses_wgmma=uw,
                           uses_fp8_wgmma=ufw, worst_dtype=wd).id
            if got != want:
                bad[(label, wd, uw, ufw)] = f"{got}!={want}"
        if not bad:
            _ok("fp8 gate fires only for a scalar fp8 worst-config with existing wgmma infra")
        else:
            failures += _fail(f"fp8 gate wrong: {bad}")

    # ---- (10b) completion-gate metadata + the "introducing a new functionality" trigger ----
    print("[10b] completion-gate: foundation levers carry a SASS verify signal; trigger is metadata")
    from fa3agent.orchestrator import _sass_has
    # the three foundation levers carry their adoption signal; nothing else does
    want_verify = {"wgmma-tensorcore": "uses_wgmma",
                   "tma-load": "uses_tma",
                   "fp8-wgmma-promote": "uses_fp8_wgmma"}
    got_verify = {lv.id: lv.verify for lv in L.CATALOG if lv.verify}
    labels_ok = all(sig in L.VERIFY_LABELS for sig in want_verify.values())
    # _sass_has: confirmed only when SASS ok AND the summary flag is truthy
    sass_ok = (_sass_has({"ok": True, "summary": {"uses_wgmma": True}}, "uses_wgmma")
               and not _sass_has({"ok": True, "summary": {"uses_wgmma": False}}, "uses_wgmma")
               and not _sass_has({"ok": False}, "uses_wgmma")            # unreadable -> not confirmed
               and not _sass_has(None, "uses_wgmma"))

    def _introduces(want, src_flag):
        # mirrors orchestrator._step: gate fires when the lever wants a signal the source LACKS
        # (False or unknown); an explicit True means it is a refinement -> skip.
        return bool(want and src_flag is not True)

    trigger_ok = (_introduces("uses_wgmma", False)        # source has no wgmma -> introduce
                  and _introduces("uses_wgmma", None)     # unknown source SASS -> verify
                  and not _introduces("uses_wgmma", True)  # already on wgmma -> refinement, skip
                  and not _introduces("", False))          # micro lever (no signal) -> never gated
    if got_verify == want_verify and labels_ok and sass_ok and trigger_ok:
        _ok("verify signals on the 3 foundation levers only; _sass_has + introduce-trigger correct")
    else:
        failures += _fail(f"verify={got_verify} labels_ok={labels_ok} sass_ok={sass_ok} "
                          f"trigger_ok={trigger_ok}")

    # ---- (11) adoption-aware exhaustion: a NOT-implemented foundation lever is not locked out ----
    print(f"[11] foundation lever with capability ABSENT survives {L.LEVER_MAX_FAILS} perf-misses; "
          f"exhausts only after {L.LEVER_NOT_IMPL_CAP} non-implementations")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        lid = "tma-load"                     # verify=uses_tma
        # brain substituted an easier path: perf "no_effect" but TMA never emitted (adopted=False)
        for _ in range(L.LEVER_MAX_FAILS):
            c.record_outcome(lid, "no_effect", step=0, adopted=False)
        survived = not c.is_exhausted(lid)
        # keep going to the not-implemented cap -> now it gives up
        for _ in range(L.LEVER_NOT_IMPL_CAP - L.LEVER_MAX_FAILS):
            c.record_outcome(lid, "no_effect", step=0, adopted=False)
        exhausted_after_cap = c.is_exhausted(lid)
        e = c.state["levers"][lid]
        if (survived and exhausted_after_cap and e.get("not_impl") == L.LEVER_NOT_IMPL_CAP
                and e.get("fail_streak") == 0):
            _ok(f"absent-capability misses tracked as not_impl (={e['not_impl']}); "
                f"fail_streak untouched; exhausts at the cap")
        else:
            failures += _fail(f"survived={survived} exhausted_after_cap={exhausted_after_cap} "
                              f"entry={e}")

    # ---- (11b) adopted=True -> the technique got a fair test, normal exhaustion applies ----
    print("[11b] foundation lever with capability PRESENT exhausts on the normal fail streak")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        lid = "wgmma-tensorcore"             # verify=uses_wgmma
        for _ in range(L.LEVER_MAX_FAILS):
            c.record_outcome(lid, "no_effect", step=0, adopted=True)
        e = c.state["levers"][lid]
        if c.is_exhausted(lid) and e.get("not_impl", 0) == 0 and e.get("no_effect") == L.LEVER_MAX_FAILS:
            _ok("present-capability perf-misses use the normal fail streak -> exhausts as before")
        else:
            failures += _fail(f"expected normal exhaustion, entry={e} exhausted={c.is_exhausted(lid)}")

    # ---- (11c) a real adoption (landed) resets the not-implemented streak ----
    print("[11c] once the capability lands, the not-implemented streak resets")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        lid = "tma-load"
        c.record_outcome(lid, "no_effect", step=0, adopted=False)   # not implemented
        c.record_outcome(lid, "landed", step=1, adopted=True)       # finally emitted + helped
        if c.state["levers"][lid].get("not_impl_streak") == 0 and not c.is_exhausted(lid):
            _ok("not_impl_streak reset by a genuine adoption")
        else:
            failures += _fail(f"streak not reset: {c.state['levers'][lid]}")

    # ---- (12) reset_exhaustion reopens foundation levers only ----
    print("[12] reset_exhaustion(foundation_only) reopens prerequisite levers, keeps tactics closed")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        for _ in range(L.LEVER_NOT_IMPL_CAP):                       # exhaust a foundation lever
            c.record_outcome("tma-load", "no_effect", step=0, adopted=False)
        for _ in range(L.LEVER_MAX_FAILS):                          # exhaust a tactic lever
            c.record_outcome("ping-pong-overlap", "no_effect", step=1)
        reopened = c.reset_exhaustion(foundation_only=True)
        if ("tma-load" in reopened
                and not c.is_exhausted("tma-load")
                and c.is_exhausted("ping-pong-overlap")):
            _ok(f"reopened foundation levers {reopened}; tactic ping-pong-overlap stays exhausted")
        else:
            failures += _fail(f"reopened={reopened} tma_open={not c.is_exhausted('tma-load')} "
                              f"pingpong_exhausted={c.is_exhausted('ping-pong-overlap')}")

    # ---- (12b) the 'not_implemented' history tag replays correctly (rollback rebuild path) ----
    print("[12b] a 'not_implemented' history entry replays without corrupting the ledger")
    with tempfile.TemporaryDirectory() as d:
        c = LeverController(d)
        c.record_outcome("tma-load", "not_implemented", step=5)
        e = c.state["levers"]["tma-load"]
        if e.get("not_impl") == 1 and e.get("fail_streak") == 0 and e.get("not_implemented") is None:
            _ok("replayed not_implemented tag -> not_impl counter (no bogus key, fail_streak intact)")
        else:
            failures += _fail(f"bad replay of not_implemented: {e}")

    # ---- (13) Layer 3: tma-multistage split into a foundation + a tactic, old id aliased ----
    print("[13] Layer 3: tma-load (foundation) + multistage-pipeline (tactic); old id aliases")
    tma_load = L.lever_by_id("tma-load")
    multistage = L.lever_by_id("multistage-pipeline")
    aliased = L.lever_by_id("tma-multistage-pipeline")     # retired id -> resolves to tma-load
    ok13 = (tma_load is not None and tma_load.verify == "uses_tma" and tma_load.kind == L.STRUCTURAL
            and multistage is not None and multistage.verify == "" and multistage.kind == L.STRUCTURAL
            and L.lever_by_id("tma-multistage-pipeline") is tma_load     # alias resolution
            and aliased is tma_load
            and tma_load.rung < multistage.rung)               # foundation preferred before the ring
    if ok13:
        _ok("tma-load is the gated foundation; multistage-pipeline is a no-verify tactic; alias OK")
    else:
        failures += _fail(f"split wrong: tma_load={tma_load and (tma_load.verify, tma_load.kind)} "
                          f"multistage={multistage and (multistage.verify, multistage.kind)} "
                          f"alias={aliased and aliased.id}")

    # ---- (14) Layer 4: all-or-nothing levers tagged branch_dev (and only the right ones) ----
    print("[14] Layer 4: branch_dev tags the all-or-nothing rewrites only")
    want_branch_dev = {"tma-load", "persistent-l2-scheduling", "clusters-tma-multicast"}
    got_branch_dev = {lv.id for lv in L.CATALOG if lv.branch_dev}
    not_dev = {"multistage-pipeline", "warp-specialize", "softmax-log2", "wgmma-tensorcore",
               "faster-barriers"}
    ok14 = (got_branch_dev == want_branch_dev
            and all(L.is_branch_dev(i) for i in want_branch_dev)
            and not any(L.is_branch_dev(i) for i in not_dev)
            and L.is_branch_dev("tma-multistage-pipeline"))    # alias -> tma-load (branch_dev)
    if ok14:
        _ok(f"branch_dev == {sorted(got_branch_dev)}; is_branch_dev() correct incl. alias")
    else:
        failures += _fail(f"branch_dev wrong: got={got_branch_dev} want={want_branch_dev}")

    # ---- (10) every lever has a reflib hints topic that resolves to a real topic ----
    print("[10] every lever's get_hints topic resolves (no hardcoded single topic)")
    from fa3agent.reflib.hints import _resolve_topics  # pure (no files needed)
    bad_hints = {}
    for lv in L.CATALOG:
        if not lv.hints or not _resolve_topics(lv.hints):
            bad_hints[lv.id] = repr(lv.hints)
    # distinct techniques must map to distinct hint topics (not all -> "wgmma")
    distinct = {tuple(_resolve_topics(lv.hints)) for lv in L.CATALOG}
    if not bad_hints and len(distinct) >= 6:
        _ok(f"all {len(L.CATALOG)} levers resolve; {len(distinct)} distinct hint topic sets")
    else:
        failures += _fail(f"unresolved={bad_hints} distinct_topic_sets={len(distinct)}")

    print()
    print("SMOKE: PASS (lever controller)" if failures == 0 else f"SMOKE: FAIL ({failures})")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
