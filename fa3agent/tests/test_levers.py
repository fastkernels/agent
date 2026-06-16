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
  - state persists across a reload (resume-safe).
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
            "latency-bound": "tma-multistage-pipeline",
            "dram-bandwidth-bound": "tma-multistage-pipeline",
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
            (("overlap-bound", False, True), "tma-multistage-pipeline"),  # wgmma but no TMA -> TMA
            (("off-tensor-core", False, True), "tma-multistage-pipeline"),
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
            c.record_outcome("tma-multistage-pipeline", "failed", step=0)
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
