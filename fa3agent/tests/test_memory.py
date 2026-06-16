"""Validation for Workstream E (cross-step memory across fresh sessions).

Run from /data-fast/agent (torch-free; no GPU/claude):
    python -m fa3agent.tests.test_memory

Covers:
  - MEMORY.md persists + GROWS across steps; its digest carries champion + dead-ends + lessons;
  - dead-ends de-duplicate and a recorded dead-end appears in the digest (so the brain is told
    NOT to retry it);
  - lessons compact to a bounded count; champion/bottleneck state is tracked;
  - state survives a kill/--resume (a fresh MemoryStore on the same dir reloads it);
  - the memory digest is injected into the refine prompt (E2);
  - the E1 causal-lesson author produces what-changed -> hypothesis -> outcome-delta -> cause
    for promoted / rejected / incorrect / build-failed outcomes, and _next_hint maps bottlenecks.
"""
from __future__ import annotations

import os
import sys
import tempfile

from fa3agent import prompts
from fa3agent.common import StepRecord
from fa3agent.memory import MemoryStore, _MAX_LESSONS
from fa3agent.orchestrator import Orchestrator, _next_hint


def _ok(m):
    print(f"  PASS: {m}")


def _fail(m):
    print(f"  FAIL: {m}")
    return 1


def _lesson(self_, rec, champ, promoted):
    # _causal_lesson does not touch `self`, so call it unbound with a dummy self.
    return Orchestrator._causal_lesson(self_, rec, champ, promoted)


def main() -> int:
    failures = 0

    with tempfile.TemporaryDirectory() as run_dir:
        # ---- (1) persists + grows + digest content ----
        print("[1] MEMORY.md persists, grows, and digests champion/lessons/dead-ends")
        m = MemoryStore(run_dir)
        m.set_champion("step001", 1, F=0.45, min_ratio=0.34, bottleneck="overlap-bound")
        m.add_lesson({"step": 1, "cid": "step001", "change": "applied: wgmma QK/PV",
                      "hypothesis": "relieve off-tensor-core", "outcome": "correct; F->0.45",
                      "cause": "new bottleneck: overlap-bound", "promoted": True})
        m.commit()
        md1 = os.path.getsize(m.md_path)
        m.add_lesson({"step": 2, "cid": "step002", "change": "applied: pipeline cp.async",
                      "hypothesis": "relieve overlap-bound", "outcome": "correct; F 0.45->0.44",
                      "cause": "still overlap-bound", "promoted": False})
        m.add_dead_end("pipeline cp.async -> rejected (bottleneck stayed overlap-bound)")
        m.commit()
        md2 = os.path.getsize(m.md_path)
        dig = m.digest()
        if (os.path.exists(m.md_path) and md2 > md1
                and "overlap-bound" in dig and "step001" in dig
                and "AVOID" in dig and "pipeline cp.async" in dig):
            _ok("MEMORY.md grew; digest carries champion + dead-end + lessons")
        else:
            failures += _fail(f"memory digest/file wrong (md1={md1} md2={md2}):\n{dig}")

        # ---- (2) dead-end de-dup ----
        print("[2] dead-ends de-duplicate")
        before = len(m.state["dead_ends"])
        m.add_dead_end("pipeline cp.async -> rejected (bottleneck stayed overlap-bound)")  # dup
        if len(m.state["dead_ends"]) == before:
            _ok("duplicate dead-end ignored")
        else:
            failures += _fail("dead-end de-dup failed")

        # ---- (3) lessons compaction (bounded) ----
        print("[3] lessons compact to a bounded count")
        for i in range(3, 3 + _MAX_LESSONS + 5):
            m.add_lesson({"step": i, "cid": f"step{i:03d}", "change": f"edit {i}",
                          "hypothesis": "", "outcome": "rejected", "cause": "", "promoted": False})
        m.commit()
        if len(m.state["lessons"]) == _MAX_LESSONS:
            _ok(f"lessons capped at {_MAX_LESSONS}")
        else:
            failures += _fail(f"lessons not capped: {len(m.state['lessons'])}")

        # ---- (4) survives resume (reload from disk) ----
        print("[4] state survives a kill/--resume (reload from the same dir)")
        m2 = MemoryStore(run_dir)
        if (m2.state.get("champion", {}).get("cid") == "step001"
                and any("pipeline cp.async" in d for d in m2.state.get("dead_ends", []))):
            _ok("a fresh MemoryStore reloaded the persisted champion + dead-ends")
        else:
            failures += _fail(f"resume reload lost state: {m2.state.get('champion')}")

    # ---- (5) digest is injected into the refine prompt (E2) ----
    print("[5] memory digest is injected into refine_prompt")
    sentinel = "MEMORY-SENTINEL-XYZ champion: step042 F=0.99"
    p = prompts.refine_prompt("/tmp/cand.cu", "/tmp/best.cu", last_eval_summary="ok",
                              profile_feedback="overlap-bound", knowledge_hints="", focus="warp-spec",
                              memory_digest=sentinel)
    if sentinel in p and "RUN MEMORY" in p:
        _ok("refine_prompt carries the RUN MEMORY digest")
    else:
        failures += _fail("memory digest not present in refine_prompt")

    # ---- (6) causal lessons (E1) for each outcome ----
    print("[6] E1 causal lessons: what-changed -> hypothesis -> outcome-delta -> cause")
    champ = {"eval": {"weighted_geomean_ratio": 0.450, "min_ratio": 0.340},
             "profile": {"bottleneck": "overlap-bound"}}

    promoted_rec = StepRecord(step=5, cid="step005", parent="step004", status="correct",
                              focus="warp-specialize producer/consumer",
                              profile={"bottleneck": "register-spill"},
                              eval={"weighted_geomean_ratio": 0.500, "min_ratio": 0.40})
    _, note = _lesson(None, promoted_rec, champ, True)
    if "PROMOTED" in note and "0.450->0.500" in note and "register-spill" in note:
        _ok(f"promoted lesson: {note}")
    else:
        failures += _fail(f"promoted lesson wrong: {note}")

    rej_rec = StepRecord(step=6, cid="step006", parent="step005", status="correct",
                         focus="pipeline tweak", profile={"bottleneck": "overlap-bound"},
                         eval={"weighted_geomean_ratio": 0.448, "min_ratio": 0.34})
    _, note = _lesson(None, rej_rec, champ, False)
    if "rejected" in note and "warp-spec" in note.lower():
        _ok(f"rejected lesson carries corrected cause: {note}")
    else:
        failures += _fail(f"rejected lesson wrong: {note}")

    inc_rec = StepRecord(step=7, cid="step007", parent="step005", status="incorrect",
                         focus="aggressive rewrite",
                         eval={"per_config": [{"config": "d_q1_k4096", "correct": False}],
                               "error": "NaN in output"})
    _, note = _lesson(None, inc_rec, champ, False)
    if "INCORRECT" in note and "NaN" in note:
        _ok(f"incorrect lesson: {note}")
    else:
        failures += _fail(f"incorrect lesson wrong: {note}")

    bf_rec = StepRecord(step=8, cid="step008", parent="step005", status="build_failed",
                        focus="tma loads", build={"error": "ptxas: too many registers"})
    _, note = _lesson(None, bf_rec, champ, False)
    if "build failed" in note and "ptxas" in note:
        _ok(f"build-failed lesson: {note}")
    else:
        failures += _fail(f"build-failed lesson wrong: {note}")

    # ---- (7) _next_hint mapping ----
    print("[7] _next_hint maps bottleneck labels to next moves")
    checks = [("overlap-bound", "warp-spec"), ("off-tensor-core", "wgmma"),
              ("register-spill", "setmaxnreg"), ("latency-bound", "TMA"),
              ("low-occupancy", "persistent"), ("unknown-xyz", "DIFFERENT")]
    if all(frag.lower() in _next_hint(bn).lower() for bn, frag in checks):
        _ok("all bottleneck->hint mappings correct")
    else:
        bad = [(bn, _next_hint(bn)) for bn, frag in checks if frag.lower() not in _next_hint(bn).lower()]
        failures += _fail(f"next_hint mapping wrong: {bad}")

    print()
    print("SMOKE: PASS (cross-step memory)" if failures == 0 else f"SMOKE: FAIL ({failures})")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
