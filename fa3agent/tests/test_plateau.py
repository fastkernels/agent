"""Validation for Workstream C2/C2b (shared PlateauState + protected rewrite branch).

Run from /data-fast/agent (torch-free):
    python -m fa3agent.tests.test_plateau

Covers (per the plan's Validation section):
  - a PlateauState replay over run_20260612_185828's candidates.jsonl asserting (i) STUCK fires in
    the micro-opt regime via the STRUCTURAL-stall trigger (b) even though F* / min_ratio are still
    creeping (so the score-plateau trigger (a) does NOT fire), and (ii) knee_cid() resolves to the
    pre-plateau ancestor (~step 17-18);
  - a branch adopt/revert decision replay (deferred adopt-or-revert at budget end);
  - the PlateauState FSM: open -> branch-local tip acceptance -> budget -> adopt/revert -> persist,
    and resume-safe reload + rebuild_from_steps.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

from fa3agent import config as C
from fa3agent import plateau as P
from fa3agent import prompts
from fa3agent.plateau import PlateauState


def _ok(m):
    print(f"  PASS: {m}")


def _fail(m):
    print(f"  FAIL: {m}")
    return 1


def _real_history():
    jl = os.path.join(C.CHECKPOINT_ROOT, "run_20260612_185828", "candidates.jsonl")
    if not os.path.exists(jl):
        return None, jl
    hist = []
    all_recs = []
    with open(jl) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            all_recs.append(d)
            if d.get("is_best"):
                ev = d.get("eval") or {}
                cu = os.path.join(C.CHECKPOINT_ROOT, "run_20260612_185828",
                                  "candidates", f"{d.get('cid')}.cu")
                loc = sum(1 for _ in open(cu)) if os.path.exists(cu) else 0
                hist.append({"step": d.get("step"), "cid": d.get("cid"),
                             "F": ev.get("weighted_geomean_ratio", 0.0), "loc": loc})
    return (hist, all_recs), jl


def main() -> int:
    failures = 0

    # ---- (1) real-run replay: knee + STUCK triggers ----
    print("[1] PlateauState replay over run_20260612_185828")
    real, jl = _real_history()
    if real is None:
        print(f"    (archived run not found at {jl}; running synthetic-only)")
        hist = [{"step": s, "cid": f"step{s:03d}", "F": f, "loc": 200 + 60 * s}
                for s, f in [(15, 0.38), (16, 0.389), (17, 0.4107), (18, 0.422),
                             (19, 0.4249), (22, 0.4275), (25, 0.4372), (27, 0.4445)]]
    else:
        hist, _ = real

    knee = P.knee_point(hist, C.KNEE_REL_BAND)
    if knee and 16 <= knee["step"] <= 18:
        _ok(f"knee resolves to the pre-plateau ancestor step {knee['step']} (cid {knee['cid']})")
    else:
        failures += _fail(f"knee resolved wrong: {knee}")

    # (a) should NOT fire: F* kept creeping > STUCK_EPS over the window
    reasons_no_stall = P.stuck_reasons(hist, rejects=0, recent_structural=[])
    if "score-plateau" not in reasons_no_stall:
        _ok("score-plateau trigger (a) does NOT fire (F* still creeping) -- matches the run")
    else:
        failures += _fail(f"score-plateau fired unexpectedly: {reasons_no_stall}")

    # (b) DOES fire when the last N structural levers all rejected (the decisive trigger here)
    reasons_stall = P.stuck_reasons(hist, rejects=0,
                                    recent_structural=["regressed", "no_effect", "regressed"])
    if "structural-stall" in reasons_stall:
        _ok("structural-stall trigger (b) fires on 3 rejected structural levers")
    else:
        failures += _fail(f"structural-stall did not fire: {reasons_stall}")

    # (c) consecutive non-promotions
    if "rejects" in P.stuck_reasons(hist, rejects=3, recent_structural=[]):
        _ok("reject trigger (c) fires at 3 consecutive non-promotions")
    else:
        failures += _fail("reject trigger (c) did not fire")

    # ---- (2) branch adopt/revert decision ----
    print("[2] deferred adopt-or-revert decision")
    fstar = P.f_star(hist)
    # a rewrite that dipped below the frozen champion (outside tol) must REVERT
    revert = not P.should_adopt(tip_F=fstar * 0.90, anchor_F=fstar, gate_ok=True)
    # one that matches within tol AND clears the gates must ADOPT
    adopt = P.should_adopt(tip_F=fstar * 0.995, anchor_F=fstar, gate_ok=True)
    # a tip that matches but FAILS the faithful gates must NOT adopt
    gate_block = not P.should_adopt(tip_F=fstar, anchor_F=fstar, gate_ok=False)
    if revert and adopt and gate_block:
        _ok(f"adopt iff tip_F>=anchor*(1-tol) AND gates pass (anchor F*={fstar:.3f})")
    else:
        failures += _fail(f"adopt/revert wrong: revert={revert} adopt={adopt} gate_block={gate_block}")

    # ---- (3) FSM lifecycle + persistence ----
    print("[3] PlateauState FSM: open -> tip acceptance -> budget -> adopt -> persist")
    with tempfile.TemporaryDirectory() as d:
        ps = PlateauState(d)
        ps.on_promotion(10, "step010", F=0.30, loc=1000)
        ps.on_promotion(18, "step018", F=0.42, loc=1700)
        ps.on_nonpromotion(); ps.on_nonpromotion()
        b = ps.open_branch(anchor_cid="step018", anchor_F=0.42,
                           seed_cid=ps.knee_cid("step000"), budget=3)
        if not ps.branch_active():
            failures += _fail("branch did not open")
        # branch-local acceptance: a correct, higher-F candidate becomes the tip; lower-F ignored
        ps.record_branch_step(19, "step019", correct=True, F=0.20, min_ratio=0.1, so_path="/x/19.so")
        ps.record_branch_step(20, "step020", correct=True, F=0.41, min_ratio=0.3, so_path="/x/20.so")
        ps.record_branch_step(21, "step021", correct=False, F=None, min_ratio=None, so_path=None)
        tip_ok = ps.branch["tip_cid"] == "step020" and abs(ps.branch["tip_F"] - 0.41) < 1e-6
        budget_ok = ps.budget_exhausted()
        adopt = ps.decide_adopt(gate_ok=True)   # 0.41 >= 0.42*0.99=0.4158? -> revert
        ps.close_branch(adopted=adopt, note="replay")
        reloaded = PlateauState(d)
        persisted = (not reloaded.branch_active() and len(reloaded.data["branch_log"]) == 1
                     and reloaded.data["champ_history"][-1]["cid"] == "step018")
        if tip_ok and budget_ok and (adopt is False) and persisted:
            _ok("tip acceptance + budget + adopt-decision + persistence all correct")
        else:
            failures += _fail(f"FSM wrong: tip_ok={tip_ok} budget={budget_ok} adopt={adopt} "
                              f"persisted={persisted}")

    # ---- (4) rebuild_from_steps (cold resume) + knee fallback ----
    print("[4] rebuild_from_steps re-derives the series; knee falls back to seed when empty")
    with tempfile.TemporaryDirectory() as d:
        ps = PlateauState(d)
        if ps.knee_cid("step000") != "step000":
            failures += _fail("empty-history knee should fall back to the seed cid")
        else:
            steps = [{"step": 1, "cid": "step001", "is_best": True,
                      "eval": {"weighted_geomean_ratio": 0.2}},
                     {"step": 2, "cid": "step002", "is_best": False,
                      "eval": {"weighted_geomean_ratio": 0.1}},
                     {"step": 3, "cid": "step003", "is_best": True,
                      "eval": {"weighted_geomean_ratio": 0.4}}]
            ps.rebuild_from_steps(steps, loc_fn=lambda cid: 100)
            if [h["cid"] for h in ps.champ_history] == ["step001", "step003"]:
                _ok("series rebuilt from is_best records; knee fallback works")
            else:
                failures += _fail(f"rebuild wrong: {ps.champ_history}")

    # ---- (5) branch prompt wiring (C3 journal + D1 rung-0 refactor focus) ----
    print("[5] branch prompt carries the rewrite journal + a structural carve-out (C3)")
    journal = "## Rungs\n- rung 0 step 20 [correct] -> NEW TIP: refactor"
    p = prompts.refine_prompt("/tmp/c.cu", "/tmp/seed.cu", last_eval_summary="ok",
                              profile_feedback="overlap-bound", knowledge_hints="",
                              focus=prompts.BRANCH_REFACTOR_FOCUS,
                              structural_carveout="You are in a PROTECTED REWRITE BRANCH (rung 0)",
                              branch_note=journal)
    if ("PROTECTED REWRITE BRANCH" in p and "REWRITE-BRANCH JOURNAL" in p
            and "RUNG 0 — REFACTOR ONLY" in p):
        _ok("branch prompt includes journal + protected-branch carve-out + rung-0 refactor focus")
    else:
        failures += _fail("branch prompt wiring missing")

    print()
    print("SMOKE: PASS (plateau + branch)" if failures == 0 else f"SMOKE: FAIL ({failures})")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
