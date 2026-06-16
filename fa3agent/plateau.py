"""Shared plateau detection + protected rewrite-branch state (Workstream C2/C2b).

ONE `PlateauState` drives BOTH the STUCK trigger and the `pre_plateau` knee seed off the same
champion-F series, so they can never disagree.  It also owns the protected rewrite-branch FSM
(freeze anchor -> budgeted branch with branch-local acceptance -> deferred adopt-or-revert).  All of
it is persisted in a ``branches.json`` sidecar in the run workdir (mirrored to the archive by F3,
restored on resume); the champion series is also rebuildable from the ``is_best`` records in
``candidates.jsonl`` so a cold resume re-derives it.

Pure logic (no GPU/torch), so the keep/STUCK/knee/adopt decisions can be replayed offline against a
run's ``candidates.jsonl`` (see tests/test_plateau.py).
"""
from __future__ import annotations

import json
import os
from typing import Optional

from . import config as C


def f_star(history: list) -> Optional[float]:
    """Running max champion F over the series (None if empty)."""
    fs = [p.get("F") for p in history if isinstance(p.get("F"), (int, float))]
    return max(fs) if fs else None


def knee_point(history: list, rel_band: float) -> Optional[dict]:
    """The lineage knee: the EARLIEST champion within ``rel_band`` of F* (smallest-loc tie-break).

    This is the pre-plateau ancestor that already has the proven gains (within the band of the
    peak) but the LEAST micro-opt encrustation -- the cleanest substrate to restructure from."""
    fs = f_star(history)
    if fs is None or fs <= 0:
        return None
    thresh = fs * (1.0 - rel_band)
    in_band = [p for p in history if isinstance(p.get("F"), (int, float)) and p["F"] >= thresh]
    if not in_band:
        return None
    # earliest step first; tie-break on smallest loc (least entangled)
    in_band.sort(key=lambda p: (p.get("step", 0), p.get("loc", 1 << 30)))
    return in_band[0]


def stuck_reasons(history: list, rejects: int, recent_structural: list,
                  eps: float = None, window: int = None,
                  structural_stall: int = None, min_rejects: int = None) -> list:
    """Return the list of STUCK triggers currently firing (empty == not stuck).

    (a) score-plateau: F* gained < eps over the last `window` promotions;
    (b) structural-stall: the last `structural_stall` STRUCTURAL levers ALL rejected (not landed);
    (c) rejects: >= `min_rejects` consecutive non-promotions.
    """
    eps = C.STUCK_EPS if eps is None else eps
    window = C.STUCK_WINDOW if window is None else window
    structural_stall = C.STUCK_STRUCTURAL_STALL if structural_stall is None else structural_stall
    min_rejects = C.STUCK_MIN_REJECTS if min_rejects is None else min_rejects

    reasons: list = []
    # (a) score-plateau: compare F* now vs F* as of `window` promotions ago.
    if len(history) > window:
        old_fstar = f_star(history[:-window])
        cur_fstar = f_star(history)
        if old_fstar is not None and cur_fstar is not None and (cur_fstar - old_fstar) < eps:
            reasons.append("score-plateau")
    # (b) structural-stall: last N structural lever outcomes all non-landing.
    if len(recent_structural) >= structural_stall:
        tail = recent_structural[-structural_stall:]
        if all(o != "landed" for o in tail):
            reasons.append("structural-stall")
    # (c) consecutive non-promotions.
    if rejects >= min_rejects:
        reasons.append("rejects")
    return reasons


def should_adopt(tip_F: Optional[float], anchor_F: Optional[float],
                 rel_tol: float = None, gate_ok: bool = True) -> bool:
    """Adopt the branch tip iff it matches-or-beats the frozen champion within `rel_tol` AND clears
    the faithful gates.  (`anchor_F` None -> there was no champion, so any correct tip adopts.)"""
    rel_tol = C.BRANCH_ADOPT_REL_TOL if rel_tol is None else rel_tol
    if tip_F is None:
        return False
    if not gate_ok:
        return False
    if anchor_F is None:
        return True
    return tip_F >= anchor_F * (1.0 - rel_tol)


class PlateauState:
    """Champion-F series + STUCK detection + knee seed + rewrite-branch FSM (persisted)."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, "branches.json")
        self.data = self._load()

    # ---- persistence ------------------------------------------------------------------
    def _default(self) -> dict:
        return {"champ_history": [], "rejects": 0, "branch": None, "branch_log": []}

    def _load(self) -> dict:
        try:
            with open(self.path) as fh:
                d = json.load(fh)
            base = self._default()
            base.update(d or {})
            return base
        except Exception:
            return self._default()

    def _save(self):
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.data, fh, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # ---- champion series --------------------------------------------------------------
    @property
    def champ_history(self) -> list:
        return self.data["champ_history"]

    def rebuild_from_steps(self, steps: list, loc_fn=None):
        """Re-derive champ_history from the is_best records in candidates.jsonl (cold resume).

        Only fills it when empty (a live branches.json already has it).  `loc_fn(cid)->int` supplies
        the champion .cu line count (entanglement proxy); 0 when unavailable."""
        if self.data["champ_history"]:
            return
        hist = []
        for s in steps:
            if not s.get("is_best"):
                continue
            F = (s.get("eval", {}) or {}).get("weighted_geomean_ratio", 0.0)
            cid = s.get("cid")
            loc = loc_fn(cid) if (loc_fn and cid) else 0
            hist.append({"step": s.get("step"), "cid": cid, "F": F, "loc": loc})
        self.data["champ_history"] = hist
        self._save()

    def on_promotion(self, step: int, cid: str, F: float, loc: int):
        self.data["champ_history"].append({"step": step, "cid": cid,
                                           "F": round(float(F), 5), "loc": int(loc)})
        self.data["rejects"] = 0
        self._save()

    def on_nonpromotion(self):
        self.data["rejects"] = self.data.get("rejects", 0) + 1
        self._save()

    def f_star(self) -> Optional[float]:
        return f_star(self.data["champ_history"])

    def knee_cid(self, seed_cid: str, rel_band: float = None) -> str:
        rel_band = C.KNEE_REL_BAND if rel_band is None else rel_band
        kp = knee_point(self.data["champ_history"], rel_band)
        return kp["cid"] if kp else seed_cid

    def stuck_reasons(self, recent_structural: list) -> list:
        return stuck_reasons(self.data["champ_history"], self.data.get("rejects", 0),
                             recent_structural)

    # ---- rewrite-branch FSM -----------------------------------------------------------
    def branch_active(self) -> bool:
        b = self.data.get("branch")
        return bool(b and b.get("active"))

    @property
    def branch(self) -> Optional[dict]:
        return self.data.get("branch")

    def open_branch(self, anchor_cid: str, anchor_F: Optional[float], seed_cid: str,
                    budget: int = None) -> dict:
        budget = C.BRANCH_BUDGET if budget is None else budget
        b = {
            "active": True,
            "anchor_cid": anchor_cid, "anchor_F": (None if anchor_F is None else round(anchor_F, 5)),
            "seed_cid": seed_cid,
            "budget": int(budget), "steps_used": 0,
            "tip_cid": seed_cid, "tip_F": None, "tip_min_ratio": None, "tip_so": None,
            "step_indices": [], "rung": 0, "escalated": False,
        }
        self.data["branch"] = b
        self._save()
        return b

    def record_branch_step(self, step: int, cid: str, correct: bool,
                           F: Optional[float], min_ratio: Optional[float], so_path: Optional[str]):
        """Branch-local acceptance: a CORRECT candidate that improves the branch's own best F
        becomes the new tip (may be below the trunk champion -- that's allowed inside a branch)."""
        b = self.data.get("branch")
        if not b:
            return
        b["steps_used"] = b.get("steps_used", 0) + 1
        b.setdefault("step_indices", []).append(step)
        if correct and F is not None and (b.get("tip_F") is None or F > b["tip_F"]):
            b["tip_cid"] = cid
            b["tip_F"] = round(float(F), 5)
            b["tip_min_ratio"] = (None if min_ratio is None else round(float(min_ratio), 5))
            b["tip_so"] = so_path
        self._save()

    def escalate_to_seed(self, seed_cid: str):
        """D1 layout-blocker escalation: restart the branch from the bare seed instead of the knee."""
        b = self.data.get("branch")
        if b:
            b["seed_cid"] = seed_cid
            b["escalated"] = True
            b["rung"] = 0
            self._save()

    def advance_rung(self):
        b = self.data.get("branch")
        if b:
            b["rung"] = b.get("rung", 0) + 1
            self._save()

    def budget_exhausted(self) -> bool:
        b = self.data.get("branch")
        return bool(b and b.get("steps_used", 0) >= b.get("budget", 0))

    def decide_adopt(self, gate_ok: bool = True) -> bool:
        b = self.data.get("branch")
        if not b:
            return False
        return should_adopt(b.get("tip_F"), b.get("anchor_F"), gate_ok=gate_ok)

    def close_branch(self, adopted: bool, note: str = "") -> Optional[dict]:
        b = self.data.get("branch")
        if not b:
            return None
        b["active"] = False
        b["adopted"] = bool(adopted)
        if note:
            b["note"] = note[:300]
        self.data.setdefault("branch_log", []).append(b)
        self.data["branch_log"] = self.data["branch_log"][-20:]
        self.data["branch"] = None
        self.data["rejects"] = 0   # branch outcome resets the trunk reject counter
        self._save()
        return b
