"""Promotion objective (C1): which correct candidate becomes the champion.

The orchestrator promotes at most one champion per step.  Two objectives, selected by
``config.PROMOTION_OBJECTIVE``:

  - ``"min"`` (DEFAULT, == legacy): promote on the worst-case ratio ``min_ratio`` with a max()
    ratchet + a measurement-noise band.  Greedy on the single worst config -- it cannot fund a
    structural refactor that temporarily dips one config even if the overall kernel improves.
  - ``"weighted_geomean"``: promote on the weighted-geomean ``F`` with a NON-monotone base
    (``champ_F := F`` on promotion) and a small match-or-improve relative tolerance, so an
    equal-overall-score refactor can LAND even without strictly beating the running peak.

``min_ratio`` remains the success/termination gate + a tie-break under BOTH objectives (it is the
spec's "<= baseline on ALL configs" success criterion).

Everything here is a PURE function of the per-step eval dict, so the keep set can be replayed
offline against a run's ``candidates.jsonl`` (see tests/test_promotion.py).
"""
from __future__ import annotations

import math
from typing import Optional

from . import config as C


def weighted_geomean(ratios, weights) -> float:
    """exp( sum(w_i * ln(x_i)) / sum(w_i) ) over x_i > 0, w_i > 0.

    Correct for FRACTIONAL weights (the legacy `for _ in range(int(weight))` duplication both
    truncated fractional weights and silently dropped configs with weight < 1)."""
    num = 0.0
    den = 0.0
    for r, w in zip(ratios, weights):
        try:
            r = float(r)
            w = float(w)
        except (TypeError, ValueError):
            continue
        if r > 0 and w > 0:
            num += w * math.log(r)
            den += w
    return math.exp(num / den) if den > 0 else 0.0


def objective_score(ev: dict, objective: Optional[str] = None) -> float:
    """The scalar the promotion decision compares, per the active objective."""
    objective = objective or C.PROMOTION_OBJECTIVE
    if objective == "weighted_geomean":
        return float((ev or {}).get("weighted_geomean_ratio") or 0.0)
    return float((ev or {}).get("min_ratio") or 0.0)


def meets_promotion_bar(score: float, champ_score: Optional[float],
                        objective: Optional[str] = None,
                        noise_band: Optional[float] = None,
                        f_rel_tol: Optional[float] = None) -> bool:
    """Does `score` clear the bar vs the current champion's `champ_score`?

    No champion yet (None / negative) always promotes.  Under "weighted_geomean" this is a
    match-or-improve test (`score >= champ_F * (1 - f_rel_tol)`) so an equal-score refactor lands;
    under "min" it is the legacy worst-case ratchet with the noise band.
    """
    objective = objective or C.PROMOTION_OBJECTIVE
    if champ_score is None or champ_score < 0:
        return True
    if objective == "weighted_geomean":
        tol = C.PROMOTION_F_REL_TOL if f_rel_tol is None else f_rel_tol
        return score >= champ_score * (1.0 - tol)
    band = C.PERF_NOISE_BAND if noise_band is None else noise_band
    return score >= champ_score - band


def next_champ_score(prev: Optional[float], score: float,
                     objective: Optional[str] = None) -> float:
    """Champion score after a promotion.  "min": monotone max() ratchet (never drops the bar).
    "weighted_geomean": NON-monotone -- track the CURRENT champion's F so an equal/lower-but-within-
    tolerance refactor isn't forever blocked by a past peak."""
    objective = objective or C.PROMOTION_OBJECTIVE
    if objective == "weighted_geomean":
        return score
    return score if prev is None else max(prev, score)


def replay_keep_set(records: list, objective: Optional[str] = None,
                    noise_band: Optional[float] = None,
                    f_rel_tol: Optional[float] = None) -> list:
    """Offline replay of the promotion rule over a sequence of step records.

    `records`: an ordered list of dicts each with `correct` (bool) and the eval fields
    (`min_ratio`, `weighted_geomean_ratio`).  Returns the list of indices that WOULD be promoted
    (the keep set), applying the same correct + meets-bar rule the orchestrator uses -- but
    WITHOUT the cost-tier veto gates (cuda-graph / in-situ / decode-equiv), which can only demote.
    """
    objective = objective or C.PROMOTION_OBJECTIVE
    champ: Optional[float] = None
    kept: list = []
    for i, r in enumerate(records):
        if not r.get("correct"):
            continue
        s = objective_score(r, objective)
        if meets_promotion_bar(s, champ, objective, noise_band, f_rel_tol):
            kept.append(i)
            champ = next_champ_score(champ, s, objective)
    return kept
