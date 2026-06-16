"""Offline replay test for C1 (promotion objective).

Run from /data-fast/agent (no GPU/torch/claude):
    python -m fa3agent.tests.test_promotion

Covers:
  - weighted_geomean handles FRACTIONAL weights correctly (the legacy int()-duplication both
    truncated fractional weights and dropped weight<1 configs);
  - replay_keep_set over a synthetic eval series: the "min" objective REJECTS a refactor that
    improves the overall (weighted-geomean) kernel but dips the single worst config, while
    "weighted_geomean" LANDS it (the whole point of C1);
  - the non-monotone base lets a near-equal-F refactor land under "weighted_geomean";
  - (best-effort) replays a real archived run's candidates.jsonl under "min" without error.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

from fa3agent import config as C
from fa3agent import promotion as P


def _ok(msg):
    print(f"  PASS: {msg}")


def _fail(msg):
    print(f"  FAIL: {msg}")
    return 1


def _rec(correct, mn, f):
    return {"correct": correct, "min_ratio": mn, "weighted_geomean_ratio": f}


def main() -> int:
    failures = 0

    # ---- (1) weighted_geomean math (fractional weights) ----
    print("[1] weighted_geomean fractional-weight correctness")
    got = P.weighted_geomean([2.0, 0.5], [1.5, 0.5])
    want = math.exp((1.5 * math.log(2.0) + 0.5 * math.log(0.5)) / 2.0)
    naive_int = 2.0  # legacy int()-duplication: range(int(1.5))=1 copy of 2.0, range(int(0.5))=0 -> [2.0]
    if abs(got - want) < 1e-9 and abs(got - naive_int) > 0.4:
        _ok(f"weighted_geomean={got:.4f} (analytic {want:.4f}); legacy int-dup would give {naive_int}")
    else:
        failures += _fail(f"weighted_geomean wrong: got {got}, want {want}, naive {naive_int}")

    # ---- (2) keep set: refactor that dips worst config but improves overall ----
    print("[2] keep set differs: min rejects the overall-better refactor, weighted_geomean lands it")
    series = [
        _rec(True, 0.30, 0.40),   # 0: first correct -> always promoted
        _rec(True, 0.34, 0.45),   # 1: improves both -> promoted under both
        _rec(True, 0.20, 0.50),   # 2: worst config dips, overall (F) up -> min REJECTS, wg PROMOTES
        _rec(True, 0.36, 0.498),  # 3: min recovers; F within 1% of champ -> promoted under both
    ]
    keep_min = P.replay_keep_set(series, "min")
    keep_wg = P.replay_keep_set(series, "weighted_geomean")
    print(f"    keep(min)={keep_min}  keep(weighted_geomean)={keep_wg}")
    if keep_min == [0, 1, 3] and keep_wg == [0, 1, 2, 3]:
        _ok("min keeps [0,1,3]; weighted_geomean funds the dip-then-better refactor [0,1,2,3]")
    else:
        failures += _fail(f"unexpected keep sets: min={keep_min} wg={keep_wg}")

    # ---- (3) incorrect candidates never promote ----
    print("[3] incorrect candidates are never in the keep set")
    series2 = [_rec(False, 0.99, 0.99), _rec(True, 0.10, 0.10), _rec(False, 2.0, 2.0)]
    keep = P.replay_keep_set(series2, "min")
    if keep == [1]:
        _ok("only the correct candidate is kept")
    else:
        failures += _fail(f"incorrect candidate leaked into keep set: {keep}")

    # ---- (4) best-effort: replay a real archived run under "min" ----
    print("[4] best-effort replay of a real run's candidates.jsonl (min objective)")
    jsonls = sorted(glob.glob(os.path.join(C.CHECKPOINT_ROOT, "run_*", "candidates.jsonl")))
    if not jsonls:
        print("    (no archived run found; skipping)")
    else:
        try:
            recs = []
            recorded_best = []
            with open(jsonls[-1]) as fh:
                for i, line in enumerate(l for l in fh if l.strip()):
                    d = json.loads(line)
                    ev = d.get("eval") or {}
                    recs.append({"correct": d.get("status") == "correct",
                                 "min_ratio": ev.get("min_ratio", 0.0),
                                 "weighted_geomean_ratio": ev.get("weighted_geomean_ratio", 0.0)})
                    if d.get("is_best"):
                        recorded_best.append(i)
            keep = P.replay_keep_set(recs, "min")
            print(f"    {os.path.basename(os.path.dirname(jsonls[-1]))}: "
                  f"replayed keep(min)={keep[:12]}{'...' if len(keep) > 12 else ''} "
                  f"recorded is_best={recorded_best[:12]}{'...' if len(recorded_best) > 12 else ''}")
            _ok("real-jsonl replay ran without error")
        except Exception as e:  # noqa: BLE001
            failures += _fail(f"real-jsonl replay raised: {e!r}")

    # ---- (5) relative band REJECTS a tiny-ratio regression (the bug fix) ----
    print("[5] min ratchet: relative band rejects a regression at tiny ratios (abs 0.02 would not)")
    # champ 0.021, candidate 0.015 (28% worst-config regression).  The OLD absolute band 0.02
    # promoted it (0.015 >= 0.021-0.02 = 0.001); the relative band must REJECT it.
    series3 = [_rec(True, 0.021, 0.10), _rec(True, 0.015, 0.10)]
    keep_rel = P.replay_keep_set(series3, "min")                       # default relative band
    keep_absloose = P.replay_keep_set(series3, "min", noise_band=0.02)  # old absolute behavior
    if keep_rel == [0] and keep_absloose == [0, 1]:
        _ok("relative band rejects 0.021->0.015; the old 0.02 absolute band would have promoted it")
    else:
        failures += _fail(f"relative band wrong: keep_rel={keep_rel} keep_absloose={keep_absloose}")

    # ---- (6) relative band still absorbs a sub-tolerance dip at high ratios ----
    print("[6] min ratchet: a <3% lateral move still lands at high ratios")
    series4 = [_rec(True, 0.80, 0.80), _rec(True, 0.79, 0.82)]   # 1.25% dip, F up
    keep_hi = P.replay_keep_set(series4, "min")
    if keep_hi == [0, 1]:
        _ok("0.80->0.79 (<3%) lands (band = 3% of 0.80 = 0.024)")
    else:
        failures += _fail(f"high-ratio lateral move wrongly rejected: {keep_hi}")

    print()
    if failures == 0:
        print("SMOKE: PASS (promotion objective)")
    else:
        print(f"SMOKE: FAIL ({failures} failing checks)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
