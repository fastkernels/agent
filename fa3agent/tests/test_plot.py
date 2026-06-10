"""Smoke test for fa3agent.plot.evolution.update_plot.

Synthesizes a small candidates.jsonl (~4 fake steps with varying min/geomean ratios,
per-config ratios, one new-best, one incorrect candidate with ratio 0) and asserts the
rendered output file exists and is non-empty.  Also exercises the missing-file and
partial-record robustness paths.

Run from /data-fast/agent:
    python -m fa3agent.tests.test_plot
"""
from __future__ import annotations

import json
import os
import tempfile

from dataclasses import asdict

from fa3agent.common import ConfigResult, EvalResult, StepRecord
from fa3agent.plot.evolution import update_plot


def _mk_record(step, cid, status, min_r, geo_r, per_cfg, is_best):
    """per_cfg: dict[config_name -> ratio]."""
    pcs = [
        ConfigResult(config=name, correct=(ratio > 0), ratio=ratio)
        for name, ratio in per_cfg.items()
    ]
    ev = EvalResult(
        correct=(status in ("correct", "promoted")),
        num_configs=len(pcs),
        num_correct=sum(1 for p in pcs if p.correct),
        min_ratio=min_r,
        geomean_ratio=geo_r,
        per_config=pcs,
    )
    return StepRecord(
        step=step,
        cid=cid,
        parent=(f"step{step-1:03d}" if step > 0 else None),
        status=status,
        eval=asdict(ev),
        is_best=is_best,
    )


def _write_jsonl(path, records):
    with open(path, "w") as fh:
        for r in records:
            fh.write(r.to_json() + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as run_dir:
        jsonl = os.path.join(run_dir, "candidates.jsonl")
        records = [
            # step 0: seed, modest speedup, new best
            _mk_record(0, "step000", "correct", 0.95, 1.02,
                       {"dev_decode": 1.05, "dev_prefill": 0.99}, is_best=True),
            # step 1: incorrect candidate -> all ratios 0 (regression to floor)
            _mk_record(1, "step001", "incorrect", 0.0, 0.0,
                       {"dev_decode": 0.0, "dev_prefill": 0.0}, is_best=False),
            # step 2: improvement -> new best
            _mk_record(2, "step002", "promoted", 1.04, 1.11,
                       {"dev_decode": 1.18, "dev_prefill": 1.04}, is_best=True),
            # step 3: slight regression vs best (not a new best)
            _mk_record(3, "step003", "correct", 1.01, 1.08,
                       {"dev_decode": 1.12, "dev_prefill": 1.01}, is_best=False),
        ]
        _write_jsonl(jsonl, records)

        # 1) default output path under run_dir/plots/
        out = update_plot(run_dir, jsonl)
        assert os.path.exists(out), f"output not created: {out}"
        size = os.path.getsize(out)
        assert size > 0, f"output empty: {out}"
        print(f"[ok] rendered {out} ({size} bytes)")

        # 2) explicit out path
        out2 = os.path.join(run_dir, "custom", "evo.png")
        r2 = update_plot(run_dir, jsonl, out_png=out2)
        assert os.path.exists(r2) and os.path.getsize(r2) > 0, f"explicit out failed: {r2}"
        print(f"[ok] rendered explicit {r2} ({os.path.getsize(r2)} bytes)")

        # 3) robustness: missing candidates file should still render (empty plot)
        missing = os.path.join(run_dir, "does_not_exist.jsonl")
        out3 = os.path.join(run_dir, "empty", "evo.png")
        r3 = update_plot(run_dir, missing, out_png=out3)
        assert os.path.exists(r3) and os.path.getsize(r3) > 0, f"missing-file path failed: {r3}"
        print(f"[ok] rendered empty/missing-input {r3} ({os.path.getsize(r3)} bytes)")

        # 4) robustness: partial / malformed lines are skipped, valid ones kept
        partial = os.path.join(run_dir, "partial.jsonl")
        with open(partial, "w") as fh:
            fh.write("not json at all\n")
            fh.write("\n")
            fh.write(json.dumps({"step": 0, "cid": "p0", "status": "correct"}) + "\n")  # no eval
            fh.write(json.dumps({"step": 1, "cid": "p1", "status": "correct",
                                 "eval": {"min_ratio": 1.0, "geomean_ratio": 1.0,
                                          "per_config": [{"config": "c", "ratio": 1.2},
                                                         "garbage"]}, "is_best": True}) + "\n")
        out4 = os.path.join(run_dir, "partial", "evo.png")
        r4 = update_plot(run_dir, partial, out_png=out4)
        assert os.path.exists(r4) and os.path.getsize(r4) > 0, f"partial path failed: {r4}"
        print(f"[ok] rendered partial-input {r4} ({os.path.getsize(r4)} bytes)")

    print("PASS: test_plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
