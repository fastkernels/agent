"""Per-step checkpointing + resume.

A run lives entirely under CHECKPOINT_ROOT/<run_id>/ so it is self-contained and resumable
after an interruption.  candidates.jsonl is the append-only evolution DAG + lineage (one
StepRecord per line).  After every step the orchestrator calls append_step(); resume reads
the latest completed step and the current best candidate.

Layout:
  <run_id>/
    meta.json                 run config (config set, model, max_steps, ...)
    candidates.jsonl          one StepRecord per step (the lineage)
    candidates/stepNNN.cu     candidate source per step
    builds/stepNNN/           per-step dir: .so + brain.log + prompt.log + per-tool <tool>.log call logs
    reports/                  ncu/sass reports
    knowledge/                typed knowledge store (notes)
    plots/evolution.png       AVO-style evolution plot
    best.cu                   current champion source
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from typing import Optional

from . import config as C
from .common import StepRecord


def _run_dir(run_id: str) -> str:
    return os.path.join(C.CHECKPOINT_ROOT, run_id)


def paths(run_id: str) -> dict:
    rd = _run_dir(run_id)
    return dict(
        run=rd,
        meta=os.path.join(rd, "meta.json"),
        jsonl=os.path.join(rd, "candidates.jsonl"),
        candidates=os.path.join(rd, "candidates"),
        builds=os.path.join(rd, "builds"),
        reports=os.path.join(rd, "reports"),
        knowledge=os.path.join(rd, "knowledge"),
        plots=os.path.join(rd, "plots"),
        best=os.path.join(rd, "best.cu"),
    )


def init_run(run_id: str, meta: dict) -> dict:
    p = paths(run_id)
    for d in ("run", "candidates", "builds", "reports", "knowledge", "plots"):
        os.makedirs(p[d], exist_ok=True)
    # don't clobber meta on resume; merge
    if os.path.exists(p["meta"]):
        try:
            old = json.load(open(p["meta"]))
            old.update({k: v for k, v in meta.items() if k not in old})
            meta = old
        except Exception:
            pass
    with open(p["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    return p


def save_candidate_cu(run_id: str, cid: str, source: str) -> str:
    p = paths(run_id)
    dst = os.path.join(p["candidates"], f"{cid}.cu")
    with open(dst, "w") as f:
        f.write(source)
    return dst


def build_dir(run_id: str, cid: str) -> str:
    d = os.path.join(paths(run_id)["builds"], cid)
    os.makedirs(d, exist_ok=True)
    return d


def append_step(run_id: str, rec: StepRecord) -> None:
    p = paths(run_id)
    with open(p["jsonl"], "a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")


def load_steps(run_id: str) -> list[dict]:
    p = paths(run_id)
    if not os.path.exists(p["jsonl"]):
        return []
    out = []
    with open(p["jsonl"]) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def latest_step(run_id: str) -> Optional[dict]:
    steps = load_steps(run_id)
    return steps[-1] if steps else None


def best_record(run_id: str) -> Optional[dict]:
    """Return the most recent promoted (is_best) step record, if any."""
    best = None
    for s in load_steps(run_id):
        if s.get("is_best"):
            best = s
    return best


def promote_best(run_id: str, cid: str) -> None:
    """Copy the candidate source to best.cu (the current champion)."""
    p = paths(run_id)
    src = os.path.join(p["candidates"], f"{cid}.cu")
    if os.path.exists(src):
        shutil.copyfile(src, p["best"])
