"""Per-step checkpointing + resume.

A run does its LIVE work under WORK_ROOT/<run_id>/ (scratch, the brain's working set) and is
MIRRORED into CHECKPOINT_ROOT/<run_id>/ (the durable, user-facing archive) after every step.  A
FRESH run wipes WORK_ROOT so the brain can only see the current run; a RESUME restores the live
workdir from the archive first.  candidates.jsonl is the append-only evolution DAG + lineage (one
StepRecord per line).  After every step the orchestrator calls append_step() then
mirror_to_archive(); resume reads the latest completed step and the current best candidate.

Layout (identical under WORK_ROOT live + CHECKPOINT_ROOT archive):
  <run_id>/
    meta.json                 run config (config set, model, max_steps, ...)
    candidates.jsonl          one StepRecord per step (the lineage)
    candidates/stepNNN.cu     candidate source per step
    builds/stepNNN/           per-step dir: .so + brain.log + prompt.log + per-tool <tool>.log call logs
    reports/                  ncu/sass reports
    knowledge/                typed knowledge store (notes)
    plots/evolution.png       AVO-style evolution plot
    best.cu                   current champion source
    MEMORY.md / branches.json / lever_state.json / rewrite_journal.md   cross-step memory + branch state
    .claude/                  brain's isolated per-run Claude config (NOT mirrored to the archive)
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
    """LIVE run dir (scratch workdir under WORK_ROOT)."""
    return os.path.join(C.WORK_ROOT, run_id)


def _archive_dir(run_id: str) -> str:
    """DURABLE archive run dir (under CHECKPOINT_ROOT)."""
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


def init_run(run_id: str, meta: dict, resume: bool = False) -> dict:
    """Prepare the LIVE workdir for a run.

    FRESH run (resume=False): wipe the ENTIRE WORK_ROOT (all prior runs' live dirs are already
    mirrored to the archive) so the brain's working set holds ONLY the current run, then recreate.
    RESUME (resume=True): restore WORK_ROOT/<run_id> from the durable archive if the live copy is
    missing (e.g. a cold resume after a wipe / on a fresh box), and NEVER wipe.
    """
    work = _run_dir(run_id)
    archive = _archive_dir(run_id)
    if resume:
        if not os.path.exists(work) and os.path.isdir(archive):
            try:
                shutil.copytree(archive, work)
            except Exception:  # noqa: BLE001  (best-effort restore; init recreates dirs below)
                pass
    else:
        shutil.rmtree(C.WORK_ROOT, ignore_errors=True)

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


# --------------------------------------------------------------------------------------
# Mirror the live workdir -> durable archive (F3).  Incremental: only new/changed files are
# copied, so per-step mirroring stays cheap.  The brain's per-run Claude config (.claude) and
# volatile ncu scratch are excluded; built .so files are included unless ARCHIVE_INCLUDE_SO is off.
# --------------------------------------------------------------------------------------
_MIRROR_EXCLUDE_DIRS = {".claude", "ncu_home"}


def _is_excluded_dir(name: str) -> bool:
    return name in _MIRROR_EXCLUDE_DIRS or name.startswith("ncu_tmp")


def _needs_copy(src: str, dst: str) -> bool:
    if not os.path.exists(dst):
        return True
    try:
        ss, ds = os.stat(src), os.stat(dst)
    except OSError:
        return True
    return ss.st_size != ds.st_size or ss.st_mtime > ds.st_mtime + 1e-6


def mirror_to_archive(run_id: str, include_so: Optional[bool] = None) -> None:
    """Incrementally copy WORK_ROOT/<run_id> -> CHECKPOINT_ROOT/<run_id> (durable, up-to-date)."""
    if include_so is None:
        include_so = C.ARCHIVE_INCLUDE_SO
    src = _run_dir(run_id)
    dst = _archive_dir(run_id)
    if not os.path.isdir(src):
        return
    try:
        os.makedirs(dst, exist_ok=True)
    except OSError:
        return
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not _is_excluded_dir(d)]
        rel = os.path.relpath(root, src)
        tgt_root = dst if rel == "." else os.path.join(dst, rel)
        for fn in files:
            if not include_so and fn.endswith(".so"):
                continue
            s = os.path.join(root, fn)
            d = os.path.join(tgt_root, fn)
            try:
                if _needs_copy(s, d):
                    os.makedirs(tgt_root, exist_ok=True)
                    shutil.copy2(s, d)
            except OSError:
                pass


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
