"""Roll a run back to an earlier completed step (`run --resume --from-step N`).

`--from-step N` controls where a resume continues:

  * not passed      -> resume from the last completed step (default behaviour);
  * N >= last step  -> no-op: you cannot resume from a *future* step;
  * N <  last step  -> ROLL BACK to step N: keep steps <= N, MOVE every artifact for the
                       rolled-back steps (> N) into one timestamped `rolled_back/<...>/` subfolder
                       of the run dir, then resume at N+1 from the champion as of step N.

Everything for the discarded steps is preserved in the durable CHECKPOINT archive (under
`rolled_back/<...>/`) but DELETED OUTRIGHT from the live workdir: per-step build dirs, candidate
sources, the removed `candidates.jsonl` tail, the run.log tail, the evolution plot, the step>N
knowledge notes, the transient profiling reports, and the brain's per-run `.claude`.  This matters
because the brain gets the workdir via ``--add-dir`` -- if the rolled-back code lived anywhere
under the workdir the brain could (and did) read it and resurrect a discarded trajectory.  Keeping
the only copy in CHECKPOINT_ROOT (which the brain is NEVER granted) makes it inspectable to the
user while invisible to the agent.

The kept cross-step state is rolled back to its step-N value in BOTH dirs: the lever ledger is
replayed over its <=N outcome history, branch state is reset (resume rebuilds the champion series
from the truncated lineage), cross-step memory is rebuilt to point at the step-N champion (lessons
+ dead-ends cleared), and `best.cu` is restored to the champion's source.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from typing import Callable, Optional

import yaml

from . import checkpoint, config as C

# A run.log step banner, e.g. "[fa3agent 21:30:11] === STEP 21 (step021, ns=...) ==="
_STEP_LINE_RE = re.compile(r"=== STEP (\d+) ")

# Cross-step state files rolled back in place (snapshotted into the subfolder first).
_STATE_FILES = ("memory_state.json", "MEMORY.md", "lever_state.json", "branches.json",
                "rewrite_journal.md", "best.cu")


# --------------------------------------------------------------------------------------
# Pure planner (no side effects; testable on an in-memory step list)
# --------------------------------------------------------------------------------------
def compute_plan(steps: list, target_step: int) -> dict:
    if not steps:
        return {"ok": False, "msg": "run has no completed steps to roll back"}
    if target_step < 0:
        return {"ok": False, "msg": f"target step must be >= 0 (got {target_step})"}
    last = max(s.get("step", -1) for s in steps)
    kept = [s for s in steps if s.get("step", 10**9) <= target_step]
    removed = [s for s in steps if s.get("step", -1) > target_step]
    if not kept:
        earliest = min(s.get("step", 0) for s in steps)
        return {"ok": False, "msg": f"no steps <= {target_step} (earliest completed step is {earliest})"}
    if not removed:
        return {"ok": True, "noop": True, "last": last, "target": target_step,
                "msg": f"last completed step is {last} <= target {target_step}; nothing to roll back"}
    champ = None
    for s in kept:
        if s.get("is_best"):
            champ = s
    return {
        "ok": True, "noop": False, "last": last, "target": target_step,
        "kept": kept, "removed": removed,
        "removed_steps": sorted(s.get("step") for s in removed),
        "champ": champ, "champ_cid": (champ or {}).get("cid"),
    }


# --------------------------------------------------------------------------------------
# CLI-facing helpers
# --------------------------------------------------------------------------------------
def ensure_workdir(run_id: str) -> str:
    """Make sure the LIVE workdir exists (restore from the durable archive on a cold resume)."""
    work = checkpoint.paths(run_id)["run"]
    archive = C.archive_dir(run_id)
    if not os.path.isdir(work) and os.path.isdir(archive):
        try:
            # exclude rolled_back/ -- the discarded data lives only in the archive, never the workdir
            shutil.copytree(archive, work, ignore=shutil.ignore_patterns("rolled_back"))
        except Exception as e:  # noqa: BLE001
            print(f"[rollback] WARN could not restore workdir from archive: {e!r}")
    return work


def last_completed_step(run_id: str) -> Optional[int]:
    steps = checkpoint.load_steps(run_id)
    if not steps:
        return None
    return max(s.get("step", -1) for s in steps)


def purge_incomplete_tail(run_id: str, log: Callable = print) -> dict:
    """Before a plain resume, DELETE the leftover on-disk artifacts of the not-yet-completed step.

    A resume re-runs step ``start = last_completed_step + 1``.  If a prior attempt at that step was
    started but killed before it was persisted to candidates.jsonl (interrupted mid-build / mid-eval),
    its workdir artifacts are stale and MUST go before the step re-runs -- most importantly the torch
    ``FileBaton`` ``lock`` left under ``builds/stepNNN/``: the rebuild would otherwise wait forever on
    a lock no live process holds and hang the whole run.  Cleared (disk-driven, so orphans too):

      * ``builds/stepNNN/`` for NNN >= start (the stale build tree incl. the FileBaton ``lock``);
      * ``candidates/stepNNN.cu`` for NNN >= start (the half-written candidate source);
      * the ``run.log`` tail from the first ``=== STEP {start}`` banner on (the killed attempt's logs);
      * ``knowledge/*.md`` notes authored for step >= start.

    Unlike a rollback this DELETES outright (no archive copy) and does NOT touch the cross-step state
    (lever / branch / memory / best.cu): those are only updated once a step is PERSISTED, so as of step
    ``start-1`` they are already correct and the discarded attempt never mutated them.  No-op when the
    tail is already clean (a fresh run, or a cold resume restored from the archive)."""
    work = checkpoint.paths(run_id)["run"]
    if not os.path.isdir(work):
        return {"purged": 0, "start": None}
    last = last_completed_step(run_id)
    start = 0 if last is None else last + 1
    N = start - 1                       # keep step <= N; purge step >= start
    discard = lambda rel: _purge(work, rel, log)  # noqa: E731

    nb = _sweep_step_artifacts(work, "builds", N, discard)
    nc = _sweep_step_artifacts(work, "candidates", N, discard)

    # run.log: drop the tail from the first step>=start banner (the killed attempt's logs).
    nlog = 0
    runlog = os.path.join(work, "run.log")
    if os.path.exists(runlog):
        with open(runlog, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        cut = None
        for i, ln in enumerate(lines):
            m = _STEP_LINE_RE.search(ln)
            if m and int(m.group(1)) > N:
                cut = i
                break
        if cut is not None:
            nlog = len(lines) - cut
            with open(runlog, "w", encoding="utf-8", errors="replace") as fh:
                fh.writelines(lines[:cut])
                fh.write(f"[resume] --- discarded the incomplete step {start} attempt "
                         f"(stale build/lock/candidate/log) before re-running it ---\n")

    # knowledge notes authored for step >= start.
    nk = 0
    kdir = os.path.join(work, "knowledge")
    if os.path.isdir(kdir):
        for fn in list(os.listdir(kdir)):
            if not fn.endswith(".md"):
                continue
            st = _note_step(os.path.join(kdir, fn))
            if st is not None and st > N:
                nk += bool(_purge(work, os.path.join("knowledge", fn), log))

    if nb or nc or nk or nlog:
        log(f"[resume] purged incomplete step {start} tail: {nb} build dir(s), {nc} candidate(s), "
            f"{nk} knowledge note(s){', run.log tail' if nlog else ''} (deleted, not archived)")
    return {"purged": nb + nc + nk, "start": start, "runlog_lines": nlog}


def plan(run_id: str, from_step: int) -> dict:
    """CLI-shaped summary of what a rollback to `from_step` discards (no side effects)."""
    cp = compute_plan(checkpoint.load_steps(run_id), from_step)
    removed = cp.get("removed_steps", []) or []
    return {
        "removed": removed,
        "n_removed": len(removed),
        "champion_cid": cp.get("champ_cid"),
        "champion_step": (cp.get("champ") or {}).get("step"),
    }


def rollback(run_id: str, from_step: int, log: Callable = print) -> dict:
    """Roll a run back to `from_step`: PRESERVE the discarded artifacts in the durable archive and
    DELETE them from the live workdir (so the brain, which sees the workdir, can't read them).
    Returns {"subfolder": ..., "archived_in": ...}."""
    steps = checkpoint.load_steps(run_id)
    p = compute_plan(steps, from_step)
    if not p.get("ok") or p.get("noop"):
        return {"subfolder": None}
    sub = os.path.join("rolled_back", f"step{from_step}_{time.strftime('%Y%m%d_%H%M%S')}")

    # Replay the lever ledger once (live + archive share the lineage) and reuse for both dirs.
    live = checkpoint.paths(run_id)["run"]
    archive = C.archive_dir(run_id)
    archive_sink = os.path.join(archive, sub)
    hist = []
    lv_path = os.path.join(live, "lever_state.json")
    if os.path.exists(lv_path):
        try:
            hist = (json.load(open(lv_path)) or {}).get("history", [])
        except Exception:  # noqa: BLE001
            hist = []
    lever_state = _rebuild_lever_state(hist, from_step)

    # 1. Archive FIRST: stow the archive's own step>N data under archive/rolled_back/<sub>/ and roll
    #    its top-level state back to step N (so the durable copy keeps both the discarded data and a
    #    clean <=N tip).
    if os.path.isdir(archive):
        _apply_to_dir(archive, p, sub, lever_state, log=log)
    # 2. Workdir: preserve any workdir-only extras (orphan partials, the brain's .claude) into the
    #    archive sink if missing, then DELETE every discarded artifact from the workdir -- no local
    #    rolled_back/ folder is created, so nothing the brain can reach survives.
    if os.path.isdir(live):
        _apply_to_dir(live, p, sub, lever_state, log=log, delete=True, sink=archive_sink)
    return {"subfolder": sub, "archived_in": archive_sink}


def apply(run_id: str, target_step: int, assume_yes: bool = False, log: Callable = print) -> bool:
    """High-level entry used by the CLI: plan -> (confirm) -> dual workdir+archive surgery -> report.

    Returns True if the caller should PROCEED to resume the run (a target >= the last completed step
    is a no-op and resumes normally), False if the rollback was rejected/aborted (do NOT start)."""
    ensure_workdir(run_id)
    p = compute_plan(checkpoint.load_steps(run_id), target_step)
    if not p.get("ok"):
        log(f"[rollback] cannot roll back {run_id} to step {target_step}: {p.get('msg')}")
        return False
    if p.get("noop"):
        log(f"[rollback] {p.get('msg')}; resuming {run_id} normally")
        return True

    champ_step = (p.get("champ") or {}).get("step")
    archive_sink = os.path.join(C.archive_dir(run_id), "rolled_back")
    log(f"[rollback] {run_id}: keep steps <= {target_step} "
        f"(champion {p['champ_cid']} @ step {champ_step}); "
        f"discard {len(p['removed_steps'])} step(s): {p['removed_steps']}")
    log(f"[rollback]   builds/candidates/run.log/plots/knowledge/reports/.claude for the discarded "
        f"steps are PRESERVED in the checkpoint archive ({archive_sink}/) and DELETED from the "
        f"workdir; memory/lever/branch state roll back to step {target_step}; next step = "
        f"{target_step + 1}")
    if not assume_yes:
        if sys.stdin and sys.stdin.isatty():
            ans = input(f"Proceed with rollback of {run_id} to step {target_step}? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                log("[rollback] aborted; nothing changed")
                return False
        else:
            log("[rollback] refusing to roll back non-interactively; re-run with --yes to confirm")
            return False

    info = rollback(run_id, target_step, log=log)
    log(f"[rollback] done; resume {run_id} at step {target_step + 1} "
        f"(archived under {info.get('subfolder')}/)")
    return True


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------
def _move(base: str, rel: str, sub: str, log: Callable) -> bool:
    """Move base/rel -> base/<sub>/rel (creating parents).  Returns True if something moved."""
    src = os.path.join(base, rel)
    if not os.path.exists(src):
        return False
    dst = os.path.join(base, sub, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.exists(dst):
            dst = f"{dst}.{int(time.time()*1000) % 100000}"
        shutil.move(src, dst)
        return True
    except Exception as e:  # noqa: BLE001
        log(f"[rollback]   WARN could not move {rel}: {e!r}")
        return False


def _copy(base: str, rel: str, sub: str) -> None:
    """Snapshot base/rel -> base/<sub>/rel (keep the live file)."""
    src = os.path.join(base, rel)
    if not os.path.exists(src):
        return
    dst = os.path.join(base, sub, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    except Exception:  # noqa: BLE001
        pass


def _preserve_to(src: str, dst: str) -> None:
    """Copy src -> dst (file or dir) only if dst is absent (don't clobber the archived copy)."""
    if not os.path.exists(src) or os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    except Exception:  # noqa: BLE001
        pass


def _evict(base: str, rel: str, sink: str, log: Callable) -> bool:
    """Preserve base/rel into sink/rel (copy if absent), then DELETE base/rel.  For the workdir, whose
    only durable copy must live in the external (checkpoint) sink, never under the workdir itself."""
    src = os.path.join(base, rel)
    if not os.path.exists(src):
        return False
    _preserve_to(src, os.path.join(sink, rel))
    try:
        if os.path.isdir(src):
            shutil.rmtree(src, ignore_errors=True)
        else:
            os.remove(src)
    except OSError as e:
        log(f"[rollback]   WARN could not delete {rel}: {e!r}")
    return True


def _purge(base: str, rel: str, log: Callable) -> bool:
    """DELETE base/rel outright (no archive copy).  Used by the resume-time tail cleanup, which
    discards the leftovers of a never-persisted step instead of preserving them like a rollback."""
    src = os.path.join(base, rel)
    if not os.path.exists(src):
        return False
    try:
        if os.path.isdir(src):
            shutil.rmtree(src, ignore_errors=True)
        else:
            os.remove(src)
    except OSError as e:
        log(f"[resume]   WARN could not delete {rel}: {e!r}")
    return True


_STEP_ARTIFACT_RE = re.compile(r"^step(\d+)(?:\.\w+)?$")


def _sweep_step_artifacts(base: str, subdir: str, target_step: int, discard: Callable) -> int:
    """Discard every base/<subdir>/stepNNN[.ext] whose step number > target_step.

    Disk-driven (not lineage-driven) so it also sweeps ORPHANS -- a candidate from a generation that
    was interrupted before it was recorded, or stale dirs left by an older/aborted trajectory -- not
    just the step records still present in candidates.jsonl.  ``discard(rel)`` stows the artifact
    (locally for the archive, or into the external sink + delete for the workdir)."""
    d = os.path.join(base, subdir)
    if not os.path.isdir(d):
        return 0
    moved = 0
    for name in sorted(os.listdir(d)):
        m = _STEP_ARTIFACT_RE.match(name)
        if m and int(m.group(1)) > target_step:
            moved += bool(discard(os.path.join(subdir, name)))
    return moved


def _note_step(path: str) -> Optional[int]:
    """The `step` frontmatter value of a knowledge note (.md), or None."""
    try:
        with open(path, encoding="utf-8") as fh:
            head = fh.read(4096)
        m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", head, re.DOTALL)
        if not m:
            return None
        fm = yaml.safe_load(m.group(1))
        if isinstance(fm, dict) and isinstance(fm.get("step"), int):
            return fm["step"]
    except Exception:  # noqa: BLE001
        pass
    return None


def _rebuild_lever_state(history: list, target_step: int) -> dict:
    """Replay the lever outcome history up to and including `target_step` -> the step-N ledger.

    Reuses LeverController.record_outcome so counters/exhaustion exactly match the step-N ledger."""
    from .levers import LeverController
    tmpd = tempfile.mkdtemp(prefix="fa3_lever_replay_")
    try:
        lc = LeverController(tmpd)
        for h in history or []:
            if not h or h[0] is None or h[0] > target_step:
                continue
            step, lid = h[0], h[1]
            outcome = h[3] if len(h) > 3 else None
            if outcome:
                lc.record_outcome(lid, outcome, step=step)
        # A rollback discards the later trajectory that "exhausted" a prerequisite capability
        # (wgmma / TMA / fp8-wgmma).  Re-open those foundation levers so the resumed run can attempt
        # the structural change again instead of inheriting a permanent lock-out from the rolled-back
        # steps.  Tactic levers keep their replayed state.
        reopened = lc.reset_exhaustion(foundation_only=True)
        if reopened:
            lc.state.setdefault("_rollback_reopened", reopened)
        return lc.state
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def _apply_to_dir(base: str, p: dict, sub: str, lever_state: dict, log: Callable = print,
                  delete: bool = False, sink: Optional[str] = None) -> None:
    """Apply the rollback to a single run directory.

    delete=False (the ARCHIVE): discarded artifacts are MOVED into base/<sub>/ (kept locally,
    durable, inspectable) and old state files are snapshotted there before being reset.
    delete=True (the WORKDIR): discarded artifacts are PRESERVED into the external `sink` (the
    archive's rolled_back/<sub>/) only if not already there, then DELETED from base -- no base/<sub>/
    is created, so nothing reachable by the brain survives under the workdir.

    Operates purely on `base` + the precomputed plan `p` / `lever_state` (no WORK_ROOT/
    CHECKPOINT_ROOT config or GPU dependency) so it is unit-testable.
    """
    if not os.path.isdir(base):
        return
    N = p["target"]
    champ_cid = p.get("champ_cid")
    champ = p.get("champ") or {}
    sink = sink or os.path.join(base, sub)

    def discard(rel: str) -> bool:
        """Remove base/rel; archive mode stows it under base/<sub>/, workdir mode evicts to sink."""
        return _evict(base, rel, sink, log) if delete else _move(base, rel, sub, log)

    def preserve_state(rel: str) -> None:
        """Preserve a state file's pre-rollback value before it is reset in place."""
        if delete:
            _preserve_to(os.path.join(base, rel), os.path.join(sink, rel))
        else:
            _copy(base, rel, sub)

    if not delete:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    log(f"[rollback] {base} -> keep step<={N}; "
        + (f"DELETE step>{N} (preserved in {sink}/)" if delete else f"stow step>{N} under {sub}/"))

    # 1. lineage: keep step<=N in place; preserve the step>N tail, drop it from the live file.
    jsonl = os.path.join(base, "candidates.jsonl")
    if os.path.exists(jsonl):
        kept_lines, removed_lines = [], []
        with open(jsonl) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    step = json.loads(line).get("step", 10**9)
                except json.JSONDecodeError:
                    kept_lines.append(line)
                    continue
                (kept_lines if step <= N else removed_lines).append(line)
        if removed_lines:
            tail = os.path.join(sink if delete else os.path.join(base, sub), "candidates.jsonl")
            if not (delete and os.path.exists(tail)):
                os.makedirs(os.path.dirname(tail), exist_ok=True)
                with open(tail, "w") as fh:
                    fh.write("\n".join(removed_lines) + "\n")
        tmp = jsonl + ".tmp"
        with open(tmp, "w") as fh:
            fh.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))
        os.replace(tmp, jsonl)
        log(f"[rollback]   lineage: kept {len(kept_lines)}, dropped {len(removed_lines)}")

    # 2. per-step build dirs + candidate sources for EVERY step > N on disk (disk-driven, not just
    #    the records still in the lineage) -- catches orphans from an interrupted generation or an
    #    older/aborted trajectory.
    nb = _sweep_step_artifacts(base, "builds", N, discard)
    nc = _sweep_step_artifacts(base, "candidates", N, discard)
    if nb or nc:
        log(f"[rollback]   {'deleted' if delete else 'stowed'} {nb} build dir(s) + {nc} "
            f"candidate source(s) for step>{N}")

    # 3. run.log: preserve the full log, then truncate the live one at the first step>N banner.
    runlog = os.path.join(base, "run.log")
    if os.path.exists(runlog):
        if delete:
            _preserve_to(runlog, os.path.join(sink, "run.log"))
        else:
            _copy(base, "run.log", sub)
        with open(runlog, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        cut = None
        for i, ln in enumerate(lines):
            m = _STEP_LINE_RE.search(ln)
            if m and int(m.group(1)) > N:
                cut = i
                break
        if cut is not None:
            with open(runlog, "w", encoding="utf-8", errors="replace") as fh:
                fh.writelines(lines[:cut])
                fh.write(f"[rollback] --- rolled back to step {N}; later steps preserved in the "
                         f"checkpoint archive ---\n")
            log(f"[rollback]   run.log truncated at step {N+1}")

    # 4. evolution plot(s) (regenerated on the next step from the truncated lineage).
    plots = os.path.join(base, "plots")
    if os.path.isdir(plots):
        for fn in list(os.listdir(plots)):
            if fn.endswith((".png", ".svg", ".pdf")):
                discard(os.path.join("plots", fn))

    # 5. knowledge notes authored after the target step.
    kdir = os.path.join(base, "knowledge")
    if os.path.isdir(kdir):
        moved = 0
        for fn in list(os.listdir(kdir)):
            if not fn.endswith(".md"):
                continue
            st = _note_step(os.path.join(kdir, fn))
            if st is not None and st > N:
                moved += bool(discard(os.path.join("knowledge", fn)))
        if moved:
            log(f"[rollback]   {'deleted' if delete else 'stowed'} {moved} step>{N} knowledge note(s)")

    # 6. transient profiling reports (no per-step granularity) -> discard wholesale, recreate empty.
    if os.path.isdir(os.path.join(base, "reports")):
        discard("reports")
        os.makedirs(os.path.join(base, "reports"), exist_ok=True)

    # 7. preserve the current cross-step state, then roll each file back to its step-N value.
    for rel in _STATE_FILES:
        preserve_state(rel)

    # 7a. lever ledger -> step-N replay (precomputed).
    with open(os.path.join(base, "lever_state.json"), "w") as fh:
        json.dump(lever_state, fh, indent=2)

    # 7b. branch state -> default (resume's rebuild_from_steps refills champ_history from <=N).
    with open(os.path.join(base, "branches.json"), "w") as fh:
        json.dump({"champ_history": [], "rejects": 0, "branch": None, "branch_log": []}, fh, indent=2)

    # 7c. rewrite journal -> reset.
    with open(os.path.join(base, "rewrite_journal.md"), "w") as fh:
        fh.write(f"# Rewrite journal (reset on rollback to step {N})\n")

    # 7d. cross-step memory -> step-N champion (lessons + dead-ends cleared: they were authored
    #     against the post-N trajectory we just discarded).
    from .levers import LeverController
    from .memory import MemoryStore
    ms = MemoryStore(base)
    ms.state["lessons"] = []
    ms.state["dead_ends"] = []
    ms.state["branch"] = None
    if champ_cid:
        ev = champ.get("eval") or {}
        bn = (champ.get("profile") or {}).get("bottleneck")
        ms.set_champion(champ_cid, champ.get("step", N), ev.get("weighted_geomean_ratio", 0.0),
                        ev.get("min_ratio", 0.0), bottleneck=bn)
    else:
        ms.state["champion"] = None
    ms.set_lever_ledger(LeverController(base).ledger_summary())  # reads the lever_state.json above
    ms.commit()

    # 7e. best.cu -> the champion's source.
    if champ_cid:
        csrc = os.path.join(base, "candidates", f"{champ_cid}.cu")
        if os.path.exists(csrc):
            shutil.copyfile(csrc, os.path.join(base, "best.cu"))
            log(f"[rollback]   best.cu <- candidates/{champ_cid}.cu")

    # 8. the brain's per-run Claude memory (so the resumed brain starts clean from step N;
    #    re-seeded on the next orchestrator init).  Workdir-only, so the workdir pass is what
    #    actually preserves it into the checkpoint sink before deleting it.
    discard(".claude")

    # 9. audit trail in meta.json.
    meta_p = os.path.join(base, "meta.json")
    try:
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        meta.setdefault("rollbacks", []).append(
            {"to_step": N, "archived_to": sub, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        with open(meta_p, "w") as fh:
            json.dump(meta, fh, indent=2)
    except Exception:  # noqa: BLE001
        pass
