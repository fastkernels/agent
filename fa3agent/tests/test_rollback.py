"""Tests for the operator rollback (fa3agent/rollback.py), torch-free / no GPU.

Run from /data-fast/agent:
    python -m fa3agent.tests.test_rollback

Exercises the pure planner (compute_plan), the lever-ledger replay (_rebuild_lever_state) and the
per-directory surgery (_apply_to_dir) on a synthetic run tree -- all operate on explicit data /
paths, so no WORK_ROOT/CHECKPOINT_ROOT config or GPU is needed.  Verifies that a rollback to step N
keeps steps <= N in place and MOVES every discarded-step artifact (build dirs, candidate sources,
lineage tail, run.log tail, plot, step>N knowledge notes, reports, .claude) into one subfolder,
while rebuilding the kept state (lever ledger replayed to step N, branches reset, memory + best.cu
restored to the step-N champion).
"""
from __future__ import annotations

import json
import os
import tempfile

from fa3agent import rollback as RB


def _ok(m):
    print(f"  PASS: {m}")


def _fail(m, fails):
    print(f"  FAIL: {m}")
    fails.append(m)


def _rec(step, is_best=False, status="correct"):
    cid = f"step{step:03d}"
    return {"step": step, "cid": cid, "status": status, "is_best": is_best,
            "eval": {"min_ratio": 0.1 + step * 0.001, "weighted_geomean_ratio": 0.2 + step * 0.001},
            "profile": {"bottleneck": f"bottleneck@{step}"}}


def _knote(step=None) -> str:
    fm = ["---", "id: ks-%04d" % (step or 0), "type: ks", "title: note", "confidence: experimental"]
    if step is not None:
        fm.append(f"step: {step}")
    fm += ["---", "body text"]
    return "\n".join(fm) + "\n"


def _build_run(base: str, last: int, best_steps: set, lever_hist: list):
    for d in ("builds", "candidates", "plots", "knowledge", "reports", ".claude"):
        os.makedirs(os.path.join(base, d), exist_ok=True)
    open(os.path.join(base, "reports", "rep.json"), "w").write("{}")
    open(os.path.join(base, ".claude", "history.jsonl"), "w").write("{}\n")

    recs = [_rec(s, is_best=(s in best_steps)) for s in range(last + 1)]
    with open(os.path.join(base, "candidates.jsonl"), "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in recs) + "\n")
    for s in range(last + 1):
        cid = f"step{s:03d}"
        os.makedirs(os.path.join(base, "builds", cid), exist_ok=True)
        open(os.path.join(base, "builds", cid, "brain.log"), "w").write(f"brain {s}")
        open(os.path.join(base, "candidates", f"{cid}.cu"), "w").write(f"// kernel step {s}\n")
        open(os.path.join(base, "knowledge", f"ks-{s:04d}-note.md"), "w").write(_knote(step=s))
    open(os.path.join(base, "knowledge", "ks-seed-note.md"), "w").write(_knote(step=None))  # survives

    with open(os.path.join(base, "run.log"), "w") as fh:
        for s in range(last + 1):
            fh.write(f"[fa3agent 00:00:0{s % 10}] === STEP {s} (step{s:03d}, ns=_ns) ===\n")
            fh.write(f"[fa3agent 00:00:0{s % 10}] [eval] step {s} ok\n")
    open(os.path.join(base, "plots", "evolution.png"), "wb").write(b"\x89PNG fake")

    json.dump({"levers": {}, "history": lever_hist},
              open(os.path.join(base, "lever_state.json"), "w"))
    json.dump({"champion": {"cid": f"step{last:03d}", "step": last}, "bottleneck": "old",
               "lessons": [{"step": s} for s in range(last + 1)],
               "dead_ends": ["stale dead end"], "lever_ledger": {}, "branch": None},
              open(os.path.join(base, "memory_state.json"), "w"))
    json.dump({"champ_history": [{"step": last}], "rejects": 3,
               "branch": {"seed_cid": "step005"}, "branch_log": ["x"]},
              open(os.path.join(base, "branches.json"), "w"))
    open(os.path.join(base, "rewrite_journal.md"), "w").write("branch journal\n")
    open(os.path.join(base, "MEMORY.md"), "w").write("# MEMORY\nchampion: step007\n")
    open(os.path.join(base, "best.cu"), "w").write("// kernel step 7\n")
    json.dump({"run": "test"}, open(os.path.join(base, "meta.json"), "w"))


def main() -> int:
    fails: list = []
    N, LAST = 3, 7
    BEST = {0, 2, 3, 5, 7}            # champion <= N is step003
    # wgmma landed across 0..3, tma only attempted (and would exhaust) at 4..5 (> N)
    HIST = [[s, "wgmma-tensorcore", "structural", "landed"] for s in range(4)]
    HIST += [[4, "tma-multistage-pipeline", "structural", "no_effect"],
             [5, "tma-multistage-pipeline", "structural", "no_effect"]]

    # --- pure planner --------------------------------------------------------------------
    steps = [_rec(s, is_best=(s in BEST)) for s in range(LAST + 1)]
    p = RB.compute_plan(steps, N)
    (_ok("compute_plan: ok, non-noop, champ=step003")
     if (p.get("ok") and not p.get("noop") and p["champ_cid"] == "step003"
         and p["removed_steps"] == [4, 5, 6, 7] and p["last"] == LAST)
     else _fail(f"compute_plan -> {p.get('champ_cid')}, removed={p.get('removed_steps')}", fails))
    noop = RB.compute_plan(steps, LAST)
    _ok("compute_plan: target==last is a noop") if noop.get("noop") else _fail("noop not detected", fails)
    bad = RB.compute_plan(steps, -1)
    _ok("compute_plan: negative target rejected") if not bad.get("ok") else _fail("neg target allowed", fails)

    # --- lever replay --------------------------------------------------------------------
    lever_state = RB._rebuild_lever_state(HIST, N)
    levers = lever_state.get("levers", {})
    wg = levers.get("wgmma-tensorcore", {})
    tma = levers.get("tma-multistage-pipeline", {})
    hist_steps = [h[0] for h in lever_state.get("history", [])]
    (_ok("lever replay: wgmma counted (4x), tma rolled away, history <= N")
     if (wg.get("tried") == 4 and not tma.get("exhausted") and all(s <= N for s in hist_steps))
     else _fail(f"lever replay wg={wg}, tma={tma}, steps={hist_steps}", fails))

    # --- per-directory surgery -----------------------------------------------------------
    with tempfile.TemporaryDirectory() as base:
        _build_run(base, LAST, BEST, HIST)
        p = RB.compute_plan(steps, N)
        sub = os.path.join("rolled_back", "test")
        RB._apply_to_dir(base, p, sub, lever_state, log=lambda m: None)
        arc = os.path.join(base, sub)

        with open(os.path.join(base, "candidates.jsonl")) as fh:
            kept = [json.loads(l)["step"] for l in fh if l.strip()]
        _ok("lineage truncated to <= N") if kept == [0, 1, 2, 3] else _fail(f"lineage={kept}", fails)
        _ok("discarded lineage archived") if os.path.exists(os.path.join(arc, "candidates.jsonl")) \
            else _fail("no archived candidates.jsonl", fails)

        kb = all(os.path.isdir(os.path.join(base, "builds", f"step{s:03d}")) for s in (0, 1, 2, 3))
        mb = all(not os.path.exists(os.path.join(base, "builds", f"step{s:03d}")) for s in (4, 5, 6, 7))
        ab = all(os.path.isdir(os.path.join(arc, "builds", f"step{s:03d}")) for s in (4, 5, 6, 7))
        (_ok("build dirs: <=N kept, >N moved") if (kb and mb and ab)
         else _fail(f"builds kept={kb} moved={mb} arc={ab}", fails))
        kc = all(os.path.exists(os.path.join(base, "candidates", f"step{s:03d}.cu")) for s in (0, 1, 2, 3))
        mc = all(not os.path.exists(os.path.join(base, "candidates", f"step{s:03d}.cu")) for s in (4, 5, 6, 7))
        _ok("candidate sources: <=N kept, >N moved") if (kc and mc) else _fail(f"cands {kc}/{mc}", fails)

        live = open(os.path.join(base, "run.log")).read()
        full = open(os.path.join(arc, "run.log")).read()
        ok_log = ("=== STEP 3 " in live and "=== STEP 4 " not in live
                  and "rolled back to step 3" in live and "=== STEP 7 " in full)
        _ok("run.log truncated live, full copy archived") if ok_log else _fail("run.log split wrong", fails)

        ok_plot = (not os.path.exists(os.path.join(base, "plots", "evolution.png"))
                   and os.path.exists(os.path.join(arc, "plots", "evolution.png")))
        _ok("evolution plot moved") if ok_plot else _fail("plot not moved", fails)

        kn = os.path.join(base, "knowledge")
        kk = all(os.path.exists(os.path.join(kn, f"ks-{s:04d}-note.md")) for s in (0, 1, 2, 3))
        sk = os.path.exists(os.path.join(kn, "ks-seed-note.md"))
        mk = all(not os.path.exists(os.path.join(kn, f"ks-{s:04d}-note.md")) for s in (4, 5, 6, 7))
        _ok("knowledge: >N moved, <=N + seed kept") if (kk and sk and mk) else _fail("knowledge wrong", fails)

        ok_rep = (os.path.isdir(os.path.join(base, "reports"))
                  and not os.listdir(os.path.join(base, "reports"))
                  and os.path.exists(os.path.join(arc, "reports", "rep.json")))
        _ok("reports archived + recreated empty") if ok_rep else _fail("reports wrong", fails)

        ok_claude = (not os.path.exists(os.path.join(base, ".claude"))
                     and os.path.exists(os.path.join(arc, ".claude", "history.jsonl")))
        _ok(".claude moved out") if ok_claude else _fail(".claude not moved", fails)

        best = open(os.path.join(base, "best.cu")).read()
        _ok("best.cu = champion step003") if "step 3" in best else _fail(f"best.cu={best!r}", fails)

        lv = json.load(open(os.path.join(base, "lever_state.json")))
        _ok("lever_state.json rolled back") if "wgmma-tensorcore" in lv.get("levers", {}) \
            and "tma-multistage-pipeline" not in lv.get("levers", {}) else _fail("lever file wrong", fails)

        br = json.load(open(os.path.join(base, "branches.json")))
        ok_br = br["champ_history"] == [] and br["branch"] is None and br["rejects"] == 0
        _ok("branches.json reset") if ok_br else _fail(f"branches={br}", fails)

        mem = json.load(open(os.path.join(base, "memory_state.json")))
        ok_mem = (mem["champion"]["cid"] == "step003" and mem["lessons"] == [] and mem["dead_ends"] == [])
        _ok("memory reset to step-N champion") if ok_mem else _fail(f"memory={mem.get('champion')}", fails)

        ok_arch = all(os.path.exists(os.path.join(arc, f)) for f in
                      ("memory_state.json", "lever_state.json", "branches.json",
                       "rewrite_journal.md", "best.cu", "MEMORY.md"))
        _ok("prior state snapshotted in subfolder") if ok_arch else _fail("state not snapshotted", fails)

        meta = json.load(open(os.path.join(base, "meta.json")))
        ok_meta = meta.get("rollbacks") and meta["rollbacks"][-1]["to_step"] == N
        _ok("meta.json rollback audit recorded") if ok_meta else _fail(f"meta={meta}", fails)

    # --- workdir PURGE mode (delete=True) -------------------------------------------------
    # The discarded artifacts must be DELETED from the workdir (the brain sees it via --add-dir)
    # and PRESERVED in an external checkpoint sink instead -- including workdir-only data (an
    # orphan partial candidate + the brain's .claude) that the archive pass never had.
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as ck:
        _build_run(work, LAST, BEST, HIST)  # _build_run already creates .claude/history.jsonl
        os.makedirs(os.path.join(work, "builds", "step099"), exist_ok=True)        # orphan, not in lineage
        open(os.path.join(work, "candidates", "step099.cu"), "w").write("// orphan\n")
        p = RB.compute_plan(steps, N)
        sub = os.path.join("rolled_back", "test")
        sink = os.path.join(ck, sub)
        RB._apply_to_dir(work, p, sub, lever_state, log=lambda m: None, delete=True, sink=sink)

        no_local = not os.path.exists(os.path.join(work, sub))
        no_claude = not os.path.exists(os.path.join(work, ".claude"))
        gone = all(not os.path.exists(os.path.join(work, "builds", f"step{s:03d}")) for s in (4, 5, 6, 7, 99))
        kept = all(os.path.isdir(os.path.join(work, "builds", f"step{s:03d}")) for s in (0, 1, 2, 3))
        (_ok("workdir purge: no local subfolder; step>N + orphan + .claude deleted; <=N kept")
         if (no_local and no_claude and gone and kept)
         else _fail(f"purge local={no_local} claude={no_claude} gone={gone} kept={kept}", fails))

        pres = (os.path.exists(os.path.join(sink, "builds", "step007"))
                and os.path.exists(os.path.join(sink, "builds", "step099"))
                and os.path.exists(os.path.join(sink, ".claude", "history.jsonl"))
                and os.path.exists(os.path.join(sink, "candidates.jsonl")))
        (_ok("workdir purge: discarded data (incl. orphan + .claude) preserved in external sink")
         if pres else _fail("external sink missing preserved data", fails))

        wlive = open(os.path.join(work, "run.log")).read()
        ok_state = ("=== STEP 4 " not in wlive and "step 3" in open(os.path.join(work, "best.cu")).read()
                    and json.load(open(os.path.join(work, "memory_state.json")))["champion"]["cid"] == "step003")
        _ok("workdir purge: run.log truncated, best.cu + memory rolled back") if ok_state \
            else _fail("purge run.log/best.cu/memory wrong", fails)

    # --- resume-time tail purge (purge_incomplete_tail): delete-only, state untouched -----------
    # A plain resume re-runs start = last_completed+1.  If a prior attempt at that step was killed
    # mid-build it leaves a stale build tree (incl. the torch FileBaton `lock` that would hang the
    # rebuild) plus a half-written candidate / run.log tail / step knowledge -- all for steps that are
    # NOT in candidates.jsonl.  These must be DELETED (no archive) while steps <= last and cross-step
    # state stay put.
    from fa3agent import config as C
    saved_root = C.WORK_ROOT
    with tempfile.TemporaryDirectory() as root:
        C.WORK_ROOT = root
        try:
            rid = "run_purge"
            work = os.path.join(root, rid)
            for d in ("candidates", "builds", "knowledge"):
                os.makedirs(os.path.join(work, d), exist_ok=True)
            with open(os.path.join(work, "candidates.jsonl"), "w") as fh:  # steps 0,1,2 persisted
                for s in (0, 1, 2):
                    fh.write(json.dumps({"step": s, "cid": f"step{s:03d}", "is_best": s == 2}) + "\n")
            for s in (0, 1, 2):
                os.makedirs(os.path.join(work, "builds", f"step{s:03d}"), exist_ok=True)
                open(os.path.join(work, "candidates", f"step{s:03d}.cu"), "w").write(f"// {s}\n")
            # leftover incomplete step 3 (never persisted) + stray orphan step 4
            os.makedirs(os.path.join(work, "builds", "step003"), exist_ok=True)
            open(os.path.join(work, "builds", "step003", "lock"), "w").write("")     # FileBaton
            open(os.path.join(work, "candidates", "step003.cu"), "w").write("// partial\n")
            os.makedirs(os.path.join(work, "builds", "step004"), exist_ok=True)
            open(os.path.join(work, "knowledge", "pf-tma.md"), "w").write("---\nstep: 3\n---\nx\n")
            open(os.path.join(work, "knowledge", "pf-old.md"), "w").write("---\nstep: 1\n---\nkeep\n")
            with open(os.path.join(work, "run.log"), "w") as fh:
                fh.write("=== STEP 2 (step002) ===\nok\n=== STEP 3 (step003) ===\nbuild\nHANG\n")

            res = RB.purge_incomplete_tail(rid, log=lambda m: None)
            gone = (res["start"] == 3
                    and not os.path.exists(os.path.join(work, "builds", "step003"))
                    and not os.path.exists(os.path.join(work, "builds", "step004"))
                    and not os.path.exists(os.path.join(work, "candidates", "step003.cu"))
                    and not os.path.exists(os.path.join(work, "knowledge", "pf-tma.md")))
            kept = (all(os.path.isdir(os.path.join(work, "builds", f"step{s:03d}")) for s in (0, 1, 2))
                    and os.path.exists(os.path.join(work, "knowledge", "pf-old.md"))
                    and [json.loads(l)["step"] for l in open(os.path.join(work, "candidates.jsonl"))
                         if l.strip()] == [0, 1, 2])   # lineage (cross-step state) untouched
            log = open(os.path.join(work, "run.log")).read()
            log_ok = "=== STEP 3" not in log and "=== STEP 2" in log and "incomplete step 3" in log
            (_ok("resume purge: step>=start (incl. FileBaton lock + orphan) deleted; <=last + state kept")
             if (gone and kept and log_ok)
             else _fail(f"resume purge gone={gone} kept={kept} log_ok={log_ok} res={res}", fails))
        finally:
            C.WORK_ROOT = saved_root

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED"))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
