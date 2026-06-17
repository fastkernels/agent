"""Command-line entrypoint:  python -m fa3agent run [options]

  run      drive the optimization loop (bootstrap -> correctness -> profile -> optimize)
  list     list checkpointed runs
  validate run per-component smoke tests (pytest-free; just runs fa3agent/tests/*)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

from . import config as C


def _latest_run() -> str | None:
    runs = sorted(glob.glob(os.path.join(C.CHECKPOINT_ROOT, "run_*")))
    return os.path.basename(runs[-1]) if runs else None


class _Tee:
    """Fan one text stream out to several (e.g. the real stdout + the run's run.log).

    Line-buffered writes with a flush after every write so ``run.log`` updates live
    as the loop prints progress -- ``tail -f .../run_*/run.log`` works regardless of
    how the process was launched (foreground, nohup, or a redirected stdout).
    """

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
                st.flush()
            except (OSError, ValueError):
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except (OSError, ValueError):
                pass

    def isatty(self):
        return False


def _maybe_rollback(run_id: str, from_step: int, resume: bool, assume_yes: bool):
    """Handle --from-step.  Returns an int exit code to ABORT cmd_run, or None to continue
    (no rollback needed, or the rollback completed and the run should resume from from_step+1).

    All the heavy lifting (plan, confirmation, dual workdir+archive surgery) lives in
    fa3agent.rollback.apply(); here we only enforce that --from-step implies a resume."""
    from . import rollback as RB

    if not resume:
        print("--from-step requires --resume (it resumes an existing run from an earlier step)")
        return 2
    return None if RB.apply(run_id, from_step, assume_yes=assume_yes) else 1


def cmd_run(a):
    from .orchestrator import Orchestrator
    run_id = a.run_id
    if a.resume and not run_id:
        run_id = _latest_run()
        if not run_id:
            print("no run to resume"); return 2
    if not run_id:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    # Optional rollback: resume from an EARLIER step.  Keep steps <= --from-step, archive the rest,
    # and continue at N+1.  Done BEFORE the orchestrator is built so its in-memory stores (memory /
    # levers / plateau) load the rolled-back state.
    if a.from_step is not None:
        rc = _maybe_rollback(run_id, a.from_step, a.resume, a.yes)
        if rc is not None:
            return rc

    print(f"run_id={run_id}  configs={a.configs}  max_steps={a.max_steps}  model={a.model}  "
          f"effort={a.effort}  resume={a.resume}")

    # Build the orchestrator FIRST: its init prepares the LIVE workdir under WORK_ROOT (a fresh run
    # WIPES WORK_ROOT, a resume restores it from the archive).  We then tee the run narrative into
    # the workdir's run.log -- the live working copy -- which F3 mirrors to the durable archive each
    # step.  (Writing run.log straight into CHECKPOINT_ROOT would put a live-written file in the
    # archive that the brain's workdir never owns; the archive is a MIRROR, not a live target.)
    orch = Orchestrator(run_id, config_set=a.configs, max_steps=a.max_steps, model=a.model,
                        effort=a.effort,
                        decode_tokens=a.decode_tokens, do_decode_equiv=not a.no_decode_equiv,
                        profile_enabled=not a.no_profile, resume=a.resume)

    # On a resume, scrub the leftovers of a step that was started but never persisted (e.g. a build
    # killed mid-flight leaves a torch FileBaton `lock` that would make the rebuild hang forever).
    # Delete-only (no archive); cross-step state is untouched.  Done after the orchestrator restored
    # the workdir but BEFORE the run.log tee opens a handle, so the run.log tail truncation is clean.
    if a.resume:
        from . import rollback as RB
        RB.purge_incomplete_tail(run_id)

    # Tee the whole run narrative into <workdir>/run.log so the live progress log lives inside the
    # live run folder (what `tail -f .../run_*/run.log` expects), in addition to whatever the
    # launcher does with stdout.  Best-effort: never let logging-setup break the run.
    run_dir = orch.paths["run"]
    log_fh = None
    saved = None
    try:
        os.makedirs(run_dir, exist_ok=True)
        log_fh = open(os.path.join(run_dir, "run.log"), "a", buffering=1,
                      encoding="utf-8", errors="replace")
        saved = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(saved[0], log_fh)
        sys.stderr = _Tee(saved[1], log_fh)
    except OSError as e:
        print(f"[warn] could not open {run_dir}/run.log ({e}); logging to stdout only")
        log_fh = None

    try:
        result = orch.run()
        print("RESULT:", result)
        print("checkpoint:", C.archive_dir(run_id))
        return 0
    finally:
        if saved is not None:
            sys.stdout, sys.stderr = saved
        if log_fh is not None:
            log_fh.close()


def cmd_list(a):
    for r in sorted(glob.glob(os.path.join(C.CHECKPOINT_ROOT, "run_*"))):
        jsonl = os.path.join(r, "candidates.jsonl")
        n = sum(1 for _ in open(jsonl)) if os.path.exists(jsonl) else 0
        print(f"{os.path.basename(r):28s} steps={n}")
    return 0


_SPARK = "▁▂▃▄▅▆▇█"


def _spark(vals, lo=0.0, hi=1.2):
    out = []
    for v in vals:
        v = max(lo, min(hi, v or 0.0))
        out.append(_SPARK[min(len(_SPARK) - 1, int((v - lo) / (hi - lo + 1e-9) * (len(_SPARK) - 1)))])
    return "".join(out)


def _fmt_row(r: dict) -> str:
    ev = r.get("eval") or {}
    pf = r.get("profile") or {}
    insitu = ev.get("insitu") or {}
    de = ev.get("decode_equiv_ok")
    ins = "-" if not insitu else ("ok" if insitu.get("ok") else ("skip" if insitu.get("skip") else "FAIL"))
    dec = "-" if de is None else ("ok" if de else "DIVERGE")
    return (f"{r.get('step'):>4} {r.get('status',''):<13} "
            f"min={ev.get('min_ratio',0):>5.3f} geo={ev.get('geomean_ratio',0):>5.3f} "
            f"insitu={ins:<5} decode={dec:<7} {'★' if r.get('is_best') else ' '} "
            f"{(pf.get('bottleneck','') or '')[:46]}")


def cmd_watch(a):
    import time
    run_id = a.run_id or _latest_run()
    if not run_id:
        print("no run to watch"); return 2
    jsonl = os.path.join(C.CHECKPOINT_ROOT, run_id, "candidates.jsonl")
    print(f"watching {run_id}  ({jsonl})")
    print(f"{'step':>4} {'status':<13} {'minratio':<9} {'geomean':<9} {'insitu':<11} {'decode':<11} best bottleneck")
    seen = 0
    mins = []
    while True:
        rows = []
        if os.path.exists(jsonl):
            with open(jsonl) as f:
                rows = [l for l in f.read().splitlines() if l.strip()]
        for line in rows[seen:]:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            print(_fmt_row(r), flush=True)
            mins.append((r.get("eval") or {}).get("min_ratio", 0.0))
        if len(rows) > seen:
            seen = len(rows)
            if len(mins) > 1:
                print(f"     min-ratio trend: {_spark(mins)}  (parity=1.0; best={max(mins):.3f})", flush=True)
        if a.once:
            return 0
        time.sleep(a.interval)



def cmd_validate(a):
    import subprocess
    tests = sorted(glob.glob(os.path.join(C.PKG_DIR, "tests", "test_*.py")))
    print(f"running {len(tests)} component smoke tests from {C.REPO_ROOT}")
    rc = 0
    for t in tests:
        print(f"\n===== {os.path.basename(t)} =====", flush=True)
        p = subprocess.run([sys.executable, t], cwd=C.REPO_ROOT)
        rc |= p.returncode
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fa3agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="drive the optimization loop")
    r.add_argument("--max-steps", type=int, default=None, help="optional step budget (loops indefinitely if unset)")
    r.add_argument("--configs", choices=["dev", "full"], default="dev")
    r.add_argument("--model", default=C.HEADLESS_MODEL, help="full model id for the headless brain")
    r.add_argument("--effort", default=C.HEADLESS_EFFORT,
                   choices=["low", "medium", "high", "xhigh", "ultracode"],
                   help="reasoning effort for the headless brain (default high)")
    r.add_argument("--run-id", default=None)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--from-step", type=int, default=None,
                   help="resume from an earlier step: keep steps <= N, archive the rest under a "
                        "rolled_back/ subfolder, and continue at N+1 (requires --resume; a step "
                        ">= the last completed step is a normal resume)")
    r.add_argument("--yes", "-y", action="store_true",
                   help="skip the interactive rollback confirmation (for non-interactive/background runs)")
    r.add_argument("--decode-tokens", type=int, default=100)
    r.add_argument("--no-decode-equiv", action="store_true", help="skip the 100-token greedy gate")
    r.add_argument("--no-profile", action="store_true", help="skip NCU profiling (faster machinery test)")
    r.set_defaults(func=cmd_run)

    sub.add_parser("list", help="list runs").set_defaults(func=cmd_list)
    sub.add_parser("validate", help="run per-component smoke tests").set_defaults(func=cmd_validate)

    w = sub.add_parser("watch", help="live per-step table from a run's candidates.jsonl")
    w.add_argument("--run-id", default=None, help="run to watch (default: latest)")
    w.add_argument("--once", action="store_true", help="print current table and exit")
    w.add_argument("--interval", type=float, default=3.0, help="poll interval seconds")
    w.set_defaults(func=cmd_watch)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
