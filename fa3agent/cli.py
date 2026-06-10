"""Command-line entrypoint:  python -m fa3agent run [options]

  run      drive the optimization loop (bootstrap -> correctness -> profile -> optimize)
  list     list checkpointed runs
  validate run per-component smoke tests (pytest-free; just runs fa3agent/tests/*)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

from . import config as C


def _latest_run() -> str | None:
    runs = sorted(glob.glob(os.path.join(C.CHECKPOINT_ROOT, "run_*")))
    return os.path.basename(runs[-1]) if runs else None


def cmd_run(a):
    from .orchestrator import Orchestrator
    run_id = a.run_id
    if a.resume and not run_id:
        run_id = _latest_run()
        if not run_id:
            print("no run to resume"); return 2
    if not run_id:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"run_id={run_id}  configs={a.configs}  max_steps={a.max_steps}  model={a.model}  "
          f"resume={a.resume}")
    orch = Orchestrator(run_id, config_set=a.configs, max_steps=a.max_steps, model=a.model,
                        decode_tokens=a.decode_tokens, do_decode_equiv=not a.no_decode_equiv,
                        profile_enabled=not a.no_profile)
    result = orch.run()
    print("RESULT:", result)
    print("checkpoint:", os.path.join(C.CHECKPOINT_ROOT, run_id))
    return 0


def cmd_list(a):
    for r in sorted(glob.glob(os.path.join(C.CHECKPOINT_ROOT, "run_*"))):
        jsonl = os.path.join(r, "candidates.jsonl")
        n = sum(1 for _ in open(jsonl)) if os.path.exists(jsonl) else 0
        print(f"{os.path.basename(r):28s} steps={n}")
    return 0


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
    r.add_argument("--model", default="opus")
    r.add_argument("--run-id", default=None)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--decode-tokens", type=int, default=100)
    r.add_argument("--no-decode-equiv", action="store_true", help="skip the 100-token greedy gate")
    r.add_argument("--no-profile", action="store_true", help="skip NCU profiling (faster machinery test)")
    r.set_defaults(func=cmd_run)

    sub.add_parser("list", help="list runs").set_defaults(func=cmd_list)
    sub.add_parser("validate", help="run per-component smoke tests").set_defaults(func=cmd_validate)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
