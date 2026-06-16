"""Validation for Workstream F (run isolation: live workdir vs durable archive).

Run from /data-fast/agent:
    python -m fa3agent.tests.test_runisolation

Deterministic (no GPU / no claude): drives checkpoint + llm helpers against temp WORK_ROOT /
CHECKPOINT_ROOT roots and asserts:
  - the LIVE run dir resolves under WORK_ROOT (not CHECKPOINT_ROOT);
  - a FRESH init_run WIPES WORK_ROOT (prior runs' live dirs vanish);
  - mirror_to_archive copies live -> archive incrementally, INCLUDING built .so but EXCLUDING the
    brain's per-run .claude config dir;
  - RESUME restores the live workdir from the archive and does NOT wipe;
  - the run_claude F4 guard refuses any CHECKPOINT_ROOT path via cwd / --add-dir / prompt;
  - the brain's CLAUDE_CONFIG_DIR resolves under the workdir.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

from fa3agent import config as C
from fa3agent import checkpoint, llm


def _ok(msg: str) -> None:
    print(f"  PASS: {msg}")


def _fail(msg: str) -> int:
    print(f"  FAIL: {msg}")
    return 1


def _touch(path: str, data: str = "x") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(data)


def main() -> int:
    failures = 0
    tmp = tempfile.mkdtemp(prefix="fa3_isolation_")
    work = os.path.join(tmp, "work")
    arch = os.path.join(tmp, "archive")
    # Redirect the roots at runtime (checkpoint/llm read C.* dynamically inside functions).
    saved = (C.WORK_ROOT, C.CHECKPOINT_ROOT)
    C.WORK_ROOT, C.CHECKPOINT_ROOT = work, arch
    try:
        # ---- (1) live run dir resolves under WORK_ROOT ----
        print("[1] live run dir under WORK_ROOT")
        p = checkpoint.paths("run_AAA")
        if p["run"].startswith(work) and not p["run"].startswith(arch):
            _ok("paths(run)['run'] under WORK_ROOT")
        else:
            failures += _fail(f"run dir not under WORK_ROOT: {p['run']}")

        # ---- (2) fresh init_run wipes WORK_ROOT ----
        print("[2] fresh init_run wipes WORK_ROOT (prior live runs gone)")
        stale = os.path.join(work, "run_OLD", "best.cu")
        _touch(stale, "stale champion from a prior run")
        checkpoint.init_run("run_AAA", {"config_set": "dev"}, resume=False)
        if os.path.exists(stale):
            failures += _fail("stale prior-run live dir survived the fresh wipe")
        elif os.path.isdir(checkpoint.paths("run_AAA")["candidates"]):
            _ok("WORK_ROOT wiped; only the new run's dirs exist")
        else:
            failures += _fail("new run dirs not created")

        # ---- (3) mirror live -> archive (include .so, exclude .claude) ----
        print("[3] mirror_to_archive: include .so, exclude .claude")
        pa = checkpoint.paths("run_AAA")
        _touch(pa["best"], "champion source")
        _touch(os.path.join(pa["candidates"], "step000.cu"), "cand")
        _touch(os.path.join(pa["builds"], "step000", "_distill.so"), "ELF")
        _touch(os.path.join(pa["run"], "MEMORY.md"), "lessons")
        _touch(os.path.join(pa["run"], ".claude", "history.jsonl"), "SECRET prior chat")
        checkpoint.mirror_to_archive("run_AAA")
        adir = os.path.join(arch, "run_AAA")
        checks = {
            "best.cu": os.path.exists(os.path.join(adir, "best.cu")),
            "candidates/step000.cu": os.path.exists(os.path.join(adir, "candidates", "step000.cu")),
            "builds/step000/_distill.so": os.path.exists(os.path.join(adir, "builds", "step000", "_distill.so")),
            "MEMORY.md": os.path.exists(os.path.join(adir, "MEMORY.md")),
        }
        claude_leaked = os.path.exists(os.path.join(adir, ".claude"))
        if all(checks.values()) and not claude_leaked:
            _ok("archive has best/candidate/.so/MEMORY.md; .claude NOT mirrored")
        else:
            failures += _fail(f"mirror wrong: present={checks} claude_leaked={claude_leaked}")

        # ---- (4) resume restores live from archive (no wipe) ----
        print("[4] resume restores the live workdir from the archive")
        shutil.rmtree(work, ignore_errors=True)   # simulate a wiped/cold box
        checkpoint.init_run("run_AAA", {"config_set": "dev"}, resume=True)
        if os.path.exists(os.path.join(work, "run_AAA", "best.cu")) and \
           os.path.exists(os.path.join(work, "run_AAA", "MEMORY.md")):
            _ok("live workdir restored from archive on resume")
        else:
            failures += _fail("resume did not restore the live workdir")

        # ---- (5) run_claude F4 guard refuses CHECKPOINT_ROOT exposure ----
        print("[5] run_claude F4 guard (cwd / --add-dir / prompt)")
        os.environ["FA3_STUB_LLM"] = "1"   # the guard runs BEFORE the stub short-circuit
        try:
            ckpt_path = os.path.join(arch, "run_AAA")
            r_cwd = llm.run_claude("hi", cwd=ckpt_path, allowed_dirs=[work])
            r_add = llm.run_claude("hi", cwd=C.REPO_ROOT, allowed_dirs=[ckpt_path])
            r_prompt = llm.run_claude(f"read {ckpt_path}/best.cu", cwd=C.REPO_ROOT, allowed_dirs=[work])
            r_clean = llm.run_claude("hi", cwd=C.REPO_ROOT, allowed_dirs=[os.path.join(work, "run_AAA")])
            if (not r_cwd.ok) and (not r_add.ok) and (not r_prompt.ok) and r_clean.ok:
                _ok("guard refuses cwd/add-dir/prompt leaks; clean workdir call allowed")
            else:
                failures += _fail(f"guard misbehaved: cwd={r_cwd.ok} add={r_add.ok} "
                                  f"prompt={r_prompt.ok} clean={r_clean.ok}")
        finally:
            os.environ.pop("FA3_STUB_LLM", None)

        # ---- (6) brain CLAUDE_CONFIG_DIR resolves under the workdir ----
        print("[6] brain Claude config dir under the workdir + slug well-formed")
        brain_cfg = os.path.join(checkpoint.paths("run_AAA")["run"], ".claude")
        slug = llm._repo_project_slug()
        if brain_cfg.startswith(work) and slug and not slug.startswith("/"):
            _ok(f"brain config under WORK_ROOT; repo slug={slug!r}")
        else:
            failures += _fail(f"brain config/slug wrong: cfg={brain_cfg} slug={slug!r}")

    finally:
        C.WORK_ROOT, C.CHECKPOINT_ROOT = saved
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures == 0:
        print("SMOKE: PASS (run isolation)")
    else:
        print(f"SMOKE: FAIL ({failures} failing checks)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
