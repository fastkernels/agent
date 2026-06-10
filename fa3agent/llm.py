"""Claude Code headless wrapper (the codegen / optimize brain).

Per spec, all (sub)agents run Claude Code headless with --dangerously-skip-permissions.
The driver (orchestrator) runs eval and feeds distilled results back next round; the brain
only writes the candidate .cu and reasons.  Session id is carried forward so refinement
rounds RESUME the same conversation (continuity, cheaper, more coherent) -- syfi pattern.
"""
from __future__ import annotations

import json
import os
import subprocess
import shutil
from dataclasses import dataclass
from typing import Optional

from . import config as C


@dataclass
class LLMResult:
    ok: bool
    text: str = ""
    session_id: Optional[str] = None
    error: str = ""
    raw: Optional[dict] = None


def claude_available() -> bool:
    return shutil.which("claude") is not None


def run_claude(prompt: str, cwd: str, model: str = "opus",
               session_id: Optional[str] = None, timeout: int = 5400,
               allowed_dirs: Optional[list[str]] = None) -> LLMResult:
    """Spawn `claude -p` headless, parse the JSON envelope, return text + session id.

    The prompt instructs the brain to WRITE the candidate .cu to a known path; the driver
    builds/evaluates it (the brain must NOT run the eval itself).
    """
    # FA3_STUB_LLM: skip the real Claude call (fast, deterministic machinery test). The
    # orchestrator then falls back to seeding from seed_kernel.cu / reusing the champion.
    if os.environ.get("FA3_STUB_LLM"):
        return LLMResult(ok=True, text="(stub: no LLM call)", session_id="stub")
    if not claude_available():
        return LLMResult(ok=False, error="claude CLI not found on PATH")

    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "--model", model,
    ]
    if session_id:
        cmd += ["--resume", session_id]

    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return LLMResult(ok=False, error=f"claude timed out after {timeout}s")
    except Exception as e:  # noqa
        return LLMResult(ok=False, error=f"claude spawn failed: {e!r}")

    if proc.returncode != 0:
        return LLMResult(ok=False, error=f"claude rc={proc.returncode}: {proc.stderr[-2000:]}")

    out = proc.stdout.strip()
    # `--output-format json` emits a single JSON object with at least {result, session_id}.
    try:
        env = json.loads(out)
    except json.JSONDecodeError:
        # fall back: maybe stream-json lines; take last json object
        env = None
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    env = json.loads(line); break
                except json.JSONDecodeError:
                    continue
        if env is None:
            return LLMResult(ok=True, text=out, raw=None)  # treat raw text as the result

    text = env.get("result") or env.get("text") or ""
    sid = env.get("session_id") or session_id
    is_err = bool(env.get("is_error", False))
    return LLMResult(ok=not is_err, text=text, session_id=sid,
                     error=("" if not is_err else text), raw=env)


def is_env_fatal(msg: str) -> bool:
    """Classify an error as unfixable-by-agent so the loop aborts instead of burning rounds."""
    m = (msg or "").lower()
    return any(sig.lower() in m for sig in C.ENV_FATAL_SIGNATURES)
