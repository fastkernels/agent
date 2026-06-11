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
import threading
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


def run_claude(prompt: str, cwd: str, model: Optional[str] = None, effort: Optional[str] = None,
               session_id: Optional[str] = None, timeout: int = 5400,
               allowed_dirs: Optional[list[str]] = None,
               stream_log: Optional[str] = None) -> LLMResult:
    """Spawn `claude -p` headless, parse the JSON envelope, return text + session id.

    The prompt instructs the brain to WRITE the candidate .cu to a known path; the driver
    builds/evaluates it (the brain must NOT run the eval itself).

    If ``stream_log`` is given, the brain runs with ``--output-format stream-json`` and every
    event line is teed LIVE to that file (tailable in real time) while the final result event
    is parsed; otherwise a single-shot ``--output-format json`` call is captured.
    """
    # FA3_STUB_LLM: skip the real Claude call (fast, deterministic machinery test). The
    # orchestrator then falls back to seeding from seed_kernel.cu / reusing the champion.
    if os.environ.get("FA3_STUB_LLM"):
        return LLMResult(ok=True, text="(stub: no LLM call)", session_id="stub")
    if not claude_available():
        return LLMResult(ok=False, error="claude CLI not found on PATH")

    model = model or C.HEADLESS_MODEL
    effort = effort if effort is not None else C.HEADLESS_EFFORT
    base = ["claude", "-p", prompt, "--dangerously-skip-permissions", "--model", model]
    if effort:
        base += ["--effort", effort]
    if session_id:
        base += ["--resume", session_id]

    if stream_log:
        # --include-partial-messages streams thinking/text DELTAS live (so brain.log advances
        # continuously during long turns, not only when an event completes).
        return _run_streamed(base + ["--output-format", "stream-json", "--verbose",
                                     "--include-partial-messages"],
                             cwd, timeout, stream_log, session_id)

    cmd = base + ["--output-format", "json"]
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
    return _envelope_to_result(env, session_id)


def _envelope_to_result(env: dict, session_id: Optional[str]) -> LLMResult:
    text = env.get("result") or env.get("text") or ""
    sid = env.get("session_id") or session_id
    is_err = bool(env.get("is_error", False))
    return LLMResult(ok=not is_err, text=text, session_id=sid,
                     error=("" if not is_err else text), raw=env)


def _run_streamed(cmd: list, cwd: str, timeout: int, stream_log: str,
                  session_id: Optional[str]) -> LLMResult:
    """Run claude with stream-json, tee every event line LIVE to ``stream_log``, and parse the
    final ``type==result`` event.  A watchdog kills the process after ``timeout`` seconds."""
    try:
        os.makedirs(os.path.dirname(stream_log) or ".", exist_ok=True)
    except Exception:
        pass
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
    except Exception as e:  # noqa
        return LLMResult(ok=False, error=f"claude spawn failed: {e!r}")

    killed = {"v": False}

    def _kill():
        killed["v"] = True
        try:
            proc.kill()
        except Exception:
            pass

    timer = threading.Timer(timeout, _kill)
    timer.start()
    result_env = None
    try:
        with open(stream_log, "w") as f:
            f.write(f"# claude stream-json (model+effort headless brain)\n"); f.flush()
            for line in proc.stdout:        # blocks per line; watchdog kills on timeout
                f.write(line); f.flush()
                s = line.strip()
                if s.startswith("{"):
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "result":
                        result_env = obj
        proc.wait()
    finally:
        timer.cancel()

    if killed["v"]:
        return LLMResult(ok=False, error=f"claude timed out after {timeout}s (see {stream_log})")
    if result_env is None:
        if proc.returncode not in (0, None):
            return LLMResult(ok=False, error=f"claude streamed rc={proc.returncode} (see {stream_log})")
        return LLMResult(ok=True, text="(streamed; no result event)", session_id=session_id)
    return _envelope_to_result(result_env, session_id)


def is_env_fatal(msg: str) -> bool:
    """Classify an error as unfixable-by-agent so the loop aborts instead of burning rounds."""
    m = (msg or "").lower()
    return any(sig.lower() in m for sig in C.ENV_FATAL_SIGNATURES)
