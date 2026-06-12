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
import time
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
    timed_out: bool = False   # killed by the wall-clock backstop (NOT retried; likely a true hang)


def claude_available() -> bool:
    return shutil.which("claude") is not None


def run_claude(prompt: str, cwd: str, model: Optional[str] = None, effort: Optional[str] = None,
               session_id: Optional[str] = None, timeout: int = 5400,
               allowed_dirs: Optional[list[str]] = None,
               stream_log: Optional[str] = None) -> LLMResult:
    """Spawn `claude -p` headless, parse the JSON envelope, return text + session id.

    The prompt instructs the brain to WRITE the candidate .cu to a known path; the driver
    builds/evaluates it (the brain must NOT run the eval itself).

    If ``stream_log`` is given, the brain runs with ``--output-format stream-json`` and the
    transcript is COALESCED to one entry per complete assistant/user/result event (tailable, no
    partial_json spam).  A wall-clock ``timeout`` bounds a true hang; a heartbeat line is written
    while the relay is silent so a slow generation does not look frozen; a clean failure is retried
    once with a fresh session.  Otherwise a single-shot ``--output-format json`` call is captured.
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
    resume_args = ["--resume", session_id] if session_id else []

    if stream_log:
        # stream-json -> a CONSOLIDATED transcript in brain.log (one entry per complete
        # assistant/user/result event).  We deliberately do NOT pass --include-partial-messages:
        # on this relay the partial deltas don't actually stream during a long generation (it
        # buffers and delivers whole messages), so they'd add log spam without a usable heartbeat.
        # Liveness instead comes from a heartbeat line _run_streamed writes while idle; a true hang
        # is bounded by the wall-clock `timeout`.
        stream_extra = ["--output-format", "stream-json", "--verbose"]
        res = _run_streamed(base + resume_args + stream_extra, cwd, timeout, stream_log, session_id)
        # Retry a CLEAN failure (nonzero exit / error result -- typically a dropped relay
        # connection) once with a FRESH session.  Do NOT retry a wall-clock timeout: it already ran
        # the full budget, so the orchestrator should fall back to the champion instead.
        retries = max(0, int(getattr(C, "HEADLESS_FAILURE_RETRIES", 1)))
        attempt = 1
        while (not res.ok) and (not res.timed_out) and attempt <= retries:
            attempt += 1
            res = _run_streamed(base + stream_extra, cwd, timeout, stream_log, None, attempt=attempt)
        return res

    cmd = base + resume_args + ["--output-format", "json"]
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


def _stringify_tool_result(content) -> str:
    """Flatten a tool_result `content` (str | list-of-blocks | other) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                out.append(b.get("text") or json.dumps(b, ensure_ascii=False))
            else:
                out.append(str(b))
        return "\n".join(out)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _render_block(block: dict) -> str:
    """Render one assistant/user content block (text / thinking / tool_use / tool_result)."""
    bt = block.get("type")
    if bt == "text":
        return block.get("text", "")
    if bt == "thinking":
        return "[thinking]\n" + block.get("thinking", "")
    if bt == "redacted_thinking":
        return "[thinking] (redacted)"
    if bt == "tool_use":
        return (f"[tool_use: {block.get('name', '?')}]\n"
                + json.dumps(block.get("input", {}), ensure_ascii=False, indent=2))
    if bt == "tool_result":
        return "[tool_result]\n" + _stringify_tool_result(block.get("content"))
    return f"[{bt}]\n" + json.dumps(block, ensure_ascii=False)


def _render_stream_event(obj: dict) -> Optional[str]:
    """Render ONE complete stream-json event as a consolidated brain.log entry, or None to skip.

    We log the COMPLETE ``assistant``/``user``/``result`` events (and the ``system:init``
    header) -- each already carries the full text of a message/block.  Partial ``stream_event``
    deltas (text_delta / thinking_delta / ``input_json_delta`` ``partial_json``) and system
    heartbeats are skipped, so the log gets one entry per message instead of one per fragment.
    """
    t = obj.get("type")
    if t in ("assistant", "user"):
        msg = obj.get("message") or {}
        role = msg.get("role", t)
        content = msg.get("content")
        if isinstance(content, str):
            body = content
        elif isinstance(content, list):
            body = "\n\n".join(_render_block(b) for b in content if isinstance(b, dict))
        else:
            body = ""
        if not body.strip():
            return None
        return f"=== {role} ===\n{body}"
    if t == "result":
        bits = [f"session={obj.get('session_id', '')}"]
        if obj.get("duration_ms") is not None:
            bits.append(f"duration_ms={obj.get('duration_ms')}")
        if obj.get("total_cost_usd") is not None:
            bits.append(f"cost_usd={obj.get('total_cost_usd')}")
        head = f"=== result:{obj.get('subtype', '')} === " + " ".join(bits)
        if obj.get("is_error"):
            head += "\n" + str(obj.get("result", ""))
        return head
    if t == "system" and obj.get("subtype") == "init":
        return f"=== system:init === session={obj.get('session_id', '')} cwd={obj.get('cwd', '')}"
    return None  # stream_event partials, system heartbeats, etc.


def _run_streamed(cmd: list, cwd: str, timeout: int, stream_log: str,
                  session_id: Optional[str], attempt: int = 1) -> LLMResult:
    """Run claude with stream-json, tee a CONSOLIDATED transcript to ``stream_log`` (one entry per
    complete event), and parse the final ``type==result`` event.

    This relay BUFFERS long generations (no output for many minutes even when the brain is healthy),
    so we must NOT kill on output-idleness.  A monitor thread instead (a) WRITES a liveness
    heartbeat into the log every ``C.HEADLESS_HEARTBEAT_S`` of silence so a slow run does not look
    frozen, and (b) kills the call ONLY at the absolute ``timeout`` backstop (-> ``timed_out``).
    """
    heartbeat_s = int(getattr(C, "HEADLESS_HEARTBEAT_S", 120))
    try:
        os.makedirs(os.path.dirname(stream_log) or ".", exist_ok=True)
    except Exception:
        pass
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
    except Exception as e:  # noqa
        return LLMResult(ok=False, error=f"claude spawn failed: {e!r}")

    state = {"reason": None, "last": time.monotonic()}
    stop = threading.Event()
    write_lock = threading.Lock()
    result_env = None

    def _kill(reason: str):
        if state["reason"] is None:
            state["reason"] = reason
        try:
            proc.kill()
        except Exception:
            pass

    try:
        with open(stream_log, "w" if attempt <= 1 else "a") as f:
            f.write(f"# claude stream-json (attempt {attempt}; model+effort headless brain)\n"); f.flush()

            def _monitor():
                start = time.monotonic()
                last_hb = start
                while not stop.wait(15.0):
                    if proc.poll() is not None:
                        return
                    now = time.monotonic()
                    if now - start > timeout:        # absolute backstop -> kill a true hang
                        _kill("total"); return
                    # The relay can be silent for minutes mid-generation -> surface liveness rather
                    # than killing.  Heartbeat at most once per heartbeat_s, only while idle.
                    if now - state["last"] >= heartbeat_s and now - last_hb >= heartbeat_s:
                        with write_lock:
                            if not stop.is_set():
                                f.write(f"# [heartbeat] brain still running: {now - start:.0f}s "
                                        f"elapsed, {now - state['last']:.0f}s since last output\n")
                                f.flush()
                        last_hb = now

            mon = threading.Thread(target=_monitor, daemon=True)
            mon.start()
            try:
                for line in proc.stdout:        # blocks per line; monitor kills only on total timeout
                    state["last"] = time.monotonic()   # mark output activity
                    s = line.strip()
                    if not s:
                        continue
                    obj = None
                    if s.startswith("{"):
                        try:
                            obj = json.loads(s)
                        except json.JSONDecodeError:
                            obj = None
                    with write_lock:
                        if obj is None:
                            f.write(line); f.flush()   # stderr / diagnostics / non-event -> keep raw
                        else:
                            if obj.get("type") == "result":
                                result_env = obj
                            entry = _render_stream_event(obj)
                            if entry:
                                f.write(entry.rstrip("\n") + "\n\n"); f.flush()
            finally:
                stop.set()
                mon.join(timeout=3)
        proc.wait()
    finally:
        stop.set()

    if state["reason"] == "total":
        return LLMResult(ok=False, timed_out=True,
                         error=f"claude timed out after {timeout}s (attempt {attempt}, see {stream_log})")
    if result_env is None:
        if proc.returncode not in (0, None):
            return LLMResult(ok=False,
                             error=f"claude streamed rc={proc.returncode} (attempt {attempt}, see {stream_log})")
        return LLMResult(ok=True, text="(streamed; no result event)", session_id=session_id)
    return _envelope_to_result(result_env, session_id)


def is_env_fatal(msg: str) -> bool:
    """Classify an error as unfixable-by-agent so the loop aborts instead of burning rounds."""
    m = (msg or "").lower()
    return any(sig.lower() in m for sig in C.ENV_FATAL_SIGNATURES)
