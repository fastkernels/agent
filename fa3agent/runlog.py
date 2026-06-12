"""Per-tool call logging for a run (component I/O capture, for double-checking a run).

Goal: for every tool/component the agent uses, persist ONE log entry per call — what it
was called with (paths / prompt / cfg / query / command) and what it returned — so a run
can be audited after the fact.  One file per component under ``<FA3_LOG_DIR>/<name>.log``.

Mechanism: the orchestrator sets ``FA3_LOG_DIR`` (-> ``<run_id>/logs``) and, per step,
``FA3_LOG_STEP``.  Every instrumented entry point (decorated with ``@logged``) appends its
inputs+output there.  Because the measure/profile/verify SUBPROCESSES and the BRAIN's own
``python -c`` tool calls all inherit the orchestrator's environment, the SAME decorated
function logs whether it was invoked by the orchestrator, a worker subprocess, or the brain
— no per-caller wiring needed.  When ``FA3_LOG_DIR`` is unset (tests / standalone use, or
``FA3_DISABLE_TOOL_LOGS`` set), every function here is a cheap no-op.

Concurrency: several processes (e.g. one measure subprocess per GPU) may append to the same
file at once, so each write is guarded by an ``flock``.
"""
from __future__ import annotations

import functools
import inspect
import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional

_ENV_DIR = "FA3_LOG_DIR"
_ENV_STEP = "FA3_LOG_STEP"
_ENV_DISABLE = "FA3_DISABLE_TOOL_LOGS"

# Generous per-value render cap.  Tool returns are bounded by design (context discipline);
# this only guards against an unexpectedly huge input/output blob.  Callers that log the
# brain prompt pass a much larger cap explicitly.
_MAX = 60_000


# ----------------------------------------------------------------------------------------
# Configuration (env-backed so child processes inherit it automatically)
# ----------------------------------------------------------------------------------------
def enabled() -> bool:
    return bool(os.environ.get(_ENV_DIR)) and not os.environ.get(_ENV_DISABLE)


def set_dir(path: str) -> None:
    """Point logging at ``path`` (creates it).  Sets the env var so children inherit it."""
    os.environ[_ENV_DIR] = path
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def set_step(step) -> None:
    """Tag subsequent entries (this process and any children it spawns) with ``step``."""
    os.environ[_ENV_STEP] = str(step)


def _log_dir() -> Optional[str]:
    if os.environ.get(_ENV_DISABLE):
        return None
    return os.environ.get(_ENV_DIR) or None


def _cur_step() -> str:
    return os.environ.get(_ENV_STEP, "")


# ----------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------
def _jsonable(v: Any, _depth: int = 0):
    if _depth > 6:
        return str(v)
    if is_dataclass(v) and not isinstance(v, type):
        try:
            v = asdict(v)
        except Exception:
            return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x, _depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, _depth + 1) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _render(v: Any, max_chars: int) -> str:
    try:
        s = json.dumps(_jsonable(v), indent=2, default=str, ensure_ascii=False)
    except Exception:
        s = str(v)
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n...[truncated {len(s) - max_chars} chars]"
    return s


def _safe_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name) or "tool"


def _append(path: str, text: str) -> None:
    try:
        import fcntl
    except Exception:
        fcntl = None  # type: ignore
    try:
        with open(path, "a", encoding="utf-8") as f:
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
            f.write(text)
            f.flush()
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
    except Exception:
        pass  # logging must never break the run


# ----------------------------------------------------------------------------------------
# Public recording API
# ----------------------------------------------------------------------------------------
def record(component: str, *, label: Optional[str] = None, inputs: Any = None,
           output: Any = None, error: Any = None, note: Optional[str] = None,
           duration_s: Optional[float] = None, max_chars: int = _MAX) -> None:
    """Append one ``<component>.log`` entry: header + INPUT + OUTPUT/ERROR.  No-op if disabled."""
    d = _log_dir()
    if not d:
        return
    path = os.path.join(d, f"{_safe_name(component)}.log")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    step = _cur_step()
    head = f"{'=' * 70}\n[{ts}] step={step} pid={os.getpid()} {label or component}"
    parts = [head]
    if note:
        parts.append(f"-- note: {note}")
    if inputs is not None:
        parts.append("-- INPUT:")
        parts.append(_render(inputs, max_chars))
    if error is not None:
        suffix = f" ({duration_s:.2f}s)" if duration_s is not None else ""
        parts.append(f"-- ERROR{suffix}:")
        parts.append(_render(error, max_chars))
    elif output is not None:
        suffix = f" ({duration_s:.2f}s)" if duration_s is not None else ""
        parts.append(f"-- OUTPUT{suffix}:")
        parts.append(_render(output, max_chars))
    _append(path, "\n".join(parts) + "\n\n")


def logged(component: Optional[str] = None, max_chars: int = _MAX) -> Callable:
    """Decorator: log each call's inputs (bound by arg name) and return value to
    ``<component>.log`` (defaults to the function name).  Exceptions are logged then re-raised.
    Zero overhead beyond a cheap env check when logging is disabled."""
    def deco(fn: Callable) -> Callable:
        comp = component or fn.__name__
        label = getattr(fn, "__qualname__", fn.__name__)
        try:
            sig: Optional[inspect.Signature] = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            if sig is not None:
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    inputs = dict(bound.arguments)
                    inputs.pop("self", None)
                    inputs.pop("cls", None)
                except Exception:
                    inputs = {"args": list(args), "kwargs": kwargs}
            else:
                inputs = {"args": list(args), "kwargs": kwargs}
            t0 = time.time()
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                record(comp, label=label, inputs=inputs,
                       error=f"{type(e).__name__}: {e}", duration_s=time.time() - t0,
                       max_chars=max_chars)
                raise
            record(comp, label=label, inputs=inputs, output=out,
                   duration_s=time.time() - t0, max_chars=max_chars)
            return out

        return wrapper

    return deco
