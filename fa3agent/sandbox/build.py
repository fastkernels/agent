"""Candidate .cu -> .so builder (component: build).

Compiles a single self-contained Hopper (sm_90a) candidate `.cu` into a torch op
library using TOOLKIT-ONLY includes (torch + CUDA; NEVER a cutlass/cute path), via
`torch.utils.cpp_extension.load`.  On success it loads the resulting `.so`, dumps the
registered `fwd` schema and compares arity/types against the EXACT contract in
`config.FWD_SCHEMA` (warning -- not failing -- on a mismatch, since the schema string is
the true gate during the real vLLM swap).  On failure it distills the FIRST real nvcc
error out of the (truncated) compile log and classifies env-fatal conditions so the
optimization loop aborts instead of burning Claude rounds.

CONTEXT DISCIPLINE: returns a bounded `common.BuildResult` -- the compile log is
truncated and the error field is a single distilled line, never the raw multi-thousand
line ninja/nvcc dump.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
import time
from typing import Optional

from .. import common
from .. import config as C
from .. import llm
from ..runlog import logged

# Bounds for the distilled BuildResult (context discipline).
_LOG_HEAD = 4000        # chars kept from the start of the compile log
_LOG_TAIL = 8000        # chars kept from the end (where ninja prints the real error)
_ERROR_MAX = 600        # max chars of the distilled single error message


# --------------------------------------------------------------------------------------
# log handling
# --------------------------------------------------------------------------------------
def _truncate_log(log: str) -> str:
    """Keep a bounded head+tail of the compile log; ninja errors live near the tail."""
    if log is None:
        return ""
    if len(log) <= _LOG_HEAD + _LOG_TAIL:
        return log
    head = log[:_LOG_HEAD]
    tail = log[-_LOG_TAIL:]
    return f"{head}\n... [truncated {len(log) - _LOG_HEAD - _LOG_TAIL} chars] ...\n{tail}"


# Patterns that mark a *real* compiler error (vs informational noise). Ordered by priority.
_ERROR_PATTERNS = [
    re.compile(r"^.*?:\s*error:\s.*$", re.IGNORECASE | re.MULTILINE),       # gcc/nvcc "file: error: ..."
    re.compile(r"^.*?\berror\s*#\d+:.*$", re.IGNORECASE | re.MULTILINE),    # nvcc "error #NNN: ..."
    re.compile(r"^.*\bfatal error\b.*$", re.IGNORECASE | re.MULTILINE),     # "fatal error: X.h: No such file"
    re.compile(r"^.*\bundefined reference\b.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^.*\bundefined symbol\b.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^.*\bld(?:\.[a-z]+)?:\s*.*$", re.IGNORECASE | re.MULTILINE),  # linker errors
]


def _distill_error(log: str) -> str:
    """Extract the FIRST meaningful compiler/linker error line from a noisy log."""
    if not log:
        return "build failed (no log captured)"
    best: Optional[tuple[int, str]] = None  # (position, line)
    for pat in _ERROR_PATTERNS:
        m = pat.search(log)
        if m:
            cand = (m.start(), m.group(0).strip())
            if best is None or cand[0] < best[0]:
                best = cand
        # First pattern (": error:") is the strongest; if it matched, prefer it outright.
        if best is not None and pat is _ERROR_PATTERNS[0] and m:
            break
    if best is not None:
        msg = best[1]
    else:
        # No structured error -> fall back to the last non-empty line (ninja summary).
        lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
        msg = lines[-1] if lines else "build failed (unrecognized error)"
    if len(msg) > _ERROR_MAX:
        msg = msg[:_ERROR_MAX] + " ..."
    return msg


# --------------------------------------------------------------------------------------
# schema comparison
# --------------------------------------------------------------------------------------
def _parse_schema_args(schema_str: str) -> list[tuple[str, str]]:
    """Parse a torch op schema DEF string into a list of (type, name) pairs.

    Accepts both our config `FWD_SCHEMA` form ("fwd(Tensor! q, Tensor k, ...) -> ...")
    and a normalized live schema ("ns::fwd(Tensor($0! -> ) q, ...) -> ..."). Annotation
    sugar (alias sets like `($0! -> )`, the bang `!`) is normalized away; we keep the
    base type token (e.g. "Tensor", "Tensor?", "int?", "float", "bool?").
    """
    # Grab the parenthesized argument list (first '(' ... matching final ')' before '->').
    open_i = schema_str.find("(")
    arrow = schema_str.rfind("->")
    close_i = schema_str.rfind(")", 0, arrow if arrow != -1 else len(schema_str))
    if open_i == -1 or close_i == -1 or close_i < open_i:
        return []
    body = schema_str[open_i + 1:close_i]
    args: list[tuple[str, str]] = []
    # Split top-level commas (no nesting expected in these schemas beyond alias parens).
    depth = 0
    cur = []
    parts = []
    for ch in body:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))

    for raw in parts:
        tok = raw.strip()
        if not tok:
            continue
        # Strip alias-annotation sugar: "Tensor($0! -> )" -> "Tensor", "Tensor($1! -> )?" -> "Tensor?"
        tok = re.sub(r"\([^)]*\)", "", tok)
        # token now like "Tensor!  q", "Tensor?  out", "int?     max_seqlen_q", "float scale"
        m = re.match(r"^([A-Za-z_][\w]*\??)\s*(!)?\s*(\??)\s+([A-Za-z_]\w*)", tok)
        if not m:
            # Fallback: split on whitespace, last token = name, rest = type.
            bits = tok.split()
            if len(bits) < 2:
                continue
            name = bits[-1]
            typ = "".join(bits[:-1])
        else:
            base, _bang, opt2, name = m.groups()
            typ = base + (opt2 or "")
        typ = typ.replace("!", "").strip()
        args.append((typ, name))
    return args


def _compare_schema(live_schema_str: str) -> Optional[str]:
    """Compare a live fwd schema against config.FWD_SCHEMA. Return a warning string on
    mismatch, or None if arity + (type,name) sequence match."""
    want = _parse_schema_args(C.FWD_SCHEMA)
    got = _parse_schema_args(live_schema_str)
    if len(want) != len(got):
        return f"fwd arity mismatch: contract={len(want)} args, candidate={len(got)} args"
    diffs = []
    for i, (w, g) in enumerate(zip(want, got)):
        if w[0] != g[0]:
            diffs.append(f"arg{i} '{g[1]}': type {g[0]!r} != contract {w[0]!r}")
        elif w[1] != g[1]:
            diffs.append(f"arg{i}: name {g[1]!r} != contract {w[1]!r}")
    if diffs:
        return "fwd schema type/name mismatch: " + "; ".join(diffs[:4]) + (
            f" (+{len(diffs) - 4} more)" if len(diffs) > 4 else "")
    return None


# --------------------------------------------------------------------------------------
# library loading (handles double-registration)
# --------------------------------------------------------------------------------------
def _load_and_get_schema(so_path: str, namespace: str) -> tuple[Optional[str], Optional[str]]:
    """Load the .so and return (live_schema_str, load_error).

    If the namespace is already registered (loading a second lib under the same op
    namespace double-registers -> torch raises), we treat that as benign IF the op is
    already resolvable: we just read the existing schema.  Returns load_error only when
    the schema cannot be obtained at all.
    """
    import torch

    def _schema_of() -> Optional[str]:
        ns = getattr(torch.ops, namespace, None)
        if ns is None:
            return None
        fwd = getattr(ns, "fwd", None)
        if fwd is None:
            return None
        try:
            return str(fwd.default._schema)
        except Exception:  # noqa: BLE001
            try:
                return str(fwd._schema)
            except Exception:  # noqa: BLE001
                return None

    # Fast path: already registered (e.g. a prior build of this namespace in-process).
    pre = _schema_of()
    try:
        torch.ops.load_library(so_path)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        # Double-registration of the same op namespace raises; tolerate if op resolves.
        if "already" in msg.lower() and ("regist" in msg.lower() or "defined" in msg.lower()):
            sch = _schema_of()
            if sch is not None:
                return sch, None
            return None, f"load_library failed (already-registered, op unresolved): {msg[:300]}"
        # Some torch builds raise the duplicate def from the static ctor with other wording.
        sch = _schema_of()
        if sch is not None and pre is None:
            # The op appeared despite the raise -> registration actually succeeded.
            return sch, None
        return None, f"load_library failed: {msg[:300]}"

    sch = _schema_of()
    if sch is None:
        return None, f"loaded {os.path.basename(so_path)} but torch.ops.{namespace}.fwd not registered"
    return sch, None


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
@logged("build")
def build_candidate(cu_path: str, namespace: str, work_dir: str,
                    extra_defs: Optional[list[str]] = None) -> common.BuildResult:
    """Compile `cu_path` into a torch op library registering namespace `namespace`.

    Returns a distilled `common.BuildResult` (bounded log + single distilled error).
    """
    t0 = time.time()

    if not os.path.isfile(cu_path):
        return common.BuildResult(
            ok=False, namespace=namespace,
            error=f"source not found: {cu_path}",
            duration_s=time.time() - t0)

    os.makedirs(work_dir, exist_ok=True)

    extra_cuda_cflags = (
        list(C.NVCC_BASE_FLAGS)
        + [f"-DFA3_LIB_NAME={namespace}"]
        + [f"-D{d}" for d in C.SEMANTIC_DEFS]
        + list(extra_defs or [])
    )

    # Capture the verbose ninja/nvcc output. cpp_extension prints to Python's
    # sys.stdout/stderr AND ninja writes to the underlying fds; redirect both.
    log_buf = io.StringIO()
    so_path: Optional[str] = None
    build_exc: Optional[BaseException] = None

    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import load as cpp_load
    except Exception as e:  # noqa: BLE001
        return common.BuildResult(
            ok=False, namespace=namespace,
            error=f"torch import failed: {e!r}",
            env_fatal=llm.is_env_fatal(str(e)),
            duration_s=time.time() - t0)

    with _capture_fds(log_buf):
        try:
            cpp_load(
                name=namespace,
                sources=[cu_path],
                is_python_module=False,
                build_directory=work_dir,
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=True,
            )
        except BaseException as e:  # noqa: BLE001  (catch ninja CalledProcessError etc.)
            build_exc = e

    log = _truncate_log(log_buf.getvalue())

    # Locate the produced .so regardless of success (load() writes <name>.so on success).
    cand_so = os.path.join(work_dir, f"{namespace}.so")
    if os.path.isfile(cand_so):
        so_path = cand_so

    if build_exc is not None or so_path is None:
        # Compile (or link) failed. Distill the first real error; classify env-fatal.
        raw = log_buf.getvalue()
        exc_text = "" if build_exc is None else f"\n{type(build_exc).__name__}: {build_exc}"
        distilled = _distill_error(raw + exc_text)
        env_fatal = llm.is_env_fatal(raw + exc_text) or llm.is_env_fatal(distilled)
        return common.BuildResult(
            ok=False, so_path=so_path, namespace=namespace, log=log,
            error=distilled, env_fatal=env_fatal,
            duration_s=time.time() - t0)

    # Build succeeded -> load and verify the schema.
    live_schema, load_err = _load_and_get_schema(so_path, namespace)
    if load_err is not None:
        env_fatal = llm.is_env_fatal(load_err)
        return common.BuildResult(
            ok=False, so_path=so_path, namespace=namespace, log=log,
            error=load_err, env_fatal=env_fatal,
            duration_s=time.time() - t0)

    schema_warning = _compare_schema(live_schema or "")
    # A schema mismatch is a WARN (per task) -- build still ok=True, but surfaced in error.
    err = "" if schema_warning is None else f"WARN: {schema_warning}"

    return common.BuildResult(
        ok=True, so_path=so_path, namespace=namespace, log=log,
        error=err, env_fatal=False,
        duration_s=time.time() - t0)


# --------------------------------------------------------------------------------------
# fd-level capture (ninja/nvcc write to the OS stdout/stderr, not just Python's)
# --------------------------------------------------------------------------------------
@contextlib.contextmanager
def _capture_fds(buf: io.StringIO):
    """Redirect both Python-level and OS-level stdout/stderr into `buf` for the block.

    cpp_extension prints with `print(...)` (Python stream) while ninja subprocess output
    goes to the inherited fds 1/2 -- both must be captured to get the real nvcc error.
    """
    # Python-level redirect.
    py_redirect = contextlib.ExitStack()
    # OS-level redirect via a temp file we drain into `buf` afterwards.
    tmp = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    saved_out_fd = os.dup(1)
    saved_err_fd = os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        py_redirect.enter_context(contextlib.redirect_stdout(buf))
        py_redirect.enter_context(contextlib.redirect_stderr(buf))
        yield
    finally:
        py_redirect.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out_fd, 1)
        os.dup2(saved_err_fd, 2)
        os.close(saved_out_fd)
        os.close(saved_err_fd)
        try:
            tmp.flush()
            tmp.seek(0)
            buf.write(tmp.read())
        except Exception:  # noqa: BLE001
            pass
        finally:
            tmp.close()
