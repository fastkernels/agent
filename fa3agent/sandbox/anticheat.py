"""Anti-cheating enforcement (component 2.5).

A candidate `.cu` is the loop's product; it MUST be a genuinely self-contained Hopper
kernel.  This component rejects any candidate that smuggles in a forbidden library (the
whole point of the distillation is to NOT lean on CUTLASS/CuTe/cuBLAS/cuDNN/flashinfer/
ThunderKittens/Triton, and to NOT link the original `_vllm_fa3_C`).

Two scans:
  (1) Source scan  -- read `cu_path`, strip line + block comments, then flag any
      `config.FORBIDDEN_INCLUDE_SUBSTR` appearing on an `#include` line and any
      `config.FORBIDDEN_SOURCE_TOKENS` appearing anywhere in the (comment-stripped) code.
  (2) Binary scan  -- if `so_path` is given, run `nm -D` (dynamic symbols) and `ldd`
      (shared-object deps) and flag any `config.FORBIDDEN_DYNAMIC_SUBSTR`.  torch/c10/
      cuda links are explicitly allowed (`config.ALLOWED_LINK_SUBSTR`).

CONTEXT DISCIPLINE: returns a bounded `common.AntiCheatResult` -- `ok` plus a short list
of human-readable violations, never the raw nm/ldd dump.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

from .. import common
from .. import config as C
from ..runlog import logged

_MAX_VIOLATIONS = 30          # bound the violations list (context discipline)
_NM_LDD_TIMEOUT = 60          # seconds per binary-scan subprocess


# --------------------------------------------------------------------------------------
# comment stripping
# --------------------------------------------------------------------------------------
def _strip_comments(src: str) -> str:
    """Remove C/C++ block (/* */) and line (//) comments, preserving string literals.

    We must NOT strip inside string/char literals (e.g. a kernel might legitimately have
    an error message), and we keep newlines so `#include` line detection still works.
    """
    out = []
    i = 0
    n = len(src)
    state = "code"  # code | line_comment | block_comment | string | char
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block_comment"
                i += 2
                continue
            if c == '"':
                state = "string"
                out.append(c)
                i += 1
                continue
            if c == "'":
                state = "char"
                out.append(c)
                i += 1
                continue
            out.append(c)
            i += 1
        elif state == "line_comment":
            if c == "\n":
                state = "code"
                out.append(c)  # keep the newline
            i += 1
        elif state == "block_comment":
            if c == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
            if c == "\n":
                out.append(c)  # keep newlines so line structure / #include detection holds
            i += 1
        elif state == "string":
            out.append(c)
            if c == "\\":
                # keep escaped char verbatim
                if nxt:
                    out.append(nxt)
                    i += 2
                    continue
            elif c == '"':
                state = "code"
            i += 1
        elif state == "char":
            out.append(c)
            if c == "\\":
                if nxt:
                    out.append(nxt)
                    i += 2
                    continue
            elif c == "'":
                state = "code"
            i += 1
    return "".join(out)


# --------------------------------------------------------------------------------------
# source scan
# --------------------------------------------------------------------------------------
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)

# Patterns that constitute a genuine REFERENCE to the external/original `_vllm_fa3_C` op
# (as opposed to the candidate legitimately registering ITS OWN op under that namespace
# via the FA3_LIB_NAME macro for the real vLLM swap). `\b` is unreliable here because the
# name starts with '_', so we anchor on the surrounding syntax instead.
_EXTERNAL_OP_REF_RES = [
    re.compile(r"torch\.ops\._vllm_fa3_C"),       # python-style op reference
    re.compile(r"\b_vllm_fa3_C::"),                # C++ namespace use of the original
    re.compile(r"\bimport\s+_vllm_fa3_C\b"),       # importing the prebuilt module
    re.compile(r"\bfrom\b[^\n]*\b_vllm_fa3_C\b"),  # from ... import _vllm_fa3_C
    re.compile(r'#\s*include[^\n]*_vllm_fa3_C'),   # including the original header/lib
]


def _external_op_refs(code: str, tok: str) -> list[str]:
    """Flag `_vllm_fa3_C` ONLY when it references the external/original op, not when it
    is the candidate's own registration namespace (FA3_LIB_NAME macro default / TORCH_LIBRARY
    argument). Returns a (possibly empty) list of violation strings."""
    hits: list[str] = []
    for pat in _EXTERNAL_OP_REF_RES:
        m = pat.search(code)
        if m:
            line_no = code.count("\n", 0, m.start()) + 1
            hits.append(
                f"forbidden reference to original op {tok!r} via {m.group(0).strip()!r} "
                f"at line ~{line_no}")
    return hits


def _scan_source(cu_path: str, violations: list[str]) -> None:
    try:
        with open(cu_path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError as e:
        violations.append(f"source unreadable: {e}")
        return

    code = _strip_comments(raw)

    # (a) forbidden includes
    for m in _INCLUDE_RE.finditer(code):
        header = m.group(1)
        low = header.lower()
        for bad in C.FORBIDDEN_INCLUDE_SUBSTR:
            if bad.lower() in low:
                violations.append(f"forbidden include '<{header}>' (matches {bad!r})")
                break

    # (b) forbidden source tokens anywhere in (comment-stripped) code.
    #     Special case: the literal "_vllm_fa3_C" is the candidate's REQUIRED registration
    #     namespace for the real vLLM swap (set as the default of its own FA3_LIB_NAME
    #     macro), so its mere appearance is legitimate. It is forbidden only when it
    #     REFERENCES the external/original op (e.g. torch.ops._vllm_fa3_C, _vllm_fa3_C::,
    #     #include of it, or `import _vllm_fa3_C`). The binary scan (nm/ldd) is the real
    #     guard against actually linking the prebuilt original.
    for tok in C.FORBIDDEN_SOURCE_TOKENS:
        if tok == "_vllm_fa3_C":
            for hit in _external_op_refs(code, tok):
                violations.append(hit)
            continue
        idx = code.find(tok)
        if idx != -1:
            line_no = code.count("\n", 0, idx) + 1
            violations.append(f"forbidden token {tok!r} at line ~{line_no}")


# --------------------------------------------------------------------------------------
# binary scan
# --------------------------------------------------------------------------------------
def _run(cmd: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=_NM_LDD_TIMEOUT)
    except FileNotFoundError:
        return False, f"{cmd[0]} not found"
    except subprocess.TimeoutExpired:
        return False, f"{cmd[0]} timed out"
    except Exception as e:  # noqa: BLE001
        return False, f"{cmd[0]} failed: {e!r}"
    # nm/ldd write useful output to stdout (and sometimes stderr); combine.
    return True, (proc.stdout or "") + (proc.stderr or "")


def _is_allowed(line_low: str) -> bool:
    return any(a.lower() in line_low for a in C.ALLOWED_LINK_SUBSTR)


def _scan_binary(so_path: str, violations: list[str]) -> None:
    if not os.path.isfile(so_path):
        violations.append(f"binary not found: {so_path}")
        return

    seen: set[str] = set()

    # ldd: shared-object dependency names.
    ok_ldd, ldd_out = _run(["ldd", so_path])
    if ok_ldd:
        for line in ldd_out.splitlines():
            low = line.lower()
            for bad in C.FORBIDDEN_DYNAMIC_SUBSTR:
                if bad.lower() in low and not _is_allowed(low):
                    key = f"ldd:{bad}"
                    if key not in seen:
                        seen.add(key)
                        violations.append(
                            f"forbidden shared-lib dep (matches {bad!r}): {line.strip()[:120]}")
                    break

    # nm -D: dynamic (imported/exported) symbols.
    ok_nm, nm_out = _run(["nm", "-D", so_path])
    if ok_nm:
        for line in nm_out.splitlines():
            low = line.lower()
            # Only look at the symbol name portion; ignore address/type columns.
            for bad in C.FORBIDDEN_DYNAMIC_SUBSTR:
                if bad.lower() in low and not _is_allowed(low):
                    key = f"nm:{bad}"
                    if key not in seen:
                        seen.add(key)
                        violations.append(
                            f"forbidden dynamic symbol (matches {bad!r}): {line.strip()[:120]}")
                    break

    if not ok_ldd and not ok_nm:
        violations.append("binary scan unavailable (nm/ldd both failed)")


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
@logged("anticheat")
def check(cu_path: str, so_path: Optional[str] = None) -> common.AntiCheatResult:
    """Scan a candidate's source (and optionally its built `.so`) for forbidden
    libraries/symbols/includes.  ok=False with explicit violations[] if any are found."""
    violations: list[str] = []

    scanned_source = False
    if cu_path and os.path.isfile(cu_path):
        _scan_source(cu_path, violations)
        scanned_source = True
    else:
        violations.append(f"source not found: {cu_path}")

    scanned_binary = False
    if so_path:
        _scan_binary(so_path, violations)
        scanned_binary = True

    # Bound the list for context discipline.
    if len(violations) > _MAX_VIOLATIONS:
        extra = len(violations) - _MAX_VIOLATIONS
        violations = violations[:_MAX_VIOLATIONS] + [f"... (+{extra} more violations)"]

    return common.AntiCheatResult(
        ok=(len(violations) == 0),
        violations=violations,
        scanned_source=scanned_source,
        scanned_binary=scanned_binary,
    )
