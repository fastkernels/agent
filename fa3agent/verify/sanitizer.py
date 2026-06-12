"""Component 2.2 (ON-DEMAND): run a candidate .so under NVIDIA compute-sanitizer.

A single forward call is executed in a throwaway subprocess under one of the four
sanitizer tools (memcheck / racecheck / initcheck / synccheck).  The raw sanitizer
dump (hundreds of `=====` lines with backtraces) is parsed down to a tiny, bounded
result so the orchestrator/LLM never sees the firehose:

    {tool, clean: bool, num_errors: int, errors: [<=5 short strings], note}

This is deliberately NOT part of the hot loop -- compute-sanitizer slows kernels by
1-2 orders of magnitude, so it is invoked only when correctness/perf signals point at
a memory/race/init/sync bug.  Runtime is bounded with the smallest dev config and a
hard timeout.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import Optional

from .. import config as C
from ..runlog import logged

# Sanitizer tools we support.  All take `--tool <name>`.
_VALID_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")

# Hard wall-clock cap for the whole subprocess.  Sanitizer is slow but a single fwd on
# the tiny dev config finishes well within this on an H200.
_TIMEOUT_S = 240

# Bound on how many error headlines we surface (context discipline).
_MAX_ERRORS = 5

# Sanitizer permission / environment signatures -> we cannot conclude clean/dirty.
_PERM_SIGNATURES = (
    "ERR_NVGPUCTRPERM",
    "insufficient permissions",
    "does not have permission",
    "requires root",
)

# A line that begins an error/hazard report headline.  The body lines are indented
# ("=====     at ...").  We only keep the *headline* of each report.
_HEADLINE_RE = re.compile(
    r"^=+\s+(ERROR|WARNING|Invalid|Uninitialized|Potential|Barrier|Malloc|Program|"
    r"Misaligned|Leaked|Race|Deadlock|Internal Sanitizer|Tensor Memory|Warpgroup)\b",
    re.IGNORECASE,
)

# Summary lines emitted by the tools, e.g. "ERROR SUMMARY: 2 errors" /
# "RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)".
_SUMMARY_RE = re.compile(r"=+\s*([A-Z]+)\s+SUMMARY:\s*(.+)$")
_SUMMARY_COUNT_RE = re.compile(r"(\d+)\s+(error|hazard)s?", re.IGNORECASE)


def _driver_source(so_path: str, namespace: str, cfg: C.Config) -> str:
    """Standalone python driver: load the .so, run ONE fwd, synchronize, exit."""
    cfg_kwargs = ", ".join(f"{k}={v!r}" for k, v in cfg.to_dict().items())
    return (
        "import torch\n"
        "import fa3agent.config as C\n"
        "import fa3agent.shapes as S\n"
        "\n"
        "def main():\n"
        f"    torch.ops.load_library({so_path!r})\n"
        f"    cfg = C.Config({cfg_kwargs})\n"
        '    inp = S.make_inputs(cfg, seed=0, device="cuda")\n'
        f"    out, lse = S.call_fwd({namespace!r}, inp)\n"
        "    torch.cuda.synchronize()\n"
        "    _ = float(out.float().sum().item())  # keep outputs live\n"
        '    print("DRIVER_OK", out.shape, lse.shape)\n'
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


def _parse(stdout: str, stderr: str) -> tuple[bool, int, list[str], Optional[str]]:
    """Return (clean, num_errors, headlines[<=5], perm_note_or_None).

    Counting strategy (robust to tool quirks):
      * Prefer the tool's own "<X> SUMMARY: N error(s)/hazard(s)" line when present.
      * Otherwise count distinct error/hazard report headlines.
    """
    text = stdout + "\n" + stderr

    # Permission / environment block -> inconclusive.
    low = text.lower()
    for sig in _PERM_SIGNATURES:
        if sig.lower() in low:
            return False, 0, [], f"compute-sanitizer permission/env blocked ({sig})"

    headlines: list[str] = []
    summary_count: Optional[int] = None
    saw_summary = False

    for raw in text.splitlines():
        line = raw.rstrip()
        m = _SUMMARY_RE.search(line)
        if m:
            saw_summary = True
            cm = _SUMMARY_COUNT_RE.search(m.group(2))
            if cm:
                # Take the max across summary lines (memcheck + leak summaries etc.).
                n = int(cm.group(1))
                summary_count = n if summary_count is None else max(summary_count, n)
            continue
        if _HEADLINE_RE.search(line):
            cleaned = re.sub(r"^=+\s*", "", line).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if len(cleaned) > 140:
                cleaned = cleaned[:137] + "..."
            headlines.append(cleaned)

    # Dedup headlines preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for h in headlines:
        if h not in seen:
            seen.add(h)
            uniq.append(h)

    if summary_count is not None:
        num_errors = summary_count
    else:
        num_errors = len(uniq)

    clean = (num_errors == 0) and (not uniq)
    return clean, num_errors, uniq[:_MAX_ERRORS], None


@logged("sanitizer")
def run_sanitizer(so_path: str, namespace: str, cfg: C.Config,
                  tool: str = "memcheck") -> dict:
    """Run ONE forward of `cfg` under compute-sanitizer `tool`; return a distilled dict.

    Returns a dict shaped:
        {tool, clean: bool, num_errors: int, errors: [<=5 short strings], note}
    `clean` is True only when the tool ran the kernel to completion and reported zero
    errors/hazards.  If the sanitizer is permission-blocked or the run failed for an
    environmental reason, `clean` is False and `note` explains why (the caller should
    treat that as inconclusive, not as a kernel bug).
    """
    if tool not in _VALID_TOOLS:
        return {"tool": tool, "clean": False, "num_errors": 0, "errors": [],
                "note": f"invalid tool {tool!r}; expected one of {_VALID_TOOLS}"}
    if not os.path.exists(so_path):
        return {"tool": tool, "clean": False, "num_errors": 0, "errors": [],
                "note": f".so not found: {so_path}"}

    with tempfile.TemporaryDirectory(prefix="fa3_sanit_") as td:
        driver = os.path.join(td, "driver.py")
        with open(driver, "w") as f:
            f.write(_driver_source(so_path, namespace, cfg))

        cmd = [
            "compute-sanitizer",
            "--tool", tool,
            "--error-exitcode", "1",
            "--target-processes", "all",
            sys.executable, driver,
        ]
        env = dict(os.environ)
        # Run python from the repo root so `import fa3agent...` resolves.
        existing = env.get("PYTHONPATH", "")
        if C.REPO_ROOT not in existing.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join(
                p for p in (C.REPO_ROOT, existing) if p)

        try:
            proc = subprocess.run(
                cmd, cwd=C.REPO_ROOT, env=env,
                capture_output=True, text=True, timeout=_TIMEOUT_S,
            )
        except FileNotFoundError:
            return {"tool": tool, "clean": False, "num_errors": 0, "errors": [],
                    "note": "compute-sanitizer not found on PATH"}
        except subprocess.TimeoutExpired:
            return {"tool": tool, "clean": False, "num_errors": 0, "errors": [],
                    "note": f"timeout after {_TIMEOUT_S}s (sanitizer too slow / hang)"}

        stdout, stderr = proc.stdout or "", proc.stderr or ""
        clean, num_errors, errors, perm = _parse(stdout, stderr)

        if perm is not None:
            return {"tool": tool, "clean": False, "num_errors": 0,
                    "errors": [], "note": perm}

        # The driver must have actually executed the kernel; otherwise the run is
        # inconclusive (load/import/env failure rather than a clean kernel).
        driver_ran = "DRIVER_OK" in stdout
        if not driver_ran and clean:
            tail = (stderr or stdout).strip().splitlines()[-3:]
            note = "driver did not complete; " + (
                " | ".join(s.strip() for s in tail) if tail
                else f"exit={proc.returncode}")
            if len(note) > 200:
                note = note[:197] + "..."
            return {"tool": tool, "clean": False, "num_errors": 0,
                    "errors": [], "note": note}

        note = "clean" if clean else f"{num_errors} issue(s) reported by {tool}"
        return {"tool": tool, "clean": clean, "num_errors": num_errors,
                "errors": errors, "note": note}
