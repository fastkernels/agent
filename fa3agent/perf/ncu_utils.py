"""Nsight-Compute report parsing helpers for the FA3-distill agent (component 2.1).

Adapted from `reference/.../ncu-report-skill/helpers/ncu_utils.py` and hardened for
THIS box: H200 / sm_90a / CUDA 13.0 / ncu 2025.3.1.

Two hard constraints discovered this session, baked in here:

  1. `import ncu_report` and `import torch` in the SAME process SEGFAULT (libstdc++/
     C-extension clash).  Therefore this module (which imports ncu_report) must be used
     ONLY in a process that does NOT import torch.  The ncu *driver* (which imports torch
     and runs the kernel) is a separate subprocess and must never import this module.

  2. Several names in `config.NCU_METRIC_SUBSET` (CUDAForge / B200 derived) DRIFT on
     sm_90a.  We never edit config (foundation, frozen).  Instead we resolve each subset
     name to a working Hopper metric via `metric_or_none()` alias chains (`HOPPER_ALIASES`)
     and verify every name with `safe()` at runtime.

This module only PARSES; it returns plain Python values.  `ncu_profile.py` does the
classification + distillation into a bounded `ProfileFeedback`.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .. import config as C


# --- Locate + import ncu_report ------------------------------------------------
def _locate_ncu_report():
    """Return a dir containing ncu_report.py, or None.

    Tries config.NCU_PYTHON_CANDIDATES first (authoritative for this box), then a
    glob probe of the usual install roots (so the helper still works if ncu moves).
    """
    cands: list[str] = list(getattr(C, "NCU_PYTHON_CANDIDATES", []))
    for root in ("/opt/nvidia", "/usr/local", "/opt/cuda"):
        p = Path(root)
        if not p.is_dir():
            continue
        for pat in ("nsight-compute-*/extras/python",
                    "nsight-compute/*/extras/python",
                    "cuda-*/nsight-compute-*/extras/python"):
            for sub in p.glob(pat):
                cands.append(str(sub))
    for c in cands:
        if Path(c).is_dir() and (Path(c) / "ncu_report.py").exists():
            return c
    return None


try:  # already importable?
    import ncu_report  # noqa: F401
except ImportError:
    _found = _locate_ncu_report()
    if _found:
        sys.path.insert(0, _found)
        import ncu_report  # noqa: F401
    else:  # defer the hard failure to the caller with a clear message
        ncu_report = None  # type: ignore


def ncu_report_available() -> bool:
    return ncu_report is not None


# --- Loading -------------------------------------------------------------------
def load_report(path):
    """Load a .ncu-rep and return (report, first_action). Keep the report alive."""
    if ncu_report is None:
        raise RuntimeError("ncu_report module not importable (locate failed)")
    r = ncu_report.load_report(str(path))
    rng = r.range_by_idx(0)        # with -c 1 there is exactly one range...
    action = rng.action_by_idx(0)  # ...and one action (kernel launch)
    return r, action


def load_action(path):
    _, action = load_report(path)
    return action


def load_all_actions(path):
    """Load a .ncu-rep and return (report, [all actions]) across every range.

    Unlike load_report (which reads only the first action), this enumerates every captured
    kernel launch -- used by the timing pass to rank a candidate's subkernels by duration.
    Keep the returned report alive while reading metrics off the actions.
    """
    if ncu_report is None:
        raise RuntimeError("ncu_report module not importable (locate failed)")
    r = ncu_report.load_report(str(path))
    try:
        n_ranges = r.num_ranges()
    except Exception:
        n_ranges = 1
    actions = []
    for ri in range(n_ranges):
        try:
            rng = r.range_by_idx(ri)
        except Exception:
            continue
        try:
            n_act = rng.num_actions()
        except Exception:
            n_act = 1
        for ai in range(n_act):
            try:
                actions.append(rng.action_by_idx(ai))
            except Exception:
                continue
    return r, actions


# --- Safe metric access --------------------------------------------------------
def safe(action, name, default=None):
    """metric value, or `default` if missing/errors. Metric names vary by GPU gen."""
    try:
        return action[name].value()
    except Exception:
        return default


def safe_many(action, names, default=None):
    return {n: safe(action, n, default) for n in names}


def metric_or_none(action, *candidates):
    """First candidate name that resolves to a non-None value (GPU-gen aliasing)."""
    for n in candidates:
        v = safe(action, n, None)
        if v is not None:
            return v
    return None


# --- Value-kind robust per-instance accessor -----------------------------------
def metric_value_at(m, i):
    """Read the i-th instance value regardless of value kind."""
    try:
        k = m.kind()
        if k == m.ValueKind_UINT64:
            return m.as_uint64(i)
        if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
            return m.as_double(i)
        if k == m.ValueKind_STRING:
            return m.as_string(i)
    except Exception:
        pass
    for fn in ("as_double", "as_uint64", "as_string"):
        try:
            return getattr(m, fn)(i)
        except Exception:
            continue
    return None


def per_instance_values(action, metric_name):
    try:
        m = action[metric_name]
        n = m.num_instances()
    except Exception:
        return None
    if not n:
        return None
    return [metric_value_at(m, i) for i in range(n)]


# --- Archive all metrics (once per candidate, so we never reopen the .ncu-rep) --
def dump_all_metrics(action, outfile):
    out = []
    for n in sorted(action.metric_names()):
        rec = {"name": n}
        try:
            m = action[n]
            try:
                rec["value"] = m.value()
            except Exception as e:
                rec["error"] = str(e)
            try:
                rec["unit"] = m.unit()
            except Exception:
                pass
        except Exception as e:
            rec["error"] = str(e)
        out.append(rec)
    Path(outfile).write_text(json.dumps(out, indent=1, default=str))
    return len(out)


# --- PC -> source line mapping (needs -lineinfo + SourceCounters) --------------
def per_pc_values(action, metric_name):
    try:
        m = action[metric_name]
        n = m.num_instances()
    except Exception:
        return []
    if not n or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        try:
            pc = cor.as_uint64(i)
        except Exception:
            try:
                pc = int(cor.as_double(i))
            except Exception:
                pc = None
        out.append((pc, metric_value_at(m, i)))
    return out


def pc_to_source_line(action, pc):
    try:
        si = action.source_info(pc)
        if si is None:
            return "?", 0
        return si.file_name(), si.line()
    except Exception:
        return "?", 0


# --- Hopper (sm_90a / ncu 2025.3) name resolution ------------------------------
#
# config.NCU_METRIC_SUBSET is the contract; several entries don't exist verbatim on
# Hopper.  For each, list working alternates (verified this session against the seed
# kernel report).  `resolve_subset()` returns {subset_name: value} using these chains.
HOPPER_ALIASES: dict[str, list[str]] = {
    # plain .sum doesn't exist on Hopper; use the per-cycle form (instructions/cycle)
    "sm__inst_executed.sum": [
        "sm__inst_executed.sum",
        "sm__inst_executed.sum.per_cycle_active",
        "sm__inst_executed.avg.per_cycle_active",
    ],
    # fp32 pipe util: config uses "fp32" pipe (B200); Hopper exposes the FMA pipe
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active": [
        "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    ],
    # tensor pipe instruction util (B200 name) -> Hopper cycles-active util
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active": [
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    ],
    # tensor cycles active: config asks _elapsed (near-0 on tiny grids); add _active
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    ],
    # DRAM throughput %peak: dram__throughput.* absent -> gpu__dram_throughput.*
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    # stall metrics: config uses *_per_warp_active.pct (B200); Hopper uses
    # *_per_issue_active.ratio (avg warps stalled per issue-active cycle).
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": [
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    ],
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": [
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    ],
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct": [
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    ],
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct": [
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    ],
    "smsp__warp_issue_stalled_membar_per_warp_active.pct": [
        "smsp__warp_issue_stalled_membar_per_warp_active.pct",
        "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    ],
}

# Stall reasons we normalize into "% of stall share" for the bottleneck decision tree.
# Keyed by short name -> alias chain (per_warp_active.pct preferred, .ratio fallback).
STALL_REASONS = [
    "long_scoreboard", "short_scoreboard", "wait", "barrier", "membar",
    "mio_throttle", "lg_throttle", "math_pipe_throttle", "tex_throttle",
    "branch_resolving", "not_selected", "dispatch_stall", "drain",
    "no_instruction", "sleeping", "misc",
]


def _stall_aliases(reason: str) -> list[str]:
    return [
        f"smsp__warp_issue_stalled_{reason}_per_warp_active.pct",
        f"smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio",
    ]


def stall_shares(action) -> dict[str, float]:
    """Return {reason: percent-of-total-stall} over the available stall reasons.

    Works whether the report exposes the B200 `_per_warp_active.pct` form or the
    Hopper `_per_issue_active.ratio` form: both are non-negative magnitudes that we
    normalize to a share, so the 'long_scoreboard > 40%' test in the decision tree
    is gen-independent.  The 'selected' (issued, not stalled) state is excluded.
    """
    raw: dict[str, float] = {}
    for reason in STALL_REASONS:
        v = metric_or_none(action, *_stall_aliases(reason))
        if v is not None:
            try:
                raw[reason] = float(v)
            except Exception:
                pass
    tot = sum(raw.values())
    if tot <= 0:
        return {k: 0.0 for k in raw}
    return {k: 100.0 * v / tot for k, v in raw.items()}


def resolve_subset(action) -> dict[str, object]:
    """Resolve config.NCU_METRIC_SUBSET to {subset_name: value} on this GPU.

    Each subset name is tried verbatim first, then via HOPPER_ALIASES.  Missing
    metrics are recorded as None (caller decides how to render).  This is the ONLY
    metric dict we ever surface upward — never the 2000+ raw dump.
    """
    out: dict[str, object] = {}
    for name in C.NCU_METRIC_SUBSET:
        chain = HOPPER_ALIASES.get(name, [name])
        out[name] = metric_or_none(action, *chain)
    return out


# --- NCU rule-engine results ---------------------------------------------------
def rule_results(action):
    try:
        return list(action.rule_results_as_dicts())
    except Exception:
        return []


_PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
# phrases that mark a % as an Est.Speedup (varies by ncu version)
_SPEEDUP_CTX = re.compile(r"(speed[\s-]?up|speedup|reduction|saving|faster|improv)", re.I)


def _extract_speedup_pct(rr: dict) -> float:
    """Best-effort Est.Speedup % from a rule dict.

    Handles two shapes:
      * documented (older ncu / skill): rr['estimated_speedup_pct'].
      * ncu 2025.3 Python API: no structured field; rr['rule_message'] is a dict
        {title,message,type}.  We scan the message for a %-in-speedup-context; if
        none, fall back to focus_metrics severity (2->mid weight) so high-severity
        rules still sort above informational ones.
    """
    est = rr.get("estimated_speedup_pct")
    if est is not None:
        try:
            return float(est)
        except Exception:
            pass
    msg = rr.get("rule_message")
    text = ""
    if isinstance(msg, dict):
        text = str(msg.get("message", "") or "")
    elif isinstance(msg, str):
        text = msg
    # %-in-speedup-context wins
    best = 0.0
    for m in _PCT_RE.finditer(text):
        s, e = m.start(), m.end()
        ctx = text[max(0, s - 40): min(len(text), e + 40)]
        if _SPEEDUP_CTX.search(ctx):
            try:
                best = max(best, float(m.group(1)))
            except Exception:
                pass
    if best > 0:
        return best
    # severity-based fallback: surface high-severity rules even without a % number
    fms = rr.get("focus_metrics") or []
    try:
        sev = max((int(f.get("severity", 0)) for f in fms), default=0)
    except Exception:
        sev = 0
    return {2: 10.0, 1: 3.0}.get(sev, 0.0)


def _rule_name(rr: dict) -> str:
    return str(rr.get("rule_name") or rr.get("name") or rr.get("rule_identifier") or "?")


def _rule_msg(rr: dict) -> str:
    msg = rr.get("message_for_display")
    if msg:
        return str(msg)
    rm = rr.get("rule_message")
    if isinstance(rm, dict):
        title = rm.get("title") or ""
        body = rm.get("message") or ""
        return (f"{title}: {body}" if title else str(body)).strip()
    return str(rm or "?")


def rule_speedups(action):
    """Return [(est_speedup_pct, rule_name, message)] sorted desc by est_speedup.

    Tolerant of both the documented dict shape and ncu-2025.3's shape.  On ncu 2025.3
    the structured speedup is not exposed, so values are heuristic (parsed from message
    / severity) and should be treated as a *ranking hint*, not ground truth.
    """
    out = []
    for rr in rule_results(action):
        out.append((_extract_speedup_pct(rr), _rule_name(rr), _rule_msg(rr)))
    out.sort(key=lambda x: -x[0])
    return out
