"""Shared result/record dataclasses used across the orchestrator and components.

Keeping these in one place lets the parallel-built components and the orchestrator
agree on a stable contract.  Everything here is JSON-serializable (for checkpoints).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Any


def now() -> float:
    return time.time()


@dataclass
class BuildResult:
    ok: bool
    so_path: Optional[str] = None
    namespace: Optional[str] = None
    log: str = ""              # truncated compile log
    error: str = ""            # distilled error (compile / schema-mismatch)
    env_fatal: bool = False    # unfixable-by-agent (ABI, no device, OOM, ...)
    duration_s: float = 0.0


@dataclass
class AntiCheatResult:
    ok: bool
    violations: list[str] = field(default_factory=list)   # human-readable reasons
    scanned_source: bool = False
    scanned_binary: bool = False


@dataclass
class ConfigResult:
    """Per-config correctness + perf outcome."""
    config: str
    correct: bool
    # correctness evidence
    out_err: float = 0.0        # candidate-vs-fp32ref error
    baseline_err: float = 0.0   # baseline-vs-fp32ref error
    lse_err: float = 0.0
    err_ratio: float = 0.0      # out_err / baseline_err (<= ERR_MULT passes)
    bias: float = 0.0           # mean|cand_out - base_out| (systematic-drift gate)
    # perf evidence (only meaningful if correct)
    ratio: float = 0.0          # baseline_latency / candidate_latency  (>=1 == match/exceed)
    cand_latency_ms: float = 0.0
    base_latency_ms: float = 0.0
    note: str = ""


@dataclass
class EvalResult:
    """Aggregate evaluation of a candidate across the config matrix."""
    correct: bool = False                 # all configs pass correctness
    num_configs: int = 0
    num_correct: int = 0
    min_ratio: float = 0.0                # the metric we optimize (worst-case speedup)
    geomean_ratio: float = 0.0
    weighted_geomean_ratio: float = 0.0
    per_config: list[ConfigResult] = field(default_factory=list)
    decode_equiv_ok: Optional[bool] = None  # 100-token greedy equality (promotion gate)
    decode_equiv_note: str = ""
    error: str = ""

    def summary(self) -> str:
        return (f"correct={self.correct} ({self.num_correct}/{self.num_configs}) "
                f"min_ratio={self.min_ratio:.3f} geomean={self.geomean_ratio:.3f} "
                f"decode_equiv={self.decode_equiv_ok}")


@dataclass
class ProfileFeedback:
    """Distilled NCU feedback (§2.1).  `terse` drives the next codegen round; `ranked`
    is the human-readable knowledge note.  NEVER the raw 2000-metric dump."""
    config: str = ""
    bottleneck: str = ""          # <=30 words, with cited metric values
    optimization: str = ""        # <=35 words
    plan: str = ""                # <=35 words
    ranked: list[str] = field(default_factory=list)  # <=5 evidence-cited priorities
    key_metrics: dict[str, Any] = field(default_factory=dict)
    report_path: str = ""


@dataclass
class Candidate:
    """One node in the evolution DAG."""
    cid: str                       # candidate id, e.g. "step007"
    step: int
    parent: Optional[str]
    cu_path: str                   # path to the candidate .cu
    status: str = "generated"      # generated|build_failed|cheat|incorrect|correct|promoted
    eval: Optional[dict] = None    # asdict(EvalResult)
    notes: str = ""
    created: float = field(default_factory=now)


@dataclass
class StepRecord:
    """Full per-step checkpoint record (one line in candidates.jsonl)."""
    step: int
    cid: str
    parent: Optional[str]
    status: str
    build: Optional[dict] = None
    anticheat: Optional[dict] = None
    eval: Optional[dict] = None
    profile: Optional[dict] = None
    sass: Optional[dict] = None
    knowledge_note: Optional[str] = None
    focus: Optional[str] = None        # the steer/lever applied this step (E1 causal lessons)
    lever_id: Optional[str] = None     # id of the B3 lever applied this step (outcome tracking)
    completion_attempts: int = 0       # in-step completion-gate follow-up turns spent this step
    session_id: Optional[str] = None   # Claude headless session for resume continuity
    is_best: bool = False
    timestamp: float = field(default_factory=now)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def dumps(obj) -> str:
    """asdict-aware JSON dump for any of the dataclasses above."""
    try:
        return json.dumps(asdict(obj))
    except TypeError:
        return json.dumps(obj)
