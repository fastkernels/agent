"""Cross-step memory across fresh `claude -p` sessions (Workstream E).

Each step runs a COLD session (no `--resume` by default), so the only durable continuity is what
the orchestrator writes to disk and re-injects.  This module owns a per-run ``MEMORY.md`` (+ a JSON
state sidecar that is the source of truth) living in the LIVE workdir (``$WORK_ROOT/<run>/``),
mirrored to the archive by F3 and restored on resume.  It holds:

  - the E1 causal lessons (what changed -> hypothesis -> outcome delta -> corrected cause),
    append-then-compact so the file stays bounded;
  - current state: champion F / min_ratio / bottleneck, a lever ledger (filled by B3 later),
    branch status (filled by C2 later), and a de-duplicated list of dead-ends not to retry.

The orchestrator injects a BOUNDED digest of this into every refine prompt (E2), and also hands the
brain ``--add-dir`` access to the workdir so it can open the FULL ``MEMORY.md`` on demand (E3).
"""
from __future__ import annotations

import json
import os
from typing import Optional

# Keep the file (and especially the injected digest) bounded so it can't crowd out the prompt.
_MAX_LESSONS = 14          # verbatim recent lessons kept in state/MEMORY.md
_MAX_DEAD_ENDS = 24        # de-duplicated negative-memory entries
_DIGEST_LESSONS = 6        # lessons shown in the per-prompt digest
_DIGEST_DEAD_ENDS = 6      # dead-ends shown in the per-prompt digest
_LESSON_FIELD_CHARS = 240  # per-field clamp


def _clip(s, n=_LESSON_FIELD_CHARS) -> str:
    s = "" if s is None else str(s)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "\u2026"


class MemoryStore:
    """Owns ``<run_dir>/MEMORY.md`` and ``<run_dir>/memory_state.json`` (the source of truth)."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.md_path = os.path.join(run_dir, "MEMORY.md")
        self.state_path = os.path.join(run_dir, "memory_state.json")
        self.state = self._load()

    # ---- persistence ----------------------------------------------------------------
    def _default_state(self) -> dict:
        return {
            "champion": None,        # {"cid","step","F","min_ratio"}
            "bottleneck": None,      # active bottleneck label (champion's)
            "lessons": [],           # [{step,cid,change,hypothesis,outcome,cause,promoted}]
            "dead_ends": [],         # ["..."] de-duplicated
            "lever_ledger": {},      # reserved for B3
            "branch": None,          # reserved for C2
        }

    def _load(self) -> dict:
        try:
            with open(self.state_path) as fh:
                st = json.load(fh)
            base = self._default_state()
            base.update(st or {})
            return base
        except Exception:
            return self._default_state()

    def _save(self):
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.state, fh, indent=2)
            os.replace(tmp, self.state_path)
            self._render_md()
        except Exception:
            pass  # memory is best-effort; never crash the loop on a write failure

    # ---- mutation ---------------------------------------------------------------------
    def set_champion(self, cid, step, F, min_ratio, bottleneck=None):
        self.state["champion"] = {"cid": cid, "step": step,
                                  "F": round(float(F), 4), "min_ratio": round(float(min_ratio), 4)}
        if bottleneck:
            self.state["bottleneck"] = _clip(bottleneck, 120)

    def set_bottleneck(self, bottleneck):
        if bottleneck:
            self.state["bottleneck"] = _clip(bottleneck, 120)

    def add_lesson(self, lesson: dict):
        """`lesson`: {step,cid,change,hypothesis,outcome,cause,promoted}.  Appends + compacts."""
        l = {
            "step": lesson.get("step"),
            "cid": lesson.get("cid"),
            "change": _clip(lesson.get("change")),
            "hypothesis": _clip(lesson.get("hypothesis")),
            "outcome": _clip(lesson.get("outcome")),
            "cause": _clip(lesson.get("cause")),
            "promoted": bool(lesson.get("promoted")),
        }
        self.state["lessons"].append(l)
        # compact: keep only the most recent _MAX_LESSONS verbatim
        if len(self.state["lessons"]) > _MAX_LESSONS:
            self.state["lessons"] = self.state["lessons"][-_MAX_LESSONS:]

    def add_dead_end(self, text: str):
        text = _clip(text, 200)
        if not text:
            return
        des = self.state["dead_ends"]
        if text not in des:
            des.append(text)
            if len(des) > _MAX_DEAD_ENDS:
                self.state["dead_ends"] = des[-_MAX_DEAD_ENDS:]

    def set_lever_ledger(self, ledger: dict):
        self.state["lever_ledger"] = ledger or {}

    def set_branch(self, branch: Optional[dict]):
        self.state["branch"] = branch

    def commit(self):
        """Persist state + re-render MEMORY.md.  Call once per step after updates."""
        self._save()

    # ---- rendering / injection --------------------------------------------------------
    def _champ_line(self) -> str:
        c = self.state.get("champion")
        if not c:
            return "champion: (none yet)"
        return (f"champion: {c.get('cid')} @ step {c.get('step')} "
                f"F={c.get('F')} min_ratio={c.get('min_ratio')}")

    def _render_md(self):
        st = self.state
        lines = ["# Run memory (auto-maintained; cross-step continuity for fresh sessions)", ""]
        lines.append("## Current state")
        lines.append(f"- {self._champ_line()}")
        lines.append(f"- active bottleneck: {st.get('bottleneck') or '(unknown)'}")
        if st.get("branch"):
            lines.append(f"- branch: {json.dumps(st['branch'])[:300]}")
        if st.get("lever_ledger"):
            lines.append(f"- lever ledger: {json.dumps(st['lever_ledger'])[:400]}")
        lines.append("")
        lines.append("## Dead-ends (do NOT retry the same way)")
        if st.get("dead_ends"):
            for d in st["dead_ends"]:
                lines.append(f"- {d}")
        else:
            lines.append("- (none recorded)")
        lines.append("")
        lines.append("## Causal lessons (most recent last)")
        if st.get("lessons"):
            for l in st["lessons"]:
                tag = "PROMOTED" if l.get("promoted") else "rejected"
                lines.append(f"- step {l.get('step')} [{tag}] change: {l.get('change')}")
                if l.get("hypothesis"):
                    lines.append(f"    hypothesis: {l.get('hypothesis')}")
                if l.get("outcome"):
                    lines.append(f"    outcome: {l.get('outcome')}")
                if l.get("cause"):
                    lines.append(f"    cause/next: {l.get('cause')}")
        else:
            lines.append("- (none yet)")
        lines.append("")
        try:
            tmp = self.md_path + ".tmp"
            with open(tmp, "w") as fh:
                fh.write("\n".join(lines))
            os.replace(tmp, self.md_path)
        except Exception:
            pass

    def digest(self) -> str:
        """A BOUNDED (~<=15 line) digest injected into EVERY refine prompt (E2)."""
        st = self.state
        out = [self._champ_line(),
               f"active bottleneck: {st.get('bottleneck') or '(unknown)'}"]
        des = st.get("dead_ends") or []
        if des:
            out.append("AVOID (already tried, did not work):")
            for d in des[-_DIGEST_DEAD_ENDS:]:
                out.append(f"  - {d}")
        lessons = st.get("lessons") or []
        if lessons:
            out.append("Recent lessons:")
            for l in lessons[-_DIGEST_LESSONS:]:
                tag = "promoted" if l.get("promoted") else "rejected"
                line = f"  - s{l.get('step')} [{tag}] {l.get('change')}"
                if l.get("cause"):
                    line += f" -> {l.get('cause')}"
                out.append(_clip(line, 320))
        return "\n".join(out)
