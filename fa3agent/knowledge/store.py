"""KnowledgeStore -- typed in-loop distillation (kernel-specific / profiling / algorithmic).

Notes are markdown files with YAML frontmatter under a notes directory.  Three types:
  ks  kernel-specific (Hopper HW, wgmma, TMA, descriptors, setmaxnreg, ref libs)
  pf  profiling      (NCU diagnosis playbook, bottleneck decision tree, metric subset)
  alg algorithmic    (methodology, FA3 structure, online-softmax, inflection optimizations)

Frontmatter: id (e.g. ks-0007), type, title, tags[], confidence
(verified > source-reported > inferred > experimental), created, step (opt), configs (opt).

This module implements the four entry points the orchestrator / agent uses:
  KnowledgeStore (store.py): the class itself + add_note/query/get/grep/seed
  query.py / get_page.py / grep.py thin wrappers re-export the same logic so callers
  may `from fa3agent.knowledge.query import query` etc.

Search/get/grep logic is adapted (NOT imported) from the KernelWiki skill scripts.

CONTEXT DISCIPLINE: query/grep return small, bounded, distilled results (id/title/score/
path/snippet, capped lists) -- never raw dumps.  get() returns one full note (bounded by
the <=250-word note convention).
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional

import yaml

from fa3agent import config
from fa3agent.runlog import logged

# Confidence ordering (higher = more trustworthy) -- used for tie-breaking in query().
_CONFIDENCE_RANK = {
    "verified": 3,
    "source-reported": 2,
    "inferred": 1,
    "experimental": 0,
}
_NOTE_TYPES = ("ks", "pf", "alg")

# Bounds (context discipline): never return more than this regardless of caller k.
_MAX_K = 20
_SNIPPET_CHARS = 240
_GREP_MAX_FILES = 40

# Directory holding the authored seed notes shipped with the package.
_SEED_DIR = os.path.join(config.PKG_DIR, "knowledge", "seed")
# Default live notes directory (overridable to a per-run checkpoint dir).
_DEFAULT_NOTES_DIR = os.path.join(config.PKG_DIR, "knowledge", "notes")

_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)", re.DOTALL)
_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_ID_RE = re.compile(r"^(ks|pf|alg)-(\d+)$")


def _split_frontmatter(content: str):
    """Parse YAML frontmatter -> (fm_dict, body).  Returns (None, None) on failure."""
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return None, None
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, None
    if not isinstance(fm, dict):
        return None, None
    return fm, m.group(2)


def _strip_code_and_comments(body: str) -> str:
    """Lowercased body text for term scoring (kept simple; full body is scored)."""
    return body.lower()


class KnowledgeStore:
    """File-backed typed knowledge store.

    Parameters
    ----------
    notes_dir : optional path to the live notes directory.  Defaults to
        fa3agent/knowledge/notes.  Pass CHECKPOINT_ROOT/<run>/knowledge for a
        run-scoped store.
    seed_dir : optional override for the authored seed-note directory.
    auto_seed : if True (default) call seed() lazily on first init when the live
        notes directory is empty.
    """

    def __init__(self, notes_dir: Optional[str] = None,
                 seed_dir: Optional[str] = None, auto_seed: bool = True):
        self.notes_dir = notes_dir or _DEFAULT_NOTES_DIR
        self.seed_dir = seed_dir or _SEED_DIR
        os.makedirs(self.notes_dir, exist_ok=True)
        self._seeded = False
        if auto_seed and not self._has_notes():
            self.seed()

    # ------------------------------------------------------------------ helpers
    def _has_notes(self) -> bool:
        try:
            return any(f.endswith(".md") for f in os.listdir(self.notes_dir))
        except FileNotFoundError:
            return False

    def _iter_note_paths(self):
        for name in sorted(os.listdir(self.notes_dir)):
            if name.endswith(".md"):
                yield os.path.join(self.notes_dir, name)

    def _load(self, path: str):
        """Read a note -> (fm, body) or (None, None)."""
        try:
            content = open(path, encoding="utf-8").read()
        except OSError:
            return None, None
        return _split_frontmatter(content)

    def _all_notes(self) -> list[dict]:
        """Load every note: list of {id, type, title, tags, confidence, path, body, fm}."""
        out = []
        for path in self._iter_note_paths():
            fm, body = self._load(path)
            if fm is None:
                continue
            out.append({
                "id": str(fm.get("id", "")),
                "type": str(fm.get("type", "")),
                "title": str(fm.get("title", "")),
                "tags": list(fm.get("tags") or []),
                "confidence": str(fm.get("confidence", "")),
                "path": path,
                "body": body or "",
                "fm": fm,
            })
        return out

    def _next_id(self, ntype: str) -> str:
        """Auto-increment id per type, e.g. ks-0007.  Scans existing ids."""
        max_n = 0
        for note in self._all_notes():
            m = _ID_RE.match(note["id"])
            if m and m.group(1) == ntype:
                max_n = max(max_n, int(m.group(2)))
        return f"{ntype}-{max_n + 1:04d}"

    def _path_for(self, note_id: str, title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48] or "note"
        return os.path.join(self.notes_dir, f"{note_id}-{slug}.md")

    def _find_by_id(self, note_id: str) -> Optional[dict]:
        for note in self._all_notes():
            if note["id"] == note_id:
                return note
        return None

    # -------------------------------------------------------------------- seed
    def seed(self) -> int:
        """Copy authored seed notes from seed_dir into the live notes dir on first
        init.  Idempotent: only writes a seed note whose id is not already present.
        Returns the number of notes newly installed."""
        installed = 0
        if not os.path.isdir(self.seed_dir):
            self._seeded = True
            return 0
        existing_ids = {n["id"] for n in self._all_notes()}
        for name in sorted(os.listdir(self.seed_dir)):
            if not name.endswith(".md"):
                continue
            src = os.path.join(self.seed_dir, name)
            fm, body = self._load(src)
            if fm is None:
                continue
            nid = str(fm.get("id", ""))
            if not nid or nid in existing_ids:
                continue
            dst = os.path.join(self.notes_dir, name)
            with open(src, encoding="utf-8") as fh:
                content = fh.read()
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(content)
            existing_ids.add(nid)
            installed += 1
        self._seeded = True
        return installed

    # ---------------------------------------------------------------- add_note
    @logged("knowledge")
    def add_note(self, ntype: str, title: str, body: str, **frontmatter) -> str:
        """Write a new typed note and return its auto-assigned id.

        ntype in {ks, pf, alg}.  Recognized frontmatter kwargs: tags (list),
        confidence (default 'experimental' for in-loop notes), step, configs.
        Unknown kwargs are passed through to the frontmatter verbatim.
        """
        if ntype not in _NOTE_TYPES:
            raise ValueError(f"ntype must be one of {_NOTE_TYPES}, got {ntype!r}")
        note_id = self._next_id(ntype)
        fm = {
            "id": note_id,
            "type": ntype,
            "title": str(title),
            "tags": list(frontmatter.pop("tags", []) or []),
            "confidence": str(frontmatter.pop("confidence", "experimental")),
            "created": frontmatter.pop("created", None) or time.strftime("%Y-%m-%d"),
        }
        # Optional fields only when provided (keep frontmatter lean).
        for k in ("step", "configs", "source"):
            if k in frontmatter and frontmatter[k] is not None:
                fm[k] = frontmatter.pop(k)
        # Any remaining kwargs pass through.
        for k, v in frontmatter.items():
            if v is not None:
                fm[k] = v

        path = self._path_for(note_id, title)
        yaml_fm = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True,
                                 default_flow_style=False).strip()
        content = f"---\n{yaml_fm}\n---\n{body.rstrip()}\n"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return note_id

    # ------------------------------------------------------------------- query
    @logged("knowledge")
    def query(self, q: str, ntype: Optional[str] = None, k: int = 5) -> list[dict]:
        """Term-scored search over notes.  Scoring: title hit *10 + tag hit *5 +
        body occurrence (capped at 3) per query term.  Optionally filter by ntype.

        Returns a bounded list of dicts: {id, title, type, score, confidence, path,
        snippet}.  Distilled -- never raw note bodies.
        """
        keywords = [t for t in re.split(r"\s+", (q or "").strip().lower()) if t]
        k = max(1, min(int(k), _MAX_K))
        notes = self._all_notes()
        if ntype:
            notes = [n for n in notes if n["type"] == ntype]

        scored = []
        for n in notes:
            title_l = n["title"].lower()
            tags_l = " ".join(str(t) for t in n["tags"]).lower()
            body_l = _strip_code_and_comments(n["body"])
            score = 0
            for kw in keywords:
                s = 0
                if kw in title_l:
                    s += 10
                if kw in tags_l:
                    s += 5
                s += min(body_l.count(kw), 3)
                score += s
            if score > 0 or not keywords:
                scored.append((score, n))

        # Sort: score desc, then confidence rank desc, then id for stability.
        scored.sort(key=lambda sn: (
            -sn[0],
            -_CONFIDENCE_RANK.get(sn[1]["confidence"], 0),
            sn[1]["id"],
        ))

        results = []
        for score, n in scored[:k]:
            results.append({
                "id": n["id"],
                "title": n["title"],
                "type": n["type"],
                "score": score,
                "confidence": n["confidence"],
                "path": n["path"],
                "snippet": self._snippet(n["body"], keywords),
            })
        return results

    @staticmethod
    def _snippet(body: str, keywords: list[str]) -> str:
        """First matching line (or the lead) trimmed to _SNIPPET_CHARS."""
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if keywords:
            for ln in lines:
                low = ln.lower()
                if any(kw in low for kw in keywords):
                    return ln[:_SNIPPET_CHARS]
        return (lines[0][:_SNIPPET_CHARS] if lines else "")

    # --------------------------------------------------------------------- get
    @logged("knowledge")
    def get(self, note_id: str, follow: bool = False) -> str:
        """Return the full text of a note (frontmatter + body).  If follow=True,
        append the bodies of notes referenced by [[links]] (one hop, deduped)."""
        note = self._find_by_id(note_id)
        if note is None:
            # Allow lookup by path too.
            if os.path.isfile(note_id):
                fm, body = self._load(note_id)
                if fm is not None:
                    note = {"id": str(fm.get("id", "")), "path": note_id,
                            "body": body or "", "fm": fm,
                            "title": str(fm.get("title", ""))}
            if note is None:
                return f"ERROR: no note '{note_id}'"
        text = open(note["path"], encoding="utf-8").read()
        if not follow:
            return text

        parts = [text]
        seen = {note["id"]}
        for target in _LINK_RE.findall(note["body"]):
            target = target.strip()
            linked = self._find_by_id(target) or self._find_by_title(target)
            if linked and linked["id"] not in seen:
                seen.add(linked["id"])
                parts.append(
                    f"\n\n--- linked: [[{target}]] -> {linked['id']} "
                    f"({linked['title']}) ---\n{linked['body'].strip()}"
                )
        return "".join(parts)

    def _find_by_title(self, title: str) -> Optional[dict]:
        tl = title.strip().lower()
        for n in self._all_notes():
            if n["title"].lower() == tl:
                return n
        return None

    # -------------------------------------------------------------------- grep
    @logged("knowledge")
    def grep(self, pattern: str) -> list[str]:
        """Regex search across note bodies.  Returns bounded list of
        'id:line_no: <line>' hits (case-insensitive), capped for context discipline.
        """
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [f"ERROR: invalid regex {pattern!r}: {e}"]
        hits: list[str] = []
        for n in self._all_notes():
            for i, line in enumerate(n["body"].splitlines(), start=1):
                if rx.search(line):
                    hits.append(f"{n['id']}:{i}: {line.strip()[:_SNIPPET_CHARS]}")
                    if len(hits) >= _GREP_MAX_FILES:
                        return hits
        return hits


# ----------------------------------------------------------------------------
# Module-level convenience: a process-wide default store + thin functional API,
# so callers can `from fa3agent.knowledge.store import query` without managing an
# instance.  query.py / get_page.py / grep.py re-export these.
# ----------------------------------------------------------------------------
_DEFAULT_STORE: Optional[KnowledgeStore] = None


def default_store() -> KnowledgeStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = KnowledgeStore()
    return _DEFAULT_STORE


def query(q: str, ntype: Optional[str] = None, k: int = 5) -> list[dict]:
    return default_store().query(q, ntype=ntype, k=k)


def get(note_id: str, follow: bool = False) -> str:
    return default_store().get(note_id, follow=follow)


def grep(pattern: str) -> list[str]:
    return default_store().grep(pattern)


def add_note(ntype: str, title: str, body: str, **frontmatter) -> str:
    return default_store().add_note(ntype, title, body, **frontmatter)
