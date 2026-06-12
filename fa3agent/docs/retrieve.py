"""Component 2.4 — NVIDIA-docs navigation / retrieval.

Three ways to read the local reference docs (reference/{cuda-docs,design-guides,architecture-specs}):
  retrieve(query, k=5, scope=None) -> str   # rank WHERE a topic lives (BM25-lite section snippets)
  get_toc(doc, max_depth=0)        -> str   # a doc's headings + line numbers (its structure)
  get_section(doc, heading)        -> str   # ONE section in FULL, incl. subsections (no clipping)

Typical flow: retrieve() to locate the topic -> get_toc() to read off the exact heading ->
get_section() to pull a dense table (e.g. a wgmma descriptor/swizzle layout) whole, where a
ranked snippet would be truncated.

retrieve() output is bounded (each snippet <= ~100 lines; total <= ~14KB); get_section() returns
a whole section up to a generous safety ceiling.  Stdlib only.
"""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache

from ..runlog import logged

try:
    from fa3agent import config as _config

    _REFERENCE_DIR = _config.REFERENCE_DIR
except Exception:  # pragma: no cover - fallback if config import path changes
    # Derive from this file's location (<repo>/fa3agent/docs/retrieve.py) rather than
    # hard-coding a box-specific path, so the fallback is still portable.
    _REFERENCE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reference",
    )

# ---------------------------------------------------------------------------
# Bounds (the whole point of this component: distilled, bounded output).
# ---------------------------------------------------------------------------
SCOPES = ("cuda-docs", "design-guides", "architecture-specs")
# Off-target architecture specs: the agent targets sm_90a (Hopper H200) ONLY.  Indexing the
# Ampere/Blackwell deep-dives lets cross-arch sections win retrieval slots and tempts the brain
# into instructions that are invalid on sm_90a (e.g. Blackwell tcgen05/tmem) -- exactly what the
# GROUNDING prompt warns against.  They are dropped from the indexed corpus (None / scope-dir
# walks); an explicit get_section('blackwell-b200.md', ...) by filename still works if ever needed.
_EXCLUDE_FROM_INDEX = frozenset({"ampere-a100.md", "blackwell-b200.md"})
# Caps sized so a full dense reference table (the wgmma SMEM descriptor / swizzle sections run
# ~30-82 lines / 1.5-4KB) comes back WHOLE instead of clipped mid-table -- truncation here was
# making the brain re-query the same thing 4-5x.  Genuinely huge sections (e.g. the 254-line
# matrix-layout section) still clip; the brain reads the file directly for those.
_MAX_SNIPPET_LINES = 100
_MAX_TOTAL_BYTES = 14 * 1024  # ~14 KB hard cap on the returned string
_MAX_SNIPPET_BYTES = 4500  # per-snippet body byte cap (before the header)
# get_section returns a WHOLE section (incl. subsections) untruncated; this is only a safety
# ceiling so a giant top-level chapter can't dump hundreds of KB. The dense tables we care about
# are well under it; if a section trips it, the caller drills into a subsection via get_toc.
_MAX_SECTION_BYTES = 40 * 1024

# BM25-lite parameters.
_BM25_K1 = 1.5
_BM25_B = 0.75

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.:]*")
# Markdown link in a heading like "[Introduction](#introduction)" -> "Introduction".
_MDLINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Leading section numbering like "1.", "9.7.16." etc.
_SECNUM_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s*")


def _clean_heading(text: str) -> str:
    """Strip markdown-link wrappers and leading section numbers from a heading."""
    text = _MDLINK_RE.sub(r"\1", text)
    text = _SECNUM_RE.sub("", text).strip()
    return text or "(intro)"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _expand_query_tokens(tokens: list[str]) -> set[str]:
    """Expand query tokens so dotted PTX-ish terms also match their pieces.

    e.g. 'cp.async.bulk.tensor' -> {'cp.async.bulk.tensor','cp','async','bulk','tensor'}
    and 'm64n128k16' -> {'m64n128k16','m64','n128','k16','64','128','16'}.
    """
    out: set[str] = set()
    for tok in tokens:
        out.add(tok)
        if "." in tok:
            for part in tok.split("."):
                if part:
                    out.add(part)
        # split alpha/num runs (m64n128k16 -> m,64,n,128,k,16) keeping useful pieces
        for m in re.finditer(r"[a-z]+\d+|\d+", tok):
            piece = m.group(0)
            if piece and piece != tok:
                out.add(piece)
                num = re.sub(r"[a-z]+", "", piece)
                if num:
                    out.add(num)
    # drop trivially-short noise tokens that aren't numeric
    return {t for t in out if len(t) > 1 or t.isdigit()}


class _Section:
    __slots__ = ("file", "heading", "start_line", "lines", "tf", "length", "head_tokens")

    def __init__(self, file: str, heading: str, start_line: int, lines: list[str]):
        self.file = file
        self.heading = heading
        self.start_line = start_line
        self.lines = lines
        head_toks = _tokenize(heading)
        # Heading tokens (plus their dotted/alnum expansions) are a strong relevance
        # signal; keep them separately so _score can give a heading-hit bonus.
        self.head_tokens = _expand_query_tokens(head_toks)
        # term frequencies over heading (weighted) + body
        toks = head_toks * 3 + _tokenize("\n".join(lines))
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        self.tf = tf
        self.length = max(1, len(toks))


def _split_sections(path: str) -> list[_Section]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read().splitlines()
    except OSError:
        return []

    fname = os.path.basename(path)
    sections: list[_Section] = []
    cur_heading = "(intro)"
    cur_start = 1
    cur_lines: list[str] = []
    # Track seen headings within a file to skip pure-TOC duplicate sections that
    # carry no real body.
    seen_heading_bodies: dict[str, bool] = {}

    def _flush(end_excl: int):
        body = cur_lines
        # Trim leading/trailing blank lines.
        while body and not body[0].strip():
            body = body[1:]
        while body and not body[-1].strip():
            body = body[:-1]
        if not body:
            return
        sections.append(_Section(fname, cur_heading, cur_start, body))

    in_fence = False
    for i, line in enumerate(raw, start=1):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            cur_lines.append(line)
            continue
        m = _HEADING_RE.match(line) if not in_fence else None
        if m:
            _flush(i)
            cur_heading = _clean_heading(m.group(2))
            cur_start = i
            cur_lines = []
        else:
            cur_lines.append(line)
    _flush(len(raw) + 1)
    return sections


def _scope_files(scope) -> list[str]:
    """Resolve `scope` to a list of absolute markdown file paths."""
    files: list[str] = []
    if scope is None:
        dirs = SCOPES
    elif scope in SCOPES:
        dirs = (scope,)
    else:
        # Treat scope as a filename (with or without .md), searched across all dirs.
        target = scope if scope.endswith(".md") else scope + ".md"
        target = os.path.basename(target)
        for d in SCOPES:
            cand = os.path.join(_REFERENCE_DIR, d, target)
            if os.path.isfile(cand):
                files.append(cand)
        return files
    for d in dirs:
        ddir = os.path.join(_REFERENCE_DIR, d)
        if not os.path.isdir(ddir):
            continue
        for fn in sorted(os.listdir(ddir)):
            if fn.endswith(".md") and fn not in _EXCLUDE_FROM_INDEX:
                files.append(os.path.join(ddir, fn))
    return files


@lru_cache(maxsize=64)
def _load_scope(scope_key) -> tuple:
    """Return (sections, avg_len, df) for a scope, cached. scope_key is hashable."""
    scope = None if scope_key == "" else scope_key
    sections: list[_Section] = []
    for path in _scope_files(scope):
        sections.extend(_split_sections(path))
    if not sections:
        return ((), 1.0, {})
    df: dict[str, int] = {}
    total_len = 0
    for s in sections:
        total_len += s.length
        for term in s.tf:
            df[term] = df.get(term, 0) + 1
    avg_len = total_len / len(sections)
    return (tuple(sections), avg_len, df)


def _score(section: _Section, q_terms: set[str], df: dict, n_docs: int, avg_len: float) -> float:
    score = 0.0
    matched = 0
    for term in q_terms:
        f = section.tf.get(term, 0)
        if f == 0:
            continue
        matched += 1
        n_t = df.get(term, 0)
        idf = math.log(1.0 + (n_docs - n_t + 0.5) / (n_t + 0.5))
        denom = f + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * section.length / avg_len)
        score += idf * (f * (_BM25_K1 + 1.0)) / denom
    if matched == 0:
        return 0.0
    # Coverage bonus: reward sections that hit many distinct query terms.
    score *= 1.0 + 0.25 * matched
    # Heading bonus: a query term appearing in the section heading is a strong
    # topicality signal (e.g. a "wgmma" query landing on the PTX wgmma section).
    head_hits = sum(1 for t in q_terms if t in section.head_tokens)
    if head_hits:
        score *= 1.0 + 0.6 * head_hits
    return score


def _render_snippet(section: _Section) -> str:
    header = f"source: {section.file} #{section.heading} (line {section.start_line})"
    body_lines = section.lines[:_MAX_SNIPPET_LINES]
    body = "\n".join(body_lines)
    if len(body.encode("utf-8")) > _MAX_SNIPPET_BYTES:
        # Truncate by bytes safely.
        enc = body.encode("utf-8")[:_MAX_SNIPPET_BYTES]
        body = enc.decode("utf-8", errors="ignore").rstrip()
        body += "\n    ...[truncated]"
    if len(body_lines) < len(section.lines):
        body += "\n    ...[+%d more lines]" % (len(section.lines) - len(body_lines))
    return header + "\n" + body


@logged("retrieve")
def retrieve(query: str, k: int = 5, scope=None) -> str:
    """Return top-k bounded doc snippets relevant to `query`.

    Args:
      query: free-text keyword query.
      k: max number of snippets to return.
      scope: None (all dirs), one of {'cuda-docs','design-guides','architecture-specs'},
             or a filename (e.g. 'parallel-thread-execution.md' / 'hopper-h200').

    Returns a single bounded string (<= ~6KB). Snippets are separated by a blank
    line and a rule. Returns a short note string if nothing matched.
    """
    if not query or not query.strip():
        return "(no query provided)"
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5
    k = max(1, min(k, 20))

    scope_key = "" if scope is None else str(scope)
    sections, avg_len, df = _load_scope(scope_key)
    if not sections:
        return f"(no documents found for scope={scope!r})"

    q_terms = _expand_query_tokens(_tokenize(query))
    if not q_terms:
        return "(query had no usable terms)"

    n_docs = len(sections)
    scored = []
    for s in sections:
        sc = _score(s, q_terms, df, n_docs, avg_len)
        if sc > 0.0:
            scored.append((sc, s))
    if not scored:
        return f"(no relevant sections for query={query!r} in scope={scope or 'all'})"

    # Sort by score desc; tie-break by shorter section (more focused) then file/line.
    scored.sort(key=lambda x: (-x[0], x[1].length, x[1].file, x[1].start_line))

    out_parts: list[str] = []
    total = 0
    sep = "\n\n" + "-" * 8 + "\n"
    for _, s in scored[:k]:
        snippet = _render_snippet(s)
        addition = (sep if out_parts else "") + snippet
        if total + len(addition.encode("utf-8")) > _MAX_TOTAL_BYTES and out_parts:
            break
        out_parts.append(snippet)
        total += len(addition.encode("utf-8"))
    return (sep.join(out_parts)).strip()


# ---------------------------------------------------------------------------
# Structured navigation (companions to retrieve): outline a doc, then pull one
# whole section by heading.  retrieve() ranks WHERE a topic lives; get_toc shows
# a doc's exact headings; get_section reads a chosen section in full (incl. its
# subsections) so a dense table is never clipped the way a ranked snippet can be.
# ---------------------------------------------------------------------------
def _read_doc_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def _resolve_doc(name) -> "str | None":
    """Map a doc filename (e.g. 'parallel-thread-execution.md', '.md' optional) to its path."""
    if not name:
        return None
    files = _scope_files(os.path.basename(str(name)))
    return files[0] if files else None


def _no_doc_msg(name) -> str:
    avail = sorted({os.path.basename(p) for p in _scope_files(None)})
    if not avail:
        return f"(no reference docs found under {_REFERENCE_DIR})"
    return f"(no reference doc {name!r}; available: {', '.join(avail)})"


def _norm_heading(text: str) -> str:
    """Normalize a heading for matching: drop section numbers/link sugar, lowercase, collapse ws."""
    return " ".join(_clean_heading(text).lower().split())


def _scan_headings(raw: list[str]) -> list[tuple[int, int, str]]:
    """Fence-aware heading scan -> [(0-based line index, level, cleaned title)] in document order."""
    out: list[tuple[int, int, str]] = []
    in_fence = False
    for i, line in enumerate(raw):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m:
            out.append((i, len(m.group(1)), _clean_heading(m.group(2))))
    return out


@logged("get_toc")
def get_toc(doc: str, max_depth: int = 0) -> str:
    """Outline a reference doc: every heading with its nesting level and line number.

    Call this to find the exact heading text to pass to get_section() -- cheaper and more
    precise than streaming a whole large file.  `doc` is a filename (e.g.
    'parallel-thread-execution.md').  `max_depth` (0 = all) caps how deep the outline goes
    (e.g. 2 = top two heading levels).
    """
    path = _resolve_doc(doc)
    if not path:
        return _no_doc_msg(doc)
    try:
        raw = _read_doc_lines(path)
    except OSError as e:
        return f"(could not read {os.path.basename(path)}: {e})"
    try:
        max_depth = int(max_depth)
    except (TypeError, ValueError):
        max_depth = 0

    fname = os.path.basename(path)
    rows = [
        f"  L{idx + 1:<6}{'  ' * (level - 1)}{title}"
        for idx, level, title in _scan_headings(raw)
        if not max_depth or level <= max_depth
    ]
    if not rows:
        return f"outline: {fname}\n  (no headings)"
    return (f"outline: {fname} ({len(rows)} headings)\n"
            + "\n".join(rows)
            + f"\n\nRead one in full with: get_section({fname!r}, '<exact heading above>')")


@logged("get_section")
def get_section(doc: str, heading: str) -> str:
    """Return ONE whole section of a reference doc: the chosen heading plus everything under it
    (including nested subsections) up to the next heading at the same or a shallower level.

    Use this to read a dense table (e.g. a wgmma shared-memory descriptor / swizzle layout) in
    FULL, where a ranked retrieve() snippet would be truncated.  `doc` is a filename; `heading`
    is the heading text as shown by get_toc()/retrieve() (a leading section number is optional).
    """
    path = _resolve_doc(doc)
    if not path:
        return _no_doc_msg(doc)
    if not heading or not str(heading).strip():
        return f"(no heading given; call get_toc({os.path.basename(path)!r}) to list headings)"
    try:
        raw = _read_doc_lines(path)
    except OSError as e:
        return f"(could not read {os.path.basename(path)}: {e})"

    fname = os.path.basename(path)
    headings = _scan_headings(raw)
    want = _norm_heading(heading)

    target = next((j for j, (_, _, t) in enumerate(headings) if _norm_heading(t) == want), None)
    if target is None:
        # No exact match: accept a unique partial match, else point back to get_toc.
        partial = [j for j, (_, _, t) in enumerate(headings) if want and want in _norm_heading(t)]
        if len(partial) == 1:
            target = partial[0]
        else:
            near = ", ".join(headings[j][2] for j in partial[:8])
            hint = f" closest: {near};" if near else ""
            return (f"(no section {heading!r} in {fname};{hint} "
                    f"call get_toc({fname!r}) for exact headings)")

    start_idx, level, title = headings[target]
    end_idx = len(raw)
    for nidx, nlevel, _t in headings[target + 1:]:
        if nlevel <= level:               # next same-or-shallower heading ends this section
            end_idx = nidx
            break

    text = "\n".join(raw[start_idx:end_idx]).rstrip()
    header = f"source: {fname} #{title} (line {start_idx + 1})"
    enc = text.encode("utf-8")
    if len(enc) > _MAX_SECTION_BYTES:
        text = enc[:_MAX_SECTION_BYTES].decode("utf-8", errors="ignore").rstrip()
        text += (f"\n... [section exceeds {_MAX_SECTION_BYTES // 1024}KB and was truncated; "
                 f"call get_toc({fname!r}) and fetch a narrower subsection]")
    return header + "\n" + text
