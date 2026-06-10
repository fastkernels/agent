"""get_page.py -- fetch one full note by id (with optional [[link]] expansion).

Adapts KernelWiki scripts/get_page.py (id->path resolution, frontmatter split,
follow-references) to the fa3agent KnowledgeStore.  Re-exports KnowledgeStore.get
so callers may `from fa3agent.knowledge.get_page import get`.

get(note_id, follow=False) returns the full note text; follow=True appends the
bodies of one-hop [[links]] (deduped, bounded by the <=250-word note convention).
"""
from __future__ import annotations

from fa3agent.knowledge.store import KnowledgeStore, default_store, get

__all__ = ["get", "KnowledgeStore", "default_store"]


def get_from(store: KnowledgeStore, note_id: str, follow: bool = False) -> str:
    """Explicit-store variant."""
    return store.get(note_id, follow=follow)
