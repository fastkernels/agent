"""query.py -- ranked keyword search over the typed knowledge store.

Adapts KernelWiki scripts/query.py term-scoring (title*10 + tags*5 + body, capped)
to the fa3agent KnowledgeStore.  Re-exports KnowledgeStore.query so callers may
`from fa3agent.knowledge.query import query`.

Returns a bounded list of {id, title, type, score, confidence, path, snippet} --
distilled, never raw note bodies (context discipline).
"""
from __future__ import annotations

from typing import Optional

from fa3agent.knowledge.store import KnowledgeStore, default_store, query

__all__ = ["query", "KnowledgeStore", "default_store"]


def query_store(store: KnowledgeStore, q: str, ntype: Optional[str] = None,
                k: int = 5) -> list[dict]:
    """Explicit-store variant for callers that hold their own (run-scoped) store."""
    return store.query(q, ntype=ntype, k=k)
