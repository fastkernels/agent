"""grep.py -- regex text search across note bodies.

Adapts KernelWiki scripts/grep_wiki.py line-matching to the fa3agent KnowledgeStore.
Re-exports KnowledgeStore.grep so callers may
`from fa3agent.knowledge.grep import grep`.

grep(pattern) returns a bounded list of 'id:line_no: <line>' hits
(case-insensitive), capped for context discipline.
"""
from __future__ import annotations

from fa3agent.knowledge.store import KnowledgeStore, default_store, grep

__all__ = ["grep", "KnowledgeStore", "default_store"]


def grep_store(store: KnowledgeStore, pattern: str) -> list[str]:
    """Explicit-store variant."""
    return store.grep(pattern)
