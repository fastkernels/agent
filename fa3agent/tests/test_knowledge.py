"""Smoke test for fa3agent.knowledge (KnowledgeStore + query/get/grep + seed).

Run from /data-fast/agent:
    python -m fa3agent.tests.test_knowledge
or
    python fa3agent/tests/test_knowledge.py

Uses a temp notes dir so it does NOT pollute fa3agent/knowledge/notes, but seeds
from the authored fa3agent/knowledge/seed/*.md.
"""
from __future__ import annotations

import os
import sys
import tempfile

# Ensure 'import fa3agent...' works no matter the CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fa3agent.knowledge.store import KnowledgeStore
from fa3agent.knowledge.query import query as mod_query
from fa3agent.knowledge.get_page import get as mod_get
from fa3agent.knowledge.grep import grep as mod_grep


def main() -> int:
    fails = []

    tmp = tempfile.mkdtemp(prefix="fa3kb_")
    notes_dir = os.path.join(tmp, "knowledge")

    # 1. construct + seed
    ks = KnowledgeStore(notes_dir=notes_dir)  # auto_seed on empty dir
    n_seeded = ks.seed()  # idempotent; should be 0 now (already auto-seeded)
    all_notes = ks._all_notes()
    print(f"[1] seeded notes on disk: {len(all_notes)} (second seed() installed {n_seeded})")
    if len(all_notes) < 10:
        fails.append(f"expected >=10 seed notes, got {len(all_notes)}")
    types = {n["type"] for n in all_notes}
    if not {"ks", "pf", "alg"}.issubset(types):
        fails.append(f"missing note types; got {types}")

    # 2. query for warp specialization / hopper / wgmma -> >=1 relevant note
    res = ks.query("warp specialization hopper wgmma")
    print(f"[2] query('warp specialization hopper wgmma') -> {len(res)} hits")
    for r in res[:5]:
        print(f"      {r['id']:9} score={r['score']:3} {r['title'][:60]}")
    if not res:
        fails.append("query returned no hits for warp/hopper/wgmma")
    else:
        top = res[0]
        blob = (top["title"] + " " + str(top.get("snippet", ""))).lower()
        if "wgmma" not in blob and "warp" not in blob:
            print(f"      (note) top hit not obviously wgmma/warp: {top['id']}")

    # 3. get(top id) -> full text
    if res:
        full = ks.get(res[0]["id"])
        print(f"[3] get('{res[0]['id']}') -> {len(full)} chars")
        if len(full) < 100 or not full.startswith("---"):
            fails.append("get() did not return full note text with frontmatter")
        full_f = ks.get(res[0]["id"], follow=True)
        if "linked:" in full_f:
            print(f"[3b] get(follow=True) expanded links -> {len(full_f)} chars")
        elif len(full_f) < len(full):
            fails.append("follow=True returned less than the base note")

    # 4. add_note('pf', ...) then query finds it; id auto-increments per type
    new_id = ks.add_note("pf", "test note", "body about a unique zylophone bottleneck",
                         tags=["x", "zylophone"])
    print(f"[4] add_note -> {new_id}")
    if not new_id.startswith("pf-"):
        fails.append(f"add_note id not pf-prefixed: {new_id}")
    res2 = ks.query("test note")
    found = any(r["id"] == new_id for r in res2)
    print(f"[4b] query('test note') found new note: {found}")
    if not found:
        fails.append("query did not find the freshly added 'test note'")
    res3 = ks.query("zylophone")
    if not any(r["id"] == new_id for r in res3):
        fails.append("query('zylophone') did not find the new note")

    # 5. ntype filter
    pf_only = ks.query("bottleneck", ntype="pf")
    if any(r["type"] != "pf" for r in pf_only):
        fails.append("ntype='pf' filter leaked non-pf notes")
    print(f"[5] query('bottleneck', ntype='pf') -> {len(pf_only)} pf hits")

    # 6. grep
    g = ks.grep("setmaxnreg")
    print(f"[6] grep('setmaxnreg') -> {len(g)} line hits")
    if not g:
        fails.append("grep('setmaxnreg') returned nothing")

    # 7. module-level functional API uses the default (package) store, seeded.
    md = mod_query("online softmax exp2")
    print(f"[7] module query('online softmax exp2') -> {len(md)} hits via default store")
    if not md:
        fails.append("module-level query returned nothing")
    else:
        _ = mod_get(md[0]["id"])
        _ = mod_grep("wgmma")

    # 8. add_note rejects bad type
    try:
        ks.add_note("bogus", "t", "b")
        fails.append("add_note accepted an invalid ntype")
    except ValueError:
        print("[8] add_note('bogus', ...) correctly rejected")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: all knowledge-store smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
