#!/usr/bin/env python3
"""
RRF vs plain concatenation — reproducible comparison for the resume.

Reads ONLY the precomputed embedding cache (no resident Chroma server, no
network) so it is fully deterministic and fast.

For a fixed query set we produce a ranked list under both fusion modes:
  - PLAIN: concatenate per-source top-k, dedup order preserved (keep-first)
  - RRF  : Reciprocal Rank Fusion reranks by sum of 1/(rank+K)

Metrics over the query set:
  n_docs      avg deterministic results returned (dedup → RRF generally ≤ plain)
  rel_hits    avg rows matching expected terms (higher = better recall)
  top3_rel    rows whose gold hit lands in the top-3
  order_stab  how often the two modes agree on the #1 doc (diversity signal)

Output is printed as a table you can copy into the resume notes.
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")

from src.rag import inmemory_cache
from agents.tools.rrf_fusion import fuse, RRF_K
from src.embeddings.ollama_client import embed_text


# (query, terms a good hit should contain)
QUERIES = [
    ("organic avocado", ["avocado"]),
    ("low fat greek yogurt", ["yogurt"]),
    ("chocolate milk", ["chocolate", "milk"]),
    ("frozen pepperoni pizza", ["pizza"]),
    ("gala apples", ["apple"]),
    ("sparkling water", ["water"]),
    ("whole wheat bread", ["bread"]),
    ("cage free eggs", ["egg"]),
    ("paper towels", ["towel"]),
    ("free delivery over 35 dollars", ["delivery"]),      # policy-flavoured
    ("when can I return items", ["return"]),               # policy
    ("do you refund damaged produce", ["return", "perishable"]),  # cross-source
]

COLLECTIONS = ["catalog_chunks", "policies"]
K_PER_SOURCE = 5


def collect(query):
    """Return list of per-source ranked lists of docs."""
    qv = embed_text(query)
    groups = []
    for name in COLLECTIONS:
        r = inmemory_cache.cosine_search(name, qv, k=K_PER_SOURCE)
        docs = [
            {"collection": name, "text": r["documents"][i],
             "meta": r["metadatas"][i], "id": r["ids"][i]}
            for i in range(len(r["ids"]))
        ]
        groups.append(docs)
    return groups


def norm(d):
    return d["text"].strip().lower()


def rel(d, terms):
    return any(t in norm(d) for t in terms)


def plain_rank(groups):
    seen, out = set(), []
    for g in groups:
        for d in g:
            k = norm(d)
            if k not in seen:
                seen.add(k)
                out.append(d)
    return out


def rrf_rank(groups):
    return fuse(groups)


def main():
    print(f"collections: {COLLECTIONS}  | K per source: {K_PER_SOURCE} | RRf K={RRF_K}\n")
    stats = {"plain": {"docs": 0, "rel": 0, "top3": 0},
             "rrf": {"docs": 0, "rel": 0, "top3": 0}}
    agree = 0
    rows = []
    for q, terms in QUERIES:
        groups = collect(q)
        p = plain_rank(groups)
        r = rrf_rank(groups)

        def scores(lst):
            rels = [rel(d, terms) for d in lst[:3]]
            return (len(lst), sum(1 for d in lst if rel(d, terms)), any(rels))

        for mode, lst in (("plain", p), ("rrf", r)):
            n, rl, t3 = scores(lst)
            stats[mode]["docs"] += n
            stats[mode]["rel"] += rl
            stats[mode]["top3"] += t3

        if p and r and norm(p[0]) == norm(r[0]):
            agree += 1
        rows.append((q, len(p), len(r), p[0]["text"][:22] if p else "-",
                     r[0]["text"][:22] if r else "-"))

    n = len(QUERIES)
    print(f"{'query':<28}{'plain_n':>8}{'rrf_n':>8}  {'plain#1':<22}{'rrf#1':<26}")
    print("-" * 96)
    for q, pn, rn, p1, r1 in rows:
        print(f"{q:<28}{pn:>8}{rn:>8}  {p1:<22}{r1:<26}")

    print("\n" + "-" * 96)
    print(f"AVERAGE over {n} queries")
    for mode in ("plain", "rrf"):
        s = stats[mode]
        print(f"  {mode.upper():>5} | docs={s['docs']/n:.1f} | rel_hits={s['rel']/n:.1f} "
              f"| query_top3_rel={s['top3']}/{n}")
    print(f"\n#1 doc agreement (plain vs rrf): {agree}/{n}")
    print(f"RRF returned {stats['rrf']['docs']/n:.1f} docs vs plain "
          f"{stats['plain']['docs']/n:.1f} → (fewer usually = less duplicate noise)")


if __name__ == "__main__":
    main()