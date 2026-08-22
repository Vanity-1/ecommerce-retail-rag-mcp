"""
Cross-source result fusion with Reciprocal Rank Fusion (RRF).

When a query is answered from several independent vector collections (product
catalog + policies + inventory), naively concatenating each collection's
top-k results mixes orders and lets a weak source dominate. RRF re-ranks the
union by per-source reciprocal ranks:

    score(doc) = sum over sources of 1 / (k_rank_in_source + K)

so a document retrieved in a high position by multiple sources rises to the
top, and a one-off hit in a single source is dampened. K is the standard
damping constant (60).

`fused_retrieve` returns reranked, deduplicated docs — an original engineering
extension on top of the upstream multi-source retrieval.
"""

RRF_K = 60.0


def _rank_scores(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def fuse(ranked_groups):
    """RRF-merge several lists of {text, meta, score}.

    ranked_groups: iterable of lists, each sorted by relevance (list[0] is the
    best result for that source). Items are deduplicated by lowercased text.
    """
    scores: dict[str, float] = {}
    docs_by_key: dict[str, dict] = {}
    for group in ranked_groups:
        for rank, doc in enumerate(group):
            key = doc.get("text", "").strip().lower()
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + _rank_scores(rank)
            docs_by_key.setdefault(key, doc)

    # Higher fusion score first.
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [docs_by_key[k] for k, _ in ordered]


async def fused_retrieve(
    query: str,
    collections=("catalog_chunks", "policies"),
    k: int = 8,
    n_final: int = 8,
) -> dict:
    """Retrieve per-source top-k, fuse by RRF, and return top-n_final docs.

    Profile: each source returns its own ranking; the fused order is the one a
    downstream LLM consumes as context, which is more stable than concatenation.
    """
    from src.rag.vector_store import get_collection
    from src.embeddings.ollama_client import embed_text

    qv = embed_text(query)
    groups = []
    for col_name in collections:
        col = get_collection(col_name)
        res = col.query(query_embeddings=[qv], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        groups.append(
            [
                {"text": docs[i], "meta": metas[i] if i < len(metas) else {}}
                for i in range(len(docs))
            ]
        )

    fused = fuse(groups)[:n_final]
    return {"count": len(fused), "docs": fused}