"""
In-memory cosine retrieval layer over a precomputed embedding cache.

Why: on this stack, chromadb 0.5.x PersistentClient drops the on-disk HNSW
vector segment (sqlite/metadata survive, query raises "Cannot open header
file"). For a ~10k-doc catalog an in-memory flat cosine scan is deterministic,
offline, sub-10ms after load, and needs no resident server — a pragmatic
engineering trade-off at this data scale.

Data files (written by scripts/_precompute_embeddings.py):
    scripts/embeddings_cache.npy     np.ndarray [N, D] float32
    scripts/embedding_index.json     list of {id, collection, text, meta}

This exposes chroma-like query semantics (query_embeddings -> {ids, documents,
metadatas}) so callers can swap retrieval backends without changing their code.
"""
import json
import os
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRE = os.environ.get("EMBED_CACHE_DIR", os.path.join(ROOT, "scripts"))
EMB_CACHE = os.path.join(PRE, "embeddings_cache.npy")
INDEX_CACHE = os.path.join(PRE, "embedding_index.json")

_lock = threading.Lock()
_STORE = None  # {collection: {"ids":[], "docs":[], "metas":[], "vecs": ndarray}}


def _build(cols, vecs, rows):
    store = {}
    for col in cols:
        store[col] = {"ids": [], "docs": [], "metas": [], "vecs": None}
    order = {}
    for c in cols:
        order[c] = []
    for i, r in enumerate(rows):
        c = order_col = r.get("collection")
        if c not in order:
            order.setdefault(r.get("collection"), []).append(i)
        else:
            order[c].append(i)
    # reorder per collection
    for c in cols:
        idx = order.get(c, [])
        if not idx:
            continue
        store[c]["ids"] = [rows[i]["id"] for i in idx]
        store[c]["docs"] = [rows[i]["text"] for i in idx]
        store[c]["metas"] = [rows[i].get("meta", {}) for i in idx]
        store[c]["vecs"] = vecs[idx]
    return store


def load():
    global _STORE
    if _STORE is not None:
        return _STORE
    with _lock:
        if _STORE is not None:
            return _STORE
        vecs = np.load(EMB_CACHE, allow_pickle=False)
        with open(INDEX_CACHE, encoding="utf-8") as f:
            rows = json.load(f)
        cols = sorted({r.get("collection", "catalog_chunks") for r in rows})
        _STORE = _build(cols, vecs, rows)
    return _STORE


def collections():
    return list(load().keys())


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine_search(collection, query_embedding, k=4):
    """Return {ids, documents, metadatas} with top-k hits for a collection."""
    store = load()
    if collection not in store:
        raise KeyError(f"collection '{collection}' not in cache: {list(store)}")
    col = store[collection]
    q = _norm(query_embedding)
    V = col["vecs"]
    if q.size != V.shape[1]:
        raise ValueError(
            f"dim mismatch: query {q.size} vs store {V.shape[1]} "
            f"(make sure query uses the same embedding model)"
        )
    sims = V @ q
    top = np.argsort(-sims)[:k]
    return {
        "ids": [col["ids"][i] for i in top],
        "documents": [col["docs"][i] for i in top],
        "metadatas": [col["metas"][i] for i in top],
    }