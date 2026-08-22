"""One-time precompute of all doc embeddings.

Reason: chroma 0.5.x PersistentClient on this stack silently drops the on-disk
HNSW vector segment (sqlite + metadata survive, but querying raises
"RuntimeError: Cannot open header file"). Rather than fight that, we precompute
embeddings once into a numpy cache and serve retrieval from an in-memory cosine
scan — deterministic, offline, and fast enough at this scale (~10k docs).

Outputs (under scripts/):
  embeddings_cache.npy   [N, D] float32  query & doc embeddings for cosine
  embedding_index.json   ids, collection, text, metadata per row

This is only a build step; runtime readers (demo / evaluator) load the cache.
"""
import sys
import os
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")

from src.embeddings.ollama_client import embed_text  # noqa: E402

OUT_DIR = os.path.join(ROOT, "scripts")
EMB_CACHE = os.path.join(OUT_DIR, "embeddings_cache.npy")
INDEX_CACHE = os.path.join(OUT_DIR, "embedding_index.json")


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect(source_path):
    rows = []
    for e in _jsonl(source_path):
        doc = e.get("text") or json.dumps(e, ensure_ascii=False)
        rows.append({
            "id": str(e.get("id") or len(rows)),
            "collection": os.path.basename(os.path.dirname(source_path)) if False else source_path.split("\\")[-2] if "\\" in source_path else "policies",
            "text": doc,
            "meta": e.get("metadata", {}),
        })
    return rows


def build(source_path, collection):
    rows = collect(source_path)
    for r in rows:
        r["collection"] = collection
    docs = [r["text"] for r in rows]
    with ThreadPoolExecutor(max_workers=8) as ex:
        vecs = list(ex.map(embed_text, docs))
    arr = np.asarray(vecs, dtype=np.float32)
    assert arr.shape == (len(rows), arr.shape[1]), arr.shape
    return rows, arr


def main():
    t0 = time.time()
    cat_rows, cat_arr = build(os.path.join(ROOT, "scripts", "catalog.jsonl"), "catalog_chunks")
    pol_rows, pol_arr = build(os.path.join(ROOT, "data", "policies.jsonl"), "policies")
    print(f"catalog {cat_arr.shape}, policies {pol_arr.shape}")

    rows = cat_rows + pol_rows
    full = np.concatenate([cat_arr, pol_arr], axis=0).astype(np.float32)
    np.save(EMB_CACHE, full)
    with open(INDEX_CACHE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"Saved {EMB_CACHE} ({full.shape}) and {INDEX_CACHE} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()