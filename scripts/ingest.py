#!/usr/bin/env python3
"""
Ingest catalog/policy/review chunks into ChromaDB so product_search, policy_qa,
and the RAG agent can retrieve them.

Usage (from repo root):
    python scripts/ingest.py --file path/to/catalog.jsonl
    python scripts/ingest.py --dir ./data

Input format (JSONL, one object per line):
    {"id": "sku-001", "text": "..." [, "metadata": {...}]}

A plain text file (.txt) is also accepted: each non-empty line becomes a chunk.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.ollama_client import embed_text  # noqa: E402
from src.rag import vector_store  # noqa: E402


def _jsonl_entries(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_entries(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        entries = list(_jsonl_entries(path))
        return [
            {
                "id": str(e.get("id") or str(i)),
                "text": e.get("text", "") if "text" in e else json.dumps(e, ensure_ascii=False),
                "metadata": e.get("metadata", {}),
            }
            for i, e in enumerate(entries)
            if e.get("text") or "text" not in e
        ]
    if ext == ".txt":
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return [
            {"id": f"{os.path.basename(path)}:{i}", "text": ln, "metadata": {}}
            for i, ln in enumerate(lines)
        ]
    raise ValueError(f"Unsupported file type: {ext} (use .jsonl or .txt)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    ap.add_argument("--file", help="Path to a .jsonl or .txt file")
    ap.add_argument("--dir", help="Directory of .jsonl/.txt files to ingest")
    ap.add_argument("--collection", default="catalog_chunks",
                    help="Chroma collection name (default: catalog_chunks)")
    ap.add_argument("--batch", type=int, default=64,
                    help="Batch size for embedding (default: 64)")
    args = ap.parse_args()

    if not (args.file or args.dir):
        ap.error("Provide either --file or --dir")

    paths: list[str] = []
    if args.file:
        paths.append(args.file)
    else:
        for root, _, files in os.walk(args.dir):
            for fn in files:
                if fn.endswith((".jsonl", ".txt")):
                    paths.append(os.path.join(root, fn))
    if not paths:
        print("No .jsonl/.txt files to ingest.")
        return

    col = vector_store.get_collection(args.collection)
    count = 0
    for path in paths:
        entries = load_entries(path)
        for i in range(0, len(entries), args.batch):
            batch = entries[i : i + args.batch]
            texts = [e["text"] for e in batch]
            embeddings = embed_text_batch(texts)
            col.upsert(
                ids=[e["id"] for e in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[e["metadata"] for e in batch],
            )
            count += len(batch)
            print(f"  {path}: upserted {len(batch)} (cumulative {count})")

    print(f"Done. Ingested {count} chunks into collection '{args.collection}'.")


def embed_text_batch(texts: list[str]) -> list[list[float]]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(embed_text, texts))


if __name__ == "__main__":
    main()