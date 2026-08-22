from typing import List, Dict, Any
import os
import chromadb

CHROMA_HOST = os.getenv("CHROMA_HOST","localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT","8001"))
# When set, talk to a local persistent store directly instead of a Chroma HTTP
# server. This avoids depending on a resident service and sidesteps an HTTP
# schema-version mismatch in Chroma 0.5.x client/server round trips.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")

_client = None

def get_client():
    global _client
    if _client is None:
        if CHROMA_PERSIST_DIR:
            _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        else:
            _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return _client

def get_collection(name: str="catalog_chunks"):
    client = get_client()
    try:
        return client.get_collection(name)
    except Exception:
        # Persisted collections may not exist on a fresh server; fall back to a
        # (possibly empty) new collection and let the caller upsert into it.
        return client.get_or_create_collection(name)

def upsert(id: str, embedding: List[float], text: str, metadata: Dict[str,Any]):
    col = get_collection()
    col.upsert(ids=[id], embeddings=[embedding], documents=[text], metadatas=[metadata])

def query(query_embedding: List[float], k: int=10):
    col = get_collection()
    return col.query(query_embeddings=[query_embedding], n_results=k)
