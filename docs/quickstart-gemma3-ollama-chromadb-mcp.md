# Quickstart: Gemma3:270m + Ollama + ChromaDB + MCP

This guide spins up a minimal local stack to test the RAG flow end‑to‑end.

## Prerequisites
- Python 3.10+
- Docker / Docker Compose
- Ollama installed and running (`ollama serve`)
- ChromaDB (Docker) or pip (`pip install chromadb`)

## 1) Start Services

### Option A: Docker Compose (recommended)
Create `docker-compose.yml` in project root:

```yaml
version: '3.9'
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama

  chroma:
    image: ghcr.io/chroma-core/chroma:latest
    ports:
      - "8001:8001"
    environment:
      CHROMA_SERVER_HOST: 0.0.0.0
      CHROMA_SERVER_HTTP_PORT: 8001

  api:
    build: .
    depends_on:
      - ollama
      - chroma
    environment:
      OLLAMA_URL: http://ollama:11434
      CHROMA_URL: http://chroma:8001
    ports:
      - "8000:8000"
volumes:
  ollama:
```

Then:
```bash
docker compose up -d
ollama pull gemma3:270m
```

### Option B: Local processes
```bash
ollama serve &
ollama pull gemma3:270m
chroma start --host 0.0.0.0 --port 8001 &
```

## 2) Install Python Deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Example `requirements.txt`:
```
chromadb
fastapi
uvicorn[standard]
requests
pydantic
python-dotenv
```

## 3) Minimal Ingestion + Query Script

Create `scripts/minimal_demo.py`:

```python
import os, requests, chromadb

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST","localhost"), port=os.getenv("CHROMA_PORT","8001"))
col = client.get_or_create_collection("catalog_chunks")

# NOTE: Replace with a real embed function if Ollama exposes embeddings for gemma3:270m.
# Otherwise use a sentence-embedding model for vectors and gemma3 for generation.

def embed_text(text: str):
    raise NotImplementedError("Provide embedding implementation")


def upsert_demo_docs():
    docs = [
        ("sku123#title", "Wireless Bluetooth Headphones with ANC", {"sku":"sku123","field":"title","brand":"Acme","price":99.0}),
        ("sku123#desc", "Over-ear headphones with 30h battery, Type-C fast charge.", {"sku":"sku123","field":"desc","brand":"Acme","price":99.0}),
        ("policy#returns", "Returns allowed within 30 days with receipt.", {"type":"policy","topic":"returns"})
    ]
    for _id, text, meta in docs:
        emb = embed_text(text)
        col.upsert(ids=[_id], embeddings=[emb], documents=[text], metadatas=[meta])


def generate(query: str, context: str) -> str:
    prompt = f"""
SYSTEM: You are a retail assistant. Answer ONLY using the provided CONTEXT. If information is missing, say you don’t have enough information and ask a concise follow-up.
USER: {query}
CONTEXT:\n{context}
STYLE: Be concise, include SKU/IDs only when present in CONTEXT, no hallucinations.
"""
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": "gemma3:270m",
        "prompt": prompt,
        "stream": False
    })
    r.raise_for_status()
    return r.json().get("response", "")


def query_pipeline(q: str):
    qv = embed_text(q)
    res = col.query(query_embeddings=[qv], n_results=5)
    docs = res.get("documents", [[]])[0]
    context = "\n\n".join(docs)
    print(generate(q, context))


if __name__ == "__main__":
    upsert_demo_docs()
    query_pipeline("Headphones under $100 with ANC?")
```

Run:
```bash
python scripts/minimal_demo.py
```

## 4) MCP Server (skeleton)

Create `src/mcp_server.py`:

```python
from fastapi import FastAPI, HTTPException
import chromadb, os, requests

app = FastAPI()
client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST","localhost"), port=os.getenv("CHROMA_PORT","8001"))
col = client.get_or_create_collection("catalog_chunks")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# TODO: Implement auth, consent, schemas per MCP spec

@app.post("/mcp/tools/call")
def tools_call(payload: dict):
    name = payload.get("params", {}).get("name")
    args = payload.get("params", {}).get("arguments", {})

    if name == "product_search":
        q = args.get("query", "")
        # embed q → query chroma → return items (placeholder)
        raise HTTPException(501, "product_search not implemented")

    raise HTTPException(404, f"unknown tool: {name}")
```

Run:
```bash
uvicorn src.mcp_server:app --reload --port 8000
```

## 5) Next Steps
- Implement actual `embed_text` using either Ollama’s embedding (if available) or a sentence-embedding model
- Add metadata filters (brand/price) in Chroma queries
- Implement MCP tools (product_search, inventory_check, policy_qa, analytics_query)
- Add auth, RBAC, rate limiting, audit logs
```
