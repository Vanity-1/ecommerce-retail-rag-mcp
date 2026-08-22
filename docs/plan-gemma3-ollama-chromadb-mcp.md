# E-commerce & Retail RAG Pipeline (Gemma3:270m + Ollama + ChromaDB + MCP)

This repository provides a production-ready plan and documentation to implement a Retrieval-Augmented Generation (RAG) system for E-commerce/Retail using:
- Gemma3:270m served via Ollama (fast, 32K context SLM)
- ChromaDB for vector search
- Model Context Protocol (MCP) for secure, standardized tool/data access

The plan emphasizes low-latency, cost-efficient SLMs with strong grounding, modular MCP tools, and measurable retrieval quality.

---

## 1. Objectives and Non-Goals

### Objectives
- Build a robust, MCP-orchestrated RAG platform for retail search, support, and insights
- Use Gemma3:270m via Ollama for generation and task prompts
- Use ChromaDB as the primary vector store with hybrid filtering
- Enforce security and data governance through MCP
- Provide stepwise implementation roadmap with KPIs and validation

### Non-Goals
- Training large foundation models from scratch
- Replacing transactional systems; this augments them via MCP tools

---

## 2. End-to-End Architecture

- Data Sources: Catalogs, Policies/FAQs, Reviews, Tickets, Inventory/Orders
- Ingestion: Connectors → normalize → multilingual normalization → chunking (atomic units)
- Embeddings: Text embedding via Gemma3:270m (through Ollama embed task or prompt), store 768d vectors in ChromaDB
- Retrieval: ChromaDB KNN with metadata filters (brand, price, availability)
- Augmentation: build a context package from top-k chunks, apply rerank if needed
- Generation: Gemma3:270m answers grounded on context; refuse/ask-clarify on insufficient context
- MCP Layer: servers for catalog_search, inventory_check, policy_qa, analytics_query; host-side consent and isolation
- Apps: Web/mobile/chat APIs; recommendation widgets; agentic ops for store teams

---

## 3. Use Cases

1) Product Discovery & Semantic Search
- Intent → query rewrite (optional) → hybrid retrieval → rerank → grounded answer with links/SKUs

2) Customer Support QA
- Order status, returns, policy Q&A; enforce policy-bound answers via MCP context

3) Recommendations & Merchandising
- Similar/complementary items using vector similarity + metadata; optional graph recall later

4) Content & Review Ops
- Attribute extraction, review summarization, title/description polishing with grounded prompts

5) Analytics Insights (read-only)
- sales/inventory trends via analytics_query MCP tool returning aggregates for natural-language summaries

---

## 4. Detailed Implementation Plan (12 Weeks)

Phase 1: Foundation (Weeks 1–2)
- Repo scaffold, CI/CD, Docker, .env templates
- Install Ollama; pull gemma3:270m
- Stand up ChromaDB; define collections (catalog, policies, reviews, tickets)
- Define schemas and chunking policy (atomic field-based chunks)

Phase 2: Naive RAG Baseline (Weeks 3–4)
- Ingest sample datasets; generate embeddings; store in ChromaDB
- Implement retrieval → context build → Gemma3:270m generate
- Add minimal evaluation set and metrics (recall@k, groundedness rubric)

Phase 3: SLM Task Tuning (Weeks 5–6)
- If LoRA/QLoRA supported: train adapters for intent classification, attribute extraction, sentiment, and response style
- If not: optimize prompting, few-shot exemplars, and retrieval precision
- Add reranker (cross-encoder or prompt-based re-ranking) and measure uplift

Phase 4: MCP Integration (Weeks 7–8)
- Implement MCP tools:
  - catalog_search: vector + filters
  - inventory_check: read-only DB/API
  - policy_qa: policy collection queries
  - analytics_query: safe aggregates
- Add auth (RBAC/API keys), consent prompts, schema validation

Phase 5: Retail Features (Weeks 9–10)
- Recommendations: similar & complementary items
- Multilingual normalizer and query rewrite stage
- Customer support flows (ticket triage, escalation detection)

Phase 6: Hardening & Deploy (Weeks 11–12)
- Perf tests & caching; latency budget: retrieval < 200ms, E2E < 600ms
- Observability (Prometheus/Grafana/logging/tracing)
- Security audit (tool allowlist, rate limits, PII minimization)
- Blue/green deploy and rollback

---

## 5. Data & Chunking Strategy

- Catalog: Keep fields distinct (title, bullets, attributes, brand), but create an aggregated representation for first-pass recall; maintain per-field chunks for rerank
- Policies/FAQs: atomic sections; optional synthetic Q pairs to improve recall
- Reviews/Tickets: summarize per-item or per-thread; store both raw and summary chunks
- Multilingual: normalize inputs; optional transliteration/translation on query; preserve language signals in metadata

---

## 6. ChromaDB Schema (example)

- Collection: catalog_chunks
  - id: string (sku#field#chunk)
  - embedding: vector(768)
  - metadata: { sku, field, lang, brand, category, price, stock, text }
- Collections: policies, reviews, tickets with analogous metadata

---

## 7. Ollama + Gemma3:270m Integration

- Run model: `ollama run gemma3:270m`
- Embedding: If Ollama exposes embedding for Gemma3:270m, use it; otherwise use a prompt-driven embedding surrogate or a compact sentence-embedder alongside Gemma3 for generation
- Generation: Ground answers strictly on retrieved context; prompt template enforces refusals when insufficient context

Prompt Template (generation)
```
SYSTEM: You are a retail assistant. Answer ONLY using the provided CONTEXT. If information is missing, say you don’t have enough information and ask a concise follow-up.
USER: {query}
CONTEXT:
{top_k_chunks}
STYLE: Be concise, include SKU/IDs only when present in CONTEXT, no hallucinations.
```

---

## 8. MCP Server Design

- Transport: HTTP/WS
- Tools:
  - product_search
  - inventory_check
  - policy_qa
  - analytics_query
- Security:
  - Host-side consent, per-tool RBAC, schema-validated inputs
  - Rate limiting, audit logs, API keys/JWTs
- Caching: query → results; chunk cache for hot SKUs

MCP Tool Request (example)
```
POST /mcp/tools/call
{
  "method": "tools/call",
  "params": {
    "name": "product_search",
    "arguments": {
      "query": "wireless headphones under $100",
      "filters": {"brand": ["Sony","Bose"], "price": [0,100]},
      "limit": 12
    }
  }
}
```

---

## 9. Python Reference Skeletons

Embedding & Storage
```python
import chromadb, requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def embed_text(text: str) -> list[float]:
    # If Ollama provides embedding endpoint for Gemma3, call it here.
    # Otherwise, replace with a compact sentence-embedder and keep Gemma3 for generation.
    raise NotImplementedError

client = chromadb.Client()
collection = client.get_or_create_collection("catalog_chunks")

def upsert_chunk(doc_id, text, metadata):
    emb = embed_text(text)
    collection.upsert(ids=[doc_id], embeddings=[emb], metadatas=[metadata], documents=[text])
```

Query → Retrieval → Generation
```python
import requests

def generate_answer(query: str, context: str) -> str:
    prompt = f"""
SYSTEM: You are a retail assistant. Answer ONLY using the provided CONTEXT. If information is missing, say you don’t have enough information and ask a concise follow-up.
USER: {query}
CONTEXT:\n{context}
STYLE: Be concise, include SKU/IDs only when present in CONTEXT, no hallucinations.
"""
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma3:270m",
        "prompt": prompt,
        "stream": False
    })
    r.raise_for_status()
    return r.json().get("response", "")


def query_pipeline(query: str, k: int = 10) -> str:
    # 1) embed query
    qv = embed_text(query)
    # 2) retrieve
    res = collection.query(query_embeddings=[qv], n_results=k)
    docs = res.get("documents", [[]])[0]
    # 3) build context
    context = "\n\n".join(docs)
    # 4) generate
    return generate_answer(query, context)
```

---

## 10. Evaluation & KPIs

Retrieval
- recall@k, MRR, nDCG on labeled query→SKU/policy pairs
- hit@k on must-have chunks (policies, returns)

Generation
- Groundedness: proportion of statements supported by retrieved chunks
- Policy‑faithfulness: no contradictions with policy text
- Support resolution: deflection rate, CSAT proxy

Latency & Cost
- Retrieval latency p95 < 200ms; E2E p95 < 600ms (CPU baseline)
- Memory footprint: keep < 2GB for service tier

---

## 11. Security & Compliance

- MCP host consent (user-visible action approvals)
- Tool allowlist and strict parameter schemas
- PII minimization; encrypt in transit/at rest; RBAC by tenant
- PCI/GDPR alignment for transactional reads; no write-backs in first release

---

## 12. Deployment

Local Dev
```
ollama serve &
ollama run gemma3:270m
chroma start --host 0.0.0.0 --port 8001
uvicorn src.mcp_server:app --reload --port 8000
```

Docker Compose (outline)
```
services:
  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: ["ollama:/root/.ollama"]
  chroma:
    image: ghcr.io/chroma-core/chroma:latest
    ports: ["8001:8001"]
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8001
  api:
    build: .
    depends_on: [ollama, chroma]
    ports: ["8000:8000"]
volumes:
  ollama:
```

---

## 13. Roadmap

- v0.1 Baseline RAG with Gemma3:270m + ChromaDB + MCP tools (search, policy)
- v0.2 Multilingual normalizer, query rewriting, better chunkers
- v0.3 Recommendations (similar/complimentary), basic analytics_query
- v0.4 Reranker and adapter-tuned tasks if training path available
- v1.0 Security hardening, A/B testing, production SLOs

---

## 14. Contributing & Support

- PRs welcome. Use black/flake8/mypy and add tests.
- Open issues for feature requests or questions.

---

## 15. Visuals

- Architecture diagram and SLM tuning flowchart are attached to the project documentation and referenced in issues/wiki.
