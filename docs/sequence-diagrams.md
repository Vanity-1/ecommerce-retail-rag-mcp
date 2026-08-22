# Sequence Diagrams (Mermaid)

This document contains additional detailed sequence diagrams for critical flows.

## 1) Product Discovery with Hybrid Retrieval and Reranker

```mermaid
sequenceDiagram
  actor U as User
  participant UI as Web UI
  participant API as API Gateway
  participant MCP as MCP Server
  participant EMB as Embedder (Ollama/Gemma)
  participant VDB as ChromaDB
  participant GEN as Generator (Ollama/Gemma)

  U->>UI: Search "laptop 16GB RAM under $1000"
  UI->>API: GET /search?q=...
  API->>MCP: tools/call product_search
  MCP->>EMB: Embed query
  EMB-->>MCP: query_vector
  MCP->>VDB: KNN + metadata filters
  VDB-->>MCP: top-50 docs
  MCP->>GEN: Rerank(top-50)
  GEN-->>MCP: top-10 ranked
  MCP->>GEN: Generate grounded summary (top-10)
  GEN-->>MCP: answer + citations
  MCP-->>API: results payload
  API-->>UI: render results
```

## 2) Customer Support with Policy Guardrails

```mermaid
sequenceDiagram
  actor C as Customer
  participant UI as Support Chat
  participant API as API Gateway
  participant MCP as MCP Server
  participant POL as Policy Store (Chroma)
  participant ORD as Order System
  participant GEN as Generator (Ollama/Gemma)

  C->>UI: "Where is my order #12345?"
  UI->>API: POST /support/chat
  API->>MCP: tools/call inventory_check + policy_qa
  MCP->>ORD: Query order status
  ORD-->>MCP: status payload
  MCP->>POL: Retrieve return policy
  POL-->>MCP: policy chunks
  MCP->>GEN: Generate grounded response
  GEN-->>MCP: reply (grounded)
  MCP-->>API: response JSON
  API-->>UI: show status + policy
```

## 3) Analytics Insight Summary (Read-only)

```mermaid
sequenceDiagram
  actor A as Analyst
  participant UI as Dashboard
  participant API as API Gateway
  participant MCP as MCP Server
  participant DWH as Analytics Warehouse
  participant GEN as Generator (Ollama/Gemma)

  A->>UI: "show last 7d sales trend for top-5 SKUs"
  UI->>API: GET /analytics/trend?range=7d
  API->>MCP: tools/call analytics_query
  MCP->>DWH: Query aggregates
  DWH-->>MCP: timeseries + SKUs
  MCP->>GEN: Summarize with rationale
  GEN-->>MCP: natural language insight
  MCP-->>API: json payload
  API-->>UI: chart + summary
```
