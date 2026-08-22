# Comparison Tables

Side-by-side comparisons to guide choices for deployment, retrieval, and modeling.

## Models (SLMs)

| Model | Params | Context | Strengths | Footprint | Notes |
|---|---|---|---|---|---|
| Gemma3:270m | 270M | 32K | Low latency, cheap | ~292MB | Great for on‑edge, tight SLOs |
| Gemma3:1B | 1B | 32K | Better reasoning | ~815MB | Higher latency/cost |
| Gemma3:4B | 4B | 128K | Stronger answers | ~3.3GB | GPU helpful |

## Vector DB Options

| DB | Pros | Cons | When to use |
|---|---|---|---|
| ChromaDB | Simple, local or server | Smaller ecosystem | Local/dev/embedded or simple prod |
| Qdrant | Robust, filters, payload | Extra infra | Managed prod with HA |
| Pinecone | Managed, scaling | Cost, vendor lock | Fully managed, elastic needs |

## Retrieval Strategies

| Strategy | Pros | Cons | Notes |
|---|---|---|---|
| Vector-only | Simple | Misses exact IDs | Start here for baseline |
| Hybrid (vector + filters) | Precision on metadata | Requires clean metadata | Retail default choice |
| Vector + Lexical | Better exact matches | More plumbing | Add for SKU/IDs |
| Add Reranker | Improves precision | Extra latency | Enable above threshold |

## Deployment

| Env | Pros | Cons | Notes |
|---|---|---|---|
| Local Compose | Fast iterate | Not prod | Dev only |
| Staging K8s | Realistic | Cost | Pre‑prod tests |
| Prod K8s/Cloud | HA/Autoscale | Complexity | SLOs, observability |
