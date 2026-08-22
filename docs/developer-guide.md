# Developer Guide: E-commerce RAG with Gemma3:270m + Ollama + ChromaDB + MCP

This guide covers architecture decisions, local setup, coding standards, directory layout, environment, CI/CD, and contribution workflow.

## Architecture Decisions (ADR Summary)
- ADR-001: Use SLM (Gemma3:270m) for low-latency grounded generation.
- ADR-002: ChromaDB for vector search with metadata filters and HNSW index.
- ADR-003: MCP servers for secure tool access (catalog_search, inventory_check, policy_qa, analytics_query).
- ADR-004: Field-aware chunking; atomic units for policies/FAQs.
- ADR-005: Quantization-first deployment; PEFT adapters if supported later.

## Directory Layout
```
/                 # repo root
docs/             # documentation & diagrams
src/              # application source
  mcp_server/     # MCP server implementation
  api/            # REST facade and gateway glue
  rag/            # retrieval & augmentation logic
  embeddings/     # embed abstraction & adapter
  eval/           # evaluation scripts
scripts/          # utilities, ingestion, demos
configs/          # env templates, yaml configs
```

## Environments
- `.env` – local development (OLLAMA_URL, CHROMA_HOST/PORT, LOG_LEVEL)
- `.env.staging`, `.env.prod` – injected via CI/CD secrets

## Coding Standards
- Python ≥ 3.10
- Format with black, lint with flake8, types via mypy
- Tests: pytest; aim for 80%+ coverage for src/
- Docstrings: Google or NumPy style

## MCP Contracts
- Tool names and JSON schema for parameters and responses stored under `configs/mcp/tools/*.json`
- Enforce strict input validation (pydantic) and RBAC per tool

## Error Handling & Observability
- Use structured logs (JSON) with trace/span IDs
- Emit Prometheus metrics: latency, throughput, hit@k, groundedness proxy
- Add SLO alerts for retrieval and E2E latency

## Contribution Workflow
- Branch naming: feature/, fix/, docs/
- Conventional commits: feat:, fix:, docs:, chore:
- PR requirements: lint, type-check, tests, updated docs
- Code review checklist provided in `.github/PULL_REQUEST_TEMPLATE.md`
