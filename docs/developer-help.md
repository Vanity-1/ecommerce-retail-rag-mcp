# Developer Help Guide

Practical steps and tips for day-to-day development.

## 1) Environments
- Copy env: `cp configs/.env.template .env`
- venv: `bash scripts/setup_venv.sh` then `source .venv/bin/activate`
- Docker: `docker compose up --build`

## 2) Running Services (Local)
- `ollama serve & && ollama pull gemma3:270m`
- `chroma start --host 0.0.0.0 --port 8001 &`
- `uvicorn src.mcp_server.main:app --host 0.0.0.0 --port 8000 --reload`
- One-shot: `bash scripts/dev_up_local.sh`

## 3) Common Scripts
- Checks: `bash scripts/checks.sh` (black, flake8, mypy, pytest)
- Docker dev: `bash scripts/dev_up_compose.sh`

## 4) Adding a New ADK Tool
- Create a tool under `agents/tools/your_tool.py`
- Register in your agent/workflow
- If it calls MCP, add an endpoint in `src/mcp_server/main.py`
- Add pydantic schemas for inputs/outputs, tests, and docs

## 5) Adding an MCP Endpoint
- Implement under `/mcp/tools/call` dispatch in `src/mcp_server/main.py`
- Validate inputs with pydantic; enforce RBAC
- Log inputs/outputs; add metrics counters and latencies

## 6) Retrieval & Embedding
- Implement `embed_text` in `src/embeddings/ollama_client.py`
  - Prefer Ollama embeddings for Gemma if available; otherwise use a compact sentence-embedder
- Ingestion: split field-aware chunks; store in Chroma with metadata

## 7) Debugging
- Use `/healthz` for quick checks
- Tail logs from uvicorn, monitor Chroma logs
- If answers hallucinate: Inspect `context` built by ADK, increase k, tune chunking, ensure strict prompts

## 8) Tests & Eval
- Unit/integration tests in `tests/` and ADK evals in `agents/eval/`
- Performance: add Locust/k6 profiles targeting `/mcp/tools/call`

## 9) Diagrams & Docs
- Diagrams index: `docs/diagrams/README.md`
- Update diagrams by editing Mermaid in each `.md`

## 10) Release Checklist
- Lint, type, and tests passing
- README and docs updated
- Security review of MCP tool schemas and rate limits
- Tag, build images, deploy to staging → prod
