# Testing Strategy & Test Plans

Comprehensive tests for retrieval, generation, tool integration, and performance.

## Test Matrix

| Area | Test | Goal | Tooling |
|---|---|---|---|
| Retrieval | recall@k, MRR, nDCG | Relevant chunks retrieved | pytest, eval scripts |
| Generation | Groundedness, Policy-faithfulness | Avoid hallucinations, comply with policy | rubric + auto checks |
| MCP Tools | Schema validation, RBAC | Safe tool calls, correct results | pydantic, httpx |
| Latency | p50/p95 E2E | SLO adherence | Locust/k6 |
| Load | RPS, saturation | Scale behavior | Locust/k6 |
| Security | Auth bypass, rate limits | Enforced controls | pytest + mocks |

## Unit Tests (pytest)
- src/rag/: chunking, rerank, context builder
- src/embeddings/: adapter contract
- src/mcp_server/: tool routing, validation, error paths

## Integration Tests
- Spin up Ollama + Chroma + API locally; exercise real flows via httpx
- Golden query set (search/support) with expected chunks and safe outputs

## Performance Tests
- Locust user classes: search-heavy, support-heavy
- Thresholds: retrieval < 200ms p95; E2E < 600ms p95

## Security Tests
- RBAC matrix per tool; fuzz invalid schemas; rate limit enforcement

## Example pytest
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_product_search_flow(app):
    async with AsyncClient(base_url="http://localhost:8000") as ac:
        payload = {
          "method":"tools/call",
          "params":{"name":"product_search","arguments":{"query":"laptop 16GB RAM"}}
        }
        r = await ac.post("/mcp/tools/call", json=payload)
        assert r.status_code == 200
        js = r.json()
        assert "results" in js
        assert len(js["results"]) > 0
```
