from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os

app = FastAPI(title="Retail RAG MCP Server")


class ToolCall(BaseModel):
    method: str
    params: dict


def _embed_query(query: str) -> list[float]:
    """Embed the query, raising a 503-friendly error if the backend is down."""
    from src.embeddings.ollama_client import embed_text

    try:
        return embed_text(query)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing():
    return """
    <html>
      <head>
        <title>Retail RAG MCP Server</title>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; }
          h1 { margin-bottom: 0.25rem; }
          .muted { color: #666; margin-top: 0; }
          ul { line-height: 1.8; }
          code { background: #f6f8fa; padding: 2px 6px; border-radius: 4px; }
        </style>
      </head>
      <body>
        <h1>Retail RAG MCP Server</h1>
        <p class=\"muted\">ADK + Gemma3:270m (Ollama) + ChromaDB + FastAPI MCP</p>
        <h2>Health</h2>
        <ul>
          <li><a href=\"/healthz\">/healthz</a></li>
        </ul>
        <h2>API</h2>
        <ul>
          <li><code>POST /mcp/tools/call</code></li>
          <li><a href=\"/docs\">/docs</a> (Swagger UI, if enabled)</li>
          <li><a href=\"/redoc\">/redoc</a> (ReDoc, if enabled)</li>
        </ul>
        <h2>Docs</h2>
        <ul>
          <li><a href=\"https://github.com/abh1hi/ecommerce-retail-rag-mcp\" target=\"_blank\">Repository</a></li>
          <li><a href=\"/\" onclick=\"return false;\">Diagrams are in <code>docs/diagrams/</code> on GitHub</a></li>
        </ul>
      </body>
    </html>
    """


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def _product_search(query: str, top_k: int = 5):
    """Semantic product search backed by the ChromaDB vector store."""
    from src.rag.vector_store import query as vdb_query

    qv = _embed_query(query)
    res = vdb_query(qv, k=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return [
        {"id": metas[i].get("product_id", metas[i].get("id")) if i < len(metas) else None,
         "text": docs[i],
         "metadata": metas[i] if i < len(metas) else {}}
        for i in range(len(docs))
    ]


def _inventory_check(sku: str):
    """Read-only inventory placeholder. Wire to a real inventory service when available."""
    return {"sku": sku, "status": "unknown", "stock_level": None}


def _policy_qa(query: str):
    """Policy answer via RAG, scoped to the policies collection.

    Uses the policies collection (not the product catalog) so policy questions
    retrieve actual policy documents rather than product chunks.
    """
    from src.rag.vector_store import get_collection

    qv = _embed_query(query)
    col = get_collection("policies")
    res = col.query(query_embeddings=[qv], n_results=3)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return [
        {"id": metas[i].get("id") if i < len(metas) else None,
         "text": docs[i],
         "metadata": metas[i] if i < len(metas) else {}}
        for i in range(len(docs))
    ]


def _analytics_query(range_: str = "7d"):
    """Analytics placeholder. Wire to a metrics store when available."""
    return {"range": range_, "summary": "analytics not configured"}


@app.post("/mcp/tools/call")
async def tools_call(payload: ToolCall):
    if payload.method != "tools/call":
        raise HTTPException(status_code=400, detail=f"unsupported method: {payload.method}")
    name = payload.params.get("name")
    args = payload.params.get("arguments", {})

    if name == "product_search":
        return {"results": _product_search(args.get("query", ""), args.get("top_k", 5))}
    if name == "inventory_check":
        return {"results": _inventory_check(args.get("sku", ""))}
    if name == "policy_qa":
        return {"results": _policy_qa(args.get("query", ""))}
    if name == "analytics_query":
        return {"results": _analytics_query(args.get("range", "7d"))}
    raise HTTPException(status_code=404, detail=f"unknown tool: {name}")
