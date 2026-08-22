from src.rag.vector_store import query as vdb_query
from src.embeddings.ollama_client import embed_text


async def retrieve(query: str, k: int = 10) -> dict:
    """ADK FunctionTool entrypoint: embed the query and search the vector store."""
    qv = embed_text(query)
    res = vdb_query(qv, k=k)
    docs = []
    for i, txt in enumerate(res.get("documents", [[]])[0]):
        md = res.get("metadatas", [[]])[0][i]
        docs.append({"text": txt, "meta": md})
    return {"count": len(docs), "docs": docs}


class VectorSearchTool:
    """Compatibility wrapper (old-style) around the async retrieve function."""

    def __init__(self, k=10):
        self.k = k

    async def call(self, ctx):
        q = ctx.state.get("query", "")
        return await retrieve(q, k=self.k)
