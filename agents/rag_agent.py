"""
Retail RAG agent built on the modern Google ADK API.

The classic example used `from adk.agents import Sequential` plus custom `Tool`
classes. With the maintained `google.adk` package (install via `google-adk`),
tools are plain async functions wrapped by ``FunctionTool``, and orchestration
is expressed either declaratively (a single ``Agent`` exposing both tools, to be
driven by an ADK ``Runner``) or programmatically (``RetailRAGAgent.run()``) —
the latter runs standalone without an LLM model / API key.
"""
from google.adk.tools import FunctionTool

from agents.tools.vector_search import retrieve
from agents.tools.generate import generate


def build_retrieve_tool(k: int = 10) -> FunctionTool:
    async def _retrieve(query: str) -> dict:
        return await retrieve(query, k=k)

    return FunctionTool(func=_retrieve)


def build_generate_tool() -> FunctionTool:
    return FunctionTool(func=generate)


class RetailRAGAgent:
    """
    Orchestrates normalize → retrieve → augment → generate.

    - normalize : trims/normalizes the raw user query.
    - retrieve  : embeds the query and fetches top-k chunks from ChromaDB.
    - generate  : produces a grounded answer from the retrieved context.

    Runnable directly (no ADK Runner / no model key required).
    """

    def __init__(self, k: int = 10):
        self.k = k
        self.retrieve_tool = build_retrieve_tool(k=k)
        self.generate_tool = build_generate_tool()

    async def normalize(self, query: str) -> dict:
        return {"query": query.strip()}

    async def run(self, query: str, context: str | None = None) -> dict:
        """Invoke the full pipeline directly (non-ADK path)."""
        norm = await self.normalize(query)
        q = norm["query"]

        if context is None:
            retrieved = await retrieve(q, k=self.k)
            docs = retrieved.get("docs", [])
            context = "\n\n".join(d.get("text", "") for d in docs)

        return await generate(q, context)


def build_pipeline(k: int = 10):
    """Declarative ADK agent exposing retrieve + generate to a Runner."""
    from google.adk.agents import Agent

    return Agent(
        name="retail_rag",
        description="Retrieve product/policy context and produce a grounded answer.",
        tools=[build_retrieve_tool(k=k), build_generate_tool()],
    )


# Compose the multi-source variant used in workflows (catalog + policies).
async def parallel_retrieve(query: str, collections=("catalog_chunks", "policies"), k: int = 8) -> dict:
    from src.rag import vector_store
    from src.embeddings.ollama_client import embed_text

    qv = embed_text(query)
    combined = []
    for col_name in collections:
        col = vector_store.get_collection(col_name)
        res = col.query(query_embeddings=[qv], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for i, txt in enumerate(docs):
            combined.append({"text": txt, "meta": metas[i] if i < len(metas) else {}})
    return {"count": len(combined), "docs": combined}