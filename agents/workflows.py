"""
Composite retail workflows built on the modern Google ADK API.

Classic example used `from adk.agents import Sequential, Parallel`. With the
maintained `google.adk` package, tools are plain async functions wrapped by
``FunctionTool`` and multi-source orchestration is provided as:

- ``multi_source_agent`` : a declarative ADK ``Agent`` (retrieve + generate).
- ``multi_source_generate`` : a direct, key-free programmatic helper.
"""
from google.adk.tools import FunctionTool

from agents.rag_agent import build_generate_tool, parallel_retrieve


# Single-source programmatic pipeline: retrieve → generate.
async def baseline_generate(query: str, k: int = 10) -> dict:
    from agents.tools.vector_search import retrieve
    from agents.tools.generate import generate

    res = await retrieve(query, k=k)
    docs = res.get("docs", [])
    context = "\n\n".join(d.get("text", "") for d in docs)
    return await generate(query, context)


def build_multi_source_tool(collections=("catalog_chunks", "policies"), k: int = 8) -> FunctionTool:
    async def _multi_source(query: str) -> dict:
        res = await parallel_retrieve(query, collections=collections, k=k)
        docs = res.get("docs", [])
        context = "\n\n".join(d.get("text", "") for d in docs)
        return {"query": query, "context": context}

    return FunctionTool(func=_multi_source)


def multi_source_agent(collections=("catalog_chunks", "policies"), k: int = 8):
    """Declarative ADK agent with multi-source retrieval + generation tools."""
    from google.adk.agents import Agent

    return Agent(
        name="multi_source_rag",
        description="Retrieve from multiple collections, then ground the answer.",
        tools=[build_multi_source_tool(collections=collections, k=k), build_generate_tool()],
    )


# Programmatic multi-source RAG entrypoint (no ADK Runner / model key required).
async def multi_source_generate(query: str, collections=("catalog_chunks", "policies"), k: int = 8) -> dict:
    from agents.tools.generate import generate

    res = await parallel_retrieve(query, collections=collections, k=k)
    docs = res.get("docs", [])
    context = "\n\n".join(d.get("text", "") for d in docs)
    return await generate(query, context)