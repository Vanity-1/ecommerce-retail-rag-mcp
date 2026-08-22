from src.embeddings.ollama_client import generate_answer

GEN_TEMPLATE = """SYSTEM: Answer ONLY using CONTEXT. If insufficient, say so and ask one follow-up.
USER: {query}
CONTEXT:
{context}
STYLE: Concise, include SKU/IDs only if present, no hallucinations.
"""


async def generate(query: str, context: str = "") -> dict:
    """ADK FunctionTool entrypoint: produce a grounded answer from context."""
    prompt = GEN_TEMPLATE.format(query=query, context=context)
    out = generate_answer(prompt)
    return {"answer": out}


class GenerateTool:
    """Compatibility wrapper (old-style Tool) around the async function."""

    def __init__(self, model="gemma3:270m"):
        self.model = model

    async def call(self, ctx):
        query = ctx.state.get("query", "")
        context = ctx.state.get("context", "")
        return await generate(query, context)
