#!/usr/bin/env python3
"""
RAG-enhanced chat skeleton:
- Embeds user input (requires embed_text implementation)
- Retrieves top-k chunks from ChromaDB
- Augments prompt with retrieved context
- Generates grounded answer via Gemma3:270m
"""
import sys
import os

# Allow running from repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List
from src.embeddings.ollama_client import generate_answer, embed_text
from src.rag.vector_store import query as vdb_query

INTRO = """Retail RAG Chat (Gemma3:270m + ChromaDB)
Type 'quit' to exit.
"""

GEN_TEMPLATE = """SYSTEM: Answer ONLY using CONTEXT. If insufficient, say you do not have enough information and ask one concise follow-up.
USER: {user_input}
CONTEXT:\n{context}
STYLE: Be concise, include SKU/IDs only if present in CONTEXT, no hallucinations.
"""

def build_context(docs: List[str], max_chars: int = 3000) -> str:
    ctx = []
    length = 0
    for d in docs:
        if d is None:
            continue
        if length + len(d) > max_chars:
            break
        ctx.append(d)
        length += len(d)
    return "\n\n".join(ctx)


def chat_rag(k: int = 8):
    print(INTRO)
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bye!")
            break

        # 1) Embed query (requires embed_text implementation)
        try:
            qv = embed_text(user_input)
        except NotImplementedError:
            print("Embedding not implemented. Please implement embed_text() in src/embeddings/ollama_client.py")
            continue

        # 2) Retrieve from ChromaDB
        try:
            res = vdb_query(qv, k=k)
            docs = res.get("documents", [[]])[0]
        except Exception as e:
            print(f"Retrieval error: {e}")
            continue

        # 3) Build context
        context = build_context(docs)

        # 4) Generate grounded answer
        prompt = GEN_TEMPLATE.format(user_input=user_input, context=context)
        try:
            answer = generate_answer(prompt)
            print(f"AI: {answer}")
        except Exception as e:
            print(f"Generation error: {e}")

if __name__ == "__main__":
    chat_rag()
