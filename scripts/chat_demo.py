#!/usr/bin/env python3
import sys
import os

# Allow running from repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.embeddings.ollama_client import generate_answer

INTRO = """Retail RAG Chat (Gemma3:270m via Ollama)
Type 'quit' to exit.
"""

PROMPT_TEMPLATE = """You are a helpful retail assistant.
Keep answers concise and avoid hallucinations.

User: {user_input}
Assistant:"""

def chat():
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

        prompt = PROMPT_TEMPLATE.format(user_input=user_input)
        try:
            response = generate_answer(prompt)
            print(f"AI: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat()
