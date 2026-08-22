# ADK Integration Guide

This guide explains how Google Agent Development Kit (ADK) orchestrates the Retail RAG stack (Gemma3:270m via Ollama, ChromaDB, FastAPI MCP).

## Overview
- Orchestration: ADK Sequential/Parallel/Loop workflows for preprocess → retrieve → augment → generate.
- Tools: ADK tools wrap MCP endpoints and local adapters (vector search, generate).
- Evaluation: Use ADK’s evaluation to score responses and step-level traces.
- Deployment: Run locally (venv/Docker) or containerize ADK runner alongside the API.

## Directory
```
agents/
  rag_agent.py
  workflows.py
  tools/
    vector_search.py
    generate.py
    mcp_tools.py
  eval/
    cases.yaml
```

## Running (venv)
```
# services
ollama serve &; ollama pull gemma3:270m
chroma start --host 0.0.0.0 --port 8001 &
uvicorn src.mcp_server.main:app --host 0.0.0.0 --port 8000 --reload

# run agent (example script to be added)
python -m agents.run_demo "wireless headphones under $100"
```

## Notes
- embed_text must be implemented (Ollama embeddings or alternate sentence embedder).
- Keep grounding strict in prompts.
