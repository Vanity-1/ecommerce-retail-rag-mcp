#!/usr/bin/env bash
set -euo pipefail
ollama serve &
sleep 2 || true
ollama pull gemma3:270m || true
chroma start --host 0.0.0.0 --port 8001 &
uvicorn src.mcp_server.main:app --host 0.0.0.0 --port 8000 --reload
