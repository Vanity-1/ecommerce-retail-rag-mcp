#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n configs/.env.template .env || true
echo "Virtual env ready. Activate with: source .venv/bin/activate"
