#!/usr/bin/env bash
set -euo pipefail
black .
flake8
mypy src || true
pytest -q
