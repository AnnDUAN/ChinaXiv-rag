#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
PYTHONUTF8=1 PYTHONWARNINGS=ignore::SyntaxWarning PYTHONPATH=src .venv/Scripts/python.exe -m rag "$@"
