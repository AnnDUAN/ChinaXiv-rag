@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONWARNINGS=ignore::SyntaxWarning
set PYTHONPATH=%~dp0src;%PYTHONPATH%
"%~dp0.venv\Scripts\python.exe" -m rag %*
