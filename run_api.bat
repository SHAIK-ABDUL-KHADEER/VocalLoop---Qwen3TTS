@echo off
REM Qwen3-TTS HTTP API on http://127.0.0.1:9000
REM   docs: /docs    demo UI: /demo    status: /health
REM
REM No settings needed - device, models-resident and ASR placement auto-tune to
REM this machine's GPU. GET /health -> "config" shows what was chosen.
"%~dp0venv\Scripts\python.exe" "%~dp0server\app.py"
pause
