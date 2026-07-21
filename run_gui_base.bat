@echo off
REM Qwen3-TTS-12Hz-1.7B-Base - 3-second voice cloning from a reference clip
"%~dp0venv\Scripts\qwen-tts-demo.exe" "%~dp0models\Qwen3-TTS-12Hz-1.7B-Base" --no-flash-attn --ip 127.0.0.1 --port 8002
pause
