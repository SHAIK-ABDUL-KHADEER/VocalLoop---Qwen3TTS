@echo off
REM Qwen3-TTS-12Hz-1.7B-CustomVoice - 9 premium voices + instruction style control
"%~dp0venv\Scripts\qwen-tts-demo.exe" "%~dp0models\Qwen3-TTS-12Hz-1.7B-CustomVoice" --no-flash-attn --ip 127.0.0.1 --port 8000
pause
