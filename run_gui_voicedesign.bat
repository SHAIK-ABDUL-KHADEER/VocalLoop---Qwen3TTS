@echo off
REM Qwen3-TTS-12Hz-1.7B-VoiceDesign - invent a voice from a text description
"%~dp0venv\Scripts\qwen-tts-demo.exe" "%~dp0models\Qwen3-TTS-12Hz-1.7B-VoiceDesign" --no-flash-attn --ip 127.0.0.1 --port 8001
pause
