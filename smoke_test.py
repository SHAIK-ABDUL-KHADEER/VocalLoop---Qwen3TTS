"""Quick smoke test: load the CustomVoice model on GPU and synthesize one line."""
import os
import time

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(REPO, "models", "Qwen3-TTS-12Hz-1.7B-CustomVoice")

t0 = time.time()
model = Qwen3TTSModel.from_pretrained(
    MODEL,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # flash_attention_2 is not available on Windows
)
print(f"loaded in {time.time() - t0:.1f}s")
print("speakers:", model.get_supported_speakers())
print("languages:", model.get_supported_languages())

t0 = time.time()
wavs, sr = model.generate_custom_voice(
    text="Hello from Qwen3 TTS, running locally on an RTX 4060.",
    language="English",
    speaker="Ryan",
)
print(f"generated in {time.time() - t0:.1f}s")

sf.write("smoke_test.wav", wavs[0], sr)
print(f"wrote smoke_test.wav  ({len(wavs[0]) / sr:.1f}s @ {sr} Hz)")
print("peak VRAM:", round(torch.cuda.max_memory_allocated() / 1024**3, 2), "GB")
