"""Benchmark steady-state throughput and batch scaling for Qwen3-TTS 1.7B.

Reports real-time factor (RTF) = seconds of audio produced per second of wall clock.
RTF > 1 means one stream runs faster than real time; RTF at batch N tells you how
many concurrent streams a single model replica can sustain.
"""
import os
import time

import torch

from qwen_tts import Qwen3TTSModel

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(REPO, "models", "Qwen3-TTS-12Hz-1.7B-CustomVoice")

SENTENCES = [
    "The quick brown fox jumps over the lazy dog, and then it runs away into the forest.",
    "She said she would be here by noon, but the traffic downtown was much worse than expected.",
    "I am not entirely convinced that this is the best approach, but I am willing to try it.",
    "Please remember to bring your passport, boarding pass, and a copy of the reservation.",
]

model = Qwen3TTSModel.from_pretrained(
    MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
)

def run(batch_size):
    text = [SENTENCES[i % len(SENTENCES)] for i in range(batch_size)]
    torch.cuda.synchronize()
    t0 = time.time()
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=["English"] * batch_size,
        speaker=["Ryan"] * batch_size,
    )
    torch.cuda.synchronize()
    wall = time.time() - t0
    audio = sum(len(w) for w in wavs) / sr
    return wall, audio, sr

# warmup - first call pays CUDA kernel autotune + allocator costs
w, a, sr = run(1)
print(f"warmup (ignored): {w:.1f}s wall for {a:.1f}s audio")

for bs in (1, 2, 4, 8, 16, 32):
    try:
        torch.cuda.reset_peak_memory_stats()
        wall, audio, sr = run(bs)
        rtf = audio / wall
        print(
            f"batch {bs:>3}: {wall:6.1f}s wall | {audio:6.1f}s audio | "
            f"RTF {rtf:5.2f}x | {audio/bs/wall:5.2f}x per stream | "
            f"peak {torch.cuda.max_memory_allocated()/1024**3:.2f} GB"
        )
    except torch.cuda.OutOfMemoryError:
        print(f"batch {bs:>3}: OOM on 8 GB (A100 80 GB would be fine)")
        torch.cuda.empty_cache()
        break
