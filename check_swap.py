"""Confirm VRAM and system RAM stay flat across model evictions."""
import requests
import psutil

B = "http://127.0.0.1:9000"


def stat(tag):
    h = requests.get(B + "/health").json()
    ram = psutil.virtual_memory()
    print(f"{tag:24} vram={h['gpu']['allocated_gb']:5.2f}GB  "
          f"resident={h['resident_models']}  ram_used={ram.percent:.0f}%")


described = {"text": "One two three.", "instruct": "A calm narrator.", "language": "english"}
builtin = {"text": "One two three.", "speaker": "ryan", "language": "english"}

stat("baseline")
for i in range(3):
    requests.post(B + "/v1/tts", json=described, timeout=900).raise_for_status()
    stat(f"round {i+1}: described")
    requests.post(B + "/v1/tts", json=builtin, timeout=900).raise_for_status()
    stat(f"round {i+1}: built-in")
