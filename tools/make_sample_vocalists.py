"""Generate placeholder vocalist clips so the voice library is testable.

These are synthesized stand-ins, not real vocalists — replace them with your own
recordings. Because we know exactly what was spoken, the transcripts are written
into voices.json verbatim rather than being guessed by Whisper later.

    python tools/make_sample_vocalists.py [--base-url http://127.0.0.1:9000]
"""

import argparse
import base64
import json
import os
import sys

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(REPO, "voices")

# Built as described voices so they do not simply duplicate the 9 built-ins.
SAMPLES = [
    {
        "id": "aria_soul",
        "name": "Aria",
        "description": "Soulful female vocalist, warm and unhurried",
        "language": "english",
        "instruct": "A soulful female singer speaking warmly and slowly, rich low-mid tone.",
        "text": "Every song starts somewhere quiet, long before anybody hears it.",
    },
    {
        "id": "marcus_baritone",
        "name": "Marcus",
        "description": "Resonant male baritone, steady and deliberate",
        "language": "english",
        "instruct": "A resonant male baritone, steady and deliberate, with a full chest voice.",
        "text": "I have sung in rooms so small the walls sang back at me.",
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:9000")
    ap.add_argument("--api-key", default=os.environ.get("QWEN_TTS_API_KEY"))
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    os.makedirs(VOICE_DIR, exist_ok=True)
    manifest_path = os.path.join(VOICE_DIR, "voices.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = {e["id"]: e for e in json.load(f).get("voices", [])}

    for s in SAMPLES:
        print(f"generating {s['id']} …", flush=True)
        r = requests.post(
            f"{base}/v1/speak",
            headers=headers,
            json={"text": s["text"], "instruct": s["instruct"], "language": s["language"]},
            timeout=1200,
        )
        r.raise_for_status()
        wav = base64.b64decode(r.json()["audio"][0])
        path = os.path.join(VOICE_DIR, f"{s['id']}.wav")
        with open(path, "wb") as f:
            f.write(wav)

        manifest[s["id"]] = {
            "id": s["id"],
            "name": s["name"],
            "description": s["description"],
            "language": s["language"],
            "ref_text": s["text"],   # exact, so no transcription guesswork
        }
        print(f"  wrote {path} ({len(wav)/1024:.0f} KB)")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"voices": list(manifest.values())}, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {manifest_path} with {len(manifest)} voice(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
