"""Exercise every Qwen3-TTS API endpoint and save the audio it returns.

    python test_endpoints.py [--base-url http://127.0.0.1:9000] [--filter clone]

Writes decoded WAVs to test_output/ so you can listen to each result.
"""

import argparse
import base64
import json
import os
import sys
import time

import requests

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
os.makedirs(OUT_DIR, exist_ok=True)

results = []
FILTER = None


def save_audio(name, b64):
    path = os.path.join(OUT_DIR, f"{name}.wav")
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return path


def check(name, method, url, *, expect=200, save_as=None, **kw):
    if FILTER and FILTER not in name:
        return None
    t0 = time.time()
    try:
        r = requests.request(method, url, timeout=1200, **kw)
    except Exception as e:
        results.append((name, "ERROR", 0, str(e)[:120]))
        print(f"  [ERROR] {name}: {e}")
        return None

    elapsed = time.time() - t0
    ok = r.status_code == expect
    note = ""

    if ok and r.headers.get("content-type", "").startswith("audio/wav"):
        with open(os.path.join(OUT_DIR, f"{save_as or name}.wav"), "wb") as f:
            f.write(r.content)
        note = f"{len(r.content)/1024:.0f} KB wav"
        hdr = r.headers.get("X-Reference-Text")
        if hdr:
            note += f' | hdr ref_text="{hdr[:40]}"'
        body = None
    elif ok:
        try:
            body = r.json()
        except Exception:
            body, note = None, "non-JSON body"
        if isinstance(body, dict) and body.get("audio"):
            for i, b64 in enumerate(body["audio"]):
                suffix = "" if len(body["audio"]) == 1 else f"_{i+1}"
                save_audio(f"{save_as or name}{suffix}", b64)
            note = (f"{body['count']} clip(s), {sum(body['durations']):.1f}s audio "
                    f"in {body.get('generation_seconds', 0):.1f}s")
            if body.get("mode"):
                note += f" | mode={body['mode']}"
            if body.get("ref_text"):
                auto = "auto" if body.get("ref_text_auto_generated") else "given"
                note += f' | {auto} ref_text="{body["ref_text"][:38]}"'
        elif isinstance(body, dict) and "text" in body:
            note = f'"{body["text"][:60]}"'
    else:
        body = None
        try:
            note = json.dumps(r.json())[:140]
        except Exception:
            note = r.text[:140]

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}  ({r.status_code}, {elapsed:.1f}s) {note}")
    results.append((name, status, elapsed, note))
    return body if ok else None


def main():
    global FILTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:9000")
    ap.add_argument("--filter", default=None, help="only run checks whose name contains this")
    args = ap.parse_args()
    B = args.base_url.rstrip("/")
    FILTER = args.filter

    ref_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
    ref_text = ("Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
                "You blew it! And thanks to you.")

    print("\n=== meta ===")
    check("health", "GET", f"{B}/health")
    check("voices", "GET", f"{B}/v1/voices")
    check("languages", "GET", f"{B}/v1/languages")

    print("\n=== /v1/tts  -  no speaker -> described voice ===")
    check("tts_described", "POST", f"{B}/v1/tts",
          json={"text": "Welcome, traveller. You have journeyed far to reach this place.",
                "instruct": "An elderly wizard, deep gravelly voice, slow and wise.",
                "language": "english"})
    check("tts_described_autolang", "POST", f"{B}/v1/tts",
          json={"text": "Bonjour, comment allez-vous aujourd'hui?",
                "instruct": "A warm, friendly woman in her thirties."})
    check("tts_described_batch", "POST", f"{B}/v1/tts",
          json={"text": ["Get down, now!", "It is going to be alright, I promise."],
                "instruct": ["Urgent military commander, clipped and forceful.",
                             "A soft, reassuring nurse, warm and calm."],
                "language": "english"})

    print("\n=== /v1/tts  -  speaker given -> built-in voice ===")
    check("tts_speaker", "POST", f"{B}/v1/tts",
          json={"text": "Hello, this is the Qwen3 text to speech engine speaking.",
                "speaker": "ryan", "language": "english"})
    check("tts_speaker_instruct", "POST", f"{B}/v1/tts",
          json={"text": "I cannot believe you did that. This is unacceptable.",
                "speaker": "vivian", "language": "english",
                "instruct": "Speak in a furious, sharp tone."})
    check("tts_speaker_autolang", "POST", f"{B}/v1/tts",
          json={"text": "Language left unspecified, so it should auto-detect.",
                "speaker": "aiden"})
    check("tts_speaker_batch", "POST", f"{B}/v1/tts",
          json={"text": ["The first sentence is spoken here.",
                         "And the second sentence follows after it."],
                "speaker": ["ryan", "aiden"], "language": "english"})
    check("tts_speaker_params", "POST", f"{B}/v1/tts",
          json={"text": "Testing sampling parameters end to end.", "speaker": "aiden",
                "language": "english", "temperature": 0.7, "top_p": 0.9, "top_k": 40,
                "do_sample": True})
    check("tts_wav_bytes", "POST", f"{B}/v1/tts",
          json={"text": "This response is raw wav bytes, not JSON.", "speaker": "ryan",
                "language": "english", "response_format": "wav"})

    print("\n=== /v1/tts  -  validation ===")
    check("tts_missing_text", "POST", f"{B}/v1/tts", expect=422,
          json={"instruct": "Text is mandatory so this must fail."})
    check("tts_no_speaker_no_instruct", "POST", f"{B}/v1/tts", expect=422,
          json={"text": "Neither speaker nor instruct given, so this must fail."})
    check("tts_bad_speaker", "POST", f"{B}/v1/tts", expect=500,
          json={"text": "This should fail.", "speaker": "not_a_real_speaker"})

    print("\n=== /v1/transcribe ===")
    tr = check("transcribe_url", "POST", f"{B}/v1/transcribe", json={"audio": ref_url})
    if tr:
        print(f"         full transcript: {tr['text'][:100]}")

    print("\n=== /v1/tts/clone  -  ref_text auto-generated ===")
    check("clone_auto_reftext", "POST", f"{B}/v1/tts/clone",
          json={"text": "This sentence is spoken in a cloned voice.",
                "language": "english", "ref_audio": ref_url})
    check("clone_given_reftext", "POST", f"{B}/v1/tts/clone",
          json={"text": "This one supplies the transcript explicitly.",
                "language": "english", "ref_audio": ref_url, "ref_text": ref_text})
    check("clone_batch", "POST", f"{B}/v1/tts/clone",
          json={"text": ["First cloned line.", "Second cloned line."],
                "language": "english", "ref_audio": ref_url})
    check("clone_wav_bytes", "POST", f"{B}/v1/tts/clone",
          json={"text": "Raw wav plus the transcript in a header.",
                "language": "english", "ref_audio": ref_url, "response_format": "wav"})

    sample = os.path.join(OUT_DIR, "tts_speaker.wav")
    if os.path.isfile(sample):
        with open(sample, "rb") as f:
            raw = f.read()
        check("clone_upload_auto", "POST", f"{B}/v1/tts/clone/upload",
              files={"ref_audio": ("ref.wav", raw, "audio/wav")},
              data={"text": "Cloned from an uploaded file, transcript auto-generated.",
                    "language": "english"})
        b64 = base64.b64encode(raw).decode("ascii")
        check("clone_raw_base64", "POST", f"{B}/v1/tts/clone",
              json={"text": "Cloned from raw base64 audio.", "language": "english",
                    "ref_audio": b64})
        check("transcribe_upload", "POST", f"{B}/v1/transcribe/upload",
              files={"audio": ("ref.wav", raw, "audio/wav")})

    print("\n" + "=" * 72)
    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    for name, status, elapsed, note in results:
        if status != "PASS":
            print(f"  {status:5}  {name}: {note}")
    print(f"  {passed}/{len(results)} passed   audio written to {OUT_DIR}")
    print("=" * 72)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
