"""Exercise every mode of POST /v1/speak and save the audio it returns.

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
API_KEY = None


def save_audio(name, b64):
    with open(os.path.join(OUT_DIR, f"{name}.wav"), "wb") as f:
        f.write(base64.b64decode(b64))


def check(name, method, url, *, expect=200, **kw):
    if FILTER and FILTER not in name:
        return None
    if API_KEY:
        kw.setdefault("headers", {})["X-API-Key"] = API_KEY

    t0 = time.time()
    try:
        r = requests.request(method, url, timeout=1800, **kw)
    except Exception as e:
        results.append((name, "ERROR", str(e)[:120]))
        print(f"  [ERROR] {name}: {e}")
        return None

    elapsed = time.time() - t0
    ok = r.status_code == expect
    note, body = "", None

    if ok and r.headers.get("content-type", "").startswith("audio/wav"):
        with open(os.path.join(OUT_DIR, f"{name}.wav"), "wb") as f:
            f.write(r.content)
        note = f"{len(r.content)/1024:.0f} KB wav | X-Mode={r.headers.get('X-Mode')}"
    elif ok:
        try:
            body = r.json()
        except Exception:
            note = "non-JSON body"
        if isinstance(body, dict) and body.get("audio"):
            for i, b64 in enumerate(body["audio"]):
                suffix = "" if len(body["audio"]) == 1 else f"_{i+1}"
                save_audio(f"{name}{suffix}", b64)
            note = (f"{body['count']} clip(s), {sum(body['durations']):.1f}s audio "
                    f"in {body.get('generation_seconds', 0):.1f}s | mode={body.get('mode')}")
            if body.get("ref_text"):
                auto = "auto" if body.get("ref_text_auto_generated") else "given"
                note += f' | {auto} ref="{body["ref_text"][:32]}"'
            if body.get("warnings"):
                note += f" | warn={len(body['warnings'])}"
        elif isinstance(body, dict) and "built_in" in body:
            note = (f"{len(body['built_in'])} built-in, {len(body['vocalist'])} vocalist, "
                    f"{len(body['languages'])} languages")
    else:
        try:
            note = json.dumps(r.json())[:150]
        except Exception:
            note = r.text[:150]

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}  ({r.status_code}, {elapsed:.1f}s) {note}")
    results.append((name, status, note))
    return body if ok else None


def main():
    global FILTER, API_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:9000")
    ap.add_argument("--filter", default=None, help="only run checks whose name contains this")
    ap.add_argument("--api-key", default=os.environ.get("QWEN_TTS_API_KEY"))
    args = ap.parse_args()
    B = args.base_url.rstrip("/")
    FILTER, API_KEY = args.filter, args.api_key

    SPEAK = f"{B}/v1/speak"
    ref_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
    ref_text = ("Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
                "You blew it! And thanks to you.")

    print("\n=== metadata ===")
    check("health", "GET", f"{B}/health")
    check("voices", "GET", f"{B}/v1/voices")

    # Fetched unconditionally, outside check(), because --filter would otherwise
    # skip it and every vocalist test would silently self-skip for lack of ids.
    vocalists = []
    try:
        hdr = {"X-API-Key": API_KEY} if API_KEY else {}
        vocalists = [v["id"] for v in
                     requests.get(f"{B}/v1/voices", headers=hdr, timeout=60).json()["vocalist"]]
        print(f"         vocalist voices: {vocalists or 'none installed'}")
    except Exception as e:
        print(f"         could not list vocalist voices: {e}")

    print("\n=== mode: described_voice  (no voice source) ===")
    check("described", "POST", SPEAK,
          json={"text": "Welcome, traveller. You have journeyed far to reach this place.",
                "instruct": "An elderly wizard, deep gravelly voice, slow and wise.",
                "language": "english"})
    check("described_autolang", "POST", SPEAK,
          json={"text": "Bonjour, comment allez-vous aujourd'hui?",
                "instruct": "A warm, friendly woman in her thirties."})
    check("described_batch", "POST", SPEAK,
          json={"text": ["Get down, now!", "It is going to be alright, I promise."],
                "instruct": ["Urgent military commander, clipped and forceful.",
                             "A soft, reassuring nurse, warm and calm."],
                "language": "english"})

    print("\n=== mode: built_in_voice  (speaker) ===")
    check("builtin", "POST", SPEAK,
          json={"text": "Hello, this is the Qwen3 text to speech engine speaking.",
                "speaker": "ryan", "language": "english"})
    check("builtin_instruct", "POST", SPEAK,
          json={"text": "I cannot believe you did that. This is unacceptable.",
                "speaker": "vivian", "language": "english",
                "instruct": "Speak in a furious, sharp tone."})
    check("builtin_batch", "POST", SPEAK,
          json={"text": ["The first sentence is spoken here.",
                         "And the second sentence follows after it."],
                "speaker": ["ryan", "aiden"], "language": "english"})
    check("builtin_params", "POST", SPEAK,
          json={"text": "Testing sampling parameters end to end.", "speaker": "aiden",
                "language": "english", "temperature": 0.7, "top_p": 0.9, "top_k": 40,
                "do_sample": True})
    check("builtin_wav_bytes", "POST", SPEAK,
          json={"text": "This response is raw wav bytes, not JSON.", "speaker": "ryan",
                "language": "english", "response_format": "wav"})

    print("\n=== mode: vocalist_clone  (voice_id) ===")
    if vocalists:
        vid = vocalists[0]
        check("vocalist", "POST", SPEAK,
              json={"text": "This line is sung by a voice from the library.",
                    "voice_id": vid, "language": "english"})
        check("vocalist_batch", "POST", SPEAK,
              json={"text": ["First vocalist line.", "Second vocalist line."],
                    "voice_id": vid, "language": "english"})
        check("vocalist_wav_bytes", "POST", SPEAK,
              json={"text": "Vocalist voice as raw wav.", "voice_id": vid,
                    "language": "english", "response_format": "wav"})
        check("vocalist_instruct_warns", "POST", SPEAK,
              json={"text": "Instruct should be ignored here.", "voice_id": vid,
                    "instruct": "Very angry.", "language": "english"})
    else:
        print("  [SKIP] no vocalist voices installed - run tools/make_sample_vocalists.py")
    check("vocalist_unknown", "POST", SPEAK, expect=404,
          json={"text": "Should fail.", "voice_id": "definitely_not_a_voice"})

    print("\n=== mode: voice_clone  (audio) ===")
    check("clone_url_auto_reftext", "POST", SPEAK,
          json={"text": "This sentence is spoken in a cloned voice.",
                "audio": ref_url, "language": "english"})
    check("clone_url_given_reftext", "POST", SPEAK,
          json={"text": "This one supplies the transcript explicitly.",
                "audio": ref_url, "ref_text": ref_text, "language": "english"})
    check("clone_batch", "POST", SPEAK,
          json={"text": ["First cloned line.", "Second cloned line."],
                "audio": ref_url, "ref_text": ref_text, "language": "english"})
    check("clone_wav_bytes", "POST", SPEAK,
          json={"text": "Raw wav plus the transcript in a header.",
                "audio": ref_url, "ref_text": ref_text,
                "language": "english", "response_format": "wav"})

    sample = os.path.join(OUT_DIR, "builtin.wav")
    sample_text = "Hello, this is the Qwen3 text to speech engine speaking."
    if os.path.isfile(sample):
        with open(sample, "rb") as f:
            raw = f.read()
        check("clone_upload", "POST", SPEAK,
              files={"audio": ("ref.wav", raw, "audio/wav")},
              data={"text": "Cloned from a multipart upload.",
                    "ref_text": sample_text, "language": "english"})
        check("clone_upload_auto_reftext", "POST", SPEAK,
              files={"audio": ("ref.wav", raw, "audio/wav")},
              data={"text": "Uploaded with no transcript supplied.", "language": "english"})
        check("clone_upload_wav_bytes", "POST", SPEAK,
              files={"audio": ("ref.wav", raw, "audio/wav")},
              data={"text": "Upload returning raw wav.", "ref_text": sample_text,
                    "language": "english", "response_format": "wav"})
        b64 = base64.b64encode(raw).decode("ascii")
        check("clone_raw_base64", "POST", SPEAK,
              json={"text": "Cloned from raw base64 audio.", "audio": b64,
                    "ref_text": sample_text, "language": "english"})
        check("clone_data_uri", "POST", SPEAK,
              json={"text": "Cloned from a data URI.",
                    "audio": "data:audio/wav;base64," + b64,
                    "ref_text": sample_text, "language": "english"})

    print("\n=== validation ===")
    check("err_missing_text", "POST", SPEAK, expect=422,
          json={"instruct": "Text is mandatory so this must fail."})
    check("err_no_voice_no_instruct", "POST", SPEAK, expect=422,
          json={"text": "Neither a voice source nor instruct, so this must fail."})
    check("err_two_voice_sources", "POST", SPEAK, expect=422,
          json={"text": "Two voice sources must be rejected.",
                "speaker": "ryan", "audio": ref_url})
    check("err_three_voice_sources", "POST", SPEAK, expect=422,
          json={"text": "Three is also wrong.", "speaker": "ryan",
                "voice_id": "aria_soul", "audio": ref_url})
    check("err_bad_speaker", "POST", SPEAK, expect=500,
          json={"text": "Unknown speaker.", "speaker": "not_a_real_speaker"})
    check("err_wav_batch", "POST", SPEAK, expect=400,
          json={"text": ["one", "two"], "speaker": "ryan", "response_format": "wav"})

    print("\n" + "=" * 74)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    for name, status, note in results:
        if status != "PASS":
            print(f"  {status:5}  {name}: {note}")
    print(f"  {passed}/{len(results)} passed   audio written to {OUT_DIR}")
    print("=" * 74)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
