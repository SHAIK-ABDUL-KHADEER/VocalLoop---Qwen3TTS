# VocalLoop — Qwen3-TTS API

An HTTP API around [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B, built to be called
from a website. Three things it does:

- **Text to speech** with one of 9 built-in voices, optionally restyled by an instruction
- **Text to speech with a voice you describe in words** ("an elderly wizard, deep and gravelly")
- **Voice cloning** from an uploaded clip, with the reference transcript generated automatically

All of it behind one origin, with optional API-key auth and configurable CORS.

```bash
curl -X POST http://127.0.0.1:9000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello there.","speaker":"ryan","language":"english","response_format":"wav"}' \
  --output hello.wav
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/tts` | TTS — built-in voice (`speaker`) or described voice (`instruct`) |
| POST | `/v1/tts/clone` | Clone a voice from a reference clip (JSON) |
| POST | `/v1/tts/clone/upload` | Clone, multipart file upload |
| POST | `/v1/transcribe` | Speech to text (JSON) |
| POST | `/v1/transcribe/upload` | Speech to text, multipart |
| GET | `/v1/voices` | The 9 built-in voices |
| GET | `/v1/languages` | Supported languages |
| GET | `/health` | Status, VRAM, resolved config |

Plus `GET /demo` — a working test UI — and `GET /docs` for interactive Swagger.

**→ [ENDPOINTS.md](ENDPOINTS.md)** for every field, type, default, and error code.

### One endpoint, two TTS modes

`/v1/tts` picks its model from whether you send `speaker`, so the frontend has a single call site:

```js
{ text: "Hello.", speaker: "vivian" }                        // built-in voice
{ text: "Hello.", speaker: "vivian", instruct: "Angrily." }  // built-in voice, restyled
{ text: "Hello.", instruct: "An elderly wizard, gravelly." } // voice invented from the description
```

## Runs on the GPU it finds

Device, how many models stay resident, where Whisper runs, and the attention kernel are all
detected at startup — the same file runs unedited on an 8 GB laptop and an 80 GB A100.
`GET /health` → `config` reports what it chose.

| | 8 GB laptop | A100 80 GB |
|---|---|---|
| `device` | `cuda:0` | `cuda:0` |
| `max_resident` | 1 | 3 |
| `asr_device` | `cpu` | `cuda:0` |
| `attn_implementation` | `sdpa` | `flash_attention_2` |

Every value can be forced with an environment variable — see [ENDPOINTS.md](ENDPOINTS.md).

## Getting started

```bash
git clone https://github.com/SHAIK-ABDUL-KHADEER/VocalLoop---Qwen3TTS.git
cd VocalLoop---Qwen3TTS
python -m venv venv && source venv/bin/activate     # Windows: .\venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -U qwen-tts fastapi "uvicorn[standard]" python-multipart requests
```

Model weights (~14.2 GB) are **not** in this repo — download them with the `hf download`
commands in [DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md). They belong in `./models/`.

Then:

```bash
python server/app.py          # http://127.0.0.1:9000
python test_endpoints.py      # 23-check suite
```

`qwen-tts` installs from PyPI — this project does not patch the upstream library, so you do
not need to clone `QwenLM/Qwen3-TTS`. The workarounds for its rough edges live in
[server/app.py](server/app.py).

## Deploying

**→ [DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md)** — RunPod A100, no Docker. Pod setup, install,
model download, auth, systemd unit, verification, troubleshooting.

No code edits are required to deploy; everything is environment-driven.

## Layout

| Path | What |
|---|---|
| [server/app.py](server/app.py) | The API |
| [server/static/demo.html](server/static/demo.html) | Demo UI served at `/demo` |
| [server/app_full.py](server/app_full.py) | Larger earlier API — saved voices, codec endpoints |
| [test_endpoints.py](test_endpoints.py) | End-to-end suite; `--filter <name>` for a subset |
| [run_family_tests.ps1](run_family_tests.ps1) | Runs each endpoint family against a fresh server (small-GPU workaround) |
| [bench.py](bench.py) | Throughput / batch-scaling benchmark |
| [smoke_test.py](smoke_test.py) | Minimal library-level generation check |

## Notes

- Audio is 24 kHz mono 16-bit WAV. `response_format: "wav"` returns raw bytes for `<audio>`;
  the default `"json"` returns base64 and supports batching.
- GPU work is serialized behind one lock — one generation at a time. For more throughput, run
  several server processes behind a load balancer rather than `uvicorn --workers`.
- Cloning works best with 3–10 seconds of clean speech and an accurate transcript. Omit
  `ref_text` and the server transcribes with Whisper; better still, call `/v1/transcribe`
  first and let the user correct it before generating.

Model weights are licensed by Alibaba under the terms in the
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) repository.
