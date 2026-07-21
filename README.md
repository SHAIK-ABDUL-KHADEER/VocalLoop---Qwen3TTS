# VocalLoop — Qwen3-TTS API

An HTTP API around [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B, built to be called
from a website. **One endpoint does everything** — the request decides which model runs.

```bash
POST /v1/speak
```

| What you send | Mode | What happens |
|---|---|---|
| `text` + `instruct` | `described_voice` | A voice is invented from your description |
| `text` + `speaker` | `built_in_voice` | One of 9 shipped voices; `instruct` restyles it |
| `text` + `voice_id` | `vocalist_clone` | Clones a voice from your curated library |
| `text` + `audio` | `voice_clone` | Clones a clip the user uploaded |

```bash
curl -X POST http://127.0.0.1:9000/v1/speak \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello there.","speaker":"ryan","response_format":"wav"}' --output hello.wav
```

Two metadata endpoints support it — `GET /v1/voices` (populate your picker) and
`GET /health` (status). That is the whole API.

**→ [ENDPOINTS.md](ENDPOINTS.md)** for every field, mode, error code, and JS examples.

## Why one endpoint

Your frontend has a single call site. A voice picker lists built-in and vocalist voices
together; pick one and you send `speaker` or `voice_id`, pick none and you send `instruct`,
upload a file and you send `audio`. No branching between endpoints, no client-side model
selection.

Exactly one voice source may be sent. Two is a `422`, not a silent preference — that
combination almost always means a frontend bug.

## The vocalist voice library

Voices **you** supply, cloned on demand and offered to users alongside the 9 built-ins.

Drop an audio file into [voices/](voices/) — the filename stem becomes the `voice_id`, and it
shows up in `GET /v1/voices` immediately, no restart. Optional display metadata goes in
`voices/voices.json`. Transcripts are generated once with Whisper and cached.

**→ [voices/README.md](voices/README.md)** for clip guidance and the manifest format.

The two clips shipped here are synthesized placeholders so the repo is testable out of the
box — replace them with real recordings.

## Runs on the GPU it finds

Device, resident-model count, ASR placement, and attention kernel are all detected at
startup, so the same file runs unedited on an 8 GB laptop and an 80 GB A100.
`GET /health` → `config` reports what it chose.

| | 8 GB laptop | A100 80 GB |
|---|---|---|
| `device` | `cuda:0` | `cuda:0` |
| `max_resident` | 1 | 3 |
| `asr_device` | `cpu` | `cuda:0` |
| `attn_implementation` | `sdpa` | `flash_attention_2` |

Any of it can be forced with an environment variable — see
[ENDPOINTS.md](ENDPOINTS.md).

## Getting started

```bash
git clone https://github.com/SHAIK-ABDUL-KHADEER/VocalLoop---Qwen3TTS.git
cd VocalLoop---Qwen3TTS
python -m venv venv && source venv/bin/activate     # Windows: .\venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ./Qwen3-TTS
pip install fastapi "uvicorn[standard]" python-multipart requests
```

The Qwen3-TTS library is **vendored** in [Qwen3-TTS/](Qwen3-TTS/), so one clone is all you
need. It is an unmodified upstream checkout — see
[Qwen3-TTS/VENDORED.md](Qwen3-TTS/VENDORED.md) for the exact commit and how to update it.
(`pip install -U qwen-tts` from PyPI is equivalent at this version.)

Model weights (~14.2 GB) are **not** in this repo — download them with the `hf download`
commands in [DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md). They go in `./models/`.

Then:

```bash
python server/app.py          # http://127.0.0.1:9000
python test_endpoints.py      # full suite, 28 checks
```

Open <http://127.0.0.1:9000/demo> for a working UI that shows which mode each request will
use as you fill the form.

This project never patches the vendored library — every workaround for its rough edges lives
in [server/app.py](server/app.py), so the folder can be swapped for a newer upstream
checkout without losing anything.

## Deploying

**→ [DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md)** — RunPod A100, no Docker. Pod setup, install,
model download, auth, systemd unit, verification, troubleshooting.

No code edits are required to deploy; everything is environment-driven.

## Layout

| Path | What |
|---|---|
| [server/app.py](server/app.py) | The API |
| [Qwen3-TTS/](Qwen3-TTS/) | Vendored upstream library — see [VENDORED.md](Qwen3-TTS/VENDORED.md) |
| [server/static/demo.html](server/static/demo.html) | Demo UI served at `/demo` |
| [voices/](voices/) | Vocalist voice library |
| [tools/make_sample_vocalists.py](tools/make_sample_vocalists.py) | Regenerates the placeholder vocalist clips |
| [test_endpoints.py](test_endpoints.py) | End-to-end suite; `--filter <name>` for a subset |
| [run_family_tests.ps1](run_family_tests.ps1) | One mode per server process (small-GPU workaround) |
| [test_concurrency.py](test_concurrency.py) | Overlapping requests across modes |
| [bench.py](bench.py) | Throughput / batch-scaling benchmark |
| [server/app_full.py](server/app_full.py) | Earlier multi-endpoint API, kept for reference |

## Notes

- Audio is 24 kHz mono 16-bit WAV. `response_format: "wav"` returns raw bytes for `<audio>`
  with the mode in an `X-Mode` header; the default `"json"` returns base64 and supports batching.
- GPU work is serialized behind one lock — one generation at a time. For more throughput, run
  several server processes behind a load balancer rather than `uvicorn --workers`.
- Cloning works best with 3–10 seconds of clean speech. Supply `ref_text` when you know the
  exact words; otherwise it is transcribed automatically.
- On an 8 GB GPU only one model fits, so switching modes reloads it. Use
  [run_family_tests.ps1](run_family_tests.ps1) rather than the plain suite there.

Model weights are licensed by Alibaba under the terms in the
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) repository. You are responsible for having
the right to clone any voice you add to the library.
