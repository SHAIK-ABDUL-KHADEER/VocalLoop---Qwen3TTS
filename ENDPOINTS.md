# Qwen3-TTS API — Endpoint Reference

**8 endpoints.** Three generate audio, two transcribe, three return metadata.

| # | Method | Path | Purpose | Auth |
|---|---|---|---|---|
| 1 | POST | `/v1/tts` | Text to speech — **built-in voice or described voice** | key |
| 2 | POST | `/v1/tts/clone` | Clone a voice from a reference clip (JSON) | key |
| 3 | POST | `/v1/tts/clone/upload` | Clone, multipart file upload | key |
| 4 | POST | `/v1/transcribe` | Speech to text (JSON) | key |
| 5 | POST | `/v1/transcribe/upload` | Speech to text, multipart | key |
| 6 | GET | `/v1/voices` | The 9 built-in voices | key |
| 7 | GET | `/v1/languages` | Supported languages | key |
| 8 | GET | `/health` | Status, VRAM, resolved config | **open** |

Plus `GET /demo` — a working test UI — and `GET /docs` for interactive Swagger. Both open.

“key” means the `X-API-Key` header is required **only when `QWEN_TTS_API_KEY` is set** on the
server. Locally it is unset, so no header is needed.

**Excluded by design:** model size (always 1.7B, the largest released) and x-vector-only
cloning (full-fidelity cloning is always used).

---

## The key idea: `/v1/tts` has two modes

Your frontend does not choose an endpoint. It always posts to `/v1/tts`; the presence of
`speaker` selects the mode.

| `speaker` | `instruct` | Mode | Model used |
|---|---|---|---|
| given | optional — restyles the voice | `built_in_voice` | CustomVoice 1.7B |
| omitted | **required** — describes the voice | `described_voice` | VoiceDesign 1.7B |

```js
{ text: "Hello there.", speaker: "vivian" }                        // voice picked
{ text: "Hello there.", speaker: "vivian", instruct: "Angrily." }  // voice + style
{ text: "Hello there.", instruct: "An elderly wizard, gravelly." } // nothing picked
```

The JSON response echoes `"mode"` so you can confirm which path ran. Sending neither
`speaker` nor `instruct` returns `422` — there would be nothing to determine the voice from.

---

## Conventions

### Language
Optional everywhere; defaults to `"auto"` (detected from the text). Values:
`auto, chinese, english, french, german, italian, japanese, korean, portuguese, russian, spanish`.
Fetch the live list from `GET /v1/languages`.

### Response format
Every audio endpoint accepts `response_format`:

- **`"json"`** (default) — base64 WAVs, supports batches.
- **`"wav"`** — raw `audio/wav` bytes, ready for `<audio>`. Single item only; a batch returns `400`.

```json
{
  "sample_rate": 24000,
  "count": 1,
  "durations": [3.12],
  "generation_seconds": 8.4,
  "mode": "built_in_voice",
  "audio": ["<base64 wav>"]
}
```

Audio is 24 kHz mono 16-bit PCM WAV.

### Batching
`text` accepts a list. `speaker` / `instruct` / `language` then take either a single value
(applied to every item) or a list of matching length; any other length returns `400`.

### Sampling parameters
All audio endpoints also accept these, all optional:
`max_new_tokens`, `do_sample`, `top_k`, `top_p`, `temperature`, `repetition_penalty`,
`subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`, `subtalker_temperature`.

### Audio inputs
`ref_audio` / `audio` accept four forms: a public **URL**, a **server-local path**, a **bare
base64** WAV string, or a **`data:audio/wav;base64,…` URI**. Or use the `/upload` variants to
post a file.

> The upstream library decides “is this base64?” by checking the string contains no `/` —
> which standard base64 uses constantly, so bare base64 gets misread as a filename. This
> server rewraps it as a data URI, so all four forms work here. If you call `qwen_tts`
> directly elsewhere, send a data URI.

### Errors

| Status | Meaning |
|---|---|
| `400` | Bad combination — e.g. `response_format:"wav"` with a batch, or a mismatched list length |
| `401` | Missing/incorrect `X-API-Key` (only when the server has a key configured) |
| `422` | Missing or malformed field; the body names the offending field |
| `500` | Unknown speaker, or a generation failure |
| `503` | A required model is not downloaded on the server |

---

## 1. `POST /v1/tts`

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `text` | string \| string[] | **yes** | — | What to say |
| `speaker` | string \| string[] | no | — | One of the 9 voices; omit to describe one |
| `instruct` | string \| string[] | conditional | — | **Required when `speaker` is omitted** |
| `language` | string \| string[] | no | `auto` | |
| `response_format` | `"json"` \| `"wav"` | no | `json` | |

```bash
# built-in voice
curl -X POST http://127.0.0.1:9000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is Qwen3 TTS.","speaker":"ryan","language":"english"}'

# built-in voice, restyled
curl -X POST http://127.0.0.1:9000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"I cannot believe you did that.","speaker":"vivian","instruct":"Furious, sharp tone."}'

# described voice (no speaker)
curl -X POST http://127.0.0.1:9000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Welcome, traveller.","instruct":"An elderly wizard, deep gravelly voice."}'
```

Batch with a different voice per line:

```json
{ "text": ["First line.", "Second line."], "speaker": ["ryan", "aiden"], "language": "english" }
```

### The 9 built-in voices

`GET /v1/voices` returns these with descriptions — build your picker from it, not from a
hardcoded copy.

| id | Voice | Native language |
|---|---|---|
| `vivian` | Bright, slightly edgy young female | Chinese |
| `serena` | Warm, gentle young female | Chinese |
| `uncle_fu` | Seasoned male, low and mellow | Chinese |
| `dylan` | Youthful Beijing male, clear and natural | Chinese (Beijing) |
| `eric` | Lively Chengdu male, husky brightness | Chinese (Sichuan) |
| `ryan` | Dynamic male, strong rhythmic drive | English |
| `aiden` | Sunny American male, clear midrange | English |
| `ono_anna` | Playful Japanese female, light and nimble | Japanese |
| `sohee` | Warm Korean female, rich emotion | Korean |

Any voice can speak any supported language, but sounds best in its native one.

> **Described voices drift.** Each call re-invents the voice from your description, so timbre
> varies between requests. Built-in voices are stable. For one consistent custom voice across
> many generations, clone from a fixed reference clip instead.

---

## 2. `POST /v1/tts/clone`

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `text` | string \| string[] | **yes** | — | What the cloned voice says |
| `ref_audio` | string | **yes** | — | URL / path / base64 / data URI |
| `ref_text` | string | no | auto | Transcript of `ref_audio`; **transcribed automatically if omitted** |
| `language` | string \| string[] | no | `auto` | |
| `response_format` | `"json"` \| `"wav"` | no | `json` | |

```bash
curl -X POST http://127.0.0.1:9000/v1/tts/clone \
  -H "Content-Type: application/json" \
  -d '{"text":"This is spoken in a cloned voice.","ref_audio":"https://example.com/sample.wav"}'
```

The response reports the transcript actually used:

```json
{ "ref_text": "Okay. Yeah. I resent you...", "ref_text_auto_generated": true, "audio": ["..."] }
```

With `response_format: "wav"` there is no JSON body, so the transcript comes back in the
**`X-Reference-Text`** response header (listed in `expose_headers`, so browser JS can read it).

**Reference clip guidance:** 3–10 seconds of clean single-speaker speech is plenty. An
accurate `ref_text` measurably improves the result — see the recommended flow below.

### `POST /v1/tts/clone/upload`

Identical, as multipart form data. Use this from a browser file input.

| Field | Type | Required |
|---|---|---|
| `text` | form field | **yes** |
| `ref_audio` | **file** | **yes** |
| `ref_text` | form field | no |
| `language` | form field | no |
| `response_format` | form field | no |

```bash
curl -X POST http://127.0.0.1:9000/v1/tts/clone/upload \
  -F "text=This is a cloned voice." \
  -F "ref_audio=@sample.wav"
```

---

## 3. `POST /v1/transcribe`

```json
{ "audio": "https://example.com/sample.wav" }
```
→
```json
{ "text": "Okay. Yeah. I resent you...", "seconds": 1.8 }
```

`POST /v1/transcribe/upload` is the multipart variant — one file field, `audio`.

### Recommended clone flow

1. User uploads a clip → call **`/v1/transcribe/upload`**.
2. Show the returned text in an **editable** box so they can fix ASR mistakes.
3. Send that corrected text as `ref_text` to **`/v1/tts/clone/upload`**.

This gives the "transcript appears automatically" UX, keeps the user in control of accuracy,
and runs ASR once instead of on every generation. If you would rather make one call, just
omit `ref_text` and let the clone endpoint transcribe internally.

A working implementation is at **`/demo`** — source in
[server/static/demo.html](server/static/demo.html), and its fetch calls are copy-pasteable.

---

## 4. Metadata

### `GET /v1/voices`
```json
{ "voices": [ { "id": "ryan", "description": "Dynamic male voice…", "native_language": "English" } ] }
```

### `GET /v1/languages`
```json
{ "languages": ["auto", "chinese", "english", "..."] }
```

### `GET /health` — open, no key required
```json
{
  "status": "ok",
  "cuda": true,
  "gpu": { "name": "NVIDIA A100-SXM4-80GB", "allocated_gb": 14.2, "total_gb": 79.2 },
  "resident_models": ["voice_design", "custom_voice", "base"],
  "asr_loaded": true,
  "config": {
    "device": "cuda:0",
    "asr_device": "cuda:0",
    "max_resident": 3,
    "attn_implementation": "flash_attention_2",
    "auth": "api_key"
  },
  "models_downloaded": { "voice_design": true, "custom_voice": true, "base": true }
}
```

`config` shows what the server auto-detected — **check `device` and `asr_device` are
`cuda:0` to confirm you are running on the GPU.**

---

## Calling it from your website

```js
const API = "https://<POD_ID>-9000.proxy.runpod.net";
const headers = { "Content-Type": "application/json", "X-API-Key": KEY };

// One call site for all TTS. Pass speaker when the user picked a voice.
async function speak({ text, speaker, instruct, language = "auto" }) {
  const res = await fetch(`${API}/v1/tts`, {
    method: "POST",
    headers,
    body: JSON.stringify({ text, speaker, instruct, language, response_format: "wav" }),
  });
  if (!res.ok) throw new Error(await res.text());
  new Audio(URL.createObjectURL(await res.blob())).play();
}

// Clone: transcribe on upload, let the user correct it, then generate.
async function transcribe(file) {
  const fd = new FormData();
  fd.append("audio", file);
  const res = await fetch(`${API}/v1/transcribe/upload`, {
    method: "POST", headers: { "X-API-Key": KEY }, body: fd,
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()).text;
}

async function speakCloned({ text, file, refText, language = "auto" }) {
  const fd = new FormData();
  fd.append("text", text);
  fd.append("ref_audio", file);
  if (refText) fd.append("ref_text", refText);   // omit to auto-transcribe
  fd.append("language", language);
  fd.append("response_format", "wav");
  const res = await fetch(`${API}/v1/tts/clone/upload`, {
    method: "POST", headers: { "X-API-Key": KEY }, body: fd,
  });
  if (!res.ok) throw new Error(await res.text());
  new Audio(URL.createObjectURL(await res.blob())).play();
}
```

⚠️ **Never ship the API key to the browser.** Proxy these calls through your own backend,
which holds the key. See [DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md) §8.

---

## Running it locally

```powershell
git clone https://github.com/SHAIK-ABDUL-KHADEER/VocalLoop---Qwen3TTS.git
cd VocalLoop---Qwen3TTS

py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -U pip
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -U qwen-tts fastapi "uvicorn[standard]" python-multipart requests

# Model weights are not in this repo - see DEPLOY_RUNPOD.md section 3 for the
# `hf download` commands. They go in ./models/ (~14.2 GB).

.\run_api.bat                                        # or: .\venv\Scripts\python.exe server\app.py
.\venv\Scripts\python.exe test_endpoints.py          # full suite
.\venv\Scripts\python.exe test_endpoints.py --filter clone
```

- Interactive docs: <http://127.0.0.1:9000/docs>
- Demo UI: <http://127.0.0.1:9000/demo>

**Launch it with the venv Python**, not the IDE's Run button — that uses the system
interpreter, which lacks these packages, and a second server fighting for port 9000 causes
confusing failures. Clear strays first:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*app.py*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

---

## Configuration

Every value auto-detects from the hardware; the environment variables are overrides. **No
code edits are needed to deploy.**

| Env var | Auto-detected default | Purpose |
|---|---|---|
| `QWEN_TTS_MODEL_DIR` | `<repo>/models` | Where the model folders live (repo-relative, so no edit needed) |
| `QWEN_TTS_DEVICE` | `cuda:0` if a GPU exists, else `cpu` | Compute device |
| `QWEN_TTS_MAX_RESIDENT` | `3` if ≥24 GB VRAM, `2` if ≥16 GB, else `1` | TTS models kept on the GPU |
| `QWEN_TTS_ASR_MODEL` | `<MODEL_DIR>/whisper-large-v3-turbo` | ASR model dir or hub id |
| `QWEN_TTS_ASR_DEVICE` | GPU if ≥16 GB VRAM, else `cpu` | Where Whisper runs |
| `QWEN_TTS_ATTN` | `flash_attention_2` if installed, else `sdpa` | Attention kernel |
| `QWEN_TTS_API_KEY` | unset → open | Require this in `X-API-Key` |
| `QWEN_TTS_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `QWEN_TTS_HOST` / `QWEN_TTS_PORT` | `127.0.0.1` / `9000` | Bind address |

### Memory

| Model | Size | Used by |
|---|---|---|
| VoiceDesign 1.7B | 4.2 GB | `/v1/tts` without `speaker` |
| CustomVoice 1.7B | 4.2 GB | `/v1/tts` with `speaker` |
| Base 1.7B | 4.2 GB | `/v1/tts/clone` |
| Whisper large-v3-turbo | 1.6 GB | `/v1/transcribe`, auto `ref_text` |

**A100 80 GB** — all four resident (14.2 GB), Whisper on GPU, nothing ever reloads.

**8 GB laptop** — one TTS model at a time, Whisper on CPU. Switching between `/v1/tts` modes
evicts and reloads a 4.2 GB model (10–30 s); repeated calls in the same mode are fast. This
is a hardware limit, not a code path you will hit in production.

---

## Concurrency

GPU work is serialized behind a single lock: one generation at a time, the rest queue,
because the models are not safe to call concurrently across threads. To raise throughput on
the A100, run several server processes on different ports behind a load balancer — see
[DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md) "Operational notes". Do not use `uvicorn --workers N`.

---

## Not exposed here

An earlier, broader API — reusable saved voices (register a clone once, reuse by `voice_id`),
design-then-clone, and codec encode/decode — is preserved in
[server/app_full.py](server/app_full.py). Saved voices are worth revisiting if your users
will reuse a cloned voice repeatedly: it avoids re-uploading the sample and re-extracting
speaker features on every call.
