# Qwen3-TTS API — Endpoint Reference

**One endpoint generates speech.** What you put in the request decides which model runs.

```
POST /v1/speak
```

Two metadata endpoints support it: `GET /v1/voices` to populate your picker, and
`GET /health` for status. That is the whole API.

| Method | Path | Purpose | Auth |
|---|---|---|---|
| POST | `/v1/speak` | **Generate speech — all four modes** | key |
| GET | `/v1/voices` | Built-in voices, vocalist voices, languages | key |
| GET | `/health` | Status, VRAM, resolved config | open |

Plus `GET /demo` (a working test UI) and `GET /docs` (Swagger). Both open.

"key" means the `X-API-Key` header is required **only when `QWEN_TTS_API_KEY` is set** on
the server. It is unset locally, so no header is needed.

---

## How the mode is chosen

A request may carry **at most one voice source**. Which one you send picks the model:

| Voice source | Mode | Model | `instruct` |
|---|---|---|---|
| *(none)* | `described_voice` | VoiceDesign 1.7B | **required** — the voice is invented from it |
| `speaker` | `built_in_voice` | CustomVoice 1.7B | optional — restyles the voice |
| `voice_id` | `vocalist_clone` | Base 1.7B | ignored |
| `audio` | `voice_clone` | Base 1.7B | ignored |

```js
{ text: "Hello." , instruct: "An elderly wizard, gravelly." }  // described_voice
{ text: "Hello." , speaker: "vivian" }                         // built_in_voice
{ text: "Hello." , speaker: "vivian", instruct: "Angrily." }   // built_in_voice, restyled
{ text: "Hello." , voice_id: "aria_soul" }                     // vocalist_clone
{ text: "Hello." , audio: "<url | base64 | file upload>" }     // voice_clone
```

The response echoes `"mode"` (and an `X-Mode` header for `wav` responses) so you can
always confirm which path ran.

**Sending two voice sources is an error**, not a silent preference — it almost always
means a frontend bug. `{"speaker": "ryan", "audio": "..."}` returns `422`. Clear the
other field when the user switches source.

Sending neither a voice source nor `instruct` also returns `422` — there would be
nothing to determine the voice from.

---

## `POST /v1/speak`

Accepts **`application/json`** or **`multipart/form-data`**. Use multipart when the user
uploads a file; everything else is easier as JSON. Field names are identical in both.

### Fields

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `text` | string \| string[] | **yes** | — | What to say. A list generates a batch. |
| `speaker` | string \| string[] | no | — | Built-in voice id → `built_in_voice` |
| `voice_id` | string | no | — | Vocalist voice id → `vocalist_clone` |
| `audio` | string \| **file** | no | — | Reference clip → `voice_clone` |
| `instruct` | string \| string[] | conditional | — | Required when no voice source is given |
| `ref_text` | string | no | auto | Transcript of `audio`; transcribed if omitted |
| `language` | string \| string[] | no | `auto` | See `GET /v1/voices` → `languages` |
| `response_format` | `"json"` \| `"wav"` | no | `json` | |

Plus optional sampling knobs: `max_new_tokens`, `do_sample`, `top_k`, `top_p`,
`temperature`, `repetition_penalty`, `subtalker_dosample`, `subtalker_top_k`,
`subtalker_top_p`, `subtalker_temperature`.

### Responses

`response_format: "json"` (default):

```json
{
  "mode": "vocalist_clone",
  "sample_rate": 24000,
  "count": 1,
  "durations": [3.12],
  "generation_seconds": 8.4,
  "audio": ["<base64 wav>"],
  "voice_id": "aria_soul",
  "ref_text": "Every song starts somewhere quiet…",
  "ref_text_auto_generated": false
}
```

`response_format: "wav"` returns raw `audio/wav` bytes — feed it straight to an `<audio>`
element. Metadata moves to headers: **`X-Mode`**, and **`X-Reference-Text`** when cloning.
Both are in `expose_headers`, so browser JS can read them. Batches are rejected with `400`;
use `"json"` for those.

Audio is 24 kHz mono 16-bit PCM WAV throughout.

### Batching

`text` accepts a list. `speaker` / `instruct` / `language` then take either one value
(applied to every item) or a list of matching length; any other length returns `400`.

```json
{ "text": ["First line.", "Second line."], "speaker": ["ryan", "aiden"] }
```

Cloning batches share one reference clip — `voice_id` / `audio` stay single-valued.

Batching requires JSON. Multipart form fields cannot express lists.

---

## The four modes

### 1. `described_voice` — invent a voice from words

No voice source; `instruct` describes the voice.

```bash
curl -X POST http://127.0.0.1:9000/v1/speak \
  -H "Content-Type: application/json" \
  -d '{"text":"Welcome, traveller.","instruct":"An elderly wizard, deep gravelly voice."}'
```

> Each call re-invents the voice, so timbre drifts slightly between requests. That is
> inherent to this mode. When a user needs one voice that stays identical across many
> generations, add it to the vocalist library instead and use `voice_id`.

### 2. `built_in_voice` — the 9 shipped voices

```bash
curl -X POST http://127.0.0.1:9000/v1/speak \
  -H "Content-Type: application/json" \
  -d '{"text":"I cannot believe you did that.","speaker":"vivian","instruct":"Furious, sharp tone."}'
```

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

Any voice can speak any supported language, but sounds best in its native one. Build your
picker from `GET /v1/voices`, not a hardcoded copy.

### 3. `vocalist_clone` — your curated voice library

Voices **you** supply, cloned on demand. Users pick these the same way they pick a
built-in voice; the difference is invisible to them.

```bash
curl -X POST http://127.0.0.1:9000/v1/speak \
  -H "Content-Type: application/json" \
  -d '{"text":"Sing me something new.","voice_id":"aria_soul"}'
```

**Adding a voice:** drop an audio file into `voices/`. The filename stem becomes the
`voice_id`, and it appears in `GET /v1/voices` immediately — no restart. Optional metadata
(display name, description, exact transcript) goes in `voices/voices.json`.

Full details, clip guidance, and the manifest format: **[voices/README.md](voices/README.md)**.

A clip's transcript is needed for good cloning. If you do not supply one, the first request
transcribes it with Whisper and **caches the result**, so it is computed once per voice
rather than once per request.

An unknown `voice_id` returns `404` listing the ids that do exist.

### 4. `voice_clone` — a clip the user uploads

From a browser, post multipart with the file in the `audio` field:

```bash
curl -X POST http://127.0.0.1:9000/v1/speak \
  -F "text=This is a cloned voice." \
  -F "audio=@sample.wav"
```

As JSON, `audio` accepts a **URL**, a **server-local path**, **bare base64**, or a
**`data:audio/wav;base64,…` URI**:

```json
{ "text": "This is a cloned voice.", "audio": "https://example.com/sample.wav" }
```

`ref_text` is optional — omit it and the clip is transcribed automatically. Supplying the
exact transcript gives better results when you know it; Whisper misreads unusual names and
heavy accents.

> The upstream library decides "is this base64?" by checking the string contains no `/`,
> which standard base64 uses constantly — so bare base64 gets misread as a filename. This
> server rewraps it as a data URI, so all four forms work here. If you call `qwen_tts`
> directly elsewhere, send a data URI.

---

## Errors

| Status | Meaning |
|---|---|
| `400` | `response_format:"wav"` with a batch, a mismatched list length, or unreadable audio |
| `401` | Missing/incorrect `X-API-Key` (only when the server has a key configured) |
| `404` | Unknown `voice_id` — the message lists the available ids |
| `422` | Missing `text`; two voice sources; or no voice source **and** no `instruct` |
| `500` | Unknown `speaker`, or a generation failure |
| `503` | A required model is not downloaded on the server |

`422` bodies use FastAPI's shape — a `detail` array with a `msg` per problem.

### Warnings

Non-fatal problems come back in a `warnings` array rather than failing the request. Sending
`instruct` while cloning is the common one:

```json
{ "mode": "voice_clone", "audio": ["…"],
  "warnings": ["`instruct` is not supported when cloning a voice and was ignored."] }
```

The Base model has no instruction channel — cloning copies the delivery in the reference
clip. To change delivery, change the reference clip.

---

## `GET /v1/voices`

Everything the picker needs, in one call:

```json
{
  "built_in": [
    { "id": "ryan", "name": "Ryan", "description": "Dynamic male voice…",
      "native_language": "English" }
  ],
  "vocalist": [
    { "id": "aria_soul", "name": "Aria", "description": "Soulful female vocalist…",
      "language": "english", "file": "aria_soul.wav", "has_transcript": true }
  ],
  "languages": ["auto", "chinese", "english", "…"]
}
```

Render `built_in` and `vocalist` as two groups in one dropdown. Send the chosen id as
`speaker` for a built-in or `voice_id` for a vocalist — that single difference is all your
frontend needs to track.

---

## `GET /health`

Open, no key required, so load balancers and RunPod probes work.

```json
{
  "status": "ok",
  "cuda": true,
  "gpu": { "name": "NVIDIA A100-SXM4-80GB", "allocated_gb": 14.2, "total_gb": 79.2 },
  "resident_models": ["voice_design", "custom_voice", "base"],
  "asr_loaded": true,
  "vocalist_voices": 2,
  "config": {
    "device": "cuda:0", "asr_device": "cuda:0", "max_resident": 3,
    "attn_implementation": "flash_attention_2", "auth": "api_key"
  },
  "models_downloaded": { "voice_design": true, "custom_voice": true, "base": true }
}
```

`config` shows what the server auto-detected at startup — **check `device` and `asr_device`
are `cuda:0` to confirm you are running on the GPU.**

---

## Calling it from your website

One function covers every mode, because there is one endpoint:

```js
const API = "https://<POD_ID>-9000.proxy.runpod.net";

async function speak({ text, voice, instruct, file, refText, language = "auto" }) {
  let res;

  if (file) {
    // Uploaded clip -> multipart. Everything else is JSON.
    const fd = new FormData();
    fd.append("text", text);
    fd.append("audio", file);
    fd.append("language", language);
    fd.append("response_format", "wav");
    if (refText) fd.append("ref_text", refText);
    res = await fetch(`${API}/v1/speak`, { method: "POST", headers: authHeader(), body: fd });
  } else {
    const body = { text, language, response_format: "wav" };
    // `voice` came from the picker as {kind, id}, matching GET /v1/voices.
    if (voice?.kind === "built_in") { body.speaker = voice.id; if (instruct) body.instruct = instruct; }
    else if (voice?.kind === "vocalist") { body.voice_id = voice.id; }
    else { body.instruct = instruct; }          // no voice picked -> describe one
    res = await fetch(`${API}/v1/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader() },
      body: JSON.stringify(body),
    });
  }

  if (!res.ok) throw new Error(await res.text());
  return { blob: await res.blob(), mode: res.headers.get("X-Mode") };
}
```

Play it with `new Audio(URL.createObjectURL(blob)).play()`.

⚠️ **Never ship the API key to the browser** — proxy these calls through your own backend,
which holds the key. See [DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md).

A complete working implementation, including the mode-preview UI, is at `/demo` —
source in [server/static/demo.html](server/static/demo.html).

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
.\venv\Scripts\python.exe test_endpoints.py --filter vocalist
```

- Interactive docs: <http://127.0.0.1:9000/docs>
- Demo UI: <http://127.0.0.1:9000/demo>

**Launch with the venv Python**, not the IDE Run button — that uses the system interpreter,
which lacks these packages, and a second server fighting for port 9000 causes confusing
failures. Clear strays first:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*app.py*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

On an 8 GB GPU, use [run_family_tests.ps1](run_family_tests.ps1) instead of the plain suite —
it runs one mode per server process. See "Memory" below for why.

---

## Configuration

Every value auto-detects from the hardware; the environment variables are overrides.
**No code edits are needed to deploy.**

| Env var | Auto-detected default | Purpose |
|---|---|---|
| `QWEN_TTS_MODEL_DIR` | `<repo>/models` | Model folders (repo-relative, so no edit needed) |
| `QWEN_TTS_VOICE_DIR` | `<repo>/voices` | Vocalist clips + `voices.json` |
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
| VoiceDesign 1.7B | 4.2 GB | `described_voice` |
| CustomVoice 1.7B | 4.2 GB | `built_in_voice` |
| Base 1.7B | 4.2 GB | `vocalist_clone`, `voice_clone` |
| Whisper large-v3-turbo | 1.6 GB | automatic `ref_text` |

**A100 80 GB** — all four resident (14.2 GB), Whisper on GPU, nothing ever reloads.

**8 GB laptop** — one TTS model at a time, Whisper on CPU. Switching modes evicts and
reloads a 4.2 GB model (10–30 s), and repeated eviction cycles can crash the CUDA process
outright on Windows. That is why `run_family_tests.ps1` exists. Neither limitation applies
on the A100, where nothing is ever evicted.

---

## Concurrency

GPU work is serialized behind a single lock: one generation at a time, the rest queue,
because the models are not safe to call concurrently across threads. To raise throughput on
the A100, run several server processes on different ports behind a load balancer — see
[DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md). Do not use `uvicorn --workers N`.

[test_concurrency.py](test_concurrency.py) fires overlapping requests across modes to verify
the locking holds.

---

## Not exposed here

An earlier, broader API — reusable saved clone prompts by `voice_id`, design-then-clone, and
codec encode/decode — is preserved in [server/app_full.py](server/app_full.py) for reference.
