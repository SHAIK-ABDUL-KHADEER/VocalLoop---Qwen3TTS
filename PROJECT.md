# VocalLoop — how the whole thing works

An end-to-end explanation of the project: what it is, what every file does, how a request
flows through it, and why it is built this way.

For task-specific docs see [ENDPOINTS.md](ENDPOINTS.md) (API reference),
[CURL_EXAMPLES.md](CURL_EXAMPLES.md) (runnable examples) and
[DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md) (deployment).

---

## 1. What it is

An HTTP API wrapping [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B so a website can
generate speech. Four capabilities behind **one endpoint**:

| The user… | Mode | Model used |
|---|---|---|
| describes a voice in words | `described_voice` | VoiceDesign 1.7B |
| picks one of 9 built-in voices | `built_in_voice` | CustomVoice 1.7B |
| picks a vocalist from your library | `vocalist_clone` | Base 1.7B |
| uploads their own clip | `voice_clone` | Base 1.7B |

Everything else — transcription, batching, model loading — happens internally.

### Design decisions

**One endpoint, not four.** The frontend has a single call site. Which fields are present
selects the model, so the browser never has to know which model exists for which feature.

**Server picks the model, not the client.** A client choosing models would hard-code
knowledge that belongs on the server, and break whenever the model lineup changed.

**Batching over parallelism.** Concurrent requests merge into one model pass rather than
running side by side. See §5 — it is the single biggest performance decision here.

**Auto-tuning over configuration.** Device, resident-model count, ASR placement, batch size
and attention kernel are all detected from the hardware at startup, so the same file runs
unedited on an 8 GB laptop and an 80 GB A100.

---

## 2. Repository layout

```
VocalLoop---Qwen3TTS/
├── server/
│   ├── app.py              the entire API (~1200 lines)
│   ├── app_full.py         earlier multi-endpoint design, kept for reference
│   └── static/demo.html    self-contained test UI served at /demo
├── Qwen3-TTS/              vendored upstream library (unmodified)
├── voices/                 vocalist clip library + voices.json manifest
├── models/                 model weights — NOT in git, ~14.2 GB, downloaded separately
├── tools/
│   └── make_sample_vocalists.py   regenerates the placeholder vocalist clips
├── test_endpoints.py       28-check end-to-end suite
├── test_concurrency.py     overlapping-request probe, reports batch metrics
├── run_family_tests.ps1    one mode per server process (small-GPU workaround)
├── bench.py                raw batch-scaling benchmark
├── smoke_test.py           library-level check, no server involved
└── *.md                    documentation
```

**`server/app.py` is the whole application.** No framework scaffolding, no package layout —
one file you can read top to bottom. Its sections in order: config, vocalist library, model
registry, helpers, request schema, batching engine, the endpoint, metadata endpoints.

---

## 3. The models

Four models, ~14.2 GB total, all downloaded from Hugging Face (see
[DEPLOY_RUNPOD.md §3](DEPLOY_RUNPOD.md)):

| Model | Size | Serves |
|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 4.2 GB | `described_voice` |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 4.2 GB | `built_in_voice` (9 speakers) |
| `Qwen3-TTS-12Hz-1.7B-Base` | 4.2 GB | both cloning modes |
| `whisper-large-v3-turbo` | 1.6 GB | transcribing reference clips |

1.7B is the largest Qwen3-TTS released; the 0.6B variants are deliberately not exposed.

### How the TTS model actually works

Speech is represented as **discrete tokens at 12 Hz** — 12 tokens per second of audio. A
transformer ("talker") predicts these autoregressively, and a codec decodes them to a 24 kHz
waveform.

Two consequences that shape everything downstream:

**Sequences are short.** Thirty seconds of speech is 360 tokens, so the KV cache is tiny —
about 112 KB per token, ~78 MB for a minute of audio. Memory is never the constraint;
compute is.

**Every frame costs 17 forward passes.** One talker step plus 16 sequential code-predictor
steps (`num_code_groups: 16`). That is a lot of small GPU launches per frame, so the workload
is bound by per-step overhead more than raw FLOPs — which is exactly why batching wins so
much (§5).

---

## 4. Request lifecycle

What happens between the browser and the audio, in order.

```
1. HTTP request arrives          POST /v1/speak, JSON or multipart
2. Auth                          X-API-Key checked, unless the path is public
3. Parse                         one body parser for both content types
4. Validate                      exactly one voice source; text/instruct rules
5. Guardrails                    item count, text length, queue depth
6. Reference audio (clone only)  fetch once, transcribe if no ref_text
7. Enqueue                       a Job with an asyncio Future; the handler awaits
8. Batch worker                  collect, group, generate, hand results back
9. Respond                       raw WAV bytes, or JSON with base64
```

### Step 3 — one parser, two content types

`/v1/speak` reads the raw `Request` rather than declaring a Pydantic body, because it must
accept **both** JSON (normal calls, batches) and multipart (browser file upload) on one path.
`parse_speak_request()` branches on `Content-Type`, builds a plain dict, and validates it
through the same `SpeakRequest` model either way — so both paths share one set of rules.

The cost is that FastAPI cannot infer a schema for `/docs`, so the OpenAPI body is supplied
by hand in `SPEAK_OPENAPI` with seven runnable examples.

> A real bug lived here: `request.form()` returns **Starlette's** `UploadFile`, and
> `fastapi.UploadFile` is a *subclass* — so `isinstance(value, fastapi.UploadFile)` silently
> missed every upload and handed Pydantic a file object where a string belonged. Multipart
> was completely broken until that import changed.

### Step 4 — mode selection

`SpeakRequest` has a `model_validator` enforcing **at most one** voice source
(`speaker` / `voice_id` / `audio`), and requiring `instruct` when there is none. Two sources
is a `422`, not a silent precedence rule, because that combination almost always means a
frontend bug — silently preferring one would hide it.

`mode` is then a pure function of which field is set.

### Step 6 — reference audio, resolved once

For the cloning modes the clip is fetched **once** into a numpy waveform and that waveform
is handed to both transcription and generation.

> Originally the URL was passed straight to the library, which downloaded it again for
> feature extraction — so a URL was fetched twice, and a network failure inside the library
> surfaced as an opaque `500`. Now it is fetched once here, and failures become `502` with
> the URL and cause.

Transcription runs only when `ref_text` is absent. For vocalist voices the result is
**written back into `voices/voices.json`**, so each clip is transcribed once ever rather than
once per request.

### Steps 7–8 — the queue

The handler does not generate. It creates a `Job`, puts it on an `asyncio.Queue`, and awaits
a Future. A single background worker owns all GPU work. Details in §5.

---

## 5. Batching — the core of the performance story

### Why it works

Generating one audio frame requires reading **all 3.4 GB of model weights** out of VRAM.
That read costs the same whether the GPU is producing for one caller or twenty.

```
1 request  : read 3.4 GB → 1 frame      3.4 GB per frame
20 batched : read 3.4 GB → 20 frames    0.17 GB per frame
```

Serving twenty people costs barely more than serving one. Serving them one at a time pays
full price twenty times.

### Measured on an 8 GB laptop (RTX 4060), warm

| | Serial | Batched |
|---|---|---|
| 4 concurrent requests | 58.3 s | **14.5 s** (4.02×) |
| 8 simultaneous users | ~80 s | **20.4 s**, 1.40× real-time |

Queue wait p95 was 0.03 s — the batching window costs nothing next to generation.

### How it is implemented

```python
# handler                              # worker (one, for the whole process)
job = Job(..., future)                 first = await job_queue.get()
await job_queue.put(job)               ... collect more until 40ms or MAX_BATCH ...
return await job.future                groups = group_by(model, sampling_kwargs)
                                       for g in groups: run in gpu_executor
                                       future.set_result(...) per job
```

**The collection window** (`QWEN_TTS_BATCH_WINDOW_MS`, default 40 ms) is how long the first
request waits for company. It only matters when idle — under load the queue is already full
when a batch finishes, so batches form for free.

**Grouping is by model *and* sampling parameters.** `generate()` takes `temperature`, `top_p`
and friends per *call*, not per item, so two requests with different sampling settings cannot
share a pass. Requests using defaults — nearly all of them — batch together.

**Failure isolation.** If a batch raises, each job is retried alone
(`run_group_isolated`). One malformed request cannot fail the others batched with it.

**Resident-model-first ordering.** When several groups are pending, the worker runs whichever
model is already loaded first. On a GPU too small to hold all three, this avoids an eviction
and 10–30 s reload in the middle of serving.

**A dedicated thread.** Generation runs on its own single-thread executor, not the shared
default pool — reference downloads and transcription also run in threads, and if they filled
the pool the worker could be left with nowhere to run.

### What batching does *not* do

Requests still wait for the batch they are in to finish completely; there is no continuous
batching (joining a pass already underway) and no streaming. vLLM-Omni does both properly,
and is worth revisiting if its online serving for Qwen3-TTS has shipped by the time you need
that level of scale.

---

## 6. Model residency

Each model is 4.2 GB and loading takes 10–30 s, so `ModelRegistry` keeps them in memory with
LRU eviction, sized from the GPU:

| VRAM | `max_resident` | Behaviour |
|---|---|---|
| ≥ 24 GB | 3 | all models stay loaded, nothing ever evicts |
| ≥ 16 GB | 2 | occasional eviction |
| < 16 GB | 1 | switching modes evicts and reloads |

**On the A100 nothing is ever evicted** — 14.2 GB of 80 GB, all resident.

Two hard-won details live in this class:

> **Eviction must move weights off the GPU, not just drop the reference.**
> `torch.cuda.empty_cache()` only returns *cached* blocks. An early version unpacked the
> evicted model into a throwaway variable that stayed alive during the next load, and VRAM
> climbed to 14.5 GB on an 8 GB card — Windows silently spills into shared system memory, so
> it grew instead of failing fast.

> **`registry.get()` must be called while holding `gpu_lock`.** Eviction frees another
> model's CUDA storage; if a second request could evict between a `get()` and its
> `generate()`, it would pull weights out from under a running call. That surfaces as
> `index is on cpu, different from other tensors on cuda:0`.

Related: **Whisper defaults to the CPU below 16 GB VRAM.** Sharing 8 GB with a 4.2 GB TTS
model left so little headroom that the next model loaded with tensors stranded on the CPU
and generation died — an intermittent failure that looked like a code bug for a long time.

---

## 7. The vocalist voice library

Curated clips users can pick and clone, distinct from the 9 built-in speakers.

**Adding a voice is dropping a file in `voices/`.** The filename stem becomes the `voice_id`,
and it appears in `GET /v1/voices` immediately — no restart, no registration call.

`voices/voices.json` is optional metadata (display name, description, exact transcript). A
clip's transcript is generated by Whisper on first use and cached back into that file, so it
costs one transcription per voice rather than one per request.

The two clips in the repo are synthesized placeholders so it works out of the box — replace
them. See [voices/README.md](voices/README.md) for clip guidance.

---

## 8. Safety and limits

| Guardrail | Default | Env var | Response |
|---|---|---|---|
| Items per request | 32 | `QWEN_TTS_MAX_ITEMS` | `400` |
| Characters per item | 3000 | `QWEN_TTS_MAX_CHARS` | `400` |
| Queued items server-wide | 128 | `QWEN_TTS_MAX_QUEUE` | `503` + `Retry-After` |
| Time queued + generating | 600 s | `QWEN_TTS_TIMEOUT` | `504` |

These exist because one caller pasting a novel would otherwise hold the GPU for minutes with
no way to shed load. Splitting long content into several requests is also *faster* for that
caller, because other work interleaves instead of blocking behind one giant job.

**Auth** is an optional `X-API-Key` header, enabled by setting `QWEN_TTS_API_KEY`.
`/health`, `/demo`, `/docs`, `/redoc` and `/openapi.json` stay open so probes and browsers
keep working. **CORS** origins come from `QWEN_TTS_CORS_ORIGINS`.

---

## 9. Observability

`GET /health` answers most operational questions in one call:

```json
{
  "status": "ok",
  "cuda": true,
  "gpu": { "name": "NVIDIA A100-SXM4-80GB", "allocated_gb": 14.2, "total_gb": 79.2 },
  "resident_models": ["voice_design", "custom_voice", "base"],
  "asr_loaded": true,
  "vocalist_voices": 2,
  "load": {
    "generating_items": 4, "queued_items": 7,
    "queue_wait_p50_seconds": 1.2, "queue_wait_p95_seconds": 8.4,
    "avg_batch_size": 5.3, "max_batch_size_seen": 16,
    "completed": 1204, "failed": 2, "rejected_overload": 0
  },
  "config": { "device": "cuda:0", "asr_device": "cuda:0", "max_resident": 3, "…": "…" },
  "models_downloaded": { "voice_design": true, "custom_voice": true, "base": true }
}
```

- **`config`** — what auto-detection chose. Check `device` and `asr_device` are `cuda:0`.
- **`load`** — the scaling signal. `queued_items` near zero and `avg_batch_size` near 1 means
  spare capacity. A climbing `queue_wait_p95_seconds` means you need a bigger batch size or
  more GPU.
- **`models_downloaded`** — catches a half-finished download before it becomes a `503`.

---

## 10. Testing

| Script | What it proves |
|---|---|
| `test_endpoints.py` | All 28 checks: every mode, batching, all four audio input forms, every error path |
| `test_concurrency.py` | Overlapping requests across modes; reports batch metrics |
| `run_family_tests.ps1` | Runs one mode per server process — needed on small GPUs |
| `bench.py` | Raw batch-scaling of the model, no server |
| `smoke_test.py` | The library works at all, independent of the API |

```bash
python test_endpoints.py                     # everything
python test_endpoints.py --filter vocalist   # one family
python test_concurrency.py --workers 8       # what 8 simultaneous users feel
```

**On an 8 GB GPU use `run_family_tests.ps1`, not the plain suite.** The suite jumps between
modes, and with one resident model that means repeated 4 GB evictions — which can kill the
CUDA process outright on Windows. Not an issue on the A100, where nothing evicts.

---

## 11. Known limitations

**No streaming.** Responses are complete files. The model supports streaming generation; the
API does not expose it. First-audio latency would improve considerably.

**No continuous batching.** A request waits for the whole batch it joined. vLLM-Omni does
this properly.

**One process owns the GPU.** Deliberate — see [DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md). Multiple
processes would each load 14.2 GB and split traffic into smaller, less efficient batches.

**Long jobs block interactive ones.** A 32-item request occupies the GPU for its duration. A
proper job queue (submit → poll) would be the honest fix if long-form becomes a primary use
case.

**Vocalist transcripts need a writable `voices/`.** On a read-only volume caching fails
(logged, non-fatal) and clips are re-transcribed each time.

---

## 12. Working on the code

`server/app.py` reads top to bottom: config → voice library → registry → helpers → schema →
batching → endpoints. Adding a mode means extending `SpeakRequest`, its validator, and the
`model_kind` mapping in the handler.

The vendored library in `Qwen3-TTS/` is **never patched** — every workaround for its rough
edges lives in `app.py`, so the folder can be swapped for a newer upstream checkout. Two
workarounds exist today, both documented at their call sites: base64 audio detection
(`normalize_audio_ref`) and fetching reference audio ourselves (`load_audio_np`).

After any change: `python test_endpoints.py` (or `run_family_tests.ps1` on a small GPU).
