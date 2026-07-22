# Deploying to RunPod (no Docker)

Target: an **A100 80 GB** pod serving the API to your website.

Every step below is a command you can paste. Roughly **45–60 minutes** end to end, most of it
waiting for the model download and the optional FlashAttention build.

---

## What you must change in the code

**Nothing.**

Everything that differs between your laptop and the A100 is either auto-detected from the
hardware or set with an environment variable. You deploy [server/app.py](server/app.py)
exactly as it is in git.

| Concern | How it is handled |
|---|---|
| GPU vs CPU | Auto-detected |
| How many models stay in VRAM | Auto — 3 on a ≥24 GB card |
| Where Whisper runs | Auto — GPU on a ≥16 GB card |
| Attention kernel | Auto — FlashAttention 2 if installed |
| Batch size | Auto — 16 on a ≥24 GB card |
| Model / voice paths | Repo-relative by default |
| API key, CORS, host, port | Environment variables (§4) |

Verify what it chose with `GET /health` → `config` (§6).

The **only** edits you might ever make are optional behaviour changes — a different queue
limit, a different timeout — and even those are environment variables first. If you find
yourself editing Python to deploy, something has gone wrong; check §4 instead.

---

## 1. Create the pod

RunPod → **Pods** → **Deploy**.

| Setting | Value |
|---|---|
| GPU | A100 80 GB (PCIe or SXM) |
| Template | **RunPod PyTorch** (any recent CUDA 12.x image) |
| Container disk | 30 GB |
| Volume disk | **60 GB**, mounted at `/workspace` |
| Expose HTTP ports | `9000` |

Put everything on the **volume** (`/workspace`). The container disk is wiped when the pod is
rebuilt, and you do not want to re-download 14 GB each time.

Once it is running, open a **Web Terminal** or connect over SSH (the pod's Connect page shows
the exact command).

---

## 2. Install

Clone this repo. It contains both the API and a vendored copy of the Qwen3-TTS library, so
nothing else needs fetching:

```bash
cd /workspace
git clone https://github.com/SHAIK-ABDUL-KHADEER/VocalLoop---Qwen3TTS.git vocalloop
cd /workspace/vocalloop

python -m venv venv
source venv/bin/activate
pip install -U pip
```

**PyTorch** — the RunPod PyTorch template usually ships it. Check first:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If that prints a CUDA build and `True`, skip ahead. Otherwise:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**The TTS engine and API dependencies:**

```bash
pip install -e ./Qwen3-TTS
pip install fastapi "uvicorn[standard]" python-multipart requests
```

> `pip install -e ./Qwen3-TTS` installs the vendored library, which pulls in transformers,
> accelerate, librosa, soundfile and onnxruntime. It is identical to `qwen-tts` 0.1.1 on
> PyPI — `pip install -U qwen-tts` is an equivalent substitute. See
> [Qwen3-TTS/VENDORED.md](Qwen3-TTS/VENDORED.md).

**FlashAttention 2** (optional, recommended). The server picks it up automatically:

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

This compiles from source — **10–30 minutes**. If it fails, skip it; the server falls back to
`sdpa` on its own. Confirm afterwards via `/health` → `config.attn_implementation`.

---

## 3. Download the models

```bash
cd /workspace/vocalloop
source venv/bin/activate
pip install -U "huggingface_hub[cli]"

export MODELS=/workspace/vocalloop/models
hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir $MODELS/Qwen3-TTS-12Hz-1.7B-VoiceDesign
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir $MODELS/Qwen3-TTS-12Hz-1.7B-CustomVoice
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base        --local-dir $MODELS/Qwen3-TTS-12Hz-1.7B-Base
hf download openai/whisper-large-v3-turbo        --local-dir $MODELS/whisper-large-v3-turbo
```

**~14.2 GB total.** The folder names must match exactly — the server looks for them by name
in `<repo>/models`, which is where the commands above put them. No configuration needed.

Verify:

```bash
du -sh $MODELS/*
```

You should see four directories, roughly 4.2 / 4.2 / 4.2 / 1.6 GB.

### Your vocalist voices

Copy your reference clips into `/workspace/vocalloop/voices/`. The filename stem becomes the
`voice_id`, and they are picked up without a restart. Delete the two placeholder clips
(`aria_soul.wav`, `marcus_baritone.wav`) once you have real ones. See
[voices/README.md](voices/README.md).

---

## 4. Configure

Create `/workspace/vocalloop/.env`:

```bash
# --- required for a public deployment ---------------------------------------
# Bind publicly so RunPod's proxy can reach it
QWEN_TTS_HOST=0.0.0.0
QWEN_TTS_PORT=9000

# Require this header on every generation request. Generate one with:
#   python -c "import secrets; print(secrets.token_urlsafe(32))"
QWEN_TTS_API_KEY=<paste-a-long-random-string>

# Only your site may call the API from a browser
QWEN_TTS_CORS_ORIGINS=https://yoursite.com,https://www.yoursite.com

# --- recommended -------------------------------------------------------------
# Load all models at boot so the first real user does not wait 10-30s
QWEN_TTS_WARMUP=1
```

Everything else auto-detects. **Do not set `QWEN_TTS_MAX_RESIDENT`, `QWEN_TTS_ASR_DEVICE`,
`QWEN_TTS_DEVICE` or `QWEN_TTS_MAX_BATCH` on the A100** — the defaults are already correct
there, and forcing them is a common way to make things worse.

### Full environment variable reference

| Variable | Default on A100 | Purpose |
|---|---|---|
| `QWEN_TTS_HOST` / `QWEN_TTS_PORT` | `127.0.0.1` / `9000` | Bind address. **Must set host to `0.0.0.0`** |
| `QWEN_TTS_API_KEY` | unset → open | Require this in an `X-API-Key` header |
| `QWEN_TTS_CORS_ORIGINS` | `*` | Comma-separated allowed browser origins |
| `QWEN_TTS_WARMUP` | off | `1` preloads every model at boot |
| `QWEN_TTS_MODEL_DIR` | `<repo>/models` | Where the model folders live |
| `QWEN_TTS_VOICE_DIR` | `<repo>/voices` | Vocalist clips + `voices.json` |
| `QWEN_TTS_DEVICE` | `cuda:0` | Compute device |
| `QWEN_TTS_MAX_RESIDENT` | `3` | Models kept in VRAM |
| `QWEN_TTS_ASR_DEVICE` | `cuda:0` | Where Whisper runs |
| `QWEN_TTS_ATTN` | `flash_attention_2` if installed | Attention kernel |
| `QWEN_TTS_MAX_BATCH` | `16` | Requests merged into one model pass |
| `QWEN_TTS_BATCH_WINDOW_MS` | `40` | How long an idle request waits for company |
| `QWEN_TTS_MAX_ITEMS` | `32` | Max `text` items per request → `400` |
| `QWEN_TTS_MAX_CHARS` | `3000` | Max characters per item → `400` |
| `QWEN_TTS_MAX_QUEUE` | `128` | Queued items before `503` |
| `QWEN_TTS_TIMEOUT` | `600` | Seconds before `504` |

### What the A100 auto-selects

| Setting | Value | Why |
|---|---|---|
| `device` | `cuda:0` | GPU present |
| `max_resident` | `3` | ≥24 GB, so all three TTS models stay loaded |
| `asr_device` | `cuda:0` | ≥16 GB, so Whisper runs on GPU (~10× faster) |
| `attn_implementation` | `flash_attention_2` | if flash-attn installed |
| `max_batch_items` | `16` | ≥24 GB, so 16 requests merge per pass |

**Total resident: 14.2 GB of 80 GB.** Nothing ever evicts or reloads.

> On the 8 GB laptop the server keeps one model resident and pushes Whisper to the CPU,
> because there is not room for more — and repeated eviction of a 4 GB model proved genuinely
> unstable there. None of that applies on the A100.

---

## 5. Run it

```bash
cd /workspace/vocalloop
source venv/bin/activate
set -a && source .env && set +a
python -m uvicorn server.app:app --host 0.0.0.0 --port 9000
```

The startup lines print the resolved configuration:

```
[startup] gpu          : NVIDIA A100-SXM4-80GB (79.2 GB)
[startup] device       : cuda:0
[startup] model dir    : /workspace/vocalloop/models
[startup] voice dir    : /workspace/vocalloop/voices (2 vocalist voices)
[startup] max resident : 3
[startup] attn impl    : flash_attention_2
[startup] asr model    : .../whisper-large-v3-turbo on cuda:0
[startup] auth         : X-API-Key required
[startup] cors origins : ['https://yoursite.com']
[startup] batching     : up to 16 items, 40ms window
[startup] limits       : 32 items/req, 3000 chars/item, queue 128, timeout 600s
[startup] warmed up    : ['voice_design', 'custom_voice', 'base']
```

**If `device` says `cpu`, or `asr model … on cpu`, stop and fix it** — you are not on the GPU
and everything will be unusably slow.

### Keeping it running

A RunPod pod is a container and **has no systemd** — `systemctl` fails with "System has not
been booted with systemd as init system". Use one of these.

**tmux**, for interactive work:

```bash
tmux new -s tts
# start the server, detach with Ctrl-B then D, return with: tmux attach -t tts
```

**A supervised background process**, to survive disconnects and restart on crash. Save as
`/workspace/vocalloop/run.sh`:

```bash
#!/usr/bin/env bash
cd /workspace/vocalloop
source venv/bin/activate
set -a && source .env && set +a
while true; do
  python -m uvicorn server.app:app --host 0.0.0.0 --port 9000
  echo "server exited ($?), restarting in 5s" >&2
  sleep 5
done
```

```bash
chmod +x /workspace/vocalloop/run.sh
nohup /workspace/vocalloop/run.sh > /workspace/vocalloop/server.log 2>&1 &
tail -f /workspace/vocalloop/server.log
```

The restart loop matters: a CUDA OOM can kill the process outright with no traceback, and
without it the pod keeps billing while serving nothing.

**Start automatically on pod boot.** Nothing under `/workspace` runs on its own — put this in
the pod's *Container Start Command* (Edit Pod → Docker Command):

```bash
bash -c "/workspace/vocalloop/run.sh"
```

---

## 6. Verify

Your public URL is `https://<POD_ID>-9000.proxy.runpod.net`, shown on the pod's Connect page.

**Health first** — `/health` needs no API key, so probes work:

```bash
curl https://<POD_ID>-9000.proxy.runpod.net/health
```

Confirm `config.device` and `config.asr_device` are both `cuda:0`, and
`models_downloaded` is all `true`.

**Generate something:**

```bash
curl -X POST https://<POD_ID>-9000.proxy.runpod.net/v1/speak \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $QWEN_TTS_API_KEY" \
  -d '{"text":"Deployment successful.","speaker":"ryan","response_format":"wav"}' \
  --output test.wav
```

**Run the full suite** (28 checks). Easiest from inside the pod, where `.env` supplies the key:

```bash
cd /workspace/vocalloop && source venv/bin/activate
set -a && source .env && set +a
python test_endpoints.py
```

From your laptop, pass the key and URL:

```bash
python test_endpoints.py --base-url https://<POD_ID>-9000.proxy.runpod.net --api-key "<key>"
```

All 28 should pass. Expect 5–10 minutes over the public proxy.

**Check concurrency** — what 8 simultaneous users actually experience:

```bash
python test_concurrency.py --workers 8 --rounds 2
```

It reports `avg_batch_size` at the end. Anything above 1 means batching is engaging.

**Record the ceiling** (optional, ~10 minutes) so future capacity planning uses real numbers:

```bash
python bench.py
```

---

## 7. Point your website at it

One endpoint, `POST /v1/speak`; the fields you send select the mode:

```js
const API = "https://<POD_ID>-9000.proxy.runpod.net";
const KEY = process.env.QWEN_TTS_API_KEY;   // server-side only

const res = await fetch(`${API}/v1/speak`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": KEY },
  body: JSON.stringify({
    text,
    // exactly one of these three, or none of them plus `instruct`:
    speaker,        // built-in voice   -> built_in_voice
    // voice_id,    // vocalist library -> vocalist_clone
    // audio,       // url/base64 clip  -> voice_clone
    instruct,       // required only when no voice source is given
    language: "auto",
    response_format: "wav",
  }),
});
```

For a user-uploaded file, post `multipart/form-data` to the same URL with the clip in an
`audio` field. Populate your voice picker from `GET /v1/voices`, which returns built-in
voices, vocalist voices and languages in one call.

⚠️ **Never put the API key in browser JavaScript.** Anyone could read it and spend your GPU
time. Proxy these calls through your own backend, which holds the key. If you must call
directly from the browser, leave `QWEN_TTS_API_KEY` unset and restrict
`QWEN_TTS_CORS_ORIGINS` to your domain — understanding that CORS is not access control, only
a browser convention.

Full reference: [ENDPOINTS.md](ENDPOINTS.md). Runnable examples:
[CURL_EXAMPLES.md](CURL_EXAMPLES.md).

---

## 8. Operating it

**Run ONE process.** The server batches concurrent requests into a single model pass, so one
process uses the GPU far better than several competing ones. Measured on an 8 GB laptop:
4 concurrent requests took 14.5 s batched versus 58.3 s serial (**4×**), and 8 simultaneous
users were all served in 20.4 s.

Several processes would be *worse*: each loads its own 14.2 GB (so only ~5 fit), and they
split traffic into smaller batches while contending for the same GPU. Do **not** use
`uvicorn --workers N` either — same problem, plus VRAM pressure.

**Scale by raising `QWEN_TTS_MAX_BATCH` first**, then by adding a second GPU. The signal:

```bash
curl -s https://<POD_ID>-9000.proxy.runpod.net/health | python -m json.tool
```

```json
"load": { "queued_items": 7, "queue_wait_p95_seconds": 8.4, "avg_batch_size": 5.3,
          "completed": 1204, "failed": 2, "rejected_overload": 0 }
```

- `queued_items` near zero → spare capacity.
- `queue_wait_p95_seconds` climbing → raise the batch size, or add GPU.
- `rejected_overload` above zero → users are getting `503`; raise `QWEN_TTS_MAX_QUEUE` only
  if the GPU can actually keep up, otherwise you are just hiding the queue.

**Cost.** A pod bills while running, whether or not it is serving. Stop it when idle. If
traffic is bursty, RunPod Serverless suits the workload better — but that route wants a
container image, which is out of scope here.

**Storage.** Keep `/workspace` on a network volume so the 14.2 GB of weights and your
vocalist clips survive pod recreation.

**Updating.** `git pull` in `/workspace/vocalloop`, then restart the server. Models and
voices live outside git, so they are untouched.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `config.device` is `cpu` | No CUDA in the venv. Reinstall torch with the right `--index-url`. |
| `config.asr_device` is `cpu` on the A100 | `QWEN_TTS_ASR_DEVICE=cpu` left in `.env`. Remove it. |
| `503 Model 'x' is not downloaded` | Wrong `QWEN_TTS_MODEL_DIR`, or a download did not finish. Folder names must match §3 exactly. |
| `503 Server is at capacity` | Queue is full — the GPU cannot keep up with arrivals. Expected under overload; raise `QWEN_TTS_MAX_BATCH` or add GPU. |
| `504 Generation exceeded 600s` | A very large request. Split the text, or raise `QWEN_TTS_TIMEOUT`. |
| `400 … exceeds the limit of 32` | More than `QWEN_TTS_MAX_ITEMS` items. Split into several requests — also faster. |
| `502 Could not fetch reference audio` | The server could not download the URL in `audio`. Check outbound access, or upload the clip instead. |
| `404 Unknown voice_id` | Clip not in `QWEN_TTS_VOICE_DIR`. The id is the filename without extension; the error lists valid ids. |
| `401 Invalid or missing X-API-Key` | Send the header, or unset `QWEN_TTS_API_KEY`. |
| `422` two voice sources | Sent more than one of `speaker` / `voice_id` / `audio`. Clear the others when the user switches. |
| `index is on cpu, different from other tensors on cuda:0` | Not enough free VRAM, so a model loaded split across devices. Should not happen on an A100 — check `nvidia-smi` for other processes. |
| `systemctl: command not found` | Expected — a pod is a container. Use `run.sh` + `nohup`, or tmux (§5). |
| `ModuleNotFoundError: qwen_tts` | venv not active, or `pip install -e ./Qwen3-TTS` skipped. |
| Server dies with no traceback | Almost always OOM. Check `nvidia-smi`; make sure only one server process is running. |
| Port not reachable | `9000` must be in the pod's exposed HTTP ports, and the server bound to `0.0.0.0`. |
| CORS errors in the browser | Set `QWEN_TTS_CORS_ORIGINS` to your exact origin, scheme included. |

`GET /health` answers most of these at once — device, ASR placement, which models are loaded,
which are downloaded, queue state, and whether auth is on.

---

## 10. Pre-launch checklist

- [ ] `/health` shows `device: cuda:0` and `asr_device: cuda:0`
- [ ] `models_downloaded` all `true`
- [ ] `QWEN_TTS_API_KEY` set, and a request without it returns `401`
- [ ] `QWEN_TTS_CORS_ORIGINS` restricted to your domain, not `*`
- [ ] API key held server-side only, never in browser JavaScript
- [ ] `test_endpoints.py` — 28/28 passing against the deployed URL
- [ ] `test_concurrency.py --workers 8` — acceptable latency, `avg_batch_size` above 1
- [ ] Your real vocalist clips in `voices/`, placeholders removed
- [ ] Server runs under `run.sh` + `nohup` (or tmux), restarts on crash
- [ ] Container Start Command set, so it survives a pod reboot
- [ ] `/workspace` is a network volume, so weights survive recreation
