# Deploying to RunPod (no Docker)

Target: an **A100 80 GB** pod running the API for your website.

> **Code edits required: none.**
> Everything that differs between your laptop and the A100 — device, how many models stay
> resident, where Whisper runs, attention kernel, auth, CORS — is auto-detected or set via
> environment variables. You deploy the same [server/app.py](server/app.py) unchanged.
> Confirm what it chose with `GET /health` → `config`.

---

## 1. Create the pod

RunPod → **Pods** → **Deploy**.

| Setting | Value |
|---|---|
| GPU | A100 80 GB (PCIe or SXM) |
| Template | **RunPod PyTorch** (any recent CUDA 12.x image) |
| Container disk | 30 GB |
| Volume disk | **80 GB**, mounted at `/workspace` |
| Expose HTTP ports | `9000` |

Put the models on the **volume** (`/workspace`), not the container disk — the container disk
is wiped when the pod is rebuilt, and you do not want to re-download 14 GB each time.

Once running, open a **Web Terminal** or connect over SSH.

---

## 2. Install

Clone **this** repo — it holds the API server, which is not part of upstream Qwen3-TTS:

```bash
cd /workspace
git clone https://github.com/SHAIK-ABDUL-KHADEER/VocalLoop---Qwen3TTS.git vocalloop
cd /workspace/vocalloop

python -m venv venv
source venv/bin/activate
pip install -U pip
```

Install PyTorch matching the pod's CUDA (the PyTorch template usually ships it already —
check with `python -c "import torch;print(torch.__version__, torch.version.cuda)"` and skip
this if it prints a CUDA build):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then the TTS engine and the API's own dependencies:

```bash
pip install -U qwen-tts
pip install fastapi "uvicorn[standard]" python-multipart requests
```

> `qwen-tts` comes straight from PyPI — this project does **not** patch the upstream library,
> so there is no need to clone `QwenLM/Qwen3-TTS`. Every workaround for its quirks lives in
> [server/app.py](server/app.py). Clone upstream only if you want the Gradio demo or the
> fine-tuning scripts.

**FlashAttention 2** — worth it on Linux, and the server picks it up automatically:

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

This compiles and takes 10–30 minutes. If it fails, skip it — the server falls back to
`sdpa` on its own. Verify afterwards with `GET /health` → `config.attn_implementation`.

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

About **14.2 GB** total. The server looks in `<repo>/models` by default, which is exactly
this path — no config needed. If you put them elsewhere, set `QWEN_TTS_MODEL_DIR`.

---

## 4. Configure

Create `/workspace/vocalloop/.env`:

```bash
# Bind publicly so RunPod's proxy can reach it
QWEN_TTS_HOST=0.0.0.0
QWEN_TTS_PORT=9000

# Require this header on every generation request. Generate with:
#   python -c "import secrets; print(secrets.token_urlsafe(32))"
QWEN_TTS_API_KEY=<paste-a-long-random-string>

# Only your site may call the API from a browser
QWEN_TTS_CORS_ORIGINS=https://yoursite.com,https://www.yoursite.com
```

Everything else auto-detects. On an A100 80 GB you should see:

| Setting | Auto-chosen value | Why |
|---|---|---|
| `device` | `cuda:0` | GPU present |
| `max_resident` | `3` | ≥24 GB VRAM, so all three TTS models stay loaded |
| `asr_device` | `cuda:0` | ≥16 GB VRAM, so Whisper runs on GPU (~10× faster) |
| `attn_implementation` | `flash_attention_2` | if flash-attn installed, else `sdpa` |

Total resident: **14.2 GB of 80 GB**. Nothing ever evicts or reloads.

> **Why this matters.** On the 8 GB laptop the server keeps only one TTS model resident and
> pushes Whisper to the CPU, because there is not enough VRAM for more. Model eviction on a
> small card also proved genuinely unstable — repeatedly freeing a 4 GB model crashed the
> CUDA process outright. On the A100 none of that applies: everything is resident on the GPU
> from the first request and eviction never happens.

---

## 5. Run it

```bash
cd /workspace/vocalloop
source venv/bin/activate
set -a && source .env && set +a
python -m uvicorn server.app:app --host 0.0.0.0 --port 9000
```

Watch the startup lines — they print the resolved config:

```
[startup] gpu          : NVIDIA A100-SXM4-80GB (79.2 GB)
[startup] device       : cuda:0
[startup] max resident : 3
[startup] attn impl    : flash_attention_2
[startup] asr model    : .../whisper-large-v3-turbo on cuda:0
[startup] auth         : X-API-Key required
```

If `device` says `cpu` or `asr model ... on cpu`, stop and fix it — you are not on the GPU.

### Keep it running after you disconnect

```bash
tmux new -s tts
# start the server, then detach with Ctrl-B then D
```

Or run it as a service so it survives a pod restart:

```bash
cat >/etc/systemd/system/qwen-tts.service <<'EOF'
[Unit]
Description=Qwen3-TTS API
After=network.target

[Service]
WorkingDirectory=/workspace/vocalloop
EnvironmentFile=/workspace/vocalloop/.env
ExecStart=/workspace/vocalloop/venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 9000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload && systemctl enable --now qwen-tts
journalctl -u qwen-tts -f
```

---

## 6. Verify

Your public URL is `https://<POD_ID>-9000.proxy.runpod.net` (shown on the pod's Connect page).

```bash
curl https://<POD_ID>-9000.proxy.runpod.net/health
```

`/health` stays open so RunPod's probes work. Confirm `config.device` is `cuda:0` and
`config.asr_device` is `cuda:0`.

Run the full suite against the deployed server (needs `requests` locally):

```bash
python test_endpoints.py --base-url https://<POD_ID>-9000.proxy.runpod.net
```

> The suite does **not** send an API key. Either run it before setting `QWEN_TTS_API_KEY`,
> or add the header to `check()` in [test_endpoints.py](test_endpoints.py). Alternatively run it
> from inside the pod against `http://127.0.0.1:9000`.

A first-call warm-up of ~10–20 s per model is normal; after that all three stay resident.

---

## 7. Point your website at it

```js
const API = "https://<POD_ID>-9000.proxy.runpod.net";
const KEY = process.env.QWEN_TTS_API_KEY;   // server-side only, never ship to the browser

const res = await fetch(`${API}/v1/tts`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": KEY },
  body: JSON.stringify({ text, speaker, instruct, language: "auto", response_format: "wav" }),
});
```

**Do not put the API key in browser JavaScript** — anyone could read it and spend your GPU
time. Proxy these calls through your own backend, which holds the key. If you must call
directly from the browser, leave `QWEN_TTS_API_KEY` unset and instead restrict
`QWEN_TTS_CORS_ORIGINS` to your domain, understanding that CORS is not real access control.

See [ENDPOINTS.md](ENDPOINTS.md) for the full request/response reference.

---

## Operational notes

**Concurrency.** The server serializes GPU work behind one lock — one generation at a time,
the rest queue. To use more of the A100, run several workers, each a separate process with
its own copy of the models (14.2 GB each, so 4–5 fit in 80 GB), behind nginx:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 9001 &
python -m uvicorn server.app:app --host 0.0.0.0 --port 9002 &
```

Do **not** use `uvicorn --workers N` — each worker would load its own models with no
coordination, and they would race for VRAM. Separate ports plus a load balancer is
predictable. Measure before scaling: run [bench.py](bench.py) on the pod first.

**Cost.** A pod bills while running, whether or not it is serving. Stop it when idle. If
traffic is bursty, RunPod Serverless suits the workload better, but that route wants a
container image — out of scope here, since you asked for no Docker.

**Storage.** Keep `/workspace` as a network volume so the 14.2 GB of weights survives pod
recreation. Rebuilding the container disk alone will not lose them.

**First request is slow.** Models load lazily on first use (~10–20 s each). To warm them up
at deploy time, fire one request per mode:

```bash
curl -X POST localhost:9000/v1/tts -H 'Content-Type: application/json' \
  -H "X-API-Key: $QWEN_TTS_API_KEY" -d '{"text":"warm up","speaker":"ryan"}'
curl -X POST localhost:9000/v1/tts -H 'Content-Type: application/json' \
  -H "X-API-Key: $QWEN_TTS_API_KEY" -d '{"text":"warm up","instruct":"a calm voice"}'
curl -X POST localhost:9000/v1/transcribe -H 'Content-Type: application/json' \
  -H "X-API-Key: $QWEN_TTS_API_KEY" \
  -d '{"audio":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"}'
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `config.device` is `cpu` | No CUDA in the venv. Reinstall torch with the right `--index-url`. |
| `config.asr_device` is `cpu` on the A100 | VRAM read below 16 GB, or `QWEN_TTS_ASR_DEVICE=cpu` left in `.env`. Remove it. |
| `503 Model 'x' is not downloaded` | Wrong `QWEN_TTS_MODEL_DIR`, or a download did not finish. Folder names must match section 3 exactly. |
| `index is on cpu, different from other tensors on cuda:0` | Not enough free VRAM, so a model loaded split across devices. Should not happen on an A100 — check nothing else is using the GPU (`nvidia-smi`). |
| `401 Invalid or missing X-API-Key` | Send the `X-API-Key` header, or unset `QWEN_TTS_API_KEY`. |
| CORS errors in the browser | Set `QWEN_TTS_CORS_ORIGINS` to your exact origin, scheme included. |
| Server dies with no traceback | Almost always OOM. Check `nvidia-smi`; make sure only one server process is running. |
| Port not reachable | Port `9000` must be in the pod's exposed HTTP ports, and the server bound to `0.0.0.0`, not `127.0.0.1`. |
