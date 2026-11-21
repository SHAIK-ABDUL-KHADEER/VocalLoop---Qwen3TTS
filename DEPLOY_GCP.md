# Deploying to Google Cloud (Cloud Run GPU + Cloud Storage)

Target: a **serverless L4 GPU** on Cloud Run that scales to zero when idle, with the model
weights and vocalist clips in a **Cloud Storage bucket**.

```
  your website ──HTTPS──▶ Cloud Run service (L4 24 GB, 0..N instances)
                             │  /mnt/gcs  (Cloud Storage FUSE mount)
                             ▼
                          gs://<bucket>/models/   14.2 GB weights
                          gs://<bucket>/voices/   vocalist clips + voices.json
```

Roughly **60–90 minutes** end to end, most of it waiting on the model copy and the image build.

---

## What you must change in the code

**Nothing.** Same as on RunPod: everything that differs is auto-detected or set by an
environment variable. The repo adds only deployment files:

| File | Purpose |
|---|---|
| [Dockerfile](Dockerfile) | The container image: Python 3.12, torch 2.11 (CUDA 12.8), the API |
| [.dockerignore](.dockerignore) / [.gcloudignore](.gcloudignore) | Keep `models/`, `venv/` and audio out of the upload and image |
| [deploy/gcp/seed-models.yaml](deploy/gcp/seed-models.yaml) | Cloud Build job: Hugging Face → your bucket |
| [deploy/gcp/env.example.yaml](deploy/gcp/env.example.yaml) | Environment variables for the service |

What the L4 auto-selects (24 GB VRAM): all 3 TTS models resident, Whisper on GPU, batch 16,
`sdpa` attention. That is 14.2 GB of 24 GB, so nothing evicts.

### Things that behave differently from a pod

| | RunPod pod | Cloud Run |
|---|---|---|
| Idle cost | Bills while running | **Scales to zero**, so you pay nothing when there are no requests |
| First request after idle | Instant | **Cold start, ~1–3 min** (container start + model load) |
| Models stored on | Volume disk | Cloud Storage bucket, mounted at `/mnt/gcs` |
| `QWEN_TTS_WARMUP` | `1` | **`0`**. Reading 14 GB before the port opens can miss Cloud Run's startup deadline |
| Restart on crash | Your `run.sh` loop | Built in |
| Auth | `X-API-Key` | Same `X-API-Key`, stored in Secret Manager |

---

## 0. Before you start

1. A Google Cloud project with **billing enabled**.
2. Pick a **region that offers Cloud Run L4 GPUs**. `us-central1` is the safe default; check
   the current list in the Cloud Run GPU docs ("Supported regions"). The bucket and the
   service **must be in the same region**, or every model load crosses regions and is slow.
3. **GPU quota.** In the Console → *IAM & Admin → Quotas*, search for
   `Total Nvidia L4 GPU allocation without zonal redundancy, per project per region` for
   your region. If it is `0`, request `1` (or more) and wait for approval. You cannot
   deploy until it is granted.

Run every command below in **Cloud Shell** (the `>_` button at the top of the Console). It has
`gcloud`, `git` and `python3` preinstalled, so nothing has to be installed on your laptop.

> Prefer your own machine? Install the Google Cloud CLI (`winget install Google.CloudSDK`),
> run `gcloud auth login`, and use the same commands in Git Bash from the repo folder.

---

## 1. Set variables and enable APIs

Paste this, after editing the first two lines. If Cloud Shell restarts, paste it again.

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1

export BUCKET=${PROJECT_ID}-qwen-tts
export SERVICE=qwen-tts
export REPO=qwen-tts
export SA_EMAIL=qwen-tts-run@${PROJECT_ID}.iam.gserviceaccount.com
export IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/server:latest

gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com storage.googleapis.com
```

Get the code:

```bash
cd ~
git clone https://github.com/SHAIK-ABDUL-KHADEER/VocalLoop---Qwen3TTS.git vocalloop
cd ~/vocalloop
```

---

## 2. Create the bucket and load the models

```bash
gcloud storage buckets create gs://$BUCKET --location=$REGION --uniform-bucket-level-access
```

Copy the four model folders from Hugging Face into the bucket. This runs as a Cloud Build
job inside Google's network, so the 14 GB never goes through your connection:

```bash
gcloud builds submit --no-source --config deploy/gcp/seed-models.yaml \
  --substitutions _BUCKET=$BUCKET
```

**~15–30 minutes.** Verify:

```bash
gcloud storage du --summarize --readable-sizes gs://$BUCKET/models/*
```

You should see four folders of roughly 4.2 / 4.2 / 4.2 / 1.6 GB. The folder names must match
exactly (`Qwen3-TTS-12Hz-1.7B-VoiceDesign`, `…-CustomVoice`, `…-Base`,
`whisper-large-v3-turbo`), and the job above already gets them right.

### Your vocalist voices

```bash
gcloud storage cp voices/*.wav voices/voices.json gs://$BUCKET/voices/
```

Replace the two placeholder clips with your real ones whenever you like. A new clip is
picked up on the next request, with no redeploy. See [voices/README.md](voices/README.md).

---

## 3. Build the container image

```bash
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION

gcloud builds submit --tag $IMAGE --machine-type=e2-highcpu-8 --disk-size=100 --timeout=3600
```

**~10–20 minutes.** The image is about 7–8 GB, mostly torch and its CUDA libraries. Only
`Dockerfile`, `Qwen3-TTS/` and `server/` are uploaded ([.gcloudignore](.gcloudignore)).

> FlashAttention is not installed in the image. Compiling it adds 30+ minutes to every
> build, and the server falls back to `sdpa` on its own, which is fine on an L4.

---

## 4. Service account, API key and config

A dedicated identity for the service, with access to the bucket only:

```bash
gcloud iam service-accounts create qwen-tts-run --display-name "Qwen3-TTS Cloud Run"

gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
  --member=serviceAccount:$SA_EMAIL --role=roles/storage.objectUser
```

`objectUser` (read **and write**) is needed because the server caches each vocalist clip's
transcript back into `voices/voices.json`.

The API key, in Secret Manager rather than in a file:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32), end='')" \
  | gcloud secrets create qwen-tts-api-key --data-file=-

gcloud secrets add-iam-policy-binding qwen-tts-api-key \
  --member=serviceAccount:$SA_EMAIL --role=roles/secretmanager.secretAccessor

# Show it, so you can put it in your website's backend:
gcloud secrets versions access latest --secret=qwen-tts-api-key; echo
```

Environment variables:

```bash
cp deploy/gcp/env.example.yaml deploy/gcp/env.yaml
nano deploy/gcp/env.yaml        # set QWEN_TTS_CORS_ORIGINS to your site
```

---

## 5. Deploy

```bash
gcloud run deploy $SERVICE \
  --image=$IMAGE \
  --region=$REGION \
  --service-account=$SA_EMAIL \
  --gpu=1 --gpu-type=nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu=8 --memory=32Gi --no-cpu-throttling \
  --execution-environment=gen2 \
  --min-instances=0 --max-instances=1 \
  --concurrency=32 \
  --timeout=900 \
  --add-volume=name=gcs,type=cloud-storage,bucket=$BUCKET,mount-options=implicit-dirs \
  --add-volume-mount=volume=gcs,mount-path=/mnt/gcs \
  --env-vars-file=deploy/gcp/env.yaml \
  --set-secrets=QWEN_TTS_API_KEY=qwen-tts-api-key:latest \
  --allow-unauthenticated
```

Why each choice:

| Flag | Why |
|---|---|
| `--gpu-type=nvidia-l4` | 24 GB is enough to keep all models resident (14.2 GB) |
| `--no-gpu-zonal-redundancy` | Cheaper, and easier to get quota for. Fine for this workload |
| `--cpu=8 --memory=32Gi` | Model loading is staged through RAM; 16 GiB is too tight for 4 models |
| `--no-cpu-throttling` | Required with a GPU; also keeps the batch worker running between requests |
| `--min-instances=0` | **Scale to zero.** Set `1` to remove cold starts (but then it bills 24/7) |
| `--max-instances=1` | One instance batches requests on one GPU, the efficient setup (see §8) |
| `--concurrency=32` | Up to 32 simultaneous requests reach the same instance, so they get batched |
| `--timeout=900` | Long generations; above `QWEN_TTS_TIMEOUT` so the server's own `504` wins |
| `mount-options=implicit-dirs` | Folders created by `gcloud storage cp` exist only as name prefixes; this makes them visible as directories |
| `--allow-unauthenticated` | Public URL; generation is still protected by `X-API-Key` |

The command prints the service URL: `https://qwen-tts-xxxxxxxx-uc.a.run.app`.

---

## 6. Verify

```bash
export URL=$(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')
export KEY=$(gcloud secrets versions access latest --secret=qwen-tts-api-key)
```

**Health first.** The first call is a cold start; give it a minute:

```bash
curl --max-time 300 $URL/health | python3 -m json.tool
```

Check:

- `cuda: true`, `config.device` and `config.asr_device` both `cuda:0`
- `gpu.name` is `NVIDIA L4`
- `models_downloaded` is all `true`. If not, see §9.

**Generate something.** The first call per mode also loads that model (~20–60 s):

```bash
curl --max-time 600 -X POST $URL/v1/speak \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"text":"Deployment successful.","speaker":"ryan","response_format":"wav"}' \
  --output test.wav && ls -lh test.wav
```

Check that auth is enforced (should print `401`):

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST $URL/v1/speak \
  -H "Content-Type: application/json" -d '{"text":"hi","speaker":"ryan"}'
```

**Full suite** (28 checks), from Cloud Shell or your laptop:

```bash
python3 test_endpoints.py --base-url $URL --api-key "$KEY"
python3 test_concurrency.py --base-url $URL --api-key "$KEY" --workers 8 --rounds 2
```

**Logs**, including the `[startup]` lines that show what was auto-selected:

```bash
gcloud run services logs read $SERVICE --region $REGION --limit 100
```

---

## 7. Point your website at it

Identical to the RunPod setup. Only the URL changes:

```js
const API = "https://qwen-tts-xxxxxxxx-uc.a.run.app";
const KEY = process.env.QWEN_TTS_API_KEY;   // server-side only

const res = await fetch(`${API}/v1/speak`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": KEY },
  body: JSON.stringify({ text, speaker, language: "auto", response_format: "wav" }),
});
```

All endpoints (`POST /v1/speak`, `GET /v1/voices`, `GET /health`, `/docs`, `/demo`) work
unchanged. Full reference: [ENDPOINTS.md](ENDPOINTS.md); examples:
[CURL_EXAMPLES.md](CURL_EXAMPLES.md).

⚠️ **Never put the API key in browser JavaScript.** Call the TTS API from your own backend.

**Plan for cold starts in the UI.** After an idle period the first request can take 1–3
minutes. Use a client timeout of at least 5 minutes and show a "warming up the voice
engine…" state. If that is not acceptable, set `--min-instances=1` (see §8).

---

## 8. Operating it

**Cost.** With GPUs, Cloud Run bills for the whole time an instance is up, not only
while it serves requests. After the last request an idle instance stays up for a while
(up to ~15 minutes) before scaling to zero, and that time is billed. Check current L4 rates
on the Cloud Run pricing page; set a **budget alert** in Billing either way.

| Setting | Effect |
|---|---|
| `--min-instances=0` | $0 when idle; cold start after idle |
| `--min-instances=1` | Always warm; bills 24/7 like a pod |

Switch any time without a rebuild:

```bash
gcloud run services update $SERVICE --region $REGION --min-instances=1
```

**Scaling.** Keep `--max-instances=1` until `/health` → `load.queue_wait_p95_seconds` stays
high. Raise `QWEN_TTS_MAX_BATCH` first, then `--max-instances` (each extra instance needs
another unit of L4 quota and loads its own copy of the models). Do not add uvicorn
`--workers`.

**Updating the code:**

```bash
cd ~/vocalloop && git pull
gcloud builds submit --tag $IMAGE --machine-type=e2-highcpu-8 --disk-size=100 --timeout=3600
gcloud run deploy $SERVICE --image=$IMAGE --region=$REGION
```

The last command keeps every other setting from §5.

**Updating voices:** `gcloud storage cp new_voice.wav gs://$BUCKET/voices/`. No redeploy.

**Changing env vars:** edit `deploy/gcp/env.yaml`, then
`gcloud run services update $SERVICE --region $REGION --env-vars-file=deploy/gcp/env.yaml`.

**Rotating the API key:** add a new secret version
(`... | gcloud secrets versions add qwen-tts-api-key --data-file=-`), then redeploy so the
service picks up `latest`.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Deploy fails: quota / "no GPU available" | L4 quota is 0 in this region (§0), or the region has no Cloud Run L4. Request quota or change `REGION` (bucket too). |
| Deploy fails: `--allow-unauthenticated` / `allUsers` refused | An org policy blocks public services. Remove the flag and front it with your backend using an ID token, or ask your org admin. |
| `models_downloaded` shows `false` | Bucket not mounted at `/mnt/gcs`, wrong folder names, or `implicit-dirs` missing. Check `gcloud storage ls gs://$BUCKET/models/`. |
| `cuda: false` / `device: cpu` | Deployed without `--gpu=1`. Redeploy with the §5 command. |
| Container fails to start / startup probe failed | `QWEN_TTS_WARMUP=1` is set. Set it to `0`. Logs show the exact error. |
| Out of memory, instance restarts | Memory below 32 GiB. Raise `--memory`. |
| `403` writing `voices.json` in logs | Service account has read-only bucket access. Grant `roles/storage.objectUser` (§4). |
| Seed job fails with permission denied on the bucket | The Cloud Build service account cannot write to the bucket. Grant it `roles/storage.objectAdmin` on `gs://$BUCKET`. |
| Image build runs out of disk | Keep `--disk-size=100`. |
| Very slow first request after idle | Expected cold start. Use `--min-instances=1` if that is unacceptable. |
| `504` from Cloud Run itself (HTML, not JSON) | Request exceeded `--timeout`. Raise it (max 3600) or split the text. |
| `401 Invalid or missing X-API-Key` | Send the header; the value is `gcloud secrets versions access latest --secret=qwen-tts-api-key`. |
| CORS errors in the browser | Set `QWEN_TTS_CORS_ORIGINS` in `env.yaml` to your exact origin, scheme included, then update the service. |

The API-level errors (`400`, `404`, `422`, `502`, `503`) mean the same as on RunPod; see
[DEPLOY_RUNPOD.md §9](DEPLOY_RUNPOD.md#9-troubleshooting).

---

## 10. Pre-launch checklist

- [ ] Bucket and service in the **same region**
- [ ] `/health` shows `cuda: true`, `device: cuda:0`, `asr_device: cuda:0`, `gpu.name: NVIDIA L4`
- [ ] `models_downloaded` all `true`
- [ ] Request without `X-API-Key` returns `401`
- [ ] `QWEN_TTS_CORS_ORIGINS` restricted to your domain, not `*`
- [ ] API key held server-side only, never in browser JavaScript
- [ ] `test_endpoints.py`: 28/28 passing against the Cloud Run URL
- [ ] Website client timeout ≥ 5 min, with a "warming up" state for cold starts
- [ ] Real vocalist clips in `gs://$BUCKET/voices/`, placeholders removed
- [ ] Budget alert set in Billing
