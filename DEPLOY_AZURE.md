# Deploying to Azure (Container Apps serverless GPU + Azure Files)

Target: an **A100 80 GB serverless GPU** on Azure Container Apps that scales to zero when idle,
with the model weights and vocalist clips on an **Azure Files** share.

```
  your website ──HTTPS──▶ Container App (A100 80 GB, 0..1 replicas)
                             │  /mnt/tts  (Azure Files SMB mount)
                             ▼
                          share "tts"/models/   14.2 GB weights
                          share "tts"/voices/   vocalist clips + voices.json
```

Roughly **60–90 minutes**, most of it waiting on the model download and the image build.
The code is unchanged. Only [Dockerfile](Dockerfile) and [deploy/azure/app.yaml](deploy/azure/app.yaml)
are used.

### Why A100 and not T4

Container Apps offers two serverless GPUs. **Use the A100.** The T4 is cheaper but a poor fit:
the models run in bfloat16, which a T4 (pre-Ampere) does not support natively. Its 16 GB also
reports as ~14.7 GB, below the server's 16 GB threshold, so only one model would stay resident
and every mode switch would evict and reload 4 GB. On the A100 everything auto-tunes exactly as
in [DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md): 3 models resident, Whisper on GPU, batch 16.

### Different from a pod

| | |
|---|---|
| Idle cost | **Scales to zero**, so nothing is billed for the GPU while idle |
| After idle | **Cold start, ~1–3 min** before the first request is served |
| Request limit | Container Apps ingress ends any HTTP request at **240 s**, so keep requests short |
| `QWEN_TTS_WARMUP` | `0`: models load on first use, so startup stays fast |

---

## 0. Before you start (Portal, manual)

1. **An Azure subscription with billing.** Free-trial and student subscriptions usually have
   no GPU quota and cannot run ACR Tasks builds. Use pay-as-you-go.
2. **Pick a region with serverless GPUs.** At the time of writing these include
   `swedencentral`, `westus3` and `australiaeast`. §1 lists them for certain.
3. **GPU quota.** Serverless GPU quota is per subscription and region. Pay-as-you-go and EA
   subscriptions often have it already. If the deploy in §5 fails with a quota error,
   request it: Portal → **Help + support → Create a support request** → *Service and
   subscription limits (quotas)* → quota type **Azure Container Apps** → ask for
   serverless **A100** GPU quota in your region.

Run everything below in **Azure Cloud Shell** (the `>_` icon at the top of the Portal, choose
**Bash**). It has `az`, `git` and `envsubst` built in.

---

## 1. Variables, providers, code

Edit `SUFFIX` to something unique (lowercase letters/digits, e.g. your initials + 4 digits).
It becomes part of globally unique names. If Cloud Shell restarts, paste this block again;
the `git clone` will then just say the folder exists.

```bash
export SUFFIX=abc1234
export LOC=swedencentral
export RG=qwen-tts-rg ENVNAME=qwen-tts-env APP=qwen-tts
export STORAGE=qwentts${SUFFIX} ACR=qwentts${SUFFIX}

az extension add --name containerapp --upgrade -y
for p in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry \
         Microsoft.ContainerInstance Microsoft.Storage; do az provider register -n $p; done

cd ~ && git clone -b azure-deploy https://github.com/SHAIK-ABDUL-KHADEER/VocalLoop---Qwen3TTS.git vocalloop
cd ~/vocalloop
```

Confirm your region offers the GPU profile:

```bash
az containerapp env workload-profile list-supported -l $LOC -o table | grep -i gpu
```

You need a row containing **`Consumption-GPU-NC24-A100`**. If there is none, change `LOC` and
re-paste the block above.

---

## 2. Storage for models and voices

A **Premium** file share. Models load from it on every cold start, and Premium gives
much higher, steadier throughput than Standard (~$16/month for 100 GiB).

```bash
az group create -n $RG -l $LOC
az storage account create -n $STORAGE -g $RG -l $LOC --kind FileStorage --sku Premium_LRS
az storage share-rm create --storage-account $STORAGE -g $RG -n tts --quota 100 --enabled-protocols SMB
export STORAGE_KEY=$(az storage account keys list -n $STORAGE -g $RG --query "[0].value" -o tsv)
```

**Download the models straight into the share** with a temporary container, so the 14 GB never
passes through your own connection (~15–30 min, a few cents):

```bash
az container create -g $RG -n qwen-tts-seed -l $LOC --os-type Linux \
  --image python:3.12-slim --cpu 4 --memory 8 --restart-policy Never \
  --environment-variables HF_HUB_DISABLE_PROGRESS_BARS=1 \
  --azure-file-volume-account-name $STORAGE --azure-file-volume-account-key "$STORAGE_KEY" \
  --azure-file-volume-share-name tts --azure-file-volume-mount-path /mnt/tts \
  --command-line "/bin/bash -c 'pip install -q -U huggingface_hub && for r in Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice Qwen/Qwen3-TTS-12Hz-1.7B-Base openai/whisper-large-v3-turbo; do hf download \$r --local-dir /mnt/tts/models/\${r#*/} || exit 1; done; rm -rf /mnt/tts/models/*/.cache; du -sh /mnt/tts/models/*'"
```

Check on it. Repeat until the state is `Succeeded`:

```bash
az container show -g $RG -n qwen-tts-seed --query "instanceView.state" -o tsv
az container logs -g $RG -n qwen-tts-seed | tail -5
```

The final log lines should list four folders of about 4.2 / 4.2 / 4.2 / 1.6 GB. Then delete
the helper container:

```bash
az container delete -g $RG -n qwen-tts-seed -y
```

**Vocalist voices:**

```bash
az storage file upload-batch --account-name $STORAGE --account-key "$STORAGE_KEY" \
  --destination tts --destination-path voices --source voices --pattern "*.wav"
az storage file upload --account-name $STORAGE --account-key "$STORAGE_KEY" \
  --share-name tts --source voices/voices.json --path voices/voices.json
```

---

## 3. Build the container image

Azure Container Registry builds the [Dockerfile](Dockerfile) in the cloud. No Docker needed
locally (~15–30 min, image ~8 GB):

```bash
az acr create -n $ACR -g $RG -l $LOC --sku Basic --admin-enabled true
az acr build -r $ACR -t qwen-tts:latest --timeout 7200 .
```

Only `Dockerfile`, `Qwen3-TTS/` and `server/` are sent ([.dockerignore](.dockerignore)).

> If `az acr build` says tasks are not allowed for your subscription, it is a free-trial
> restriction. Upgrade to pay-as-you-go and retry.

---

## 4. Container Apps environment with an A100 profile

```bash
az containerapp env create -n $ENVNAME -g $RG -l $LOC --enable-workload-profiles
az containerapp env workload-profile add -g $RG -n $ENVNAME \
  --workload-profile-name gpu --workload-profile-type Consumption-GPU-NC24-A100

az containerapp env storage set -n $ENVNAME -g $RG --storage-name tts \
  --azure-file-account-name $STORAGE --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name tts --access-mode ReadWrite
```

`ReadWrite` is needed because the server caches each vocalist clip's transcript into
`voices/voices.json`.

---

## 5. Deploy

Set your website's origin. `*` is fine while testing. Generate the API key and **save the
printed value**: your website's backend sends it as `X-API-Key`.

```bash
export CORS="https://yoursite.com"
export API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
echo "API KEY: $API_KEY"

export ENV_ID=$(az containerapp env show -n $ENVNAME -g $RG --query id -o tsv)
export ACR_PASSWORD=$(az acr credential show -n $ACR --query "passwords[0].value" -o tsv)

envsubst '$LOC $ENV_ID $ACR $ACR_PASSWORD $API_KEY $CORS' < deploy/azure/app.yaml > /tmp/app.yaml
az containerapp create -n $APP -g $RG --yaml /tmp/app.yaml
rm /tmp/app.yaml
```

The app definition is in [deploy/azure/app.yaml](deploy/azure/app.yaml): 1 GPU replica max,
scale to zero, the share mounted at `/mnt/tts`, the API key stored as a Container Apps secret.

---

## 6. Verify

```bash
export URL=https://$(az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv)
echo $URL
curl --max-time 240 $URL/health | python3 -m json.tool
```

The first call wakes the app from zero; retry if it times out. Check:

- `cuda: true` and `gpu.name` contains `A100`
- `config.device` and `config.asr_device` are both `cuda:0`, `max_resident` is `3`
- `models_downloaded` all `true`

Then generate. The first request per mode also loads that model (~30–90 s from the share):

```bash
curl --max-time 240 -X POST $URL/v1/speak -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"text":"Deployment successful.","speaker":"ryan","response_format":"wav"}' -o test.wav && ls -lh test.wav
```

Download `test.wav` from Cloud Shell (**Manage files → Download**, path
`vocalloop/test.wav`) and listen. A request without the key must return `401`:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST $URL/v1/speak -H "Content-Type: application/json" -d '{"text":"hi","speaker":"ryan"}'
```

Full suite, from Cloud Shell or your laptop:

```bash
python3 test_endpoints.py --base-url $URL --api-key "$API_KEY"
```

Logs, including the `[startup]` lines:

```bash
az containerapp logs show -n $APP -g $RG --tail 100
```

---

## 7. Point your website at it

Same API as everywhere else. Only the URL changes:

```js
const API = "https://qwen-tts.<random>.<region>.azurecontainerapps.io";
const res = await fetch(`${API}/v1/speak`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": process.env.QWEN_TTS_API_KEY },
  body: JSON.stringify({ text, speaker, language: "auto", response_format: "wav" }),
});
```

- ⚠️ Keep the key in your backend, never in browser JavaScript.
- **Cold starts:** after idle, the first request can take minutes and may hit the 240 s
  ingress limit. Have your backend call `GET /health` first (it wakes the app without
  loading a model), retry once on `504`, and show a "warming up…" state.
- **240 s per request:** keep each request to a few sentences. Long scripts should be split
  into several requests (or one batched JSON request of short items).

Reference: [ENDPOINTS.md](ENDPOINTS.md), [CURL_EXAMPLES.md](CURL_EXAMPLES.md).

---

## 8. Operating it

**Cost.** You pay per second while a replica runs. It scales to zero after a few idle minutes
(cooldown, default 300 s). Check current serverless A100 rates on the Container Apps pricing
page and **set a budget alert** (Cost Management → Budgets).

**Always warm** (no cold starts, but bills 24/7):

```bash
az containerapp update -n $APP -g $RG --min-replicas 1
```

**Update the code:**

```bash
cd ~/vocalloop && git pull
az acr build -r $ACR -t qwen-tts:latest --timeout 7200 .
az containerapp update -n $APP -g $RG --image $ACR.azurecr.io/qwen-tts:latest \
  --revision-suffix v$(date +%s)
```

**Add voices:** upload a clip to the share's `voices/` folder (`az storage file upload ...`,
or Portal → storage account → File shares → tts → voices). No redeploy.

**Change env vars:** `az containerapp update -n $APP -g $RG --set-env-vars QWEN_TTS_CORS_ORIGINS=https://yoursite.com`

**Rotate the API key:**

```bash
az containerapp secret set -n $APP -g $RG --secrets api-key=<new-key>
az containerapp update -n $APP -g $RG --revision-suffix k$(date +%s)
```

**Delete everything:** `az group delete -n $RG -y`

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `workload-profile add` or `create` fails on quota / profile not available | No serverless A100 quota in this region (§0.3), or the region lacks the profile (§1 check). |
| `create` rejects `cpu`/`memory` | Run `az containerapp env workload-profile list-supported -l $LOC -o table` and set `resources` in `app.yaml` to the A100 profile's values. |
| `az acr build` refused | Free-trial subscription. Upgrade to pay-as-you-go. |
| Seed container stuck `Waiting`, or image pull error | Docker Hub rate limit. Delete and re-run the `az container create` command. |
| `models_downloaded` shows `false` | Share not mounted, or seed did not finish. Portal → storage account → File shares → tts → models should show four folders. |
| `cuda: false` | App not on the `gpu` workload profile. Check `workloadProfileName` in `app.yaml`. |
| `504` / `upstream request timeout` after 240 s | Ingress limit. Cold start plus first model load, or too much text in one request. Call `/health` first; split long text. |
| `401` | Wrong or missing `X-API-Key`. |
| CORS errors | Set `QWEN_TTS_CORS_ORIGINS` to your exact origin including `https://`. |
| Replica keeps restarting | `az containerapp logs show -n $APP -g $RG --tail 200`. Usually a startup error printed in the log. |

API-level errors (`400`, `404`, `422`, `502`, `503`) mean the same as in
[DEPLOY_RUNPOD.md §9](DEPLOY_RUNPOD.md#9-troubleshooting).
