# Qwen3-TTS API for Cloud Run (GPU). See DEPLOY_GCP.md.
#
# Model weights are NOT baked in. They live in a Cloud Storage bucket mounted at
# /mnt/gcs, which keeps this image small enough to cold-start quickly and lets you
# swap models or vocalist clips without rebuilding.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf

# sox/ffmpeg/libsndfile: audio decoding for uploaded and vocalist reference clips.
RUN apt-get update \
 && apt-get install -y --no-install-recommends sox libsox-fmt-all ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Same torch build as the dev machine. The pip wheel bundles its own CUDA 12.8
# runtime, so it only needs a CUDA 12.x driver on the host, which Cloud Run provides.
RUN pip install torch==2.11.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128

# Vendored TTS library first (changes rarely), then the API code (changes often),
# so ordinary code edits rebuild only the last layer.
COPY Qwen3-TTS /app/Qwen3-TTS
RUN pip install ./Qwen3-TTS \
      fastapi==0.139.2 "uvicorn[standard]==0.51.0" python-multipart requests

COPY server /app/server

# Where the Cloud Storage bucket is mounted (see the deploy command in DEPLOY_GCP.md).
ENV QWEN_TTS_MODEL_DIR=/mnt/gcs/models \
    QWEN_TTS_VOICE_DIR=/mnt/gcs/voices \
    PORT=8080

EXPOSE 8080

# One process, no --workers: the server batches concurrent requests on one GPU.
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 75"]
