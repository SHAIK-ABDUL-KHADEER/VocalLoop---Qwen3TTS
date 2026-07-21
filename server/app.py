"""Qwen3-TTS HTTP API.

Three audio endpoints:

    POST /v1/tts             one endpoint for both TTS modes:
                               speaker given   -> CustomVoice model (instruct optional)
                               no speaker      -> VoiceDesign model (instruct required)
    POST /v1/tts/clone       clone a voice from a reference clip
    POST /v1/transcribe      speech-to-text, so the frontend can show the
                             auto-generated reference transcript for editing

Reference transcripts are produced automatically: if `ref_text` is omitted from a
clone request the server transcribes `ref_audio` with Whisper. Callers may still
pass `ref_text` to skip that step (faster, and exact).

Deliberately not exposed, per spec:
  - model size: always 1.7B, the largest released
  - x-vector-only cloning: full ICL cloning is always used

`language` is optional everywhere and defaults to "auto".

Everything auto-tunes to the GPU present, so the same file runs unedited on an
8 GB laptop and an 80 GB A100. Check GET /health -> "config" to see what it chose.

Config via environment variables (all optional):
    QWEN_TTS_MODEL_DIR     model folders (default <repo>/models)
    QWEN_TTS_DEVICE        cuda:0 if a GPU is present, else cpu
    QWEN_TTS_MAX_RESIDENT  TTS models on the GPU at once (3 if >=24 GB, 2 if >=16, else 1)
    QWEN_TTS_ASR_MODEL     ASR model dir or hub id (default <MODEL_DIR>/whisper-large-v3-turbo)
    QWEN_TTS_ASR_DEVICE    GPU when >=16 GB VRAM, else cpu (see the note by ASR_DEVICE)
    QWEN_TTS_ATTN          flash_attention_2 when flash-attn is installed, else sdpa
    QWEN_TTS_API_KEY       require this value in an X-API-Key header (unset = open)
    QWEN_TTS_CORS_ORIGINS  comma-separated allowed origins (default *)
    QWEN_TTS_HOST/PORT     bind address (default 127.0.0.1:9000)
"""

from __future__ import annotations

import base64
import gc
import importlib.util
import io
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

from qwen_tts import Qwen3TTSModel

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Everything below auto-tunes to the GPU it finds, so the same code runs unedited
# on an 8 GB laptop and an 80 GB A100. Every value can still be forced via env.
_TOTAL_VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1024**3
    if torch.cuda.is_available() else 0.0
)

MODEL_DIR = os.environ.get("QWEN_TTS_MODEL_DIR", os.path.join(REPO_ROOT, "models"))
DEVICE = os.environ.get("QWEN_TTS_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")

# Each TTS model is ~4.2 GB. Keeping all three resident needs headroom for KV
# cache and activations too, so only do it on a genuinely large card.
MAX_RESIDENT = int(os.environ.get("QWEN_TTS_MAX_RESIDENT") or (
    3 if _TOTAL_VRAM_GB >= 24 else 2 if _TOTAL_VRAM_GB >= 16 else 1
))

ASR_MODEL = os.environ.get(
    "QWEN_TTS_ASR_MODEL", os.path.join(MODEL_DIR, "whisper-large-v3-turbo")
)
# Whisper (~1.6 GB) goes on the GPU whenever there is room to spare. On a small
# card it must not: sharing 8 GB with a 4.2 GB TTS model leaves so little headroom
# that the next TTS model loads with tensors stranded on the CPU, and generation
# dies with "index is on cpu, different from other tensors on cuda:0" rather than
# a clean OOM. Above ~16 GB that pressure disappears and GPU ASR is ~10x faster.
ASR_DEVICE = os.environ.get("QWEN_TTS_ASR_DEVICE") or (
    DEVICE if (DEVICE != "cpu" and _TOTAL_VRAM_GB >= 16) else "cpu"
)

# flash-attn has no Windows build; use it automatically wherever it is installed.
_HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None
ATTN = os.environ.get("QWEN_TTS_ATTN") or ("flash_attention_2" if _HAS_FLASH_ATTN else "sdpa")

# Auth + CORS are env-driven so deploying needs no code edits.
# Unset API key = open server (fine locally, not on a public host).
API_KEY = os.environ.get("QWEN_TTS_API_KEY") or None
CORS_ORIGINS = [
    o.strip() for o in os.environ.get("QWEN_TTS_CORS_ORIGINS", "*").split(",") if o.strip()
]

MODEL_PATHS = {
    "voice_design": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    "custom_voice": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "base": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-Base"),
}

# Speaker descriptions from the model card, surfaced through /v1/voices so the
# frontend can render a labelled picker.
SPEAKER_INFO = {
    "vivian": ("Bright, slightly edgy young female voice.", "Chinese"),
    "serena": ("Warm, gentle young female voice.", "Chinese"),
    "uncle_fu": ("Seasoned male voice with a low, mellow timbre.", "Chinese"),
    "dylan": ("Youthful Beijing male voice with a clear, natural timbre.", "Chinese (Beijing Dialect)"),
    "eric": ("Lively Chengdu male voice with a slightly husky brightness.", "Chinese (Sichuan Dialect)"),
    "ryan": ("Dynamic male voice with strong rhythmic drive.", "English"),
    "aiden": ("Sunny American male voice with a clear midrange.", "English"),
    "ono_anna": ("Playful Japanese female voice with a light, nimble timbre.", "Japanese"),
    "sohee": ("Warm Korean female voice with rich emotion.", "Korean"),
}

LANGUAGES = [
    "auto", "chinese", "english", "french", "german", "italian",
    "japanese", "korean", "portuguese", "russian", "spanish",
]


# --------------------------------------------------------------------------- #
# Model registry - lazy load with LRU eviction so this runs on small GPUs too
# --------------------------------------------------------------------------- #


class ModelRegistry:
    def __init__(self) -> None:
        self._models: "OrderedDict[str, Qwen3TTSModel]" = OrderedDict()
        self._load_lock = threading.Lock()

    def get(self, kind: str) -> Qwen3TTSModel:
        """Fetch a model, loading and evicting as needed.

        MUST be called while holding `gpu_lock`, and the returned model must only
        be used while that same lock is still held. Eviction frees the evicted
        model's CUDA storage, so if another request could evict between the get
        and the generate, it would pull the weights out from under a running call
        and raise "index is on cpu, different from other tensors on cuda:0".

        Note that eviction itself is fragile on small GPUs: repeatedly loading and
        freeing a ~4 GB model kills the CUDA process outright on Windows after a
        handful of cycles (native crash, no traceback). Keeping MAX_RESIDENT >=
        the number of models in use avoids eviction entirely, which is what the
        auto-tuned default does on any card with room for it.
        """
        with self._load_lock:
            if kind in self._models:
                self._models.move_to_end(kind)
                return self._models[kind]

            path = MODEL_PATHS[kind]
            if not os.path.isdir(path):
                raise HTTPException(
                    status_code=503, detail=f"Model '{kind}' is not downloaded at {path}."
                )

            while len(self._models) >= MAX_RESIDENT:
                evicted_kind, evicted_model = self._models.popitem(last=False)
                self._release(evicted_kind, evicted_model)
                del evicted_model
                gc.collect()
                torch.cuda.empty_cache()

            print(f"[registry] loading '{kind}'")
            t0 = time.time()
            model = Qwen3TTSModel.from_pretrained(
                path, device_map=DEVICE, dtype=torch.bfloat16, attn_implementation=ATTN
            )
            print(f"[registry] loaded '{kind}' in {time.time() - t0:.1f}s")
            self._models[kind] = model
            return model

    @staticmethod
    def _release(kind: str, model: Qwen3TTSModel) -> None:
        """Drop an evicted model. The caller deletes its reference, then runs
        gc.collect() + empty_cache(), which is what actually frees the VRAM.

        Do NOT `.to("cpu")` here. That copies ~4.2 GB into system RAM on every
        swap; with the incoming model also staging through RAM, a 16 GB machine
        runs out and Windows kills the process outright - no traceback, just a
        dead server. Dropping the last reference frees the CUDA storage directly.
        """
        print(f"[registry] evicted '{kind}' to free VRAM")

    def resident(self) -> List[str]:
        return list(self._models.keys())


registry = ModelRegistry()

# All GPU work is serialized: HF `generate` blocks, and the models are not safe
# to call concurrently from multiple threads.
gpu_lock = threading.Lock()

# The ASR model is deliberately kept OUT of the LRU pool above. It is small
# (~1.6 GB) and every clone request needs it, so evicting a TTS model to make
# room for it - then evicting it again to reload the TTS model - would thrash.
_asr = None
_asr_lock = threading.Lock()


def get_asr():
    global _asr
    with _asr_lock:
        if _asr is None:
            from transformers import pipeline

            src = ASR_MODEL
            if not os.path.isdir(src) and os.path.sep in src:
                raise HTTPException(
                    status_code=503,
                    detail=f"ASR model not found at {src}. Set QWEN_TTS_ASR_MODEL.",
                )
            print(f"[asr] loading {src} on {ASR_DEVICE}")
            t0 = time.time()
            _asr = pipeline(
                "automatic-speech-recognition",
                model=src,
                dtype=torch.float32 if ASR_DEVICE == "cpu" else torch.float16,
                device=ASR_DEVICE,
            )
            print(f"[asr] loaded in {time.time() - t0:.1f}s")
        return _asr


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

GEN_PARAM_FIELDS = (
    "max_new_tokens", "do_sample", "top_k", "top_p", "temperature",
    "repetition_penalty", "subtalker_dosample", "subtalker_top_k",
    "subtalker_top_p", "subtalker_temperature",
)

ResponseFormat = Literal["json", "wav"]


class GenParams(BaseModel):
    """Optional sampling knobs shared by every TTS endpoint."""

    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    subtalker_dosample: Optional[bool] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None
    subtalker_temperature: Optional[float] = None

    def kwargs(self) -> Dict[str, Any]:
        # Only the generation fields - subclasses add text/speaker/etc, and those
        # are passed explicitly, so leaking them here would duplicate them.
        dumped = self.model_dump()
        return {k: dumped[k] for k in GEN_PARAM_FIELDS if dumped.get(k) is not None}


def wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def audio_response(
    wavs: List[np.ndarray],
    sr: int,
    response_format: str,
    elapsed: float,
    extra: Optional[Dict[str, Any]] = None,
):
    """Raw WAV bytes (single item only) or JSON with base64 audio."""
    if response_format == "wav":
        if len(wavs) != 1:
            raise HTTPException(
                status_code=400,
                detail="response_format='wav' requires exactly one text item; use 'json' for batches.",
            )
        headers = {}
        # The transcript can't go in a JSON body here, so surface it in a header.
        if extra and extra.get("ref_text"):
            headers["X-Reference-Text"] = extra["ref_text"].encode("ascii", "replace").decode()
        return Response(content=wav_bytes(wavs[0], sr), media_type="audio/wav", headers=headers)

    payload: Dict[str, Any] = {
        "sample_rate": sr,
        "count": len(wavs),
        "durations": [round(len(w) / sr, 3) for w in wavs],
        "generation_seconds": round(elapsed, 3),
        "audio": [base64.b64encode(wav_bytes(w, sr)).decode("ascii") for w in wavs],
    }
    if extra:
        payload.update(extra)
    return JSONResponse(payload)


def as_list(x: Union[str, List[str]]) -> List[str]:
    return x if isinstance(x, list) else [x]


def broadcast(value, n: int, name: str) -> List[Any]:
    """Accept a scalar or an n-length list and always return an n-length list."""
    if value is None:
        return [None] * n
    if isinstance(value, list):
        if len(value) == 1:
            return value * n
        if len(value) != n:
            raise HTTPException(
                status_code=400, detail=f"'{name}' has length {len(value)}, expected 1 or {n}"
            )
        return value
    return [value] * n


def normalize_audio_ref(s: str) -> str:
    """Make a caller-supplied audio reference safe for the library's input sniffing.

    qwen_tts decides "is this base64?" with `"/" not in s and "\\" not in s`, but
    standard base64 uses '/' as a symbol - so most real base64 audio is misread as
    a file path and fails to open. A `data:audio/...` URI is always recognised, so
    anything that looks like bare base64 gets rewrapped as one.
    """
    if not isinstance(s, str) or s.startswith("data:") or s.startswith(("http://", "https://")):
        return s
    if os.path.exists(s):
        return s
    if len(s) > 256 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", s):
        return "data:audio/wav;base64," + s
    return s


def decode_upload(data: bytes) -> str:
    """Uploaded audio -> data URI (see normalize_audio_ref for why not raw base64)."""
    return "data:audio/wav;base64," + base64.b64encode(data).decode("ascii")


def load_audio_np(ref: str) -> Tuple[np.ndarray, int]:
    """Resolve a URL / path / base64 / data-URI reference into a waveform."""
    ref = normalize_audio_ref(ref)
    if ref.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(ref) as resp:
            raw = resp.read()
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    elif ref.startswith("data:"):
        raw = base64.b64decode(ref.split(",", 1)[1])
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    else:
        wav, sr = sf.read(ref, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav.astype(np.float32), int(sr)


def transcribe(ref_audio: str) -> str:
    """Speech-to-text for a reference clip. Raises 400 if nothing intelligible."""
    wav, sr = load_audio_np(ref_audio)
    asr = get_asr()
    if ASR_DEVICE == "cpu":
        # No GPU involved, so don't block TTS generation behind it.
        out = asr({"array": wav, "sampling_rate": sr})
    else:
        with gpu_lock:
            out = asr({"array": wav, "sampling_rate": sr})
    text = (out.get("text") or "").strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Could not transcribe the reference audio. Pass ref_text explicitly.",
        )
    return text


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class TTSRequest(GenParams):
    """One endpoint, two modes.

    speaker present -> CustomVoice model, instruct optional (style control)
    speaker absent  -> VoiceDesign model, instruct required (voice description)
    """

    text: Union[str, List[str]] = Field(..., description="Text to speak. Required.")
    speaker: Optional[Union[str, List[str]]] = Field(
        None,
        description="One of the 9 built-in voices (see GET /v1/voices). "
                    "Omit to describe a voice with `instruct` instead.",
    )
    instruct: Optional[Union[str, List[str]]] = Field(
        None,
        description="Voice description. Required when `speaker` is omitted; "
                    "optional style control when `speaker` is given.",
    )
    language: Optional[Union[str, List[str]]] = Field(None, description="Defaults to 'auto'.")
    response_format: ResponseFormat = "json"

    @model_validator(mode="after")
    def _require_instruct_without_speaker(self):
        if not self.speaker and not self.instruct:
            raise ValueError(
                "Provide `speaker` to use a built-in voice, or `instruct` to describe one."
            )
        return self


class CloneRequest(GenParams):
    """Base model: reproduce the voice in a reference clip."""

    text: Union[str, List[str]] = Field(..., description="Text to speak. Required.")
    ref_audio: str = Field(
        ..., description="Reference clip: URL, server path, base64 WAV, or data URI. Required."
    )
    ref_text: Optional[str] = Field(
        None,
        description="Transcript of ref_audio. Omit and the server transcribes it automatically.",
    )
    language: Optional[Union[str, List[str]]] = Field(None, description="Defaults to 'auto'.")
    response_format: ResponseFormat = "json"


class TranscribeRequest(BaseModel):
    audio: str = Field(..., description="URL, server path, base64 WAV, or data URI.")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


# Left open even when a key is set: /health so load balancers and RunPod probes
# can reach it, /demo and the docs so they are usable in a browser.
PUBLIC_PATHS = {"/health", "/demo", "/docs", "/redoc", "/openapi.json"}


def require_api_key(
    request: Request, x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """No-op when QWEN_TTS_API_KEY is unset, so local use needs no header."""
    if API_KEY is None or request.url.path in PUBLIC_PATHS:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"[startup] gpu          : {gpu} ({_TOTAL_VRAM_GB:.1f} GB)")
    print(f"[startup] device       : {DEVICE}")
    print(f"[startup] model dir    : {MODEL_DIR}")
    print(f"[startup] max resident : {MAX_RESIDENT}")
    print(f"[startup] attn impl    : {ATTN}")
    print(f"[startup] asr model    : {ASR_MODEL} on {ASR_DEVICE}")
    print(f"[startup] auth         : {'X-API-Key required' if API_KEY else 'OPEN (no api key)'}")
    print(f"[startup] cors origins : {CORS_ORIGINS}")
    if DEVICE == "cpu":
        print("[startup] WARNING no CUDA device found - generation will be extremely slow.")
    missing = [k for k, p in MODEL_PATHS.items() if not os.path.isdir(p)]
    if missing:
        print(f"[startup] WARNING missing models: {missing}")
    yield


app = FastAPI(
    title="Qwen3-TTS API",
    version="3.0.0",
    description="Text-to-speech with built-in voices, described voices, and voice cloning.",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # set QWEN_TTS_CORS_ORIGINS to your site before going public
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Reference-Text"],
)


# ------------------------------ TTS endpoints ------------------------------ #


@app.post("/v1/tts", tags=["tts"])
def tts(req: TTSRequest):
    """Text to speech.

    Pass `speaker` to use one of the 9 built-in voices (`instruct` then optionally
    restyles it). Omit `speaker` and pass `instruct` to have a voice generated from
    your description. `language` defaults to auto-detect.
    """
    texts = as_list(req.text)
    n = len(texts)

    if req.speaker:
        with gpu_lock:
            model = registry.get("custom_voice")
            t0 = time.time()
            wavs, sr = model.generate_custom_voice(
                text=texts,
                speaker=broadcast(req.speaker, n, "speaker"),
                language=broadcast(req.language or "auto", n, "language"),
                instruct=broadcast(req.instruct, n, "instruct") if req.instruct else None,
                **req.kwargs(),
            )
            elapsed = time.time() - t0
        return audio_response(wavs, sr, req.response_format, elapsed, {"mode": "built_in_voice"})

    with gpu_lock:
        model = registry.get("voice_design")
        t0 = time.time()
        wavs, sr = model.generate_voice_design(
            text=texts,
            instruct=broadcast(req.instruct, n, "instruct"),
            language=broadcast(req.language or "auto", n, "language"),
            **req.kwargs(),
        )
        elapsed = time.time() - t0
    return audio_response(wavs, sr, req.response_format, elapsed, {"mode": "described_voice"})


@app.post("/v1/tts/clone", tags=["tts"])
def tts_clone(req: CloneRequest):
    """Clone the voice in a reference clip.

    `ref_text` is optional - omit it and the reference audio is transcribed
    automatically. The transcript used is returned as `ref_text` in the response
    (or the `X-Reference-Text` header when response_format is 'wav').
    """
    ref_audio = normalize_audio_ref(req.ref_audio)
    ref_text = req.ref_text.strip() if req.ref_text else None
    auto = ref_text is None
    if auto:
        ref_text = transcribe(ref_audio)

    texts = as_list(req.text)
    n = len(texts)
    with gpu_lock:
        model = registry.get("base")
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=texts,
            language=broadcast(req.language or "auto", n, "language"),
            ref_audio=ref_audio,
            ref_text=ref_text,
            **req.kwargs(),
        )
        elapsed = time.time() - t0
    return audio_response(
        wavs, sr, req.response_format, elapsed,
        {"ref_text": ref_text, "ref_text_auto_generated": auto},
    )


@app.post("/v1/tts/clone/upload", tags=["tts"])
async def tts_clone_upload(
    text: str = Form(..., description="Text to speak."),
    ref_audio: UploadFile = File(..., description="Reference audio file."),
    ref_text: Optional[str] = Form(None, description="Optional; transcribed automatically if omitted."),
    language: Optional[str] = Form(None),
    response_format: ResponseFormat = Form("json"),
):
    """Same as /v1/tts/clone but takes a multipart file upload - use this from a browser."""
    data = await ref_audio.read()
    return tts_clone(
        CloneRequest(
            text=text,
            ref_audio=decode_upload(data),
            ref_text=ref_text,
            language=language,
            response_format=response_format,
        )
    )


# ------------------------------ transcription ------------------------------ #


@app.post("/v1/transcribe", tags=["tts"])
def transcribe_json(req: TranscribeRequest):
    """Transcribe audio. Call this when the user uploads a clip so you can show
    them the auto-generated reference text and let them correct it."""
    t0 = time.time()
    text = transcribe(req.audio)
    return {"text": text, "seconds": round(time.time() - t0, 3)}


@app.post("/v1/transcribe/upload", tags=["tts"])
async def transcribe_upload(audio: UploadFile = File(...)):
    """Multipart variant of /v1/transcribe."""
    data = await audio.read()
    t0 = time.time()
    text = transcribe(decode_upload(data))
    return {"text": text, "seconds": round(time.time() - t0, 3)}


# ----------------------------- meta endpoints ------------------------------ #


@app.get("/demo", tags=["meta"], include_in_schema=False)
def demo_page():
    """A small self-contained page exercising both endpoints, including the
    upload -> transcribe -> review -> clone flow. Handy for manual testing."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "demo.html")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="demo.html not found")
    with open(path, encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/html")


@app.get("/health", tags=["meta"])
def health():
    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        }
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "gpu": gpu,
        "resident_models": registry.resident(),
        "asr_loaded": _asr is not None,
        # Auto-tuned at startup - check these to confirm you are on the GPU.
        "config": {
            "device": DEVICE,
            "asr_device": ASR_DEVICE,
            "max_resident": MAX_RESIDENT,
            "attn_implementation": ATTN,
            "auth": "api_key" if API_KEY else "open",
        },
        "models_downloaded": {k: os.path.isdir(p) for k, p in MODEL_PATHS.items()},
    }


@app.get("/v1/voices", tags=["meta"])
def list_voices():
    """The 9 built-in voices. Populate your voice picker from this."""
    return {
        "voices": [
            {"id": sid, "description": desc, "native_language": lang}
            for sid, (desc, lang) in SPEAKER_INFO.items()
        ]
    }


@app.get("/v1/languages", tags=["meta"])
def list_languages():
    """Language dropdown. 'auto' is the default everywhere."""
    return {"languages": LANGUAGES}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("QWEN_TTS_HOST", "127.0.0.1"),
        port=int(os.environ.get("QWEN_TTS_PORT", "9000")),
    )
