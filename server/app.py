"""Qwen3-TTS HTTP API — one endpoint for every mode.

    POST /v1/speak

The request decides which model runs. Exactly one *voice source* may be given:

    voice source        mode              model
    ------------------  ----------------  ---------------------
    (none)              described_voice   VoiceDesign 1.7B   instruct REQUIRED
    speaker             built_in_voice    CustomVoice 1.7B   instruct optional
    voice_id            vocalist_clone    Base 1.7B          curated vocalist clip
    audio               voice_clone       Base 1.7B          user-supplied clip

Supporting endpoints are metadata only: GET /v1/voices (populate the picker) and
GET /health.

Reference transcripts for cloning are produced automatically with Whisper when
not supplied; for vocalist voices the transcript is cached so it is computed once.

Deliberately not exposed, per spec:
  - model size: always 1.7B, the largest released
  - x-vector-only cloning: full ICL cloning is always used

`language` is optional everywhere and defaults to "auto".

Everything auto-tunes to the GPU present, so the same file runs unedited on an
8 GB laptop and an 80 GB A100. Check GET /health -> "config" to see what it chose.

Config via environment variables (all optional):
    QWEN_TTS_MODEL_DIR     model folders (default <repo>/models)
    QWEN_TTS_VOICE_DIR     vocalist clips + voices.json (default <repo>/voices)
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
import json
import os
import re
import secrets
import threading
import time
import urllib.request
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError, model_validator

# Starlette's UploadFile, NOT fastapi.UploadFile: request.form() yields the former,
# and fastapi's is a subclass - so isinstance against fastapi's silently misses.
from starlette.datastructures import UploadFile

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
VOICE_DIR = os.environ.get("QWEN_TTS_VOICE_DIR", os.path.join(REPO_ROOT, "voices"))
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
API_KEY = os.environ.get("QWEN_TTS_API_KEY") or None
CORS_ORIGINS = [
    o.strip() for o in os.environ.get("QWEN_TTS_CORS_ORIGINS", "*").split(",") if o.strip()
]

MODEL_PATHS = {
    "voice_design": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    "custom_voice": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "base": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-Base"),
}

# The 9 voices shipped with the CustomVoice model.
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

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")


# --------------------------------------------------------------------------- #
# Vocalist voice library
# --------------------------------------------------------------------------- #
#
# Curated reference clips you supply, cloned on demand. Drop an audio file into
# VOICE_DIR and it becomes selectable; voices.json only adds display metadata.
# A clip's transcript is needed for high-quality cloning, so it is transcribed
# once on first use and written back to voices.json.

_voice_lib_lock = threading.Lock()


def _manifest_path() -> str:
    return os.path.join(VOICE_DIR, "voices.json")


def _read_manifest() -> Dict[str, Dict[str, Any]]:
    path = _manifest_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[voices] WARNING could not read {path}: {e}")
        return {}
    entries = data.get("voices", data) if isinstance(data, dict) else data
    return {e["id"]: e for e in entries if isinstance(e, dict) and e.get("id")}


def _write_manifest(by_id: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(VOICE_DIR, exist_ok=True)
    tmp = _manifest_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"voices": list(by_id.values())}, f, indent=2, ensure_ascii=False)
    os.replace(tmp, _manifest_path())


def list_vocalists() -> List[Dict[str, Any]]:
    """Every audio file in VOICE_DIR, enriched with any voices.json metadata."""
    if not os.path.isdir(VOICE_DIR):
        return []
    meta = _read_manifest()
    out = []
    for fn in sorted(os.listdir(VOICE_DIR)):
        if not fn.lower().endswith(AUDIO_EXTS):
            continue
        vid = os.path.splitext(fn)[0]
        m = meta.get(vid, {})
        out.append({
            "id": vid,
            "name": m.get("name") or vid.replace("_", " ").title(),
            "description": m.get("description", ""),
            "language": m.get("language", "auto"),
            "file": fn,
            "has_transcript": bool(m.get("ref_text")),
        })
    return out


def vocalist_clip(voice_id: str) -> Tuple[str, Optional[str]]:
    """Resolve a vocalist id to (audio path, cached transcript or None)."""
    safe = os.path.basename(voice_id)
    if not os.path.isdir(VOICE_DIR):
        raise HTTPException(status_code=404, detail=f"No voice library at {VOICE_DIR}.")
    for fn in sorted(os.listdir(VOICE_DIR)):
        if os.path.splitext(fn)[0] == safe and fn.lower().endswith(AUDIO_EXTS):
            meta = _read_manifest().get(safe, {})
            return os.path.join(VOICE_DIR, fn), (meta.get("ref_text") or None)
    available = [v["id"] for v in list_vocalists()]
    raise HTTPException(
        status_code=404,
        detail=f"Unknown voice_id '{voice_id}'. Available: {available or 'none'}",
    )


def cache_vocalist_transcript(voice_id: str, ref_text: str) -> None:
    """Persist a transcript so a vocalist clip is only ever transcribed once."""
    with _voice_lib_lock:
        meta = _read_manifest()
        entry = meta.get(voice_id, {"id": voice_id})
        entry["ref_text"] = ref_text
        meta[voice_id] = entry
        try:
            _write_manifest(meta)
        except OSError as e:  # a read-only volume must not break generation
            print(f"[voices] WARNING could not cache transcript for {voice_id}: {e}")


# --------------------------------------------------------------------------- #
# Model registry - lazy load with LRU eviction so this runs on small GPUs too
# --------------------------------------------------------------------------- #


class ModelRegistry:
    def __init__(self) -> None:
        self._models: "OrderedDict[str, Qwen3TTSModel]" = OrderedDict()
        self._load_lock = threading.Lock()

    def get(self, kind: str) -> Qwen3TTSModel:
        """Fetch a model, loading and evicting as needed.

        MUST be called while holding `gpu_lock`, and the returned model may only
        be used while that lock is still held. Eviction frees the evicted model's
        CUDA storage, so if another request could evict between the get and the
        generate it would pull the weights out from under a running call and raise
        "index is on cpu, different from other tensors on cuda:0".

        Eviction is also fragile on small GPUs: repeatedly loading and freeing a
        ~4 GB model can kill the CUDA process outright on Windows (native crash,
        no traceback). Keeping MAX_RESIDENT >= the number of models in use avoids
        eviction entirely, which is what the auto-tuned default does when there is
        room for it.
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
        """Move weights off the GPU before dropping the reference.

        empty_cache() alone only returns *cached* blocks; the parameters have to
        actually lose their CUDA storage or VRAM keeps climbing until the card
        (or, on Windows, shared system memory) runs out.
        """
        try:
            model.model.to("cpu")
        except Exception as e:
            print(f"[registry] warning: could not move '{kind}' to cpu: {e}")
        print(f"[registry] evicted '{kind}' to free VRAM")

    def resident(self) -> List[str]:
        return list(self._models.keys())


registry = ModelRegistry()

# All GPU work is serialized: HF `generate` blocks, and the models are not safe
# to call concurrently from multiple threads.
gpu_lock = threading.Lock()

# The ASR model is deliberately kept OUT of the LRU pool above: it is small and
# every un-transcribed clone needs it, so evicting a TTS model to make room and
# then evicting it again would thrash.
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


def wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


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
    """Resolve a URL / path / base64 / data-URI reference into a waveform.

    Callers resolve the clip once and pass the waveform on to both transcription
    and generation. Letting the library re-resolve a URL itself would download it
    a second time, and a network failure there escapes as an opaque 500 instead of
    the 502 raised here.
    """
    ref = normalize_audio_ref(ref)
    try:
        if ref.startswith(("http://", "https://")):
            try:
                with urllib.request.urlopen(ref, timeout=60) as resp:
                    raw = resp.read()
            except (urllib.error.URLError, OSError) as e:
                raise HTTPException(
                    status_code=502, detail=f"Could not fetch reference audio from {ref}: {e}"
                ) from e
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        elif ref.startswith("data:"):
            raw = base64.b64decode(ref.split(",", 1)[1])
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        else:
            import librosa

            wav, sr = librosa.load(ref, sr=None, mono=True)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}") from e
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav.astype(np.float32), int(sr)


def transcribe_wav(wav: np.ndarray, sr: int) -> str:
    """Speech-to-text for an already-loaded waveform. 400 if nothing intelligible."""
    asr = get_asr()
    if ASR_DEVICE == "cpu":
        # No GPU involved, so don't block generation behind it.
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
# Request schema
# --------------------------------------------------------------------------- #


class SpeakRequest(BaseModel):
    """The single request shape. Which fields are set decides which model runs."""

    text: Union[str, List[str]] = Field(..., description="Text to speak. Required.")

    # --- voice sources: at most one --------------------------------------- #
    speaker: Optional[Union[str, List[str]]] = Field(
        None, description="A built-in voice id. See GET /v1/voices -> built_in."
    )
    voice_id: Optional[str] = Field(
        None, description="A vocalist voice id to clone. See GET /v1/voices -> vocalist."
    )
    audio: Optional[str] = Field(
        None,
        description="Reference clip to clone: URL, server path, base64, or data URI. "
                    "With multipart, send a file in this field instead.",
    )

    # --- modifiers --------------------------------------------------------- #
    instruct: Optional[Union[str, List[str]]] = Field(
        None,
        description="Voice description (required when no voice source is given) or "
                    "style instruction for a built-in voice. Ignored when cloning.",
    )
    ref_text: Optional[str] = Field(
        None, description="Transcript of the reference clip. Transcribed automatically if omitted."
    )
    language: Optional[Union[str, List[str]]] = Field(None, description="Defaults to 'auto'.")
    response_format: ResponseFormat = "json"

    # --- generation knobs -------------------------------------------------- #
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

    def gen_kwargs(self) -> Dict[str, Any]:
        dumped = self.model_dump()
        return {k: dumped[k] for k in GEN_PARAM_FIELDS if dumped.get(k) is not None}

    @model_validator(mode="after")
    def _one_voice_source(self):
        given = [n for n in ("speaker", "voice_id", "audio") if getattr(self, n)]
        if len(given) > 1:
            raise ValueError(
                f"Give at most one voice source, got {given}. "
                "Use `speaker` for a built-in voice, `voice_id` for a vocalist voice, "
                "`audio` to clone an uploaded clip, or none of them with `instruct` "
                "to have a voice generated from your description."
            )
        if not given and not self.instruct:
            raise ValueError(
                "Provide a voice source (`speaker`, `voice_id`, or `audio`), or "
                "`instruct` to describe the voice you want."
            )
        return self

    @property
    def mode(self) -> str:
        if self.audio:
            return "voice_clone"
        if self.voice_id:
            return "vocalist_clone"
        if self.speaker:
            return "built_in_voice"
        return "described_voice"


# Form fields that are not part of the model but carry the uploaded file.
_FILE_FIELDS = ("audio", "file", "ref_audio")


async def parse_speak_request(request: Request) -> SpeakRequest:
    """Build a SpeakRequest from either a JSON body or multipart form data.

    One endpoint has to accept both: JSON for ordinary calls and batches,
    multipart so a browser can post a file straight from an <input type=file>.
    """
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()

    if ctype == "multipart/form-data":
        form = await request.form()
        payload: Dict[str, Any] = {}
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                if key in _FILE_FIELDS:
                    payload["audio"] = decode_upload(await value.read())
                continue
            if value == "":  # empty form inputs mean "not provided"
                continue
            payload[key] = value
    else:
        try:
            payload = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Body must be JSON, or multipart/form-data when uploading a file.",
            ) from e
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object.")

    try:
        return SpeakRequest(**payload)
    except ValidationError as e:
        # 422 with the same shape FastAPI produces for declared models.
        raise HTTPException(status_code=422, detail=json.loads(e.json())) from e


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

# Left open even when a key is set: /health for load balancers and RunPod probes,
# /demo and the docs so they stay usable in a browser.
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
    print(f"[startup] voice dir    : {VOICE_DIR} ({len(list_vocalists())} vocalist voices)")
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
    version="4.0.0",
    description="One endpoint for described voices, built-in voices, vocalist voices, and cloning.",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # set QWEN_TTS_CORS_ORIGINS to your site before going public
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Mode", "X-Reference-Text"],
)


# ------------------------------ the endpoint ------------------------------- #


@app.post("/v1/speak", tags=["speech"])
async def speak(request: Request):
    """Generate speech. The request decides which model runs.

    | voice source | mode | notes |
    |---|---|---|
    | none | `described_voice` | `instruct` required - the voice is invented from it |
    | `speaker` | `built_in_voice` | one of the 9 built-ins; `instruct` optionally restyles |
    | `voice_id` | `vocalist_clone` | clones a curated vocalist clip |
    | `audio` | `voice_clone` | clones the supplied clip (JSON string or multipart file) |

    Accepts `application/json` or `multipart/form-data`.
    """
    req = await parse_speak_request(request)
    texts = as_list(req.text)
    n = len(texts)
    mode = req.mode
    warnings: List[str] = []
    info: Dict[str, Any] = {}

    # --- resolve the reference clip for the two cloning modes -------------- #
    ref_clip: Optional[Tuple[np.ndarray, int]] = None
    ref_text: Optional[str] = req.ref_text.strip() if req.ref_text else None

    if mode in ("vocalist_clone", "voice_clone"):
        if mode == "vocalist_clone":
            path, cached = vocalist_clip(req.voice_id)
            source, ref_text = path, (ref_text or cached)
            info["voice_id"] = req.voice_id
        else:
            source = normalize_audio_ref(req.audio)

        if req.instruct:
            warnings.append(
                "`instruct` is not supported when cloning a voice and was ignored."
            )

        # Resolved once here and handed to both steps below.
        ref_clip = load_audio_np(source)

        if not ref_text:
            ref_text = transcribe_wav(*ref_clip)
            info["ref_text_auto_generated"] = True
            if mode == "vocalist_clone":
                cache_vocalist_transcript(req.voice_id, ref_text)
        else:
            info["ref_text_auto_generated"] = False
        info["ref_text"] = ref_text

    # --- generate ----------------------------------------------------------- #
    languages = broadcast(req.language or "auto", n, "language")
    gen = req.gen_kwargs()

    with gpu_lock:
        t0 = time.time()
        if mode == "built_in_voice":
            model = registry.get("custom_voice")
            wavs, sr = model.generate_custom_voice(
                text=texts,
                speaker=broadcast(req.speaker, n, "speaker"),
                language=languages,
                instruct=broadcast(req.instruct, n, "instruct") if req.instruct else None,
                **gen,
            )
        elif mode == "described_voice":
            model = registry.get("voice_design")
            wavs, sr = model.generate_voice_design(
                text=texts,
                instruct=broadcast(req.instruct, n, "instruct"),
                language=languages,
                **gen,
            )
        else:
            model = registry.get("base")
            wavs, sr = model.generate_voice_clone(
                text=texts,
                language=languages,
                ref_audio=ref_clip,   # (waveform, sample_rate), already resolved
                ref_text=ref_text,
                **gen,
            )
        elapsed = time.time() - t0

    # --- respond ------------------------------------------------------------ #
    if req.response_format == "wav":
        if len(wavs) != 1:
            raise HTTPException(
                status_code=400,
                detail="response_format='wav' returns a single file; use 'json' for batches.",
            )
        headers = {"X-Mode": mode}
        if info.get("ref_text"):
            # Header values must be latin-1; fall back rather than 500 on non-ASCII.
            try:
                info["ref_text"].encode("latin-1")
                headers["X-Reference-Text"] = info["ref_text"]
            except UnicodeEncodeError:
                pass
        return Response(content=wav_bytes(wavs[0], sr), media_type="audio/wav", headers=headers)

    payload: Dict[str, Any] = {
        "mode": mode,
        "sample_rate": sr,
        "count": len(wavs),
        "durations": [round(len(w) / sr, 3) for w in wavs],
        "generation_seconds": round(elapsed, 3),
        "audio": [base64.b64encode(wav_bytes(w, sr)).decode("ascii") for w in wavs],
        **info,
    }
    if warnings:
        payload["warnings"] = warnings
    return JSONResponse(payload)


# ----------------------------- metadata ------------------------------------ #


@app.get("/v1/voices", tags=["meta"])
def list_voices():
    """Everything the voice picker needs: built-in voices, vocalist voices, languages."""
    return {
        "built_in": [
            {"id": sid, "name": sid.replace("_", " ").title(),
             "description": desc, "native_language": lang}
            for sid, (desc, lang) in SPEAKER_INFO.items()
        ],
        "vocalist": list_vocalists(),
        "languages": LANGUAGES,
    }


@app.get("/demo", tags=["meta"], include_in_schema=False)
def demo_page():
    """Self-contained page exercising every mode. Handy for manual testing."""
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
        "vocalist_voices": len(list_vocalists()),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("QWEN_TTS_HOST", "127.0.0.1"),
        port=int(os.environ.get("QWEN_TTS_PORT", "9000")),
    )
