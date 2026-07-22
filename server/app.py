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

import asyncio
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
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
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

# --- batching -------------------------------------------------------------- #
# Decoding one step reads the whole 3.4 GB of weights out of VRAM whether it is
# producing for one caller or twenty, so generating 20 clips together costs barely
# more than generating one. Requests are therefore queued and merged into single
# model calls. Measured on an 8 GB laptop, batch 16 did ~6.8x the work of batch 1
# and was still scaling, so the ceiling here is well above the default.
MAX_BATCH_ITEMS = int(os.environ.get("QWEN_TTS_MAX_BATCH") or (16 if _TOTAL_VRAM_GB >= 24 else 4))
# How long to hold the first request waiting for company. Only felt when idle -
# under load the queue is already full when a batch finishes, so batches form for
# free. Keep it far below generation time (seconds) or every caller pays for it.
BATCH_WINDOW_S = float(os.environ.get("QWEN_TTS_BATCH_WINDOW_MS") or 40) / 1000.0

# --- guardrails ------------------------------------------------------------ #
# One caller pasting a novel would otherwise hold the GPU for minutes and stall
# everyone else, with no way to shed load.
MAX_ITEMS_PER_REQUEST = int(os.environ.get("QWEN_TTS_MAX_ITEMS") or 32)
MAX_CHARS_PER_ITEM = int(os.environ.get("QWEN_TTS_MAX_CHARS") or 3000)
MAX_QUEUE_ITEMS = int(os.environ.get("QWEN_TTS_MAX_QUEUE") or 128)
REQUEST_TIMEOUT_S = float(os.environ.get("QWEN_TTS_TIMEOUT") or 600)
WARMUP = os.environ.get("QWEN_TTS_WARMUP", "").lower() in ("1", "true", "yes")

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
_asr_lock = threading.Lock()          # guards loading
_asr_infer_lock = threading.Lock()    # guards inference (one Whisper call at a time)


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
    # Its own lock, deliberately not gpu_lock: Whisper is a different model, and
    # PyTorch is happy running it alongside a TTS batch. Sharing gpu_lock would
    # stall transcription behind every generation (and vice versa) for no reason.
    with _asr_infer_lock:
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

# The endpoint reads the raw Request so it can accept JSON *and* multipart on one
# path, which means FastAPI cannot infer a body schema. Supply it by hand, with a
# runnable example per mode so "Try it out" in /docs works without typing.
DEMO_REF_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"

SPEAK_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": SpeakRequest.model_json_schema(),
                "examples": {
                    "described_voice": {
                        "summary": "1. Describe a voice (no voice source)",
                        "description": "No speaker, no clip - the voice is invented from "
                                       "`instruct`, which is required in this mode.",
                        "value": {
                            "text": "Welcome, traveller. You have journeyed far.",
                            "instruct": "An elderly wizard, deep gravelly voice, slow and wise.",
                            "language": "english",
                            "response_format": "wav",
                        },
                    },
                    "built_in_voice": {
                        "summary": "2. Built-in voice",
                        "description": "One of the 9 shipped voices. See GET /v1/voices.",
                        "value": {
                            "text": "Hello, this is the Qwen3 text to speech engine speaking.",
                            "speaker": "ryan",
                            "language": "english",
                            "response_format": "wav",
                        },
                    },
                    "built_in_voice_restyled": {
                        "summary": "3. Built-in voice + style instruction",
                        "description": "`instruct` is optional here and restyles the speaker.",
                        "value": {
                            "text": "I cannot believe you did that. This is unacceptable.",
                            "speaker": "vivian",
                            "instruct": "Speak in a furious, sharp tone.",
                            "language": "english",
                            "response_format": "wav",
                        },
                    },
                    "vocalist_clone": {
                        "summary": "4. Vocalist voice from the library",
                        "description": "Clones a curated clip. Ids come from "
                                       "GET /v1/voices -> vocalist.",
                        "value": {
                            "text": "This line is spoken by a voice from the library.",
                            "voice_id": "aria_soul",
                            "language": "english",
                            "response_format": "wav",
                        },
                    },
                    "voice_clone_url": {
                        "summary": "5. Clone a clip by URL",
                        "description": "`ref_text` omitted, so the clip is transcribed "
                                       "automatically. Add it when you know the exact words.",
                        "value": {
                            "text": "This sentence is spoken in a cloned voice.",
                            "audio": DEMO_REF_URL,
                            "language": "english",
                            "response_format": "wav",
                        },
                    },
                    "batch": {
                        "summary": "6. Batch, one voice per line",
                        "description": "`text` as a list. Other fields take one value or a "
                                       "list of matching length. JSON only.",
                        "value": {
                            "text": ["The first sentence is spoken here.",
                                     "And the second follows after it."],
                            "speaker": ["ryan", "aiden"],
                            "language": "english",
                        },
                    },
                    "raw_wav_response": {
                        "summary": "7. Return raw WAV instead of JSON",
                        "description": "Returns audio/wav bytes with the mode in an X-Mode "
                                       "header. Single item only.",
                        "value": {
                            "text": "This response is raw wav bytes, not JSON.",
                            "speaker": "aiden",
                            "language": "english",
                            "response_format": "wav",
                        },
                    },
                },
            },
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["text", "audio"],
                    "properties": {
                        "text": {"type": "string", "title": "Text to speak",
                                 "example": "Cloned from an uploaded file."},
                        "audio": {"type": "string", "format": "binary",
                                  "title": "Reference clip to clone"},
                        "ref_text": {"type": "string", "title": "Transcript (optional)",
                                     "description": "Omit and the clip is transcribed for you."},
                        "language": {"type": "string", "default": "auto", "enum": LANGUAGES},
                        "response_format": {"type": "string", "default": "json",
                                            "enum": ["json", "wav"]},
                    },
                },
            },
        },
    },
}


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
    print(f"[startup] batching     : up to {MAX_BATCH_ITEMS} items, "
          f"{BATCH_WINDOW_S * 1000:.0f}ms window")
    print(f"[startup] limits       : {MAX_ITEMS_PER_REQUEST} items/req, "
          f"{MAX_CHARS_PER_ITEM} chars/item, queue {MAX_QUEUE_ITEMS}, "
          f"timeout {REQUEST_TIMEOUT_S:.0f}s")
    if DEVICE == "cpu":
        print("[startup] WARNING no CUDA device found - generation will be extremely slow.")
    missing = [k for k, p in MODEL_PATHS.items() if not os.path.isdir(p)]
    if missing:
        print(f"[startup] WARNING missing models: {missing}")
    if MAX_RESIDENT < len([k for k in MODEL_PATHS if os.path.isdir(MODEL_PATHS[k])]):
        print(f"[startup] NOTE max_resident={MAX_RESIDENT} is below the number of models, "
              "so switching modes will evict and reload (10-30s). Fine on a small GPU, "
              "but set QWEN_TTS_MAX_RESIDENT=3 where there is VRAM for it.")

    worker = asyncio.create_task(batch_worker())

    if WARMUP:
        # Load every model up front so the first real caller does not eat the
        # 10-30s load. Only sane when they all stay resident.
        def _load(kind: str):
            with gpu_lock:               # registry.get requires it
                registry.get(kind)

        for kind in MODEL_PATHS:
            if os.path.isdir(MODEL_PATHS[kind]):
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        gpu_executor, _load, kind
                    )
                except Exception as e:
                    print(f"[startup] warmup of '{kind}' failed: {e}")
        print(f"[startup] warmed up  : {registry.resident()}")

    try:
        yield
    finally:
        worker.cancel()


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


# --------------------------------------------------------------------------- #
# Batching engine
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    """One in-flight /v1/speak request, waiting for the batch worker."""

    mode: str
    model_kind: str
    gen_key: Tuple            # requests with different sampling knobs cannot merge
    gen_kwargs: Dict[str, Any]
    texts: List[str]
    speakers: List[Optional[str]]
    instructs: List[str]
    languages: List[str]
    ref_clip: Optional[Tuple[np.ndarray, int]]
    ref_text: Optional[str]
    info: Dict[str, Any]
    warnings: List[str]
    future: "asyncio.Future"
    queued_at: float

    @property
    def n(self) -> int:
        return len(self.texts)


class Metrics:
    """Just enough to answer 'is the queue backing up?' from /health."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.waits = deque(maxlen=200)       # queue wait, seconds
        self.batch_sizes = deque(maxlen=200)
        self.completed = 0
        self.failed = 0
        self.rejected = 0
        self.generating = 0

    def record(self, wait: float) -> None:
        with self.lock:
            self.waits.append(wait)

    def snapshot(self, queued_items: int) -> Dict[str, Any]:
        with self.lock:
            waits = sorted(self.waits)
            sizes = list(self.batch_sizes)

        def pct(p):
            if not waits:
                return 0.0
            return round(waits[min(len(waits) - 1, int(len(waits) * p))], 2)

        return {
            "generating_items": self.generating,
            "queued_items": queued_items,
            "queue_wait_p50_seconds": pct(0.50),
            "queue_wait_p95_seconds": pct(0.95),
            "avg_batch_size": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
            "max_batch_size_seen": max(sizes) if sizes else 0,
            "completed": self.completed,
            "failed": self.failed,
            "rejected_overload": self.rejected,
        }


metrics = Metrics()
job_queue: "asyncio.Queue[Job]" = asyncio.Queue()
_queued_items = 0                 # items (not requests) currently waiting
_queued_lock = threading.Lock()

# Generation gets its own single thread rather than the default executor. Reference
# downloads and transcriptions also run in threads, and if they filled the shared
# pool the batch worker could be left with no thread to run on.
gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-gpu")


def _adjust_queued(delta: int) -> int:
    global _queued_items
    with _queued_lock:
        _queued_items += delta
        return _queued_items


def run_group(jobs: List[Job]) -> List[Tuple[Job, Any]]:
    """Generate for one compatible group in a single model call. Blocking - runs
    in a worker thread.

    Compatible means: same model, same sampling kwargs. Every job's items are
    flattened into one list, generated together, then sliced back apart.
    """
    texts, speakers, instructs, languages, ref_audios, ref_texts = [], [], [], [], [], []
    for j in jobs:
        texts += j.texts
        speakers += j.speakers
        instructs += j.instructs
        languages += j.languages
        if j.model_kind == "base":
            ref_audios += [j.ref_clip] * j.n
            ref_texts += [j.ref_text] * j.n

    kind = jobs[0].model_kind
    gen = jobs[0].gen_kwargs

    with gpu_lock:
        metrics.generating = len(texts)
        t0 = time.time()
        try:
            model = registry.get(kind)
            if kind == "custom_voice":
                wavs, sr = model.generate_custom_voice(
                    text=texts, speaker=speakers, language=languages,
                    instruct=instructs, **gen,
                )
            elif kind == "voice_design":
                wavs, sr = model.generate_voice_design(
                    text=texts, instruct=instructs, language=languages, **gen,
                )
            else:
                wavs, sr = model.generate_voice_clone(
                    text=texts, language=languages,
                    ref_audio=ref_audios, ref_text=ref_texts, **gen,
                )
        finally:
            metrics.generating = 0
        elapsed = time.time() - t0

    with metrics.lock:
        metrics.batch_sizes.append(len(texts))

    out, cursor = [], 0
    for j in jobs:
        out.append((j, (wavs[cursor:cursor + j.n], sr, elapsed)))
        cursor += j.n
    return out


def run_group_isolated(jobs: List[Job]) -> List[Tuple[Job, Any]]:
    """Run a group, and if it blows up, retry each job alone.

    One malformed request must not fail everyone batched alongside it, so the
    fallback narrows the blast radius to the job that actually breaks.
    """
    try:
        return run_group(jobs)
    except Exception as e:
        if len(jobs) == 1:
            return [(jobs[0], e)]
        print(f"[batch] group of {len(jobs)} failed ({e}); retrying individually")
        out = []
        for j in jobs:
            try:
                out += run_group([j])
            except Exception as inner:
                out.append((j, inner))
        return out


def _group_key(job: Job) -> Tuple:
    return (job.model_kind, job.gen_key)


async def batch_worker() -> None:
    """Collect queued jobs, merge compatible ones, and run them on the GPU."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            first = await job_queue.get()
            batch, items = [first], first.n
            deadline = loop.time() + BATCH_WINDOW_S

            # Fill the window. Under load the queue is already populated, so this
            # returns immediately and batches form without anyone waiting.
            while items < MAX_BATCH_ITEMS:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(job_queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                batch.append(nxt)
                items += nxt.n

            groups: Dict[Tuple, List[Job]] = defaultdict(list)
            for j in batch:
                groups[_group_key(j)].append(j)

            # Run whichever model is already resident first. With MAX_RESIDENT
            # below the number of models, this avoids an eviction + 10-30s reload
            # in the middle of serving.
            resident = set(registry.resident())
            for key in sorted(groups, key=lambda k: k[0] not in resident):
                jobs = groups[key]
                for j in jobs:
                    metrics.record(time.time() - j.queued_at)
                _adjust_queued(-sum(j.n for j in jobs))
                results = await loop.run_in_executor(gpu_executor, run_group_isolated, jobs)
                for job, result in results:
                    if job.future.done():
                        continue          # client disconnected
                    if isinstance(result, Exception):
                        metrics.failed += 1
                        job.future.set_exception(result)
                    else:
                        metrics.completed += 1
                        job.future.set_result(result)
        except asyncio.CancelledError:
            raise
        except Exception as e:                      # the worker must never die
            print(f"[batch] worker error: {e}")


# ------------------------------ the endpoint ------------------------------- #


@app.post("/v1/speak", tags=["speech"], openapi_extra=SPEAK_OPENAPI)
async def speak(request: Request):
    """Generate speech. The request decides which model runs.

    | voice source | mode | notes |
    |---|---|---|
    | none | `described_voice` | `instruct` required - the voice is invented from it |
    | `speaker` | `built_in_voice` | one of the 9 built-ins; `instruct` optionally restyles |
    | `voice_id` | `vocalist_clone` | clones a curated vocalist clip |
    | `audio` | `voice_clone` | clones the supplied clip (JSON string or multipart file) |

    Accepts `application/json` or `multipart/form-data`. Requests are batched with
    others arriving at the same time, so concurrent callers share one model pass.
    """
    loop = asyncio.get_running_loop()
    req = await parse_speak_request(request)
    texts = as_list(req.text)
    n = len(texts)
    mode = req.mode
    warnings: List[str] = []
    info: Dict[str, Any] = {}

    # --- guardrails --------------------------------------------------------- #
    if n > MAX_ITEMS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"{n} items in one request exceeds the limit of {MAX_ITEMS_PER_REQUEST}. "
                   "Split it into several requests - that also lets other callers interleave.",
        )
    for t in texts:
        if len(t) > MAX_CHARS_PER_ITEM:
            raise HTTPException(
                status_code=400,
                detail=f"A text item is {len(t)} characters, over the {MAX_CHARS_PER_ITEM} limit.",
            )

    if _queued_items + n > MAX_QUEUE_ITEMS:
        metrics.rejected += 1
        raise HTTPException(
            status_code=503,
            detail="Server is at capacity, please retry shortly.",
            headers={"Retry-After": "5"},
        )

    # --- resolve the reference clip for the two cloning modes --------------- #
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

        # Fetching and transcribing happen off the event loop, and outside the
        # GPU queue, so a slow download never blocks generation for others.
        ref_clip = await loop.run_in_executor(None, load_audio_np, source)

        if not ref_text:
            ref_text = await loop.run_in_executor(None, transcribe_wav, *ref_clip)
            info["ref_text_auto_generated"] = True
            if mode == "vocalist_clone":
                cache_vocalist_transcript(req.voice_id, ref_text)
        else:
            info["ref_text_auto_generated"] = False
        info["ref_text"] = ref_text

    # --- queue it ------------------------------------------------------------ #
    gen = req.gen_kwargs()
    job = Job(
        mode=mode,
        model_kind={"built_in_voice": "custom_voice",
                    "described_voice": "voice_design"}.get(mode, "base"),
        gen_key=tuple(sorted(gen.items())),
        gen_kwargs=gen,
        texts=texts,
        speakers=broadcast(req.speaker, n, "speaker"),
        # "" rather than None: the library treats an empty instruction as none,
        # and a list mixing None with strings is not safe to batch.
        instructs=[i or "" for i in broadcast(req.instruct, n, "instruct")],
        languages=broadcast(req.language or "auto", n, "language"),
        ref_clip=ref_clip,
        ref_text=ref_text,
        info=info,
        warnings=warnings,
        future=loop.create_future(),
        queued_at=time.time(),
    )

    _adjust_queued(n)
    await job_queue.put(job)
    try:
        wavs, sr, elapsed = await asyncio.wait_for(job.future, REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        job.future.cancel()
        raise HTTPException(
            status_code=504,
            detail=f"Generation exceeded {REQUEST_TIMEOUT_S:.0f}s. Try shorter text.",
        ) from None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e

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
        # Watch queued_items and queue_wait_p95_seconds. Consistently near zero
        # means batching has nothing to merge and the GPU is keeping up.
        "load": metrics.snapshot(_queued_items),
        # Auto-tuned at startup - check these to confirm you are on the GPU.
        "config": {
            "device": DEVICE,
            "asr_device": ASR_DEVICE,
            "max_resident": MAX_RESIDENT,
            "attn_implementation": ATTN,
            "auth": "api_key" if API_KEY else "open",
            "max_batch_items": MAX_BATCH_ITEMS,
            "batch_window_ms": round(BATCH_WINDOW_S * 1000),
            "max_items_per_request": MAX_ITEMS_PER_REQUEST,
            "max_queue_items": MAX_QUEUE_ITEMS,
            "request_timeout_seconds": REQUEST_TIMEOUT_S,
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
