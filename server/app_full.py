"""Qwen3-TTS HTTP API.

Exposes every capability of the Qwen3-TTS 1.7B model family behind one origin:
custom-voice TTS, voice design, voice cloning, reusable saved voices, and raw
codec encode/decode.

Config via environment variables:
    QWEN_TTS_MODEL_DIR    parent dir holding the three model folders
    QWEN_TTS_MAX_RESIDENT how many models may sit on the GPU at once (default 1)
    QWEN_TTS_ATTN         attn implementation (default sdpa; use flash_attention_2 on Linux)
    QWEN_TTS_DEVICE       device_map target (default cuda:0)
    QWEN_TTS_VOICE_DIR    where saved clone voices are persisted
"""

from __future__ import annotations

import base64
import gc
import io
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import soundfile as sf
import torch
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("QWEN_TTS_MODEL_DIR", os.path.join(REPO_ROOT, "models"))
VOICE_DIR = os.environ.get("QWEN_TTS_VOICE_DIR", os.path.join(REPO_ROOT, "saved_voices"))
DEVICE = os.environ.get("QWEN_TTS_DEVICE", "cuda:0")
ATTN = os.environ.get("QWEN_TTS_ATTN", "sdpa")
MAX_RESIDENT = int(os.environ.get("QWEN_TTS_MAX_RESIDENT", "1"))

MODEL_PATHS = {
    "custom_voice": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "voice_design": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    "base": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-Base"),
}

# Speaker descriptions from the model card, surfaced through /v1/voices.
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

os.makedirs(VOICE_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model registry - lazy load with LRU eviction so this runs on small GPUs too
# --------------------------------------------------------------------------- #


class ModelRegistry:
    def __init__(self) -> None:
        self._models: "OrderedDict[str, Qwen3TTSModel]" = OrderedDict()
        self._load_lock = threading.Lock()
        self.load_times: Dict[str, float] = {}

    def get(self, kind: str) -> Qwen3TTSModel:
        with self._load_lock:
            if kind in self._models:
                self._models.move_to_end(kind)
                return self._models[kind]

            path = MODEL_PATHS[kind]
            if not os.path.isdir(path):
                raise HTTPException(
                    status_code=503,
                    detail=f"Model '{kind}' is not downloaded at {path}.",
                )

            while len(self._models) >= MAX_RESIDENT:
                evicted_kind, evicted_model = self._models.popitem(last=False)
                self._release(evicted_kind, evicted_model)
                del evicted_model
                gc.collect()
                torch.cuda.empty_cache()

            print(f"[registry] loading '{kind}' from {path}")
            t0 = time.time()
            model = Qwen3TTSModel.from_pretrained(
                path, device_map=DEVICE, dtype=torch.bfloat16, attn_implementation=ATTN
            )
            self.load_times[kind] = time.time() - t0
            print(f"[registry] loaded '{kind}' in {self.load_times[kind]:.1f}s")
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

# All GPU work is serialized. HF `generate` is blocking and the models are not
# safe to call concurrently from multiple threads, so one request runs at a time.
gpu_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

GEN_PARAM_FIELDS = (
    "max_new_tokens",
    "do_sample",
    "top_k",
    "top_p",
    "temperature",
    "repetition_penalty",
    "subtalker_dosample",
    "subtalker_top_k",
    "subtalker_top_p",
    "subtalker_temperature",
)


class GenParams(BaseModel):
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
        # are passed positionally, so leaking them here would duplicate them.
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
    """Return either raw WAV bytes (single item only) or JSON with base64 audio."""
    if response_format == "wav":
        if len(wavs) != 1:
            raise HTTPException(
                status_code=400,
                detail="response_format='wav' requires exactly one text item; use 'json' for batches.",
            )
        return Response(content=wav_bytes(wavs[0], sr), media_type="audio/wav")

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


def decode_upload(data: bytes) -> str:
    """Uploaded audio -> data URI (see normalize_audio_ref for why not raw base64)."""
    return "data:audio/wav;base64," + base64.b64encode(data).decode("ascii")


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


def design_reference_clip(
    instruct: str, ref_text: str, language: Optional[str], gen_kwargs: Dict[str, Any]
):
    """Synthesize a reference clip with the VoiceDesign model.

    Deliberately its own function: the local reference to the design model has to
    go out of scope before the Base model is requested, otherwise eviction cannot
    free it and both models sit on the GPU at once (fatal when MAX_RESIDENT=1).
    """
    design = registry.get("voice_design")
    with gpu_lock:
        wavs, sr = design.generate_voice_design(
            text=[ref_text],
            instruct=[instruct],
            language=[language or "auto"],
            **gen_kwargs,
        )
    return wavs[0], sr


def voice_path(voice_id: str) -> str:
    safe = os.path.basename(voice_id)
    return os.path.join(VOICE_DIR, f"{safe}.pt")


def save_voice(voice_id: str, items: List[VoiceClonePromptItem], meta: Dict[str, Any]) -> None:
    torch.save(
        {"items": [asdict(i) for i in items], "meta": meta},
        voice_path(voice_id),
    )


def load_voice(voice_id: str) -> List[VoiceClonePromptItem]:
    path = voice_path(voice_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Saved voice '{voice_id}' not found.")
    blob = torch.load(path, weights_only=False)
    return [VoiceClonePromptItem(**d) for d in blob["items"]]


def voice_meta(voice_id: str) -> Dict[str, Any]:
    blob = torch.load(voice_path(voice_id), weights_only=False)
    return blob.get("meta", {})


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

ResponseFormat = Literal["json", "wav"]


class CustomVoiceRequest(GenParams):
    text: Union[str, List[str]] = Field(..., description="Text to speak; string or list for batch.")
    speaker: Union[str, List[str]] = Field("ryan", description="One of the 9 built-in speakers.")
    language: Optional[Union[str, List[str]]] = Field(
        None, description="Language name, or omit / 'auto' to auto-detect."
    )
    instruct: Optional[Union[str, List[str]]] = Field(
        None, description="Natural-language style control, e.g. 'speak very angrily'."
    )
    response_format: ResponseFormat = "json"


class VoiceDesignRequest(GenParams):
    text: Union[str, List[str]]
    instruct: Union[str, List[str]] = Field(..., description="Description of the voice to invent.")
    language: Optional[Union[str, List[str]]] = None
    response_format: ResponseFormat = "json"


class CloneRequest(GenParams):
    text: Union[str, List[str]]
    ref_audio: str = Field(..., description="Reference clip: URL, local path, or base64 WAV.")
    ref_text: Optional[str] = Field(
        None, description="Transcript of ref_audio. Required unless x_vector_only_mode is true."
    )
    language: Optional[Union[str, List[str]]] = None
    x_vector_only_mode: bool = Field(
        False, description="Use speaker embedding only; no transcript needed, lower fidelity."
    )
    response_format: ResponseFormat = "json"


class SavedVoiceTTSRequest(GenParams):
    text: Union[str, List[str]]
    voice_id: str
    language: Optional[Union[str, List[str]]] = None
    response_format: ResponseFormat = "json"


class CreateCloneVoiceRequest(BaseModel):
    ref_audio: str
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False
    name: Optional[str] = None


class CreateDesignedVoiceRequest(GenParams):
    instruct: str = Field(..., description="Description of the voice to invent and then persist.")
    ref_text: str = Field(
        "Hello there, this is a quick sample of how this particular voice sounds when speaking naturally.",
        description="Sentence spoken to create the reference clip.",
    )
    language: Optional[str] = None
    name: Optional[str] = None


class DesignThenCloneRequest(GenParams):
    text: Union[str, List[str]]
    instruct: str
    ref_text: str = "Hello there, this is a quick sample of how this particular voice sounds when speaking naturally."
    language: Optional[Union[str, List[str]]] = None
    response_format: ResponseFormat = "json"


class EncodeRequest(BaseModel):
    audio: Union[str, List[str]] = Field(..., description="URL, path, or base64 WAV.")


class DecodeRequest(BaseModel):
    codes: List[List[List[int]]] = Field(..., description="audio_codes from /v1/codec/encode.")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] model dir      : {MODEL_DIR}")
    print(f"[startup] max resident   : {MAX_RESIDENT}")
    print(f"[startup] attn impl      : {ATTN}")
    print(f"[startup] available      : {[k for k, p in MODEL_PATHS.items() if os.path.isdir(p)]}")
    yield


app = FastAPI(
    title="Qwen3-TTS API",
    version="1.0.0",
    description="All Qwen3-TTS 1.7B capabilities behind one HTTP origin.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your site's domain before going public
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------- meta endpoints ------------------------------ #


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
        "max_resident": MAX_RESIDENT,
    }


@app.get("/v1/models", tags=["meta"])
def list_models():
    return {
        "models": [
            {
                "id": kind,
                "path": path,
                "downloaded": os.path.isdir(path),
                "loaded": kind in registry.resident(),
                "tasks": {
                    "custom_voice": ["/v1/tts/custom-voice"],
                    "voice_design": ["/v1/tts/voice-design", "/v1/voices/designed"],
                    "base": ["/v1/tts/clone", "/v1/voices/cloned", "/v1/codec/encode"],
                }[kind],
            }
            for kind, path in MODEL_PATHS.items()
        ]
    }


@app.get("/v1/voices", tags=["meta"])
def list_builtin_voices():
    return {
        "voices": [
            {"id": sid, "description": desc, "native_language": lang}
            for sid, (desc, lang) in SPEAKER_INFO.items()
        ]
    }


@app.get("/v1/languages", tags=["meta"])
def list_languages():
    return {
        "languages": [
            "auto", "chinese", "english", "french", "german", "italian",
            "japanese", "korean", "portuguese", "russian", "spanish",
        ]
    }


# ------------------------------ TTS endpoints ------------------------------ #


@app.post("/v1/tts/custom-voice", tags=["tts"])
def tts_custom_voice(req: CustomVoiceRequest):
    """Built-in speaker TTS. `instruct` is optional style control."""
    texts = as_list(req.text)
    n = len(texts)
    model = registry.get("custom_voice")
    with gpu_lock:
        t0 = time.time()
        wavs, sr = model.generate_custom_voice(
            text=texts,
            speaker=broadcast(req.speaker, n, "speaker"),
            language=broadcast(req.language or "auto", n, "language"),
            instruct=broadcast(req.instruct, n, "instruct") if req.instruct else None,
            **req.kwargs(),
        )
        elapsed = time.time() - t0
    return audio_response(wavs, sr, req.response_format, elapsed)


@app.post("/v1/tts/voice-design", tags=["tts"])
def tts_voice_design(req: VoiceDesignRequest):
    """Invent a voice from a natural-language description and speak with it."""
    texts = as_list(req.text)
    n = len(texts)
    model = registry.get("voice_design")
    with gpu_lock:
        t0 = time.time()
        wavs, sr = model.generate_voice_design(
            text=texts,
            instruct=broadcast(req.instruct, n, "instruct"),
            language=broadcast(req.language or "auto", n, "language"),
            **req.kwargs(),
        )
        elapsed = time.time() - t0
    return audio_response(wavs, sr, req.response_format, elapsed)


@app.post("/v1/tts/clone", tags=["tts"])
def tts_clone(req: CloneRequest):
    """One-shot voice clone from a reference clip supplied as URL / path / base64."""
    if not req.x_vector_only_mode and not req.ref_text:
        raise HTTPException(
            status_code=400,
            detail="ref_text is required unless x_vector_only_mode is true.",
        )
    texts = as_list(req.text)
    n = len(texts)
    model = registry.get("base")
    with gpu_lock:
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=texts,
            language=broadcast(req.language or "auto", n, "language"),
            ref_audio=normalize_audio_ref(req.ref_audio),
            ref_text=req.ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
            **req.kwargs(),
        )
        elapsed = time.time() - t0
    return audio_response(wavs, sr, req.response_format, elapsed)


@app.post("/v1/tts/clone/upload", tags=["tts"])
async def tts_clone_upload(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    x_vector_only_mode: bool = Form(False),
    response_format: ResponseFormat = Form("json"),
):
    """Same as /v1/tts/clone but takes a multipart file upload."""
    data = await ref_audio.read()
    req = CloneRequest(
        text=text,
        ref_audio=decode_upload(data),
        ref_text=ref_text,
        language=language,
        x_vector_only_mode=x_vector_only_mode,
        response_format=response_format,
    )
    return tts_clone(req)


@app.post("/v1/tts/saved-voice", tags=["tts"])
def tts_saved_voice(req: SavedVoiceTTSRequest):
    """Speak using a previously registered voice, without re-sending reference audio."""
    items = load_voice(req.voice_id)
    texts = as_list(req.text)
    n = len(texts)
    model = registry.get("base")
    prompt = items * n if len(items) == 1 else items
    with gpu_lock:
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=texts,
            language=broadcast(req.language or "auto", n, "language"),
            voice_clone_prompt=prompt,
            **req.kwargs(),
        )
        elapsed = time.time() - t0
    return audio_response(wavs, sr, req.response_format, elapsed, {"voice_id": req.voice_id})


@app.post("/v1/tts/design-then-clone", tags=["tts"])
def tts_design_then_clone(req: DesignThenCloneRequest):
    """Design a voice, then immediately speak arbitrary text in it (uses both models)."""
    t0 = time.time()
    ref_wav, ref_sr = design_reference_clip(
        req.instruct,
        req.ref_text,
        req.language if isinstance(req.language, str) else None,
        {},
    )
    base = registry.get("base")
    texts = as_list(req.text)
    n = len(texts)
    with gpu_lock:
        items = base.create_voice_clone_prompt(
            ref_audio=(ref_wav, ref_sr), ref_text=req.ref_text
        )
        wavs, sr = base.generate_voice_clone(
            text=texts,
            language=broadcast(req.language or "auto", n, "language"),
            voice_clone_prompt=items * n if len(items) == 1 else items,
            **req.kwargs(),
        )
        elapsed = time.time() - t0
    return audio_response(
        wavs, sr, req.response_format, elapsed,
        {"reference_audio": base64.b64encode(wav_bytes(ref_wav, ref_sr)).decode("ascii")},
    )


# --------------------------- saved voice management ------------------------ #


@app.post("/v1/voices/cloned", tags=["voices"])
def create_cloned_voice(req: CreateCloneVoiceRequest):
    """Register a reusable voice from reference audio. Returns a voice_id."""
    if not req.x_vector_only_mode and not req.ref_text:
        raise HTTPException(
            status_code=400, detail="ref_text is required unless x_vector_only_mode is true."
        )
    model = registry.get("base")
    with gpu_lock:
        items = model.create_voice_clone_prompt(
            ref_audio=normalize_audio_ref(req.ref_audio),
            ref_text=req.ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
        )
    voice_id = f"vc_{uuid.uuid4().hex[:12]}"
    meta = {
        "voice_id": voice_id,
        "name": req.name or voice_id,
        "kind": "cloned",
        "ref_text": req.ref_text,
        "x_vector_only_mode": req.x_vector_only_mode,
        "created_at": time.time(),
    }
    save_voice(voice_id, items, meta)
    return meta


@app.post("/v1/voices/cloned/upload", tags=["voices"])
async def create_cloned_voice_upload(
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    x_vector_only_mode: bool = Form(False),
    name: Optional[str] = Form(None),
):
    """Register a reusable voice from an uploaded audio file."""
    data = await ref_audio.read()
    return create_cloned_voice(
        CreateCloneVoiceRequest(
            ref_audio=decode_upload(data),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
            name=name,
        )
    )


@app.post("/v1/voices/designed", tags=["voices"])
def create_designed_voice(req: CreateDesignedVoiceRequest):
    """Invent a voice from a description and persist it as a reusable voice_id."""
    ref_wav, ref_sr = design_reference_clip(
        req.instruct, req.ref_text, req.language, req.kwargs()
    )
    base = registry.get("base")
    with gpu_lock:
        items = base.create_voice_clone_prompt(
            ref_audio=(ref_wav, ref_sr), ref_text=req.ref_text
        )
    voice_id = f"vd_{uuid.uuid4().hex[:12]}"
    meta = {
        "voice_id": voice_id,
        "name": req.name or voice_id,
        "kind": "designed",
        "instruct": req.instruct,
        "ref_text": req.ref_text,
        "created_at": time.time(),
    }
    save_voice(voice_id, items, meta)
    return {
        **meta,
        "reference_audio": base64.b64encode(wav_bytes(ref_wav, ref_sr)).decode("ascii"),
        "sample_rate": ref_sr,
    }


@app.get("/v1/voices/saved", tags=["voices"])
def list_saved_voices():
    out = []
    for fn in sorted(os.listdir(VOICE_DIR)):
        if fn.endswith(".pt"):
            try:
                out.append(voice_meta(fn[:-3]))
            except Exception as e:  # a corrupt file shouldn't break the listing
                out.append({"voice_id": fn[:-3], "error": str(e)})
    return {"voices": out, "count": len(out)}


@app.get("/v1/voices/saved/{voice_id}", tags=["voices"])
def get_saved_voice(voice_id: str):
    if not os.path.isfile(voice_path(voice_id)):
        raise HTTPException(status_code=404, detail=f"Saved voice '{voice_id}' not found.")
    return voice_meta(voice_id)


@app.delete("/v1/voices/saved/{voice_id}", tags=["voices"])
def delete_saved_voice(voice_id: str):
    path = voice_path(voice_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Saved voice '{voice_id}' not found.")
    os.remove(path)
    return {"deleted": voice_id}


# ------------------------------ codec endpoints ---------------------------- #


@app.post("/v1/codec/encode", tags=["codec"])
def codec_encode(req: EncodeRequest):
    """Encode audio into discrete 12 Hz codes."""
    model = registry.get("base")
    audio = [normalize_audio_ref(a) for a in req.audio] if isinstance(req.audio, list) else normalize_audio_ref(req.audio)
    with gpu_lock:
        enc = model.model.speech_tokenizer.encode(audio)
    codes = [c.tolist() for c in enc.audio_codes]
    return {
        "count": len(codes),
        "shapes": [list(np.array(c).shape) for c in codes],
        "audio_codes": codes,
    }


@app.post("/v1/codec/decode", tags=["codec"])
def codec_decode(req: DecodeRequest):
    """Decode 12 Hz codes back into audio."""
    model = registry.get("base")
    payload = [{"audio_codes": torch.tensor(c, dtype=torch.long)} for c in req.codes]
    with gpu_lock:
        wavs, sr = model.model.speech_tokenizer.decode(payload)
    return audio_response(wavs, sr, "json", 0.0)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("QWEN_TTS_HOST", "127.0.0.1"),
        port=int(os.environ.get("QWEN_TTS_PORT", "9000")),
    )
