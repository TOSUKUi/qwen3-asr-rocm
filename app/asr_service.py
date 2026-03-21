import tempfile
import time
from pathlib import Path
from threading import Lock

import torch
from qwen_asr import Qwen3ASRModel


_MODEL = None
_MODEL_CONFIG = None
_MODEL_LOCK = Lock()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False. Check ROCm container access.")


def get_model(
    model_name: str,
    dtype_name: str,
    batch_size: int,
    max_new_tokens: int,
    timestamps: bool,
    forced_aligner: str,
):
    global _MODEL
    global _MODEL_CONFIG

    ensure_cuda_available()

    config = {
        "model_name": model_name,
        "dtype_name": dtype_name,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "timestamps": timestamps,
        "forced_aligner": forced_aligner,
    }

    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_CONFIG == config:
            return _MODEL, 0.0

        dtype = resolve_dtype(dtype_name)
        model_kwargs = {
            "dtype": dtype,
            "device_map": "cuda:0",
            "max_inference_batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
        }
        if timestamps:
            model_kwargs["forced_aligner"] = forced_aligner
            model_kwargs["forced_aligner_kwargs"] = {
                "dtype": dtype,
                "device_map": "cuda:0",
            }

        t0 = time.perf_counter()
        model = Qwen3ASRModel.from_pretrained(model_name, **model_kwargs)
        load_sec = time.perf_counter() - t0

        _MODEL = model
        _MODEL_CONFIG = config
        return _MODEL, load_sec


def transcribe_path(
    audio_path: str,
    model_name: str,
    language: str | None,
    dtype_name: str,
    batch_size: int,
    max_new_tokens: int,
    timestamps: bool,
    forced_aligner: str,
):
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    model, load_sec = get_model(
        model_name=model_name,
        dtype_name=dtype_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        timestamps=timestamps,
        forced_aligner=forced_aligner,
    )

    t0 = time.perf_counter()
    results = model.transcribe(
        audio=str(path),
        language=language,
        return_time_stamps=timestamps,
    )
    infer_sec = time.perf_counter() - t0

    first = results[0]
    payload = {
        "audio": str(path),
        "model": model_name,
        "load_sec": round(load_sec, 3),
        "infer_sec": round(infer_sec, 3),
        "language": getattr(first, "language", None),
        "text": getattr(first, "text", None),
    }
    if timestamps:
        payload["time_stamps"] = getattr(first, "time_stamps", None)
    return payload


def transcribe_bytes(
    audio_bytes: bytes,
    suffix: str,
    model_name: str,
    language: str | None,
    dtype_name: str,
    batch_size: int,
    max_new_tokens: int,
    timestamps: bool,
    forced_aligner: str,
):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        return transcribe_path(
            audio_path=tmp.name,
            model_name=model_name,
            language=language,
            dtype_name=dtype_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            timestamps=timestamps,
            forced_aligner=forced_aligner,
        )
