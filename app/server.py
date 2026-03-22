from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.asr_service import transcribe_bytes

app = FastAPI(title="Qwen3-ASR ROCm API", version="0.1.0")

DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"


def read_stamp_field(stamp, field: str):
    if isinstance(stamp, dict):
        return stamp.get(field)
    if field == "start":
        return getattr(stamp, "start", getattr(stamp, "start_time", None))
    if field == "end":
        return getattr(stamp, "end", getattr(stamp, "end_time", None))
    return getattr(stamp, field, None)


@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_transcription(
    filename: str,
    audio_bytes: bytes,
    language: str,
    model: str,
    dtype: str,
    max_new_tokens: int,
    batch_size: int,
    timestamps: bool,
    forced_aligner: str,
):
    suffix = Path(filename).suffix or ".wav"
    language_value = None if language.lower() == "auto" else language
    return transcribe_bytes(
        audio_bytes=audio_bytes,
        suffix=suffix,
        model_name=model,
        language=language_value,
        dtype_name=dtype,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        timestamps=timestamps,
        forced_aligner=forced_aligner,
    )


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "owned_by": "local-qwen",
            }
        ],
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("Japanese"),
    model: str = Form(DEFAULT_MODEL),
    dtype: str = Form("bfloat16"),
    max_new_tokens: int = Form(256),
    batch_size: int = Form(1),
    timestamps: bool = Form(False),
    forced_aligner: str = Form(DEFAULT_ALIGNER),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        return run_transcription(
            filename=file.filename,
            audio_bytes=audio_bytes,
            language=language,
            model=model,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            timestamps=timestamps,
            forced_aligner=forced_aligner,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/audio/transcriptions")
async def openai_audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: str = Form("Japanese"),
    response_format: str = Form("json"),
    timestamp_granularities: str | None = Form(None),
    temperature: float | None = Form(None),
    prompt: str | None = Form(None),
    authorization: str | None = Header(default=None),
):
    del temperature
    del prompt
    del authorization

    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    if model != DEFAULT_MODEL:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported model: {model}. expected {DEFAULT_MODEL}",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    timestamps = False
    if timestamp_granularities:
        normalized = {item.strip() for item in timestamp_granularities.split(",") if item.strip()}
        unsupported = normalized.difference({"segment"})
        if unsupported:
            raise HTTPException(
                status_code=400,
                detail="only timestamp_granularities=segment is supported",
            )
        timestamps = "segment" in normalized

    try:
        result = run_transcription(
            filename=file.filename,
            audio_bytes=audio_bytes,
            language=language,
            model=model,
            dtype="bfloat16",
            max_new_tokens=256,
            batch_size=1,
            timestamps=timestamps,
            forced_aligner=DEFAULT_ALIGNER,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if response_format == "text":
        return PlainTextResponse(result["text"] or "")

    if response_format == "json":
        return {"text": result["text"] or ""}

    if response_format == "verbose_json":
        payload = {
            "task": "transcribe",
            "language": result["language"],
            "duration": None,
            "text": result["text"] or "",
        }
        if timestamps:
            raw_stamps = result.get("time_stamps") or []
            payload["segments"] = [
                {
                    "id": idx,
                    "seek": 0,
                    "start": read_stamp_field(stamp, "start"),
                    "end": read_stamp_field(stamp, "end"),
                    "text": read_stamp_field(stamp, "text"),
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": None,
                    "compression_ratio": None,
                    "no_speech_prob": None,
                }
                for idx, stamp in enumerate(raw_stamps)
            ]
        return payload

    raise HTTPException(
        status_code=400,
        detail="unsupported response_format. use text, json, or verbose_json",
    )
