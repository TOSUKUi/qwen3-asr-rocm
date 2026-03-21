from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.asr_service import transcribe_bytes

app = FastAPI(title="Qwen3-ASR ROCm API", version="0.1.0")


@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("Japanese"),
    model: str = Form("Qwen/Qwen3-ASR-1.7B"),
    dtype: str = Form("bfloat16"),
    max_new_tokens: int = Form(256),
    batch_size: int = Form(1),
    timestamps: bool = Form(False),
    forced_aligner: str = Form("Qwen/Qwen3-ForcedAligner-0.6B"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    suffix = Path(file.filename).suffix or ".wav"
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    try:
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
