import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with VibeVoice ASR on ROCm."
    )
    parser.add_argument("audio", help="Path to an audio file inside the container.")
    parser.add_argument(
        "--model",
        default="microsoft/VibeVoice-ASR-HF",
        help="Model ID to load from Hugging Face.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional context / hotword prompt.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Optional acoustic_tokenizer_chunk_size override.",
    )
    parser.add_argument(
        "--return-format",
        default="transcription_only",
        help="Decoder return format. Example: transcription_only",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False. Check ROCm container access.")

    dtype = resolve_dtype(args.dtype)

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="cuda:0",
    )
    t1 = time.perf_counter()

    if args.prompt:
        inputs = processor.apply_transcription_request(
            audio=str(audio_path),
            prompt=args.prompt,
        )
    else:
        inputs = processor.apply_transcription_request(audio=str(audio_path))

    inputs = inputs.to(model.device, model.dtype)
    generate_kwargs = {}
    if args.chunk_size > 0:
        generate_kwargs["acoustic_tokenizer_chunk_size"] = args.chunk_size

    output_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    decoded = processor.decode(
        generated_ids,
        return_format=args.return_format,
    )
    t2 = time.perf_counter()

    payload = {
        "audio": str(audio_path),
        "model": args.model,
        "load_sec": round(t1 - t0, 3),
        "infer_sec": round(t2 - t1, 3),
        "dtype": args.dtype,
        "prompt": args.prompt or None,
        "return_format": args.return_format,
        "result": decoded[0] if isinstance(decoded, list) else decoded,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
