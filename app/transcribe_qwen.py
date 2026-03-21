import argparse
import json
import time
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Qwen3-ASR-1.7B on ROCm."
    )
    parser.add_argument("audio", help="Path to an audio file inside the container.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model ID to load from Hugging Face.",
    )
    parser.add_argument(
        "--language",
        default="Japanese",
        help="Language hint. Use auto for automatic language detection.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum decoder tokens per utterance.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Maximum inference batch size.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Enable forced alignment timestamps.",
    )
    parser.add_argument(
        "--forced-aligner",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Forced aligner model ID used only with --timestamps.",
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
    language = None if args.language.lower() == "auto" else args.language

    model_kwargs = {
        "dtype": dtype,
        "device_map": "cuda:0",
        "max_inference_batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.timestamps:
        model_kwargs["forced_aligner"] = args.forced_aligner
        model_kwargs["forced_aligner_kwargs"] = {
            "dtype": dtype,
            "device_map": "cuda:0",
        }

    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(args.model, **model_kwargs)
    t1 = time.perf_counter()

    results = model.transcribe(
        audio=str(audio_path),
        language=language,
        return_time_stamps=args.timestamps,
    )
    t2 = time.perf_counter()

    first = results[0]
    payload = {
        "audio": str(audio_path),
        "model": args.model,
        "load_sec": round(t1 - t0, 3),
        "infer_sec": round(t2 - t1, 3),
        "language": getattr(first, "language", None),
        "text": getattr(first, "text", None),
    }
    if args.timestamps:
        payload["time_stamps"] = getattr(first, "time_stamps", None)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
