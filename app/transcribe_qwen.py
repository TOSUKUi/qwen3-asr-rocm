import argparse
import json

from app.asr_service import transcribe_path


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


def main() -> int:
    args = parse_args()
    language = None if args.language.lower() == "auto" else args.language
    payload = transcribe_path(
        audio_path=args.audio,
        model_name=args.model,
        language=language,
        dtype_name=args.dtype,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        timestamps=args.timestamps,
        forced_aligner=args.forced_aligner,
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
