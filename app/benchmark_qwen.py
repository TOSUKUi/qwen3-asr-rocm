import argparse
import json
import time
from pathlib import Path

from app.asr_service import transcribe_path


def probe_duration_sec(path: str) -> float:
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR variants.")
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help=(
            "Space-separated benchmark cases in the form "
            "label:model:dtype:timestamps. Example: "
            "1.7b_bf16:Qwen/Qwen3-ASR-1.7B:bfloat16:false"
        ),
    )
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--forced-aligner",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    parser.add_argument("--output", default="", help="Optional JSON output path")
    return parser.parse_args()


def parse_case(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(f"invalid case spec: {spec}")
    label, model_name, dtype_name, timestamps_raw = parts
    timestamps = timestamps_raw.lower() in {"1", "true", "yes", "on"}
    return {
        "label": label,
        "model_name": model_name,
        "dtype_name": dtype_name,
        "timestamps": timestamps,
    }


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    audio_sec = probe_duration_sec(str(audio_path))
    language = None if args.language.lower() == "auto" else args.language

    results = []
    for spec in args.cases:
        case = parse_case(spec)
        wall0 = time.perf_counter()
        payload = transcribe_path(
            audio_path=str(audio_path),
            model_name=case["model_name"],
            language=language,
            dtype_name=case["dtype_name"],
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            timestamps=case["timestamps"],
            forced_aligner=args.forced_aligner,
        )
        wall1 = time.perf_counter()
        infer_sec = float(payload["infer_sec"])
        results.append(
            {
                "label": case["label"],
                "model": case["model_name"],
                "dtype": case["dtype_name"],
                "timestamps": case["timestamps"],
                "audio_sec": round(audio_sec, 3),
                "load_sec": payload["load_sec"],
                "infer_sec": payload["infer_sec"],
                "wall_sec": round(wall1 - wall0, 3),
                "rtf": round(infer_sec / audio_sec, 4),
                "chars": len(payload.get("text") or ""),
                "segments": len(payload.get("time_stamps") or []),
                "preview": (payload.get("text") or "")[:120],
            }
        )

    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
