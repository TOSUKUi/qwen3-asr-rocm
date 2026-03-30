# Benchmark

計測日: 2026-03-22

## 条件

- ハードウェア: Ryzen AI Max+ 395
- 実行方法: Docker
- ROCm / PyTorch ベース: `rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0`
- 音声ファイル: `samples/benchmark_sample.mp3`
- 音声長: `507.288 sec`
- 言語指定: `Japanese`

## 結果

| Case | Model | DType | Timestamps | Load sec | Infer sec | RTF | Chars | Segments |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1.7b_bf16` | `Qwen/Qwen3-ASR-1.7B` | `bfloat16` | `false` | 5.397 | 226.020 | 0.4455 | 372 | 0 |
| `1.7b_fp16` | `Qwen/Qwen3-ASR-1.7B` | `float16` | `false` | 6.358 | 217.870 | 0.4295 | 373 | 0 |
| `0.6b_bf16` | `Qwen/Qwen3-ASR-0.6B` | `bfloat16` | `false` | 5.911 | 38.936 | 0.0768 | 332 | 0 |
| `1.7b_bf16_ts` | `Qwen/Qwen3-ASR-1.7B` | `bfloat16` | `true` | 9.632 | 260.822 | 0.5141 | 1131 | 635 |

## 所見

- この環境では `1.7B float16` が `1.7B bfloat16` より少し速かった
- `0.6B` は `1.7B` より大幅に速く、RTF は `0.0768`
- `1.7B + timestamps` は通常転写より遅く、RTF は `0.5141`
- ROCm 7.2 系ではセグフォが出たが、ROCm 7.1 系では計測完走

## VibeVoice 短尺メモ

計測日: 2026-03-30

### 条件

- モデル: `microsoft/VibeVoice-ASR-HF`
- ハードウェア: Ryzen AI Max+ 395
- 実行方法: Docker
- ROCm / PyTorch ベース: `rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0`
- 音声ファイル: `samples/benchmark_sample_short_30s.wav`
- 音声長: `30 sec`
- 言語指定: なし

### 結果

| Case | DType | AOTriton | Load sec | Infer sec | RTF | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `vibevoice_short_bf16` | `bfloat16` | `false` | 15.829 | 97.672 | 3.2557 | 30 秒短尺は完走 |
| `vibevoice_short_bf16_aotriton` | `bfloat16` | `true` | 16.278 | 79.924 | 2.6641 | 通常設定より少し速い |

### 所見

- VibeVoice はこの環境で短尺 30 秒なら ROCm 上で完走した
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` で短尺は少し改善した
- ただし `RTF > 2` なので、現時点ではかなり遅い
- フル尺 `samples/benchmark_sample.mp3` はモデルロードと生成開始までは到達したが、この時点では完走未確認
