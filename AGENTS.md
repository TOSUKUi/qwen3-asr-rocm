# AGENTS.md

## Qwen ROCm Baseline

- Qwen 側の既定 ROCm は `7.1` のまま扱う。
- 根拠は [README.md](/home/amemiya/work/qwen3-asr/README.md) にある通り、このリポジトリでは `ROCm 7.2 系より、ROCm 7.1 系の AMD PyTorch イメージを既定値にした方が安定していました` という運用判断になっているため。
- 既定のベースイメージは `rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0`。
- Qwen 用サービス `qwen3-asr` と `qwen3-asr-api` は、明示的な依頼と再検証なしに `7.2` へ上げない。

## VibeVoice

- VibeVoice は Qwen 用環境と依存がズレるため、別イメージとして扱う。
- README 上でも、VibeVoice は短尺 30 秒なら ROCm 上で完走した前提で扱う。
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` は短尺で改善が出たので、再現試験時の候補に含める。
- フル尺はモデルロードと生成開始までは到達しているが、完走確認前の前提で扱う。

## Editing Notes

- README の方針と食い違う ROCm 変更を入れる前に、まず README の記述と現行の実測結果を確認する。
- `docker-compose.yml` には未コミット変更が入ることがあるので、既存差分を消さない。
