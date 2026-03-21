# Qwen3-ASR-1.7B on ROCm Docker

Ryzen AI Max+ 395 / ROCm 7.2 を前提に、`Qwen/Qwen3-ASR-1.7B` を Docker でまず動かすための最小構成です。最初は `transformers` バックエンドで動作確認し、vLLM は後段に回す前提です。

## 前提

- Ubuntu 24.04.3
- ROCm 7.2 がホストに導入済み
- Docker が使える
- `/dev/kfd` と `/dev/dri` が見える

ホスト側の確認:

```bash
ls -l /dev/kfd /dev/dri
docker --version
```

## 使い方

### 1. イメージをビルド

```bash
docker compose build
```

初回ビルドでは `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1` を取得し、`qwen-asr` と `ffmpeg` を入れます。

### 2. ROCm が見えるか確認

```bash
docker compose run --rm qwen3-asr python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

期待値:

- 1 行目が `True`
- 2 行目に AMD GPU 名が出る

### 3. 音声認識を実行

音声ファイルをこのリポジトリ配下に置いて実行します。例では `samples/sample.wav` を使います。

```bash
mkdir -p samples outputs
docker compose run --rm qwen3-asr \
  python3 app/transcribe_qwen.py samples/sample.wav | tee outputs/result.json
```

初回推論時は Hugging Face からモデルを取得するので時間がかかります。キャッシュは `./cache/huggingface` に残ります。

## オプション

言語自動判定:

```bash
docker compose run --rm qwen3-asr \
  python3 app/transcribe_qwen.py samples/sample.wav --language auto
```

タイムスタンプ付き:

```bash
docker compose run --rm qwen3-asr \
  python3 app/transcribe_qwen.py samples/sample.wav --timestamps
```

`--timestamps` を使うと `Qwen/Qwen3-ForcedAligner-0.6B` も読み込みます。

## トラブルシュート

`torch.cuda.is_available()` が `False`

- ホスト側の ROCm 導入を確認
- `docker compose` 実行ユーザーが Docker を扱えるか確認
- `/dev/kfd` と `/dev/dri` がコンテナに渡っているか確認

モデル取得で失敗する

- ネットワーク到達性を確認
- Hugging Face 側の認証が必要な場合は `huggingface-cli login` 相当のトークン設定を追加

長い音声で重い

- `--max-new-tokens` を下げる
- 先に音声を分割する

## Git 初期化

このディレクトリを Git 管理するなら:

```bash
git init
git add .
git commit -m "Initial Qwen3-ASR ROCm Docker setup"
```
