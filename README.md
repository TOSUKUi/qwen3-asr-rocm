# Qwen3-ASR-1.7B on ROCm Docker

Ryzen AI Max+ 395 上で `Qwen/Qwen3-ASR-1.7B` を Docker で動かすための最小構成です。現状は ROCm 7.2 系より、ROCm 7.1 系の AMD PyTorch イメージを既定値にした方が安定していました。

## 前提

- Ubuntu 24.04.3
- ROCm がホストに導入済み
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

初回ビルドでは既定で `rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0` を取得し、`qwen-asr` と `ffmpeg` を入れます。

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
  python3 -m app.transcribe_qwen samples/sample.wav | tee outputs/result.json
```

初回推論時は Hugging Face からモデルを取得するので時間がかかります。キャッシュは `./cache/huggingface` に残ります。

## オプション

言語自動判定:

```bash
docker compose run --rm qwen3-asr \
  python3 -m app.transcribe_qwen samples/sample.wav --language auto
```

タイムスタンプ付き:

```bash
docker compose run --rm qwen3-asr \
  python3 -m app.transcribe_qwen samples/sample.wav --timestamps
```

`--timestamps` を使うと `Qwen/Qwen3-ForcedAligner-0.6B` も読み込みます。

## API サーバー

常駐プロセスとして使う場合は API サーバーを起動します。

### 1. サーバー起動

```bash
docker compose up -d qwen3-asr-api
```

### 2. ヘルスチェック

```bash
curl http://localhost:8000/health
```

### 3. 音声を投げる

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@samples/sample.wav" \
  -F "language=Japanese"
```

自動判定なら `-F "language=auto"` を使います。

タイムスタンプ付き:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@samples/sample.wav" \
  -F "language=Japanese" \
  -F "timestamps=true"
```

停止:

```bash
docker compose stop qwen3-asr-api
```

## OpenAI 互換 API

公開仕様としては `POST /v1/audio/transcriptions` を用意しています。OpenAI SDK や既存クライアントを流用しやすい形です。

モデル一覧:

```bash
curl http://localhost:8000/v1/models
```

文字起こし:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local-dev-token" \
  -F "file=@samples/sample.wav" \
  -F "model=Qwen/Qwen3-ASR-1.7B" \
  -F "language=Japanese"
```

`response_format=text` と `response_format=verbose_json` にも対応しています。詳細は [docs/openai-compatible-api.md](/home/amemiya/work/qwen3-asr/docs/openai-compatible-api.md) を参照してください。

## VibeVoice ASR

`transformers` 側の VibeVoice ASR は Qwen 用環境と依存がズレるので、別イメージ `vibevoice-asr` として分離しています。

現状メモ:

- 専用イメージのビルドは通る
- `transformers` の `VibeVoiceAsrForConditionalGeneration` import は通る
- この ROCm 7.1 / Ryzen AI Max+ 395 環境では、30 秒短尺は ROCm 上で完走した
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` を付けると短尺は少し速かった
- フル尺はモデルロードと生成開始までは到達したが、README 更新時点では完走未確認

### 1. イメージをビルド

```bash
docker compose build vibevoice-asr
```

### 2. ヘルプ確認

```bash
docker compose run --rm vibevoice-asr python3 -m app.transcribe_vibevoice --help
```

### 3. 文字起こし

```bash
docker compose run --rm vibevoice-asr \
  python3 -m app.transcribe_vibevoice samples/sample.wav
```

任意のコンテキストを与える例:

```bash
docker compose run --rm vibevoice-asr \
  python3 -m app.transcribe_vibevoice samples/sample.wav \
  --prompt "About horse racing"
```

短尺で `AOTriton` を有効にする例:

```bash
docker compose run --rm -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 vibevoice-asr \
  python3 -m app.transcribe_vibevoice samples/sample.wav
```

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
