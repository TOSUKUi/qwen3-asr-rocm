# OpenAI Compatible API

このサーバーはローカル実装ですが、`POST /v1/audio/transcriptions` については OpenAI 互換の呼び出し形に寄せています。

## Base URL

```text
http://localhost:8000/v1
```

## サポート済みエンドポイント

### `GET /v1/models`

利用可能モデルを返します。

レスポンス例:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-ASR-1.7B",
      "object": "model",
      "owned_by": "local-qwen"
    }
  ]
}
```

### `POST /v1/audio/transcriptions`

multipart/form-data で音声ファイルを送ります。

サポートするフォーム項目:

- `file`: 必須
- `model`: 必須。現状は `Qwen/Qwen3-ASR-1.7B` のみ
- `language`: 任意。`Japanese` か `auto`
- `response_format`: 任意。`json`、`text`、`verbose_json`
- `timestamp_granularities`: 任意。`segment` のみ対応

未対応:

- `stream`
- `timestamp_granularities[]=word`
- モデル切り替え
- 翻訳 API

## curl 例

### JSON

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local-dev-token" \
  -F "file=@samples/sample.wav" \
  -F "model=Qwen/Qwen3-ASR-1.7B" \
  -F "language=Japanese"
```

### text

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@samples/sample.wav" \
  -F "model=Qwen/Qwen3-ASR-1.7B" \
  -F "response_format=text"
```

### verbose_json

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@samples/sample.wav" \
  -F "model=Qwen/Qwen3-ASR-1.7B" \
  -F "language=Japanese" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities=segment"
```

## OpenAI SDK 例

OpenAI Python SDK を使う場合は `base_url` をこのサーバーに向けます。

```python
from openai import OpenAI

client = OpenAI(
    api_key="local-dev-token",
    base_url="http://localhost:8000/v1",
)

with open("samples/sample.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="Qwen/Qwen3-ASR-1.7B",
        file=f,
        language="Japanese",
    )

print(transcript.text)
```
