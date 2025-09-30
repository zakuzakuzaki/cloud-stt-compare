# cloud-stt-compare

クラウド音声認識サービス（GCP、AWS）の比較プロジェクト

## セットアップ

このプロジェクトは [uv](https://github.com/astral-sh/uv) を使用してパッケージを管理しています。

### uvのインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 依存関係のインストール

使用するサービスに応じて依存関係をインストールします。

```bash
# GCP用
uv sync --extra gcp

# AWS用
uv sync --extra aws

# 両方
uv sync --all-extras
```

## 環境変数の設定

`.env.example`をコピーして`.env`ファイルを作成し、必要な値を設定してください。

```bash
cp .env.example .env
# .envファイルを編集して実際の値を入力
```

## 実行手順

### GCP Speech-to-Text

```bash
# 実行
uv run --env-file .env python gcp/streaming_microphone.py
```

スクリプトは自動的にマイク入力を開始し、リアルタイムで音声認識を行います。デフォルトで日本語（ja-JP）に設定されています。

**言語を変更する場合:**

`gcp/streaming_microphone.py`の167行目を編集：
```python
# 英語認識の場合
transcribe_streaming_mic(language_code="en-US")
```

### AWS Transcribe

```bash
# 実行
uv run --env-file .env python aws/simple_mic.py
```

**注意:** `aws/simple_mic.py`内のリージョンと言語コードを必要に応じて変更してください。

## 前提条件

### GCP
- Google Cloud プロジェクトの作成
- Speech-to-Text API の有効化
- サービスアカウントの作成と認証情報JSONファイルの取得

#### サービスアカウントに必要な権限

以下のいずれかのロールを付与してください：

**最小権限:**
- `roles/speech.client` - Cloud Speech-to-Text Client (Cloud Speech クライアント)

**権限の付与方法:**
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/speech.client"
```

### AWS
- AWS アカウントの作成
- Transcribe サービスへのアクセス権限
- AWS認証情報の設定