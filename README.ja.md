# soarm101-isaac-lab-sagemaker-tensorboard

SO-ARM101 の Reach タスクを Isaac Lab で強化学習し、Amazon SageMaker Training Job + Managed Spot Training で実行するサンプルコードに、TensorBoard で学習進捗を可視化する機能を加えたものです。

学習済みログをローカル TensorBoard で後追い参照する方法（Part 1）と、学習中のジョブのログを S3 経由でほぼリアルタイムに参照する方法（Part 2）の 2 通りに対応しています。

関連ブログ記事：[SageMaker Training Job + TensorBoard で Isaac Lab の強化学習を可視化する](https://dev.classmethod.jp/articles/)（公開後にリンク差し替え）

> English version: [README.md](README.md)

## 概要

- ベースイメージ: `nvcr.io/nvidia/isaac-lab:2.3.2`（Isaac Lab 2.3.2 + Isaac Sim 5.1.x 同梱）
- タスク: `Isaac-SO-ARM101-Reach-v0`（[MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) main ブランチ）
- 学習基盤: SageMaker Training Job、デフォルト **ml.g5.2xlarge**（NVIDIA A10G 24 GB）、Managed Spot
- リージョン: `ap-northeast-1`
- 実測コスト（Reach 1 試行、`max_iterations=1000`）：オンデマンド **約 0.31 USD** / Managed Spot **約 0.13 USD（58% 削減）**

## リポジトリ構成

```
.
├── cdk/                    # AWS CDK（TypeScript）: S3 / ECR / IAM Role
├── scripts/
│   ├── push_to_ecr.sh      # 学習用 image を build して ECR に push
│   ├── download_logs.sh    # Part 1: 学習済ログを S3 から DL してローカル TB で参照
│   └── tb_live.sh          # Part 2: 学習中ジョブの TB ログを S3 直接参照で起動
├── src/
│   ├── train.py            # SageMaker entrypoint（SIGTERM 対応、ckpt 自動再開、TB symlink）
│   └── entrypoint.sh
├── Dockerfile              # NGC isaac-lab:2.3.2 を継承
├── submit.py               # SageMaker Estimator 起動スクリプト
└── README.md / README.ja.md
```

## 前提条件

- SageMaker / S3 / ECR 権限のある AWS アカウント
- `ap-northeast-1` 用に設定済みの AWS CLI v2
- Docker（`linux/amd64` ビルド対応）
- Node.js 20.x と AWS CDK v2（`pnpm add -g aws-cdk`）
- NVIDIA NGC アカウントと API key（`docker login nvcr.io`）
- Python 3.11 と SageMaker Python SDK（`pip install sagemaker`）

## セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/furuya02/soarm101-isaac-lab-sagemaker-tensorboard.git
cd soarm101-isaac-lab-sagemaker-tensorboard
```

### 2. CDK で AWS リソースを構築

```bash
cd cdk
pnpm install

export AWS_REGION=ap-northeast-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

cdk bootstrap aws://${ACCOUNT_ID}/${AWS_REGION}
cdk deploy \
  -c account_id=${ACCOUNT_ID} \
  -c region=${AWS_REGION}
```

このスタックで作成されるリソース：

- S3 バケット: `soarm101-isaac-lab-sagemaker-tensorboard-<ACCOUNT_ID>`
- ECR リポジトリ: `soarm101-isaac-lab-sagemaker-tensorboard`
- IAM ロール: `soarm101-isaac-lab-sagemaker-tensorboard-execution-role`

bucket suffix を上書きする場合：

```bash
cdk deploy \
  -c account_id=${ACCOUNT_ID} \
  -c bucket_suffix=20260503
```

### 3. 学習用 image を build して ECR に push

```bash
cd ..

# NGC にログイン（NGC API key が必要）
docker login nvcr.io

./scripts/push_to_ecr.sh
```

初回 push は約 15 GB の転送が発生し、回線速度に応じて 30〜60 分程度かかります。同一リージョンの ECR から SageMaker への pull はリージョン間データ転送料金が発生しません。

### 4. SageMaker Training Job を投入

```bash
export SAGEMAKER_ROLE_ARN=$(aws iam get-role \
  --role-name soarm101-isaac-lab-sagemaker-tensorboard-execution-role \
  --query 'Role.Arn' --output text)
export ECR_IMAGE_URI=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/soarm101-isaac-lab-sagemaker-tensorboard:latest
export S3_BUCKET=soarm101-isaac-lab-sagemaker-tensorboard-${ACCOUNT_ID}

# On-demand で動作確認
USE_SPOT=false MAX_RUN_MINUTES=15 python submit.py

# Managed Spot 実行
USE_SPOT=true MAX_RUN_MINUTES=15 MAX_WAIT_MINUTES=16 python submit.py
```

### 5. 学習済みモデルの取得

```bash
JOB_NAME=<submit.py の出力に表示される job name>
aws s3 cp s3://${S3_BUCKET}/output/${JOB_NAME}/output/model.tar.gz .
tar xzf model.tar.gz
# rsl_rl/<task>/<run>/model_<iter>.pt が取り出せる
```

### 6. TensorBoard で学習進捗を見る

rsl_rl は `--logger tensorboard` で起動されており、エピソード平均報酬・PPO の loss・探索ノイズなどを TensorBoard 形式で出力しています。本リポジトリでは `src/train.py` 内で TensorBoard ログ出力先を `/opt/ml/checkpoints/tensorboard/` への symlink として作成しているため、SageMaker の `checkpoint_s3_uri` 経由で **学習中もリアルタイムで S3 へ連続同期** されます。

事前準備：

```bash
pip install tensorboard tensorflow-io
```

#### Part 1: 学習済みログをローカル TensorBoard で後追い参照する

学習が完了したジョブの `model.tar.gz` を S3 から DL し、解凍してローカル TensorBoard を起動します。

```bash
./scripts/download_logs.sh ${S3_BUCKET} ${JOB_NAME}
tensorboard --logdir ./logs/${JOB_NAME}/rsl_rl/
# ブラウザで http://localhost:6006
```

#### Part 2: 学習中のジョブのログをリアルタイムで参照する

ジョブ投入直後にローカル TensorBoard を S3 ダイレクト参照モードで起動し、学習中の curve をブラウザで眺めます。

```bash
./scripts/tb_live.sh ${S3_BUCKET} ${JOB_NAME}
# ブラウザで http://localhost:6006
```

注意事項：

- TensorBoard が S3 を継続スキャンするため、**長時間放置すると S3 GET API のコストが膨らむ**事例が報告されています（[tensorboard issue #6564](https://github.com/tensorflow/tensorboard/issues/6564)）。確認したらブラウザを閉じ、TensorBoard プロセス（Ctrl-C）を終了させてください。
- 同期間隔は `checkpoint_s3_uri` の Continuous upload mode に依存し、数十秒〜数分のラグがあります。

### 7. （任意）学習済モデルから mp4 動画を生成する

学習済モデルの動作を視覚的に確認したい場合は、もう 1 つ SageMaker ジョブを投入して `play.py --headless --video` を実行します。GUI 環境は不要です。`Dockerfile` で `ffmpeg` を導入しているのは、Isaac Lab の `RecordVideo` ラッパーが frame buffers を mp4 にエンコードする際に `ffmpeg` を呼び出すためです（NGC の `isaac-lab` ベースイメージには ffmpeg は同梱されていません）。

```bash
MODE=play \
MODEL_S3_URI=s3://${S3_BUCKET}/output/${JOB_NAME}/output/model.tar.gz \
USE_SPOT=true MAX_RUN_MINUTES=15 MAX_WAIT_MINUTES=16 \
python submit.py

# play ジョブ完了後、動画をダウンロード:
PLAY_JOB=<play 用 submit の出力に表示される job name>
aws s3 cp s3://${S3_BUCKET}/output/${PLAY_JOB}/output/model.tar.gz play_output.tar.gz
tar xzf play_output.tar.gz   # videos/*.mp4
```

play ジョブは 5〜10 分で完了し、Spot なら 0.10 USD 未満で済みます。

## 実測コスト（ap-northeast-1、2026 年 5 月時点）

| 項目 | オンデマンド | Managed Spot |
|---|---|---|
| インスタンス | ml.g5.2xlarge（A10G 24GB） | ml.g5.2xlarge（A10G 24GB） |
| 学習時間（実時間） | 727 秒（12 分） | 733 秒（12 分） |
| 課金時間 | 727 秒 | 309 秒 |
| **1 試行コスト** | **0.306 USD** | **0.130 USD（58% off）** |

そのほか継続発生するコスト：

| リソース | コスト |
|---|---|
| ECR ストレージ（約 40 GB の image を保持する間） | 約 4.0 USD / 月 |
| S3（成果物 + チェックポイント、100 MB 未満） | 月額 0.10 USD 未満 |

`ml.g6.2xlarge`（NVIDIA L4、オンデマンド約 1.81 USD/時、Spot 約 0.54 USD/時）も quota が払い出されれば有力な選択肢です（注意事項を参照）。

## クリーンアップ

```bash
cd cdk
cdk destroy
# S3 バケット内のオブジェクトと ECR の image は、必要に応じて手動で削除してください。
```

## 注意事項

- **Managed Spot にはチェックポイント実装が必須**。チェックポイントがないと `max_wait` の上限が 1 時間に制限されます。`src/train.py` は `/opt/ml/checkpoints/model_*.pt` から自動的に再開します。
- **`max_run` は必ず指定**。学習暴走による課金事故を防ぎます。
- **リージョン固定**。ECR / S3 / SageMaker をすべて `ap-northeast-1` に揃え、リージョン間データ転送料金を回避します。
- **Image サイズ**。`nvcr.io/nvidia/isaac-lab:2.3.2` のベース image は uncompressed 約 40 GB。家庭回線では初回 ECR push に 30〜60 分、SageMaker のジョブ起動時の pull で 5〜10 分の起動オーバーヘッドが発生します。
- **Service Quotas**。新規 AWS アカウントでは SageMaker GPU 系インスタンスの quota が 0 のことが多く、特に最新の g6（L4）系は 0 のままです。本リポジトリでデフォルトとしている `ml.g5.2xlarge` は標準で quota = 1 立っていることが多く動作します。`ml.g6.2xlarge` を使う場合は事前に Service Quotas で「ml.g6.2xlarge for training job usage」と「for spot training job usage」を申請してください。
- **`isaac_so_arm101` のレイアウト**。`v1.2.0` タグは古い `source/SO_100/` レイアウトのため `pip install -e .` がそのままでは通りません。Dockerfile の build arg `ISAAC_SO_ARM101_REF` で main ブランチ（または直近の commit）を指定し、標準的な `src/isaac_so_arm101/` レイアウトを使ってください。
- **Mac Docker のディスク上限**。macOS の Docker Desktop は仮想ディスクのデフォルト上限が 64 GB で、40 GB の Isaac Lab image を扱うには不足します。`docker builder prune -a -f` で空き容量を確保するか、Docker Desktop -> Settings -> Resources -> Advanced で 200 GB に拡張してください。

## ライセンス

本サンプルコードは MIT License で公開しています。

`isaac_so_arm101` は BSD-3-Clause です。NVIDIA Isaac Sim および Isaac Lab は [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/install-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html) に従います。
