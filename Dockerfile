# =====================================================================
# 1. ベースイメージ
# =====================================================================
# NGC が配布する Isaac Lab 公式コンテナ。
# OS / NVIDIA driver 互換層 / Isaac Sim 5.1 / Isaac Lab 2.3.2 /
# Isaac Sim 同梱の embedded Python（torch[CUDA], isaaclab, isaaclab_rl 等）/
# Isaac Lab 公式リポジトリ全体（/workspace/isaaclab/, isaaclab.sh 含む）
# が image 内にプリインストール済み。
FROM nvcr.io/nvidia/isaac-lab:2.3.2


# =====================================================================
# 2. OS パッケージ追加
# =====================================================================
# ffmpeg : Isaac Lab の RecordVideo wrapper が mp4 エンコードに使用。
# python3: ビルド時パッチを実行するため。base image 同梱の Python は
#          /workspace/isaaclab/_isaac_sim/python.sh 経由でしか呼べず、
#          PATH 上に python3 コマンドが無いので別途追加する。
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg python3 \
 && rm -rf /var/lib/apt/lists/*


# =====================================================================
# 3. isaac_so_arm101 (SO-ARM101 用の Reach タスク実装)
# =====================================================================
# サードパーティ製の OSS パッケージ。SO-ARM101 用の URDF / Reach 環境
# 定義 / RSL-RL 用 cfg を提供し、`Isaac-SO-ARM101-Reach-v0` などを
# embedded Python の gym レジストリに登録する。
# 本記事執筆時点の main HEAD (2025-12-22) に pin。upstream の不意の
# 更新で手順が壊れないようにするため。別 ref を試したい場合は
# --build-arg ISAAC_SO_ARM101_REF=<commit_or_branch> で上書き可能。
ARG ISAAC_SO_ARM101_REF=e4624dea075b00a36dbc66bebd531d191c92e8cd
RUN git clone https://github.com/MuammerBay/isaac_so_arm101.git /opt/isaac_so_arm101 \
 && git -C /opt/isaac_so_arm101 checkout ${ISAAC_SO_ARM101_REF} \
 && /workspace/isaaclab/isaaclab.sh -p -m pip install -e /opt/isaac_so_arm101 --no-deps


# =====================================================================
# 4. ビルド時パッチ (isaac_so_arm101 を base image と整合させる互換性レイヤ)
# =====================================================================
# patch_play.py             : pretrained_checkpoint の import を
#                             旧 path / 新 path / ダミー の 3 段
#                             フォールバックに書き換え。
# patch_reach_visualizer.py : ゴールマーカーをマゼンタ球に変更し、
#                             カメラ位置を近接配置（動画の見やすさ）。
COPY scripts/patch_play.py scripts/patch_reach_visualizer.py /tmp/
RUN python3 /tmp/patch_play.py \
 && python3 /tmp/patch_reach_visualizer.py \
 && rm /tmp/patch_play.py /tmp/patch_reach_visualizer.py


# =====================================================================
# 5. SageMaker 連携 (wrapper / boto3 / EULA / ENTRYPOINT)
# =====================================================================
# NVIDIA Omniverse の利用規約を非対話環境向けに環境変数で受諾。
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y

# AWS SDK。play.py が S3 から学習済モデル (model.tar.gz) を download
# する際に使用する。
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install boto3

# 学習・動画生成 wrapper を SageMaker 予約パスに配置。
# entrypoint.sh が MODE 環境変数 (train|play) で 2 つの wrapper を
# 切り替える。
WORKDIR /opt/ml/code
COPY src/train.py src/play.py src/entrypoint.sh /opt/ml/code/
RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
