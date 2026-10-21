#!/usr/bin/env bash
# =====================================================================
# syncnet_python 安装: 独立 venv (.venv_syncnet) + 权重 + 冒烟测试
# - venv 用 --system-site-packages 复用 base 环境的 torch / scipy / numpy,
#   只在 venv 里装旧 API 的 scenedetect 0.5.6.1 (base 的新版缺 VideoManager)
# - 权重 (data/syncnet_v2.model, detectors/s3fd/weights/sfd_face.pth) 缺失时
#   用 download_model.sh 下载
# - 由 setup_all.sh 调用, 也可直接 `bash setup.sh`
# =====================================================================
set -euo pipefail

PY_SYS="${PYTHON_BIN:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv_syncnet"
cd "$HERE"
# 独立 venv (scenedetect 0.5.6.1 旧 API), 隔离 base 的新版 scenedetect (VBench2 用).
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "== 建 venv $VENV (复用 base 大包 torch/cv2/scipy) =="
    "$PY_SYS" -m venv --system-site-packages "$VENV"
fi
PY="$VENV/bin/python"
PIP="$PY -m pip"

echo "==[1/4] ffmpeg=="
ffmpeg -version 2>/dev/null | head -n 1 || { echo "ERROR: 未找到系统 ffmpeg, 请先安装"; exit 1; }

echo "==[2/4] 装 pip 依赖=="
# inference.py 兼容 0.5.6.1 的 scenedetect API; opencv-contrib-python 含 face 模块
$PIP install --upgrade pip
$PIP install \
    "scenedetect==0.5.6.1" \
    "opencv-contrib-python" \
    "python_speech_features" \
    "scipy" "numpy"
# torch 来自 base 环境 (--system-site-packages)
$PY -c "import torch; print('torch ok', torch.__version__)" \
    || { echo "ERROR: venv 里 import torch 失败, 请先完成 setup_all.sh 的 base_env 步骤"; exit 1; }

echo "==[3/4] 权重=="
if [[ ! -s data/syncnet_v2.model || ! -s data/example.avi || ! -s detectors/s3fd/weights/sfd_face.pth ]]; then
    echo "权重缺失, 运行 download_model.sh 下载"
    bash download_model.sh
fi
ls -lh data/syncnet_v2.model data/example.avi detectors/s3fd/weights/sfd_face.pth

echo "==[4/4] 冒烟: 跑官方 example.avi=="
$PY inference.py --videofile data/example.avi --data_dir data/work_smoke
echo
echo "DONE. 真实评测请跑: bash run_all.sh"
