#!/usr/bin/env bash
# =====================================================================
# Syncformer 安装: 独立 venv (.venv_syncformer) + 权重检查 + 冒烟测试
# - venv 用 --system-site-packages 复用 base 环境的 torch / torchaudio / av,
#   只在 venv 里装与 base 冲突的 transformers 4.40.* / timm 0.9.16
# - 权重: logs/sync_models/<EXP>/ (若 $MODEL_ROOT/Syncformer 存在则软链过去);
#   缺失时 example.py / batch_eval.py 首次运行会从官方地址自动下载
# - 由 setup_all.sh 调用, 也可直接 `bash setup.sh`
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
export REPO_ROOT MODEL_ROOT

PY_SYS="${PYTHON_BIN:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv_syncformer"
cd "$HERE"
# 独立 venv (transformers 4.40.*), 隔离 base 的 transformers 4.57.6.
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "== 建 venv $VENV (复用 base 大包 torch/torchaudio/av) =="
    "$PY_SYS" -m venv --system-site-packages "$VENV"
fi
PY="$VENV/bin/python"
PIP="$PY -m pip"

echo "==[1/4] ffmpeg=="
ffmpeg -version 2>/dev/null | head -n 1 || { echo "ERROR: 未找到系统 ffmpeg, 请先安装"; exit 1; }

echo "==[2/4] pip 依赖="
# 选用有 Py3.12 预编译 wheel 的版本 (避免 Rust / ffmpeg-dev 源码编译):
#   - transformers 4.40.* (仓库里只用了 ASTConfig / AutoTokenizer 这类稳定 API)
#     连带 tokenizers>=0.19 (0.13.3 无 cp312 wheel 会拉 Rust 编译)
#   - timm 0.9.16 (0.6.12 的 maxxvit.py 在 Py3.11+ 上 @dataclass mutable default 会报错)
#     仓库只用 trunc_normal_/DropPath/to_2tuple 这类 API, 0.9.16 全部保留
#   - av: torchvision.io.read_video (dataset_utils.py / example.py) 依赖 PyAV
#   - omegaconf / einops 都是纯 py3 wheel
$PIP install --upgrade pip
$PIP install \
    "omegaconf" "einops" "timm==0.9.16" \
    "transformers==4.40.*" \
    "tokenizers>=0.19,<0.20" \
    "huggingface_hub" \
    "av"

# torch / torchaudio 来自 base 环境 (--system-site-packages)
$PY -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available()); print('torchaudio', torchaudio.__version__)"

echo "==[3/4] 权重 (logs/sync_models)="
if [[ ! -e logs/sync_models && -d "$MODEL_ROOT/Syncformer" ]]; then
    mkdir -p logs
    ln -sfn "$MODEL_ROOT/Syncformer" logs/sync_models
fi
# 默认 EXP 用于评测; 另外两个 EXP 可选
ls -lh logs/sync_models/24-01-04T16-39-21/ 2>/dev/null \
    || echo "默认权重 logs/sync_models/24-01-04T16-39-21/ 不存在, 首次运行时会从官方地址自动下载"
for exp in 23-12-23T18-33-57 24-01-22T20-34-52; do
    [[ -d "logs/sync_models/$exp" ]] && ls -lh "logs/sync_models/$exp/" || echo "(optional) $exp 未下载, 跳过"
done
echo "==[4/4] 冒烟: 用 AudioSet 模型跑 example.py="
# 依次尝试: LTX-2.3 输出 -> Ovi 输出 -> Ovi/assets 示例视频
SMOKE_VID=""
for cand_root in \
    $REPO_ROOT/LTX-2.3_test/output \
    $REPO_ROOT/Ovi_test/output \
    $REPO_ROOT/daVinci-MagiHuman_test/output \
    $REPO_ROOT/Ovi_test/Ovi/assets ; do
    found=$(find "$cand_root" -name "*.mp4" 2>/dev/null | head -n 1 || true)
    if [[ -n "$found" ]]; then SMOKE_VID="$found"; break; fi
done
if [[ -n "$SMOKE_VID" ]]; then
    echo "smoke video: $SMOKE_VID"
    $PY example.py --exp_name 24-01-04T16-39-21 --vid_path "$SMOKE_VID" --offset_sec 0.0
else
    echo "未找到任何 mp4 冒烟视频, 跳过冒烟. 真实跑用: bash run_all.sh"
fi
echo
echo "DONE."
