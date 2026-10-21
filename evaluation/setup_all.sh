#!/usr/bin/env bash
# =============================================================================
# evaluation/setup_all.sh
#
# 一次性配置全部评测工程的环境: 一个 Python 3.12 base 环境 + 3 个独立 venv
# (Q-Align / syncnet_python / Syncformer).
#
# 前置 (见 README.md §1.1):
#   - Python 3.12 (conda 环境或系统 Python; base 不要用 venv, 否则 syncnet_python /
#     Syncformer 的 --system-site-packages venv 继承不到 base 的包)
#   - NVIDIA 驱动 (CUDA 12.8) + CUDA Toolkit 12.8 (nvcc, 源码编译 mmcv / flash_attn 用)
#   - 系统包: gcc/g++, git, wget, curl, ffmpeg (tmux 由 system_tmux 步骤安装)
#   - 模型权重放在 $MODEL_ROOT (默认 $REPO_ROOT/models), 见 README.md §1.3
#
# 执行顺序 (即 ALL_EXPERTS):
#   base_env                 按 requirements-freeze-clean.txt 安装 base 环境
#   system_tmux              装 tmux (run_all_experts.sh 用它管 Qwen3-VL 服务)
#   qwen3_omni_service       Qwen3-Omni 推理依赖 (qwen3omni_deploy/setup.sh)
#   audio_consistency / asr_eval / emtion_consistency / style_consistency /
#   subject_consistency / id_consistency / action_eval
#                            LLM 客户端工程, 补少量辅助包; asr_eval 自动下载 Qwen3-ASR-1.7B
#   syncnet_python           独立 venv (.venv_syncnet, scenedetect 0.5.6.1) + 权重
#   TransNetV2               transnetv2-pytorch
#   audiobox-aesthetics      pip install -e . + checkpoint
#   facial_consistency       insightface + onnxruntime-gpu 1.20.2 (CUDA 12)
#   VBench2-Human_Anatomy    mmcv 2.1.0 / mmdet 3.3.0 / mmyolo 0.6.0 (源码) / CLIP
#   Q-Align                  独立 venv ($ENV_ROOT/qalign, torch 2.2.2+cu121)
#   consistency_segmented    SAM2 (源码) + SAM2 / GroundingDINO 权重
#   ocr                      paddlepaddle-gpu 3.1.0 (cu126) + paddleocr / paddlex
#   Syncformer               独立 venv (.venv_syncformer, transformers 4.40)
#   post_setup_fixes         post_setup_fixes.sh: 修正上面各步骤之间已知的包冲突
#
# 关键设计:
#   * 每个步骤用独立子 shell (bash -c '...') 跑, 避免环境变量污染下一步
#   * 单步失败不中止, 继续下一个, 最后打印 ✅/❌ 汇总表
#   * 每步单独日志: _setup_logs/<step>.<timestamp>.log
#   * 全部前台跑 + tee 落盘, 不使用 nohup/&/setsid
#
# 用法:
#   cd evaluation
#   bash setup_all.sh                                # 全量顺序装
#   bash setup_all.sh --only audio_consistency,ocr   # 只装某几个
#   bash setup_all.sh --skip base_env,Q-Align        # 跳过某几个
#   bash setup_all.sh --dry-run                      # 只打印计划
#   bash setup_all.sh --help                         # 显示帮助
#
# 建议实际跑用 tee 落顶层日志:
#   mkdir -p _setup_logs && bash setup_all.sh 2>&1 | tee _setup_logs/setup_all.$(date +%Y%m%d_%H%M).log
# =============================================================================

set -uo pipefail   # 不加 -e, 因为 continue-on-failure 是策略
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
ENV_ROOT="${ENV_ROOT:-$REPO_ROOT/envs}"
export REPO_ROOT MODEL_ROOT ENV_ROOT

# -----------------------------------------------------------------------------
# 基础路径
# -----------------------------------------------------------------------------
EVAL_ROOT="$REPO_ROOT/evaluation"
LOG_ROOT="$EVAL_ROOT/_setup_logs"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_ROOT"

# -----------------------------------------------------------------------------
# 颜色 & 日志
# -----------------------------------------------------------------------------
if [[ -t 1 ]]; then
    C_RED='\033[0;31m'; C_GRN='\033[0;32m'; C_YEL='\033[1;33m'
    C_BLU='\033[0;34m'; C_CYN='\033[0;36m'; C_RST='\033[0m'
else
    C_RED=''; C_GRN=''; C_YEL=''; C_BLU=''; C_CYN=''; C_RST=''
fi

log()    { echo -e "${C_BLU}[setup_all]${C_RST} $*"; }
ok()     { echo -e "${C_GRN}[setup_all]${C_RST} $*"; }
warn()   { echo -e "${C_YEL}[setup_all]${C_RST} $*"; }
err()    { echo -e "${C_RED}[setup_all]${C_RST} $*" >&2; }
section(){
    echo
    echo "============================================================"
    echo -e "${C_CYN}  $*${C_RST}"
    echo "============================================================"
}

# -----------------------------------------------------------------------------
# 步骤清单 (顺序即执行顺序)
# -----------------------------------------------------------------------------
ALL_EXPERTS=(
    base_env
    system_tmux
    qwen3_omni_service
    audio_consistency
    asr_eval
    emtion_consistency
    style_consistency
    subject_consistency
    id_consistency
    action_eval
    syncnet_python
    TransNetV2
    audiobox-aesthetics
    facial_consistency
    VBench2-Human_Anatomy
    Q-Align
    consistency_segmented
    ocr
    Syncformer
    post_setup_fixes
)

# -----------------------------------------------------------------------------
# 每个步骤的 setup 命令 (作为字符串, 由 orchestrator 通过 bash -c 起干净子 shell 执行)
#
# 设计原则:
#   * base_env 按 requirements-freeze-clean.txt 装好 torch / vllm / transformers /
#     flash_attn 等主库; 之后各 LLM 客户端工程只补少量辅助包并做 import 自检.
#   * 专家侧复用工程内的 setup*.sh; TransNetV2 / audiobox-aesthetics /
#     consistency_segmented 没有独立脚本, 由本文件内嵌 pip 步骤替代.
#   * 与 base 版本冲突的工程放进独立 venv: Q-Align (torch 2.2.2), syncnet_python
#     (scenedetect 0.5.6.1), Syncformer (transformers 4.40 / timm 0.9.16).
# -----------------------------------------------------------------------------
declare -A CMD=()

# =============================================================================
# base 环境 & 系统级准备
# =============================================================================

# ---- base_env ----
# 按 requirements-freeze-clean.txt 装 base 环境, 三条命令及参数含义见该文件顶部注释.
# 已按清单装好时可用 --skip base_env 跳过.
read -r -d '' _CMD_BASE <<'EOF' || true
set -uo pipefail
cd "$REPO_ROOT/evaluation"
REQ=requirements-freeze-clean.txt

python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' || {
    echo "[base_env] ❌ 需要 Python 3.12, 当前: $(python3 --version 2>&1)"
    exit 1
}

echo "[base_env] 1/3 构建工具"
python3 -m pip install --no-deps \
    Cython==3.2.9 numpy==2.2.6 setuptools==79.0.1 wheel==0.48.0 packaging==26.3 \
    ninja==1.13.0 psutil==7.1.0 cmake==4.1.0 || exit 1

echo "[base_env] 2/3 torch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0"
python3 -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 || exit 1

echo "[base_env] 3/3 $REQ"
python3 -m pip install --no-deps --no-build-isolation --ignore-installed \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    -r "$REQ" || exit 1

python3 - <<'PY' || exit 1
import torch, transformers, vllm
print("torch       :", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("vllm        :", vllm.__version__)
print("transformers:", transformers.__version__)
PY
EOF
CMD[base_env]="$_CMD_BASE"

# ---- 00. system_tmux ----
# orchestrator 用 tmux 后台跑 Omni + VL 服务, 必须先装
read -r -d '' _CMD_SYSTMUX <<'EOF' || true
set -uo pipefail
if command -v tmux >/dev/null 2>&1; then
    echo "[system_tmux] ✅ tmux 已就位: $(tmux -V)"
    exit 0
fi

echo "[system_tmux] tmux 未装, 尝试装 (需要 root 或 sudo)"
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq tmux || exit 1
elif command -v yum >/dev/null 2>&1; then
    yum install -y -q tmux || exit 1
else
    echo "[system_tmux] ❌ 找不到 apt / yum, 请自行装 tmux"
    exit 1
fi
echo "[system_tmux] ✅ 装完: $(tmux -V)"
EOF
CMD[system_tmux]="$_CMD_SYSTMUX"

# ---- 01. qwen3_omni_service ----
# 装 Qwen3-Omni 推理依赖 (qwen-omni-utils + decord + flash-attn, 由 qwen3omni_deploy/setup.sh 负责;
# 已装则跳过). Phase A 的工程在进程内加载 Qwen3-Omni, 不需要起服务.
# 同时检查 Phase B 的 Qwen3-VL 服务启动脚本是否就位 (本步不启动服务).
read -r -d '' _CMD_OMNISERVICE <<'EOF' || true
set -uo pipefail
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

OMNI_DEPLOY=$REPO_ROOT/qwen3omni_deploy
if [[ ! -f "$OMNI_DEPLOY/setup.sh" ]]; then
    echo "[qwen3_omni_service] qwen3omni_deploy/ not included in this release; skipping (needed only for prompt construction)."
    exit 0
fi

# 补丁: qwen_omni_utils.v2_5.audio_process 里 hard import audioread,
# 但 qwen_omni_utils wheel 的 setup.py 没把它列成传递依赖 → 手工补装
# ❗ 必须在 bash setup.sh 之前装, 因为工程内部 setup.sh 末尾会自行 import qwen_omni_utils 自检,
#    audioread 缺失则子脚本直接以非零退出, 往后拼接的补丁命令根本跑不到.
echo "[qwen3_omni_service] 前置补装 qwen_omni_utils 的隐式依赖: audioread"
python3 -m pip install audioread || exit 1

echo "[qwen3_omni_service] 调用 $OMNI_DEPLOY/setup.sh (幂等: 已装则跳过)"
cd "$OMNI_DEPLOY"
bash setup.sh || exit 1

echo "[qwen3_omni_service] 检查 VL 服务启动脚本 (Phase B 用)"
VL_DEPLOY=$REPO_ROOT/Qwen3-VL-235B_prompt
if [[ ! -f "$VL_DEPLOY/scripts/start_qwen3vl.sh" ]]; then
    echo "[qwen3_omni_service] ⚠️  找不到 $VL_DEPLOY/scripts/start_qwen3vl.sh (Phase B 会跑不了)"
else
    echo "[qwen3_omni_service] ✅ start_qwen3vl.sh 就位, 可由 run_all_experts.sh Phase B 拉起"
fi
EOF
CMD[qwen3_omni_service]="$_CMD_OMNISERVICE"

# =============================================================================
# 7 个 LLM 客户端工程 (共享 base 环境, 只需补少量辅助包)
# =============================================================================

# ---- 1. audio_consistency ----
# 依赖: qwen_omni_utils / transnetv2-pytorch / ffmpeg-python / librosa / decord
# (torch / transformers / vllm / flash_attn 由 base_env 安装; ffmpeg 为系统包)
read -r -d '' _CMD_AUDIOCONS <<'EOF' || true
set -uo pipefail
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

echo "[audio_consistency] 装 Qwen3-Omni 侧辅助包"
# 注意 audioread 是 qwen_omni_utils 的隐式依赖 (v2_5/audio_process.py hard import,
# 但未写进 wheel 的传递依赖), 必须显式加进来, 否则 import qwen_omni_utils 直接崩
python3 -m pip install \
    "qwen-omni-utils" \
    "transnetv2-pytorch" \
    "ffmpeg-python" \
    "librosa" \
    "decord" \
    "audioread" || exit 1

echo "[audio_consistency] import 自检"
python3 - <<'PY' || exit 1
import torch, transformers, qwen_omni_utils
from transnetv2_pytorch import TransNetV2
import ffmpeg, librosa, decord
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("qwen_omni_utils: ok")
print("transnetv2  :", sum(p.numel() for p in TransNetV2().parameters()), "params")
print("librosa     :", librosa.__version__)
print("decord      :", decord.__version__)
PY

echo "[audio_consistency] 检查 Qwen3-Omni 权重"
CKPT_DIR=$MODEL_ROOT/Qwen3-Omni/Qwen3-Omni-30B-A3B-Instruct
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "[audio_consistency] [WARN] 权重缺失: $CKPT_DIR"
else
    echo "[audio_consistency] ✅ 权重就位: $CKPT_DIR ($(ls "$CKPT_DIR" | wc -l) items)"
fi
EOF
CMD[audio_consistency]="$_CMD_AUDIOCONS"

# ---- 2. asr_eval ----
# 依赖: transformers==4.57.6 (base_env 已装, qwen-asr 固定此版本) + librosa + soundfile
# (Qwen3-ASR-1.7B 权重缺失时自动下载到 asr_eval/models/Qwen3-ASR-1.7B/)
read -r -d '' _CMD_ASR <<'EOF' || true
set -uo pipefail
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

echo "[asr_eval] 装辅助包"
python3 -m pip install \
    "librosa" \
    "jiwer" \
    "zhconv" || exit 1

echo "[asr_eval] import 自检"
python3 - <<'PY' || exit 1
import torch, transformers, librosa
try:
    import jiwer; print("jiwer:", jiwer.__version__)
except Exception as e:
    print("jiwer FAIL:", e)
try:
    import zhconv; print("zhconv: ok")
except Exception as e:
    print("zhconv FAIL:", e)
print("librosa:", librosa.__version__)
PY

echo "[asr_eval] 检查 Qwen3-ASR 权重"
CKPT_DIR=$REPO_ROOT/evaluation/asr_eval/models/Qwen3-ASR-1.7B
if [[ ! -f "$CKPT_DIR/config.json" ]]; then
    echo "[asr_eval] 下载 Qwen/Qwen3-ASR-1.7B -> $CKPT_DIR"
    CKPT_DIR="$CKPT_DIR" python3 - <<'PY' || exit 1
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-ASR-1.7B", local_dir=os.environ["CKPT_DIR"])
PY
fi
echo "[asr_eval] ✅ 权重就位: $CKPT_DIR ($(ls "$CKPT_DIR" | wc -l) items)"
EOF
CMD[asr_eval]="$_CMD_ASR"

# ---- 3. emtion_consistency ----
# 与 audio_consistency 同一技术栈 (Qwen3-Omni)
read -r -d '' _CMD_EMO <<'EOF' || true
set -uo pipefail
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

echo "[emtion_consistency] 装 Qwen3-Omni 侧辅助包 (与 audio_consistency 共享)"
# audioread: qwen_omni_utils 隐式依赖 (同 audio_consistency 说明)
python3 -m pip install \
    "qwen-omni-utils" \
    "ffmpeg-python" \
    "librosa" \
    "decord" \
    "audioread" || exit 1

echo "[emtion_consistency] import 自检"
python3 -c "
import qwen_omni_utils, librosa, decord, ffmpeg
print('emtion_consistency deps OK')
" || exit 1
EOF
CMD[emtion_consistency]="$_CMD_EMO"

# ---- 4. style_consistency ----
# 走 vLLM HTTP OpenAI 兼容 (openai 客户端); 只需要 openai + 图像/视频编解码
read -r -d '' _CMD_STYLE <<'EOF' || true
set -uo pipefail
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

echo "[style_consistency] 装 vLLM 客户端侧辅助包 (openai 客户端 + 编解码工具)"
python3 -m pip install \
    "openai" \
    "qwen-vl-utils" \
    "ffmpeg-python" \
    "decord" || exit 1

echo "[style_consistency] import 自检"
python3 -c "
import openai, qwen_vl_utils, ffmpeg, decord
print('style_consistency deps OK')
print('openai:', openai.__version__)
" || exit 1

# 说明: style/subject/id/action_eval 走 Qwen3-VL vLLM 服务, 由 run_all_experts.sh 在 Phase B
# 自动启停 (手动启动见 t2av-bench/SETUP.md §3). 本 setup 不启动 vLLM.
echo "[style_consistency] NOTE: Qwen3-VL vLLM 服务由 run_all_experts.sh 在 Phase B 自动启停"
EOF
CMD[style_consistency]="$_CMD_STYLE"

# ---- 5. subject_consistency ----
# 与 style_consistency 完全同一技术栈, 依赖包已在 style_consistency 装完时全就位
read -r -d '' _CMD_SUBJ <<'EOF' || true
set -uo pipefail
echo "[subject_consistency] 与 style_consistency 共享依赖 (openai/qwen_vl_utils/ffmpeg-python/decord)"
python3 -c "
import openai, qwen_vl_utils, ffmpeg, decord
print('subject_consistency deps OK (all shared)')
" || {
    echo "[subject_consistency] 依赖缺失, 请先 bash setup_all.sh --only style_consistency 再重试"
    exit 1
}
echo "[subject_consistency] NOTE: Qwen3-VL vLLM 服务由 run_all_experts.sh 在 Phase B 自动启停"
EOF
CMD[subject_consistency]="$_CMD_SUBJ"

# ---- 6. id_consistency ----
read -r -d '' _CMD_ID <<'EOF' || true
set -uo pipefail
echo "[id_consistency] 与 style_consistency 共享依赖 (openai/qwen_vl_utils/ffmpeg-python/decord)"
python3 -c "
import openai, qwen_vl_utils, ffmpeg, decord
print('id_consistency deps OK (all shared)')
" || {
    echo "[id_consistency] 依赖缺失, 请先 bash setup_all.sh --only style_consistency 再重试"
    exit 1
}
echo "[id_consistency] NOTE: Qwen3-VL vLLM 服务由 run_all_experts.sh 在 Phase B 自动启停"
EOF
CMD[id_consistency]="$_CMD_ID"

# ---- 7. action_eval (VL judge yes/no, 依赖同 style_consistency) ----
# 依赖: openai + qwen-vl-utils + ffmpeg-python + decord (与 style_consistency 共享)
# 前置: 阶段 1 需要 audio_consistency 已切好 clips (软链接复用), 由 run_all.sh 内部检查
read -r -d '' _CMD_ACTION <<'EOF' || true
set -uo pipefail
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

echo "[action_eval] 检查 VL 客户端侧辅助包 (openai/qwen_vl_utils/ffmpeg-python/decord)"
python3 -c "
import openai, qwen_vl_utils, ffmpeg, decord
print('action_eval deps OK (与 style_consistency 共享)')
" || {
    echo "[action_eval] 依赖缺失, 请先 bash setup_all.sh --only style_consistency 装齐"
    exit 1
}

echo "[action_eval] 检查 questions 与 system prompt 就位"
QDIR=$REPO_ROOT/t2av-bench/questions
Q_FILE="$QDIR/action_eval_llm.jsonl"
if [[ ! -f "$Q_FILE" ]]; then
    echo "[action_eval] ⚠️  questions 文件缺失: $Q_FILE"
    echo "[action_eval]    如需重建: cd action_eval && python3 build_questions.py"
else
    echo "[action_eval] ✅ questions 就位: $Q_FILE ($(wc -l < "$Q_FILE") 行)"
fi

echo "[action_eval] NOTE: 跑评测前 Phase B 的 Qwen3-VL vLLM 服务需就绪"
echo "[action_eval] NOTE: 依赖 audio_consistency 的 shotcut_manifest.jsonl (软链接复用)"
EOF
CMD[action_eval]="$_CMD_ACTION"

# =============================================================================
# 9 个专家工程 (Syncformer 最后)
# =============================================================================

# ---- 7. syncnet_python ----
CMD[syncnet_python]="cd '$EVAL_ROOT/syncnet_python' && bash setup.sh"

# ---- 8. TransNetV2 (无独立 setup 脚本) ----
read -r -d '' _CMD_TRANSNETV2 <<'EOF' || true
set -uo pipefail
cd $REPO_ROOT/evaluation/TransNetV2
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

echo '[transnetv2] pip install transnetv2-pytorch + ffmpeg-python (imageio-ffmpeg 视需要)'
python3 -m pip install \
    transnetv2-pytorch ffmpeg-python || exit 1

echo "[transnetv2] ffmpeg = $(command -v ffmpeg)"
ffmpeg -version 2>/dev/null | head -1 || true

echo '[transnetv2] 自检'
python3 - <<'PY' || exit 1
import torch, numpy, pandas, ffmpeg
from transnetv2_pytorch import TransNetV2
m = TransNetV2()
print("torch     :", torch.__version__, "cuda:", torch.cuda.is_available())
print("numpy     :", numpy.__version__, " pandas:", pandas.__version__)
print("TransNetV2:", sum(p.numel() for p in m.parameters()), "params")
PY
EOF
CMD[TransNetV2]="$_CMD_TRANSNETV2"

# ---- 9. audiobox-aesthetics (无独立脚本; 使用系统 ffmpeg / ffprobe) ----
read -r -d '' _CMD_AUDIOBOX <<'EOF' || true
set -uo pipefail
cd $REPO_ROOT/evaluation/audiobox-aesthetics

echo '[audiobox] 系统 ffmpeg / ffprobe:'
which ffmpeg && which ffprobe
ffmpeg  -version | head -1
ffprobe -version | head -1

echo '[audiobox] pip install -e . (audiobox_aesthetics editable)'
python3 -m pip install -e . || exit 1

echo '[audiobox] import 自检'
python3 - <<'PY' || exit 1
import audiobox_aesthetics, torch, torchaudio
print('audiobox_aesthetics:', audiobox_aesthetics.__file__)
print('torch:', torch.__version__)
print('torchaudio:', torchaudio.__version__)
PY

CKPT=$MODEL_ROOT/audiobox-aesthetics/checkpoint.pt
if [[ ! -s "$CKPT" ]]; then
    echo "[audiobox] 下载 checkpoint -> $CKPT"
    mkdir -p "$(dirname "$CKPT")"
    wget -O "$CKPT" https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt || exit 1
fi
echo "[audiobox] 权重就位: $CKPT ($(stat -c%s "$CKPT") bytes)"
EOF
CMD[audiobox-aesthetics]="$_CMD_AUDIOBOX"

# ---- 10. facial_consistency (insightface + onnxruntime-gpu cu12) ----
# onnxruntime-gpu 1.20.2 (CUDA 12 build, 与 requirements-freeze-clean.txt 一致) 的
# CUDAExecutionProvider 需要 libcudnn_adv.so.9 / libcublas.so / libcudart.so 等动态库.
# 这些库在 base 的 nvidia-* 包里 (site-packages/nvidia/*/lib), 但默认 LD_LIBRARY_PATH
# 不包含, 会静默回退 CPU, 所以先把它们加进 LD_LIBRARY_PATH.
read -r -d '' _CMD_FACIAL <<'EOF' || true
set -uo pipefail
NV="$(python3 -c 'import nvidia; print(list(nvidia.__path__)[0])')" || {
    echo "[facial_consistency] ❌ 找不到 nvidia-* 包, 请先跑 base_env"
    exit 1
}
export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cublas/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO_ROOT/evaluation/facial_consistency"
ONNXRT_VERSION=1.20.2 bash setup_env.sh
EOF
CMD[facial_consistency]="$_CMD_FACIAL"

# ---- 11. VBench2-Human_Anatomy ----
# 内含 mmengine / mmcv==2.1.0 / mmdet==3.3.0 / mmyolo (源码) / lvis / albumentations + CLIP 权重
# ⚠️ mmcv==2.1.0 在 numpy 2.x 上可能装不上 (要看). 若失败, 详情看 log 决定是否降 numpy.
# 补丁: VBench2 setup_env.sh 里 `pip install -U openmim gdown ...` 会拖来 openxlab,
# 而 openxlab 精确 pin setuptools==60.2.0, 会把 base 的 setuptools 从 79 拉低到 60.2,
# 导致 py3.12 的 pkg_resources 因 pkgutil.ImpImporter 缺失而崩, 连带把 filelock/rich/timm
# 也降版, 破坏 vllm/torch 生态. 用 PIP_CONSTRAINT 全局约束禁止降低这些包版本;
# setup 完毕后卸掉 openxlab/opendatalab 避免残留.
_VBENCH_CONSTRAINT="/tmp/vbench_pip_constraint.$$"
read -r -d '' _CMD_VBENCH <<'EOF' || true
set -uo pipefail

# 写一份 pip 全局约束文件, 禁止关键包被降版
CONSTRAINT="/tmp/vbench_pip_constraint.$$"
cat > "$CONSTRAINT" <<'CONS'
setuptools>=77.0.3,<80
filelock>=3.16.1
rich>=13.9.4
packaging>=24.0
pytz>=2024.1
requests>=2.32,<3
urllib3>=2.0,<3
CONS
export PIP_CONSTRAINT="$CONSTRAINT"
echo "[VBench2] PIP_CONSTRAINT=$PIP_CONSTRAINT (禁止 setuptools/filelock/rich 被降版)"
cat "$CONSTRAINT"

cd $REPO_ROOT/evaluation/VBench2-Human_Anatomy

# 前置手工 pip 装 mm 全家桶 (--no-build-isolation 关键)
# 背景: mmcv==2.1.0 的 setup.py 第 5 行 `from pkg_resources import ...`, 但
#       setuptools 81+ 已官方剔离 pkg_resources. 如果走默认 pip build isolation,
#       pip 会在 /tmp/pip-build-env-*/ 创建隔离环境并自动拉最新 setuptools 84,
#       进入隔离环境后 mmcv setup.py 直接报 No module named 'pkg_resources'.
#       --no-build-isolation 强制 mmcv 编译时使用 base 的 setuptools 79
#       (旧到还带 pkg_resources, 又新到 py3.12 能用), 因此可以编译通过.
# 装完后, setup_env.sh 里的 need_mm 自检会发现已符合版本要求, 直接跳过 mim install.
echo "[VBench2] 前置编译装 mmengine / mmcv==2.1.0 / mmdet==3.3.0 (--no-build-isolation, 以免 setuptools 81+ 无 pkg_resources)"
python3 -m pip install --no-build-isolation \
    -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.8.0/index.html \
    "mmengine>=0.10.5" \
    "mmcv==2.1.0" \
    "mmdet==3.3.0" || { rm -f "$CONSTRAINT"; exit 1; }

bash setup_env.sh || { rm -f "$CONSTRAINT"; exit 1; }

# 清理: 卸掉 openxlab/opendatalab (VBench2 不真正使用, 只是 openmim 拖来的; 它们的
# 精确 pin 是破坏 base 的元凶, 卸掉能防止后续 pip 操作被它们干扰)
echo "[VBench2] 清理 openxlab/opendatalab (释放对 setuptools/filelock 的 pin)"
python3 -m pip uninstall -y openxlab opendatalab 2>/dev/null || true

unset PIP_CONSTRAINT
rm -f "$CONSTRAINT"
echo "[VBench2] ✅ setup_env.sh 完成 + 清理完毕"
EOF
CMD[VBench2-Human_Anatomy]="$_CMD_VBENCH"

# ---- 12. Q-Align (独立 venv, 不动主环境) ----
# $ENV_ROOT/qalign/ (torch 2.2.2+cu121) 已存在时只做检查, 不重装; 不存在时跑 Q-Align/setup_env.sh 建 venv.
read -r -d '' _CMD_QALIGN <<'EOF' || true
set -uo pipefail
VENV=$ENV_ROOT/qalign
if [[ ! -d "$VENV" ]]; then
    echo "[Q-Align] venv 不存在, 需要跑 Q-Align/setup_env.sh 建 venv"
    echo "[Q-Align] 建 venv (会在 base 之外的独立 python 环境里装 torch 2.2.2+cu121)"
    cd $REPO_ROOT/evaluation/Q-Align
    bash setup_env.sh || exit 1
else
    echo "[Q-Align] ✅ venv 已就位: $VENV"
    "$VENV/bin/python" -c "
import torch, transformers
print('qalign torch      :', torch.__version__)
print('qalign transformers:', transformers.__version__)
try:
    from q_align.evaluate.scorer import QAlignVideoScorer
    print('q_align import OK')
except Exception as e:
    print('q_align import FAIL:', e)
" || {
        echo "[Q-Align] venv 存在但 import 失败, 重跑 setup_env.sh"
        cd $REPO_ROOT/evaluation/Q-Align
        bash setup_env.sh || exit 1
    }
fi
EOF
CMD[Q-Align]="$_CMD_QALIGN"

# ---- 13. consistency_segmented ----
# 复用 base 的 torch 2.8.0 (满足 SAM2 的 torch>=2.5.1 要求), 只装:
#   - requirements.txt (含 CLIP)
#   - SAM2 (源码 -e, --no-deps)
#   - SAM2 权重 + GroundingDINO 权重
read -r -d '' _CMD_CS <<'EOF' || true
set -uo pipefail

PROJ=$REPO_ROOT/evaluation/consistency_segmented
cd "$PROJ"

echo "[consistency_segmented] step 1: 装 requirements.txt (不含 torch)"
if [[ -f requirements.txt ]]; then
    python3 -m pip install -r requirements.txt || exit 1
fi

echo "[consistency_segmented] step 2: 检查 base 的 torch"
python3 -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'avail:', torch.cuda.is_available())
assert tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2,5), \
    f'torch {torch.__version__} < 2.5, SAM2 可能不兼容'
"

echo "[consistency_segmented] step 3: 装 SAM 2 (源码, --no-deps 防止拉 torch)"
# 补丁: git clone 偶发网络 502, git 本身不像 pip 一样自动重试, 手工 5 次
SAM2_SRC="$ENV_ROOT/src/sam2"
if [[ ! -d "$SAM2_SRC/.git" ]]; then
    mkdir -p "$ENV_ROOT/src"
    rm -rf "$SAM2_SRC"
    _clone_ok=0
    for _i in 1 2 3 4 5; do
        if git clone https://github.com/facebookresearch/sam2.git "$SAM2_SRC"; then
            _clone_ok=1; break
        fi
        echo "[consistency_segmented] git clone sam2 尝试 $_i 失败, 15s 后重试..."
        rm -rf "$SAM2_SRC"
        sleep 15
    done
    if [[ "$_clone_ok" -ne 1 ]]; then
        echo "[consistency_segmented] ❌ git clone sam2 连续 5 次失败, 请检查网络"
        exit 1
    fi
fi
python3 -m pip install -e "$SAM2_SRC" --no-deps || exit 1

# SAM2 有一些非 torch 的运行时依赖, 单独补齐 (不覆盖 torch)
python3 -m pip install \
    "hydra-core" "iopath" "pillow>=9.4.0" || exit 1

echo "[consistency_segmented] step 4: 验证 sam2 + clip 能 import"
python3 -c "
import torch, sam2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
import clip
print('sam2 + clip ok')
"

echo "[consistency_segmented] step 5: SAM 2 权重"
mkdir -p "$PROJ/pretrained"
SAM2_CKPT="$PROJ/pretrained/sam2.1_hiera_large.pt"
if [[ -f "$SAM2_CKPT" ]] && [[ $(stat -c%s "$SAM2_CKPT") -gt 800000000 ]]; then
    echo "[consistency_segmented] SAM2 权重已就位: $SAM2_CKPT"
else
    wget -O "$SAM2_CKPT" \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt || exit 1
fi

echo "[consistency_segmented] step 6: GroundingDINO 权重"
GDINO_DIR="$PROJ/pretrained/grounding-dino-tiny"
GDINO_MAIN="$GDINO_DIR/model.safetensors"
if [[ -f "$GDINO_MAIN" ]] && [[ $(stat -c%s "$GDINO_MAIN") -gt 600000000 ]]; then
    echo "[consistency_segmented] GroundingDINO 权重已就位"
else
    python3 - <<'PY' || exit 1
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="IDEA-Research/grounding-dino-tiny",
    local_dir="./pretrained/grounding-dino-tiny",
)
PY
fi

echo "[consistency_segmented] step 7: 验证 grounding dino"
python3 -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
p = './pretrained/grounding-dino-tiny'
AutoProcessor.from_pretrained(p)
m = AutoModelForZeroShotObjectDetection.from_pretrained(p)
print('gdino params:', sum(x.numel() for x in m.parameters())/1e6, 'M')
"
EOF
CMD[consistency_segmented]="$_CMD_CS"

# ---- 14. ocr (paddlepaddle-gpu 3.1.0 cu126) ----
# 直接调工程内的 rebuild_env.sh (它做 cu126 wheel + paddleocr + paddlex 三件套)
# 注意: 本 orchestrator 会走 CPU 参考路线跑评测, 所以运行时不需要 source activate_paddle_gpu.sh
#
# 补丁: paddlepaddle-gpu 声明依赖 nvidia-nccl-cu12==2.25.1, pip 装完会把 base 的
#      nvidia-nccl-cu12 从 2.27.3 (torch 2.8+cu128 需要) 降到 2.25.1, 导致
#      `import torch` 报 undefined symbol: ncclCommWindowRegister (NCCL 2.26+ API).
#      rebuild_env.sh 末尾自带 paddleocr 全链路自检, 而自检会 import torch, 一定失败.
# 策略: 用 ";" + "_RC=$?" + "exit \$_RC" 确保:
#   1) 无论 rebuild_env.sh 自检成败, 都强制拉回 nccl 2.27.3 和 cudnn 9.10.2.21;
#   2) SUMMARY 中 ocr 的 ✅/❌ 状态仍忠实反映 rebuild_env.sh 本身的 rc, 不被补丁掩盖.
# 注意: paddle 不仅会把 nvidia-nccl-cu12 降到 2.25.1, 也会把 nvidia-cudnn-cu12 从
#       base 的 9.10.2.21 降到 9.5.1.17, 导致 torch.backends.cudnn.version() 报
#       "PyTorch was compiled against (9,8,0) but found runtime (9,5,1)". 两个都需拉回.
CMD[ocr]="cd '$EVAL_ROOT/ocr' && bash rebuild_env.sh; _RC=\$?; \
  echo '[ocr] 无条件拉回 nvidia-nccl-cu12 到 2.27.3 + nvidia-cudnn-cu12 到 9.10.2.21 (paddle 会双降版破坏 torch)' && \
  python3 -m pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.27.3 nvidia-cudnn-cu12==9.10.2.21 && \
  python3 -c 'import torch; print(\"[ocr] torch NCCL:\", torch.cuda.nccl.version(), \"cudnn:\", torch.backends.cudnn.version(), \"cuda_avail:\", torch.cuda.is_available())'; \
  exit \$_RC"

# ---- 15. Syncformer (独立 venv) ----
# Syncformer 需要 transformers 4.40 / timm 0.9.16, 与 base 冲突. Syncformer/setup.sh 建
# 独立 venv (.venv_syncformer, --system-site-packages 复用 base 的 torch / av), 并准备权重;
# Syncformer/run_all.sh 自动使用该 venv.
CMD[Syncformer]="cd '$EVAL_ROOT/Syncformer' && bash setup.sh"

# ---- 16. post_setup_fixes ----
# 修正上面各步骤之间已知的包冲突 (opencv / transformers / librosa 等), 详见脚本内注释.
CMD[post_setup_fixes]="bash '$EVAL_ROOT/post_setup_fixes.sh'"

# -----------------------------------------------------------------------------
# 参数解析
# -----------------------------------------------------------------------------
ONLY_LIST=""
SKIP_LIST=""
DRY_RUN=0

show_help() {
    sed -n '2,49p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)     ONLY_LIST="$2"; shift 2 ;;
        --only=*)   ONLY_LIST="${1#*=}"; shift ;;
        --skip)     SKIP_LIST="$2"; shift 2 ;;
        --skip=*)   SKIP_LIST="${1#*=}"; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  show_help ;;
        *)  err "unknown argument: $1"; err "run 'bash setup_all.sh --help' for usage"; exit 2 ;;
    esac
done

declare -A ONLY_SET=()
declare -A SKIP_SET=()
if [[ -n "$ONLY_LIST" ]]; then
    IFS=',' read -r -a _tmp <<<"$ONLY_LIST"
    for x in "${_tmp[@]}"; do
        x="$(echo "$x" | xargs)"; [[ -z "$x" ]] && continue
        ONLY_SET["$x"]=1
    done
fi
if [[ -n "$SKIP_LIST" ]]; then
    IFS=',' read -r -a _tmp <<<"$SKIP_LIST"
    for x in "${_tmp[@]}"; do
        x="$(echo "$x" | xargs)"; [[ -z "$x" ]] && continue
        SKIP_SET["$x"]=1
    done
fi

_valid_expert() {
    local name="$1"
    for e in "${ALL_EXPERTS[@]}"; do
        [[ "$e" == "$name" ]] && return 0
    done
    return 1
}
for name in "${!ONLY_SET[@]}"; do
    if ! _valid_expert "$name"; then
        err "--only 中的 '$name' 不是合法专家名, 合法列表: ${ALL_EXPERTS[*]}"; exit 2
    fi
done
for name in "${!SKIP_SET[@]}"; do
    if ! _valid_expert "$name"; then
        err "--skip 中的 '$name' 不是合法专家名, 合法列表: ${ALL_EXPERTS[*]}"; exit 2
    fi
done

PLAN=()
for e in "${ALL_EXPERTS[@]}"; do
    if [[ ${#ONLY_SET[@]} -gt 0 ]] && [[ -z "${ONLY_SET[$e]:-}" ]]; then
        continue
    fi
    if [[ -n "${SKIP_SET[$e]:-}" ]]; then
        continue
    fi
    PLAN+=("$e")
done

if [[ ${#PLAN[@]} -eq 0 ]]; then
    err "PLAN is empty. Check --only / --skip filters."; exit 2
fi

# -----------------------------------------------------------------------------
# 打印计划
# -----------------------------------------------------------------------------
section "SETUP PLAN  (ts=$TS)"
log "EVAL_ROOT = $EVAL_ROOT"
log "LOG_ROOT  = $LOG_ROOT"
log "DRY_RUN   = $DRY_RUN"
log "python3   = $(command -v python3) ($(python3 --version 2>&1))"
log "will setup ${#PLAN[@]} experts in this order:"
i=0
for e in "${PLAN[@]}"; do
    i=$((i+1))
    printf "   %2d. %s\n" "$i" "$e"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    warn "DRY_RUN=1, exit without doing anything."
    exit 0
fi

# -----------------------------------------------------------------------------
# 主循环
# -----------------------------------------------------------------------------
declare -A RESULT_STATUS=()
declare -A RESULT_ELAPSED=()
declare -A RESULT_LOGFILE=()

TOTAL_START="$(date +%s)"

for expert in "${PLAN[@]}"; do
    section "SETUP  [$expert]  ($(date '+%H:%M:%S'))"
    LOGFILE="$LOG_ROOT/${expert}.${TS}.log"
    RESULT_LOGFILE["$expert"]="$LOGFILE"

    if [[ -z "${CMD[$expert]:-}" ]]; then
        err "[$expert] 未定义 CMD, skip"
        RESULT_STATUS["$expert"]="fail"
        RESULT_ELAPSED["$expert"]=0
        continue
    fi

    start="$(date +%s)"
    log "logfile: $LOGFILE"

    bash -c "${CMD[$expert]}" 2>&1 | tee "$LOGFILE"
    ret="${PIPESTATUS[0]}"

    end="$(date +%s)"
    elapsed=$((end - start))
    RESULT_ELAPSED["$expert"]=$elapsed

    if [[ "$ret" -eq 0 ]]; then
        RESULT_STATUS["$expert"]="ok"
        ok "[$expert] SUCCESS in ${elapsed}s  (log: $LOGFILE)"
    else
        RESULT_STATUS["$expert"]="fail"
        err "[$expert] FAILED  rc=$ret  elapsed=${elapsed}s  (log: $LOGFILE)"
        err "[$expert] 继续下一个 (continue-on-failure)."
    fi
done

# -----------------------------------------------------------------------------
# 汇总
# -----------------------------------------------------------------------------
TOTAL_END="$(date +%s)"
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

section "SETUP SUMMARY"
n_ok=0; n_fail=0
for expert in "${PLAN[@]}"; do
    st="${RESULT_STATUS[$expert]}"
    el="${RESULT_ELAPSED[$expert]}"
    lg="${RESULT_LOGFILE[$expert]}"
    if [[ "$st" == "ok" ]]; then
        printf "  ${C_GRN}✅  %-25s${C_RST}  %5ds   log=%s\n" "$expert" "$el" "$lg"
        n_ok=$((n_ok+1))
    else
        printf "  ${C_RED}❌  %-25s${C_RST}  %5ds   log=%s\n" "$expert" "$el" "$lg"
        n_fail=$((n_fail+1))
    fi
done

echo
log "total: ${n_ok}/${#PLAN[@]} succeeded, ${n_fail}/${#PLAN[@]} failed."
log "total elapsed: ${TOTAL_ELAPSED}s"
log "detail logs in: $LOG_ROOT/"

if [[ "$n_fail" -gt 0 ]]; then
    warn "有失败项. 请查看对应 log 文件, 手动排查后单独重跑:"
    warn "  bash setup_all.sh --only <expert-name>"
    exit 1
fi
exit 0
