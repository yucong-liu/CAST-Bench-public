#!/usr/bin/env bash
# =============================================================================
# post_setup_fixes.sh  —  修正 setup_all.sh 各步骤之间已知的包冲突 (8 类)
#
# setup_all.sh 的最后一步 (post_setup_fixes) 会自动调用本脚本, 也可单独运行:
#   mkdir -p _setup_logs && bash post_setup_fixes.sh 2>&1 | tee _setup_logs/post_setup.$(date +%Y%m%d_%H%M).log
# (如需代理才能访问 pypi / huggingface, 请先自行 export http_proxy / https_proxy)
#
# 幂等: 可重复运行. 已到位的步骤会跳过.
# =============================================================================
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_ROOT

EVAL_ROOT=$REPO_ROOT/evaluation
SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ -z "$SITE" || ! -d "$SITE" ]]; then
    echo "[ERROR] 找不到 python3 的 site-packages 目录: '$SITE'" >&2
    exit 1
fi
PIP="python3 -m pip install --no-cache-dir"

# 代理提示 (下载 DINOv2 / pip 可能需要访问外网)
if [[ -z "${http_proxy:-}" ]]; then
    echo "[INFO] 未检测到 http_proxy. 若节点无法直连外网, 请先 export http_proxy / https_proxy 再跑, 否则 DINOv2 下载可能卡住."
fi

echo "==================================================================="
echo " [1/8] opencv: 清理三包冲突, 只保留单一 headless (修 VL 服务 500)"
echo "   症状: cv2.videoio_registry has no attribute getStreamBufferedBackends"
echo "==================================================================="
# 卸掉所有 opencv 变体 + 手动清残留 cv2 目录 (多包混装会残留)
python3 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true
rm -rf "$SITE/cv2"
$PIP "opencv-python-headless==4.12.0.88"
python3 -c "import cv2; assert hasattr(cv2.videoio_registry,'getStreamBufferedBackends'), 'cv2 修复失败'; print('  cv2', cv2.__version__, 'OK')"

echo "==================================================================="
echo " [2/8] transformers: 固定为 4.57.6"
echo "   4.57.6 同时满足 asr_eval(==4.57.6) / vllm(>=4.55.2) / Qwen3-Omni / Qwen3-VL"
echo "==================================================================="
$PIP "transformers==4.57.6" "tokenizers==0.22.2"
python3 - <<'PY'
from transformers.integrations import use_kernel_forward_from_hub
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3VLForConditionalGeneration
import transformers; print("  transformers", transformers.__version__, "(asr/Omni/VL 能力 OK)")
PY

echo "==================================================================="
echo " [3/8] librosa: 固定为 0.10.2.post1 (1.0.0 移除了 audioread fallback)"
echo "   症状: id_consistency transcribe 100 条全 err: Invalid file <FFmpegAudioFile>"
echo "==================================================================="
$PIP "librosa==0.10.2.post1"
python3 -c "import librosa; print('  librosa', librosa.__version__)"

echo "==================================================================="
echo " [4/8] 缺包: nagisa (asr_eval) + meeteval (id_consistency judge)"
echo "==================================================================="
$PIP nagisa meeteval
python3 -c "import nagisa, meeteval; print('  nagisa/meeteval OK')"

echo "==================================================================="
echo " [5/8] ocr: 装 paddlex[ocr] 缺的文档依赖 (不含 opencv-contrib, 避免污染 cv2)"
echo "==================================================================="
$PIP latex2mathml lxml openpyxl premailer cssselect cssutils \
     einops ftfy imagesize Jinja2 pyclipper pypdfium2 python-bidi \
     regex "safetensors>=0.7.0" scikit-learn scipy sentencepiece shapely tiktoken "tokenizers>=0.19"
# paddlex 的 ocr extra 检查按发行包名查 opencv-contrib-python; 我们用 headless 代替,
# 故伪造一个空 dist-info 骗过检查, 不实际安装 opencv-contrib (否则重演 [1] 的 cv2 冲突).
CONTRIB_STUB="$SITE/opencv_contrib_python-4.10.0.84.dist-info"
if ! python3 -c "import importlib.metadata as m; m.version('opencv-contrib-python')" 2>/dev/null; then
    mkdir -p "$CONTRIB_STUB"
    printf 'Metadata-Version: 2.1\nName: opencv-contrib-python\nVersion: 4.10.0.84\n' > "$CONTRIB_STUB/METADATA"
    : > "$CONTRIB_STUB/RECORD"
    echo "manual-stub" > "$CONTRIB_STUB/INSTALLER"
fi
python3 - <<'PY'
from paddlex.utils import deps
deps.is_dep_available.cache_clear(); deps.is_extra_available.cache_clear()
assert deps.is_extra_available("ocr"), "paddlex[ocr] 仍缺依赖"
import cv2; assert hasattr(cv2.videoio_registry,'getStreamBufferedBackends'), "cv2 被 ocr 步骤污染了!"
print("  paddlex[ocr] OK, cv2 仍干净", cv2.__version__)
PY

echo "==================================================================="
echo " [6/8] DINOv2 权重: 校验完整性, 残缺则重下"
echo "   症状: consistency_segmented: PytorchStreamReader failed finding central directory"
echo "==================================================================="
DINO="$(python3 -c 'import torch; print(torch.hub.get_dir())')/checkpoints/dinov2_vitb14_pretrain.pth"
NEED_DL=1
if [[ -f "$DINO" ]]; then
    SZ=$(stat -c%s "$DINO" 2>/dev/null || echo 0)
    [[ "$SZ" -gt 300000000 ]] && NEED_DL=0 && echo "  已存在完整权重 ($((SZ/1024/1024))MB), 跳过"
fi
if [[ "$NEED_DL" == "1" ]]; then
    mkdir -p "$(dirname "$DINO")"; rm -f "$DINO"
    wget -q --show-progress -O "$DINO" \
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
    python3 -c "import torch; torch.load('$DINO',map_location='cpu',weights_only=True); print('  DINOv2 权重 OK')"
fi

echo "==================================================================="
echo " [7/8] Syncformer: 独立 venv (transformers 4.40.*), 隔离 base 4.57.6"
echo "   Syncformer/run_all.sh 自动使用该 venv"
echo "==================================================================="
SF_VENV="$EVAL_ROOT/Syncformer/.venv_syncformer"
if [[ ! -x "$SF_VENV/bin/python" ]]; then
    python3 -m venv --system-site-packages "$SF_VENV"
    "$SF_VENV/bin/pip" install --no-cache-dir \
        "transformers==4.40.*" "tokenizers>=0.19,<0.20" "timm==0.9.16" omegaconf einops huggingface_hub
fi
"$SF_VENV/bin/python" -c "import transformers; assert transformers.__version__.startswith('4.40'); print('  Syncformer venv transformers', transformers.__version__)"

echo "==================================================================="
echo " [8/8] syncnet_python: 独立 venv (scenedetect 0.5.6.1 旧 API), 隔离 base 的 scenedetect"
echo "   症状: 全部 fail/nan: scenedetect 新版本缺 VideoManager"
echo "   syncnet_python/run_all.sh 自动使用该 venv"
echo "==================================================================="
SN_VENV="$EVAL_ROOT/syncnet_python/.venv_syncnet"
if [[ ! -x "$SN_VENV/bin/python" ]]; then
    python3 -m venv --system-site-packages "$SN_VENV"
    "$SN_VENV/bin/pip" install --no-cache-dir \
        "scenedetect==0.5.6.1" python_speech_features
fi
"$SN_VENV/bin/python" -c "from scenedetect.video_manager import VideoManager; import scenedetect; print('  syncnet venv scenedetect', scenedetect.__version__, '(旧 API OK)')"

echo ""
echo "==================================================================="
echo " 全部补丁完成. base 关键版本快照:"
echo "==================================================================="
python3 - <<'PY' 2>&1 | grep -viE "RequestsDep|warnings.warn"
import importlib.metadata as m
for p in ["transformers","tokenizers","librosa","opencv-python-headless","scenedetect","nagisa","meeteval"]:
    try: print(f"   {p:28s} {m.version(p)}")
    except: print(f"   {p:28s} (未装)")
PY
echo "   Syncformer venv  -> transformers 4.40.*  (隔离)"
echo "   syncnet venv     -> scenedetect 0.5.6.1  (隔离)"
echo "   Q-Align venv     -> 由 setup_all.sh 就位  (隔离)"
