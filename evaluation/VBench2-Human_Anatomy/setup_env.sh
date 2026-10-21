#!/usr/bin/env bash
# =====================================================================
# setup_env.sh — VBench2 Human_Anatomy 一键配置环境脚本（幂等可重入）
#
# 目的: 在一台新的(或半新)的 GPU 节点上, 一次执行就把 batch_eval.py /
#       run_all.sh 跑评测所需的一切准备好, 跑完之后直接可以:
#           bash run_all.sh
#
# 包含的踩坑点:
#   1. mmcv 必须 == 2.1.0 (2.0.x 在 py3.12 上无 wheel 编译失败)
#   2. mmengine >= 0.10.5 (旧版与 torch 2.4+ 的 Adafactor 注册冲突)
#   3. mmyolo v0.6.0 自带 mmcv<2.1.0 上限, 安装前必须 sed 放宽
#   4. lvis 包必须装 (mmdet 配置里的 LVIS minival 验证集需要它解析)
#   5. CLIP (openai/clip-vit-base-patch32) 必须本地完整下载
#   6. 新版 transformers (因 CVE-2025-32434) 拒绝 .bin, 必须有 .safetensors
#   7. HF 缓存的 manifest 里没登记后补的 .safetensors,
#      detector config 里 text_model_name 必须用本地 snapshot 绝对路径
#
# 用法:
#   bash setup_env.sh                  # 全套都跑(已就位的会跳过)
#   FORCE_REINSTALL=1 bash setup_env.sh  # 即使已经装好也强制重装(慎用)
#
# 可选环境变量:
#   PROXY_URL=http://<proxy-host>:<port>        设置后 pip / HF 下载走该代理
#   PYTHON_BIN=/path/to/python3                指定 python 可执行 (默认 PATH 中的 python3)
#   SRC_ROOT=...                               mmyolo 源码目录的父目录 (默认 $ENV_ROOT/src)
#   VBENCH2_CACHE_DIR=...                      ViT 权重缓存目录
#                                              (默认 = $SCRIPT_DIR/vbench2/vbench2_weights)
#
# 不会自动跑模型评测! 跑完此脚本后再:
#   bash run_all.sh
# =====================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-${ENV_ROOT:-$REPO_ROOT/envs}/src}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
# 权重默认放在工程内 vbench2/vbench2_weights/
# 想放别处: export VBENCH2_CACHE_DIR=/your/path/to/vbench2_weights 后再跑本脚本
VBENCH2_CACHE_DIR="${VBENCH2_CACHE_DIR:-$SCRIPT_DIR/vbench2/vbench2_weights}"

CLIP_REPO="openai/clip-vit-base-patch32"
CLIP_HUB_DIR="$HOME/.cache/huggingface/hub/models--openai--clip-vit-base-patch32"
DETECTOR_CFG="$SCRIPT_DIR/vbench2/third_party/YOLO-World/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"

# ---------- 颜色与日志 ----------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()   { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()    { echo -e "${RED}[err]${NC}   $*" >&2; }
section(){ echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }

# ---------- 0. 代理(可选) ----------
section "0. 代理设置"
if [[ -n "${PROXY_URL:-}" ]]; then
    export http_proxy="$PROXY_URL"
    export https_proxy="$PROXY_URL"
    export no_proxy="localhost,127.0.0.1,::1"
    log "已启用代理: $PROXY_URL"
else
    log "未设置 PROXY_URL, 直接连外网"
    log "  如果外网不通(HF/PyPI), 重新跑:  PROXY_URL=http://<proxy-host>:<port> bash $0"
fi

# ---------- 1. python 基础工具 ----------
section "1. 基础 pip 工具 (pip / wheel / openmim / gdown / scenedetect / yacs / termcolor ...)"
if [[ "$FORCE_REINSTALL" -eq 1 ]] || ! "$PYTHON_BIN" -c "import openmim, gdown, scenedetect, timm, einops, addict, yapf, yacs, termcolor" >/dev/null 2>&1; then
    log "安装/升级基础 pip 工具..."
    "$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools
    "$PYTHON_BIN" -m pip install -U openmim gdown "timm>=0.9" "yapf>=0.40.1" \
        einops tqdm addict scenedetect yacs termcolor
else
    log "基础 pip 工具已就位, 跳过"
fi

# ---------- 2. mm 全家桶 ----------
section "2. mmengine / mmcv / mmdet"
need_mm=0
if [[ "$FORCE_REINSTALL" -eq 1 ]]; then
    need_mm=1
else
    "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1 || need_mm=1
import importlib.metadata as md
from packaging.version import Version
assert Version(md.version("mmengine")) >= Version("0.10.5")
assert Version(md.version("mmcv"))     == Version("2.1.0")
assert Version(md.version("mmdet"))    >= Version("3.3.0")
PY
fi
if [[ "$need_mm" -eq 1 ]]; then
    log "安装 mmengine>=0.10.5 / mmcv==2.1.0 / mmdet==3.3.0 ..."
    "$PYTHON_BIN" -m mim install "mmengine>=0.10.5"
    "$PYTHON_BIN" -m mim install "mmcv==2.1.0"
    "$PYTHON_BIN" -m mim install "mmdet==3.3.0"
else
    log "mmengine / mmcv / mmdet 已就位, 跳过"
fi

# ---------- 3. mmyolo 源码安装 (放宽 mmcv 上限) ----------
section "3. mmyolo 0.6.0 源码安装"
need_mmyolo=0
if [[ "$FORCE_REINSTALL" -eq 1 ]]; then
    need_mmyolo=1
elif ! "$PYTHON_BIN" -c "import mmyolo" >/dev/null 2>&1; then
    need_mmyolo=1
fi
if [[ "$need_mmyolo" -eq 1 ]]; then
    log "下载 mmyolo v0.6.0 源码到 $SRC_ROOT/mmyolo ..."
    mkdir -p "$SRC_ROOT"
    cd "$SRC_ROOT"
    rm -rf mmyolo mmyolo-0.6.0
    curl -fL --connect-timeout 15 --max-time 300 \
        -o mmyolo.tar.gz \
        https://github.com/open-mmlab/mmyolo/archive/refs/tags/v0.6.0.tar.gz
    tar -xzf mmyolo.tar.gz && mv mmyolo-0.6.0 mmyolo && cd mmyolo
    log "放宽 mmcv 版本上限 (允许 2.1.0~2.3.0) ..."
    sed -i 's/mmcv>=2\.0\.0rc4,<2\.1\.0/mmcv>=2.0.0rc4,<2.3.0/g' requirements/mminstall.txt
    sed -i 's/mmcv_maximum_version *= *.*/mmcv_maximum_version = "2.3.0"/g' mmyolo/__init__.py
    # --no-deps: 防止 mmcv 被强降回 2.0.x
    # --no-build-isolation: mmyolo 的 setup.py 顶部 `import torch`, 若走 pip 默认 build isolation
    #   pip 会临时建独立环境, 该环境里没有 torch -> ModuleNotFoundError: No module named 'torch'.
    #   加 --no-build-isolation 让 build 阶段直接使用 base 的 torch, 复用 base setuptools/wheel.
    "$PYTHON_BIN" -m pip install -v -e . --no-deps --no-build-isolation
    "$PYTHON_BIN" -m pip install "albumentations>=1.3.0"
    cd "$SCRIPT_DIR"
else
    log "mmyolo 已就位, 跳过"
fi

# ---------- 4. lvis 包 (mmdet LVIS 验证集需要) ----------
section "4. lvis 包"
if [[ "$FORCE_REINSTALL" -eq 1 ]] || ! "$PYTHON_BIN" -c "import lvis" >/dev/null 2>&1; then
    log "安装 lvis ..."
    "$PYTHON_BIN" -m pip install lvis
else
    log "lvis 已就位, 跳过"
fi

# ---------- 5. ViT 权重检查 ----------
section "5. VBench2 ViT 权重检查 (4 个文件, 不自动下载)"
WEIGHTS=(
    "$VBENCH2_CACHE_DIR/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
    "$VBENCH2_CACHE_DIR/anomaly_detector/human.pth"
    "$VBENCH2_CACHE_DIR/anomaly_detector/face.pth"
    "$VBENCH2_CACHE_DIR/anomaly_detector/hand.pth"
)
missing=0
for w in "${WEIGHTS[@]}"; do
    if [[ -f "$w" ]]; then
        log "  ok: $w"
    else
        warn "  MISSING: $w"
        missing=1
    fi
done
if [[ "$missing" -eq 1 ]]; then
    cat <<EOF
${YELLOW}
[WARN] 上述 ViT 权重未就位.
请按 evaluation/README.md §1.3 把 4 个权重放到对应路径 (可用 gdown 下载).
本一键脚本不会自动下载这 4 个权重(避免无网环境卡死).
${NC}
EOF
fi

# ---------- 6. CLIP 仓库下载 ----------
section "6. CLIP (openai/clip-vit-base-patch32) 整仓下载到 HF 缓存"
clip_snap=""
if [[ -d "$CLIP_HUB_DIR/snapshots" ]]; then
    clip_snap=$(find "$CLIP_HUB_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n1 || true)
fi
need_clip=0
if [[ -z "$clip_snap" ]]; then
    need_clip=1
elif [[ ! -f "$clip_snap/pytorch_model.bin" ]] && [[ ! -f "$clip_snap/model.safetensors" ]]; then
    need_clip=1
elif [[ -f "$clip_snap/pytorch_model.bin" ]]; then
    sz=$(stat -c%s "$clip_snap/pytorch_model.bin" 2>/dev/null || echo 0)
    if [[ "$sz" -lt 600000000 ]]; then need_clip=1; fi
fi

if [[ "$need_clip" -eq 1 ]] || [[ "$FORCE_REINSTALL" -eq 1 ]]; then
    log "下载 CLIP (~605MB)..."
    # 关 OFFLINE, 这一步必须联网
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_ENDPOINT
    # 清掉之前残留的 incomplete / 0 字节假软链
    rm -f "$CLIP_HUB_DIR/blobs/"*.incomplete 2>/dev/null || true
    "$PYTHON_BIN" - <<PY
import os, sys
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="$CLIP_REPO",
    repo_type="model",
    allow_patterns=[
        "config.json", "tokenizer*", "vocab.json", "merges.txt",
        "special_tokens_map.json", "preprocessor_config.json",
        "pytorch_model.bin",
    ],
    max_workers=4,
)
print("DOWNLOADED_TO:", p)
bin_path = os.path.join(p, "pytorch_model.bin")
sz = os.path.getsize(os.path.realpath(bin_path)) if os.path.exists(bin_path) else 0
assert sz > 600000000, f"pytorch_model.bin size invalid: {sz}"
print("[ok] pytorch_model.bin size =", sz)
PY
    clip_snap=$(find "$CLIP_HUB_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n1)
else
    log "CLIP 已下载: $clip_snap"
fi

# ---------- 7. bin -> safetensors 转换 ----------
section "7. CLIP pytorch_model.bin -> model.safetensors 转换"
if [[ -f "$clip_snap/model.safetensors" ]] && [[ "$FORCE_REINSTALL" -ne 1 ]]; then
    log "model.safetensors 已存在, 跳过转换"
else
    log "转换中... (~10s)"
    "$PYTHON_BIN" - <<PY
import os
from pathlib import Path
import torch
from safetensors.torch import save_file

snap = Path("$clip_snap")
blob_dir = snap.parent.parent / "blobs"
src_bin = (snap / "pytorch_model.bin").resolve()
print("[load]", src_bin, "size=", src_bin.stat().st_size)
state = torch.load(src_bin, map_location="cpu", weights_only=False)
print("[info] tensor count:", len(state))

clean = {}
for k, v in state.items():
    if isinstance(v, torch.Tensor):
        clean[k] = v.contiguous().clone()

dst_blob = blob_dir / "model.safetensors"
# 必须带 metadata={"format":"pt"} — 否则 transformers.from_pretrained 会在
# modeling_utils.py 里 metadata.get("format") 报 'NoneType' has no attribute 'get'
save_file(clean, str(dst_blob), metadata={"format": "pt"})
print("[save]", dst_blob, "size=", dst_blob.stat().st_size)

dst_link = snap / "model.safetensors"
if dst_link.exists() or dst_link.is_symlink():
    dst_link.unlink()
os.symlink("../../blobs/model.safetensors", dst_link)
print("[link]", dst_link, "->", os.readlink(dst_link))
PY
fi

# ---------- 8. 修补 detector config 的 text_model_name ----------
section "8. 修补 detector config (text_model_name -> 本地 snapshot 绝对路径)"
if [[ ! -f "$DETECTOR_CFG" ]]; then
    err "detector config not found: $DETECTOR_CFG"
    exit 1
fi
target_line="text_model_name = '$clip_snap'"
if grep -qF "$target_line" "$DETECTOR_CFG"; then
    log "detector config 已指向本地 snapshot, 跳过"
else
    log "改写 text_model_name -> $clip_snap"
    # 仅替换"未注释的 text_model_name = '...' "这一行
    "$PYTHON_BIN" - <<PY
import re
from pathlib import Path
p = Path("$DETECTOR_CFG")
s = p.read_text()
new_line = "text_model_name = '$clip_snap'"
# 只替换不在注释里的 text_model_name 赋值, 旧值可能是 'openai/...' 也可能是其它本地路径
s2, n = re.subn(r"^[ \t]*text_model_name[ \t]*=[ \t]*['\"][^'\"\n]+['\"]",
                new_line, s, count=1, flags=re.MULTILINE)
if n == 0:
    raise SystemExit("[err] 未在 detector config 找到 text_model_name 赋值, 请人工检查")
p.write_text(s2)
print("[ok] patched")
PY
fi

# ---------- 9. 自检 ----------
section "9. 自检 import / CLIP 离线 load / detector config 解析"
"$PYTHON_BIN" - <<'PY'
import importlib, sys
print("python:", sys.version.split()[0])
ok = True
for name in ["torch","numpy","mmengine","mmcv","mmdet","mmyolo","timm",
             "decord","scenedetect","albumentations","lvis","huggingface_hub","safetensors"]:
    try:
        m = importlib.import_module(name)
        print(f"  {name:18s} {getattr(m,'__version__','?')}")
    except Exception as e:
        print(f"  {name:18s} FAIL: {type(e).__name__}: {e}")
        ok = False
print("--- key import chain ---")
for stmt in [
    "from mmdet.apis import init_detector",
    "from mmyolo.registry import VISUALIZERS",
    "from mmengine.dataset import Compose",
    "from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD",
    "import lvis",
]:
    try:
        exec(stmt); print(f"  OK   {stmt}")
    except Exception as e:
        print(f"  FAIL {stmt}  -> {type(e).__name__}: {e}")
        ok = False
sys.exit(0 if ok else 1)
PY

log "CLIP 离线 load 验证..."
"$PYTHON_BIN" - <<PY
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
LOCAL = "$clip_snap"
tok = AutoTokenizer.from_pretrained(LOCAL)
print("[OK] tokenizer:", type(tok).__name__, "vocab_size=", tok.vocab_size)
cfg = CLIPTextConfig.from_pretrained(LOCAL)
m = CLIPTextModel.from_pretrained(LOCAL, config=cfg)
print(f"[OK] CLIPTextModel params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
PY

# ---------- 10. 完成 ----------
section "10. ALL DONE"
cat <<EOF
${GREEN}
环境就位. 现在可以跑评测了:

  cd $SCRIPT_DIR
  bash run_all.sh                          # 跑 models.tsv 里的全部模型
  MODELS=seedance bash run_all.sh          # 仅跑 seedance
  CATEGORIES=all bash run_all.sh           # 不过滤 category, 全量评测

CLIP snapshot dir: $clip_snap
ViT  cache  dir:   $VBENCH2_CACHE_DIR
${NC}
EOF
