#!/usr/bin/env bash
# ============================================================================
# Q-Align (OneAlign) 视频质量评测 —— 环境搭建脚本
#
# 用法（在 GPU 节点前台执行）：
#   cd evaluation/Q-Align
#   bash setup_env.sh
#
# 在独立 venv ($ENV_ROOT/qalign, torch 2.2.2+cu121) 里装 Q-Align, 与 base 环境完全隔离;
# setup_all.sh 在该 venv 缺失或 import 失败时调用本脚本.
# 任一步失败即退出；已经装好的话直接跑也没问题，pip 自己会识别到已装。
# 想跳过某一段（如已经装过 torch），把对应块用 : <<'SKIP' ... SKIP 注释掉即可。
# ============================================================================

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
ENV_ROOT="${ENV_ROOT:-$REPO_ROOT/envs}"
export REPO_ROOT MODEL_ROOT ENV_ROOT

# ============================================================================
# 1. 建 venv（python 3.12）+ 激活 + 升级 pip 工具链
# ============================================================================
mkdir -p $ENV_ROOT
python3.12 -m venv $ENV_ROOT/qalign

# ---------------------------------------------------------------------------
# shebang 自愈：venv 被整体搬迁到其他路径后，
# `python -m venv <已存在目录>` 只会更新 pyvenv.cfg + bin/python* 软链，不会
# 重写 bin/ 下第三方脚本 (pip / accelerate / huggingface-cli ...) 的 shebang，
# 导致 `pip --version` 报 bad interpreter。
# 这里主动扫 bin/ 下的每个普通文件：如果首行 shebang 指向的是任意路径下的
# ".../envs/qalign/bin/python[版本号]"，但和当前 VENV_DIR 不一致，就把它改
# 成当前 VENV_DIR 对应的路径。用 python heredoc + 参数传值，避开 sed/bash
# 对 `#!`、`$` 的转义踩坑；幂等，多跑几次无副作用。
# ---------------------------------------------------------------------------
VENV_DIR=$ENV_ROOT/qalign
python3 - "$VENV_DIR" <<'PY'
import os, re, sys, pathlib
venv_dir = sys.argv[1]
bin_dir = pathlib.Path(venv_dir) / "bin"
# shebang 形如: #!<任意前缀>/envs/qalign/bin/python 或 python3 / python3.12
pat = re.compile(r'^#!(\S+)/envs/qalign/bin/(python[0-9.]*)\s*$')
fixed, skipped = 0, 0
for entry in bin_dir.iterdir():
    if not entry.is_file() or entry.is_symlink():
        continue
    try:
        with entry.open('rb') as f:
            head = f.readline()
    except Exception:
        continue
    try:
        first = head.decode('utf-8', errors='strict').rstrip('\n')
    except UnicodeDecodeError:
        continue  # 二进制文件跳过
    m = pat.match(first)
    if not m:
        continue
    old_prefix, py_name = m.group(1), m.group(2)
    if old_prefix == venv_dir.rsplit('/envs/qalign', 1)[0]:
        skipped += 1
        continue  # 已经是新路径，无需改
    new_shebang = f"#!{venv_dir}/bin/{py_name}\n"
    with entry.open('rb') as f:
        rest = f.read()[len(head):]
    with entry.open('wb') as f:
        f.write(new_shebang.encode('utf-8'))
        f.write(rest)
    print(f"[shebang-heal] fixed: {entry.name}  (was: {first})")
    fixed += 1
print(f"[shebang-heal] done. fixed={fixed}, already-ok={skipped}")
PY

# 激活
source $ENV_ROOT/qalign/bin/activate

# 升级 pip 工具链
python -m pip install --upgrade pip wheel

# ============================================================================
# 2. 装 PyTorch（必须 cu121，不要走 cu118）
# ============================================================================
pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121

# 验证
python -c "
import torch
print('torch =', torch.__version__)
print('cuda  =', torch.version.cuda)
print('avail =', torch.cuda.is_available())
print('dev   =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
print('cap   =', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
"

# ============================================================================
# 3. 强制把 numpy 钉回 1.x
# ============================================================================
pip install "numpy<2"   # numpy==1.26.4

# 验证
python -c "
import numpy, torch
print('numpy =', numpy.__version__)
print('torch =', torch.__version__)
x = torch.from_numpy(numpy.array([1.0,2,3])).cuda() + 3
print('numpy<->torch<->cuda roundtrip OK:', x.sum().item())
"

# ============================================================================
# 4. 装 setuptools（修 distutils 缺失）
# ============================================================================
pip install "setuptools<81"

# ============================================================================
# 5. 装 Q-Align 项目依赖（精简清单 + 项目本体不带依赖）
# ============================================================================
cd $REPO_ROOT/evaluation/Q-Align

pip install \
  transformers==4.36.1 \
  tokenizers==0.15.0 \
  sentencepiece==0.1.99 \
  protobuf \
  accelerate==0.21.0 \
  peft==0.4.0 \
  "pydantic<2,>=1" \
  scikit-learn==1.2.2 \
  einops==0.6.1 einops-exts==0.0.4 \
  timm==0.6.13 \
  decord \
  scipy \
  shortuuid \
  icecream \
  Pillow

# 项目本体（不带依赖，避免覆盖上面的版本）
pip install -e . --no-deps

# ============================================================================
# 6. sanity check（只做导入，不下载、不跑模型）
# ============================================================================
python -c "
import sys, os, torch, transformers
print('python =', sys.version.split()[0])
print('torch  =', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())
print('trf    =', transformers.__version__)
from q_align.evaluate.scorer import QAlignVideoScorer
print('q_align import OK')
"

# ============================================================================
# 7. 模型权重: $MODEL_ROOT/q-align/one-align/ (缺失时从 HF q-future/one-align 下载)
# ============================================================================
if [[ ! -f "$MODEL_ROOT/q-align/one-align/config.json" ]]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='q-future/one-align', local_dir='$MODEL_ROOT/q-align/one-align')
"
fi

# ============================================================================
# 8. single-video smoke test（验证整链路通）
# 会真加载模型 (~22s) + 抽帧打分，需要约 16.5 GB 显存; 找不到 mp4 时跳过
# ============================================================================
source $ENV_ROOT/qalign/bin/activate
cd $REPO_ROOT/evaluation/Q-Align

SMOKE_VID="$(find "$REPO_ROOT"/*_test/output -name '*.mp4' 2>/dev/null | head -n 1 || true)"
if [[ -z "$SMOKE_VID" ]]; then
    echo "未找到 mp4 冒烟视频, 跳过 smoke test"
    exit 0
fi

python -c "
import torch, time
from q_align.evaluate.scorer import QAlignVideoScorer, load_video

mp = '$MODEL_ROOT/q-align/one-align'
vp = '$SMOKE_VID'

t0 = time.time()
scorer = QAlignVideoScorer(pretrained=mp, device='cuda:0')
print(f'>>> model loaded in {time.time()-t0:.1f}s')

video = load_video(vp)
print(f'>>> video frames sampled = {len(video)}')

with torch.inference_mode():
    score = scorer([video])
print('>>> raw score tensor =', score)
print('>>> quality MOS      =', score.flatten().tolist())
"
