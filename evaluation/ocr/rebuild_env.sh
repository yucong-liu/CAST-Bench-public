#!/usr/bin/env bash
# =============================================================================
# rebuild_env.sh
#   安装 PaddleOCR GPU 环境 (paddlepaddle-gpu 3.1.0 cu126 + paddleocr 3.7.0 + paddlex 3.7.2)
#   并做 GPU 自检. 由 setup_all.sh 的 ocr 步骤调用, 也可单独运行.
#
# 用法 (前台运行, 建议同时落日志):
#   bash rebuild_env.sh 2>&1 | tee ~/rebuild_env.$(date +%Y%m%d_%H%M%S).log
#
# 任何一步失败立刻停 (set -e).
# =============================================================================

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

PROJ_DIR="$REPO_ROOT/evaluation/ocr"
ACT_SCRIPT="${PROJ_DIR}/activate_paddle_gpu.sh"

log() { echo -e "\n========================================================================"; echo "[rebuild_env] $*"; echo "========================================================================"; }

# -----------------------------------------------------------------------------
log "Step 1. 当前 shell 清掉 LD_LIBRARY_PATH / LD_PRELOAD"
# -----------------------------------------------------------------------------
unset LD_LIBRARY_PATH || true
unset LD_PRELOAD      || true
echo "LD_LIBRARY_PATH = '${LD_LIBRARY_PATH:-<unset>}'"
echo "LD_PRELOAD      = '${LD_PRELOAD:-<unset>}'"

# -----------------------------------------------------------------------------
log "Step 2. 装 paddlepaddle-gpu 3.1.0 cu126 (~1.2 GB wheel) + paddleocr/paddlex"
# -----------------------------------------------------------------------------
# ⚠️ 版本选择:
#   - 用 cu126 wheel (自带 sm_90 kernel), 版本锁 3.1.0.
#   - cu118 wheel 未编译 sm_90 kernel, 在 sm_9.0 卡 (compute capability 9.0) 上会报
#     "CUDA error 209 no kernel image is available for execution on the device",
#     matmul 静默返回 0.0.
# ⚠️ 不要装 paddlepaddle-gpu 3.0.0 — libphi_core / libcommon ABI 错位, 起不来.
# ⚠️ 不要装 paddlepaddle-gpu 3.3.x — 撞 paddle_flags ABI 符号 bug.
python3 -m pip uninstall -y paddlepaddle paddlepaddle-gpu || true    # 先把任何其它版本/半装状态清掉
python3 -m pip install --no-cache-dir paddlepaddle-gpu==3.1.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# 期望: Successfully installed paddlepaddle-gpu-3.1.0
python3 -m pip show paddlepaddle-gpu | grep -E 'Name|Version|Location'

# --- PaddleOCR 全链路依赖 (只装 paddle 会 ModuleNotFoundError) ---
python3 -m pip install --no-cache-dir paddleocr==3.7.0 paddlex==3.7.2
python3 -m pip show paddleocr paddlex | grep -E 'Name|Version'

# -----------------------------------------------------------------------------
log "Step 3. 创建 / 同步激活脚本 (软链到 home)"
# -----------------------------------------------------------------------------
# 工程内 (本仓库已带): ${ACT_SCRIPT}
# 建议 home 下做软链, 任意目录都能 `source ~/activate_paddle_gpu.sh`:
ln -sf "${ACT_SCRIPT}" ~/activate_paddle_gpu.sh
ls -la ~/activate_paddle_gpu.sh

# -----------------------------------------------------------------------------
log "Step 4. 自检, 确认全 ✅"
# -----------------------------------------------------------------------------
# 先 source 激活脚本, 让 LD_LIBRARY_PATH 指向 site-packages/nvidia/* 下的 CUDA 库
# shellcheck disable=SC1090
source ~/activate_paddle_gpu.sh

# (1) paddle GPU 自检
log "Step 4.(1) Paddle GPU 自检"
python3 - <<'PY'
import paddle, numpy as np
print('cuda ok =', paddle.is_compiled_with_cuda())   # True
print('cuda v  =', paddle.version.cuda())            # 12.6 (cu126 wheel)
print('cudnn v =', paddle.version.cudnn())
print('devices =', paddle.device.cuda.device_count())  # >=1
x = paddle.to_tensor(np.ones((4,4),'float32'), place=paddle.CUDAPlace(0))
val = float((x @ x).sum().numpy())
print('matmul =', val)                                # 期望 64.0
assert abs(val - 64.0) < 1e-3, (
    f"❌ FAIL: matmul got {val}, expected 64.0. "
    "很可能 paddle wheel 没编译当前 GPU 的 sm kernel "
    "(如 cu118 wheel 跑 sm_9.0 卡). 请换 cu126 wheel.")
print('==> ✅ Paddle GPU OK')
PY

# (2) PaddleOCR 全链路自检
log "Step 4.(2) PaddleOCR 全链路自检"
python3 - <<'PY'
import numpy as np, time
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, device='gpu:0')
img = np.ones((480,640,3),'uint8')*255
t=time.time(); _=ocr.predict(input=[img]); print(f'warm = {time.time()-t:.2f}s')
t=time.time(); _=ocr.predict(input=[img]); print(f'steady = {time.time()-t:.2f}s')
print('==> ✅ PaddleOCR (GPU) 全链路 OK')
PY

log "全部完成 ✅   PaddleOCR GPU 环境已重建并通过自检"
