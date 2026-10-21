#!/usr/bin/env bash
# =====================================================================
# facial_consistency · setup_env.sh
#
# 依赖安装 / 真 GPU 验证 / buffalo_l 权重 smoke 测试, 把环境搭建 + 验证全部自动化.
#
# * 前台运行, 不使用 nohup/&/setsid; 遇错立刻退出 (set -euo pipefail).
# * 只做「装 + 验证」, 不启动任何评测 (run_all.sh 由用户手跑).
# * 4 个开关全部走环境变量:
#     SKIP_INSTALL=1     跳过 Phase 1 (已装过, 只想复验 GPU)
#     PIP_BREAK=1        pip 加 --break-system-packages (PEP 668 节点)
#     PIP_USER=1         pip 加 --user (与 PIP_BREAK 互斥, PIP_BREAK 优先)
#     ONNXRT_VERSION=…   默认 1.20.2 (与 requirements-freeze-clean.txt 一致, CUDA 12 build)
#     PY=…               默认 PATH 中的 python3 (base 环境)
#
# 常用调用姿势:
#     bash setup_env.sh                        # 首次完整安装 + 验证
#     PIP_BREAK=1 bash setup_env.sh            # PEP 668 报错时
#     SKIP_INSTALL=1 bash setup_env.sh         # 只跑验证 (Phase 3/4)
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

PY="${PY:-python3}"
ONNXRT_VERSION="${ONNXRT_VERSION:-1.20.2}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# ----- pip flag 组合 -----
PIP_EXTRA=()
if [[ "${PIP_BREAK:-0}" == "1" ]]; then
    PIP_EXTRA+=(--break-system-packages)
elif [[ "${PIP_USER:-0}" == "1" ]]; then
    PIP_EXTRA+=(--user)
fi

banner() {
    echo
    echo "=========================================================="
    echo "[setup_env] $*"
    echo "=========================================================="
}

# =====================================================================
# Phase 0. 节点信息自检 (只读, 帮定位 CUDA 12 vs 11)
#
# 探测命令随便怎么退出 (SIGPIPE / nvcc 缺失 / grep 无匹配) 都不该打断脚本,
# 所以整段临时关掉 pipefail + 每条命令 || true 兜底.
# =====================================================================
banner "Phase 0 · 节点信息自检"

set +o pipefail  # ← 临时关 pipefail, 防 `nvidia-smi | head -20` 触发 SIGPIPE 引 set -e 退出

echo "[env] which python : $(command -v "$PY" || echo NOT-FOUND)"
"$PY" --version || true
"$PY" -c "import sys; print('[env] sys.executable =', sys.executable)" || true
"$PY" -c "import sys; print('[env] sys.version    =', sys.version.split()[0])" || true

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | head -20 || true
else
    echo "[WARN] nvidia-smi 不存在, 无法确认 GPU 状态" >&2
fi

# CUDA 版本: onnxruntime-gpu 1.20.x (CUDA 12 build) 只兼容 CUDA 12
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    /usr/local/cuda/bin/nvcc --version | tail -2 || true
elif command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -2 || true
else
    echo "[WARN] 找不到 nvcc, 无法确认 CUDA toolkit 版本 (只装了 driver + runtime, 无 toolkit, 一般不影响推理)" >&2
fi

set -o pipefail  # ← Phase 0 结束, 恢复 pipefail 给后续 Phase 1/3 用

# =====================================================================
# Phase 1. 依赖安装 (含前置清理 CUDA-11 版 onnxruntime-gpu)
# =====================================================================
if [[ "$SKIP_INSTALL" == "1" ]]; then
    banner "Phase 1 · SKIP_INSTALL=1, 跳过依赖安装"
else
    banner "Phase 1 · 依赖安装 (insightface / scikit-learn / onnxruntime-gpu==${ONNXRT_VERSION})"

    # 1a. 如果已装 1.17.x / 1.18.x 的 onnxruntime-gpu, 先清掉
    CUR_ORT="$("$PY" -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null || echo none)"
    echo "[pip] existing onnxruntime version: $CUR_ORT"
    case "$CUR_ORT" in
        1.17.*|1.18.*)
            echo "[pip] 检测到 CUDA-11-only 版 onnxruntime-gpu ($CUR_ORT), 卸载"
            "$PY" -m pip uninstall -y onnxruntime onnxruntime-gpu "${PIP_EXTRA[@]}" || true
            ;;
        none|"")
            echo "[pip] 当前无 onnxruntime, 跳过卸载步骤"
            ;;
        *)
            echo "[pip] 现有 onnxruntime 版本 $CUR_ORT 允许保留 (不在黑名单 1.17.*/1.18.*)"
            ;;
    esac

    # 1b. insightface + scikit-learn
    echo
    echo "[pip] install insightface==0.7.3  scikit-learn"
    "$PY" -m pip install "${PIP_EXTRA[@]}" \
        'insightface==0.7.3' \
        'scikit-learn'

    # 1c. onnxruntime-gpu · CUDA 12 专用 wheel (onnxruntime-cuda-12 index)
    echo
    echo "[pip] install onnxruntime-gpu==${ONNXRT_VERSION}  (onnxruntime-cuda-12 index)"
    "$PY" -m pip install "${PIP_EXTRA[@]}" \
        "onnxruntime-gpu==${ONNXRT_VERSION}" \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
fi

# =====================================================================
# Phase 2. 版本核验
# =====================================================================
banner "Phase 2 · pip 版本核验"
"$PY" -m pip list 2>/dev/null | grep -Ei "(^insightface|^onnxruntime|^scikit-learn|^numpy|^opencv)" || {
    echo "[WARN] 关键包没匹配到, 请人工核对" >&2
}

# =====================================================================
# Phase 3. onnxruntime 真 GPU session 验证 (以实际使用的 provider 为准)
#
# 只看 get_available_providers() 不够, 必须建一个真实 InferenceSession
# 然后调 session.get_providers(), 才知道是否被静默回退到 CPU.
# 这一步同时会触发 buffalo_l 权重首次下载 (§5), 顺便完成 Phase 4.
# =====================================================================
banner "Phase 3 · onnxruntime 真实 GPU session 验证 (会顺带触发 buffalo_l 首次下载)"

set +e
"$PY" - <<'PY'
import sys
try:
    import onnxruntime as ort
except Exception as e:
    print(f"[FAIL] import onnxruntime failed: {e}", file=sys.stderr); sys.exit(3)

print(f"[ort] version   = {ort.__version__}")
print(f"[ort] available = {ort.get_available_providers()}")

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    print(f"[FAIL] import insightface failed: {e}", file=sys.stderr); sys.exit(3)

# 强制优先 CUDA, 用 buffalo_l 建一个真实 session
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
)
app.prepare(ctx_id=0, det_size=(640, 640))

det_sess = app.models['detection'].session
real = det_sess.get_providers()
print(f"[REAL] providers used by detection session = {real}")

if 'CUDAExecutionProvider' not in real:
    print("", file=sys.stderr)
    print("[FAIL] onnxruntime 静默回退到 CPU !", file=sys.stderr)
    print("       典型症状 = CUDA-11-only 版 onnxruntime-gpu 在 CUDA 12 节点上找不到 libcufft.so.10", file=sys.stderr)
    print("       修复: 卸载 CUDA-11 版 onnxruntime-gpu, 用 ONNXRT_VERSION 指定 CUDA-12 wheel 重跑本脚本", file=sys.stderr)
    sys.exit(4)

print("[PASS] CUDA session 真的建起来了 (REAL providers 含 CUDAExecutionProvider)")
PY
RC=$?
set -e

if [[ "$RC" -ne 0 ]]; then
    echo
    echo "[setup_env] Phase 3 FAILED (rc=$RC). onnxruntime-gpu 未能使用 CUDA, 按上面的提示排查后重跑." >&2
    exit "$RC"
fi

# =====================================================================
# Phase 4. buffalo_l 权重就位确认
#
# Phase 3 那步 FaceAnalysis(name='buffalo_l').prepare(...) 会自动下载,
# 这里只是 ls 复核一下 5 个 onnx 都在.
# =====================================================================
banner "Phase 4 · buffalo_l 权重复核"

BUFFALO_DIR="${HOME}/.insightface/models/buffalo_l"
if [[ ! -d "$BUFFALO_DIR" ]]; then
    echo "[FAIL] $BUFFALO_DIR 不存在. 按理 Phase 3 会触发下载, 请检查网络, 或手动下载 buffalo_l.zip 解压到 ~/.insightface/models/" >&2
    exit 5
fi

echo "[ls] $BUFFALO_DIR"
ls -la "$BUFFALO_DIR" | grep -E "\.onnx$" || {
    echo "[FAIL] $BUFFALO_DIR 下没有任何 .onnx 文件 (期望 5 个)" >&2
    exit 5
}

EXPECTED=(1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx w600k_r50.onnx)
missing=0
for f in "${EXPECTED[@]}"; do
    if [[ ! -f "$BUFFALO_DIR/$f" ]]; then
        echo "[FAIL] missing weight file: $BUFFALO_DIR/$f" >&2
        missing=$((missing+1))
    fi
done
if [[ "$missing" -gt 0 ]]; then
    echo "[FAIL] buffalo_l 缺 $missing 个权重文件, 可手动下载 buffalo_l.zip 解压到 ~/.insightface/models/" >&2
    exit 5
fi

echo "[PASS] buffalo_l 5 个 .onnx 权重齐全"

# =====================================================================
# Done
# =====================================================================
banner "[DONE] facial_consistency 环境搭建 & 验证全部通过"
cat <<EOF
下一步 (前台跑):

  cd $REPO_ROOT/evaluation/facial_consistency

  # 追加 3 个新模型 (最常用):
  mkdir -p runs
  MODELS=veo3.1,kling-v3-omni,happyhorse-1.1 \\
      bash run_all.sh 2>&1 | tee runs/run_new3_\$(date +%Y%m%d_%H%M).log

  # 或全量跑 (已跑过的自动 skip):
  bash run_all.sh
EOF
