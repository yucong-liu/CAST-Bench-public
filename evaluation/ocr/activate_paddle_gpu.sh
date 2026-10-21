#!/bin/bash
# ============================================================================
# PaddleOCR GPU 环境激活脚本
#
# 用法（必须 source，不能 ./ 直接跑）：
#     source evaluation/ocr/activate_paddle_gpu.sh
#   或更短（rebuild_env.sh 会在 ~/ 下放一份软链）：
#     source ~/activate_paddle_gpu.sh
#
# 验证：
#     python3 -c 'import paddle; print(paddle.is_compiled_with_cuda(), paddle.device.cuda.device_count())'
#     # 期望：True  >=1
#
# 适用：
#     paddlepaddle-gpu 3.1.0 (cu126) + paddleocr 3.7.0 + paddlex 3.7.2, Python 3.12
#
# 为什么需要这个脚本？
#   paddle 需要优先加载 site-packages/nvidia/* 下的 cuDNN / cuBLAS / cudart / NVRTC。
#   如果 LD_LIBRARY_PATH 里带着系统里其它 CUDA 版本、nsight、gcc-toolset 等目录，
#   同名库 (libnvrtc / libstdc++ / libcommon 等) 可能被抢先加载，导致 paddle
#   ImportError 或 undefined symbol。
#
#   因此把 LD_LIBRARY_PATH **独占赋值** 成这 4 个 nvidia 路径
#   (不带 :$LD_LIBRARY_PATH 后缀)，顺序与 libphi_core.so 的依赖一致：
#       cudnn → cublas → cuda_runtime → cuda_nvrtc
#
# 副作用提醒：
#   source 之后当前 shell 看不到其它 CUDA / nsight 等路径，**只**适合跑
#   PaddleOCR / paddle GPU 任务。评测流水线走 CPU 参考路线, 不需要 source 本脚本;
#   需要 nvcc 编译等请新开一个 shell。
# ============================================================================
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# ---- 0) 必须 source 执行；如果是 ./activate_paddle_gpu.sh 直接执行就报错退出
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "❌ 这个脚本必须用 source 执行，例如："
    echo "     source ~/activate_paddle_gpu.sh"
    echo "     source $REPO_ROOT/evaluation/ocr/activate_paddle_gpu.sh"
    echo "   直接 ./activate_paddle_gpu.sh 不会改变当前 shell 的环境变量。"
    exit 1
fi

# ---- 1) 清掉 LD_PRELOAD
unset LD_PRELOAD

# ---- 2) 独占赋值 LD_LIBRARY_PATH —— 注意末尾 **没有** :$LD_LIBRARY_PATH
_PADDLE_NV="$(python3 -c 'import nvidia; print(list(nvidia.__path__)[0])')"
export LD_LIBRARY_PATH="$_PADDLE_NV/cudnn/lib:$_PADDLE_NV/cublas/lib:$_PADDLE_NV/cuda_runtime/lib:$_PADDLE_NV/cuda_nvrtc/lib"
unset _PADDLE_NV

# ---- 3) 静音 PaddleX 启动时的「检查模型源连通性」网络探测
#       (模型缓存在 ~/.paddlex/official_models/ 后不需要联网校验)
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# ---- 4) 状态提示
echo "✅ PaddleOCR GPU 环境已激活 (paddlepaddle-gpu 3.1.0 cu126 + paddleocr 3.7.0)"
echo "   LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | sed 's/^/     - /'
echo "   一键自检： python3 -c 'import paddle; print(paddle.is_compiled_with_cuda(), paddle.device.cuda.device_count())'"
