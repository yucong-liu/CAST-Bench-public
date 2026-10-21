#!/usr/bin/env bash
# run_judge.sh
# ============
# 运行 ID 一致性 judge (per-video 给 subject/line 各打 1-5 分)
#
# 用法:
#   bash run_judge.sh                                # 跑全部 5 个模型
#   bash run_judge.sh --skip-existing                # 续跑, 已评过的跳过
#   bash run_judge.sh --models LTX-2.3,Ovi           # 只跑指定模型
#   bash run_judge.sh --ids speech_0001,speech_0002  # 只跑指定 vid
#
# 也可临时覆盖 LLM 服务:
#   BASE_URL=http://127.0.0.1:8000/v1 \
#   LLM_MODEL=Qwen3-VL-235B-A22B-Instruct \
#   bash run_judge.sh
#
# ! 不使用任何后台启动方式 (nohup / & / disown / setsid)
# ! 前台运行 + tee 同时落盘日志, Ctrl-C 可随时终止

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_DIR="${THIS_DIR}/score_out/judge"
LOG_DIR="${THIS_DIR}/score_out"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/judge_${TS}.log"

EXTRA_ARGS=("$@")

CMD=(python "${THIS_DIR}/judge_qwen.py")

if [[ -n "${BASE_URL:-}" ]]; then
  CMD+=(--base-url "${BASE_URL}")
fi
if [[ -n "${LLM_MODEL:-}" ]]; then
  CMD+=(--model "${LLM_MODEL}")
fi
if [[ -n "${API_KEY:-}" ]]; then
  CMD+=(--api-key "${API_KEY}")
fi
if [[ -n "${TEMPERATURE:-}" ]]; then
  CMD+=(--temperature "${TEMPERATURE}")
fi
if [[ -n "${MAX_TOKENS:-}" ]]; then
  CMD+=(--max-tokens "${MAX_TOKENS}")
fi
if [[ -n "${TIMEOUT_SEC:-}" ]]; then
  CMD+=(--timeout-sec "${TIMEOUT_SEC}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "==============================================" | tee -a "${LOG_FILE}"
echo "[run_judge] $(date '+%F %T')" | tee -a "${LOG_FILE}"
echo "[run_judge] CMD = ${CMD[*]}"   | tee -a "${LOG_FILE}"
echo "[run_judge] LOG = ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==============================================" | tee -a "${LOG_FILE}"

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
