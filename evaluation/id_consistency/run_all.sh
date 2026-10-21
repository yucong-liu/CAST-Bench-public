#!/usr/bin/env bash
# =====================================================================
# id_consistency · 多模型 ID 一致性评测 (Qwen3-Omni 端到端 ASR + 说话人编号)
#
# 流水线:
#   1) transcribe_omni.py
#        加载 Qwen3-Omni-30B-A3B-Instruct 1 次, 跨多模型循环, 每条视频
#        一次 generate -> runs/<model>/transcripts/<vid>.json
#        DP N x 1: 起 N 个进程, 每个绑 1 张卡, 按 (model,video) 全局序列切片.
#   2) merge_shards.py: 合并 summary.shard*.json -> summary.json
#   3) score_meeteval.py: 跟 specs 对比, 输出 cpwer / greedy_orcwer / Δcp
#                         -> score_out/compare_summary.csv
#
# 默认仅跑 5 个模型: LTX-2.3 / Ovi / daVinci-MagiHuman / seedance / wan2.7
# 不跑 kling-v3 / kling-v3-omni. 可通过 ONLY_MODELS 覆盖.
#
# 全部前台运行, 不使用 nohup/&/setsid 等后台手段; 单卡可前台直接看输出.
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
export REPO_ROOT MODEL_ROOT

# ---------------- 用户可调参数 ----------------
EVAL_ROOT=$REPO_ROOT/evaluation/id_consistency
RUN_DIR="${RUN_DIR:-$EVAL_ROOT/runs}"
SCORE_DIR="${SCORE_DIR:-$EVAL_ROOT/score_out}"

SPECS="${SPECS:-$REPO_ROOT/t2av-bench/specs/speech.jsonl}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-$EVAL_ROOT/system_prompt.md}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen3-Omni/Qwen3-Omni-30B-A3B-Instruct}"

# Backend 选择: local (默认) | api (走 qwen3.5-omni-plus 平台)
#   local 输出: runs/qwen3_omni/<model>/transcripts/...     score_out/qwen3_omni/compare_summary.csv
#   api   输出: runs/qwen3.5_omni/<model>/transcripts/...   score_out/qwen3.5_omni/compare_summary.csv
BACKEND="${BACKEND:-local}"
LOCAL_SUBDIR="${LOCAL_SUBDIR:-qwen3_omni}"
API_SUBDIR="${API_SUBDIR:-qwen3.5_omni}"

TRANSCRIBE_PY=$EVAL_ROOT/transcribe_omni.py
MERGE_PY=$EVAL_ROOT/merge_shards.py
SCORE_PY=$EVAL_ROOT/score_meeteval.py

CATEGORIES="${CATEGORIES:-speech}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
LIMIT="${LIMIT:-0}"           # per-model, 0 = 不限制
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# DP: 默认 8 卡; 单卡跑设 N_SHARDS=1. api 模式强制 1
N_SHARDS="${N_SHARDS:-8}"
if [[ "$BACKEND" == "api" ]]; then
    N_SHARDS=1
fi

MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"
# 默认仅 5 个模型 (排除 kling)
ONLY_MODELS="${ONLY_MODELS:-LTX-2.3,Ovi,daVinci-MagiHuman,seedance,wan2.7}"

# 阶段开关
RUN_TRANSCRIBE="${RUN_TRANSCRIBE:-1}"
RUN_SCORE="${RUN_SCORE:-1}"

# APPEND_MODE=1 (仅阶段 2 生效):
#   给 score_meeteval.py 传 --append-models "$ONLY_MODELS",
#   读旧 compare_summary*.csv 保留其他模型的列, 只重算 ONLY_MODELS 的列.
#   per_sample.csv / details.json 也不覆盖原表, 改名写到 *.append.csv / *.append.json.
# APPEND_MODE=0 (默认):
#   score_meeteval.py 走原有全量模式 (--models "$ONLY_MODELS"), 会整表重写.
APPEND_MODE="${APPEND_MODE:-0}"
# ----------------------------------------------

if [[ ! -f "$MODELS_TSV" ]]; then
    echo "[ERROR] models.tsv not found: $MODELS_TSV" >&2; exit 1
fi
if [[ "$BACKEND" == "local" && ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Qwen3-Omni weight dir not found: $MODEL_PATH" >&2; exit 1
fi
if [[ "$BACKEND" == "api" ]]; then
    if [[ -z "${QWEN35_APP_ID:-}" || -z "${QWEN35_APP_KEY:-}" ]]; then
        echo "[ERROR] BACKEND=api 需要 export QWEN35_APP_ID / QWEN35_APP_KEY" >&2
        exit 1
    fi
fi
if [[ ! -f "$TRANSCRIBE_PY" ]]; then
    echo "[ERROR] transcribe_omni.py not found: $TRANSCRIBE_PY" >&2; exit 1
fi
if [[ ! -f "$SYSTEM_PROMPT" ]]; then
    echo "[ERROR] system_prompt.md not found: $SYSTEM_PROMPT" >&2; exit 1
fi

mkdir -p "$RUN_DIR" "$SCORE_DIR"

echo "=========================================================="
echo "[id_consistency] EVAL_ROOT     = $EVAL_ROOT"
echo "[id_consistency] BACKEND       = $BACKEND"
echo "[id_consistency] MODELS_TSV    = $MODELS_TSV"
echo "[id_consistency] ONLY_MODELS   = $ONLY_MODELS"
echo "[id_consistency] MODEL_PATH    = $MODEL_PATH"
echo "[id_consistency] RUN_DIR       = $RUN_DIR"
echo "[id_consistency] SPECS         = $SPECS"
echo "[id_consistency] CATEGORIES    = $CATEGORIES"
echo "[id_consistency] N_SHARDS      = $N_SHARDS  (model loaded ONCE per shard)"
echo "[id_consistency] LIMIT=$LIMIT (per-model)  MAX_NEW_TOKENS=$MAX_NEW_TOKENS  ATTN=$ATTN_IMPL"
echo "[id_consistency] SKIP_EXISTING = $SKIP_EXISTING"
echo "=========================================================="

# 供下面调用作为 backend 公用参数
BACKEND_ARGS=(--backend "$BACKEND")
if [[ "$BACKEND" == "local" ]]; then
    BACKEND_ARGS+=(--model-path "$MODEL_PATH")
fi

# RUN_DIR_BASE = 原始 runs/;
#   local 模式产物落在 RUN_DIR_BASE/qwen3_omni/<model>/
#   api   模式产物落在 RUN_DIR_BASE/qwen3.5_omni/<model>/
RUN_DIR_BASE="$RUN_DIR"
if [[ "$BACKEND" == "api" ]]; then
    RUNS_ROOT_FOR_TRANSCRIBE="$RUN_DIR_BASE/$API_SUBDIR"
else
    RUNS_ROOT_FOR_TRANSCRIBE="$RUN_DIR_BASE/$LOCAL_SUBDIR"
fi
mkdir -p "$RUNS_ROOT_FOR_TRANSCRIBE"

COMMON_EXTRA=()
[[ "$LIMIT" != "0" ]]      && COMMON_EXTRA+=(--limit "$LIMIT")
[[ "$SKIP_EXISTING" == "1" ]] && COMMON_EXTRA+=(--skip-existing)
[[ -n "$ONLY_MODELS" ]]    && COMMON_EXTRA+=(--only-models "$ONLY_MODELS")

# ============== 阶段 1: 转写 ==============
if [[ "$RUN_TRANSCRIBE" == "1" ]]; then
    if [[ "$N_SHARDS" -le 1 ]]; then
        LOG="$RUNS_ROOT_FOR_TRANSCRIBE/omni.log"
        echo "[id_consistency] single-shard run (backend=$BACKEND), log=$LOG"
        if [[ "$BACKEND" == "api" ]]; then
            python3 -u "$TRANSCRIBE_PY" \
                --models-tsv     "$MODELS_TSV" \
                --runs-root      "$RUNS_ROOT_FOR_TRANSCRIBE" \
                --system-prompt  "$SYSTEM_PROMPT" \
                "${BACKEND_ARGS[@]}" \
                --categories     "$CATEGORIES" \
                --attn-impl      "$ATTN_IMPL" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --shard-idx      0 \
                --shard-num      1 \
                --device         "auto" \
                "${COMMON_EXTRA[@]}" \
                2>&1 | tee "$LOG"
        else
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
            python3 -u "$TRANSCRIBE_PY" \
                --models-tsv     "$MODELS_TSV" \
                --runs-root      "$RUNS_ROOT_FOR_TRANSCRIBE" \
                --system-prompt  "$SYSTEM_PROMPT" \
                "${BACKEND_ARGS[@]}" \
                --categories     "$CATEGORIES" \
                --attn-impl      "$ATTN_IMPL" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --shard-idx      0 \
                --shard-num      1 \
                --device         "cuda:0" \
                "${COMMON_EXTRA[@]}" \
                2>&1 | tee "$LOG"
        fi
    else
        # DP N x 1: 仅 local 模式需要 (api 模式 N_SHARDS 被强制=1, 走不到这里)
        rm -f "$RUNS_ROOT_FOR_TRANSCRIBE"/omni.shard*.log 2>/dev/null || true
        for sub in "$RUNS_ROOT_FOR_TRANSCRIBE"/*/ ; do
            [[ -d "$sub" ]] || continue
            rm -f "$sub"summary.shard*.json 2>/dev/null || true
        done

        pids=()
        for ((i=0; i<N_SHARDS; i++)); do
            LOG="$RUNS_ROOT_FOR_TRANSCRIBE/omni.shard${i}.log"
            CUDA_VISIBLE_DEVICES=$i \
            python3 -u "$TRANSCRIBE_PY" \
                --models-tsv     "$MODELS_TSV" \
                --runs-root      "$RUNS_ROOT_FOR_TRANSCRIBE" \
                --system-prompt  "$SYSTEM_PROMPT" \
                "${BACKEND_ARGS[@]}" \
                --categories     "$CATEGORIES" \
                --attn-impl      "$ATTN_IMPL" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --shard-idx      "$i" \
                --shard-num      "$N_SHARDS" \
                --device         "cuda:0" \
                "${COMMON_EXTRA[@]}" \
                > "$LOG" 2>&1 &
            pids+=($!)
            echo "[id_consistency]   spawned shard $i  pid=${pids[$i]}  log=$LOG"
        done

        fail=0
        for ((i=0; i<N_SHARDS; i++)); do
            rc=0
            wait "${pids[$i]}" || rc=$?
            if [[ "$rc" -ne 0 ]]; then
                echo "[id_consistency]   shard $i FAILED rc=$rc (see $RUNS_ROOT_FOR_TRANSCRIBE/omni.shard${i}.log)" >&2
                fail=$((fail+1))
            else
                echo "[id_consistency]   shard $i done"
            fi
        done

        if [[ "$fail" -gt 0 ]]; then
            echo "[id_consistency] $fail / $N_SHARDS shards failed, skip merge & score" >&2
            exit 1
        fi

        for sub in "$RUNS_ROOT_FOR_TRANSCRIBE"/*/ ; do
            [[ -d "$sub" ]] || continue
            if compgen -G "$sub"summary.shard*.json > /dev/null; then
                python3 "$MERGE_PY" --out-dir "$sub" --shard-num "$N_SHARDS"
            fi
        done
    fi
fi

# ============== 阶段 2: 评分 ==============
if [[ "$RUN_SCORE" == "1" ]]; then
    echo
    echo "=========================================================="
    echo "[id_consistency] score (meeteval cpwer/greedy_orcwer/delta_cp)"
    echo "=========================================================="
    if [[ "$BACKEND" == "api" ]]; then
        SCORE_OUT_DIR="$SCORE_DIR/$API_SUBDIR"
    else
        SCORE_OUT_DIR="$SCORE_DIR/$LOCAL_SUBDIR"
    fi
    mkdir -p "$SCORE_OUT_DIR"
    if [[ "$APPEND_MODE" == "1" ]]; then
        echo "[id_consistency] APPEND_MODE=1  -> score_meeteval 追加模式"
        python3 "$SCORE_PY" \
            --specs         "$SPECS" \
            --runs-root     "$RUNS_ROOT_FOR_TRANSCRIBE" \
            --append-models "$ONLY_MODELS" \
            --out-dir       "$SCORE_OUT_DIR" \
            2>&1 | tee "$SCORE_OUT_DIR/_score.append.log"
    else
        python3 "$SCORE_PY" \
            --specs     "$SPECS" \
            --runs-root "$RUNS_ROOT_FOR_TRANSCRIBE" \
            --models    "$ONLY_MODELS" \
            --out-dir   "$SCORE_OUT_DIR" \
            2>&1 | tee "$SCORE_OUT_DIR/_score.log"
    fi
fi

echo
echo "========================================================="
echo "[DONE] hyp under     : $RUNS_ROOT_FOR_TRANSCRIBE/<model>/transcripts/"
if [[ "$BACKEND" == "api" ]]; then
    echo "       compare table : $SCORE_DIR/$API_SUBDIR/compare_summary.csv"
else
    echo "       compare table : $SCORE_DIR/$LOCAL_SUBDIR/compare_summary.csv"
fi
echo "========================================================="