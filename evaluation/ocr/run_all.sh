#!/usr/bin/env bash
# =====================================================================
# ocr · 多模型 OCR 屏幕文字质量评测 (PaddleOCR + 本地 Qwen3-VL judge)
#
# 流水线 (与 id_consistency 同构):
#   0) filter_specs.py        扫 t2av-bench/specs/*.jsonl, 过滤 text_on_screen != []
#                             -> ocr_specs.jsonl
#   1) extract_tracks.py      stage1: PaddleOCR 跑视频 -> tracks/<vid>.json (DP×N)
#   2) merge_shards.py        合并 summary.shard*.json -> summary.json
#   3) judge_qwen.py          stage2: 本地 Qwen3-VL 多维 0-100 分
#   4) score_summary.py       跨 model 二次汇总 -> compare_summary.csv / per_category.csv
#
# 默认仅跑 5 个模型: LTX-2.3 / Ovi / daVinci-MagiHuman / seedance / wan2.7
# (与 id_consistency 一致, 不跑 kling-v3 / kling-v3-omni).
#
# 全部前台运行, 不使用 nohup/&/setsid 等后台手段; 单卡可直接看输出, Ctrl-C 可断.
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# ---------------- 用户可调参数 ----------------
EVAL_ROOT=$REPO_ROOT/evaluation/ocr
RUN_DIR="${RUN_DIR:-$EVAL_ROOT/runs/paddleocr}"
SCORE_DIR="${SCORE_DIR:-$EVAL_ROOT/score_out/qwen}"

OCR_SPECS="${OCR_SPECS:-$EVAL_ROOT/ocr_specs.jsonl}"
SPECS_DIR="${SPECS_DIR:-$REPO_ROOT/t2av-bench/specs}"
PROMPTS_DIR="${PROMPTS_DIR:-$REPO_ROOT/t2av-bench/prompts}"

FILTER_PY=$EVAL_ROOT/filter_specs.py
EXTRACT_PY=$EVAL_ROOT/extract_tracks.py
MERGE_PY=$EVAL_ROOT/merge_shards.py
JUDGE_PY=$EVAL_ROOT/judge_qwen.py
SUMMARY_PY=$EVAL_ROOT/score_summary.py

LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# DP: 默认 8 卡跑 stage1
N_SHARDS="${N_SHARDS:-8}"

MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"
ONLY_MODELS="${ONLY_MODELS:-LTX-2.3,Ovi,daVinci-MagiHuman,seedance,wan2.7}"

# stage2 (本地 Qwen) 并发线程数
JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
JUDGE_VARIANT="${JUDGE_VARIANT:-original}"

# 阶段开关 (默认全部开)
RUN_FILTER="${RUN_FILTER:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
# ----------------------------------------------

if [[ ! -f "$MODELS_TSV" ]]; then
    echo "[ERROR] models.tsv not found: $MODELS_TSV" >&2; exit 1
fi
if [[ ! -f "$EXTRACT_PY" ]]; then
    echo "[ERROR] extract_tracks.py not found: $EXTRACT_PY" >&2; exit 1
fi
if [[ ! -f "$JUDGE_PY" ]]; then
    echo "[ERROR] judge_qwen.py not found: $JUDGE_PY" >&2; exit 1
fi

mkdir -p "$RUN_DIR" "$SCORE_DIR"

echo "=========================================================="
echo "[ocr] EVAL_ROOT     = $EVAL_ROOT"
echo "[ocr] MODELS_TSV    = $MODELS_TSV"
echo "[ocr] ONLY_MODELS   = $ONLY_MODELS"
echo "[ocr] OCR_SPECS     = $OCR_SPECS"
echo "[ocr] PROMPTS_DIR   = $PROMPTS_DIR"
echo "[ocr] RUN_DIR       = $RUN_DIR"
echo "[ocr] SCORE_DIR     = $SCORE_DIR"
echo "[ocr] N_SHARDS      = $N_SHARDS"
echo "[ocr] LIMIT=$LIMIT (per-model)  SKIP_EXISTING=$SKIP_EXISTING"
echo "[ocr] JUDGE_WORKERS = $JUDGE_WORKERS  JUDGE_VARIANT=$JUDGE_VARIANT"
echo "[ocr] STAGES: filter=$RUN_FILTER extract=$RUN_EXTRACT judge=$RUN_JUDGE summary=$RUN_SUMMARY"
echo "=========================================================="

# ============== 阶段 0: 过滤 specs ==============
if [[ "$RUN_FILTER" == "1" ]]; then
    echo
    echo "[ocr] === stage 0: filter_specs ==="
    python3 -u "$FILTER_PY" --specs-dir "$SPECS_DIR" --out "$OCR_SPECS"
fi

if [[ ! -f "$OCR_SPECS" ]]; then
    echo "[ERROR] ocr_specs not found: $OCR_SPECS (启用 RUN_FILTER=1 先过滤)" >&2
    exit 1
fi

EXTRACT_EXTRA=()
[[ "$LIMIT" != "0" ]]         && EXTRACT_EXTRA+=(--limit "$LIMIT")
[[ "$SKIP_EXISTING" == "1" ]] && EXTRACT_EXTRA+=(--skip-existing)
[[ -n "$ONLY_MODELS" ]]       && EXTRACT_EXTRA+=(--only-models "$ONLY_MODELS")

# ============== 阶段 1: PaddleOCR 抽 tracks ==============
if [[ "$RUN_EXTRACT" == "1" ]]; then
    echo
    echo "[ocr] === stage 1: extract_tracks (PaddleOCR, DP×$N_SHARDS) ==="
    if [[ "$N_SHARDS" -le 1 ]]; then
        LOG="$RUN_DIR/extract.log"
        echo "[ocr] single-shard, log=$LOG"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        OCR_DEVICE="gpu:0" \
        python3 -u "$EXTRACT_PY" \
            --models-tsv "$MODELS_TSV" \
            --runs-root  "$RUN_DIR" \
            --ocr-specs  "$OCR_SPECS" \
            --shard-idx  0 --shard-num 1 \
            "${EXTRACT_EXTRA[@]}" \
            2>&1 | tee "$LOG"
    else
        # DP N×1: 起 N 个进程, 每个绑 1 张卡 (cuda:0)
        # 删掉旧 shard summary, 避免合并误读
        for sub in "$RUN_DIR"/*/ ; do
            [[ -d "$sub" ]] || continue
            rm -f "$sub"summary.shard*.json 2>/dev/null || true
        done
        rm -f "$RUN_DIR"/extract.shard*.log 2>/dev/null || true

        pids=()
        for ((i=0; i<N_SHARDS; i++)); do
            LOG="$RUN_DIR/extract.shard${i}.log"
            CUDA_VISIBLE_DEVICES=$i \
            OCR_DEVICE="gpu:0" \
            python3 -u "$EXTRACT_PY" \
                --models-tsv "$MODELS_TSV" \
                --runs-root  "$RUN_DIR" \
                --ocr-specs  "$OCR_SPECS" \
                --shard-idx  "$i" --shard-num "$N_SHARDS" \
                "${EXTRACT_EXTRA[@]}" \
                > "$LOG" 2>&1 &
            pids+=($!)
            echo "[ocr]   spawned shard $i  pid=${pids[$i]}  log=$LOG"
        done

        fail=0
        for ((i=0; i<N_SHARDS; i++)); do
            rc=0
            wait "${pids[$i]}" || rc=$?
            if [[ "$rc" -ne 0 ]]; then
                echo "[ocr]   shard $i FAILED rc=$rc (see $RUN_DIR/extract.shard${i}.log)" >&2
                fail=$((fail+1))
            else
                echo "[ocr]   shard $i done"
            fi
        done

        if [[ "$fail" -gt 0 ]]; then
            echo "[ocr] $fail / $N_SHARDS shards failed, skip merge & judge" >&2
            exit 1
        fi

        for sub in "$RUN_DIR"/*/ ; do
            [[ -d "$sub" ]] || continue
            if compgen -G "$sub"summary.shard*.json > /dev/null; then
                python3 "$MERGE_PY" --out-dir "$sub" --shard-num "$N_SHARDS"
            fi
        done
    fi
fi

# ============== 阶段 2: 本地 Qwen 裁判 ==============
if [[ "$RUN_JUDGE" == "1" ]]; then
    echo
    echo "[ocr] === stage 2: judge_qwen (本地 Qwen3-VL OpenAI 兼容服务) ==="
    JUDGE_EXTRA=()
    [[ "$LIMIT" != "0" ]]         && JUDGE_EXTRA+=(--limit "$LIMIT")
    [[ "$SKIP_EXISTING" == "1" ]] && JUDGE_EXTRA+=(--skip-existing)

    LOG="$SCORE_DIR/_judge.log"
    mkdir -p "$SCORE_DIR"
    python3 -u "$JUDGE_PY" \
        --runs-root   "$RUN_DIR" \
        --out-dir     "$SCORE_DIR" \
        --ocr-specs   "$OCR_SPECS" \
        --prompts-dir "$PROMPTS_DIR" \
        --models      "$ONLY_MODELS" \
        --prompt-variant "$JUDGE_VARIANT" \
        --num-workers "$JUDGE_WORKERS" \
        "${JUDGE_EXTRA[@]}" \
        2>&1 | tee "$LOG"
fi

# ============== 阶段 3: 二次汇总 ==============
if [[ "$RUN_SUMMARY" == "1" ]]; then
    echo
    echo "[ocr] === stage 3: score_summary ==="
    python3 -u "$SUMMARY_PY" --out-dir "$SCORE_DIR" --models "$ONLY_MODELS"
fi

echo
echo "========================================================="
echo "[DONE] tracks under   : $RUN_DIR/<model>/tracks/"
echo "       judge under    : $SCORE_DIR/<model>/<vid>.json"
echo "       compare table  : $SCORE_DIR/compare_summary.csv"
echo "       per-category   : $SCORE_DIR/per_category.csv"
echo "       model summary  : $SCORE_DIR/judge_summary.csv"
echo "========================================================="
