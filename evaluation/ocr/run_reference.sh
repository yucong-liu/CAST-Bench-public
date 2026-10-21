#!/usr/bin/env bash
# =====================================================================
# ocr · run_reference.sh
#
# 参考式 OCR 打分流水线 (对齐 MSAVBench pp_ocrv5, 不启 Qwen 裁判)
#
# 阶段:
#   0) filter_specs.py       扫 t2av-bench/specs/*.jsonl 过滤 text_on_screen 非空
#                            -> ocr_specs.jsonl  (若已存在且 RUN_FILTER=0 则跳过)
#   1) extract_tracks.py     stage1: PaddleOCR (use_textline_orientation=True) 抽 tracks (DP×N)
#                            -> runs/paddleocr/<model>/tracks/<vid>.json
#   2) score_reference.py    参考式打分, 用 spec.text_on_screen[*].text 作为 GT
#                            -> score_out/reference/<model>/<vid>.json
#                            -> score_out/reference/compare_summary.csv
#
# 全部前台运行, 不使用 nohup/&/setsid.
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

EVAL_ROOT=$REPO_ROOT/evaluation/ocr
RUN_DIR="${RUN_DIR:-$EVAL_ROOT/runs/paddleocr}"
SCORE_DIR="${SCORE_DIR:-$EVAL_ROOT/score_out/reference}"

OCR_SPECS="${OCR_SPECS:-$EVAL_ROOT/ocr_specs.jsonl}"
SPECS_DIR="${SPECS_DIR:-$REPO_ROOT/t2av-bench/specs}"

FILTER_PY=$EVAL_ROOT/filter_specs.py
EXTRACT_PY=$EVAL_ROOT/extract_tracks.py
MERGE_PY=$EVAL_ROOT/merge_shards.py
SCORE_PY=$EVAL_ROOT/score_reference.py

MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"
# 6 个模型, 与 id_consistency 相比额外加 kling-v3
ONLY_MODELS="${ONLY_MODELS:-LTX-2.3,Ovi,daVinci-MagiHuman,seedance,wan2.7,kling-v3}"

# DP: stage1 分片数 (每卡 1 进程)
N_SHARDS="${N_SHARDS:-8}"

LIMIT="${LIMIT:-0}"
# textline orientation 已从 False 改为 True, 之前的 tracks 不再等价 → 默认全量重跑
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# 官方 pp_ocrv5 打分参数 (已作为 score_reference.py 默认值, 一般不用改)
NUM_FRAMES="${NUM_FRAMES:-5}"
MIN_TEXT_SCORE="${MIN_TEXT_SCORE:-0.5}"
FUZZY_THRESHOLD="${FUZZY_THRESHOLD:-0.7}"
MAX_VIDEO_DURATION_SEC="${MAX_VIDEO_DURATION_SEC:-30.0}"

# 阶段开关
# ocr_specs.jsonl 不存在时默认先跑 stage 0 生成; 已存在则默认不再重跑
if [[ -f "$OCR_SPECS" ]]; then RUN_FILTER="${RUN_FILTER:-0}"; else RUN_FILTER="${RUN_FILTER:-1}"; fi
RUN_EXTRACT="${RUN_EXTRACT:-1}"        # stage1 需要重跑 (方向分类打开)
RUN_SCORE="${RUN_SCORE:-1}"

# 若需备份旧的 tracks (方向分类=False 版本), 打包到 runs/paddleocr_v4_no_orient/
BACKUP_OLD_TRACKS="${BACKUP_OLD_TRACKS:-1}"

# ---------------------------------------------------------------
if [[ ! -f "$MODELS_TSV" ]]; then
    echo "[ERROR] models.tsv not found: $MODELS_TSV" >&2; exit 1
fi
if [[ ! -f "$EXTRACT_PY" ]]; then
    echo "[ERROR] extract_tracks.py not found: $EXTRACT_PY" >&2; exit 1
fi
if [[ ! -f "$SCORE_PY" ]]; then
    echo "[ERROR] score_reference.py not found: $SCORE_PY" >&2; exit 1
fi

mkdir -p "$RUN_DIR" "$SCORE_DIR"

echo "=========================================================="
echo "[ocr.ref] EVAL_ROOT      = $EVAL_ROOT"
echo "[ocr.ref] MODELS_TSV     = $MODELS_TSV"
echo "[ocr.ref] ONLY_MODELS    = $ONLY_MODELS"
echo "[ocr.ref] OCR_SPECS      = $OCR_SPECS"
echo "[ocr.ref] RUN_DIR        = $RUN_DIR"
echo "[ocr.ref] SCORE_DIR      = $SCORE_DIR"
echo "[ocr.ref] N_SHARDS       = $N_SHARDS"
echo "[ocr.ref] LIMIT          = $LIMIT (per-model)"
echo "[ocr.ref] SKIP_EXISTING  = $SKIP_EXISTING"
echo "[ocr.ref] BACKUP_OLD     = $BACKUP_OLD_TRACKS"
echo "[ocr.ref] STAGES: filter=$RUN_FILTER extract=$RUN_EXTRACT score=$RUN_SCORE"
echo "[ocr.ref] SCORE PARAMS: num_frames=$NUM_FRAMES min_text_score=$MIN_TEXT_SCORE"
echo "[ocr.ref]              fuzzy_threshold=$FUZZY_THRESHOLD max_dur=$MAX_VIDEO_DURATION_SEC"
echo "=========================================================="

# ============== 阶段 0: 过滤 specs (可选) ==============
if [[ "$RUN_FILTER" == "1" ]]; then
    echo
    echo "[ocr.ref] === stage 0: filter_specs ==="
    python3 -u "$FILTER_PY" --specs-dir "$SPECS_DIR" --out "$OCR_SPECS"
fi

if [[ ! -f "$OCR_SPECS" ]]; then
    echo "[ERROR] ocr_specs not found: $OCR_SPECS  (先设 RUN_FILTER=1)" >&2; exit 1
fi

# ============== 阶段 1: PaddleOCR 抽 tracks (方向分类=True) ==============
if [[ "$RUN_EXTRACT" == "1" ]]; then
    # 先把旧的 tracks 备份改名 (只在第一次 SKIP_EXISTING=0 时做)
    if [[ "$BACKUP_OLD_TRACKS" == "1" && "$SKIP_EXISTING" == "0" ]]; then
        BAK_DIR="$EVAL_ROOT/runs/paddleocr_v4_no_orient"
        if [[ -d "$RUN_DIR" ]] && compgen -G "$RUN_DIR/*/tracks" > /dev/null; then
            if [[ ! -d "$BAK_DIR" ]]; then
                echo "[ocr.ref] backup old tracks -> $BAK_DIR"
                mv "$RUN_DIR" "$BAK_DIR"
                mkdir -p "$RUN_DIR"
            else
                echo "[ocr.ref] backup dir already exists, skip: $BAK_DIR"
            fi
        fi
    fi

    echo
    echo "[ocr.ref] === stage 1: extract_tracks (PaddleOCR use_textline_orientation=True, DP×$N_SHARDS) ==="

    EXTRACT_EXTRA=()
    [[ "$LIMIT" != "0" ]]         && EXTRACT_EXTRA+=(--limit "$LIMIT")
    [[ "$SKIP_EXISTING" == "1" ]] && EXTRACT_EXTRA+=(--skip-existing)
    [[ -n "$ONLY_MODELS" ]]       && EXTRACT_EXTRA+=(--only-models "$ONLY_MODELS")

    if [[ "$N_SHARDS" -le 1 ]]; then
        LOG="$RUN_DIR/extract.log"
        echo "[ocr.ref] single-shard, log=$LOG"
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
            echo "[ocr.ref]   spawned shard $i  pid=${pids[$i]}  log=$LOG"
        done

        fail=0
        for ((i=0; i<N_SHARDS; i++)); do
            rc=0; wait "${pids[$i]}" || rc=$?
            if [[ "$rc" -ne 0 ]]; then
                echo "[ocr.ref]   shard $i FAILED rc=$rc (see $RUN_DIR/extract.shard${i}.log)" >&2
                fail=$((fail+1))
            else
                echo "[ocr.ref]   shard $i done"
            fi
        done
        if [[ "$fail" -gt 0 ]]; then
            echo "[ocr.ref] $fail / $N_SHARDS shards failed, skip score" >&2; exit 1
        fi

        for sub in "$RUN_DIR"/*/ ; do
            [[ -d "$sub" ]] || continue
            if compgen -G "$sub"summary.shard*.json > /dev/null; then
                python3 "$MERGE_PY" --out-dir "$sub" --shard-num "$N_SHARDS"
            fi
        done
    fi
fi

# ============== 阶段 2: 参考式打分 (纯 CPU, 秒级) ==============
if [[ "$RUN_SCORE" == "1" ]]; then
    echo
    echo "[ocr.ref] === stage 2: score_reference (offline, no LLM) ==="
    LOG="$SCORE_DIR/_score.log"
    mkdir -p "$SCORE_DIR"
    python3 -u "$SCORE_PY" \
        --runs-root  "$RUN_DIR" \
        --ocr-specs  "$OCR_SPECS" \
        --models     "$ONLY_MODELS" \
        --out-dir    "$SCORE_DIR" \
        --num-frames "$NUM_FRAMES" \
        --min-text-score "$MIN_TEXT_SCORE" \
        --fuzzy-threshold "$FUZZY_THRESHOLD" \
        --max-video-duration-sec "$MAX_VIDEO_DURATION_SEC" \
        2>&1 | tee "$LOG"
fi

echo
echo "========================================================="
echo "[DONE] tracks under      : $RUN_DIR/<model>/tracks/"
echo "       per-video score   : $SCORE_DIR/<model>/<vid>.json"
echo "       per-model summary : $SCORE_DIR/<model>/_summary.csv"
echo "       compare table     : $SCORE_DIR/compare_summary.csv"
echo "========================================================="
