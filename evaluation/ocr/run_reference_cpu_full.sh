#!/usr/bin/env bash
# =====================================================================
# ocr · run_reference_cpu_full.sh
#
# CPU 全量参考式 OCR 打分流水线 (方案 A: 时空轨迹 + 官方打分函数)
#
# 与 run_reference.sh 的区别:
#   - Stage 1 强制走 CPU (OCR_DEVICE=cpu), 不再依赖 GPU cudnn
#   - 用 shard-idx / shard-num 做 CPU 多进程并行 (每个进程 1 个 CPU worker)
#   - 默认 --skip-existing, 支持中断续跑
#   - 输出目录另起一份, 不覆盖已有 runs/ 与 score_out/reference/
#
# 阶段:
#   0) filter_specs.py       (仅当 ocr_specs.jsonl 不存在时)
#        -> ocr_specs.jsonl
#   1) extract_tracks.py     (CPU × N_SHARDS 并行)
#        -> runs_full_cpu/paddleocr/<model>/tracks/<vid>.json
#   2) score_reference.py    (纯 CPU, 秒级)
#        -> score_out/reference_full_cpu/<model>/<vid>.json
#        -> score_out/reference_full_cpu/compare_summary.csv
#   3) summarize_by_shot.py  (按 single_shot / multi_shot 汇总 6 个模型)
#        -> score_out/reference_full_cpu/compare_summary_ocr{,_single,_multi}.csv
#
# 全部前台运行, 不使用 nohup/&/setsid.
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

EVAL_ROOT=$REPO_ROOT/evaluation/ocr

RUN_DIR="${RUN_DIR:-$EVAL_ROOT/runs_full_cpu/paddleocr}"
SCORE_DIR="${SCORE_DIR:-$EVAL_ROOT/score_out/reference_full_cpu}"

OCR_SPECS="${OCR_SPECS:-$EVAL_ROOT/ocr_specs.jsonl}"
SPECS_DIR="${SPECS_DIR:-$REPO_ROOT/t2av-bench/specs}"

FILTER_PY=$EVAL_ROOT/filter_specs.py
EXTRACT_PY=$EVAL_ROOT/extract_tracks.py
MERGE_PY=$EVAL_ROOT/merge_shards.py
SCORE_PY=$EVAL_ROOT/score_reference.py
SHOT_PY=$EVAL_ROOT/summarize_by_shot.py

MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"
# ONLY_MODELS  = stage 1 (extract_tracks) 想跑哪些模型 (逗号分隔). 追加新模型时改这个.
#                空 = 跑 models.tsv 全部; 配合 --skip-existing 天然增量.
ONLY_MODELS="${ONLY_MODELS:-LTX-2.3,Ovi,daVinci-MagiHuman,seedance,wan2.7,kling-v3}"

# SUMMARY_MODELS = stage 2/3 汇总时用哪些模型. 空 = 自动扫 RUN_DIR 下所有 tracks/*.json 非空的模型,
#                  这样即使本次 stage 1 只跑了部分模型, compare_summary*.csv 也会带上其它模型的列.
SUMMARY_MODELS="${SUMMARY_MODELS:-}"

# CPU 分片并行进程数 (机器 56 核, 默认 16, 留富余)
N_SHARDS="${N_SHARDS:-16}"
# 每个 shard 内部 paddle/OMP 线程数; 保持 2 让 16 * 2 = 32 << 56 核
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"       # 默认续跑, 已有 tracks 不重算

# 官方 pp_ocrv5 打分参数 (score_reference.py 默认值一致)
NUM_FRAMES="${NUM_FRAMES:-5}"
MIN_TEXT_SCORE="${MIN_TEXT_SCORE:-0.5}"
FUZZY_THRESHOLD="${FUZZY_THRESHOLD:-0.7}"
MAX_VIDEO_DURATION_SEC="${MAX_VIDEO_DURATION_SEC:-30.0}"

RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_SCORE="${RUN_SCORE:-1}"
RUN_SHOT_SUMMARY="${RUN_SHOT_SUMMARY:-1}"

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
if [[ ! -f "$OCR_SPECS" ]]; then
    echo "[ocr.ref.cpu] ocr_specs 不存在, 先用 filter_specs.py 从 $SPECS_DIR 生成: $OCR_SPECS"
    python3 -u "$FILTER_PY" --specs-dir "$SPECS_DIR" --out "$OCR_SPECS"
fi

mkdir -p "$RUN_DIR" "$SCORE_DIR"

echo "=========================================================="
echo "[ocr.ref.cpu] EVAL_ROOT       = $EVAL_ROOT"
echo "[ocr.ref.cpu] MODELS_TSV      = $MODELS_TSV"
echo "[ocr.ref.cpu] ONLY_MODELS     = $ONLY_MODELS"
echo "[ocr.ref.cpu] OCR_SPECS       = $OCR_SPECS"
echo "[ocr.ref.cpu] RUN_DIR         = $RUN_DIR"
echo "[ocr.ref.cpu] SCORE_DIR       = $SCORE_DIR"
echo "[ocr.ref.cpu] N_SHARDS (CPU)  = $N_SHARDS"
echo "[ocr.ref.cpu] OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "[ocr.ref.cpu] LIMIT           = $LIMIT (per-model)"
echo "[ocr.ref.cpu] SKIP_EXISTING   = $SKIP_EXISTING"
echo "[ocr.ref.cpu] STAGES: extract=$RUN_EXTRACT score=$RUN_SCORE shot=$RUN_SHOT_SUMMARY"
echo "=========================================================="

# ---------------------------------------------------------------
# 自动发现已完成 extract 的模型 (RUN_DIR/<model>/tracks/*.json 非空):
# 作为 SUMMARY_MODELS 的默认值, 保证 stage 2/3 天然带上历史模型.
# 若用户显式指定了 SUMMARY_MODELS, 就照他给的用.
#
# 注意: 这个函数在 stage 2 开始时才调用 (不在脚本启动时), 这样 stage 1 刚刚
# 落盘的新模型 tracks 也能被发现进入汇总; 否则会出现 "追加跑新模型时汇总只带旧模型" 的坑.
# ---------------------------------------------------------------
discover_summary_models() {
    local out=""
    if [[ ! -d "$RUN_DIR" ]]; then
        echo ""; return
    fi
    for sub in "$RUN_DIR"/*/ ; do
        [[ -d "$sub" ]] || continue
        local name; name="$(basename "$sub")"
        # 跳过异常/隐藏目录
        case "$name" in _*|.*) continue ;; esac
        # 只要 tracks 目录下有 *.json 就算数
        if compgen -G "$sub"tracks/"*.json" > /dev/null 2>&1; then
            [[ -n "$out" ]] && out="$out,$name" || out="$name"
        fi
    done
    echo "$out"
}

# ============== 阶段 1: PaddleOCR CPU 抽 tracks ==============
if [[ "$RUN_EXTRACT" == "1" ]]; then
    echo
    echo "[ocr.ref.cpu] === stage 1: extract_tracks (CPU × $N_SHARDS shards) ==="

    EXTRACT_EXTRA=()
    [[ "$LIMIT" != "0" ]]         && EXTRACT_EXTRA+=(--limit "$LIMIT")
    [[ "$SKIP_EXISTING" == "1" ]] && EXTRACT_EXTRA+=(--skip-existing)
    [[ -n "$ONLY_MODELS" ]]       && EXTRACT_EXTRA+=(--only-models "$ONLY_MODELS")

    if [[ "$N_SHARDS" -le 1 ]]; then
        LOG="$RUN_DIR/extract.cpu.log"
        echo "[ocr.ref.cpu] single-shard, log=$LOG"
        CUDA_VISIBLE_DEVICES="" \
        OCR_DEVICE="cpu" \
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
        rm -f "$RUN_DIR"/extract.cpu.shard*.log 2>/dev/null || true

        pids=()
        for ((i=0; i<N_SHARDS; i++)); do
            LOG="$RUN_DIR/extract.cpu.shard${i}.log"
            CUDA_VISIBLE_DEVICES="" \
            OCR_DEVICE="cpu" \
            python3 -u "$EXTRACT_PY" \
                --models-tsv "$MODELS_TSV" \
                --runs-root  "$RUN_DIR" \
                --ocr-specs  "$OCR_SPECS" \
                --shard-idx  "$i" --shard-num "$N_SHARDS" \
                "${EXTRACT_EXTRA[@]}" \
                > "$LOG" 2>&1 &
            pids+=($!)
            echo "[ocr.ref.cpu]   spawned shard $i  pid=${pids[$i]}  log=$LOG"
        done

        # 前台等所有子进程; 收到 Ctrl-C 会一并退出 (set -e + trap)
        trap 'echo "[ocr.ref.cpu] Ctrl-C -> killing shards"; kill "${pids[@]}" 2>/dev/null || true; exit 130' INT TERM

        fail=0
        for ((i=0; i<N_SHARDS; i++)); do
            rc=0; wait "${pids[$i]}" || rc=$?
            if [[ "$rc" -ne 0 ]]; then
                echo "[ocr.ref.cpu]   shard $i FAILED rc=$rc (see $RUN_DIR/extract.cpu.shard${i}.log)" >&2
                fail=$((fail+1))
            else
                echo "[ocr.ref.cpu]   shard $i done"
            fi
        done
        trap - INT TERM
        if [[ "$fail" -gt 0 ]]; then
            echo "[ocr.ref.cpu] $fail / $N_SHARDS shards failed, skip score" >&2; exit 1
        fi

        for sub in "$RUN_DIR"/*/ ; do
            [[ -d "$sub" ]] || continue
            if compgen -G "$sub"summary.shard*.json > /dev/null; then
                python3 "$MERGE_PY" --out-dir "$sub" --shard-num "$N_SHARDS"
            fi
        done
    fi
fi

# ============== 阶段 2: 参考式打分 ==============
if [[ "$RUN_SCORE" == "1" ]]; then
    echo
    echo "[ocr.ref.cpu] === stage 2: score_reference (offline, no LLM) ==="

    # 在 stage 2 开始时才解析 SUMMARY_MODELS: 这样 stage 1 刚落盘的新模型也会进汇总.
    if [[ -z "$SUMMARY_MODELS" ]]; then
        SUMMARY_MODELS="$(discover_summary_models)"
        if [[ -z "$SUMMARY_MODELS" ]]; then
            echo "[ocr.ref.cpu] [WARN] no model with tracks/ under $RUN_DIR, SUMMARY_MODELS 兜底 ONLY_MODELS=$ONLY_MODELS"
            SUMMARY_MODELS="$ONLY_MODELS"
        else
            echo "[ocr.ref.cpu] SUMMARY_MODELS auto-discovered = $SUMMARY_MODELS"
        fi
    else
        echo "[ocr.ref.cpu] SUMMARY_MODELS (from env) = $SUMMARY_MODELS"
    fi

    echo "[ocr.ref.cpu]   using SUMMARY_MODELS = $SUMMARY_MODELS"
    LOG="$SCORE_DIR/_score.log"
    mkdir -p "$SCORE_DIR"
    python3 -u "$SCORE_PY" \
        --runs-root  "$RUN_DIR" \
        --ocr-specs  "$OCR_SPECS" \
        --models     "$SUMMARY_MODELS" \
        --out-dir    "$SCORE_DIR" \
        --num-frames "$NUM_FRAMES" \
        --min-text-score "$MIN_TEXT_SCORE" \
        --fuzzy-threshold "$FUZZY_THRESHOLD" \
        --max-video-duration-sec "$MAX_VIDEO_DURATION_SEC" \
        2>&1 | tee "$LOG"
fi

# ============== 阶段 3: 按镜头数汇总 ==============
if [[ "$RUN_SHOT_SUMMARY" == "1" ]]; then
    echo
    echo "[ocr.ref.cpu] === stage 3: summarize_by_shot (single vs multi) ==="
    if [[ ! -f "$SHOT_PY" ]]; then
        echo "[ERROR] summarize_by_shot.py not found: $SHOT_PY" >&2; exit 1
    fi

    # 若跳过 stage 2 直接跑 stage 3 (RUN_SCORE=0), 这里也要保证 SUMMARY_MODELS 有值.
    if [[ -z "$SUMMARY_MODELS" ]]; then
        SUMMARY_MODELS="$(discover_summary_models)"
        if [[ -z "$SUMMARY_MODELS" ]]; then
            echo "[ocr.ref.cpu] [WARN] no model with tracks/ under $RUN_DIR, SUMMARY_MODELS 兜底 ONLY_MODELS=$ONLY_MODELS"
            SUMMARY_MODELS="$ONLY_MODELS"
        else
            echo "[ocr.ref.cpu] SUMMARY_MODELS auto-discovered = $SUMMARY_MODELS"
        fi
    fi

    LOG="$SCORE_DIR/_shot_summary.log"
    echo "[ocr.ref.cpu]   using SUMMARY_MODELS = $SUMMARY_MODELS"
    python3 -u "$SHOT_PY" \
        --score-root "$SCORE_DIR" \
        --ocr-specs  "$OCR_SPECS" \
        --models     "$SUMMARY_MODELS" \
        --out-dir    "$SCORE_DIR" \
        2>&1 | tee "$LOG"
fi

echo
echo "========================================================="
echo "[DONE] tracks under      : $RUN_DIR/<model>/tracks/"
echo "       per-video score   : $SCORE_DIR/<model>/<vid>.json"
echo "       per-model summary : $SCORE_DIR/<model>/_summary.csv"
echo "       compare table     : $SCORE_DIR/compare_summary.csv"
echo "       compare all/single/multi: $SCORE_DIR/compare_summary_ocr{,_single,_multi}.csv"
echo "========================================================="
