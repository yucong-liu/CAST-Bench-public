#!/usr/bin/env bash
# =====================================================================
# subject_consistency 评测点 · 多模型批跑
#
# 流水线 (一次性串完):
#   1) prepare_manifest.py
#        - 不切片. 跨 9 个 category 扫 <video-root>/<cat>/<spec_id>.mp4
#        - 输出 runs/<judge>/<model>/manifest.jsonl
#   2) batch_eval.py
#        - judge_backend = qwen_vl (本地 vLLM Qwen3-VL)
#        - 走 OpenAI 兼容 API (默认 http://127.0.0.1:8080/v1, 模型
#          Qwen3-VL-235B-A22B-Instruct), 与 t2av-bench/SETUP.md 同一套服务
#        - 整段视频 base64 发给 VLM, 单 spec 内多题 grouped 一次推理
#        - 多 job 模式: 一个进程内顺序跑完所有视频模型 (jobs.tsv)
#        - DP 并发 (N_SHARDS>=2 时): 起 N 个 python 进程, 按 spec 取模分片,
#          每个进程独立向 vLLM 发请求, 利用服务端并发
#   3) merge_shards.py
#        - 把每个模型的 per_question.shard*.jsonl 合并成最终 per_question.jsonl
#   4) score_calculation.py
#        - 每题答对 +1 分; 每个视频归一化到 0-1; 按 spec 平均, 拆 single / multi
#   5) aggregate_summary.py
#        - 跨模型按 category 拆 compare_summary*.csv
#
# 模型清单: evaluation/models.tsv (两列 TSV: 模型名\t视频 output 根目录)
# 新增模型 = 在 models.tsv 加一行, 不需要改本脚本.
#
# 全部前台运行, 不使用 nohup/&/setsid. 内部并发进程仍是 & + wait, 顶层
# 脚本始终前台 (Ctrl-C 可终止整条流水线).
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# ---------------- 用户可调参数 ----------------
EVAL_ROOT=$REPO_ROOT/evaluation/subject_consistency
# RUN_DIR_BASE = runs/; 实际写到 runs/<JUDGE_SUBDIR>/<video_model>/
#   JUDGE_BACKEND=qwen_vl -> runs/qwen3_vl/<video_model>/
# 不同 judge 模型完全隔离, 互不覆盖.
RUN_DIR_BASE=$EVAL_ROOT/runs
SPECS_DIR=$REPO_ROOT/t2av-bench/specs
QUESTIONS="${QUESTIONS:-$REPO_ROOT/t2av-bench/questions/subject_consistency_llm.jsonl}"
SYSTEM_PROMPT=$EVAL_ROOT/system_prompt.md

PREPARE_PY=$EVAL_ROOT/prepare_manifest.py
EVAL_PY=$EVAL_ROOT/batch_eval.py
MERGE_PY=$EVAL_ROOT/merge_shards.py
AGG_PY=$EVAL_ROOT/aggregate_summary.py
SCORE_PY=$EVAL_ROOT/score_calculation.py

# Judge backend: qwen_vl (本地 vLLM Qwen3-VL-235B)
JUDGE_BACKEND="${JUDGE_BACKEND:-qwen_vl}"
QWEN_VL_SUBDIR="${QWEN_VL_SUBDIR:-qwen3_vl}"
case "$JUDGE_BACKEND" in
    qwen_vl)  RUN_DIR="$RUN_DIR_BASE/$QWEN_VL_SUBDIR" ;;
    *) echo "[ERROR] unknown JUDGE_BACKEND=$JUDGE_BACKEND (allow: qwen_vl)" >&2; exit 1 ;;
esac

# qwen_vl backend: 与 t2av-bench/SETUP.md 一致 (本地 vLLM)
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8080/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3-VL-235B-A22B-Instruct}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_TIMEOUT_SEC="${JUDGE_TIMEOUT_SEC:-600}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_TOP_P="${JUDGE_TOP_P:-0.9}"
JUDGE_MAX_RETRY="${JUDGE_MAX_RETRY:-3}"
JUDGE_VIDEO_LOCAL_PATH="${JUDGE_VIDEO_LOCAL_PATH:-0}"  # 1=传 file:// 给 vLLM, 0=base64 data URL

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LIMIT="${LIMIT:-0}"

# DP: vLLM 服务自带并发, 我们用 N 个客户端进程分摊视频, 每个进程串行调 API.
# 起多少个客户端不依赖 GPU 卡数, 但太多会把 vLLM 排队队列拉满, 4~8 较稳.
N_SHARDS="${N_SHARDS:-4}"

MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
RUN_SCORE_CALCULATION="${RUN_SCORE_CALCULATION:-1}"
# ----------------------------------------------

# 关键: vLLM 在 127.0.0.1, 一定要把 localhost 加到 NO_PROXY, 否则可能被
# HTTPS 代理拦截 (与 t2av-bench/SETUP.md §3.3 同一坑位).
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

declare -A SRC_MAP=()
MODELS=()
load_models() {
    if [[ ! -f "$MODELS_TSV" ]]; then
        echo "[ERROR] models.tsv not found: $MODELS_TSV" >&2; exit 1
    fi
    while IFS=$'\t' read -r name src || [[ -n "$name" ]]; do
        [[ -n "${src:-}" && "$src" != /* ]] && src="$REPO_ROOT/$src"
        [[ -z "$name" || "$name" =~ ^[[:space:]]*# ]] && continue
        MODELS+=("$name"); SRC_MAP["$name"]="$src"
    done < "$MODELS_TSV"
}
load_models

# ---- 可选: ONLY_MODELS=<name>[,<name>...] 只跑指定模型 (追加模式) ----
#   - 默认不设 => 走全量流程 (行为一字不变)
#   - 设了则:
#       * 阶段 1/2/3 只处理这些模型 (prepare/eval/score_calc 都只对它们跑)
#       * 阶段 4 aggregate 自动加 --append-models "$ONLY_MODELS", 不改动
#         其他模型在 3 张 CSV 里的原值
#   - 拼错的名字 (models.tsv 里没有) 直接报错退出, 避免静默跳过
ONLY_MODELS="${ONLY_MODELS:-}"
if [[ -n "$ONLY_MODELS" ]]; then
    IFS=', ' read -r -a _wanted <<< "$ONLY_MODELS"
    declare -A _wanted_set=()
    for w in "${_wanted[@]}"; do
        [[ -z "$w" ]] && continue
        _wanted_set["$w"]=1
    done
    _filtered=()
    for M in "${MODELS[@]}"; do
        if [[ -n "${_wanted_set[$M]:-}" ]]; then
            _filtered+=("$M")
            unset '_wanted_set[$M]'
        fi
    done
    if [[ ${#_wanted_set[@]} -gt 0 ]]; then
        echo "[ERROR] ONLY_MODELS 里这些名字在 $MODELS_TSV 中不存在: ${!_wanted_set[*]}" >&2
        exit 1
    fi
    if [[ ${#_filtered[@]} -eq 0 ]]; then
        echo "[ERROR] ONLY_MODELS 过滤后 MODELS 为空" >&2
        exit 1
    fi
    MODELS=("${_filtered[@]}")
    echo "[run_all] ONLY_MODELS 过滤后 MODELS = ${MODELS[*]}  (追加模式)"
fi

# 模型名 -> 视频根目录:
#   1) <root> 本身含 category 子目录 (一般情况)
#   2) <root>/<中间一层>/ 含 category 子目录 (兼容 daVinci-MagiHuman: output/540p)
src_root_for() {
    local root="${SRC_MAP[$1]:-}"
    [[ -z "$root" ]] && { echo ""; return; }
    if [[ -d "$root/speech" || -d "$root/action" || -d "$root/object" \
       || -d "$root/sfx" || -d "$root/ambient" || -d "$root/animal" \
       || -d "$root/bgm" || -d "$root/instrument" || -d "$root/non_verbal_vocal" ]]; then
        echo "$root"; return
    fi
    local cand
    cand=$(find "$root" -mindepth 2 -maxdepth 2 -type d \
           \( -name speech -o -name action -o -name object -o -name sfx \
              -o -name ambient -o -name animal -o -name bgm \
              -o -name instrument -o -name non_verbal_vocal \) \
           2>/dev/null | head -n 1)
    if [[ -n "$cand" ]]; then
        echo "$(dirname "$cand")"; return
    fi
    echo "$root"
}

mkdir -p "$RUN_DIR"

out_dir_for() {
    local m="$1"
    echo "$RUN_DIR/$m"
}

echo "=========================================================="
echo "[run_all] EVAL_ROOT      = $EVAL_ROOT"
echo "[run_all] JUDGE_BACKEND  = $JUDGE_BACKEND"
echo "[run_all] RUN_DIR        = $RUN_DIR"
echo "[run_all] QUESTIONS      = $QUESTIONS"
echo "[run_all] SPECS_DIR      = $SPECS_DIR"
echo "[run_all] MODELS_TSV     = $MODELS_TSV"
if [[ "$JUDGE_BACKEND" == "qwen_vl" ]]; then
    echo "[run_all] JUDGE_BASE_URL = $JUDGE_BASE_URL"
    echo "[run_all] JUDGE_MODEL    = $JUDGE_MODEL"
fi
echo "[run_all] RUN_PREPARE=$RUN_PREPARE  RUN_EVAL=$RUN_EVAL  RUN_AGGREGATE=$RUN_AGGREGATE  RUN_SCORE_CALCULATION=$RUN_SCORE_CALCULATION"
echo "[run_all] LIMIT=$LIMIT  MAX_NEW_TOKENS=$MAX_NEW_TOKENS  N_SHARDS=$N_SHARDS"
echo "[run_all] models from tsv: ${MODELS[*]}"
echo "=========================================================="

# ============== 阶段 1: 生成 manifest (不切片) ==============
if [[ "$RUN_PREPARE" == "1" ]]; then
    for M in "${MODELS[@]}"; do
        SRC="$(src_root_for "$M")"
        OUT="$(out_dir_for "$M")"
        echo
        echo "############ [prepare_manifest] $M ############"
        echo "  src = $SRC"
        echo "  out = $OUT"
        if [[ -z "$SRC" || ! -d "$SRC" ]]; then
            echo "[run_all] $M: src not found, skip"; continue
        fi
        mkdir -p "$OUT"
        python3 "$PREPARE_PY" \
            --video-root "$SRC" \
            --specs-dir  "$SPECS_DIR" \
            --questions  "$QUESTIONS" \
            --out-dir    "$OUT" \
            --tag        "$M"
    done
fi

# ============== 阶段 2: VLM 评测 (单进程顺序 or 多进程客户端并发) ==============
if [[ "$RUN_EVAL" == "1" ]]; then
    JOBS_TSV="$RUN_DIR/_jobs.tsv"
    : > "$JOBS_TSV"
    JOB_MODELS=()
    for M in "${MODELS[@]}"; do
        OUT="$(out_dir_for "$M")"
        MANIFEST="$OUT/manifest.jsonl"
        if [[ ! -f "$MANIFEST" ]]; then
            echo "[run_all] $M: manifest not found ($MANIFEST), skip"; continue
        fi
        printf "%s\t%s\t%s\n" "$M" "$MANIFEST" "$OUT" >> "$JOBS_TSV"
        JOB_MODELS+=("$M")
        rm -f "$OUT"/per_question.shard*.jsonl "$OUT"/summary.shard*.json
    done

    if [[ ${#JOB_MODELS[@]} -eq 0 ]]; then
        echo "[run_all] no manifest found for any model, skip eval stage"
    else
        echo
        echo "############ [eval] jobs=${#JOB_MODELS[@]}  DP x ${N_SHARDS} (clients) ############"
        echo "[run_all]   jobs_tsv = $JOBS_TSV"
        echo "[run_all]   models   : ${JOB_MODELS[*]}"
        cat "$JOBS_TSV" | sed 's/^/[run_all]     /'

        EXTRA_ARGS=()
        if [[ "$LIMIT" != "0" ]]; then
            EXTRA_ARGS+=(--limit "$LIMIT")
        fi
        if [[ "$JUDGE_VIDEO_LOCAL_PATH" == "1" ]]; then
            EXTRA_ARGS+=(--judge-video-local-path)
        fi

        rm -f "$RUN_DIR"/eval.shard*.log

        pids=()
        for ((i=0; i<N_SHARDS; i++)); do
            LOG="$RUN_DIR/eval.shard${i}.log"
            python3 -u "$EVAL_PY" \
                --jobs-tsv         "$JOBS_TSV" \
                --questions        "$QUESTIONS" \
                --system-prompt    "$SYSTEM_PROMPT" \
                --max-new-tokens   "$MAX_NEW_TOKENS" \
                --shard-idx        "$i" \
                --shard-num        "$N_SHARDS" \
                --judge-backend    "$JUDGE_BACKEND" \
                --judge-base-url   "$JUDGE_BASE_URL" \
                --judge-model      "$JUDGE_MODEL" \
                --judge-api-key    "$JUDGE_API_KEY" \
                --judge-timeout-sec "$JUDGE_TIMEOUT_SEC" \
                --judge-temperature "$JUDGE_TEMPERATURE" \
                --judge-top-p       "$JUDGE_TOP_P" \
                --judge-max-retry   "$JUDGE_MAX_RETRY" \
                "${EXTRA_ARGS[@]}" \
                > "$LOG" 2>&1 &
            pids+=($!)
            echo "[run_all]   spawned shard $i  pid=${pids[$i]}  log=$LOG  (handles all ${#JOB_MODELS[@]} models)"
        done

        fail=0
        for ((i=0; i<N_SHARDS; i++)); do
            rc=0
            wait "${pids[$i]}" || rc=$?
            if [[ "$rc" -ne 0 ]]; then
                echo "[run_all]   shard $i FAILED rc=$rc (see $RUN_DIR/eval.shard${i}.log)" >&2
                fail=$((fail+1))
            else
                echo "[run_all]   shard $i done"
            fi
        done
        if [[ "$fail" -gt 0 ]]; then
            echo "[run_all] ERROR: $fail / $N_SHARDS shards failed, skip ALL merge_shards" >&2
        else
            if [[ "$N_SHARDS" -le 1 ]]; then
                echo "[run_all] N_SHARDS=1, skip merge_shards (per_question.jsonl already final)"
            else
                for M in "${JOB_MODELS[@]}"; do
                    OUT="$(out_dir_for "$M")"
                    echo
                    echo "############ [merge_shards] $M ############"
                    python3 "$MERGE_PY" --out-dir "$OUT" --shard-num "$N_SHARDS"
                done
            fi
        fi
    fi
fi

# ============== 阶段 3: 分数计算 ==============
# 注意: 必须先跑 score_calculation, 再跑 aggregate_summary
if [[ "$RUN_SCORE_CALCULATION" == "1" ]]; then
    echo
    echo "=========================================================="
    echo "[run_all] score_calculation"
    echo "=========================================================="
    for M in "${MODELS[@]}"; do
        OUT="$(out_dir_for "$M")"
        PER_QUESTION="$OUT/per_question.jsonl"
        echo
        echo "############ [score_calculation] $M ############"
        if [[ ! -f "$PER_QUESTION" ]]; then
            echo "[run_all] $M: per_question.jsonl not found, skip"
            continue
        fi
        python3 "$SCORE_PY" \
            --per-question "$PER_QUESTION" \
            --out-dir "$OUT" \
            --tag "$M"
    done
fi

# ============== 阶段 4: 跨模型汇总 ==============
if [[ "$RUN_AGGREGATE" == "1" ]]; then
    echo
    echo "=========================================================="
    echo "[run_all] aggregate_summary"
    echo "=========================================================="
    if [[ -n "$ONLY_MODELS" ]]; then
        python3 "$AGG_PY" --runs-dir "$RUN_DIR" --append-models "$ONLY_MODELS"
    else
        python3 "$AGG_PY" --runs-dir "$RUN_DIR"
    fi
fi

echo
echo "=========================================================="
echo "[DONE] outputs under: $RUN_DIR"
echo "=========================================================="
