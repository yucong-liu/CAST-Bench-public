#!/usr/bin/env bash
# =====================================================================
# emtion_consistency 评测点 · 多模型批跑
#
# 流水线 (一次性串完):
#   1) prepare_manifest.py  (按视频模型循环, 仅生成 manifest, 不加载大模型)
#        - 不切片. 仅扫 <video-root>/speech/<spec_id>.mp4
#        - 输出 runs/<model>/manifest.jsonl
#   2) batch_eval.py  (一次加载 Qwen, 跑完所有视频模型)
#        - 把所有 (model, manifest, out_dir) 写到 runs/_jobs.tsv
#        - 起 N_SHARDS 个进程 (每张卡 1 个), 每个进程**只加载 1 次** Qwen3-Omni,
#          然后顺序跑完 jobs.tsv 里的所有视频模型
#        - 每个视频模型的输出仍写到 runs/<model>/per_question.shard*.jsonl
#        - 全部跑完后, 对每个模型逐一调用 merge_shards.py
#   3) aggregate_summary.py  自动汇总到 runs/compare_summary*.csv
#   4) score_calculation.py
#        - 计算每个模型的分数 (per-spec stair-bonus + single/multi/overall 分桶)
#        - 更新 runs/<model>/summary.json
#
# 模型清单: evaluation/models.tsv (两列 TSV: 模型名\t视频 output 根目录)
# 新增模型 = 在 models.tsv 加一行, 不需要改本脚本.
#
# 全部前台运行, 不使用 nohup/&/setsid 等后台手段 (用户用 tmux 自己管会话).
# (Python 进程通过 'cmd > LOG 2>&1 &' + wait 并发, 但脚本本身始终前台.)
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
export REPO_ROOT MODEL_ROOT

# ---------------- 用户可调参数 ----------------
EVAL_ROOT=$REPO_ROOT/evaluation/emtion_consistency
# RUN_DIR_BASE = 原始 runs/; RUN_DIR = 本脚本实际输出根
#   local -> $RUN_DIR_BASE/qwen3_omni
#   api   -> $RUN_DIR_BASE/qwen3.5_omni
RUN_DIR_BASE=$EVAL_ROOT/runs
RUN_DIR=$RUN_DIR_BASE
SPECS_DIR=$REPO_ROOT/t2av-bench/specs
QUESTIONS="${QUESTIONS:-$REPO_ROOT/t2av-bench/questions/emotion_consistency.jsonl}"
SYSTEM_PROMPT=$EVAL_ROOT/system_prompt.md
MODEL_PATH=$MODEL_ROOT/Qwen3-Omni/Qwen3-Omni-30B-A3B-Instruct

# Backend 选择: local (默认) | api (走 qwen3.5-omni-plus 平台)
#   local 输出: runs/qwen3_omni/<model>/
#   api   输出: runs/qwen3.5_omni/<model>/    (与 local 结果独立)
BACKEND="${BACKEND:-local}"
LOCAL_SUBDIR="${LOCAL_SUBDIR:-qwen3_omni}"
API_SUBDIR="${API_SUBDIR:-qwen3.5_omni}"

# SPLIT_MODALITY: 0 (默认) = 旧行为 (视频+音轨一起送)
#                 1         = 新行为: audio-only 题只送 wav (ffmpeg -vn),
#                              video-only 题只送去音视频 (use_audio_in_video=False)
#                              输出至独立子目录, 不覆盖老结果
SPLIT_MODALITY="${SPLIT_MODALITY:-0}"
LOCAL_SUBDIR_SPLIT="${LOCAL_SUBDIR_SPLIT:-qwen3_omni_split}"
API_SUBDIR_SPLIT="${API_SUBDIR_SPLIT:-qwen3.5_omni_split}"

if [[ "$BACKEND" == "api" ]]; then
    if [[ "$SPLIT_MODALITY" == "1" ]]; then
        RUN_DIR="$RUN_DIR_BASE/$API_SUBDIR_SPLIT"
    else
        RUN_DIR="$RUN_DIR_BASE/$API_SUBDIR"
    fi
else
    if [[ "$SPLIT_MODALITY" == "1" ]]; then
        RUN_DIR="$RUN_DIR_BASE/$LOCAL_SUBDIR_SPLIT"
    else
        RUN_DIR="$RUN_DIR_BASE/$LOCAL_SUBDIR"
    fi
fi

PREPARE_PY=$EVAL_ROOT/prepare_manifest.py
EVAL_PY=$EVAL_ROOT/batch_eval.py
MERGE_PY=$EVAL_ROOT/merge_shards.py
AGG_PY=$EVAL_ROOT/aggregate_summary.py
SCORE_PY=$EVAL_ROOT/score_calculation.py

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
LIMIT="${LIMIT:-0}"   # 0 = 不限制 (调试时可设为 5 等小值)

# DP: N 个进程, 每个绑一张卡, 各跑 1/N 切片 (按 spec_id 维度切)
N_SHARDS="${N_SHARDS:-8}"
if [[ "$BACKEND" == "api" ]]; then
    N_SHARDS=1
fi

MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"

# 阶段开关 (默认全跑)
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_OMNI="${RUN_OMNI:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
RUN_SCORE_CALCULATION="${RUN_SCORE_CALCULATION:-1}"
# ----------------------------------------------

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
#       * 阶段 1/2/3 只处理这些模型 (prepare/omni/score_calc 都只对它们跑)
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
#   1) <root>/speech/  存在 => 用 <root>
#   2) <root>/<中间一层>/speech/ 存在 (例如 daVinci-MagiHuman: output/540p) => 用中间层
src_root_for() {
    local root="${SRC_MAP[$1]:-}"
    [[ -z "$root" ]] && { echo ""; return; }

    if [[ -d "$root/speech" ]]; then
        echo "$root"; return
    fi

    local cand
    cand=$(find "$root" -mindepth 2 -maxdepth 2 -type d -name speech 2>/dev/null | head -n 1)
    if [[ -n "$cand" ]]; then
        echo "$(dirname "$cand")"; return
    fi

    echo "$root"   # fallback: 让下游打印 not-found
}

mkdir -p "$RUN_DIR"

if [[ "$BACKEND" == "api" ]]; then
    if [[ -z "${QWEN35_APP_ID:-}" || -z "${QWEN35_APP_KEY:-}" ]]; then
        echo "[ERROR] BACKEND=api 需要 export QWEN35_APP_ID / QWEN35_APP_KEY" >&2
        exit 1
    fi
fi

out_dir_for() {
    local m="$1"
    echo "$RUN_DIR/$m"
}

echo "========================================================="
echo "[run_all] EVAL_ROOT     = $EVAL_ROOT"
echo "[run_all] BACKEND       = $BACKEND"
echo "[run_all] SPLIT_MODALITY= $SPLIT_MODALITY"
echo "[run_all] RUN_DIR       = $RUN_DIR"
echo "[run_all] MODELS_TSV    = $MODELS_TSV"
echo "[run_all] MODEL_PATH    = $MODEL_PATH"
echo "[run_all] RUN_PREPARE=$RUN_PREPARE  RUN_OMNI=$RUN_OMNI  RUN_AGGREGATE=$RUN_AGGREGATE"
echo "[run_all] LIMIT=$LIMIT  MAX_NEW_TOKENS=$MAX_NEW_TOKENS  N_SHARDS=$N_SHARDS"
echo "[run_all] models from tsv: ${MODELS[*]}"
echo "========================================================="
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
        # API 模式: 优先复用 local 模式已生成的 manifest (位于
        # $RUN_DIR_BASE/$LOCAL_SUBDIR/$M/ 或老的 $RUN_DIR_BASE/$M/);
        # 没有 local 缓存就 skip (API 模式不在本环境扫视频).
        if [[ "$BACKEND" == "api" ]]; then
            # split-modality 模式下优先复用 local split 的 manifest, 其次扭回普通 local
            if [[ "$SPLIT_MODALITY" == "1" ]]; then
                LOCAL_OUT="$RUN_DIR_BASE/$LOCAL_SUBDIR_SPLIT/$M"
                if [[ ! -f "$LOCAL_OUT/manifest.jsonl" ]]; then
                    LOCAL_OUT="$RUN_DIR_BASE/$LOCAL_SUBDIR/$M"
                fi
            else
                LOCAL_OUT="$RUN_DIR_BASE/$LOCAL_SUBDIR/$M"
            fi
            if [[ ! -f "$LOCAL_OUT/manifest.jsonl" && -f "$RUN_DIR_BASE/$M/manifest.jsonl" ]]; then
                LOCAL_OUT="$RUN_DIR_BASE/$M"
            fi
            if [[ -f "$LOCAL_OUT/manifest.jsonl" ]]; then
                echo "[run_all] $M: reuse local manifest ($LOCAL_OUT/manifest.jsonl) by symlink"
                ln -sfn "$LOCAL_OUT/manifest.jsonl" "$OUT/manifest.jsonl"
            else
                echo "[run_all] $M: no local manifest, SKIP (api 模式不扫视频)"
            fi
            continue
        fi
        python3 "$PREPARE_PY" \
            --video-root "$SRC" \
            --specs-dir  "$SPECS_DIR" \
            --questions  "$QUESTIONS" \
            --out-dir    "$OUT" \
            --tag        "$M"
    done
fi

# ============== 阶段 2: Omni 评测 (DP 多卡并发, 一次加载 Qwen 跑所有模型) ==============
# 优化目标: 不再 "外层按视频模型循环 -> 反复加载 Qwen3-Omni"; 而是把所有模型的
# (manifest, out_dir) 写到一个 jobs.tsv 里, 只起 N_SHARDS 个进程 (每张卡 1 个),
# 每个进程加载 1 次 Qwen, 然后顺序跑完 jobs.tsv 里的所有模型 -> Qwen 加载次数
# 从 (#models * N_SHARDS) 降到 N_SHARDS.
if [[ "$RUN_OMNI" == "1" ]]; then
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
        echo "[run_all] no manifest found for any model, skip omni stage"
    else
        echo
        echo "############ [omni_eval]  jobs=${#JOB_MODELS[@]}  DP x ${N_SHARDS} ############"
        echo "[run_all]   jobs_tsv = $JOBS_TSV"
        echo "[run_all]   models   : ${JOB_MODELS[*]}"
        cat "$JOBS_TSV" | sed 's/^/[run_all]     /'

        EXTRA_ARGS=()
        if [[ "$LIMIT" != "0" ]]; then
            EXTRA_ARGS+=(--limit "$LIMIT")
        fi
        if [[ "$SPLIT_MODALITY" == "1" ]]; then
            EXTRA_ARGS+=(--split-modality)
        fi

        rm -f "$RUN_DIR"/omni.shard*.log
        for M in "${JOB_MODELS[@]}"; do
            OUT="$(out_dir_for "$M")"
            rm -f "$OUT"/omni.shard*.log
        done

        pids=()
        for ((i=0; i<N_SHARDS; i++)); do
            LOG="$RUN_DIR/omni.shard${i}.log"
            if [[ "$BACKEND" == "api" ]]; then
                python3 -u "$EVAL_PY" \
                    --jobs-tsv      "$JOBS_TSV" \
                    --questions     "$QUESTIONS" \
                    --system-prompt "$SYSTEM_PROMPT" \
                    --backend       api \
                    --attn-impl     "$ATTN_IMPL" \
                    --max-new-tokens "$MAX_NEW_TOKENS" \
                    --shard-idx     "$i" \
                    --shard-num     "$N_SHARDS" \
                    --device        "auto" \
                    "${EXTRA_ARGS[@]}" \
                    > "$LOG" 2>&1 &
            else
                CUDA_VISIBLE_DEVICES=$i python3 -u "$EVAL_PY" \
                    --jobs-tsv      "$JOBS_TSV" \
                    --questions     "$QUESTIONS" \
                    --system-prompt "$SYSTEM_PROMPT" \
                    --backend       local \
                    --model-path    "$MODEL_PATH" \
                    --attn-impl     "$ATTN_IMPL" \
                    --max-new-tokens "$MAX_NEW_TOKENS" \
                    --shard-idx     "$i" \
                    --shard-num     "$N_SHARDS" \
                    --device        "cuda:0" \
                    "${EXTRA_ARGS[@]}" \
                    > "$LOG" 2>&1 &
            fi
            pids+=($!)
            echo "[run_all]   spawned shard $i  pid=${pids[$i]}  log=$LOG  (handles all ${#JOB_MODELS[@]} models, backend=$BACKEND)"
        done

        # 前台 wait, Ctrl-C 可直接终止整条流水线
        fail=0
        for ((i=0; i<N_SHARDS; i++)); do
            rc=0
            wait "${pids[$i]}" || rc=$?
            if [[ "$rc" -ne 0 ]]; then
                echo "[run_all]   shard $i FAILED rc=$rc (see $RUN_DIR/omni.shard${i}.log)" >&2
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

# ============== 阶段 3: 分数计算 (问题正确性 + 一致性奖励) ==============
# 注意: 必须先跑 score_calculation, 再跑 aggregate_summary —— aggregate_summary
# 里的 score_<m> 列要从 <model>/summary.json 的 score_calculation 段里读数, 反过
# 来跑会拿不到分.
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
        # 追加模式: 只刷新指定模型的列, 保留 3 张 CSV 里其他模型的分数不动
        python3 "$AGG_PY" --runs-dir "$RUN_DIR" --append-models "$ONLY_MODELS"
    else
        python3 "$AGG_PY" --runs-dir "$RUN_DIR"
    fi
fi

echo
echo "=========================================================="
echo "[DONE] outputs under: $RUN_DIR"
echo "=========================================================="
