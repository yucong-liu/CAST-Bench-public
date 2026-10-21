#!/usr/bin/env bash
# =====================================================================
# audio_consistency 评测点 · 多模型批跑
#
# 流水线 (一次性串完):
#   1) pipeline/prepare_clips.py  (按视频模型循环, 仅切镜头, 不加载大模型)
#        - single_shot 直通, multi_shot 用 TransNetV2 切镜头
#        - 输出 runs/<model>/clips/  和  runs/<model>/shotcut_manifest.jsonl
#   2) batch_eval.py  (一次加载 Qwen, 跑完所有视频模型)
#        - 把所有 (model, manifest, out_dir) 写到 runs/_jobs.tsv
#        - 起 N_SHARDS 个进程 (每张卡 1 个), 每个进程**只加载 1 次** Qwen3-Omni,
#          然后顺序跑完 jobs.tsv 里的所有视频模型
#        - 每个视频模型的输出仍写到 runs/<model>/per_question.shard*.jsonl
#        - 全部跑完后, 对每个模型逐一调用 merge_shards.py
#   3) aggregate_summary.py  自动汇总到 runs/compare_summary*.csv
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
EVAL_ROOT=$REPO_ROOT/evaluation/audio_consistency
# RUN_DIR_BASE = 原始 runs 根目录 (prepare 阶段的 local 缓存复用查这里的子目录)
# RUN_DIR      = 本脚本实际用的输出根:
#                  local 模式 = RUN_DIR_BASE/qwen3_omni       (Qwen3-Omni 30B 本地推理)
#                  api   模式 = RUN_DIR_BASE/qwen3.5_omni     (qwen3.5-omni-plus 平台)
#                两套结果完全隔离, 互不覆盖, 方便左右对照.
RUN_DIR_BASE=$EVAL_ROOT/runs
RUN_DIR=$RUN_DIR_BASE
SPECS_DIR=$REPO_ROOT/t2av-bench/specs
# 评测用 questions: 默认走 t2av-bench 统一维护的 audio_consistency_llm.jsonl
QUESTIONS="${QUESTIONS:-$REPO_ROOT/t2av-bench/questions/audio_consistency_llm.jsonl}"
SYSTEM_PROMPT=$EVAL_ROOT/system_prompt.md
MODEL_PATH=$MODEL_ROOT/Qwen3-Omni/Qwen3-Omni-30B-A3B-Instruct

# Backend 选择:
#   local  -> 本地加载 Qwen3-Omni 30B (默认); 输出: runs/qwen3_omni/<model>/
#   api    -> 走平台 qwen3.5-omni-plus API ; 输出: runs/qwen3.5_omni/<model>/
#            需要提前 export QWEN35_APP_ID / QWEN35_APP_KEY
BACKEND="${BACKEND:-local}"
# 两套 backend 的输出子目录 (可用环境变量覆盖, 一般无需改)
LOCAL_SUBDIR="${LOCAL_SUBDIR:-qwen3_omni}"
API_SUBDIR="${API_SUBDIR:-qwen3.5_omni}"
if [[ "$BACKEND" == "api" ]]; then
    RUN_DIR="$RUN_DIR_BASE/$API_SUBDIR"
else
    RUN_DIR="$RUN_DIR_BASE/$LOCAL_SUBDIR"
fi

PIPE_DIR=$EVAL_ROOT/pipeline
PREPARE_PY=$PIPE_DIR/prepare_clips.py
EVAL_PY=$EVAL_ROOT/batch_eval.py
MERGE_PY=$EVAL_ROOT/merge_shards.py
AGG_PY=$EVAL_ROOT/aggregate_summary.py

THRESHOLD="${THRESHOLD:-0.5}"
# grouped 模式: 一次性问完同一个 clip 下的全部问题, 单次 generate 的 token 预算
# (batch_eval.py 内部还会按 N 题动态再加: min(4096, base + 256*N))
# Instruct 变体直接吐 JSON, 单题 ~80 tokens 就够, 默认 512 已留足余量
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
LIMIT="${LIMIT:-0}"   # 0 = 不限制 (调试时可设为 5 等小值)

# DP 8x1: 8 个进程, 每个绑一张卡, 各跑 1/N 切片
# (local backend 默认 8; api backend 默认 1, 都允许通过环境变量 N_SHARDS 覆盖)
# api 模式下 N_SHARDS=K 表示起 K 个进程并行跑 API (无 GPU 占用), 用于提升吞吐.
if [[ -z "${N_SHARDS:-}" ]]; then
    if [[ "$BACKEND" == "api" ]]; then
        N_SHARDS=1
    else
        N_SHARDS=8
    fi
fi

MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"

# 阶段开关 (默认两个阶段都跑)
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_OMNI="${RUN_OMNI:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
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
#       * 阶段 1/2 只处理这些模型 (prepare/omni 只对它们跑)
#       * 阶段 3 aggregate 自动加 --append-models "$ONLY_MODELS", 不改动
#         其他模型在 3 张 CSV 里的原值;
#         compare_summary_multi_intersect.csv 追加模式下整张不动.
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
#   1) <root> 本身 (常规)
#   2) <root> 下唯一的中间层 (例如 daVinci-MagiHuman: output/540p)
src_root_for() {
    local root="${SRC_MAP[$1]:-}"
    [[ -z "$root" ]] && { echo ""; return; }
    # 直接在 root 下找到 action/ambient/.../speech 之类的子目录就用它
    if [[ -d "$root/action" || -d "$root/speech" || -d "$root/object" \
       || -d "$root/sfx" || -d "$root/ambient" || -d "$root/animal" \
       || -d "$root/bgm" || -d "$root/instrument" || -d "$root/non_verbal_vocal" ]]; then
        echo "$root"; return
    fi
    # 兜底: root 下找一层中间目录, 该中间目录下含 speech/action/...
    local cand
    cand=$(find "$root" -mindepth 2 -maxdepth 2 -type d \
           \( -name speech -o -name action -o -name object -o -name sfx \
              -o -name ambient -o -name animal -o -name bgm \
              -o -name instrument -o -name non_verbal_vocal \) \
           2>/dev/null | head -n 1)
    if [[ -n "$cand" ]]; then
        echo "$(dirname "$cand")"; return
    fi
    echo "$root"   # fallback: 让下游自己打印 not-found
}

mkdir -p "$RUN_DIR"

# api backend: 检查鉴权环境变量
if [[ "$BACKEND" == "api" ]]; then
    if [[ -z "${QWEN35_APP_ID:-}" || -z "${QWEN35_APP_KEY:-}" ]]; then
        echo "[ERROR] BACKEND=api 需要 export QWEN35_APP_ID / QWEN35_APP_KEY" >&2
        exit 1
    fi
fi

# 输出目录: 统一走 $RUN_DIR/<model>/
# (api 模式下 RUN_DIR 已被重定向到 runs/qwen3.5_omni/, 所以自然得到 runs/qwen3.5_omni/<model>/)
out_dir_for() {
    local m="$1"
    echo "$RUN_DIR/$m"
}

echo "=========================================================="
echo "[run_all] EVAL_ROOT     = $EVAL_ROOT"
echo "[run_all] BACKEND       = $BACKEND"
echo "[run_all] RUN_DIR       = $RUN_DIR"
echo "[run_all] MODELS_TSV    = $MODELS_TSV"
echo "[run_all] MODEL_PATH    = $MODEL_PATH"
echo "[run_all] RUN_PREPARE=$RUN_PREPARE  RUN_OMNI=$RUN_OMNI  RUN_AGGREGATE=$RUN_AGGREGATE"
echo "[run_all] LIMIT=$LIMIT  THRESHOLD=$THRESHOLD  MAX_NEW_TOKENS=$MAX_NEW_TOKENS  N_SHARDS=$N_SHARDS"
echo "[run_all] models from tsv: ${MODELS[*]}"
echo "=========================================================="

# ============== 阶段 1: 切镜头 ==============
if [[ "$RUN_PREPARE" == "1" ]]; then
    for M in "${MODELS[@]}"; do
        SRC="$(src_root_for "$M")"
        OUT="$(out_dir_for "$M")"
        echo
        echo "############ [prepare_clips] $M ############"
        echo "  src = $SRC"
        echo "  out = $OUT"
        if [[ -z "$SRC" || ! -d "$SRC" ]]; then
            echo "[run_all] $M: src not found, skip"; continue
        fi
        mkdir -p "$OUT"
        # API 模式: 切片不依赖大模型, 优先复用 local 模式已切好的 manifest/clips
        # (位于 $RUN_DIR_BASE/$LOCAL_SUBDIR/$M/ 或老的 $RUN_DIR_BASE/$M/);
        # 若无 local 缓存, 也直接在本机自切片 (TransNetV2 仅占少量 GPU).
        # 已存在的本目录 manifest/clips 直接复用, 不重切.
        if [[ "$BACKEND" == "api" ]]; then
            if [[ -f "$OUT/shotcut_manifest.jsonl" && -d "$OUT/clips" ]]; then
                echo "[run_all] $M: $OUT 已有 manifest+clips, reuse (skip prepare)"
                continue
            fi
            LOCAL_OUT="$RUN_DIR_BASE/$LOCAL_SUBDIR/$M"
            if [[ ! -f "$LOCAL_OUT/shotcut_manifest.jsonl" || ! -d "$LOCAL_OUT/clips" ]]; then
                # 回退到老的扁平布局 (兼容历史 runs/<model>/)
                if [[ -f "$RUN_DIR_BASE/$M/shotcut_manifest.jsonl" && -d "$RUN_DIR_BASE/$M/clips" ]]; then
                    LOCAL_OUT="$RUN_DIR_BASE/$M"
                fi
            fi
            if [[ -f "$LOCAL_OUT/shotcut_manifest.jsonl" && -d "$LOCAL_OUT/clips" ]]; then
                echo "[run_all] $M: reuse local prepare ($LOCAL_OUT) by symlink"
                ln -sfn "$LOCAL_OUT/shotcut_manifest.jsonl" "$OUT/shotcut_manifest.jsonl"
                ln -sfn "$LOCAL_OUT/clips" "$OUT/clips"
                continue
            fi
            echo "[run_all] $M: no local prepare cache, run prepare_clips.py (api 模式自切片)"
            python3 "$PREPARE_PY" \
                --video-root "$SRC" \
                --specs-dir  "$SPECS_DIR" \
                --out-dir    "$OUT" \
                --tag        "$M" \
                --threshold  "$THRESHOLD"
            continue
        fi
        python3 "$PREPARE_PY" \
            --video-root "$SRC" \
            --specs-dir  "$SPECS_DIR" \
            --out-dir    "$OUT" \
            --tag        "$M" \
            --threshold  "$THRESHOLD"
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
        MANIFEST="$OUT/shotcut_manifest.jsonl"
        if [[ ! -f "$MANIFEST" ]]; then
            echo "[run_all] $M: manifest not found ($MANIFEST), skip"; continue
        fi
        # tag<TAB>manifest<TAB>out_dir
        printf "%s\t%s\t%s\n" "$M" "$MANIFEST" "$OUT" >> "$JOBS_TSV"
        JOB_MODELS+=("$M")
        # 清掉本模型上一次的 shard 残留
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

        # 进程级日志统一放在 $RUN_DIR/omni.shard${i}.log
        rm -f "$RUN_DIR"/omni.shard*.log
        for M in "${JOB_MODELS[@]}"; do
            OUT="$(out_dir_for "$M")"
            rm -f "$OUT"/omni.shard*.log
        done

        pids=()
        for ((i=0; i<N_SHARDS; i++)); do
            LOG="$RUN_DIR/omni.shard${i}.log"
            if [[ "$BACKEND" == "api" ]]; then
                # API 无 GPU; 不设 CUDA_VISIBLE_DEVICES, 不传 --model-path
                python3 -u "$EVAL_PY" \
                    --jobs-tsv      "$JOBS_TSV" \
                    --questions     "$QUESTIONS" \
                    --specs-dir     "$SPECS_DIR" \
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
                    --specs-dir     "$SPECS_DIR" \
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
            # N_SHARDS=1 时 batch_eval.py 直接输出 per_question.jsonl, 不需 merge
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

# ============== 阶段 3: 跨模型汇总 ==============
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
