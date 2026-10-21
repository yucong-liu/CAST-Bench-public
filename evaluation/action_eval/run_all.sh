#!/usr/bin/env bash
# =====================================================================
# action_eval 评测点 · 多模型批跑 (VLM judge, 不走 omni)
#
# 流水线 (一次性串完):
#   阶段 1) reuse_clips: 复用 evaluation/audio_consistency 已切好的
#                        shotcut_manifest.jsonl + clips/ , 软链接挂到
#                        evaluation/action_eval/runs/<judge>/<model>/.
#                        若 audio_consistency 那边没跑过, 跳过该模型.
#   阶段 2) batch_eval.py:
#                        judge_backend = qwen_vl (默认, 本地 vLLM Qwen3-VL)
#                        每个 (spec, shot) 一个 bucket (5 道 yes/no), clip 送 VLM 一次.
#                        多 job 模式: 一个进程顺序跑完所有视频模型 (jobs.tsv);
#                        DP 客户端并发: 起 N 个 python 进程, 按 bucket 取模分片.
#   阶段 3) merge_shards.py + aggregate_summary.py: 合并/汇总.
#
# 模型清单: evaluation/models.tsv (与其他评测点共享)
# judge 输出目录: runs/qwen3_vl/<m>/.
# 全部前台运行, 不使用 nohup/&/setsid 等后台手段.
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# ---------------- 用户可调参数 ----------------
EVAL_ROOT=$REPO_ROOT/evaluation/action_eval
RUN_DIR_BASE=$EVAL_ROOT/runs
SPECS_DIR=$REPO_ROOT/t2av-bench/specs
QUESTIONS="${QUESTIONS:-$REPO_ROOT/t2av-bench/questions/action_eval_llm.jsonl}"
SYSTEM_PROMPT=$EVAL_ROOT/system_prompt.md
# yes/no 版下只有一份通用 rubric; 若要重新生成 questions, 手动跑
#   python t2av-bench/scripts/15_build_action_eval_questions_llm.py

# 新增 humanoid_integrity 分项 rubric (整段视频 1-5 分严格评分).
# 传空字串就会跳过该分项.
SYSTEM_PROMPT_HUMANOID="${SYSTEM_PROMPT_HUMANOID:-$EVAL_ROOT/system_prompt_humanoid_integrity.md}"
# 只跑 humanoid 分项 (不跑 yes/no) 的开关. 用于已经跑过 yes/no 的
# 模型只补 humanoid 新分项. 注意: 这会把旧 per_question.jsonl 覆盖, 旧 yes/no
# 记录会丢! 若需保留旧 yes/no, 请先备份 out 目录, humanoid 跑完后手工 merge 两个
# jsonl 再跑 aggregate_summary --append-models (追加模式不覆盖其它模型 CSV 列).
ONLY_HUMANOID="${ONLY_HUMANOID:-0}"

EVAL_PY=$EVAL_ROOT/batch_eval.py
MERGE_PY=$EVAL_ROOT/merge_shards.py
AGG_PY=$EVAL_ROOT/aggregate_summary.py

# 复用 audio_consistency 已切好的 clips 来源根目录:
#   audio_consistency/runs/$AC_SUBDIR/<model>/{shotcut_manifest.jsonl, clips/}
AC_RUNS_BASE="${AC_RUNS_BASE:-$REPO_ROOT/evaluation/audio_consistency/runs}"
AC_SUBDIR="${AC_SUBDIR:-qwen3_omni}"
AC_RUN_DIR="$AC_RUNS_BASE/$AC_SUBDIR"

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

# 视频采样密度 (仅 qwen_vl backend 生效, 透传到 vLLM Qwen3-VL processor)
# 默认 fps=12, max_frames=128: single-shot 10s 也能吃到 ~120 帧, 多-shot 短 clip 也会获得更密采样.
# qwen_vl 视频抽帧默认: fps=6.0, 不锁 num_frames (=0 时 batch_eval 起手就走 fps_only).
# 若想回到固定帧数模式, 显式设置 VIDEO_MAX_FRAMES=96 之类即可.
VIDEO_FPS="${VIDEO_FPS:-6.0}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-0}"

# DP: 起 N 个客户端进程分摊 bucket, 每个进程串行调 API.
# vLLM 服务端自带并发, 4~8 较稳.
N_SHARDS="${N_SHARDS:-4}"

MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"

# 阶段开关
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
# ----------------------------------------------

# 关键: vLLM 在 127.0.0.1, 务必把 localhost 加到 NO_PROXY,
# 否则可能被HTTPS 代理拦截 (与 t2av-bench/SETUP.md §3.3 同坑).
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
#       * 阶段 1/2 只处理这些模型 (reuse_clips/eval 只对它们跑)
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

mkdir -p "$RUN_DIR"

out_dir_for() {
    local m="$1"
    echo "$RUN_DIR/$m"
}
ac_dir_for() {
    local m="$1"
    echo "$AC_RUN_DIR/$m"
}

echo "=========================================================="
echo "[run_all] EVAL_ROOT      = $EVAL_ROOT"
echo "[run_all] JUDGE_BACKEND  = $JUDGE_BACKEND"
echo "[run_all] RUN_DIR        = $RUN_DIR"
echo "[run_all] AC_RUN_DIR     = $AC_RUN_DIR  (clip 复用源)"
echo "[run_all] QUESTIONS      = $QUESTIONS"
echo "[run_all] SPECS_DIR      = $SPECS_DIR"
echo "[run_all] MODELS_TSV     = $MODELS_TSV"
echo "[run_all] SYS_PROMPT_HUMANOID = $SYSTEM_PROMPT_HUMANOID"
echo "[run_all] ONLY_HUMANOID  = $ONLY_HUMANOID"
if [[ "$JUDGE_BACKEND" == "qwen_vl" ]]; then
    echo "[run_all] JUDGE_BASE_URL = $JUDGE_BASE_URL"
    echo "[run_all] JUDGE_MODEL    = $JUDGE_MODEL"
fi
echo "[run_all] RUN_PREPARE=$RUN_PREPARE  RUN_EVAL=$RUN_EVAL  RUN_AGGREGATE=$RUN_AGGREGATE"
echo "[run_all] LIMIT=$LIMIT  MAX_NEW_TOKENS=$MAX_NEW_TOKENS  N_SHARDS=$N_SHARDS"
if [[ "$JUDGE_BACKEND" == "qwen_vl" ]]; then
    echo "[run_all] VIDEO_FPS=$VIDEO_FPS  VIDEO_MAX_FRAMES=$VIDEO_MAX_FRAMES"
fi
echo "[run_all] models from tsv: ${MODELS[*]}"
echo "=========================================================="

# ============== 阶段 1: 复用 audio_consistency 的 clips ==============
if [[ "$RUN_PREPARE" == "1" ]]; then
    # yes/no 版下 questions 由 t2av-bench/scripts/15_build_action_eval_questions_llm.py 生成
    # (需要连本地 Qwen3-VL-235B, 慢, 跑一次即可, 不在每次 run_all 里自动重跑).
    if [[ ! -f "$QUESTIONS" ]]; then
        echo "[ERROR] questions file not found: $QUESTIONS" >&2
        echo "[ERROR] 请先跑: python t2av-bench/scripts/15_build_action_eval_questions_llm.py" >&2
        exit 1
    fi

    for M in "${MODELS[@]}"; do
        AC="$(ac_dir_for "$M")"
        OUT="$(out_dir_for "$M")"
        echo
        echo "############ [reuse_clips] $M ############"
        echo "  ac  = $AC"
        echo "  out = $OUT"
        if [[ ! -f "$AC/shotcut_manifest.jsonl" ]]; then
            echo "[run_all] $M: AC manifest not found ($AC/shotcut_manifest.jsonl)"
            echo "[run_all]    -> 请先在 evaluation/audio_consistency/ 跑 run_all.sh 把 clips 切好."
            continue
        fi
        if [[ ! -d "$AC/clips" ]]; then
            echo "[run_all] $M: AC clips dir not found ($AC/clips), skip"
            continue
        fi
        mkdir -p "$OUT"
        # clips 目录直接软链 (其它 category 的 clip 文件留在那也无所谓, 不会被引用)
        ln -sfn "$AC/clips"                   "$OUT/clips"
        # manifest: 不能简单软链, 必须按 category=="action" 过滤后写真实文件,
        # 否则 batch_eval 会把 animal/speech/non_verbal_vocal 也跑了.
        # 同时给每条 spec 补上 original_video_path (humanoid 分项用到整段原视频).
        # 旧软链如果存在, 先删掉再写文件.
        rm -f "$OUT/shotcut_manifest.jsonl"
        _SRC_ROOT="${SRC_MAP[$M]}"
        python3 -c "
import json, os, sys
src = '$AC/shotcut_manifest.jsonl'
dst = '$OUT/shotcut_manifest.jsonl'
src_root = '${_SRC_ROOT}'
n_in = n_out = n_ov_ok = n_ov_missing = 0
def _find_original_video(sid: str, root: str):
    if not (sid and root):
        return ''
    for cand in (
        os.path.join(root, 'action', sid + '.mp4'),
        os.path.join(root, 'output', 'action', sid + '.mp4'),
        os.path.join(root, sid + '.mp4'),
    ):
        if os.path.isfile(cand):
            return cand
    return ''
with open(src, 'r', encoding='utf-8') as fi, open(dst, 'w', encoding='utf-8') as fo:
    for line in fi:
        line = line.strip()
        if not line:
            continue
        n_in += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue
        cat = (obj.get('category') or '').strip()
        sid = (obj.get('spec_id') or '').strip()
        if cat == 'action' or sid.startswith('action_'):
            ov = _find_original_video(sid, src_root)
            if ov:
                obj['original_video_path'] = ov
                n_ov_ok += 1
            else:
                # single_shot 时 shots[0].clip_path 本身就是原视频, batch_eval 可回落
                n_ov_missing += 1
            fo.write(json.dumps(obj, ensure_ascii=False) + '\n')
            n_out += 1
print(f'[run_all]   manifest filter: {n_out}/{n_in} kept (category==action)  '
      f'original_video_path: ok={n_ov_ok} missing={n_ov_missing}')
"
        echo "[run_all] $M: filtered manifest (action-only) + linked clips from $AC"
    done
fi

# ============== 阶段 2: VLM 评测 (DP 多客户端并发) ==============
if [[ "$RUN_EVAL" == "1" ]]; then
    JOBS_TSV="$RUN_DIR/_jobs.tsv"
    : > "$JOBS_TSV"
    JOB_MODELS=()
    for M in "${MODELS[@]}"; do
        OUT="$(out_dir_for "$M")"
        MANIFEST="$OUT/shotcut_manifest.jsonl"
        if [[ ! -e "$MANIFEST" ]]; then
            echo "[run_all] $M: manifest link missing ($MANIFEST), skip"; continue
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
        # humanoid_integrity 分项 (整段视频 1-5 分). 只要 SYSTEM_PROMPT_HUMANOID 非空
        # 且文件存在, 就会开启新分项 (不影响旧 yes/no).
        if [[ -n "$SYSTEM_PROMPT_HUMANOID" && -f "$SYSTEM_PROMPT_HUMANOID" ]]; then
            EXTRA_ARGS+=(--system-prompt-humanoid "$SYSTEM_PROMPT_HUMANOID")
            echo "[run_all]   humanoid_integrity 分项已启用: $SYSTEM_PROMPT_HUMANOID"
            if [[ "$ONLY_HUMANOID" == "1" ]]; then
                EXTRA_ARGS+=(--only-humanoid)
                echo "[run_all]   ONLY_HUMANOID=1: 将跳过 yes/no, 只跑 humanoid 分项"
                echo "[run_all]   警告: per_question.jsonl 会被重写, 旧 yes/no 记录将丢失!"
            fi
        else
            echo "[run_all]   humanoid_integrity 分项未启用 (SYSTEM_PROMPT_HUMANOID 未设或文件不存在)"
        fi

        rm -f "$RUN_DIR"/eval.shard*.log

        pids=()
        for ((i=0; i<N_SHARDS; i++)); do
            LOG="$RUN_DIR/eval.shard${i}.log"
            python3 -u "$EVAL_PY" \
                --jobs-tsv          "$JOBS_TSV" \
                --questions         "$QUESTIONS" \
                --specs-dir         "$SPECS_DIR" \
                --system-prompt     "$SYSTEM_PROMPT" \
                --max-new-tokens    "$MAX_NEW_TOKENS" \
                --shard-idx         "$i" \
                --shard-num         "$N_SHARDS" \
                --judge-backend     "$JUDGE_BACKEND" \
                --judge-base-url    "$JUDGE_BASE_URL" \
                --judge-model       "$JUDGE_MODEL" \
                --judge-api-key     "$JUDGE_API_KEY" \
                --judge-timeout-sec "$JUDGE_TIMEOUT_SEC" \
                --judge-temperature "$JUDGE_TEMPERATURE" \
                --judge-top-p       "$JUDGE_TOP_P" \
                --judge-max-retry   "$JUDGE_MAX_RETRY" \
                --video-fps         "$VIDEO_FPS" \
                --video-max-frames  "$VIDEO_MAX_FRAMES" \
                "${EXTRA_ARGS[@]}" \
                > "$LOG" 2>&1 &
            pids+=($!)
            echo "[run_all]   spawned shard $i  pid=${pids[$i]}  log=$LOG  (handles all ${#JOB_MODELS[@]} models, judge=$JUDGE_BACKEND)"
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
