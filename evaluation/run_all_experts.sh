#!/usr/bin/env bash
# =============================================================================
# evaluation/run_all_experts.sh
#
# 一次性串起全部评测工程. 前置: 已按 setup_all.sh 装好环境 (见 README.md §1).
# 按 3 阶段管理 GPU 显存:
#
#   Phase A — Qwen3-Omni 本地推理窗口 (每个工程内部 8 shard 各自 from_pretrained
#           加载 30B, 前后串行, 一个跑完 (含其内部汇总) 再起下一个;
#           不需要任何 vLLM 服务)
#     A1) audio_consistency          (Qwen3-Omni 本地 from_pretrained)
#     A2) emtion_consistency         (Qwen3-Omni 本地 from_pretrained)
#     A3) id_consistency::transcribe (id_consistency 阶段 1, Omni 本地转写)
#       权重: $MODEL_ROOT/Qwen3-Omni/Qwen3-Omni-30B-A3B-Instruct
#
#   Phase B — Qwen3-VL 服务窗口 (orchestrator 起 vLLM @ 8080)
#     B1) action_eval                (Qwen3-VL API)
#     B2) style_consistency          (Qwen3-VL API)
#     B3) subject_consistency        (Qwen3-VL API)
#     B4) id_consistency::judge      (id_consistency 阶段 2, CPU 计分 + VL judge)
#     [关闭 VL 服务, 释放显存]
#
#   Phase C — 专家窗口 (无服务, 各工程独占本地 GPU/CPU)
#     C1) asr_eval                   (Qwen3-ASR 本地)
#     C2) syncnet_python             (独立 venv)
#     C3) TransNetV2
#     C4) audiobox-aesthetics
#     C5) facial_consistency
#     C6) VBench2-Human_Anatomy
#     C7) Q-Align                    (独立 venv)
#     C8) consistency_segmented      (依赖 audio_consistency/runs/qwen3_omni/)
#     C9) ocr                        (CPU 参考路线)
#     C10) Syncformer                (独立 venv)
#
#   Phase D — 收尾: 18 指标汇总 (aggregate_18_metrics.py)
#
# Qwen3-VL 服务 (Phase B):
#   cd Qwen3-VL-235B_prompt && ./scripts/start_qwen3vl.sh
#       脚本内部自建 tmux 会话 qwen3vl_serve
#       监听 8080, served-model-name=Qwen3-VL-235B-A22B-Instruct
#       权重: $MODEL_ROOT/Qwen3-VL-235B-A22B-Instruct-FP8/
#       首次加载 ~5-10 min
#
# 智能启停:
#   - PLAN 里有 B 阶段的工程才起 VL 服务, B 阶段跑完立即关闭
#   - A / C 阶段不起任何服务
#
# 模型选择语义 (追加新模型友好):
#   * 不传 MODELS  → 按 models.tsv 全量跑
#   * MODELS=my_new bash run_all_experts.sh
#       - 转发给每个工程 (ONLY_MODELS / MODELS / MODELS_TSV / --models)
#       - 每工程只对 my_new 跑一遍, 其他模型旧结果保留
#
# 用法:
#   cd evaluation
#   bash run_all_experts.sh                                       # 全量 A→B→C→D
#   MODELS=my_new bash run_all_experts.sh                         # 追加新模型
#   MODELS=seedance,wan2.7 bash run_all_experts.sh                # 只跑这两个模型
#   EXPERTS=Q-Align,Syncformer bash run_all_experts.sh            # 只跑 C 阶段的这两个
#   PHASES=A,B bash run_all_experts.sh                            # 只跑 A + B 阶段
#   SKIP_SERVICE_MGMT=1 bash run_all_experts.sh                   # 假设服务已由用户起好,
#                                                                 # orchestrator 不管服务生命周期
#   SKIP_IF_DONE=1 MODELS=my_new bash run_all_experts.sh          # 断点续跑: 若某工程
#                                                                 # 对 my_new 已有 sentinel 产物,
#                                                                 # 则整个工程跳过 (skipped-cached)
#                                                                 # 默认 0
#   DRY_RUN=1 bash run_all_experts.sh                             # 只打印计划
#
# 建议加 tee 落顶层日志:
#   mkdir -p _run_logs && bash run_all_experts.sh 2>&1 | tee _run_logs/run_all.$(date +%Y%m%d_%H%M).log
# =============================================================================

set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-$REPO_ROOT/envs}"
export REPO_ROOT ENV_ROOT

# -----------------------------------------------------------------------------
# 基础路径
# -----------------------------------------------------------------------------
EVAL_ROOT="$REPO_ROOT/evaluation"
MODELS_TSV_MASTER="$EVAL_ROOT/models.tsv"
LOG_ROOT="$EVAL_ROOT/_run_logs"
TMP_ROOT="$EVAL_ROOT/_run_logs/_tmp_tsv"
TS="$(date +%Y%m%d_%H%M%S)"

# Qwen3-VL 服务侧路径
VL_DEPLOY_DIR="$REPO_ROOT/Qwen3-VL-235B_prompt"
VL_LAUNCH_SH="$VL_DEPLOY_DIR/scripts/start_qwen3vl.sh"
VL_STATUS_SH="$VL_DEPLOY_DIR/scripts/status.sh"
VL_TMUX_SESSION="qwen3vl_serve"
VL_HEALTH_PORT=8080
VL_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"

# 服务等待就绪的超时 (秒)
VL_READY_TIMEOUT="${VL_READY_TIMEOUT:-900}"        # 15 min (FP8 权重首次加载约 5-10 min)

mkdir -p "$LOG_ROOT" "$TMP_ROOT"

# -----------------------------------------------------------------------------
# 颜色 & 日志
# -----------------------------------------------------------------------------
if [[ -t 1 ]]; then
    C_RED='\033[0;31m'; C_GRN='\033[0;32m'; C_YEL='\033[1;33m'
    C_BLU='\033[0;34m'; C_CYN='\033[0;36m'; C_MAG='\033[0;35m'; C_RST='\033[0m'
else
    C_RED=''; C_GRN=''; C_YEL=''; C_BLU=''; C_CYN=''; C_MAG=''; C_RST=''
fi

log()    { echo -e "${C_BLU}[run_all]${C_RST} $*"; }
ok()     { echo -e "${C_GRN}[run_all]${C_RST} $*"; }
warn()   { echo -e "${C_YEL}[run_all]${C_RST} $*"; }
err()    { echo -e "${C_RED}[run_all]${C_RST} $*" >&2; }
section(){
    echo
    echo "============================================================"
    echo -e "${C_CYN}  $*${C_RST}"
    echo "============================================================"
}

# -----------------------------------------------------------------------------
# 阶段分组: A/B/C
# 每个虚拟条目名字要唯一 (id_consistency 拆成两个)
# -----------------------------------------------------------------------------
PHASE_A_EXPERTS=(
    audio_consistency
    emtion_consistency
    id_consistency_transcribe
)

PHASE_B_EXPERTS=(
    action_eval
    style_consistency
    subject_consistency
    id_consistency_judge
)

PHASE_C_EXPERTS=(
    asr_eval
    syncnet_python
    TransNetV2
    audiobox-aesthetics
    facial_consistency
    VBench2-Human_Anatomy
    Q-Align
    consistency_segmented
    ocr
    Syncformer
)

ALL_EXPERTS=(
    "${PHASE_A_EXPERTS[@]}"
    "${PHASE_B_EXPERTS[@]}"
    "${PHASE_C_EXPERTS[@]}"
)

# 每个专家所属阶段 (供智能启停查询)
declare -A EXPERT_PHASE=()
for e in "${PHASE_A_EXPERTS[@]}"; do EXPERT_PHASE["$e"]="A"; done
for e in "${PHASE_B_EXPERTS[@]}"; do EXPERT_PHASE["$e"]="B"; done
for e in "${PHASE_C_EXPERTS[@]}"; do EXPERT_PHASE["$e"]="C"; done

# 用户友好的 alias: 传 EXPERTS=id_consistency 自动展开为两个虚拟条目
declare -A EXPERTS_ALIAS=(
    ["id_consistency"]="id_consistency_transcribe,id_consistency_judge"
)

_valid_expert() {
    local name="$1"
    local _e
    for _e in "${ALL_EXPERTS[@]}"; do
        [[ "$_e" == "$name" ]] && return 0
    done
    # 允许 alias
    [[ -n "${EXPERTS_ALIAS[$name]:-}" ]] && return 0
    return 1
}

# -----------------------------------------------------------------------------
# 参数 & 帮助
# -----------------------------------------------------------------------------
show_help() {
    sed -n '2,71p' "$0"
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        -h|--help)  show_help ;;
        *)  err "unknown argument: $arg"; err "run 'bash run_all_experts.sh --help' for usage"; exit 2 ;;
    esac
done

MODELS_FILTER="${MODELS:-}"
EXPERTS_FILTER="${EXPERTS:-}"
PHASES_FILTER="${PHASES:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SERVICE_MGMT="${SKIP_SERVICE_MGMT:-0}"
# SKIP_IF_DONE=1 时启用产物级跳过 (方案 B, opt-in):
#   对每个 expert, 检查其 sentinel 产物路径 (见下方 EXPERT_SENTINEL 表);
#   当 SELECTED_MODELS 里 *全部* 模型的 sentinel 都存在时, 跳过该工程,
#   RESULT_STATUS 标为 "skipped-cached". 只要有任一 model 缺失就照常跑.
#   默认 0 = 全量重跑.
SKIP_IF_DONE="${SKIP_IF_DONE:-0}"

# -----------------------------------------------------------------------------
# 前置检查 A: models.tsv 存在, 读全量模型清单
# -----------------------------------------------------------------------------
if [[ ! -f "$MODELS_TSV_MASTER" ]]; then
    err "models.tsv not found: $MODELS_TSV_MASTER"
    exit 2
fi

ALL_MODELS=()
while IFS=$'\t' read -r name src || [[ -n "$name" ]]; do
    [[ -z "$name" || "$name" =~ ^[[:space:]]*# ]] && continue
    ALL_MODELS+=("$name")
done < "$MODELS_TSV_MASTER"

if [[ ${#ALL_MODELS[@]} -eq 0 ]]; then
    err "models.tsv 里没有任何模型条目"; exit 2
fi

# -----------------------------------------------------------------------------
# 前置检查 B: 校验 MODELS 里的名字都合法
# -----------------------------------------------------------------------------
declare -A ALL_MODEL_SET=()
for m in "${ALL_MODELS[@]}"; do ALL_MODEL_SET["$m"]=1; done

SELECTED_MODELS=()
if [[ -n "$MODELS_FILTER" ]]; then
    IFS=',' read -r -a _tmp <<<"$MODELS_FILTER"
    for m in "${_tmp[@]}"; do
        m="$(echo "$m" | xargs)"; [[ -z "$m" ]] && continue
        if [[ -z "${ALL_MODEL_SET[$m]:-}" ]]; then
            err "MODELS 中的 '$m' 不在 $MODELS_TSV_MASTER 里"
            err "  合法模型: ${ALL_MODELS[*]}"
            exit 2
        fi
        SELECTED_MODELS+=("$m")
    done
    if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
        err "MODELS 过滤后为空"; exit 2
    fi
else
    SELECTED_MODELS=("${ALL_MODELS[@]}")
fi

SELECTED_MODELS_CSV="$(IFS=','; echo "${SELECTED_MODELS[*]}")"

# -----------------------------------------------------------------------------
# 前置检查 C: 校验 EXPERTS / PHASES 参数合法, 展开 alias, 生成 PLAN
# -----------------------------------------------------------------------------
declare -A PHASES_SET=()
if [[ -n "$PHASES_FILTER" ]]; then
    IFS=',' read -r -a _tmp <<<"$PHASES_FILTER"
    for p in "${_tmp[@]}"; do
        p="$(echo "$p" | xargs | tr 'a-z' 'A-Z')"
        [[ -z "$p" ]] && continue
        case "$p" in
            A|B|C|D)  PHASES_SET["$p"]=1 ;;
            *) err "PHASES 中的 '$p' 不合法, 允许: A / B / C / D"; exit 2 ;;
        esac
    done
else
    # 默认全阶段
    PHASES_SET[A]=1; PHASES_SET[B]=1; PHASES_SET[C]=1; PHASES_SET[D]=1
fi

# 展开 EXPERTS 里可能出现的 alias (如 id_consistency → transcribe+judge)
_expand_expert_alias() {
    local raw="$1"
    local val
    val="${EXPERTS_ALIAS[$raw]:-}"
    if [[ -n "$val" ]]; then
        printf '%s' "$val"
    else
        printf '%s' "$raw"
    fi
}

# 先构造用户想跑的 experts 集合 (若无 EXPERTS 则默认全部)
declare -A WANT_EXPERT=()
if [[ -n "$EXPERTS_FILTER" ]]; then
    IFS=',' read -r -a _tmp <<<"$EXPERTS_FILTER"
    for e in "${_tmp[@]}"; do
        e="$(echo "$e" | xargs)"; [[ -z "$e" ]] && continue
        if ! _valid_expert "$e"; then
            err "EXPERTS 中的 '$e' 不是合法专家名"
            err "  合法列表: ${ALL_EXPERTS[*]}  (或 alias: id_consistency)"
            exit 2
        fi
        expanded="$(_expand_expert_alias "$e")"
        IFS=',' read -r -a _sub <<<"$expanded"
        for x in "${_sub[@]}"; do
            WANT_EXPERT["$x"]=1
        done
    done
else
    for e in "${ALL_EXPERTS[@]}"; do WANT_EXPERT["$e"]=1; done
fi

# 按 PHASE + WANT_EXPERT 双条件过滤, 生成分阶段 PLAN
PLAN_A=(); PLAN_B=(); PLAN_C=()
for e in "${PHASE_A_EXPERTS[@]}"; do
    [[ -n "${WANT_EXPERT[$e]:-}" && -n "${PHASES_SET[A]:-}" ]] && PLAN_A+=("$e")
done
for e in "${PHASE_B_EXPERTS[@]}"; do
    [[ -n "${WANT_EXPERT[$e]:-}" && -n "${PHASES_SET[B]:-}" ]] && PLAN_B+=("$e")
done
for e in "${PHASE_C_EXPERTS[@]}"; do
    [[ -n "${WANT_EXPERT[$e]:-}" && -n "${PHASES_SET[C]:-}" ]] && PLAN_C+=("$e")
done

PLAN=("${PLAN_A[@]:-}" "${PLAN_B[@]:-}" "${PLAN_C[@]:-}")
# 清除数组里可能因空展开产生的空字符串
PLAN=("${PLAN[@]/#/}")
_tmp_plan=()
for e in "${PLAN[@]}"; do [[ -n "$e" ]] && _tmp_plan+=("$e"); done
PLAN=("${_tmp_plan[@]}")

if [[ ${#PLAN[@]} -eq 0 ]]; then
    err "PLAN 为空. 检查 EXPERTS / PHASES 参数."
    exit 2
fi

# -----------------------------------------------------------------------------
# 前置检查 D: consistency_segmented 依赖 audio_consistency/runs/qwen3_omni/
# -----------------------------------------------------------------------------
declare -A SKIP_REASON=()

_is_in_plan_c() {
    local _e
    for _e in "${PLAN_C[@]}"; do [[ "$_e" == "$1" ]] && return 0; done
    return 1
}

if _is_in_plan_c consistency_segmented; then
    CS_DEP_DIR="$EVAL_ROOT/audio_consistency/runs/qwen3_omni"
    _need_run_ac_now=0
    for e in "${PLAN_A[@]}"; do [[ "$e" == "audio_consistency" ]] && _need_run_ac_now=1; done
    if [[ "$_need_run_ac_now" -eq 1 ]]; then
        log "consistency_segmented 依赖 audio_consistency, 本次会先跑 (PLAN_A 里有 audio_consistency)"
    elif [[ ! -d "$CS_DEP_DIR" ]] || [[ -z "$(ls -A "$CS_DEP_DIR" 2>/dev/null)" ]]; then
        warn "consistency_segmented 依赖的 audio_consistency 产物缺失或为空:"
        warn "  $CS_DEP_DIR"
        warn "  → 从 PLAN 摘掉, 汇总时标 SKIPPED-DEPENDENCY-MISSING"
        warn "  → 建议: 先跑 audio_consistency, 或 PHASES=A 单独跑 A 阶段"
        SKIP_REASON["consistency_segmented"]="audio_consistency/runs/qwen3_omni/ missing (前置检查)"
        _tmp_plan_c=()
        for e in "${PLAN_C[@]}"; do
            [[ "$e" == "consistency_segmented" ]] && continue
            _tmp_plan_c+=("$e")
        done
        PLAN_C=("${_tmp_plan_c[@]:-}")
        _tmp_plan=()
        for e in "${PLAN[@]}"; do
            [[ "$e" == "consistency_segmented" ]] && continue
            _tmp_plan+=("$e")
        done
        PLAN=("${_tmp_plan[@]:-}")
    else
        log "consistency_segmented 依赖检查 OK: $CS_DEP_DIR ($(ls -1 "$CS_DEP_DIR" | wc -l) items)"
    fi
fi

# -----------------------------------------------------------------------------
# 生成 subset models.tsv (给 TransNetV2 / Syncformer 用)
# -----------------------------------------------------------------------------
SUBSET_TSV="$TMP_ROOT/models.subset.${TS}.tsv"
{
    echo "# ===== auto-generated by run_all_experts.sh at $TS ====="
    echo "# subset of $MODELS_TSV_MASTER for selected models: $SELECTED_MODELS_CSV"
    while IFS=$'\t' read -r name src || [[ -n "$name" ]]; do
        [[ -z "$name" || "$name" =~ ^[[:space:]]*# ]] && continue
        for want in "${SELECTED_MODELS[@]}"; do
            if [[ "$name" == "$want" ]]; then
                printf "%s\t%s\n" "$name" "$src"
                break
            fi
        done
    done < "$MODELS_TSV_MASTER"
} > "$SUBSET_TSV"

if [[ -n "$MODELS_FILTER" ]]; then
    USE_TSV="$SUBSET_TSV"
else
    USE_TSV="$MODELS_TSV_MASTER"
fi

# -----------------------------------------------------------------------------
# Qwen3-VL 服务生命周期函数 (手动启动方式见 t2av-bench/SETUP.md §3)
# ⚠️ 唯一允许后台运行的场景: 长驻 vLLM 服务进程, 用 tmux -d 管理
# -----------------------------------------------------------------------------

# 检查 tmux 可用
_svc_check_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        err "tmux 未安装. 请先运行: apt-get update && apt-get install -y tmux"
        return 1
    fi
    return 0
}

# 探测某端口 vLLM /v1/models 是否就绪
_svc_probe_port() {
    local port="$1"
    curl -s --connect-timeout 2 --max-time 5 \
        --noproxy '*' \
        "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q '"data"'
}

# --- VL 服务 ---
_svc_vl_is_running() {
    tmux has-session -t "$VL_TMUX_SESSION" 2>/dev/null
}

_svc_vl_start() {
    section "服务启动 [Qwen3-VL]  (tmux: $VL_TMUX_SESSION, port $VL_HEALTH_PORT)"

    _svc_check_tmux || return 1

    if _svc_vl_is_running; then
        warn "tmux 会话 $VL_TMUX_SESSION 已存在, 复用"
    else
        if [[ ! -f "$VL_LAUNCH_SH" ]]; then
            err "VL 启动脚本不存在: $VL_LAUNCH_SH"
            return 1
        fi
        log "调用 VL 启动脚本: $VL_LAUNCH_SH"
        log "  (脚本内部自建 tmux 会话 $VL_TMUX_SESSION, 前台一次调用即返回)"
        # start_qwen3vl.sh 自己会 tmux new -s qwen3vl_serve -d
        # 我们只需前台调用它, 它会立即返回
        ( cd "$VL_DEPLOY_DIR" && bash "$VL_LAUNCH_SH" ) || {
            err "start_qwen3vl.sh 返回非零. 查 tmux attach -t $VL_TMUX_SESSION 排查."
            return 1
        }
    fi

    # 健康检查
    local start_ts; start_ts="$(date +%s)"
    local timeout=$VL_READY_TIMEOUT
    log "等待 VL 就绪 (timeout=${timeout}s, 首次加载 FP8 权重 5-10 min)..."
    while true; do
        if _svc_probe_port "$VL_HEALTH_PORT"; then
            ok "Qwen3-VL 就绪 (elapsed=$(( $(date +%s) - start_ts ))s)"
            log "  ✓ http://127.0.0.1:${VL_HEALTH_PORT}/v1/models  ($VL_MODEL_NAME)"
            return 0
        fi
        local elapsed=$(( $(date +%s) - start_ts ))
        if [[ "$elapsed" -ge "$timeout" ]]; then
            err "Qwen3-VL 服务在 ${timeout}s 内未就绪. 查 tmux attach -t $VL_TMUX_SESSION 排查."
            return 1
        fi
        if (( elapsed % 60 < 5 )); then
            log "  等待中... (elapsed=${elapsed}s)"
        fi
        sleep 5
    done
}

_svc_vl_stop() {
    section "服务关闭 [Qwen3-VL]  (释放显存)"
    if _svc_vl_is_running; then
        log "kill tmux session: $VL_TMUX_SESSION"
        tmux kill-session -t "$VL_TMUX_SESSION" 2>/dev/null || true
    else
        log "tmux 会话 $VL_TMUX_SESSION 不存在, 跳过"
    fi
    log "兜底 pkill -f 'vllm serve.*Qwen3-VL'"
    pkill -9 -f 'vllm serve.*Qwen3-VL' 2>/dev/null || true
    sleep 10
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "nvidia-smi 显存快照:"
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/  /'
    fi
    ok "VL 服务已关闭"
}

# -----------------------------------------------------------------------------
# 每个工程的 run 命令
# -----------------------------------------------------------------------------
declare -A CMD=()

_llm_env_prefix() {
    if [[ -n "$MODELS_FILTER" ]]; then
        echo "ONLY_MODELS='$SELECTED_MODELS_CSV' "
    else
        echo ""
    fi
}
_LLM_ENV="$(_llm_env_prefix)"

# 通用 NO_PROXY 前缀, 保证 curl / openai client 能连本地 vLLM
_NO_PROXY_EXPORT="export NO_PROXY=localhost,127.0.0.1,::1 && export no_proxy=\$NO_PROXY && "

# ===== Phase A: Omni 本地推理窗口 (进程内 from_pretrained, 非 HTTP 服务) =====

CMD[audio_consistency]="${_NO_PROXY_EXPORT}cd '$EVAL_ROOT/audio_consistency' && ${_LLM_ENV}bash run_all.sh"

CMD[emtion_consistency]="${_NO_PROXY_EXPORT}cd '$EVAL_ROOT/emtion_consistency' && ${_LLM_ENV}SPLIT_MODALITY=1 bash run_all.sh"

# id_consistency 拆两段:
# 阶段 1 = transcribe (仅 Omni 转写, RUN_TRANSCRIBE=1 RUN_SCORE=0)
CMD[id_consistency_transcribe]="${_NO_PROXY_EXPORT}cd '$EVAL_ROOT/id_consistency' && RUN_TRANSCRIBE=1 RUN_SCORE=0 ${_LLM_ENV}bash run_all.sh"

# ===== Phase B: VL 窗口 =====

# action_eval: JUDGE_BACKEND=qwen_vl 是默认
CMD[action_eval]="${_NO_PROXY_EXPORT}cd '$EVAL_ROOT/action_eval' && JUDGE_BACKEND=qwen_vl ${_LLM_ENV}bash run_all.sh"

CMD[style_consistency]="${_NO_PROXY_EXPORT}cd '$EVAL_ROOT/style_consistency' && ${_LLM_ENV}bash run_all.sh"
CMD[subject_consistency]="${_NO_PROXY_EXPORT}cd '$EVAL_ROOT/subject_consistency' && ${_LLM_ENV}bash run_all.sh"

# id_consistency 阶段 2 = CPU 计分 (score_meeteval) + VL judge
#   - RUN_TRANSCRIBE=0 RUN_SCORE=1 bash run_all.sh  → cpwer / orcwer / delta_cp
#   - bash run_judge.sh                             → 用 VL judge 打分
# 两步顺序无关, 但都在 B 窗口 (VL 服务窗口) 里跑, 因为 judge_qwen.py 依赖 VL 服务
#
# 说明: run_judge.sh 只在 BASE_URL / LLM_MODEL 环境变量存在时才向 judge_qwen.py 透传
# --base-url / --model; 否则 judge_qwen.py 会依赖 t2av_bench.config, 而该 py 包
# 在当前 python 环境未安装, 会因 "缺 base_url" 直接 rc=2 秒退. 因此这里显式注入.
#
# 另外 judge_qwen.py 有两种模型工作集参数, 语义完全不同:
#   --models         : 覆盖式. 直接用作 models 列表, 顶层 judge_summary*.csv 会
#                       **整体覆盖**, 不在 --models 里的模型行会被清空.
#   --append-models  : 合并式. 只跑指定模型, 但顶层 judge_summary*.csv 用
#                       "已有行原位保留 + 本次追加/覆写" 方式合并, 不清空其它模型.
# 追加新模型时必须用 --append-models 以保留其它模型已有的分数; 再叠加
# --skip-existing 让 per-vid <M>/<vid>.json 已有缓存的直接命中不重调 LLM.
_JUDGE_ENV="BASE_URL='http://127.0.0.1:8080/v1' LLM_MODEL='Qwen3-VL-235B-A22B-Instruct' "
if [[ -n "$MODELS_FILTER" ]]; then
    _JUDGE_ARGS="--append-models '$SELECTED_MODELS_CSV' --skip-existing"
else
    _JUDGE_ARGS=""
fi
CMD[id_consistency_judge]="${_NO_PROXY_EXPORT}cd '$EVAL_ROOT/id_consistency' && RUN_TRANSCRIBE=0 RUN_SCORE=1 ${_LLM_ENV}bash run_all.sh && ${_JUDGE_ENV}bash run_judge.sh ${_JUDGE_ARGS}"

# ===== Phase C: 专家窗口 (无服务) =====

CMD[asr_eval]="cd '$EVAL_ROOT/asr_eval' && ${_LLM_ENV}bash run_all.sh"

CMD[syncnet_python]="cd '$EVAL_ROOT/syncnet_python' && ${_LLM_ENV}bash run_all.sh"

# TransNetV2: 保留 master TSV 作为完整清单 (供 Step2 汇总 compare 表用),
# 用 ONLY_MODELS 追加模式指定本次只跑哪些模型.
# 这样 compare_summary_multi.csv 的列会保留 tsv 里所有模型, 不会被覆盖.
CMD[TransNetV2]="cd '$EVAL_ROOT/TransNetV2' && MODELS_TSV='$MODELS_TSV_MASTER' ONLY_MODELS='$SELECTED_MODELS_CSV' bash run_all.sh"

if [[ -n "$MODELS_FILTER" ]]; then
    CMD[audiobox-aesthetics]="cd '$EVAL_ROOT/audiobox-aesthetics' && MODELS='$SELECTED_MODELS_CSV' bash run_all.sh"
    CMD[facial_consistency]="cd '$EVAL_ROOT/facial_consistency' && MODELS='$SELECTED_MODELS_CSV' bash run_all.sh"
    CMD[VBench2-Human_Anatomy]="cd '$EVAL_ROOT/VBench2-Human_Anatomy' && MODELS='$SELECTED_MODELS_CSV' bash run_all.sh"
else
    CMD[audiobox-aesthetics]="cd '$EVAL_ROOT/audiobox-aesthetics' && bash run_all.sh"
    CMD[facial_consistency]="cd '$EVAL_ROOT/facial_consistency' && bash run_all.sh"
    CMD[VBench2-Human_Anatomy]="cd '$EVAL_ROOT/VBench2-Human_Anatomy' && bash run_all.sh"
fi

QALIGN_VENV="$ENV_ROOT/qalign/bin/activate"
if [[ -n "$MODELS_FILTER" ]]; then
    CMD[Q-Align]="source '$QALIGN_VENV' && cd '$EVAL_ROOT/Q-Align' && MODELS='$SELECTED_MODELS_CSV' bash run_all.sh"
else
    CMD[Q-Align]="source '$QALIGN_VENV' && cd '$EVAL_ROOT/Q-Align' && bash run_all.sh"
fi

_CS_ARGS=(
    "--audio_runs_dir '$EVAL_ROOT/audio_consistency/runs/qwen3_omni'"
    "--sam2_ckpt      './pretrained/sam2.1_hiera_large.pt'"
    "--sam2_cfg       'configs/sam2.1/sam2.1_hiera_l.yaml'"
    "--subject_encoder dinov2"
    "--clip_model      'ViT-B/32'"
    "--device          cuda"
    "--max_frames      32"
    "--categories      speech,non_verbal_vocal"
    "--use_text_prompt"
    "--grounding_ckpt  './pretrained/grounding-dino-tiny'"
    "--specs_root      '$REPO_ROOT/t2av-bench/specs'"
    "--output_root     './runs'"
    # --append_summary: 合并模式. 读已有的 compare_summary_<kind>.csv, 保留其它模型的列/值,
    # 只把本次 --models 里的模型作为新列追加. 若不加, 与 --models 过滤组合时默认的全量覆盖
    # 会抹掉其它模型已有的分数.
    "--append_summary"
)
_CS_CMD_BASE="cd '$EVAL_ROOT/consistency_segmented' && mkdir -p runs && python3 batch_eval.py ${_CS_ARGS[*]}"
if [[ -n "$MODELS_FILTER" ]]; then
    CMD[consistency_segmented]="$_CS_CMD_BASE --models '$SELECTED_MODELS_CSV'"
else
    CMD[consistency_segmented]="$_CS_CMD_BASE"
fi

CMD[ocr]="cd '$EVAL_ROOT/ocr' && ONLY_MODELS='$SELECTED_MODELS_CSV' bash run_reference_cpu_full.sh"

# Syncformer: 在独立 venv (.venv_syncformer, transformers 4.40.*) 中运行
read -r -d '' _CMD_SYNCFORMER_RUN <<EOF || true
set -uo pipefail

echo "[Syncformer] 使用独立 venv (.venv_syncformer, transformers 4.40.*), 与 base 的 transformers 4.57.6 隔离."
echo "[Syncformer] run_all.sh 内部走 venv; 若 venv 缺失会回退 PATH 中的 python3 (会用到 base transformers)."
echo "[Syncformer] 如需(重)建 venv: cd '$EVAL_ROOT/Syncformer' && bash setup.sh"

echo "[Syncformer] 跑评测"
cd '$EVAL_ROOT/Syncformer'
# 同 TransNetV2: master TSV 作完整清单, ONLY_MODELS 指定本次实跑范围,
# 保证 compare_summary_<EXP>*.csv 不丢其它模型的列.
MODELS_TSV='$MODELS_TSV_MASTER' ONLY_MODELS='$SELECTED_MODELS_CSV' bash run_all.sh
EOF
CMD[Syncformer]="$_CMD_SYNCFORMER_RUN"

# -----------------------------------------------------------------------------
# SKIP_IF_DONE=1 时使用的 sentinel 表 (每个 expert 一条判定规则)
# -----------------------------------------------------------------------------
# 语义:
#   * 每条规则 = "对某个具体 model 判定是否已完成" 的谓词.
#   * 值前缀 "file:"  → 判定该文件是否存在 (非空判定用 -s).
#   * 值前缀 "dir:"   → 判定该目录是否存在且非空.
#   * 值里可用占位符 <MODEL>, 运行时替换为具体 model 名.
#   * 路径都是相对 $EVAL_ROOT 的绝对路径.
#
# 判定入口 _expert_all_done "$expert":
#   * SELECTED_MODELS 里 *全部* model 的 sentinel 都命中 → 返回 0 (跳过)
#   * 只要任一 miss → 返回 1 (照常跑)
#   * 无 sentinel 规则 → 保守返回 1 (照常跑)
declare -A EXPERT_SENTINEL=(
    [audio_consistency]="file:$EVAL_ROOT/audio_consistency/runs/qwen3_omni/<MODEL>/summary.json"
    [emtion_consistency]="file:$EVAL_ROOT/emtion_consistency/runs/qwen3_omni_split/<MODEL>/summary.json"
    [id_consistency_transcribe]="dir:$EVAL_ROOT/id_consistency/runs/qwen3_omni/<MODEL>/transcripts"

    [action_eval]="file:$EVAL_ROOT/action_eval/runs/qwen3_vl/<MODEL>/summary.json"
    [style_consistency]="file:$EVAL_ROOT/style_consistency/runs/qwen3_vl/<MODEL>/summary.json"
    [subject_consistency]="file:$EVAL_ROOT/subject_consistency/runs/qwen3_vl/<MODEL>/summary.json"
    [id_consistency_judge]="dir:$EVAL_ROOT/id_consistency/score_out/judge/<MODEL>"

    [asr_eval]="file:$EVAL_ROOT/asr_eval/runs/<MODEL>/wer_results.json"
    [syncnet_python]="file:$EVAL_ROOT/syncnet_python/runs/<MODEL>/result.csv"
    [TransNetV2]="file:$EVAL_ROOT/TransNetV2/runs/<MODEL>/summary_multi.csv"
    [audiobox-aesthetics]="file:$EVAL_ROOT/audiobox-aesthetics/runs/<MODEL>/summary.csv"
    [facial_consistency]="file:$EVAL_ROOT/facial_consistency/runs/<MODEL>/summary.json"
    [VBench2-Human_Anatomy]="file:$EVAL_ROOT/VBench2-Human_Anatomy/runs/<MODEL>/summary.csv"
    [Q-Align]="file:$EVAL_ROOT/Q-Align/runs/<MODEL>/summary.csv"
    [consistency_segmented]="dir:$EVAL_ROOT/consistency_segmented/runs/<MODEL>"
    [ocr]="dir:$EVAL_ROOT/ocr/runs_full_cpu/paddleocr/<MODEL>/tracks"
    [Syncformer]="file:$EVAL_ROOT/Syncformer/runs/<MODEL>/24-01-04T16-39-21/summary.csv"
)

_sentinel_hit() {
    # $1 = expert, $2 = model
    local rule="${EXPERT_SENTINEL[$1]:-}"
    [[ -z "$rule" ]] && return 1
    local path="${rule#*:}"
    path="${path//<MODEL>/$2}"
    case "$rule" in
        file:*)
            [[ -s "$path" ]] && return 0 || return 1 ;;
        dir:*)
            [[ -d "$path" ]] && [[ -n "$(ls -A "$path" 2>/dev/null)" ]] && return 0 || return 1 ;;
        *)
            return 1 ;;
    esac
}

_expert_all_done() {
    # 对 SELECTED_MODELS 里每个 model 判定 sentinel, 全部命中才返回 0
    local expert="$1"
    [[ -z "${EXPERT_SENTINEL[$expert]:-}" ]] && return 1
    local m
    for m in "${SELECTED_MODELS[@]}"; do
        if ! _sentinel_hit "$expert" "$m"; then
            return 1
        fi
    done
    return 0
}

# 预扫描: 在打印计划阶段, 提前算出哪些 expert 会被 cached-skip
# 存到 CACHED_SKIP 里, 供打印计划和主循环共用 (避免重复判定)
declare -A CACHED_SKIP=()
if [[ "$SKIP_IF_DONE" == "1" ]]; then
    for e in "${PLAN[@]}"; do
        if _expert_all_done "$e"; then
            CACHED_SKIP["$e"]=1
        fi
    done
fi

# -----------------------------------------------------------------------------
# 打印计划
# -----------------------------------------------------------------------------
section "RUN PLAN  (ts=$TS)"
log "EVAL_ROOT         = $EVAL_ROOT"
log "LOG_ROOT          = $LOG_ROOT"
log "MODELS_TSV_MASTER = $MODELS_TSV_MASTER"
log "python3           = $(command -v python3)"
if [[ -n "$MODELS_FILTER" ]]; then
    log "MODELS filter     = $MODELS_FILTER  (subset tsv: $SUBSET_TSV)"
else
    log "MODELS filter     = (empty, run all ${#ALL_MODELS[@]} models from tsv)"
fi
log "SELECTED_MODELS   = $SELECTED_MODELS_CSV"
log "USE_TSV           = $USE_TSV"
log "PHASES            = ${!PHASES_SET[*]}"
log "SKIP_SERVICE_MGMT = $SKIP_SERVICE_MGMT  (=1 时假设服务已就绪, orchestrator 不起/关服务)"
log "SKIP_IF_DONE      = $SKIP_IF_DONE  (=1 时若所有 selected model 的 sentinel 都命中则整个 expert 跳过)"
log "DRY_RUN           = $DRY_RUN"

echo
if [[ ${#PLAN_A[@]} -gt 0 ]]; then
    log "${C_MAG}Phase A (Omni 本地推理)${C_RST}  ${#PLAN_A[@]} 个工程:"
    i=0; for e in "${PLAN_A[@]}"; do i=$((i+1)); printf "   A%d. %s\n" "$i" "$e"; done
else
    log "${C_MAG}Phase A${C_RST}: (empty)"
fi
if [[ ${#PLAN_B[@]} -gt 0 ]]; then
    log "${C_MAG}Phase B (VL 窗口)${C_RST}    ${#PLAN_B[@]} 个工程:"
    i=0; for e in "${PLAN_B[@]}"; do i=$((i+1)); printf "   B%d. %s\n" "$i" "$e"; done
else
    log "${C_MAG}Phase B${C_RST}: (empty)"
fi
if [[ ${#PLAN_C[@]} -gt 0 ]]; then
    log "${C_MAG}Phase C (专家窗口)${C_RST}  ${#PLAN_C[@]} 个工程:"
    i=0; for e in "${PLAN_C[@]}"; do i=$((i+1)); printf "   C%d. %s\n" "$i" "$e"; done
else
    log "${C_MAG}Phase C${C_RST}: (empty)"
fi
if [[ -n "${PHASES_SET[D]:-}" ]]; then
    log "${C_MAG}Phase D (收尾)${C_RST}     18 项汇总"
fi

if [[ ${#SKIP_REASON[@]} -gt 0 ]]; then
    log "pre-skipped (dependency missing):"
    for k in "${!SKIP_REASON[@]}"; do
        printf "   ⏭  %s  (reason: %s)\n" "$k" "${SKIP_REASON[$k]}"
    done
fi

if [[ "$SKIP_IF_DONE" == "1" ]]; then
    if [[ ${#CACHED_SKIP[@]} -gt 0 ]]; then
        log "cached-skipped (SKIP_IF_DONE=1, sentinel hit for all selected models):"
        for k in "${!CACHED_SKIP[@]}"; do
            rule="${EXPERT_SENTINEL[$k]:-}"
            printf "   ⏭  %s  (sentinel: %s)\n" "$k" "$rule"
        done
    else
        log "cached-skipped: (empty) — no expert has full sentinel hit for all selected models"
    fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
    warn "DRY_RUN=1, printing commands and exit (not executing)."
    echo
    for e in "${PLAN[@]}"; do
        phase="${EXPERT_PHASE[$e]:-?}"
        if [[ -n "${CACHED_SKIP[$e]:-}" ]]; then
            echo -e "${C_YEL}[phase=$phase] [$e] (would be CACHED-SKIPPED)${C_RST}"
        else
            echo -e "${C_MAG}[phase=$phase] [$e]${C_RST}"
        fi
        echo "  ${CMD[$e]}"
    done
    exit 0
fi

# -----------------------------------------------------------------------------
# 主循环: 分阶段执行 (智能启停服务)
# -----------------------------------------------------------------------------
declare -A RESULT_STATUS=()
declare -A RESULT_ELAPSED=()
declare -A RESULT_LOGFILE=()
declare -A RESULT_PHASE=()

for k in "${!SKIP_REASON[@]}"; do
    RESULT_STATUS["$k"]="skipped"
    RESULT_ELAPSED["$k"]=0
    RESULT_LOGFILE["$k"]="-"
    RESULT_PHASE["$k"]="${EXPERT_PHASE[$k]:-?}"
done

TOTAL_START="$(date +%s)"

# 定义"跑一个工程"的公共函数
_run_one_expert() {
    local expert="$1"
    local phase="${EXPERT_PHASE[$expert]:-?}"
    RESULT_PHASE["$expert"]="$phase"

    # SKIP_IF_DONE=1 且预扫描判定该 expert 已完成 → 快速跳过, 不进 section
    if [[ "$SKIP_IF_DONE" == "1" && -n "${CACHED_SKIP[$expert]:-}" ]]; then
        RESULT_STATUS["$expert"]="skipped-cached"
        RESULT_ELAPSED["$expert"]=0
        RESULT_LOGFILE["$expert"]="-"
        local rule="${EXPERT_SENTINEL[$expert]:-}"
        log "${C_YEL}⏭  [phase=$phase] [$expert]${C_RST} cached-skip (SKIP_IF_DONE=1, sentinel: $rule)"
        return 0
    fi

    section "RUN  [phase=$phase] [$expert]  ($(date '+%H:%M:%S'))"
    local LOGFILE="$LOG_ROOT/${expert}.${TS}.log"
    RESULT_LOGFILE["$expert"]="$LOGFILE"

    if [[ -z "${CMD[$expert]:-}" ]]; then
        err "[$expert] 未定义 CMD, skip"
        RESULT_STATUS["$expert"]="fail"
        RESULT_ELAPSED["$expert"]=0
        return 1
    fi

    local start; start="$(date +%s)"
    log "logfile: $LOGFILE"
    log "cmd (first line): $(echo "${CMD[$expert]}" | head -1)"

    bash -c "${CMD[$expert]}" 2>&1 | tee "$LOGFILE"
    local ret="${PIPESTATUS[0]}"

    local end; end="$(date +%s)"
    local elapsed=$((end - start))
    RESULT_ELAPSED["$expert"]=$elapsed

    if [[ "$ret" -eq 0 ]]; then
        RESULT_STATUS["$expert"]="ok"
        ok "[$expert] SUCCESS in ${elapsed}s  (log: $LOGFILE)"
        return 0
    else
        RESULT_STATUS["$expert"]="fail"
        err "[$expert] FAILED  rc=$ret  elapsed=${elapsed}s  (log: $LOGFILE)"
        err "[$expert] 继续下一个 (continue-on-failure)."
        return 1
    fi
}

# --- Phase A ---
# 用到 Qwen3-Omni 的工程 (audio_consistency / emtion_consistency /
# id_consistency_transcribe) 都在自己的 Python 进程里 from_pretrained 加载 30B
# 权重 (每张卡 1 个 shard 进程), 不走 HTTP 服务, 因此无需 orchestrator 起 vLLM.
# 这里就是简单的串行: 一个工程跑完 (含其内部汇总), 再起下一个.
if [[ ${#PLAN_A[@]} -gt 0 ]]; then
    section "PHASE A — Qwen3-Omni 本地推理窗口  (${#PLAN_A[@]} 个工程, 串行)"
    for e in "${PLAN_A[@]}"; do
        _run_one_expert "$e" || true
    done
fi

# --- Phase B ---
if [[ ${#PLAN_B[@]} -gt 0 ]]; then
    section "PHASE B — Qwen3-VL 窗口  (${#PLAN_B[@]} 个工程)"

    # 若 SKIP_IF_DONE=1 且 PLAN_B 里每个工程都 cached-skip, 则跳过起 VL 服务
    _phase_b_all_cached=1
    if [[ "$SKIP_IF_DONE" == "1" ]]; then
        for e in "${PLAN_B[@]}"; do
            if [[ -z "${CACHED_SKIP[$e]:-}" ]]; then
                _phase_b_all_cached=0
                break
            fi
        done
    else
        _phase_b_all_cached=0
    fi

    if [[ "$_phase_b_all_cached" == "1" ]]; then
        warn "Phase B 全部工程已 cached-skip (SKIP_IF_DONE=1), 跳过 VL 服务启停以省 5-10 min 加载时间"
        for e in "${PLAN_B[@]}"; do
            _run_one_expert "$e" || true
        done
    elif [[ "$SKIP_SERVICE_MGMT" != "1" ]]; then
        if ! _svc_vl_start; then
            err "VL 服务启动失败, Phase B 全部标 fail"
            for e in "${PLAN_B[@]}"; do
                RESULT_STATUS["$e"]="fail"
                RESULT_ELAPSED["$e"]=0
                RESULT_LOGFILE["$e"]="-"
                RESULT_PHASE["$e"]="B"
            done
        else
            for e in "${PLAN_B[@]}"; do
                _run_one_expert "$e" || true
            done
            _svc_vl_stop
        fi
    else
        warn "SKIP_SERVICE_MGMT=1, 跳过 VL 服务启停 (假设用户已启动)"
        for e in "${PLAN_B[@]}"; do
            _run_one_expert "$e" || true
        done
    fi
fi

# --- Phase C ---
if [[ ${#PLAN_C[@]} -gt 0 ]]; then
    section "PHASE C — 专家窗口  (${#PLAN_C[@]} 个工程, 无服务)"
    for e in "${PLAN_C[@]}"; do
        _run_one_expert "$e" || true
    done
fi

# -----------------------------------------------------------------------------
# Phase D: 18 指标汇总
# -----------------------------------------------------------------------------
if [[ -n "${PHASES_SET[D]:-}" ]]; then
    section "PHASE D — AGGREGATE 18 METRICS  →  $EVAL_ROOT/runs_summary/"

    AGG_SCRIPT="$EVAL_ROOT/aggregate_18_metrics.py"
    AGG_LOG="$LOG_ROOT/aggregate_18metrics.${TS}.log"

    if [[ ! -f "$AGG_SCRIPT" ]]; then
        warn "汇总脚本不存在, 跳过: $AGG_SCRIPT"
    else
        _agg_cmd=(python3 "$AGG_SCRIPT" --output-dir "$EVAL_ROOT/runs_summary")
        if [[ -n "$MODELS_FILTER" ]]; then
            _agg_cmd+=(--models "$SELECTED_MODELS_CSV")
        fi
        log "cmd: ${_agg_cmd[*]}"
        log "logfile: $AGG_LOG"
        "${_agg_cmd[@]}" 2>&1 | tee "$AGG_LOG"
        _agg_ret="${PIPESTATUS[0]}"
        if [[ "$_agg_ret" -eq 0 ]]; then
            ok "18 项汇总已生成:"
            ok "  → $EVAL_ROOT/runs_summary/all_models_18metrics.csv         (合并表)"
            ok "  → $EVAL_ROOT/runs_summary/all_models_18metrics_single.csv  (single 档)"
            ok "  → $EVAL_ROOT/runs_summary/all_models_18metrics_multi.csv   (multi 档)"
            ok "  → $EVAL_ROOT/runs_summary/all_models_18metrics.md          (Markdown 版)"
            ok "  → $EVAL_ROOT/runs_summary/all_models_18metrics.json        (结构化)"
        else
            warn "18 项汇总 rc=$_agg_ret (各工程原产物不受影响, 汇总可稍后手动重跑)"
            warn "  手动重跑: python3 $AGG_SCRIPT [--models $SELECTED_MODELS_CSV]"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# 汇总
# -----------------------------------------------------------------------------
TOTAL_END="$(date +%s)"
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

section "RUN SUMMARY  (models = $SELECTED_MODELS_CSV)"
n_ok=0; n_fail=0; n_skip=0; n_cached=0

for expert in "${ALL_EXPERTS[@]}"; do
    st="${RESULT_STATUS[$expert]:-not-selected}"
    el="${RESULT_ELAPSED[$expert]:-0}"
    lg="${RESULT_LOGFILE[$expert]:--}"
    ph="${RESULT_PHASE[$expert]:-${EXPERT_PHASE[$expert]:-?}}"
    case "$st" in
        ok)
            printf "  ${C_GRN}✅  [%s] %-30s${C_RST}  %5ds   log=%s\n" "$ph" "$expert" "$el" "$lg"
            n_ok=$((n_ok+1)) ;;
        fail)
            printf "  ${C_RED}❌  [%s] %-30s${C_RST}  %5ds   log=%s\n" "$ph" "$expert" "$el" "$lg"
            n_fail=$((n_fail+1)) ;;
        skipped)
            reason="${SKIP_REASON[$expert]:-}"
            printf "  ${C_YEL}⏭   [%s] %-30s${C_RST}  (skipped: %s)\n" "$ph" "$expert" "$reason"
            n_skip=$((n_skip+1)) ;;
        skipped-cached)
            rule="${EXPERT_SENTINEL[$expert]:-}"
            printf "  ${C_YEL}⏭   [%s] %-30s${C_RST}  (cached: %s)\n" "$ph" "$expert" "$rule"
            n_cached=$((n_cached+1)) ;;
        not-selected)
            printf "  ${C_BLU}·   [%s] %-30s${C_RST}  (not selected)\n" "$ph" "$expert" ;;
    esac
done

echo
log "total: ${n_ok} ok, ${n_fail} fail, ${n_skip} skipped, ${n_cached} cached-skipped, out of ${#PLAN[@]} planned."
log "total elapsed: ${TOTAL_ELAPSED}s"
log "detail logs in: $LOG_ROOT/"

echo
log "查看各工程产出:"
for e in "${PLAN[@]}"; do
    case "$e" in
        audio_consistency)          echo "  audio_consistency         : $EVAL_ROOT/audio_consistency/runs/qwen3_omni/*/summary.json + compare_summary*.csv" ;;
        emtion_consistency)         echo "  emtion_consistency        : $EVAL_ROOT/emtion_consistency/runs/qwen3_omni_split/*/summary.json + compare_summary*.csv" ;;
        id_consistency_transcribe)  echo "  id_consistency::transcribe: $EVAL_ROOT/id_consistency/runs/qwen3_omni/<model>/transcripts/*.json" ;;
        action_eval)                echo "  action_eval               : $EVAL_ROOT/action_eval/runs/qwen3_vl/*/summary.json + compare_summary*.csv" ;;
        style_consistency)          echo "  style_consistency         : $EVAL_ROOT/style_consistency/runs/qwen3_vl/*/summary.json + compare_summary*.csv" ;;
        subject_consistency)        echo "  subject_consistency       : $EVAL_ROOT/subject_consistency/runs/qwen3_vl/*/summary.json + compare_summary*.csv" ;;
        id_consistency_judge)       echo "  id_consistency::judge     : $EVAL_ROOT/id_consistency/score_out/(compare_summary*.csv | judge/judge_summary*.csv)" ;;
        asr_eval)                   echo "  asr_eval                  : $EVAL_ROOT/asr_eval/runs/*/wer_results.json + compare_summary*.csv" ;;
        syncnet_python)             echo "  syncnet_python            : $EVAL_ROOT/syncnet_python/runs/*/result.csv + compare_summary*.csv" ;;
        TransNetV2)                 echo "  TransNetV2                : $EVAL_ROOT/TransNetV2/runs/*/summary_multi.csv + compare_summary_multi.csv" ;;
        audiobox-aesthetics)        echo "  audiobox-aesthetics       : $EVAL_ROOT/audiobox-aesthetics/runs/*/summary.csv + compare_summary_{CE,CU,PC,PQ}.csv" ;;
        facial_consistency)         echo "  facial_consistency        : $EVAL_ROOT/facial_consistency/runs/*/summary.json + compare_summary.csv" ;;
        VBench2-Human_Anatomy)      echo "  VBench2-Human_Anatomy     : $EVAL_ROOT/VBench2-Human_Anatomy/runs/*/summary.csv + compare_summary*.csv" ;;
        Q-Align)                    echo "  Q-Align                   : $EVAL_ROOT/Q-Align/runs/*/summary.csv + compare_summary*.csv" ;;
        consistency_segmented)      echo "  consistency_segmented     : $EVAL_ROOT/consistency_segmented/runs/*/  + compare_summary*.csv" ;;
        ocr)                        echo "  ocr                       : $EVAL_ROOT/ocr/runs_full_cpu/paddleocr/*/tracks/ + score_out/reference_full_cpu/compare_summary*.csv" ;;
        Syncformer)                 echo "  Syncformer                : $EVAL_ROOT/Syncformer/runs/*/24-01-04T16-39-21/summary.csv + compare_summary*.csv" ;;
    esac
done

echo
log "18 项跨工程分数汇总 (Phase D):"
log "  → $EVAL_ROOT/runs_summary/all_models_18metrics.csv         (合并表: single+multi)"
log "  → $EVAL_ROOT/runs_summary/all_models_18metrics_single.csv  (只 single 档)"
log "  → $EVAL_ROOT/runs_summary/all_models_18metrics_multi.csv   (只 multi 档)"
log "  → $EVAL_ROOT/runs_summary/all_models_18metrics.md          (Markdown 表格)"
log "  → $EVAL_ROOT/runs_summary/all_models_18metrics.json        (结构化)"

if [[ "$n_fail" -gt 0 ]]; then
    warn "有失败项. 请查看对应 log 文件, 手动排查后单独重跑:"
    warn "  EXPERTS=<expert-name> [MODELS=...] bash run_all_experts.sh"
    warn "  可选: SKIP_SERVICE_MGMT=1 (若你自己已管好服务)"
    exit 1
fi
exit 0
