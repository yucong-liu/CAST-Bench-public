#!/usr/bin/env bash
# =====================================================================
# syncnet_python 一键多模型评测  (lip <-> speech sync, syncnet 任务)
#   筛选规则: category == "speech" AND audio_situation != "off_screen"
#   = 每生成模型 8 条 speech_*.mp4 (speech_0004 / speech_0008 因 off_screen 排除)
#
# 输出:
#   runs/<MODEL>/_filtered_input/speech/<id>.mp4    # 评测前建的 symlink 子集
#   runs/<MODEL>/result.csv                          # 类目级 + TOTAL
# 已存在 result.csv 的模型自动跳过, 重跑请先 rm.
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# 用独立 venv (scenedetect 0.5.6.1 旧 API), 避免与 base 的新版 scenedetect (VBench2 用) 冲突.
# 可用 SYNCNET_PY 覆盖. venv 缺失时回退 PATH 中的 python3 (但 scenedetect 会 import 失败).
HERE=$REPO_ROOT/evaluation/syncnet_python
PY="${SYNCNET_PY:-$HERE/.venv_syncnet/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
ROOT=$REPO_ROOT
cd "$HERE"

WORKERS="${SYNCNET_WORKERS:-1}"
CONF_TH="${SYNCNET_CONF_TH:-1.0}"
SPECS_DIR="${SPECS_DIR:-$REPO_ROOT/t2av-bench/specs}"

# ---- 模型清单从共享 TSV 读入：新增模型只需在 models.tsv 加一行 ----
MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"
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
src_root_for() { echo "${SRC_MAP[$1]:-}"; }
load_models

# ---- 可选: ONLY_MODELS=<name>[,<name>...] 只跑指定模型 (追加模式) ----
#   - 默认不设 => 走全量流程 (行为一字不变)
#   - 设了则:
#       * for loop 只处理这些模型 (filter+eval 只对它们跑)
#       * 最后 aggregate 自动加 --append_models "$ONLY_MODELS",
#         保留旧的 compare_summary*.csv 里其他模型的行不变
#   - 拼错的名字 (models.tsv 里没有) 直接报错退出
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

for m in "${MODELS[@]}"; do
    src=$(src_root_for "$m")
    out_csv="runs/$m/result.csv"
    work_dir="data/work_$m"
    filt_dir="runs/$m/_filtered_input"

    if [[ -z "$src" || ! -d "$src" ]]; then
        echo "[skip] $m: src not found ($src)"; continue
    fi

    mkdir -p "runs/$m" "$work_dir"
    echo "==[filter] $m  task=syncnet (speech only, on-screen)=="
    $PY filter_specs.py --task syncnet --src "$src" --dst "$filt_dir"

    if [[ -z "$(find "$filt_dir" -name '*.mp4' -print -quit 2>/dev/null)" ]]; then
        echo "[skip] $m: 筛选后无可评测视频"; continue
    fi

    echo "==[run] $m  src=$filt_dir  conf_th=$CONF_TH  workers=$WORKERS=="
    SYNCNET_WORKERS="$WORKERS" $PY batch_eval.py \
        --video_root "$filt_dir" \
        --save_csv   "$out_csv" \
        --specs_dir  "$SPECS_DIR" \
        --inference_py inference.py \
        --data_dir   "$work_dir" \
        --conf_th    "$CONF_TH"
done

echo
echo "==[ALL DONE] 结果在 runs/<MODEL>/result.csv=="
ls -1 runs/*/result.csv 2>/dev/null || true

echo
echo "==[aggregate] 生成跨模型总表 runs/compare_summary*.csv=="
# 全样本 / single / multi 三份汇总. 开源模型不生成多镜头视频, 因此其
# result_multi.csv 可能不存在或全空, aggregate_summary.py 会自动跳过 / 留空.
#
# 追加模式 (ONLY_MODELS 非空): 保留 3 张 CSV 里其他模型的行不变, 只
# 重算/追加指定模型的行.
if [[ -n "$ONLY_MODELS" ]]; then
    $PY aggregate_summary.py --runs_dir runs --out_csv runs/compare_summary.csv \
        --result_basename result.csv        --append_models "$ONLY_MODELS" || true
    $PY aggregate_summary.py --runs_dir runs --out_csv runs/compare_summary_single.csv \
        --result_basename result_single.csv --append_models "$ONLY_MODELS" || true
    $PY aggregate_summary.py --runs_dir runs --out_csv runs/compare_summary_multi.csv \
        --result_basename result_multi.csv  --append_models "$ONLY_MODELS" || true
else
    $PY aggregate_summary.py --runs_dir runs --out_csv runs/compare_summary.csv \
        --result_basename result.csv || true
    $PY aggregate_summary.py --runs_dir runs --out_csv runs/compare_summary_single.csv \
        --result_basename result_single.csv || true
    $PY aggregate_summary.py --runs_dir runs --out_csv runs/compare_summary_multi.csv \
        --result_basename result_multi.csv  || true
fi
