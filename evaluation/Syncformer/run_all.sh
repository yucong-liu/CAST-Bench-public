#!/usr/bin/env bash
# =====================================================================
# Syncformer 一键多模型评测  (通用音画同步, syncformer 任务)
#   筛选规则: category NOT IN {ambient, bgm} AND audio_situation != "off_screen"
#   = 每生成模型 56 条 (7 类: speech / non_verbal_vocal / object / action /
#                                animal / instrument / sfx, 每类 8 条)
#
# 输出:
#   runs/<MODEL>/<EXP>/_filtered_input/<class>/<id>.mp4   # 评测前建的 symlink 子集
#   runs/<MODEL>/<EXP>/results.csv     # 每条视频
#   runs/<MODEL>/<EXP>/summary.csv     # 类目级 + __ALL__
# 已存在 results.csv 的模型自动跳过, 重跑请先 rm.
#
# 可换模型权重:
#   SYNCF_EXP=23-12-23T18-33-57 bash run_all.sh   # LRS3
#   SYNCF_EXP=24-01-22T20-34-52 bash run_all.sh   # synchability
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# 用独立 venv (transformers 4.40.*), 避免污染 base 的 transformers 4.57.6.
# 可用 SYNCF_PY 覆盖. venv 缺失时回退 PATH 中的 python3 (但会用到 base 的 transformers).
HERE=$REPO_ROOT/evaluation/Syncformer
PY="${SYNCF_PY:-$HERE/.venv_syncformer/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
ROOT=$REPO_ROOT
cd "$HERE"

EXP="${SYNCF_EXP:-24-01-04T16-39-21}"
DEVICE="${SYNCF_DEVICE:-cuda:0}"
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

# ---- 保留完整模型清单副本, 供最后的跨模型汇总保持 CSV 列不丢失 ----
ALL_MODELS=("${MODELS[@]}")

# ---- ONLY_MODELS 追加模式: 只跑白名单里的模型 (逗号或空格分隔) ----
# 用法:
#   ONLY_MODELS="seedance-2.5,MiniMax-H3,LTX-2.5" bash run_all.sh
# 语义:
#   - 只对这些模型跑 Syncformer 推理;
#   - 已存在 results.csv 的模型依旧自动 skip (上面循环里的行为);
#   - 末尾跨模型汇总仍然遍历 ALL_MODELS (老模型读各自已存在的 summary.csv,
#     新模型读刚跑完的), 保证 compare_summary_<EXP>*.csv 列完整不丢失;
#   - 拼错模型名会报错退出, 不静默跳过.
ONLY_MODELS="${ONLY_MODELS:-}"
if [[ -n "$ONLY_MODELS" ]]; then
    ONLY_MODELS_NORM="${ONLY_MODELS//,/ }"
    declare -A _KNOWN_MODELS=()
    for m in "${ALL_MODELS[@]}"; do _KNOWN_MODELS["$m"]=1; done
    FILTERED=()
    for m in $ONLY_MODELS_NORM; do
        [[ -z "$m" ]] && continue
        if [[ -z "${_KNOWN_MODELS[$m]:-}" ]]; then
            echo "[ERROR] ONLY_MODELS 中的模型不在 models.tsv: $m" >&2
            echo "[ERROR] models.tsv 已注册: ${ALL_MODELS[*]}" >&2
            exit 1
        fi
        FILTERED+=("$m")
    done
    if [[ ${#FILTERED[@]} -eq 0 ]]; then
        echo "[ERROR] ONLY_MODELS 过滤后为空" >&2
        exit 1
    fi
    MODELS=("${FILTERED[@]}")
    echo "[run_all] ONLY_MODELS 过滤后 MODELS = ${MODELS[*]} (追加模式)"
fi
# ----------------------------------------------

for m in "${MODELS[@]}"; do
    src=$(src_root_for "$m")
    out_dir="runs/$m/$EXP"
    out_csv="$out_dir/results.csv"
    sum_csv="$out_dir/summary.csv"
    filt_dir="$out_dir/_filtered_input"

    if [[ -z "$src" || ! -d "$src" ]]; then
        echo "[skip] $m: src not found ($src)"; continue
    fi

    # 已经跑过 (results.csv 非空) 则跳过, 保留旧结果参与末尾的跨模型汇总;
    # 想强制重跑某模型: rm -rf runs/<MODEL>/$EXP
    if [[ -s "$out_csv" ]]; then
        echo "[skip] $m: results.csv already exists ($out_csv), 保留旧结果参与汇总"
        continue
    fi

    mkdir -p "$out_dir"
    echo "==[filter] $m  task=syncformer (no ambient / no bgm, on-screen only)=="
    $PY filter_specs.py --task syncformer --src "$src" --dst "$filt_dir"

    # 没筛到任何视频就跳过
    if [[ -z "$(find "$filt_dir" -name '*.mp4' -print -quit 2>/dev/null)" ]]; then
        echo "[skip] $m: 筛选后无可评测视频"; continue
    fi

    echo "==[run] $m  src=$filt_dir  exp=$EXP=="
    $PY batch_eval.py \
        --root "$filt_dir" \
        --exp_name "$EXP" \
        --device "$DEVICE" \
        --save_csv "$out_csv" \
        --save_summary_csv "$sum_csv" \
        --specs_dir "$SPECS_DIR"
done

echo
echo "==[ALL DONE per-model]=="
ls -1 runs/*/$EXP/summary.csv 2>/dev/null || true

# ============== 跨模型汇总 (cross-model comparison) ==============
# 输出三张表 (列结构相同, 区别仅在数据来源):
#   compare_summary_<EXP>.csv         <- 由各模型 summary.csv         (全样本)
#   compare_summary_<EXP>_single.csv  <- 由各模型 summary_single.csv  (single_shot 子集)
#   compare_summary_<EXP>_multi.csv   <- 由各模型 summary_multi.csv   (multi_shot  子集)
# 开源模型 (LTX-2.3 / Ovi / daVinci-MagiHuman) 不生成多镜头视频, 因此
# compare_summary_<EXP>_multi.csv 中它们的列会留空, 这是预期行为.
RUN_DIR=$HERE/runs
$PY - <<PY
import os, csv
from collections import defaultdict

run_dir = "$RUN_DIR"
exp     = "$EXP"
# 用 ALL_MODELS (完整清单) 而非 MODELS (可能被 ONLY_MODELS 过滤过的),
# 以保证 compare_summary_<EXP>*.csv 的列不会因追加模式而丢失老模型的数据.
models  = "${ALL_MODELS[*]}".split()

VARIANTS = [
    (f"compare_summary_{exp}.csv",        "summary.csv"),
    (f"compare_summary_{exp}_single.csv", "summary_single.csv"),
    (f"compare_summary_{exp}_multi.csv",  "summary_multi.csv"),
]

for out_name, sum_name in VARIANTS:
    data_cont   = defaultdict(dict)
    data_argmax = defaultdict(dict)
    n_map       = defaultdict(dict)
    all_cats    = set()

    for m in models:
        sum_csv = os.path.join(run_dir, m, exp, sum_name)
        if not os.path.isfile(sum_csv):
            print(f"[WARN] missing {sum_csv}, skip model {m}")
            continue
        with open(sum_csv) as f:
            r = csv.DictReader(f)
            for row in r:
                cat = (row.get("folder") or "").strip()
                if not cat:
                    continue
                try:
                    cont   = float(row["mean_abs_offset_cont_sec"])
                    argmax = float(row["mean_abs_offset_argmax_sec"])
                except (ValueError, TypeError, KeyError):
                    cont = argmax = None
                try:
                    n_val = int(row.get("n", ""))
                except (ValueError, TypeError):
                    n_val = ""
                all_cats.add(cat)
                if cont is not None:
                    data_cont[cat][m]   = cont
                    data_argmax[cat][m] = argmax
                n_map[cat][m] = n_val

    if not all_cats:
        print(f"[WARN] no data for {sum_name}, skip {out_name}")
        continue

    cats_sorted = sorted([c for c in all_cats if c != "__ALL__"])
    if "__ALL__" in all_cats:
        cats_sorted.append("__ALL__")

    out_csv = os.path.join(run_dir, out_name)
    header  = (["category"]
               + [f"{m}_cont"   for m in models]
               + [f"{m}_argmax" for m in models]
               + [f"n_{m}"      for m in models])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cat in cats_sorted:
            row = [cat]
            for m in models:
                v = data_cont[cat].get(m, "")
                row.append(f"{v:.6f}" if isinstance(v, float) else "")
            for m in models:
                v = data_argmax[cat].get(m, "")
                row.append(f"{v:.6f}" if isinstance(v, float) else "")
            for m in models:
                row.append(n_map[cat].get(m, ""))
            w.writerow(row)
    print(f"[write] {out_csv}")

    # 终端预览 (只看主指标 cont, 仅打印 全样本表 简明)
    if sum_name == "summary.csv":
        print()
        print(f"====== compare_summary preview ({sum_name}, mean_abs_offset_cont_sec, lower=better) ======")
        head = ["category".ljust(18)] + [m.rjust(14) for m in models]
        print(" | ".join(head))
        print("-" * len(" | ".join(head)))
        for cat in cats_sorted:
            cells = [cat.ljust(18)]
            for m in models:
                v = data_cont[cat].get(m, None)
                cells.append((f"{v:.4f}" if isinstance(v, float) else "-").rjust(14))
            print(" | ".join(cells))
PY

echo
echo "==[ALL DONE]=="
