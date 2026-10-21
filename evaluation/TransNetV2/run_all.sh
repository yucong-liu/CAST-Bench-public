#!/usr/bin/env bash
# =====================================================================
# TransNetV2 镜头数批量评测脚本 (PyTorch 版) — *仅多镜头*
# 输出布局:
#   每个商业模型一份 results_multi.csv + summary_multi.csv,
#   顶层再生成 compare_summary_multi.csv (多模型 mean_score 对比).
#
# 真值: t2av-bench/specs/<category>.jsonl 的 shot_count / shots
#       *只评 shot_count == "multi_shot" 的样本; single_shot 直接跳过*
# 公式: score = max(0, 1 - |A - E| / E)
# 模型: 开源三家 (LTX-2.3 / Ovi / daVinci-MagiHuman) 不生成多镜头视频, 自动跳过
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# ---------------- 用户可调参数 ----------------
TNV2_ROOT=$REPO_ROOT/evaluation/TransNetV2
RUN_DIR="$TNV2_ROOT/runs"
SPECS_DIR=$REPO_ROOT/t2av-bench/specs
# 留空 = 用 transnetv2-pytorch pip 包内置权重
WEIGHTS=""
THRESHOLD=0.5

# multi 模式下默认跳过的模型（开源三家不生成多镜头）
# 可以用 SKIP_MODELS="" 强制全跑 (一般不需要)
SKIP_MODELS="${SKIP_MODELS-LTX-2.3 Ovi daVinci-MagiHuman}"

# 解释器: 允许用 PY 环境变量覆盖, 否则用 PATH 中的 python3 (base 环境)
if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
    PY=$(command -v python3 || true)
fi
if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
    echo "[ERROR] no usable python found, set env PY=/path/to/python and rerun"
    exit 1
fi
echo "[INFO] using python: $PY ($($PY -V 2>&1))"
echo "[INFO] mode: multi-only (single_shot samples are skipped)"

# 想跑哪些模型就在这里列; 新增模型记得同步在 src_root_for() 里加 case
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

# ---- 保留完整模型清单副本, 供 Step 2 汇总时保持 CSV 列不丢失 ----
ALL_MODELS=("${MODELS[@]}")

# ---- ONLY_MODELS 追加模式: 只跑白名单里的模型 (逗号或空格分隔) ----
# 用法:
#   ONLY_MODELS="kling-v3-omni,veo3.1,happyhorse-1.1" bash run_all.sh
# 语义:
#   - Step 1 只对这些模型跑 TransNetV2 推理;
#   - Step 2 依然汇总 models.tsv 里所有模型 (其它模型读各自已存在的 summary_multi.csv,
#     本次模型读刚跑完的 summary_multi.csv), 保证 compare_summary_multi.csv 列完整不丢失;
#   - SKIP_MODELS 依然生效, 白名单会先减去黑名单.
#   - 拼错模型名 (models.tsv 中不存在) 会报错退出, 不静默跳过.
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

mkdir -p "$RUN_DIR"
cd "$TNV2_ROOT"

# 辅助: 判断某个模型是否在跳过名单中
should_skip_model() {
    local m="$1"
    for skip in $SKIP_MODELS; do
        [[ "$m" == "$skip" ]] && return 0
    done
    return 1
}

# ============== Step 1: 每个商业模型跑 TransNetV2 (仅 multi_shot) ==============
for MODEL in "${MODELS[@]}"; do
    if should_skip_model "$MODEL"; then
        echo "[INFO] 跳过开源模型: $MODEL (不生成多镜头视频)"
        continue
    fi
    SRC=$(src_root_for "$MODEL")
    if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
        echo "[ERROR] model=$MODEL src=$SRC not found, skip"
        continue
    fi
    OUT_DIR="$RUN_DIR/$MODEL"
    mkdir -p "$OUT_DIR"

    echo "============================================================"
    echo "[INFO] running model=$MODEL  src=$SRC  out=$OUT_DIR  (multi_shot only)"
    echo "[INFO]   results -> $OUT_DIR/results_multi.csv"
    echo "[INFO]   summary -> $OUT_DIR/summary_multi.csv"
    echo "============================================================"

    "$PY" "$TNV2_ROOT/batch_eval.py" \
        --root "$SRC" \
        --specs_dir "$SPECS_DIR" \
        --weights "$WEIGHTS" \
        --threshold "$THRESHOLD" \
        --shot_count_kind multi \
        --save_csv "$OUT_DIR/results_multi.csv" \
        --save_summary_csv "$OUT_DIR/summary_multi.csv"
done

# ============== Step 2: 跨模型对比, 仅生成 compare_summary_multi.csv ==============
# 跳过名单中的模型 (开源三家) 不读它们可能存在的 summary, 在汇总表中列空.
# 注意: 这里用 ALL_MODELS (完整清单) 而非 MODELS (可能被 ONLY_MODELS 过滤过的),
# 以保证 compare_summary_multi.csv 的列不会因为追加模式而丢失其它模型的数据.
export SKIP_MODELS_FOR_PY="$SKIP_MODELS"
"$PY" - <<PY
import os, csv
from collections import defaultdict

run_dir = "$RUN_DIR"
models = "${ALL_MODELS[*]}".split()
skip_models = set(os.environ.get("SKIP_MODELS_FOR_PY", "").split())

out_name = "compare_summary_multi.csv"
sum_name = "summary_multi.csv"

data = defaultdict(dict)
n_map = defaultdict(dict)
all_categories = set()

for m in models:
    if m in skip_models:
        continue
    sum_csv = os.path.join(run_dir, m, sum_name)
    if not os.path.isfile(sum_csv):
        print(f"[WARN] missing {sum_csv}, skip model {m}")
        continue
    with open(sum_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            cat = row["folder"]
            try:
                mean_s = float(row["mean_score"])
            except (ValueError, TypeError):
                mean_s = None
            try:
                n_matched = int(row["n_matched"])
            except (ValueError, TypeError):
                n_matched = ""
            all_categories.add(cat)
            if mean_s is not None:
                data[cat][m] = mean_s
            n_map[cat][m] = n_matched

cats_sorted = sorted([c for c in all_categories if c != "__ALL__"])
if "__ALL__" in all_categories:
    cats_sorted.append("__ALL__")

out_csv = os.path.join(run_dir, out_name)
header = ["category"] + models + [f"n_{m}" for m in models]
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for cat in cats_sorted:
        row = [cat]
        for m in models:
            v = data[cat].get(m, "")
            row.append(f"{v:.4f}" if isinstance(v, float) else "")
        for m in models:
            row.append(n_map[cat].get(m, ""))
        w.writerow(row)
print(f"[write] {out_csv}")
PY

echo
echo "============================================================"
echo "[DONE] all commercial models finished. outputs under: $RUN_DIR"
echo "       summary -> $RUN_DIR/compare_summary_multi.csv"
echo "============================================================"
