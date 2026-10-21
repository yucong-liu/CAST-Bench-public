#!/usr/bin/env bash
# =====================================================================
# regen_compare.sh
#   ⚠️ 仅重新聚合 runs/compare_summary_multi.csv, 不重跑任何推理.
#
# 用途:
#   当 compare_summary_multi.csv 不完整 (例如只含部分模型的列)
#   时, 用它重新从 runs/<MODEL>/summary_multi.csv 聚合出完整的 compare 表.
#
# 前提:
#   runs/<MODEL>/summary_multi.csv 必须已经存在 (即已跑过推理).
#   缺失 summary_multi.csv 的模型会打印 [WARN], 在 compare 表里列空.
#
# 用法:
#   bash regen_compare.sh
#     从 models.tsv 读全部模型清单, 聚合每个模型的 summary_multi.csv,
#     覆盖写 runs/compare_summary_multi.csv.
#
#   MODELS_TSV=/path/to/models.tsv bash regen_compare.sh
#     指定自定义 tsv.
#
#   SKIP_MODELS="LTX-2.3 Ovi daVinci-MagiHuman" bash regen_compare.sh
#     指定跳过列表 (默认与 run_all.sh 一致: 开源三家不生成多镜头视频).
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

TNV2_ROOT=$REPO_ROOT/evaluation/TransNetV2
RUN_DIR="$TNV2_ROOT/runs"

MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"
SKIP_MODELS="${SKIP_MODELS-LTX-2.3 Ovi daVinci-MagiHuman}"

if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
    PY=$(command -v python3 || true)
fi
if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
    echo "[ERROR] no usable python found, set env PY=/path/to/python and rerun"
    exit 1
fi
echo "[INFO] using python: $PY"
echo "[INFO] MODELS_TSV: $MODELS_TSV"
echo "[INFO] SKIP_MODELS: $SKIP_MODELS"

# 读全量模型清单
if [[ ! -f "$MODELS_TSV" ]]; then
    echo "[ERROR] models.tsv not found: $MODELS_TSV" >&2; exit 1
fi
ALL_MODELS=()
while IFS=$'\t' read -r name src || [[ -n "$name" ]]; do
    [[ -n "${src:-}" && "$src" != /* ]] && src="$REPO_ROOT/$src"
    [[ -z "$name" || "$name" =~ ^[[:space:]]*# ]] && continue
    ALL_MODELS+=("$name")
done < "$MODELS_TSV"
echo "[INFO] ALL_MODELS (${#ALL_MODELS[@]}): ${ALL_MODELS[*]}"

# 备份旧 compare_summary_multi.csv (以防万一)
OUT_CSV="$RUN_DIR/compare_summary_multi.csv"
if [[ -f "$OUT_CSV" ]]; then
    BACKUP="$OUT_CSV.bak.$(date +%Y%m%d_%H%M%S)"
    cp "$OUT_CSV" "$BACKUP"
    echo "[INFO] backed up existing compare table -> $BACKUP"
fi

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
        print(f"[INFO] skip (in SKIP_MODELS): {m}")
        continue
    sum_csv = os.path.join(run_dir, m, sum_name)
    if not os.path.isfile(sum_csv):
        print(f"[WARN] missing {sum_csv}, column for {m} will be empty")
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
print(f"[write] header cols = {len(header)}; models cols = {len(models)}")
PY

echo
echo "============================================================"
echo "[DONE] compare table regenerated: $OUT_CSV"
echo "[TIP]  head -1 shows all columns:"
head -1 "$OUT_CSV"
echo "============================================================"
