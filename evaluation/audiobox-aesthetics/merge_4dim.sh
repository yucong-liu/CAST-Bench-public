#!/usr/bin/env bash
# =====================================================================
# 仅重新生成 compare_summary_4dim{,_single,_multi}.csv
# 输入: runs/compare_summary_{CE,CU,PC,PQ}{,_single,_multi}.csv (已存在)
# 输出: runs/compare_summary_4dim{,_single,_multi}.csv
# 列结构: category | CE_<m1> ... CE_<mK> | CU_... | PC_... | PQ_... | n_<m1> ... n_<mK>
#         (按指标分组: 每个指标连续 K 列对应 K 个模型)
# 用途 : 只想刷新合并表, 不想重跑模型时执行
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

RUN_DIR="${RUN_DIR:-$REPO_ROOT/evaluation/audiobox-aesthetics/runs}"

RUN_DIR="$RUN_DIR" python3 - <<'PY'
import os, csv

run_dir = os.environ["RUN_DIR"]
DIMS = ["CE", "CU", "PC", "PQ"]
KINDS = ["", "_single", "_multi"]

def read_compare(path):
    with open(path) as f:
        r = csv.reader(f); rows = list(r)
    if not rows:
        return [], [], {}, {}
    header = rows[0]
    n_models = (len(header) - 1) // 2
    models = header[1:1 + n_models]
    rows_by_cat, n_by_cat = {}, {}
    for row in rows[1:]:
        cat = row[0]
        rows_by_cat[cat] = {m: row[1 + i] for i, m in enumerate(models)}
        n_by_cat[cat]    = {m: row[1 + n_models + i] for i, m in enumerate(models)}
    return header, models, rows_by_cat, n_by_cat

for suffix in KINDS:
    per_dim, models, cats = {}, None, []
    for d in DIMS:
        path = os.path.join(run_dir, f"compare_summary_{d}{suffix}.csv")
        if not os.path.isfile(path):
            continue
        _, ms, rows_by_cat, n_by_cat = read_compare(path)
        per_dim[d] = (rows_by_cat, n_by_cat)
        if models is None:
            models = ms
        if not cats:
            cats = [c for c in rows_by_cat.keys() if c != "__ALL__"]
            if "__ALL__" in rows_by_cat:
                cats.append("__ALL__")
    if not per_dim or models is None:
        continue
    out_csv = os.path.join(run_dir, f"compare_summary_4dim{suffix}.csv")
    header = ["category"]
    for d in DIMS:
        for m in models:
            header.append(f"{d}_{m}")
    for m in models:
        header.append(f"n_{m}")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cat in cats:
            row = [cat]
            for d in DIMS:
                rows_by_cat, _ = per_dim.get(d, ({}, {}))
                for m in models:
                    row.append(rows_by_cat.get(cat, {}).get(m, ""))
            n_src = per_dim.get("CE") or next(iter(per_dim.values()))
            _, n_by_cat = n_src
            for m in models:
                row.append(n_by_cat.get(cat, {}).get(m, ""))
            w.writerow(row)
    print(f"[write] {out_csv}")
PY
