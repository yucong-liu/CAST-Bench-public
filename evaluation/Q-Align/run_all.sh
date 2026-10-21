#!/usr/bin/env bash
# =====================================================================
# Q-Align (one-align) batch evaluation script
# Output layout aligned with ../audiobox-aesthetics/runs:
#   <RUN_DIR>/<MODEL>/results.csv     # per-video score
#   <RUN_DIR>/<MODEL>/summary.csv     # per-folder mean (folder,n,mean_score)
#   <RUN_DIR>/compare_summary.csv     # cross-model comparison (single metric)
#
# Q-Align outputs only ONE score per video (visual aesthetic),
# so there is no compare_summary_PQ/CE/PC like audiobox; only ONE compare table.
#
# To add a new model:
#   1) append to MODELS=( ... )
#   2) add a case in src_root_for() returning its mp4 root directory
#
# No run.log, no compare_per_video.csv (per user request).
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
export REPO_ROOT MODEL_ROOT

# ---------------- user-tunable parameters ----------------
RUN_DIR=$REPO_ROOT/evaluation/Q-Align/runs
QALIGN_DIR=$REPO_ROOT/evaluation/Q-Align
MODEL_PATH=$MODEL_ROOT/q-align/one-align
DEVICE=cuda

# --- FIX (offline HF): q_align/model/modeling_mplug_owl2.py 里 tokenizer / image_processor
#     的 repo id 硬编码为 "q-future/one-align", 需靠 QALIGN_LOCAL_PATH 环境变量指向本地权重,
#     否则在离线/受限网络 (HF 不可达) 环境下会抛 ConnectionError.
export QALIGN_LOCAL_PATH="$MODEL_PATH"

# Models to evaluate. Add new ones here AND in src_root_for() below.
# ---- 模型清单从共享 TSV 读入：新增模型只需在 models.tsv 加一行 ----
MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"
# 先把外部传进来的 MODELS=a,b,c 子集过滤值保存下来，避免被下面的 MODELS=() 数组覆盖
MODELS_FILTER="${MODELS:-}"
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

# ---- 可选: 通过环境变量 MODELS=a,b,c 选择子集（不在 models.tsv 中的名字直接报错） ----
# 用法:
#   bash run_all.sh                          # 跑 models.tsv 全部
#   MODELS=seedance bash run_all.sh          # 只跑 seedance
#   MODELS=veo3.1,kling-v3-omni bash run_all.sh   # 只跑这两个
MODELS_ALL=("${MODELS[@]}")
if [[ -n "$MODELS_FILTER" ]]; then
    IFS=',' read -r -a WANT <<<"$MODELS_FILTER"
    SUB=()
    for w in "${WANT[@]}"; do
        w="$(echo "$w" | xargs)"   # trim
        [[ -z "$w" ]] && continue
        if [[ -z "${SRC_MAP[$w]:-}" ]]; then
            echo "[ERROR] MODELS contains '$w' which is not in $MODELS_TSV" >&2
            echo "        available: ${MODELS_ALL[*]}" >&2
            exit 1
        fi
        SUB+=("$w")
    done
    MODELS=("${SUB[@]}")
    echo "[INFO] MODELS filter active, will run: ${MODELS[*]}"
else
    echo "[INFO] no MODELS filter, will run all: ${MODELS[*]}"
fi
# ---------------------------------------------------------

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# ============== Step 1: per-model evaluation ==============
for MODEL in "${MODELS[@]}"; do
    SRC=$(src_root_for "$MODEL")
    if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
        echo "[ERROR] model=$MODEL src=$SRC not found, skip"
        continue
    fi
    OUT_DIR="$RUN_DIR/$MODEL"

    # 已经跑过 (results.csv 非空) 则跳过, 保留旧结果参与末尾的跨模型汇总;
    # 想强制重跑某模型: rm -rf runs/<MODEL>
    if [[ -s "$OUT_DIR/results.csv" ]]; then
        echo "[skip] $MODEL: results.csv already exists ($OUT_DIR/results.csv), 保留旧结果参与汇总"
        continue
    fi

    mkdir -p "$OUT_DIR"

    echo "============================================================"
    echo "[INFO] running model=$MODEL  src=$SRC  out=$OUT_DIR"
    echo "============================================================"

    python "$QALIGN_DIR/batch_eval.py" \
        --root "$SRC" \
        --recursive_root \
        --device "$DEVICE" \
        --model_path "$MODEL_PATH" \
        --save_csv "$OUT_DIR/results.csv" \
        --save_summary_csv "$OUT_DIR/summary.csv"
done

# ============== Step 2: cross-model comparison ==============
# 输出三张表 (列结构相同, 区别仅在数据来源):
#   compare_summary.csv         全样本
#   compare_summary_single.csv  仅 single_shot 子集
#   compare_summary_multi.csv   仅 multi_shot  子集
# 直接基于 results.csv (folder, video_path, score) + spec_kind_map 拆分,
# 不再依赖 summary.csv (因为 summary.csv 已经按 folder 聚合, 没法再拆 single/multi).
# 开源模型 (LTX-2.3 / Ovi / daVinci-MagiHuman) 不生成多镜头视频, 因此
# compare_summary_multi.csv 中它们的列会留空, 这是预期行为.
SPECS_DIR="${SPECS_DIR:-$REPO_ROOT/t2av-bench/specs}"
EVAL_COMMON="$REPO_ROOT/evaluation/_common"

python - <<PY
import os, csv, sys
from statistics import mean
from collections import defaultdict

sys.path.insert(0, "$EVAL_COMMON")
try:
    from spec_kind import load_spec_kind_map
except Exception as e:
    print(f"[WARN] cannot import spec_kind: {e}; will only emit full compare_summary.csv")
    load_spec_kind_map = None

spec_kind_map = load_spec_kind_map("$SPECS_DIR", verbose=True) if load_spec_kind_map else {}

run_dir = "$RUN_DIR"
# 汇总始终扫 runs/ 下所有含 results.csv 的模型目录, 按 models.tsv 顺序排列.
# 这样即使本次 Step 1 只跑了新模型的子集, 旧模型也会自动带入汇总, 不丢列.
tsv_order = "${MODELS_ALL[*]}".split()
existing = []
if os.path.isdir(run_dir):
    for name in sorted(os.listdir(run_dir)):
        rp = os.path.join(run_dir, name, "results.csv")
        if os.path.isfile(rp):
            existing.append(name)
ordered = [m for m in tsv_order if m in existing]
ordered += [m for m in existing if m not in tsv_order]
models = ordered
print(f"[compare] models in compare_summary_*.csv = {models}")

# ---- 类目归一化 & 白名单 ----
# 各模型 src 根目录深浅不一, 有的会在 folder 前缀里带 "540p/" 或 "output/",
# 导致同一个类目 (如 action) 被拆成 "action" / "540p/action" / "output/action" 三行,
# 三张 compare 表因此无法横向对齐. 这里做两件事:
#   1) folder 只取最后一段 (basename), 把上述三种情况统一成 "action";
#   2) 只保留 8 类白名单; 白名单外的 (例如 seedance 里的 hard/emtion) 视为脏数据丢弃.
CANON_CATS = {
    "action", "ambient", "animal", "bgm", "instrument",
    "non_verbal_vocal", "object", "sfx", "speech",
}
_dropped_report = defaultdict(int)  # (model, raw_folder) -> count

def _canon(cat_raw: str):
    """把带前缀的 folder 归一到 8 类中的一个; 不在白名单则返回 None."""
    if not cat_raw:
        return None
    tail = os.path.basename(cat_raw.rstrip("/"))
    return tail if tail in CANON_CATS else None

# (kind_filter_name) -> {category: {model: [scores]}}
buckets = {
    "all":    defaultdict(lambda: defaultdict(list)),
    "single": defaultdict(lambda: defaultdict(list)),
    "multi":  defaultdict(lambda: defaultdict(list)),
}

def _stem(p):
    name = os.path.basename(str(p))
    return os.path.splitext(name)[0]

for m in models:
    res_csv = os.path.join(run_dir, m, "results.csv")
    if not os.path.isfile(res_csv):
        print(f"[WARN] missing {res_csv}, skip model {m}")
        continue
    with open(res_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            cat_raw = (row.get("folder") or "").strip()
            cat = _canon(cat_raw)
            if cat is None:
                _dropped_report[(m, cat_raw)] += 1
                continue
            try:
                s = float(row.get("score", ""))
            except (ValueError, TypeError):
                continue
            sid = _stem(row.get("video_path", ""))
            kind = spec_kind_map.get(sid, "")

            buckets["all"][cat][m].append(s)
            if kind == "single_shot":
                buckets["single"][cat][m].append(s)
            elif kind == "multi_shot":
                buckets["multi"][cat][m].append(s)

if _dropped_report:
    print("[warn] dropped rows whose folder is not in CANON_CATS whitelist:")
    for (m, raw), n in sorted(_dropped_report.items()):
        print(f"       model={m}  raw_folder={raw!r}  n={n}")

OUTS = [
    ("all",    "compare_summary.csv"),
    ("single", "compare_summary_single.csv"),
    ("multi",  "compare_summary_multi.csv"),
]

for key, name in OUTS:
    data = buckets[key]
    cats_present = sorted(data.keys())

    out_csv = os.path.join(run_dir, name)
    header = ["category"] + models + [f"n_{m}" for m in models]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        # 每个 folder 一行
        for cat in cats_present:
            row = [cat]
            for m in models:
                vs = data[cat].get(m, [])
                row.append(f"{mean(vs):.6f}" if vs else "")
            for m in models:
                vs = data[cat].get(m, [])
                row.append(len(vs))
            w.writerow(row)
        # __ALL__ 行
        row = ["__ALL__"]
        all_per_model = {m: [s for c in cats_present for s in data[c].get(m, [])]
                         for m in models}
        for m in models:
            vs = all_per_model[m]
            row.append(f"{mean(vs):.6f}" if vs else "")
        for m in models:
            row.append(len(all_per_model[m]))
        w.writerow(row)
    print(f"[write] {out_csv}")
PY

echo
echo "============================================================"
echo "[DONE] all models finished. outputs under: $RUN_DIR"
echo "============================================================"
