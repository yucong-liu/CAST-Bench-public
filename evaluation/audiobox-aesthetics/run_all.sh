#!/usr/bin/env bash
# =====================================================================
# audiobox-aesthetics 批量评测脚本
# 输出布局对齐 ../Q-Align/runs：每个模型一份 results.csv + summary.csv，
# 顶层再生成 4 张 compare_summary_{CE,CU,PC,PQ}.csv（4 个维度全裂）。
#
# 新增模型 = 在 MODELS 数组里追加一项 + 在 src_root_for() 里加一个 case。
# 不输出 run.log，不输出 compare_per_video.csv。
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models}"
export REPO_ROOT MODEL_ROOT

# ---------------- 用户可调参数 ----------------
RUN_DIR=$REPO_ROOT/evaluation/audiobox-aesthetics/runs
CKPT=$MODEL_ROOT/audiobox-aesthetics/checkpoint.pt
BATCH_SIZE=16

# 想跑哪些模型就在这里列；新增模型记得同步在 src_root_for() 里加 case
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
#   MODELS=seedance,wan2.7 bash run_all.sh   # 只跑这两个
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
# ----------------------------------------------

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# ============== Step 1: 每个模型跑 audio-aes，生成 results.csv / summary.csv ==============
for MODEL in "${MODELS[@]}"; do
    SRC=$(src_root_for "$MODEL")
    if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
        echo "[ERROR] model=$MODEL src=$SRC not found, skip"
        continue
    fi
    OUT_DIR="$RUN_DIR/$MODEL"
    mkdir -p "$OUT_DIR"

    echo "============================================================"
    echo "[INFO] running model=$MODEL  src=$SRC  out=$OUT_DIR"
    echo "============================================================"

    # 1) input.jsonl  (先用 ffprobe 过滤掉无音轨的 mp4，再 ffmpeg 抽出 wav 缓存)
    #    说明: 当前 torchaudio 只剩 soundfile backend，不能直接读 mp4，所以
    #         在这里先把每个 mp4 抽成 16kHz mono wav 缓存到 $OUT_DIR/wav_cache/，
    #         input.jsonl 里写 wav 路径喂给 audio-aes；results.csv 后面再回填原 mp4 路径。
    python - <<PY
import glob, json, os, subprocess
src = "$SRC"
out_dir = "$OUT_DIR"
out = os.path.join(out_dir, "input.jsonl")
map_path = os.path.join(out_dir, "wav2mp4.jsonl")
skip_log = os.path.join(out_dir, "skipped_no_audio.txt")
fail_log = os.path.join(out_dir, "skipped_extract_failed.txt")
wav_cache_root = os.path.join(out_dir, "wav_cache")
os.makedirs(wav_cache_root, exist_ok=True)
mp4s = sorted(glob.glob(os.path.join(src, "**", "*.mp4"), recursive=True))

def has_audio(p):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","a",
             "-show_entries","stream=codec_type","-of","csv=p=0", p],
            capture_output=True, text=True, timeout=30,
        )
        return "audio" in r.stdout
    except Exception as e:
        print(f"[ffprobe-err] {p}: {e}")
        return False

def extract_wav(mp4_path):
    # 缓存路径：保留 src 下的相对目录结构，避免不同子目录同名 mp4 撞车
    rel = os.path.relpath(mp4_path, src)
    wav_rel = os.path.splitext(rel)[0] + ".wav"
    wav_path = os.path.join(wav_cache_root, wav_rel)
    if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
        return wav_path  # 已缓存，跳过
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    try:
        r = subprocess.run(
            ["ffmpeg","-y","-loglevel","error","-i", mp4_path,
             "-vn","-ac","1","-ar","16000", wav_path],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0 or not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
            print(f"[ffmpeg-err] {mp4_path}: rc={r.returncode}  stderr={r.stderr.strip()[:200]}")
            return None
        return wav_path
    except Exception as e:
        print(f"[ffmpeg-exc] {mp4_path}: {e}")
        return None

kept_pairs, skipped, failed = [], [], []
for p in mp4s:
    if not has_audio(p):
        skipped.append(p); continue
    wav = extract_wav(p)
    if wav is None:
        failed.append(p); continue
    kept_pairs.append((wav, p))

with open(out, "w") as f:
    for wav, _mp4 in kept_pairs:
        f.write(json.dumps({"path": wav}) + "\n")
with open(map_path, "w") as f:
    for wav, mp4 in kept_pairs:
        f.write(json.dumps({"wav": wav, "mp4": mp4}) + "\n")
with open(skip_log, "w") as f:
    for p in skipped:
        f.write(p + "\n")
with open(fail_log, "w") as f:
    for p in failed:
        f.write(p + "\n")
print(f"[gen] {out}  total={len(mp4s)}  with_audio_extracted={len(kept_pairs)}  skipped_no_audio={len(skipped)}  extract_failed={len(failed)}")
PY

    # 2) 跑 audio-aes（stderr 不写文件，直接打到屏幕）
    audio-aes "$OUT_DIR/input.jsonl" \
        --batch-size "$BATCH_SIZE" \
        --ckpt "$CKPT" \
        > "$OUT_DIR/output.jsonl"

    # 3) 合并 input + output 为 results.csv，再聚合 summary.csv
    #    注意: input.jsonl 里是 wav 缓存路径，需要通过 wav2mp4.jsonl 映射回原 mp4 路径，
    #         保证 results.csv 的 video_path / folder 仍以原始 mp4 为准。
    python - <<PY
import json, os, csv, statistics
from collections import defaultdict

out_dir = "$OUT_DIR"
inp = os.path.join(out_dir, "input.jsonl")
oup = os.path.join(out_dir, "output.jsonl")
map_path = os.path.join(out_dir, "wav2mp4.jsonl")
skip_log = os.path.join(out_dir, "skipped_no_audio.txt")
fail_log = os.path.join(out_dir, "skipped_extract_failed.txt")

wav2mp4 = {}
if os.path.isfile(map_path):
    with open(map_path) as f:
        for line in f:
            d = json.loads(line)
            wav2mp4[d["wav"]] = d["mp4"]

paths, scores = [], []
with open(inp) as f:
    for line in f:
        wav = json.loads(line)["path"]
        paths.append(wav2mp4.get(wav, wav))  # 回填到原始 mp4 路径
with open(oup) as f:
    for line in f:
        scores.append(json.loads(line))
assert len(paths) == len(scores), f"len mismatch: {len(paths)} vs {len(scores)}"

n_skipped = 0
if os.path.isfile(skip_log):
    with open(skip_log) as f:
        n_skipped = sum(1 for ln in f if ln.strip())
n_failed = 0
if os.path.isfile(fail_log):
    with open(fail_log) as f:
        n_failed = sum(1 for ln in f if ln.strip())

DIMS = ["CE", "CU", "PC", "PQ"]

# results.csv: folder, video_path, CE, CU, PC, PQ
res_csv = os.path.join(out_dir, "results.csv")
with open(res_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["folder", "video_path"] + DIMS)
    for p, s in zip(paths, scores):
        folder = os.path.basename(os.path.dirname(p))
        w.writerow([folder, p] + [s.get(d, "") for d in DIMS])
print(f"[write] {res_csv}  rows={len(paths)}")

# summary.csv: folder, n, CE, CU, PC, PQ  (+ __ALL__)
buckets = defaultdict(list)
for p, s in zip(paths, scores):
    folder = os.path.basename(os.path.dirname(p))
    buckets[folder].append(s)

sum_csv = os.path.join(out_dir, "summary.csv")
with open(sum_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["folder", "n"] + DIMS)
    for folder in sorted(buckets.keys()):
        items = buckets[folder]
        means = [statistics.mean([it[d] for it in items]) for d in DIMS]
        w.writerow([folder, len(items)] + means)
    # __ALL__
    all_items = [s for s in scores]
    if all_items:
        means_all = [statistics.mean([it[d] for it in all_items]) for d in DIMS]
        w.writerow(["__ALL__", len(all_items)] + means_all)
    # 无音轨被剔除的样本数（仅留痕，不参与均值计算）
    if n_skipped > 0:
        w.writerow(["__SKIPPED_NO_AUDIO__", n_skipped, "", "", "", ""])
    # ffmpeg 抽轨失败的样本数（仅留痕，不参与均值计算）
    if n_failed > 0:
        w.writerow(["__EXTRACT_FAILED__", n_failed, "", "", "", ""])
print(f"[write] {sum_csv}  (skipped_no_audio={n_skipped}, extract_failed={n_failed})")
PY
done

# ============== Step 2: 跨模型对比，4 个维度各生成三张 compare_summary_*.csv ==============
# 输出 (per-dim ∈ {CE, CU, PC, PQ}):
#   compare_summary_<DIM>.csv         全样本
#   compare_summary_<DIM>_single.csv  仅 single_shot 子集
#   compare_summary_<DIM>_multi.csv   仅 multi_shot  子集
# 数据直接来自各模型的 results.csv (per-video 4 维分数), 通过 spec_id (= 文件名 stem)
# 反查 t2av-bench specs 拿 shot_count 拆分; summary.csv 仍然保留为模型自身的全样本文件.
# 开源模型不生成多镜头视频, 因此 *_multi.csv 里它们的列会留空, 这是预期行为.
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
    print(f"[WARN] cannot import spec_kind: {e}; will only emit full compare_summary.")
    load_spec_kind_map = None

spec_kind_map = load_spec_kind_map("$SPECS_DIR", verbose=True) if load_spec_kind_map else {}

run_dir = "$RUN_DIR"
tsv_order = "${MODELS_ALL[*]}".split()  # models.tsv 中的顺序
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

DIMS = ["CE", "CU", "PC", "PQ"]
KINDS = [("all", None), ("single", "single_shot"), ("multi", "multi_shot")]

def _stem(p):
    name = os.path.basename(str(p))
    return os.path.splitext(name)[0]

# bucket[kind][dim][category][model] = [val, val, ...]
bucket = {k[0]: {d: defaultdict(lambda: defaultdict(list)) for d in DIMS} for k in KINDS}

for m in models:
    res_csv = os.path.join(run_dir, m, "results.csv")
    if not os.path.isfile(res_csv):
        print(f"[WARN] missing {res_csv}, skip model {m}")
        continue
    with open(res_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            cat = (row.get("folder") or "").strip()
            sid = _stem(row.get("video_path", ""))
            kind = spec_kind_map.get(sid, "")
            for d in DIMS:
                try:
                    v = float(row.get(d, ""))
                except (ValueError, TypeError):
                    continue
                bucket["all"][d][cat][m].append(v)
                if kind == "single_shot":
                    bucket["single"][d][cat][m].append(v)
                elif kind == "multi_shot":
                    bucket["multi"][d][cat][m].append(v)

for kind_name, _kind_filter in KINDS:
    for d in DIMS:
        data = bucket[kind_name][d]
        cats_present = sorted(data.keys())

        suffix = "" if kind_name == "all" else f"_{kind_name}"
        out_csv = os.path.join(run_dir, f"compare_summary_{d}{suffix}.csv")
        header = ["category"] + models + [f"n_{m}" for m in models]
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for cat in cats_present:
                row = [cat]
                for m in models:
                    vs = data[cat].get(m, [])
                    row.append(f"{mean(vs):.6f}" if vs else "")
                for m in models:
                    row.append(len(data[cat].get(m, [])))
                w.writerow(row)
            # __ALL__
            row = ["__ALL__"]
            agg_per_model = {m: [v for c in cats_present for v in data[c].get(m, [])]
                             for m in models}
            for m in models:
                vs = agg_per_model[m]
                row.append(f"{mean(vs):.6f}" if vs else "")
            for m in models:
                row.append(len(agg_per_model[m]))
            w.writerow(row)
        print(f"[write] {out_csv}")
PY

# ============== Step 3: 把 4 个维度合并到同一张表，便于一眼看完 ==============
# 输入  : compare_summary_{CE,CU,PC,PQ}{suffix}.csv   (suffix ∈ {"", "_single", "_multi"})
# 输出  : compare_summary_4dim{suffix}.csv
# 列结构: category | CE_<m1> ... CE_<mK> | CU_<m1> ... | PC_... | PQ_... | n_<m1> ... n_<mK>
#         （按指标分组：每个指标下连续 K 列对应 K 个模型）
# 行    : 各 category + __ALL__
python - <<PY
import os, csv

run_dir = "$RUN_DIR"
DIMS = ["CE", "CU", "PC", "PQ"]
KINDS = ["", "_single", "_multi"]

# 4 个 dim 共用同一个 model 顺序与同一组 category，只要任一存在就以它为准
def read_compare(path):
    """returns (header, models, rows_by_cat, n_by_cat)"""
    with open(path) as f:
        r = csv.reader(f); rows = list(r)
    if not rows:
        return [], [], {}, {}
    header = rows[0]
    # header = ["category", *models, *("n_<m>" for m in models)]
    n_models = (len(header) - 1) // 2
    models = header[1:1 + n_models]
    rows_by_cat, n_by_cat = {}, {}
    for row in rows[1:]:
        cat = row[0]
        rows_by_cat[cat] = {m: row[1 + i] for i, m in enumerate(models)}
        n_by_cat[cat]    = {m: row[1 + n_models + i] for i, m in enumerate(models)}
    return header, models, rows_by_cat, n_by_cat

for suffix in KINDS:
    per_dim = {}     # dim -> (rows_by_cat, n_by_cat)
    models  = None
    cats    = []
    for d in DIMS:
        path = os.path.join(run_dir, f"compare_summary_{d}{suffix}.csv")
        if not os.path.isfile(path):
            continue
        _, ms, rows_by_cat, n_by_cat = read_compare(path)
        per_dim[d] = (rows_by_cat, n_by_cat)
        if models is None:
            models = ms
        # 保留首次出现的 category 顺序（来自 CE 表），以 __ALL__ 收尾
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
            # n: 4 个 dim 在同一条 (cat, model) 上的样本数应当一致, 取 CE 为准, 缺则按出现顺序
            n_src = per_dim.get("CE") or next(iter(per_dim.values()))
            _, n_by_cat = n_src
            for m in models:
                row.append(n_by_cat.get(cat, {}).get(m, ""))
            w.writerow(row)
    print(f"[write] {out_csv}")
PY

echo
echo "============================================================"
echo "[DONE] all models finished. outputs under: $RUN_DIR"
echo "============================================================"
