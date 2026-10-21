#!/usr/bin/env bash
# =====================================================================
# VBench2 Human_Anatomy 批量评测脚本
#
# 输出布局对齐 ../motion_smoothness/runs / ../audiobox-aesthetics/runs:
#   - 每个模型一个子目录, 内含 results.csv / summary.csv / failed.txt
#   - 顶层 3 张 compare_summary{,_single,_multi}.csv 跨模型对比
#
# 默认只评测 "运动类" category: action,
# 跳过 speech / non_verbal_vocal 以及 animal / object 等无人/非运动类, 聚焦运动人体解剖学评测。
#
# 新增模型 = 在 ../models.tsv 里加一行即可。
#
# 用法:
#   bash run_all.sh                              # 跑 models.tsv 全部
#   MODELS=seedance bash run_all.sh              # 只跑 seedance
#   MODELS=seedance,wan2.7 bash run_all.sh       # 只跑这两个
#   CATEGORIES=all bash run_all.sh               # 取消 category 过滤, 全量评测
#   CATEGORIES=speech,non_verbal_vocal,action bash run_all.sh  # 回退到旧的三类过滤
#   NUM_GPUS=8 bash run_all.sh                   # 同时占用 8 张卡, 每卡独立跑一个完整模型 (默认自动探测)
#   BATCH_SIZE=256 bash run_all.sh               # ViT 分类器 batch_size (默认 256)
#
# 调度模型: ·按模型并行· (不是在同一模型内部切片)
#   同时最多 NUM_GPUS 个模型在跑, 每个占一张卡. 模型多于卡时按 FIFO 排队.
#   这样避免多进程争 CPU/IO 导致的单视频耗时暴涨.
#
# 注: 长任务请用 tee 自己落盘日志, 例如:
#   mkdir -p runs && bash run_all.sh 2>&1 | tee runs/run_$(date +%Y%m%d_%H%M).log
#
# 前置: 第一次跑前必须先执行  bash setup_env.sh  把环境配齐.
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# ---------------- 用户可调参数 ----------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- 运行时必备环境变量 (避免每次跑前还要手动 export) ----
# (1) YOLO-World 源码不是 pip 包, 必须把它的目录放到 PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/vbench2/third_party/YOLO-World:${PYTHONPATH:-}"
# (2) CLIP 已被 setup_env.sh 下载并转好 safetensors, detector config 也指向本地路径,
#     全程不需要联网, 强制走 HF 离线缓存(避免被 SSL/代理抖动连累)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# (3) 抑制 TF 的多余日志(与 batch_eval.py 顶部一致)
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/runs}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CATEGORIES="${CATEGORIES:-action}"
# 权重默认放在工程内 vbench2/vbench2_weights/
# 想放别处: export VBENCH2_CACHE_DIR=/your/path/to/vbench2_weights
CACHE_DIR="${VBENCH2_CACHE_DIR:-$SCRIPT_DIR/vbench2/vbench2_weights}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# SKIP_EXISTING=1 时, 若 <RUN_DIR>/<model>/results.csv 已存在, 则该模型直接 [skip]
# 不发起 batch_eval.py 进程 (但仍然会被 Step 2 自动扫描到, 进入 compare_summary*.csv 汇总).
# 默认 0 = 保持旧行为 (每次持之以恒 rerun 本次 MODELS 内的每个模型).
# 使用场景: 追加新模型 + 进入已有汇总表 → SKIP_EXISTING=1 MODELS=new1,new2,旧模型1 bash run_all.sh
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# ---- 多卡数据并行 ----
# 默认自动探测 GPU 数: 优先看 CUDA_VISIBLE_DEVICES, 其次看 /dev/nvidia*
detect_num_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        # 逗号分隔计数 (跳过空项)
        local n=0
        IFS=',' read -r -a _devs <<<"$CUDA_VISIBLE_DEVICES"
        for d in "${_devs[@]}"; do [[ -n "$(echo "$d" | xargs)" ]] && n=$((n+1)); done
        echo "$n"
        return
    fi
    local cnt
    cnt=$(ls -d /dev/nvidia[0-9]* 2>/dev/null | wc -l)
    if [[ "$cnt" -gt 0 ]]; then echo "$cnt"; return; fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l
        return
    fi
    echo 1
}
NUM_GPUS="${NUM_GPUS:-$(detect_num_gpus)}"
if [[ "$NUM_GPUS" -lt 1 ]]; then NUM_GPUS=1; fi
echo "[INFO] NUM_GPUS = $NUM_GPUS  BATCH_SIZE = $BATCH_SIZE"

# ---- 模型清单从共享 TSV 读入: 新增模型只需在 models.tsv 加一行 ----
MODELS_TSV="${MODELS_TSV:-$SCRIPT_DIR/../models.tsv}"
# 先把外部传进来的 MODELS=a,b,c 子集过滤值保存下来, 避免被下面的 MODELS=() 数组覆盖
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

MODELS_ALL=("${MODELS[@]}")
if [[ -n "$MODELS_FILTER" ]]; then
    IFS=',' read -r -a WANT <<<"$MODELS_FILTER"
    SUB=()
    for w in "${WANT[@]}"; do
        w="$(echo "$w" | xargs)"
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
echo "[INFO] CATEGORIES filter = $CATEGORIES"
echo "[INFO] cache_dir = $CACHE_DIR"
echo "[INFO] SKIP_EXISTING = $SKIP_EXISTING (=1 则已有 results.csv 的模型不重跑, 但仍会进 Step 2 汇总)"
# ----------------------------------------------

mkdir -p "$RUN_DIR"

# 基本前置检查: 4 个权重必须就位
WEIGHTS=(
    "$CACHE_DIR/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
    "$CACHE_DIR/anomaly_detector/human.pth"
    "$CACHE_DIR/anomaly_detector/face.pth"
    "$CACHE_DIR/anomaly_detector/hand.pth"
)
missing=0
for w in "${WEIGHTS[@]}"; do
    if [[ ! -f "$w" ]]; then
        echo "[ERROR] weight not found: $w" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    echo "[ERROR] please place the missing weights, see evaluation/README.md §1.3." >&2
    exit 1
fi

# ============== Step 1: ·按模型并行· 调度每个模型独占一张卡 ==============
# 调度思路:
#   · 同时运行中的模型数 ≤ NUM_GPUS
#   · 每个进程 CUDA_VISIBLE_DEVICES=<gpu_id> 占独一张卡, 不走 batch_eval.py 的
#     --num_shards 分片逻辑 → 输出仍是原生的 results.csv / summary.csv
#   · 可并发同时跑的模型个数取 min(NUM_GPUS, 剩余模型数)
# ----------------------------------------------------------------

# 可分配的 GPU 号列表: 优先尊重 CUDA_VISIBLE_DEVICES (避免踩到别人),
# 否则默认从 0 起连编号
GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _vis <<<"$CUDA_VISIBLE_DEVICES"
    for d in "${_vis[@]}"; do
        d="$(echo "$d" | xargs)"
        [[ -n "$d" ]] && GPU_IDS+=("$d")
    done
else
    for ((i=0; i<NUM_GPUS; i++)); do GPU_IDS+=("$i"); done
fi
# 如果可见 GPU 多于 NUM_GPUS, 以 NUM_GPUS 为准裁剪
if [[ "${#GPU_IDS[@]}" -gt "$NUM_GPUS" ]]; then
    GPU_IDS=("${GPU_IDS[@]:0:$NUM_GPUS}")
fi
echo "[INFO] GPU_IDS = (${GPU_IDS[*]})  parallel models = ${#GPU_IDS[@]}"

# slot 状态: SLOT_PID[i] = 占用 GPU_IDS[i] 的进程 pid (0 表示空闲)
#         SLOT_MODEL[i] = 进程跑的模型名 (供报错使用)
NSLOT="${#GPU_IDS[@]}"
SLOT_PID=()
SLOT_MODEL=()
for ((s=0; s<NSLOT; s++)); do SLOT_PID+=(0); SLOT_MODEL+=(""); done

# 全局失败计数
GLOBAL_FAIL=0
FAILED_MODELS=()

# 启动一个模型到指定 slot (后台进程, 但不是 nohup/disown, 受主脚本控制)
launch_model_on_slot() {
    local slot_idx="$1" model="$2" src="$3"
    local gpu_id="${GPU_IDS[$slot_idx]}"
    local out_dir="$RUN_DIR/$model"
    mkdir -p "$out_dir"

    # 清掉上次跑的残留 (包括旧的 shard 产物)
    rm -f "$out_dir"/results.shard*.csv \
          "$out_dir"/failed.shard*.txt \
          "$out_dir"/shard*.log \
          "$out_dir"/results.csv \
          "$out_dir"/summary.csv \
          "$out_dir"/failed.txt

    local log="$out_dir/run.log"
    echo "[launch] model=$model  GPU=$gpu_id  slot=$slot_idx  log=$log"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    "$PYTHON_BIN" "$SCRIPT_DIR/batch_eval.py" \
        --root       "$src" \
        --out_dir    "$out_dir" \
        --cache_dir  "$CACHE_DIR" \
        --categories "$CATEGORIES" \
        --batch_size "$BATCH_SIZE" \
        --device     "$DEVICE" \
        >"$log" 2>&1 &
    SLOT_PID[$slot_idx]=$!
    SLOT_MODEL[$slot_idx]="$model"
}

# 等待 ·任意一个· slot 空出来 (返回该 slot 索引)
# 如果全部 slot 都空闲返回 "-1"
wait_any_slot() {
    while :; do
        local s
        for ((s=0; s<NSLOT; s++)); do
            local pid="${SLOT_PID[$s]}"
            if [[ "$pid" == "0" ]]; then
                echo "$s"; return
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                # 已退出, 取退出码
                local rc=0
                wait "$pid" 2>/dev/null || rc=$?
                local m="${SLOT_MODEL[$s]}"
                if [[ "$rc" -ne 0 ]]; then
                    echo "[ERROR] model=$m (pid=$pid, GPU=${GPU_IDS[$s]}) exited rc=$rc, see $RUN_DIR/$m/run.log" >&2
                    GLOBAL_FAIL=$((GLOBAL_FAIL+1))
                    FAILED_MODELS+=("$m")
                else
                    echo "[ok]  model=$m (pid=$pid, GPU=${GPU_IDS[$s]}) finished"
                fi
                SLOT_PID[$s]=0
                SLOT_MODEL[$s]=""
                echo "$s"; return
            fi
        done
        sleep 2
    done
}

# 主调度循环: 逐个模型入队, 有空 slot 就发出去
for MODEL in "${MODELS[@]}"; do
    SRC=$(src_root_for "$MODEL")
    if [[ -z "$SRC" || ! -d "$SRC" ]]; then
        echo "[WARN] model=$MODEL src=$SRC not found, skip"
        continue
    fi

    # 增量补跑: SKIP_EXISTING=1 且已有 results.csv → 该模型不重跑,
    # 但 Step 2 扫 runs/*/results.csv 时会自动拉上它, compare_summary*.csv 自然包含.
    if [[ "$SKIP_EXISTING" == "1" && -f "$RUN_DIR/$MODEL/results.csv" ]]; then
        n_rows=$(( $(wc -l < "$RUN_DIR/$MODEL/results.csv") - 1 ))
        echo "[skip] model=$MODEL already has $RUN_DIR/$MODEL/results.csv (${n_rows} rows), skip stage1"
        continue
    fi

    # 找一个空闲 slot, 没有就等
    free_slot=""
    # 先看是否有现成空位
    for ((s=0; s<NSLOT; s++)); do
        if [[ "${SLOT_PID[$s]}" == "0" ]]; then free_slot="$s"; break; fi
    done
    # 没有的话阻塞等到有一个进程退出
    if [[ -z "$free_slot" ]]; then
        free_slot=$(wait_any_slot)
    fi

    echo "============================================================"
    echo "[INFO] dispatch model=$MODEL  -> slot=$free_slot  GPU=${GPU_IDS[$free_slot]}"
    echo "        src=$SRC"
    echo "        BATCH_SIZE=$BATCH_SIZE  CATEGORIES=$CATEGORIES"
    echo "============================================================"
    launch_model_on_slot "$free_slot" "$MODEL" "$SRC"
done

# 等所有还在跑的 slot 收完
for ((s=0; s<NSLOT; s++)); do
    pid="${SLOT_PID[$s]}"
    [[ "$pid" == "0" ]] && continue
    m="${SLOT_MODEL[$s]}"
    if wait "$pid"; then
        echo "[ok]  model=$m (pid=$pid, GPU=${GPU_IDS[$s]}) finished"
    else
        rc=$?
        echo "[ERROR] model=$m (pid=$pid, GPU=${GPU_IDS[$s]}) exited rc=$rc, see $RUN_DIR/$m/run.log" >&2
        GLOBAL_FAIL=$((GLOBAL_FAIL+1))
        FAILED_MODELS+=("$m")
    fi
    SLOT_PID[$s]=0
    SLOT_MODEL[$s]=""
done

if [[ "$GLOBAL_FAIL" -ne 0 ]]; then
    echo "[ERROR] ${GLOBAL_FAIL} model(s) failed: ${FAILED_MODELS[*]}" >&2
    echo "        compare 阶段仍会尝试跟已成功的模型跑, 以便看部分结果" >&2
fi

# ============== Step 2: 跨模型对比, 输出 3 张 compare_summary*.csv ==============
# 输出:
#   compare_summary.csv          全样本
#   compare_summary_single.csv   仅 single_shot
#   compare_summary_multi.csv    仅 multi_shot
SPECS_DIR="${SPECS_DIR:-$REPO_ROOT/t2av-bench/specs}"
EVAL_COMMON="$SCRIPT_DIR/../_common"

"$PYTHON_BIN" - <<PY
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
print(f"[compare] models = {models}")

KINDS = [("all", None), ("single", "single_shot"), ("multi", "multi_shot")]

def _stem(p):
    name = os.path.basename(str(p))
    return os.path.splitext(name)[0]

# bucket[kind][category][model] = [val, val, ...]
bucket = {k[0]: defaultdict(lambda: defaultdict(list)) for k in KINDS}

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
            try:
                v = float(row.get("human_anatomy", ""))
            except (ValueError, TypeError):
                continue
            bucket["all"][cat][m].append(v)
            if kind == "single_shot":
                bucket["single"][cat][m].append(v)
            elif kind == "multi_shot":
                bucket["multi"][cat][m].append(v)

for kind_name, _kf in KINDS:
    data = bucket[kind_name]
    cats_present = sorted(data.keys())

    suffix = "" if kind_name == "all" else f"_{kind_name}"
    out_csv = os.path.join(run_dir, f"compare_summary{suffix}.csv")
    # 列顺序: category | <model score>... | n_<model>...
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

echo
echo "============================================================"
echo "[DONE] all models finished. outputs under: $RUN_DIR"
echo "       per-model: <model>/results.csv, <model>/summary.csv"
echo "       compare  : compare_summary{,_single,_multi}.csv"
echo "============================================================"
