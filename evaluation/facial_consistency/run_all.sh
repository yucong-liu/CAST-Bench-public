#!/usr/bin/env bash
# =====================================================================
# facial_consistency 评测点 · 多模型批跑 (无 VLM, 纯 InsightFace + DBSCAN)
#
# 流水线 (一次性串完):
#   1) batch_eval.py
#        - 不切片. 跨 9 个 category 扫 <video-root>/<cat>/<spec_id>.mp4
#        - 期望主角脸数 = specs[*].subjects 中 type=='human' 的数量
#          (不再依赖 Gemini 弱标签)
#        - 算法 = InsightFace buffalo_l 检测 -> 贪心追踪 -> DBSCAN 聚类
#                 -> 数量 40 + 稳定性 60 打分 (0-100)
#        - 多 job 模式: 一个进程内顺序跑完所有视频模型 (jobs.tsv)
#        - DP 多客户端 (N_SHARDS>=2): 每个 shard 进程独立加载一份 InsightFace,
#          按 spec_id hash 取模分片. 每个进程都需要一张 GPU (CUDA Provider),
#          多卡才有意义; 单卡建议 N_SHARDS=1.
#   2) merge_shards.py (N_SHARDS>=2 才执行)
#        - 把每个模型的 per_video.shard*.jsonl + summary.shard*.json 合并
#   3) aggregate_summary.py
#        - 跨模型按 category 拆 compare_summary*.csv
#
# 模型清单: evaluation/models.tsv (两列 TSV: 模型名\t视频 output 根目录)
# 新增模型 = 在 models.tsv 加一行, 不需要改本脚本.
#
# 全部前台运行, 不使用 nohup/&/setsid. 内部并发进程仍是 & + wait, 顶层
# 脚本始终前台 (Ctrl-C 可终止整条流水线).
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# -------- 让 onnxruntime CUDA EP 能加载 torch wheel 自带的 cuDNN 9 / cuBLAS / NVRTC ------
# 现象: onnxruntime-gpu==1.20.x 需要 libcudnn_adv.so.9 / libcublasLt.so.12 / libnvrtc.so.12,
# 但 pip 装的 nvidia-cudnn-cu12 / nvidia-cublas-cu12 / nvidia-cuda-nvrtc-cu12 只把 .so 放
# 在 site-packages/nvidia/*/lib/ 下, 不会写进 ldconfig, 于是 dlopen 找不到, 报
# "libcudnn_adv.so.9: cannot open shared object file". 这里把这几个私有目录加进
# LD_LIBRARY_PATH, 走"零系统改动、零 pip"的方式打通.
_ORT_NV="$(python3 -c 'import nvidia; print(list(nvidia.__path__)[0])')"
export LD_LIBRARY_PATH="${_ORT_NV}/cudnn/lib:${_ORT_NV}/cublas/lib:${_ORT_NV}/cuda_nvrtc/lib:${_ORT_NV}/cuda_runtime/lib:${_ORT_NV}/cufft/lib:${_ORT_NV}/curand/lib:${_ORT_NV}/cusolver/lib:${_ORT_NV}/cusparse/lib:${_ORT_NV}/nccl/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---------------- 用户可调参数 ----------------
EVAL_ROOT=$REPO_ROOT/evaluation/facial_consistency
RUN_DIR=$EVAL_ROOT/runs
SPECS_DIR=$REPO_ROOT/t2av-bench/specs

EVAL_PY=$EVAL_ROOT/batch_eval.py
MERGE_PY=$EVAL_ROOT/merge_shards.py
AGG_PY=$EVAL_ROOT/aggregate_summary.py

# InsightFace 设备 / 检测器
CTX_ID="${CTX_ID:-0}"            # GPU 卡号 (CUDAExecutionProvider)
DET_SIZE="${DET_SIZE:-640,640}"  # InsightFace det_size (W,H)

# pipeline cfg (与原 batch_eval 默认值保持一致)
SAMPLE_RATE="${SAMPLE_RATE:-1}"
MIN_FACE_PX="${MIN_FACE_PX:-30}"
MIN_DET_SCORE="${MIN_DET_SCORE:-0.6}"
MIN_REL_AREA="${MIN_REL_AREA:-0.0005}"
MAX_GAP_FRAMES="${MAX_GAP_FRAMES:-3}"
MIN_IOU="${MIN_IOU:-0.1}"
MIN_COS="${MIN_COS:-0.35}"
MIN_TRACKLET_LEN="${MIN_TRACKLET_LEN:-3}"
DBSCAN_EPS="${DBSCAN_EPS:-0.65}"
DBSCAN_MIN_SAMPLES="${DBSCAN_MIN_SAMPLES:-1}"
PRIMARY_RATIO_THRESH="${PRIMARY_RATIO_THRESH:-0.10}"
MIN_PRIMARY_CLUSTER_DETS="${MIN_PRIMARY_CLUSTER_DETS:-30}"

# anthropomorphic (会说话的卡通动物等) 是否计入期望人脸数. InsightFace 训练
# 数据是真实人脸, 卡通脸通常检不到, 默认 0=不计入.
INCLUDE_ANTHROPOMORPHIC="${INCLUDE_ANTHROPOMORPHIC:-0}"

# Category 过滤: 只评测有人的 3 个桶 (对话 / 动作 / 非语言人声).
# 传 CATEGORIES=all 取消过滤, 跑全部 9 个桶.
CATEGORIES="${CATEGORIES:-speech,non_verbal_vocal,action}"

LIMIT="${LIMIT:-0}"  # 调试用: 每个模型只跑前 N 条 spec, 0=全跑

# DP: InsightFace 是 GPU 重资源, 多 shard 会各自加载一份模型. 单卡 80GB 上
# 建议 N_SHARDS=1; 多卡机可设 N_SHARDS=卡数, 每个 shard 用 CTX_ID=$i.
N_SHARDS="${N_SHARDS:-1}"

MODELS_TSV="${MODELS_TSV:-$REPO_ROOT/evaluation/models.tsv}"

RUN_EVAL="${RUN_EVAL:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
# ----------------------------------------------

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
load_models

# ---- 可选: 通过环境变量 MODELS=a,b,c 选择子集 (不在 models.tsv 中的名字直接报错) ----
# 用法:
#   bash run_all.sh                          # 跑 models.tsv 全部 (已跑过的自动 [skip])
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

# 模型名 -> 视频根目录:
#   1) <root> 本身含 category 子目录 (一般情况)
#   2) <root>/<中间一层>/ 含 category 子目录 (兼容 daVinci-MagiHuman: output/540p,
#      也兼容 kling-v3 配在 generation/kling-v3 但视频在 generation/kling-v3/output)
src_root_for() {
    local root="${SRC_MAP[$1]:-}"
    [[ -z "$root" ]] && { echo ""; return; }
    if [[ -d "$root/speech" || -d "$root/action" || -d "$root/object" \
       || -d "$root/sfx" || -d "$root/ambient" || -d "$root/animal" \
       || -d "$root/bgm" || -d "$root/instrument" || -d "$root/non_verbal_vocal" ]]; then
        echo "$root"; return
    fi
    local cand
    cand=$(find "$root" -mindepth 2 -maxdepth 2 -type d \
           \( -name speech -o -name action -o -name object -o -name sfx \
              -o -name ambient -o -name animal -o -name bgm \
              -o -name instrument -o -name non_verbal_vocal \) \
           2>/dev/null | head -n 1)
    if [[ -n "$cand" ]]; then
        echo "$(dirname "$cand")"; return
    fi
    echo "$root"
}

mkdir -p "$RUN_DIR"

out_dir_for() {
    local m="$1"
    echo "$RUN_DIR/$m"
}

echo "=========================================================="
echo "[run_all] EVAL_ROOT      = $EVAL_ROOT"
echo "[run_all] RUN_DIR        = $RUN_DIR"
echo "[run_all] SPECS_DIR      = $SPECS_DIR"
echo "[run_all] MODELS_TSV     = $MODELS_TSV"
echo "[run_all] N_SHARDS       = $N_SHARDS  (CTX_ID=$CTX_ID, DET_SIZE=$DET_SIZE)"
echo "[run_all] RUN_EVAL=$RUN_EVAL  RUN_AGGREGATE=$RUN_AGGREGATE  LIMIT=$LIMIT"
echo "[run_all] INCLUDE_ANTHROPOMORPHIC=$INCLUDE_ANTHROPOMORPHIC"
echo "[run_all] CATEGORIES     = $CATEGORIES"
echo "[run_all] models from tsv: ${MODELS[*]}"
echo "=========================================================="

# ============== 阶段 1: 评测 ==============
if [[ "$RUN_EVAL" == "1" ]]; then
    JOBS_TSV="$RUN_DIR/_jobs.tsv"
    : > "$JOBS_TSV"
    JOB_MODELS=()
    for M in "${MODELS[@]}"; do
        SRC="$(src_root_for "$M")"
        OUT="$(out_dir_for "$M")"
        if [[ -z "$SRC" || ! -d "$SRC" ]]; then
            echo "[run_all] $M: src not found ($SRC), skip"; continue
        fi

        # 已跑过判定: per_video.jsonl 非空 AND 行数 >= 应有 spec 数 才 skip;
        # 若非空但行数远少于应有数(残破产物, 上次被中断/kill), 会自动把旧文件备份改名后重跑,
        # 避免用坏数据参与 aggregate. (历史 bug: 只判 -s, 12 行破碎产物也被当"完成态".)
        # 想强制重跑: FORCE_RERUN=1 bash run_all.sh   或者   rm -rf runs/<MODEL>
        if [[ "${FORCE_RERUN:-0}" != "1" && -s "$OUT/per_video.jsonl" ]]; then
            # 计算应有 spec 数 = 匹配 CATEGORIES 的 specs jsonl 行数之和
            if [[ "$CATEGORIES" == "all" ]]; then
                _CATS_LIST="speech non_verbal_vocal action object sfx ambient animal bgm instrument"
            else
                _CATS_LIST="${CATEGORIES//,/ }"
            fi
            _EXPECTED_N=0
            for _C in $_CATS_LIST; do
                _F="$SPECS_DIR/${_C}.jsonl"
                [[ -f "$_F" ]] && _EXPECTED_N=$(( _EXPECTED_N + $(grep -c . "$_F" 2>/dev/null || echo 0) ))
            done
            _ACTUAL_N=$(grep -c . "$OUT/per_video.jsonl" 2>/dev/null || echo 0)
            if [[ "$_EXPECTED_N" -gt 0 && "$_ACTUAL_N" -ge "$_EXPECTED_N" ]]; then
                echo "[run_all] $M: per_video.jsonl 完整 ($_ACTUAL_N / $_EXPECTED_N), 保留旧结果参与汇总"
                continue
            else
                _TS=$(date +%Y%m%d_%H%M%S)
                _BAK="$OUT/per_video.jsonl.bak_incomplete_${_ACTUAL_N}of${_EXPECTED_N}_${_TS}"
                echo "[run_all] $M: per_video.jsonl 残破 ($_ACTUAL_N / $_EXPECTED_N), 备份改名后重跑: $_BAK"
                mv "$OUT/per_video.jsonl" "$_BAK"
                # summary.json 若存在也备份, 否则 aggregate 会拿旧 summary 混
                if [[ -f "$OUT/summary.json" ]]; then
                    mv "$OUT/summary.json" "$OUT/summary.json.bak_incomplete_${_TS}"
                fi
                # 继续走下面 mkdir + 加入 jobs.tsv 的正常重跑流程
            fi
        fi

        mkdir -p "$OUT"
        printf "%s\t%s\t%s\n" "$M" "$SRC" "$OUT" >> "$JOBS_TSV"
        JOB_MODELS+=("$M")
        # 清理上次的 shard 文件 (per_video.jsonl/summary.json 不动, 让 batch_eval 自己覆盖)
        rm -f "$OUT"/per_video.shard*.jsonl "$OUT"/summary.shard*.json
    done

    if [[ ${#JOB_MODELS[@]} -eq 0 ]]; then
        echo "[run_all] no model has a valid video_root, skip eval stage"
    else
        echo
        echo "############ [eval] jobs=${#JOB_MODELS[@]}  N_SHARDS=${N_SHARDS} ############"
        echo "[run_all]   jobs_tsv = $JOBS_TSV"
        echo "[run_all]   models   : ${JOB_MODELS[*]}"
        cat "$JOBS_TSV" | sed 's/^/[run_all]     /'

        EXTRA_ARGS=()
        if [[ "$LIMIT" != "0" ]]; then
            EXTRA_ARGS+=(--limit "$LIMIT")
        fi
        if [[ "$INCLUDE_ANTHROPOMORPHIC" == "1" ]]; then
            EXTRA_ARGS+=(--include-anthropomorphic)
        fi

        rm -f "$RUN_DIR"/eval.shard*.log

        if [[ "$N_SHARDS" -le 1 ]]; then
            # 单进程: 直接前台跑, 输出到终端 + 落盘日志
            LOG="$RUN_DIR/eval.shard0.log"
            echo "[run_all]   single-shard mode, log=$LOG"
            python3 -u "$EVAL_PY" \
                --jobs-tsv             "$JOBS_TSV" \
                --specs-dir            "$SPECS_DIR" \
                --categories           "$CATEGORIES" \
                --shard-idx            0 \
                --shard-num            1 \
                --ctx-id               "$CTX_ID" \
                --det-size             "$DET_SIZE" \
                --sample-rate          "$SAMPLE_RATE" \
                --min-face-px          "$MIN_FACE_PX" \
                --min-det-score        "$MIN_DET_SCORE" \
                --min-rel-area         "$MIN_REL_AREA" \
                --max-gap-frames       "$MAX_GAP_FRAMES" \
                --min-iou              "$MIN_IOU" \
                --min-cos              "$MIN_COS" \
                --min-tracklet-len     "$MIN_TRACKLET_LEN" \
                --dbscan-eps           "$DBSCAN_EPS" \
                --dbscan-min-samples   "$DBSCAN_MIN_SAMPLES" \
                --primary-ratio-thresh "$PRIMARY_RATIO_THRESH" \
                --min-primary-cluster-dets "$MIN_PRIMARY_CLUSTER_DETS" \
                "${EXTRA_ARGS[@]}" \
                2>&1 | tee "$LOG"
        else
            # 多 shard: 每个 shard 用 CTX_ID = base + i (假定 i 张连续 GPU 可用),
            # 失败时自行调小 N_SHARDS 或显式 export CTX_ID 固定一张卡.
            pids=()
            for ((i=0; i<N_SHARDS; i++)); do
                LOG="$RUN_DIR/eval.shard${i}.log"
                CUR_CTX=$((CTX_ID + i))
                echo "[run_all]   spawn shard $i  ctx_id=$CUR_CTX  log=$LOG"
                python3 -u "$EVAL_PY" \
                    --jobs-tsv             "$JOBS_TSV" \
                    --specs-dir            "$SPECS_DIR" \
                    --categories           "$CATEGORIES" \
                    --shard-idx            "$i" \
                    --shard-num            "$N_SHARDS" \
                    --ctx-id               "$CUR_CTX" \
                    --det-size             "$DET_SIZE" \
                    --sample-rate          "$SAMPLE_RATE" \
                    --min-face-px          "$MIN_FACE_PX" \
                    --min-det-score        "$MIN_DET_SCORE" \
                    --min-rel-area         "$MIN_REL_AREA" \
                    --max-gap-frames       "$MAX_GAP_FRAMES" \
                    --min-iou              "$MIN_IOU" \
                    --min-cos              "$MIN_COS" \
                    --min-tracklet-len     "$MIN_TRACKLET_LEN" \
                    --dbscan-eps           "$DBSCAN_EPS" \
                    --dbscan-min-samples   "$DBSCAN_MIN_SAMPLES" \
                    --primary-ratio-thresh "$PRIMARY_RATIO_THRESH" \
                    --min-primary-cluster-dets "$MIN_PRIMARY_CLUSTER_DETS" \
                    "${EXTRA_ARGS[@]}" \
                    > "$LOG" 2>&1 &
                pids+=($!)
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

# ============== 阶段 2: 跨模型汇总 ==============
if [[ "$RUN_AGGREGATE" == "1" ]]; then
    echo
    echo "=========================================================="
    echo "[run_all] aggregate_summary"
    echo "=========================================================="
    python3 "$AGG_PY" --runs-dir "$RUN_DIR" --specs-dir "$SPECS_DIR"
fi

echo
echo "=========================================================="
echo "[DONE] outputs under: $RUN_DIR"
echo "=========================================================="
