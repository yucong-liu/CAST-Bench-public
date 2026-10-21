#!/usr/bin/env bash
# =====================================================================
# T2AV-Bench Speech 子集 · ASR (Qwen3-ASR-1.7B) 批量评测
#
# 输出布局对齐 ../audiobox_runs / ../qalign_runs：
#   asr_eval/runs/
#     ├── LTX-2.3/wer_results.json
#     ├── Ovi/wer_results.json
#     ├── daVinci-MagiHuman/wer_results.json
#     ├── compare_summary.csv           # 全样本 (all)
#     ├── compare_summary_single.csv    # 仅 single_shot   (新增, 自动产出)
#     └── compare_summary_multi.csv     # 仅 multi_shot    (新增, 自动产出)
#
# 三张表由 aggregate_summary.py 在跑完模型后统一生成 (只读 details, 不上 GPU),
# single / multi 的划分约定见 evaluation/README.md §4.
#
# 想新增模型 = MODELS 里追加 + src_for() 里加一个 case
# =====================================================================
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# ---------------- 用户可调参数 ----------------
ASR_EVAL_ROOT=$REPO_ROOT/evaluation/asr_eval
RUN_DIR=$ASR_EVAL_ROOT/runs
SPEC_JSONL=$REPO_ROOT/t2av-bench/specs/speech.jsonl
MODEL_PATH=$ASR_EVAL_ROOT/models/Qwen3-ASR-1.7B
SCRIPT_DIR=$ASR_EVAL_ROOT/scripts/Qwen3-ASR
DEVICE=cuda:0
BATCH_SIZE=4
# 评测样本范围（按 spec.shot_count 过滤）：single | multi | all
# 默认 all，即 single_shot + multi_shot 全评。
# multi_shot 的 ref 来自 spec.subjects[].speech（已合并各分镜台词），与 ASR 整段转写直接对齐；
# 标点已在 normalize() 里全部 strip，对 multi_shot 同样适用。
#
# ★ 无论 SHOT_COUNT 取何值, 跑完之后都会自动调用
#   aggregate_summary.py 按 details[].shot_count 分桶产出三张表:
#     runs/compare_summary.csv          (全样本)
#     runs/compare_summary_single.csv   (仅 single_shot)
#     runs/compare_summary_multi.csv    (仅 multi_shot)
#   与 TransNetV2 / audio_consistency 等评测点的输出布局对齐 (见 evaluation/README.md §4).
#
# 因此推荐直接:
#   bash run_all.sh                     # -> 一次跑 all, 直接得到三张表 (最省心)
#
# 如果希望完全隔离 single / multi 两轮 ASR (例如避免 all 里做过滤),
# 也可以分两次跑:
#   SHOT_COUNT=single bash run_all.sh   # -> runs/<m>/wer_results_single.json
#   SHOT_COUNT=multi  bash run_all.sh   # -> runs/<m>/wer_results_multi.json
#   (每一次结束都会再 aggregate 一次, 三张表会被增量刷新)
#
# 开源模型 (LTX-2.3 / Ovi / daVinci-MagiHuman) 不生成多镜头视频, 所以
# 在 multi 桶里它们的 score/wer/n 列会留空, 这是预期行为.
SHOT_COUNT="${SHOT_COUNT:-all}"

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
load_models

# 保留完整模型清单副本, 供最后 aggregate_summary.py 汇总时使用
# (ONLY_MODELS 追加模式下, MODELS 会被裁剪, 但 ALL_MODELS 始终保持全量)
ALL_MODELS=("${MODELS[@]}")

# ---- ONLY_MODELS 追加模式: 只跑白名单里的模型 (逗号或空格分隔) ----
# 用法:
#   ONLY_MODELS="kling-v3-omni,veo3.1,happyhorse-1.1" bash run_all.sh
# 语义:
#   - 阶段1 (ASR 推理) 只处理这些模型;
#   - 阶段2 (内嵌 python 汇总) 在追加模式下跳过, 避免把老模型的列写空;
#   - 阶段3 (aggregate_summary.py) 天然扫全部模型的 wer_results.json,
#     追加模式和全量模式产物一致 (老模型 wer_results.json 未变, 结果不受影响).
#   - 拼错模型名 (models.tsv 中不存在) 会报错退出.
ONLY_MODELS="${ONLY_MODELS:-}"
if [[ -n "$ONLY_MODELS" ]]; then
    ONLY_MODELS_NORM="${ONLY_MODELS//,/ }"
    declare -A _KNOWN_MODELS=()
    for _m in "${ALL_MODELS[@]}"; do _KNOWN_MODELS["$_m"]=1; done
    FILTERED=()
    for _m in $ONLY_MODELS_NORM; do
        [[ -z "$_m" ]] && continue
        if [[ -z "${_KNOWN_MODELS[$_m]:-}" ]]; then
            echo "[ERROR] ONLY_MODELS 中的模型不在 models.tsv: $_m" >&2
            echo "[ERROR] models.tsv 已注册: ${ALL_MODELS[*]}" >&2
            exit 1
        fi
        FILTERED+=("$_m")
    done
    if [[ ${#FILTERED[@]} -eq 0 ]]; then
        echo "[ERROR] ONLY_MODELS 过滤后为空" >&2
        exit 1
    fi
    MODELS=("${FILTERED[@]}")
    echo "[run_all] ONLY_MODELS 追加模式生效, MODELS = ${MODELS[*]}"
    echo "[run_all] (阶段3 aggregate_summary.py 仍会汇总全部 ${#ALL_MODELS[@]} 个模型)"
fi

# 模型名 -> mp4 所在的 speech 子目录
# models.tsv 给的是 output 根；这里自动兜底寻找 speech 子目录:
#   1) <root>/speech
#   2) <root>/<任意中间层>/speech (例如 daVinci-MagiHuman: output/540p/speech)
src_for() {
    local root="${SRC_MAP[$1]:-}"
    [[ -z "$root" ]] && { echo ""; return; }
    if [[ -d "$root/speech" ]]; then
        echo "$root/speech"; return
    fi
    local cand
    cand=$(find "$root" -mindepth 2 -maxdepth 3 -type d -name speech 2>/dev/null | head -n 1)
    [[ -n "$cand" ]] && { echo "$cand"; return; }
    echo "$root/speech"   # fallback: 让下游打印 not-found 提示
}
# ----------------------------------------------

mkdir -p "$RUN_DIR"

# 把 MODELS 序列化成 python 能解析的形式：每行  MODEL\tSRC
SPEC_FOR_PY=""
for M in "${MODELS[@]}"; do
    S=$(src_for "$M")
    SPEC_FOR_PY+="$M"$'\t'"$S"$'\n'
done

# ============== 主体：单进程加载一次 ASR 模型 → 串行评 3 个模型 ==============
# 用 env 把变量传进 python；这里不用 here-doc 替换变量（避免 bash 报警）
export RUN_DIR SPEC_JSONL MODEL_PATH SCRIPT_DIR DEVICE BATCH_SIZE SHOT_COUNT
export ASR_MODEL_LIST="$SPEC_FOR_PY"
export ASR_ONLY_MODE="${ONLY_MODELS:+1}"

python3 - <<'PY'
import os, sys, json, csv, time

run_dir    = os.environ["RUN_DIR"]
spec_jsonl = os.environ["SPEC_JSONL"]
model_path = os.environ["MODEL_PATH"]
script_dir = os.environ["SCRIPT_DIR"]
device     = os.environ["DEVICE"]
batch_size = int(os.environ["BATCH_SIZE"])
shot_count = os.environ.get("SHOT_COUNT", "single")
model_list = [
    line.split("\t", 1)
    for line in os.environ["ASR_MODEL_LIST"].strip().splitlines()
    if line.strip()
]

# 把 Qwen3-ASR 脚本目录加进 sys.path 才能 import test_wer_t2av
sys.path.insert(0, script_dir)
from test_wer_t2av import load_asr_model, run_one_model  # noqa: E402

# 加载一次模型，三模型共享
print("[run_all] loading ASR model once ...")
print(f"[run_all] shot_count filter = {shot_count}")
t0 = time.time()
asr = load_asr_model(model_path, device, batch_size)
print(f"[run_all] model loaded in {time.time()-t0:.1f}s")

results = {}
for model_name, src_dir in model_list:
    print(f"\n############ {model_name} ############")
    if not src_dir or not os.path.isdir(src_dir):
        print(f"[run_all] {model_name}: src '{src_dir}' not found, skip")
        continue

    out_dir   = os.path.join(run_dir, model_name)
    # 当 shot_count=single/multi 时, 把 wer_results.json 也加后缀,
    # 否则两次跑 (single 和 multi) 会互相覆盖结果.
    if shot_count == "single":
        out_json = os.path.join(out_dir, "wer_results_single.json")
    elif shot_count == "multi":
        out_json = os.path.join(out_dir, "wer_results_multi.json")
    else:
        out_json = os.path.join(out_dir, "wer_results.json")
    audio_dir = os.path.join(out_dir, "audio_extracted")
    os.makedirs(out_dir, exist_ok=True)

    res = run_one_model(
        asr,
        video_dir   = src_dir,
        spec_jsonl  = spec_jsonl,
        audio_dir   = audio_dir,
        out_json    = out_json,
        batch_size  = batch_size,
        auto_language = False,
        tag         = model_name,
        shot_count  = shot_count,
    )
    if res is not None:
        results[model_name] = res

# ============== 汇总 compare_summary.csv ==============
# 输出文件名跟随 SHOT_COUNT (single -> compare_summary_single.csv,
# multi -> compare_summary_multi.csv, all -> compare_summary.csv).
# 用法:
#   SHOT_COUNT=single bash run_all.sh   # 只跑单镜头并写到 _single 表
#   SHOT_COUNT=multi  bash run_all.sh   # 只跑多镜头并写到 _multi  表
#   bash run_all.sh                     # 默认 all, 写到 compare_summary.csv (历史行为)
#
# ★ ONLY_MODELS 追加模式下, 这里的内嵌汇总只覆盖本轮跑的模型, 会把老模型列写空;
#   因此追加模式直接跳过这段, 完全依赖阶段3的 aggregate_summary.py (它会独立扫
#   全部模型的 wer_results.json, 老模型 CSV 列不会丢失).
if os.environ.get("ASR_ONLY_MODE") == "1":
    print("[run_all] ONLY_MODELS 追加模式: 跳过内嵌汇总, 完全依赖 aggregate_summary.py")
    for m, _ in model_list:
        r = results.get(m, {})
        if r:
            print(f"  {m:<20}  final_score = {r['final_score']:.4f}   overall_wer = {r['overall_wer']:.4f}   n = {r['num_samples']}")
        else:
            print(f"  {m:<20}  (no result)")
    raise SystemExit(0)

if shot_count == "single":
    sum_basename = "compare_summary_single.csv"
elif shot_count == "multi":
    sum_basename = "compare_summary_multi.csv"
else:
    sum_basename = "compare_summary.csv"

sum_csv = os.path.join(run_dir, sum_basename)
header = (
    ["category"]
    + [f"score_{m}" for m, _ in model_list]
    + [f"wer_{m}"   for m, _ in model_list]
    + [f"n_{m}"     for m, _ in model_list]
)
with open(sum_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    row = ["__ALL__"]
    for m, _ in model_list:                       # score_*
        v = results.get(m, {}).get("final_score", "")
        row.append(f"{v:.6f}" if isinstance(v, float) else "")
    for m, _ in model_list:                       # wer_*  (新增展示)
        v = results.get(m, {}).get("overall_wer", "")
        row.append(f"{v:.6f}" if isinstance(v, float) else "")
    for m, _ in model_list:                       # n_*
        n = results.get(m, {}).get("num_samples", "")
        row.append(n)
    w.writerow(row)

print()
print("=" * 60)
print(f"[run_all] compare_summary written to: {sum_csv}")
print("=" * 60)
for m, _ in model_list:
    r = results.get(m, {})
    if r:
        print(f"  {m:<20}  final_score = {r['final_score']:.4f}   overall_wer = {r['overall_wer']:.4f}   n = {r['num_samples']}")
    else:
        print(f"  {m:<20}  (no result)")
PY

# ============== 离线汇总：按 shot_count 拆三张表 ==============
# 无论 SHOT_COUNT 是 all / single / multi, 只要 runs/<m>/wer_results*.json
# 里有 details, aggregate_summary.py 就会读回来按 details[].shot_count 分组,
# 覆盖式写出:
#   runs/compare_summary.csv          (全样本, 与历史 all 表兼容)
#   runs/compare_summary_single.csv   (仅 single_shot)
#   runs/compare_summary_multi.csv    (仅 multi_shot)
# 这样一次 `bash run_all.sh` 就能直接产出三张表, 与 TransNetV2 / audio_consistency
# 等评测点的输出布局对齐 (见 evaluation/README.md §4).
echo
echo "[run_all] aggregating compare_summary{,_single,_multi}.csv ..."
python3 "$ASR_EVAL_ROOT/aggregate_summary.py" \
    --run-dir "$RUN_DIR" \
    --models-tsv "$MODELS_TSV" || {
    echo "[WARN] aggregate_summary.py failed (non-fatal); 保留主流程已写入的老格式 compare_summary CSV" >&2
}

echo
echo "=========================================================="
echo "[DONE] all models finished. outputs under: $RUN_DIR"
echo "=========================================================="
