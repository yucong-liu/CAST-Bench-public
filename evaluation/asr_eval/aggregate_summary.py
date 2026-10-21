#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/asr_eval/aggregate_summary.py
==========================================

离线汇总脚本：扫 ``runs/<model>/wer_results*.json``，按
``details[].shot_count`` 字段分组重算 ``final_score`` / ``overall_wer`` /
``num_samples``，产出三份可直接跟其它评测点对齐的 compare_summary CSV：

    runs/compare_summary.csv          # 全样本 (all)
    runs/compare_summary_single.csv   # 仅 single_shot
    runs/compare_summary_multi.csv    # 仅 multi_shot

设计要点
--------
* **不重跑 ASR 模型，不用 GPU**。评分公式与 ``run_all.sh``/``test_wer_t2av.py``
  完全一致：

  - ``score  = max(0, 1 - min(wer, 1))``  (对每条样本，NaN → 0)
  - ``final_score = mean(score over samples in bucket)``
  - ``overall_wer = jiwer.wer(norm_refs_of_bucket, norm_hyps_of_bucket)``
    (对该桶内所有 ref / hyp 先 ``normalize`` 再整体求 WER)

* 输入优先级：
  1) ``runs/<m>/wer_results.json``     (SHOT_COUNT=all 跑出来的, 含所有样本)
  2) 若 1) 不存在，则 fallback 到
     ``runs/<m>/wer_results_single.json`` + ``runs/<m>/wer_results_multi.json``
     两个文件的 details 合并后再分桶。

  这样无论用户是 ``bash run_all.sh`` (老流程) 还是
  ``SHOT_COUNT=single/multi bash run_all.sh`` (两轮流程) 都能被兼容。

* Header 与 ``run_all.sh`` 内嵌的汇总保持一致::

    category, score_<m>..., wer_<m>..., n_<m>...

  ``category`` 列写 ``__ALL__``，与 TransNetV2 等评测点的语义对齐。

用法
----
默认从 ``models.tsv`` 里读模型列表 (顺序保持一致)，扫 ``runs/`` 下的
json，写三张表：

    python3 aggregate_summary.py

也可以显式指定：

    python3 aggregate_summary.py \\
        --run-dir /.../asr_eval/runs \\
        --models-tsv /.../evaluation/models.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 依赖：jiwer + 内联 normalize (与 test_wer.py 保持逐字节一致，但去掉 torch 依赖，
# 这样离线汇总可以在纯 CPU 登录节点/沙箱里跑)
#
# 注意：jiwer 只在计算 corpus-level ``overall_wer`` 时才需要。per-sample 的
# ``wer`` 已经在 wer_results.json 里存好了，``final_score`` 只依赖它，因此
# 即便 jiwer 未安装，脚本依然能产出正确的 ``final_score`` / ``n`` 列，
# 仅 ``wer_<m>`` 列会留空并给出 warning。
# ---------------------------------------------------------------------------
try:
    import jiwer  # type: ignore  # noqa: E402
    _HAS_JIWER = True
except Exception:  # pragma: no cover
    jiwer = None  # type: ignore
    _HAS_JIWER = False

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _is_chinese(text: str) -> bool:
    """Return True if the text contains a significant proportion of CJK characters."""
    cjk_count = len(_CJK_RE.findall(text))
    return cjk_count > len(text) * 0.2


def _strip_unicode_punct(text: str) -> str:
    """剥离所有 Unicode 标点 (类别以 P 开头)。"""
    return "".join(
        " " if unicodedata.category(c).startswith("P") else c for c in text
    )


def normalize(text: str, for_display: bool = False) -> str:
    """与 scripts/Qwen3-ASR/test_wer.py::normalize 完全一致。

    - Strip ALL Unicode punctuation
    - Lowercase
    - For Chinese: insert spaces between every character (char-level WER)
    - For English/其他语种: collapse whitespace (word-level WER)
    """
    text = _strip_unicode_punct(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    if not for_display and _is_chinese(text):
        text = " ".join(list(text.replace(" ", "")))
    return text


_THIS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
KIND_SINGLE = "single_shot"
KIND_MULTI = "multi_shot"
BUCKETS = ("all", "single", "multi")
BUCKET_TO_KIND = {"single": KIND_SINGLE, "multi": KIND_MULTI, "all": None}
BUCKET_TO_CSV = {
    "all": "compare_summary.csv",
    "single": "compare_summary_single.csv",
    "multi": "compare_summary_multi.csv",
}

DEFAULT_RUN_DIR = _THIS_DIR / "runs"
DEFAULT_MODELS_TSV = _THIS_DIR.parent / "models.tsv"


# ---------------------------------------------------------------------------
# 模型列表 / json 收集
# ---------------------------------------------------------------------------

def load_models(models_tsv: Path) -> List[str]:
    """从 models.tsv 里读模型名（第一列），保持顺序、忽略注释/空行。"""
    if not models_tsv.is_file():
        raise FileNotFoundError(f"models.tsv not found: {models_tsv}")
    models: List[str] = []
    with models_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            name = s.split("\t", 1)[0].strip()
            if name:
                models.append(name)
    return models


def load_details_for_model(run_dir: Path, model: str) -> List[Dict]:
    """收集一个模型的所有可用 detail 记录，按 id 去重（后写入的覆盖先写入的）。

    优先读 ``wer_results.json``；若不存在，则合并 ``wer_results_single.json``
    与 ``wer_results_multi.json``。
    """
    model_dir = run_dir / model
    if not model_dir.is_dir():
        return []

    candidates = [
        model_dir / "wer_results.json",
        model_dir / "wer_results_single.json",
        model_dir / "wer_results_multi.json",
    ]

    merged: Dict[str, Dict] = {}
    for jp in candidates:
        if not jp.is_file():
            continue
        try:
            with jp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[WARN] cannot read {jp}: {e}", file=sys.stderr)
            continue
        for d in data.get("details") or []:
            sid = d.get("id")
            if not sid:
                continue
            merged[sid] = d
    return list(merged.values())


# ---------------------------------------------------------------------------
# 分桶重算
# ---------------------------------------------------------------------------

def _safe_float(x) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return f


def compute_bucket_metrics(details: List[Dict], want_kind: Optional[str]
                           ) -> Tuple[Optional[float], Optional[float], int]:
    """在指定 shot_count 桶下重算 (final_score, overall_wer, n)。

    * ``want_kind`` = None 表示 "all"，不过滤。
    * 若桶内样本数为 0，返回 (None, None, 0)。
    * 严格复用 run_all.sh 的公式：
      - score  = max(0, 1 - min(wer, 1))；per-sample NaN → 0
      - final_score = mean(score)
      - overall_wer = jiwer.wer(norm_refs, norm_hyps) over the bucket
    """
    refs: List[str] = []
    hyps: List[str] = []
    per_sample_wer: List[float] = []

    for d in details:
        if want_kind is not None and d.get("shot_count") != want_kind:
            continue
        ref = d.get("ref_text") or ""
        hyp = d.get("hyp_text") or ""
        refs.append(normalize(ref))
        hyps.append(normalize(hyp))
        per_sample_wer.append(_safe_float(d.get("wer")))

    n = len(refs)
    if n == 0:
        return None, None, 0

    per_sample_score = [
        max(0.0, 1.0 - min(w, 1.0)) if w == w else 0.0  # NaN check: w != w
        for w in per_sample_wer
    ]
    final_score = sum(per_sample_score) / n

    # overall_wer: 用桶内合并后的 refs/hyps 整体算一次
    if not _HAS_JIWER:
        overall_wer = float("nan")
    else:
        try:
            # jiwer.wer 允许传两个等长 list
            overall_wer = float(jiwer.wer(refs, hyps))
        except Exception as e:
            print(f"[WARN] jiwer.wer failed on bucket (kind={want_kind}): {e}",
                  file=sys.stderr)
            overall_wer = float("nan")

    return final_score, overall_wer, n


# ---------------------------------------------------------------------------
# CSV 写入
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    return f"{v:.6f}"


def write_compare_csv(out_csv: Path, models: List[str],
                      results_by_model: Dict[str, Tuple[Optional[float], Optional[float], int]]):
    """写出一张 compare_summary*.csv，格式与 run_all.sh 内嵌的完全一致。"""
    header = (
        ["category"]
        + [f"score_{m}" for m in models]
        + [f"wer_{m}"   for m in models]
        + [f"n_{m}"     for m in models]
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        row: List[str] = ["__ALL__"]
        for m in models:  # score_*
            fs, _, _ = results_by_model.get(m, (None, None, 0))
            row.append(_fmt(fs))
        for m in models:  # wer_*
            _, ow, _ = results_by_model.get(m, (None, None, 0))
            row.append(_fmt(ow))
        for m in models:  # n_*
            _, _, n = results_by_model.get(m, (None, None, 0))
            row.append(str(n) if n else "")
        w.writerow(row)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def aggregate(run_dir: Path, models_tsv: Path, verbose: bool = True) -> Dict[str, Path]:
    models = load_models(models_tsv)
    if verbose:
        print(f"[aggregate] run_dir    = {run_dir}")
        print(f"[aggregate] models_tsv = {models_tsv}")
        print(f"[aggregate] models     = {models}")
        if not _HAS_JIWER:
            print("[aggregate] [WARN] jiwer 未安装：wer_<m> 列将留空 (per-sample "
                  "wer 已经在 wer_results.json 里, final_score / n 仍会正确产出)")

    # 1) 一次性把所有模型的 details 读进来
    details_by_model: Dict[str, List[Dict]] = {}
    for m in models:
        details = load_details_for_model(run_dir, m)
        details_by_model[m] = details
        if verbose:
            n_all = len(details)
            n_s = sum(1 for d in details if d.get("shot_count") == KIND_SINGLE)
            n_m = sum(1 for d in details if d.get("shot_count") == KIND_MULTI)
            n_u = n_all - n_s - n_m
            print(f"  {m:<22}  loaded {n_all:>3} details  "
                  f"(single={n_s}, multi={n_m}, other/unknown={n_u})")

    # 2) 分桶重算 + 写 csv
    out_paths: Dict[str, Path] = {}
    for bucket in BUCKETS:
        want_kind = BUCKET_TO_KIND[bucket]
        results_by_model: Dict[str, Tuple[Optional[float], Optional[float], int]] = {}
        for m in models:
            results_by_model[m] = compute_bucket_metrics(
                details_by_model[m], want_kind
            )

        out_csv = run_dir / BUCKET_TO_CSV[bucket]
        write_compare_csv(out_csv, models, results_by_model)
        out_paths[bucket] = out_csv

        if verbose:
            print()
            print("=" * 72)
            print(f"[aggregate] bucket = {bucket!r}  ->  {out_csv}")
            print("=" * 72)
            for m in models:
                fs, ow, n = results_by_model[m]
                if n == 0:
                    print(f"  {m:<22}  (no samples)")
                else:
                    print(f"  {m:<22}  final_score={_fmt(fs):>8}  "
                          f"overall_wer={_fmt(ow):>8}  n={n}")

    return out_paths


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline aggregate compare_summary{,_single,_multi}.csv for asr_eval."
    )
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR),
                    help=f"asr_eval runs 目录 (默认: {DEFAULT_RUN_DIR})")
    ap.add_argument("--models-tsv", default=str(DEFAULT_MODELS_TSV),
                    help=f"模型清单 tsv (默认: {DEFAULT_MODELS_TSV})")
    ap.add_argument("--quiet", action="store_true", help="减少输出")
    args = ap.parse_args()

    aggregate(
        run_dir=Path(args.run_dir).expanduser().resolve(),
        models_tsv=Path(args.models_tsv).expanduser().resolve(),
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
