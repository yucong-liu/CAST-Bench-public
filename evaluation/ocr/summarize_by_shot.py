#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr · summarize_by_shot.py
==========================

按 shot_count (single_shot / multi_shot) 汇总 6 个模型的 OCR 参考式打分结果.

汇总表 (每行 = 一个模型 + 结尾 __ALL__ 汇总), 只输出 3 张:
    <out-dir>/compare_summary_ocr.csv
    <out-dir>/compare_summary_ocr_single.csv
    <out-dir>/compare_summary_ocr_multi.csv
  列: model, n, mean_score, mean_match_rate, n_valid, mean_score_valid, mean_match_rate_valid
    - n:                   该档下评测过的样本总数 (含 error)
    - mean_score:           所有样本 score 的平均 (error 视为 0.0, 惩罚模型没抽出可评估文本)
    - mean_match_rate:      所有样本 match_rate 的平均 (error 视为 0.0)
    - n_valid:              error 为 null 的样本数
    - mean_score_valid:     仅 error 为 null 的样本 score 平均
    - mean_match_rate_valid:仅 error 为 null 的样本 match_rate 平均

stdout 打印三张对齐表: OVERALL / SINGLE-SHOT / MULTI-SHOT.

输入:
  --score-root <dir>     score_reference.py 的产物根目录, 内含 <model>/<vid>.json
  --ocr-specs  <path>    ocr_specs.jsonl (从中读取每个 vid 的 shot_count)
  --models     "A,B,C"   参与汇总的模型名列表 (逗号分隔; 排序时优先前置商业模型)
  --out-dir    <dir>     3 张 CSV 的输出目录
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
KIND_SINGLE = "single_shot"
KIND_MULTI = "multi_shot"
KIND_UNKNOWN = ""

# 与 voice_consistency 对齐: 商业模型固定排在前, 其余按字母序追加
_MODEL_ORDER_PREFIX = ["seedance", "wan2.7", "kling-v3"]

COMPARE_COLS = [
    "model",
    "n",
    "mean_score",
    "mean_match_rate",
    "n_valid",
    "mean_score_valid",
    "mean_match_rate_valid",
]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_shot_map(specs_path: Path) -> Dict[str, str]:
    """id -> shot_count (single_shot / multi_shot / other)"""
    m: Dict[str, str] = {}
    with specs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            vid = d.get("id")
            sc = d.get("shot_count") or ""
            if vid:
                m[vid] = sc
    return m


def _flat_row(d: Dict[str, Any], model: str, spec_id: str,
              shot_kind: str) -> Dict[str, Any]:
    """把 score_reference.py 输出的单个 json 摊平成明细行."""
    sub = d.get("sub_scores") or {}
    meta = d.get("meta") or {}
    err = d.get("error", None)
    return {
        "model": model,
        "spec_id": spec_id,
        "shot_count_kind": shot_kind,
        "category": d.get("category", ""),
        "score": d.get("score"),
        "match_rate": sub.get("match_rate"),
        "total_matched": sub.get("total_matched"),
        "total_expected": sub.get("total_expected"),
        "num_distinct_texts": sub.get("num_distinct_texts"),
        "num_frames_with_text": (
            sub.get("num_frames_with_text")
            if sub.get("num_frames_with_text") is not None
            else meta.get("num_frames_with_text")
        ),
        "error": err,
    }


def load_all_rows(score_root: Path, models: List[str],
                  shot_map: Dict[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in models:
        mdir = score_root / m
        if not mdir.is_dir():
            print(f"[shot_summary][WARN] model dir missing: {mdir}",
                  file=sys.stderr)
            continue
        n_before = len(rows)
        for jf in sorted(mdir.glob("*.json")):
            try:
                with jf.open("r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception as e:
                print(f"[shot_summary][WARN] fail to parse {jf}: {e}",
                      file=sys.stderr)
                continue
            if not isinstance(d, dict):
                continue
            spec_id = d.get("video_id") or jf.stem
            shot_kind = shot_map.get(spec_id, KIND_UNKNOWN)
            rows.append(_flat_row(d, m, spec_id, shot_kind))
        print(f"[shot_summary] scan {mdir}: +{len(rows) - n_before} rows")
    return rows


# ---------------------------------------------------------------------------
# 排序 & 汇总
# ---------------------------------------------------------------------------
def _order_models(models: List[str]) -> List[str]:
    head = [m for m in _MODEL_ORDER_PREFIX if m in models]
    tail = sorted(m for m in models if m not in _MODEL_ORDER_PREFIX)
    return head + tail


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _agg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """一个模型 (或全体) 的汇总统计.

    mean_* 对 error 样本按 0.0 计入 (惩罚); mean_*_valid 只算 error is None 的样本.
    """
    scores_all: List[float] = []
    mrs_all: List[float] = []
    scores_valid: List[float] = []
    mrs_valid: List[float] = []

    for r in rs:
        err = r.get("error")
        s = _num(r.get("score"))
        mr = _num(r.get("match_rate"))
        # 全量 (含 error): 缺失就补 0.0, 让"没跑成"也计入惩罚
        scores_all.append(s if s is not None else 0.0)
        mrs_all.append(mr if mr is not None else 0.0)
        if err is None:
            if s is not None:
                scores_valid.append(s)
            if mr is not None:
                mrs_valid.append(mr)

    def _mean(vs: List[float]) -> Optional[float]:
        return round(statistics.fmean(vs), 4) if vs else None

    return {
        "n": len(rs),
        "mean_score": _mean(scores_all),
        "mean_match_rate": _mean(mrs_all),
        "n_valid": len(scores_valid),
        "mean_score_valid": _mean(scores_valid),
        "mean_match_rate_valid": _mean(mrs_valid),
    }


# ---------------------------------------------------------------------------
# 写 CSV
# ---------------------------------------------------------------------------
def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_compare_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = _order_models(sorted({r["model"] for r in rows}))

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COMPARE_COLS)
        for m in models:
            rs = [r for r in rows if r["model"] == m]
            a = _agg(rs)
            w.writerow([
                m, a["n"],
                _fmt_cell(a["mean_score"]),
                _fmt_cell(a["mean_match_rate"]),
                a["n_valid"],
                _fmt_cell(a["mean_score_valid"]),
                _fmt_cell(a["mean_match_rate_valid"]),
            ])
        # __ALL__ 综合平均 (跨模型的样本一锅平均)
        a_all = _agg(rows)
        w.writerow([
            "__ALL__", a_all["n"],
            _fmt_cell(a_all["mean_score"]),
            _fmt_cell(a_all["mean_match_rate"]),
            a_all["n_valid"],
            _fmt_cell(a_all["mean_score_valid"]),
            _fmt_cell(a_all["mean_match_rate_valid"]),
        ])
    print(f"[shot_summary] wrote {path}  ({len(models)} models)")


# ---------------------------------------------------------------------------
# 打印对齐表
# ---------------------------------------------------------------------------
def _print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n=== {title}  (n={len(rows)}) ===")
    if not rows:
        return
    models = _order_models(sorted({r["model"] for r in rows}))
    header = ("model", "n", "mean_score", "mean_match", "n_valid",
              "mean_s_valid", "mean_mr_valid")
    print(f"{header[0]:<20} {header[1]:>4}  {header[2]:>10}  "
          f"{header[3]:>10}  {header[4]:>7}  {header[5]:>12}  {header[6]:>13}")
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        a = _agg(rs)
        ms  = f"{a['mean_score']:.4f}"           if a["mean_score"] is not None else "None"
        mmr = f"{a['mean_match_rate']:.4f}"      if a["mean_match_rate"] is not None else "None"
        msv = f"{a['mean_score_valid']:.4f}"     if a["mean_score_valid"] is not None else "None"
        mmv = f"{a['mean_match_rate_valid']:.4f}" if a["mean_match_rate_valid"] is not None else "None"
        print(f"{m:<20} {a['n']:>4}  {ms:>10}  {mmr:>10}  "
              f"{a['n_valid']:>7}  {msv:>12}  {mmv:>13}")
    # 结尾 __ALL__
    a_all = _agg(rows)
    ms  = f"{a_all['mean_score']:.4f}"           if a_all["mean_score"] is not None else "None"
    mmr = f"{a_all['mean_match_rate']:.4f}"      if a_all["mean_match_rate"] is not None else "None"
    msv = f"{a_all['mean_score_valid']:.4f}"     if a_all["mean_score_valid"] is not None else "None"
    mmv = f"{a_all['mean_match_rate_valid']:.4f}" if a_all["mean_match_rate_valid"] is not None else "None"
    print(f"{'__ALL__':<20} {a_all['n']:>4}  {ms:>10}  {mmr:>10}  "
          f"{a_all['n_valid']:>7}  {msv:>12}  {mmv:>13}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-root", required=True,
                    help="score_reference.py 输出根目录 (含 <model>/<vid>.json)")
    ap.add_argument("--ocr-specs", required=True,
                    help="ocr_specs.jsonl (提供 shot_count)")
    ap.add_argument("--models", required=True,
                    help="逗号分隔模型名列表 (决定扫哪些 <model>/ 子目录)")
    ap.add_argument("--out-dir", required=True,
                    help="输出目录; 生成 compare_summary_ocr{,_single,_multi}.csv 共 3 个 CSV")
    args = ap.parse_args()

    score_root = Path(args.score_root)
    ocr_specs = Path(args.ocr_specs)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载 shot_map
    shot_map = load_shot_map(ocr_specs)
    print(f"[shot_summary] loaded shot_map: {len(shot_map)} specs "
          f"(single={sum(1 for v in shot_map.values() if v == KIND_SINGLE)}, "
          f"multi={sum(1 for v in shot_map.values() if v == KIND_MULTI)})")

    # 2) 扫全部明细行
    rows = load_all_rows(score_root, models, shot_map)
    if not rows:
        print("[shot_summary][ERROR] no per-video json found under any model dir",
              file=sys.stderr)
        return 1

    single_rows  = [r for r in rows if r["shot_count_kind"] == KIND_SINGLE]
    multi_rows   = [r for r in rows if r["shot_count_kind"] == KIND_MULTI]
    unknown_rows = [r for r in rows if r["shot_count_kind"] not in (KIND_SINGLE, KIND_MULTI)]
    if unknown_rows:
        uniq_ids = sorted({r["spec_id"] for r in unknown_rows})
        print(
            f"[shot_summary][WARN] {len(unknown_rows)} rows ({len(uniq_ids)} spec_ids) "
            f"have unknown shot_count_kind and will only appear in the ALL csv: "
            f"{uniq_ids[:10]}{' ...' if len(uniq_ids) > 10 else ''}",
            file=sys.stderr,
        )

    # 3) 写 3 张汇总 CSV: all / single / multi
    cmp_all    = out_dir / "compare_summary_ocr.csv"
    cmp_single = out_dir / "compare_summary_ocr_single.csv"
    cmp_multi  = out_dir / "compare_summary_ocr_multi.csv"

    _write_compare_csv(cmp_all,    rows)
    _write_compare_csv(cmp_single, single_rows)
    _write_compare_csv(cmp_multi,  multi_rows)

    # 4) stdout 打印三张对齐表
    _print_table("OVERALL", rows)
    _print_table("SINGLE-SHOT (specs.shot_count=single_shot)", single_rows)
    _print_table("MULTI-SHOT  (specs.shot_count=multi_shot)",  multi_rows)
    if unknown_rows:
        _print_table("UNKNOWN-KIND (not in specs)", unknown_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
