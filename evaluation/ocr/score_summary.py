#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr · score_summary.py

对 score_out/qwen/<model>/<vid>.json 做二次汇总:
  - per-category × per-model 的均值 (overall + 4 个 subscore)
  - 落盘:
      score_out/qwen/compare_summary.csv     跨 model 横向 overall_avg
      score_out/qwen/per_category.csv        category × model 的 overall_avg

跟 judge_qwen.py 自带的 judge_summary.csv 互补.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "score_out" / "qwen"
DEFAULT_MODELS = ["LTX-2.3", "Ovi", "daVinci-MagiHuman", "seedance", "wan2.7"]


def avg(lst: List[float]) -> Optional[float]:
    return round(statistics.mean(lst), 2) if lst else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="judge_qwen.py 的 --out-dir, 内含 <model>/<vid>.json")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.out_dir.is_dir():
        print(f"[ERROR] out-dir 不存在: {args.out_dir}", file=sys.stderr)
        return 2

    # model -> {category -> [overall_scores]}, 同时保存 4 个 subscore
    by_model_cat: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for m in models:
        mdir = args.out_dir / m
        if not mdir.is_dir():
            print(f"[warn] 模型目录不存在: {mdir}", file=sys.stderr)
            continue
        for f in sorted(mdir.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("judge_failed"):
                continue
            ov = rec.get("overall_text_quality_score")
            if not isinstance(ov, int):
                continue
            cat = rec.get("category") or rec.get("video_id", "").split("_", 1)[0] or "unknown"
            subs = rec.get("subscores") or {}
            slot = by_model_cat.setdefault(m, {}).setdefault(cat, {
                "overall": [], "legibility": [], "temporal": [], "spatial": [], "completeness": [],
            })
            slot["overall"].append(ov)
            for k, key in (("legibility", "legibility_accuracy"),
                           ("temporal", "temporal_stability"),
                           ("spatial", "spatial_stability"),
                           ("completeness", "completeness")):
                v = subs.get(key)
                if isinstance(v, int):
                    slot[k].append(v)

    # ---- compare_summary.csv: 跨 model 横向 (含全量 + 各 subscore) ----
    cs = args.out_dir / "compare_summary.csv"
    with cs.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["model", "n_all",
                    "overall_avg", "legibility_avg", "temporal_avg",
                    "spatial_avg", "completeness_avg"])
        for m in models:
            agg = {"overall": [], "legibility": [], "temporal": [],
                   "spatial": [], "completeness": []}
            for cat, slot in (by_model_cat.get(m) or {}).items():
                for k in agg:
                    agg[k].extend(slot[k])
            w.writerow([
                m, len(agg["overall"]),
                avg(agg["overall"]), avg(agg["legibility"]),
                avg(agg["temporal"]), avg(agg["spatial"]),
                avg(agg["completeness"]),
            ])
    print(f"[summary] wrote {cs}")

    # ---- per_category.csv: category 行 × model 列 (overall_avg) ----
    all_cats = sorted({cat for d in by_model_cat.values() for cat in d.keys()})
    pc = args.out_dir / "per_category.csv"
    with pc.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["category"] + [f"{m}_overall" for m in models]
                   + [f"{m}_n" for m in models])
        for cat in all_cats:
            row = [cat]
            for m in models:
                slot = (by_model_cat.get(m) or {}).get(cat) or {"overall": []}
                row.append(avg(slot["overall"]))
            for m in models:
                slot = (by_model_cat.get(m) or {}).get(cat) or {"overall": []}
                row.append(len(slot["overall"]))
            w.writerow(row)
    print(f"[summary] wrote {pc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
