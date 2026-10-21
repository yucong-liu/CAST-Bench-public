#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
facial_consistency · aggregate_summary.py
=========================================

跨模型汇总. 自动 listdir(<runs_dir>) 发现各模型, 读
<runs_dir>/<model>/per_video.jsonl + summary.json, 输出:

    <runs_dir>/compare_summary.csv          # 全部 spec
    <runs_dir>/compare_summary_single.csv   # 仅 single_shot spec
    <runs_dir>/compare_summary_multi.csv    # 仅 multi_shot spec

每份 CSV 行布局:
    第 1..N 行: 单 category (按字典序), 9 个 L1 桶 + 实际遇到的桶
    末尾行  : __ALL__ (跨所有 category 的统计)

列布局:
    category,
    score_<m>... ...        平均 final_score (0-100, 该 category 内 status 落在 SCORED_STATUSES)
    s_count_<m>... ...      平均 s_count (0-1)
    s_stab_<m>... ...       平均 s_stability (0-1)
    n_specs_<m>... ...      参与平均的 spec 数
    n_total_<m>... ...      该 category 的 spec 总数 (含 missing / failed / expected=0)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]


# 与 batch_eval / merge_shards 保持一致: 这些状态会出 final_score 并参与平均
SCORED_STATUSES = {"ok", "ok_no_faces", "no_faces_detected", "no_tracklets", "no_clusters"}


def discover_models(runs_dir: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(runs_dir)):
        sub = os.path.join(runs_dir, name)
        if not os.path.isdir(sub):
            continue
        if name.startswith(("_", ".")):
            continue
        if os.path.isfile(os.path.join(sub, "per_video.jsonl")) \
                or os.path.isfile(os.path.join(sub, "summary.json")):
            out.append(name)
    return out


def load_per_video(runs_dir: str, model: str) -> List[dict]:
    p = os.path.join(runs_dir, model, "per_video.jsonl")
    if not os.path.isfile(p):
        return []
    out: List[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def aggregate_one(records: List[dict]) -> Dict[str, dict]:
    """{category_or_'__ALL__': stats}"""
    bucket: Dict[str, dict] = defaultdict(lambda: {
        "n_total": 0,        # 该 category 进入 per_video 的总条数 (包括 missing / failed)
        "n_scored": 0,       # status in SCORED_STATUSES 的条数
        "score_sum": 0.0,
        "s_count_sum": 0.0,
        "s_stab_sum": 0.0,
        "spec_set": set(),
    })
    for r in records:
        cat = r.get("category") or "unknown"
        sid = r.get("spec_id") or ""
        for key in ("__ALL__", cat):
            if not key:
                continue
            b = bucket[key]
            b["n_total"] += 1
            if sid:
                b["spec_set"].add(sid)
            if r.get("status") in SCORED_STATUSES:
                b["n_scored"] += 1
                try:
                    b["score_sum"] += float(r.get("final_score", 0.0) or 0.0)
                except Exception:
                    pass
                br = r.get("breakdown") or {}
                try:
                    b["s_count_sum"] += float(br.get("s_count", 0.0) or 0.0)
                except Exception:
                    pass
                try:
                    b["s_stab_sum"] += float(br.get("s_stability", 0.0) or 0.0)
                except Exception:
                    pass
    return bucket


def _fmt_avg(num: float, den: int, fmt: str) -> str:
    if den <= 0:
        return ""
    return fmt.format(num / den)


def _update_model_summary(runs_dir: str, model: str, b_all: dict) -> None:
    """把 __ALL__ 的 mean_score / mean_s_count / mean_s_stability 写回 model/summary.json."""
    p = os.path.join(runs_dir, model, "summary.json")
    if not os.path.isfile(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    n = int(b_all.get("n_scored", 0))
    if n > 0:
        data["mean_score"] = round(b_all.get("score_sum", 0.0) / n, 4)
        data["mean_s_count"] = round(b_all.get("s_count_sum", 0.0) / n, 4)
        data["mean_s_stability"] = round(b_all.get("s_stab_sum", 0.0) / n, 4)
    data["n_specs_with_scored"] = len(b_all.get("spec_set") or set())

    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True,
                    help="如 evaluation/facial_consistency/runs")
    ap.add_argument("--specs-dir", default=os.environ.get(
        "T2AV_SPECS_DIR",
        str(_REPO_ROOT / "t2av-bench/specs"),
    ), help="t2av-bench specs 目录, 用于按 shot_count 拆 single/multi 子表")
    args = ap.parse_args()

    runs_dir = args.runs_dir
    if not os.path.isdir(runs_dir):
        print(f"[ERROR] runs dir not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    models = discover_models(runs_dir)
    if not models:
        print(f"[WARN] no model under {runs_dir}")
        return

    # 加载 spec_id -> shot_count_kind, 反查
    _here = os.path.dirname(os.path.abspath(__file__))
    _common = os.path.normpath(os.path.join(_here, "..", "_common"))
    if _common not in sys.path:
        sys.path.insert(0, _common)
    try:
        from spec_kind import load_spec_kind_map  # noqa: E402
        spec_kind_map = load_spec_kind_map(args.specs_dir, verbose=True)
    except Exception as e:
        print(f"[WARN] load_spec_kind_map failed: {e}; will only emit full compare_summary.csv",
              file=sys.stderr)
        spec_kind_map = {}

    per_v_full = {m: load_per_video(runs_dir, m) for m in models}

    def _filter_by_kind(records, kind):
        if kind is None:
            return records
        return [r for r in records
                if spec_kind_map.get(r.get("spec_id") or "", "") == kind]

    def _aggregate_one_variant(per_v, out_csv, write_back_summary, label):
        stats = {m: aggregate_one(per_v[m]) for m in models}

        cats = sorted({c for m in models for c in stats[m].keys()
                       if c and c != "__ALL__"})
        rows_order = cats + ["__ALL__"]

        header = ["category"]
        for m in models: header.append(f"score_{m}")
        for m in models: header.append(f"s_count_{m}")
        for m in models: header.append(f"s_stab_{m}")
        for m in models: header.append(f"n_specs_{m}")
        for m in models: header.append(f"n_total_{m}")

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for cat in rows_order:
                row = [cat]
                for m in models:
                    b = stats[m].get(cat) or {}
                    row.append(_fmt_avg(b.get("score_sum", 0.0),
                                        int(b.get("n_scored", 0) or 0),
                                        "{:.4f}"))
                for m in models:
                    b = stats[m].get(cat) or {}
                    row.append(_fmt_avg(b.get("s_count_sum", 0.0),
                                        int(b.get("n_scored", 0) or 0),
                                        "{:.4f}"))
                for m in models:
                    b = stats[m].get(cat) or {}
                    row.append(_fmt_avg(b.get("s_stab_sum", 0.0),
                                        int(b.get("n_scored", 0) or 0),
                                        "{:.4f}"))
                for m in models:
                    b = stats[m].get(cat) or {}
                    row.append(int(b.get("n_scored", 0) or 0))
                for m in models:
                    b = stats[m].get(cat) or {}
                    row.append(int(b.get("n_total", 0) or 0))
                w.writerow(row)
        print(f"[aggregate:{label}] written: {out_csv}")

        if write_back_summary:
            for m in models:
                b_all = stats[m].get("__ALL__") or {}
                _update_model_summary(runs_dir, m, b_all)

        for m in models:
            b = stats[m].get("__ALL__") or {}
            n = int(b.get("n_scored", 0) or 0)
            n_total = int(b.get("n_total", 0) or 0)
            if n > 0:
                avg_score = b.get("score_sum", 0.0) / n
                avg_sc = b.get("s_count_sum", 0.0) / n
                avg_st = b.get("s_stab_sum", 0.0) / n
                print(f"  [{label:<6}] {m:<24}  n_scored={n:>4} / n_total={n_total:>4}  "
                      f"score={avg_score:.4f}  s_count={avg_sc:.4f}  s_stab={avg_st:.4f}")
            else:
                print(f"  [{label:<6}] {m:<24}  n_scored=0 / n_total={n_total}  -")
        print()

    # ---- 三份输出 ----
    _aggregate_one_variant(
        per_v_full,
        out_csv=os.path.join(runs_dir, "compare_summary.csv"),
        write_back_summary=True,
        label="all",
    )

    if spec_kind_map:
        per_v_single = {m: _filter_by_kind(per_v_full[m], "single_shot") for m in models}
        _aggregate_one_variant(
            per_v_single,
            out_csv=os.path.join(runs_dir, "compare_summary_single.csv"),
            write_back_summary=False,
            label="single",
        )
        per_v_multi = {m: _filter_by_kind(per_v_full[m], "multi_shot") for m in models}
        _aggregate_one_variant(
            per_v_multi,
            out_csv=os.path.join(runs_dir, "compare_summary_multi.csv"),
            write_back_summary=False,
            label="multi",
        )


if __name__ == "__main__":
    main()
