#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject_consistency · 分数计算

计分规则 (per spec / 视频):
  - 每条 question 对错: answer == "yes" 且 parse_ok == True 则视为正确 (+1)
  - 该 spec 的 raw_score        = n_correct
  - 该 spec 的 max_score        = n_questions
  - 该 spec 的 normalized_score = raw_score / max_score, 范围 [0,1]
    (即一道一分, 然后把该视频归一化到 0-1)

汇总:
  - 把每个 spec 的 normalized 直接做"按 spec 平均"得到三档分数:
        overall_normalized_score   : 所有 spec
        single_shot_normalized     : shot_count == single_shot
        multi_shot_normalized      : shot_count == multi_shot
  - 单镜头 / 多镜头标签来自 t2av-bench/specs (经 _common/spec_kind.py 反查).
  - 另外按 t2av-bench 的 9 个 category 各自再算一份 "by_category" 均值
    (仅供横向对比, 不影响 final_combined).

输出:
  - <out-dir>/detailed_scores.json
  - 更新 <out-dir>/summary.json:
        score_calculation: { ... overall / single / multi 三个分数 + 规则文本 ... }
        final_combined          = overall_normalized_score
        final_combined_rule     = "subject_consistency normalized score (0-1)..."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]

_HERE = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_HERE, "..", "_common"))
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)
try:
    from spec_kind import (  # type: ignore  # noqa: E402
        load_spec_kind_map,
        KIND_SINGLE,
        KIND_MULTI,
        KIND_UNKNOWN,
    )
except Exception as _e:  # pragma: no cover
    print(f"[score_calc][WARN] cannot import spec_kind: {_e}", file=sys.stderr)
    load_spec_kind_map = None  # type: ignore
    KIND_SINGLE = "single_shot"
    KIND_MULTI = "multi_shot"
    KIND_UNKNOWN = ""


def load_per_question(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] bad json line skipped: {e}", file=sys.stderr)
    return out


def _is_correct(q: dict) -> bool:
    """题目按设计正确答案恒为 yes; 沿用与 emtion_consistency 相同的口径."""
    return (q.get("answer") == "yes") and (q.get("parse_ok") is True)


def calculate_scores(records: List[dict],
                     spec_kind_map: Optional[Dict[str, str]] = None) -> Dict[str, any]:
    spec_kind_map = spec_kind_map or {}

    spec_groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("spec_id", "")
        if sid:
            spec_groups[sid].append(r)

    spec_results: Dict[str, dict] = {}
    for sid, questions in spec_groups.items():
        per_q_correct = [1 if _is_correct(q) else 0 for q in questions]
        n_q = len(questions)
        n_correct = sum(per_q_correct)
        # 每题对得 1 分, 该视频上限 = 题数, 归一化到 [0,1].
        raw = n_correct
        max_raw = n_q
        norm = (raw / max_raw) if max_raw > 0 else 0.0
        kind = spec_kind_map.get(sid, KIND_UNKNOWN)
        cat = (questions[0].get("category") if questions else "") or ""
        spec_results[sid] = {
            "category": cat,
            "n_questions": n_q,
            "n_correct": n_correct,
            "question_scores": per_q_correct,
            "raw_score": raw,
            "max_score": max_raw,
            "normalized_score": norm,
            "shot_kind": kind,
        }

    def _avg(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    all_norms = [v["normalized_score"] for v in spec_results.values()]
    single_norms = [v["normalized_score"] for v in spec_results.values()
                    if v["shot_kind"] == KIND_SINGLE]
    multi_norms = [v["normalized_score"] for v in spec_results.values()
                   if v["shot_kind"] == KIND_MULTI]
    unknown_specs = [sid for sid, v in spec_results.items()
                     if v["shot_kind"] not in (KIND_SINGLE, KIND_MULTI)]

    overall = _avg(all_norms)
    single = _avg(single_norms)
    multi = _avg(multi_norms)

    # 顺带给每个 category 也算一个均值 (按 category 内 spec 平均归一化分)
    by_cat_norms: Dict[str, List[float]] = defaultdict(list)
    for v in spec_results.values():
        c = v.get("category") or "unknown"
        by_cat_norms[c].append(v["normalized_score"])
    by_cat_avg = {c: _avg(vs) for c, vs in by_cat_norms.items()}

    return {
        "n_specs": len(spec_results),
        "n_specs_single": len(single_norms),
        "n_specs_multi": len(multi_norms),
        "n_specs_unknown_kind": len(unknown_specs),
        "unknown_kind_spec_ids": unknown_specs,
        "overall_normalized_score": overall if overall is not None else 0.0,
        "single_shot_normalized_score": single,
        "multi_shot_normalized_score": multi,
        "by_category_normalized_score": by_cat_avg,
        "spec_results": spec_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-question", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--specs-dir", default=os.environ.get(
        "T2AV_SPECS_DIR",
        str(_REPO_ROOT / "t2av-bench/specs"),
    ))
    args = ap.parse_args()

    if not args.tag:
        args.tag = os.path.basename(os.path.dirname(args.per_question)) or "model"

    print(f"[score_calc] tag={args.tag}")
    print(f"  per_question = {args.per_question}")
    print(f"  out_dir      = {args.out_dir}")
    print(f"  specs_dir    = {args.specs_dir}")

    records = load_per_question(args.per_question)
    print(f"[score_calc] loaded {len(records)} question records")
    if not records:
        print("[ERROR] no records found", file=sys.stderr)
        sys.exit(1)

    if load_spec_kind_map is None:
        spec_kind_map: Dict[str, str] = {}
        print("[score_calc][WARN] spec_kind helper unavailable; "
              "single/multi split disabled", file=sys.stderr)
    else:
        spec_kind_map = load_spec_kind_map(args.specs_dir, verbose=True)

    scores = calculate_scores(records, spec_kind_map=spec_kind_map)

    os.makedirs(args.out_dir, exist_ok=True)
    detailed_path = os.path.join(args.out_dir, "detailed_scores.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(args.out_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}

    overall = scores["overall_normalized_score"]
    single = scores["single_shot_normalized_score"]
    multi = scores["multi_shot_normalized_score"]

    summary["score_calculation"] = {
        "method": "per-spec yes-rate, then mean-of-spec",
        "rules": [
            "Each correct question (answer=yes && parse_ok): +1 point",
            "Per spec normalized = n_correct / n_questions (range [0, 1])",
            "Final score = arithmetic mean of per-spec normalized "
            "(overall / single_shot / multi_shot computed separately)",
            "single_shot / multi_shot kind is read from t2av-bench/specs[*].shot_count",
        ],
        "n_specs_scored": scores["n_specs"],
        "n_specs_single": scores["n_specs_single"],
        "n_specs_multi": scores["n_specs_multi"],
        "n_specs_unknown_kind": scores["n_specs_unknown_kind"],
        "overall_normalized_score": round(overall, 6),
        "single_shot_normalized_score": (round(single, 6) if single is not None else None),
        "multi_shot_normalized_score": (round(multi, 6) if multi is not None else None),
        "by_category_normalized_score": {
            c: (round(v, 6) if v is not None else None)
            for c, v in (scores.get("by_category_normalized_score") or {}).items()
        },
    }
    summary["final_combined"] = round(overall, 6)
    summary["final_combined_rule"] = (
        "subject_consistency normalized score (0-1): "
        "per-question +1 then per-spec normalize, then mean-of-spec (overall)"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    def _fmt(v):
        return "-" if v is None else f"{v:.4f}"

    print()
    print("=" * 70)
    print("SCORE CALCULATION RESULTS")
    print("=" * 70)
    print(f"Total specs evaluated      : {scores['n_specs']}  "
          f"(single={scores['n_specs_single']}, "
          f"multi={scores['n_specs_multi']}, "
          f"unknown_kind={scores['n_specs_unknown_kind']})")
    print(f"Overall  normalized score  : {_fmt(overall)}")
    print(f"Single-shot normalized     : {_fmt(single)}")
    print(f"Multi-shot  normalized     : {_fmt(multi)}")
    if scores.get("by_category_normalized_score"):
        print("By category:")
        for c, v in sorted(scores["by_category_normalized_score"].items()):
            print(f"  {c:<20}  {_fmt(v)}")
    print()
    print("Per-spec results:")
    print("-" * 90)
    for sid in sorted(scores["spec_results"].keys()):
        r = scores["spec_results"][sid]
        kind_short = {KIND_SINGLE: "single", KIND_MULTI: "multi"}.get(r["shot_kind"], "?")
        cat = r.get("category") or "-"
        print(f"  {sid:14}  cat={cat:<18}  kind={kind_short:<6}  "
              f"Qs={r['n_questions']}  correct={r['n_correct']}  "
              f"raw={r['raw_score']}/{r['max_score']}  "
              f"norm={r['normalized_score']:.3f}")
    print("-" * 90)
    print(f"FINAL (overall) : {_fmt(overall)}")
    print(f"FINAL (single)  : {_fmt(single)}")
    print(f"FINAL (multi)   : {_fmt(multi)}")
    print()
    print(f"[score_calc] detailed results: {detailed_path}")
    print(f"[score_calc] updated summary : {summary_path}")


if __name__ == "__main__":
    main()
