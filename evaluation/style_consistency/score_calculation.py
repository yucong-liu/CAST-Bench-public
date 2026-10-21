#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
style_consistency · 分数计算 (0-5 Likert, sum-of-questions per spec, 十分制)

规则:
  - 每条 question: VLM 返回 0..5 整数分 (写在 per_question.jsonl 的 "score" 字段).
    0 = 风格/类型完全不对 或 视频损坏不可看; 5 = 世界一流参考级.
  - parse_ok=False 的题视为最差档 score=0 (与 batch_eval fallback 一致).
  - per-spec score = sum(per-question score) / 2 题  (本评测每个 spec 恒为 2 题,
    所以 spec 分息好是 [0, 10] 这个十分制区间).
  - single_shot / multi_shot / overall 分别 = 该集合下 spec 分的算术平均.
  - by_category 分: 该 category 内所有 spec 分的算术平均.

输入:
  --per-question  <out-dir>/per_question.jsonl
  --out-dir       <out-dir>
  --tag           模型名 (打印用; 默认从 per_question 路径 basename 推断)
  --specs-dir     t2av-bench/specs, 用于拿 single/multi 标签

输出:
  - <out-dir>/detailed_scores.json
  - 更新 <out-dir>/summary.json:
        score_calculation: {
            method, rules,
            n_specs_scored, n_specs_single, n_specs_multi, n_specs_unknown_kind,
            overall_score_10            # 0~10
            single_shot_score_10        # 0~10
            multi_shot_score_10         # 0~10
            overall_normalized_score    # 0~1, 兼容上层 aggregate_summary
            single_shot_normalized_score
            multi_shot_normalized_score
            by_category_score_10
            by_category_normalized_score
        }
        final_combined          = overall_score_10  (十分制)
        final_combined_rule     = 文本说明
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional
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


# 每题最高分; 每个 spec 题数硬假设 = 2 (style_consistency 设计); 十分制 = 2*5 = 10
PER_Q_MAX = 5
PER_Q_MIN = 0  # 完全不对 或 视频损坏 记 0
SPEC_MAX_10 = 10  # 仅用于打印; 实际计算用 sum(score) / (n_q*5) * 10


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


def _question_score(q: dict) -> int:
    """读取单题 0-5 分; parse_ok=False 或非法 score 一律视为 0 (最差档).
    与 batch_eval.py 的 fallback (score=0) 语义保持一致.
    """
    if not q.get("parse_ok"):
        return PER_Q_MIN
    s = q.get("score", None)
    try:
        if isinstance(s, bool):
            return PER_Q_MIN
        if isinstance(s, int):
            v = int(s)
        elif isinstance(s, float):
            v = int(round(float(s)))
        elif isinstance(s, str):
            v = int(round(float(s.strip())))
        else:
            return PER_Q_MIN
    except Exception:
        return PER_Q_MIN
    if v < PER_Q_MIN:
        return PER_Q_MIN
    if v > PER_Q_MAX:
        return PER_Q_MAX
    return v


def calculate_scores(records: List[dict],
                     spec_kind_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    spec_kind_map = spec_kind_map or {}

    # 按 spec 分桶
    spec_groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("spec_id", "")
        if sid:
            spec_groups[sid].append(r)

    # 每个 spec: 取该 spec 下所有 question 的 0..5 分相加 = spec 十分制分 (2 题 x 5)
    spec_results: Dict[str, dict] = {}
    for sid, questions in spec_groups.items():
        per_q_scores = [_question_score(q) for q in questions]
        n_q = len(questions)
        sum_score = sum(per_q_scores)
        # 直接十分制: 2 题各 0..5, 加起来就是 [0, 10]; 兼容 (理论上) n_q != 2 的极端情形,
        # 用 n_q*5 作为该 spec 满分, 归一回 0..10
        max_for_spec = max(1, n_q) * PER_Q_MAX
        score_10 = (sum_score / max_for_spec) * SPEC_MAX_10  # 0..10
        score_norm = score_10 / SPEC_MAX_10  # 0..1
        kind = spec_kind_map.get(sid, KIND_UNKNOWN)
        cat = (questions[0].get("category") if questions else "") or ""
        spec_results[sid] = {
            "category": cat,
            "n_questions": n_q,
            "question_scores": per_q_scores,
            "sum_score": sum_score,
            "score_10": score_10,            # 十分制
            "normalized_score": score_norm,  # 0..1
            "shot_kind": kind,
        }

    def _avg(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    all_10 = [v["score_10"] for v in spec_results.values()]
    single_10 = [v["score_10"] for v in spec_results.values()
                 if v["shot_kind"] == KIND_SINGLE]
    multi_10 = [v["score_10"] for v in spec_results.values()
                if v["shot_kind"] == KIND_MULTI]
    unknown_specs = [sid for sid, v in spec_results.items()
                     if v["shot_kind"] not in (KIND_SINGLE, KIND_MULTI)]

    overall_10 = _avg(all_10)
    single_10_avg = _avg(single_10)
    multi_10_avg = _avg(multi_10)

    # by category (基于全部 spec, 不区分 shot kind)
    by_cat_10: Dict[str, List[float]] = defaultdict(list)
    for v in spec_results.values():
        c = v.get("category") or "unknown"
        by_cat_10[c].append(v["score_10"])
    by_cat_avg_10 = {c: _avg(vs) for c, vs in by_cat_10.items()}

    return {
        "n_specs": len(spec_results),
        "n_specs_single": len(single_10),
        "n_specs_multi": len(multi_10),
        "n_specs_unknown_kind": len(unknown_specs),
        "unknown_kind_spec_ids": unknown_specs,

        # 十分制
        "overall_score_10": (overall_10 if overall_10 is not None else 0.0),
        "single_shot_score_10": single_10_avg,
        "multi_shot_score_10": multi_10_avg,
        "by_category_score_10": by_cat_avg_10,

        # 兼容上层 aggregate_summary 的 0..1 字段
        "overall_normalized_score": ((overall_10 / SPEC_MAX_10) if overall_10 is not None else 0.0),
        "single_shot_normalized_score": ((single_10_avg / SPEC_MAX_10) if single_10_avg is not None else None),
        "multi_shot_normalized_score": ((multi_10_avg / SPEC_MAX_10) if multi_10_avg is not None else None),
        "by_category_normalized_score": {
            c: ((v / SPEC_MAX_10) if v is not None else None)
            for c, v in by_cat_avg_10.items()
        },

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

    overall_10 = scores["overall_score_10"]
    single_10 = scores["single_shot_score_10"]
    multi_10 = scores["multi_shot_score_10"]

    overall_norm = scores["overall_normalized_score"]
    single_norm = scores["single_shot_normalized_score"]
    multi_norm = scores["multi_shot_normalized_score"]

    summary["score_calculation"] = {
        "method": "Likert 0-5 per question; sum 2 questions per spec -> 10-pt spec score; "
                  "arithmetic mean of spec scores for single / multi / overall",
        "rules": [
            "Each question: integer 0..5 from VLM (parse_fail or out-of-range -> 0). "
            "0 = style/type completely wrong or video broken; 5 = world-class reference quality.",
            "Per spec score (10-pt) = sum(question scores) / (n_q * 5) * 10  "
            "(this evaluation has 2 questions per spec, so spec score lives in [0, 10]).",
            "single_shot / multi_shot kind read from t2av-bench/specs[*].shot_count.",
            "overall_score_10        = mean over ALL specs (10-pt scale).",
            "single_shot_score_10    = mean over single_shot specs (10-pt).",
            "multi_shot_score_10     = mean over multi_shot specs (10-pt).",
            "by_category_score_10    = mean over specs in that category (10-pt).",
            "*_normalized_score      = corresponding *_score_10 / 10  (0..1, for cross-eval compat).",
        ],
        "n_specs_scored": scores["n_specs"],
        "n_specs_single": scores["n_specs_single"],
        "n_specs_multi": scores["n_specs_multi"],
        "n_specs_unknown_kind": scores["n_specs_unknown_kind"],

        # 十分制 (主输出)
        "overall_score_10": round(overall_10, 6),
        "single_shot_score_10": (round(single_10, 6) if single_10 is not None else None),
        "multi_shot_score_10": (round(multi_10, 6) if multi_10 is not None else None),
        "by_category_score_10": {
            c: (round(v, 6) if v is not None else None)
            for c, v in (scores.get("by_category_score_10") or {}).items()
        },

        # 0..1 (兼容)
        "overall_normalized_score": round(overall_norm, 6),
        "single_shot_normalized_score": (round(single_norm, 6) if single_norm is not None else None),
        "multi_shot_normalized_score": (round(multi_norm, 6) if multi_norm is not None else None),
        "by_category_normalized_score": {
            c: (round(v, 6) if v is not None else None)
            for c, v in (scores.get("by_category_normalized_score") or {}).items()
        },
    }
    summary["final_combined"] = round(overall_10, 6)
    summary["final_combined_rule"] = (
        "style_consistency 0-5 Likert score (10-pt scale): per-question 0..5, "
        "sum 2 questions per spec -> spec score in [0, 10], then arithmetic mean over specs."
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    def _fmt10(v):
        return "-" if v is None else f"{v:.4f}"

    print()
    print("=" * 70)
    print("SCORE CALCULATION RESULTS  (0-5 Likert, sum-of-2-questions = 10-pt)")
    print("=" * 70)
    print(f"Total specs evaluated      : {scores['n_specs']}  "
          f"(single={scores['n_specs_single']}, "
          f"multi={scores['n_specs_multi']}, "
          f"unknown_kind={scores['n_specs_unknown_kind']})")
    print(f"Overall   score (0-10)     : {_fmt10(overall_10)}")
    print(f"Single-shot score (0-10)   : {_fmt10(single_10)}")
    print(f"Multi-shot  score (0-10)   : {_fmt10(multi_10)}")
    if scores.get("by_category_score_10"):
        print("By category (0-10):")
        for c, v in sorted(scores["by_category_score_10"].items()):
            print(f"  {c:<20}  {_fmt10(v)}")
    print()
    print(f"[score_calc] detailed results: {detailed_path}")
    print(f"[score_calc] updated summary : {summary_path}")


if __name__ == "__main__":
    main()
