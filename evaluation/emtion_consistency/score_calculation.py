#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emtion_consistency · 分数计算脚本

计分规则 (per spec / 视频):
  - 基础分: 每答对 1 题 +1 分
  - 一致性奖励 (bonus, 按答对题数累加):
      答对 0 题 -> bonus = 0
      答对 1 题 -> bonus = 0
      答对 2 题 -> bonus = +1
      答对 3 题 -> bonus = +2
    即 "答对一题一分; 两题 2+1 分; 三题 3+2 分".
  - 该 spec 的 raw_score  = base + bonus
  - 该 spec 的 max_score  = n_questions + bonus(n_questions)
        n=2 -> max=3, n=3 -> max=5
  - 该 spec 的 normalized = raw_score / max_score, 范围 [0,1]

汇总:
  - 把每个 spec 的 normalized 直接做"按 spec 平均"得到三档分数:
        overall_normalized_score  : 所有 spec
        single_shot_normalized    : shot_count == single_shot
        multi_shot_normalized     : shot_count == multi_shot
  - 单镜头 / 多镜头标签来自 t2av-bench/specs (经 _common/spec_kind.py 反查).

输出:
  - <out-dir>/detailed_scores.json
  - 更新 <out-dir>/summary.json:
        score_calculation: { ... overall / single / multi 三个分数 + 规则文本 ... }
        final_combined          = overall_normalized_score
        final_combined_rule     = "emotion_consistency normalized score (0-1)..."
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

# ---------------------------------------------------------------------------
# spec_kind helper (复用 evaluation/_common/spec_kind.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 计分规则常量
# ---------------------------------------------------------------------------
# bonus[k] = 答对 k 题时的额外一致性奖励
#   0 题 -> 0; 1 题 -> 0; 2 题 -> +1; 3 题 -> +2
# 题数最多到 3.
BONUS_BY_CORRECT = {0: 0, 1: 0, 2: 1, 3: 2}


def _bonus_of(n_correct: int) -> int:
    return BONUS_BY_CORRECT.get(int(n_correct), 0)


def _max_raw_of(n_questions: int) -> int:
    """该 spec 全对时的 raw 上限 = n_questions + bonus(n_questions)."""
    return int(n_questions) + _bonus_of(int(n_questions))


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 核心计算
# ---------------------------------------------------------------------------
def _is_correct(q: dict) -> bool:
    """题目设计上正确答案恒为 yes; 沿用原判定逻辑."""
    return (q.get("answer") == "yes") and (q.get("parse_ok") is True)


def calculate_scores(records: List[dict],
                     spec_kind_map: Optional[Dict[str, str]] = None) -> Dict[str, any]:
    """按新规则计算每个 spec 的归一化分数, 以及 overall/single/multi 三档平均."""
    spec_kind_map = spec_kind_map or {}

    # 1) 按 spec_id 分组
    spec_groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("spec_id", "")
        if sid:
            spec_groups[sid].append(r)

    # 2) 每个 spec 单独算分
    spec_results: Dict[str, dict] = {}
    for sid, questions in spec_groups.items():
        per_q_correct = [1 if _is_correct(q) else 0 for q in questions]
        n_q = len(questions)
        n_correct = sum(per_q_correct)

        base_score = n_correct
        bonus = _bonus_of(n_correct)
        raw = base_score + bonus
        max_raw = _max_raw_of(n_q)
        norm = (raw / max_raw) if max_raw > 0 else 0.0
        kind = spec_kind_map.get(sid, KIND_UNKNOWN)

        spec_results[sid] = {
            "n_questions": n_q,
            "n_correct": n_correct,
            "question_scores": per_q_correct,
            "base_score": base_score,
            "consistency_bonus": bonus,
            "raw_score": raw,
            "max_score": max_raw,
            "normalized_score": norm,
            "shot_kind": kind,
        }

    # 3) 按 spec 做"算术平均", 拆 overall / single / multi
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

    return {
        "n_specs": len(spec_results),
        "n_specs_single": len(single_norms),
        "n_specs_multi": len(multi_norms),
        "n_specs_unknown_kind": len(unknown_specs),
        "unknown_kind_spec_ids": unknown_specs,
        "overall_normalized_score": overall if overall is not None else 0.0,
        "single_shot_normalized_score": single,   # 可能为 None
        "multi_shot_normalized_score": multi,     # 可能为 None
        "spec_results": spec_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-question", required=True,
                    help="per_question.jsonl 文件路径")
    ap.add_argument("--out-dir", required=True,
                    help="输出目录 (一般就是 runs/<model>/)")
    ap.add_argument("--tag", default="",
                    help="模型标签 (仅打印用)")
    ap.add_argument("--specs-dir", default=os.environ.get(
        "T2AV_SPECS_DIR",
        str(_REPO_ROOT / "t2av-bench/specs"),
    ), help="t2av-bench specs 目录, 用于按 shot_count 拆 single/multi")
    args = ap.parse_args()

    if not args.tag:
        args.tag = os.path.basename(os.path.dirname(args.per_question)) or "model"

    print(f"[score_calc] tag={args.tag}")
    print(f"  per_question = {args.per_question}")
    print(f"  out_dir      = {args.out_dir}")
    print(f"  specs_dir    = {args.specs_dir}")

    # 1) 读题目结果
    records = load_per_question(args.per_question)
    print(f"[score_calc] loaded {len(records)} question records")
    if not records:
        print("[ERROR] no records found", file=sys.stderr)
        sys.exit(1)

    # 2) 读 shot_count 映射
    if load_spec_kind_map is None:
        spec_kind_map: Dict[str, str] = {}
        print("[score_calc][WARN] spec_kind helper unavailable; "
              "single/multi split disabled", file=sys.stderr)
    else:
        spec_kind_map = load_spec_kind_map(args.specs_dir, verbose=True)

    # 3) 算分
    scores = calculate_scores(records, spec_kind_map=spec_kind_map)

    # 4) 写出
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
        "method": "per-spec stair-bonus, then mean-of-spec",
        "rules": [
            "Each correct question: +1 point",
            "If 2 questions correct in same spec: additional +1 bonus",
            "If 3 questions correct in same spec: additional +2 bonus "
            "(i.e. base k + bonus[0,0,1,2])",
            "Per spec normalized = raw / max_raw "
            "(max_raw=3 for 2Q spec, =5 for 3Q spec)",
            "Final score = arithmetic mean of per-spec normalized "
            "(overall / single_shot / multi_shot separately)",
        ],
        "n_specs_scored": scores["n_specs"],
        "n_specs_single": scores["n_specs_single"],
        "n_specs_multi": scores["n_specs_multi"],
        "n_specs_unknown_kind": scores["n_specs_unknown_kind"],
        "overall_normalized_score": round(overall, 6),
        "single_shot_normalized_score": (round(single, 6) if single is not None else None),
        "multi_shot_normalized_score": (round(multi, 6) if multi is not None else None),
    }
    summary["final_combined"] = round(overall, 6)
    summary["final_combined_rule"] = (
        "emotion_consistency normalized score (0-1): "
        "per-spec stair bonus then mean-of-spec (overall)"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 5) 终端摘要
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
    print()
    print("Per-spec results:")
    print("-" * 90)
    for sid in sorted(scores["spec_results"].keys()):
        r = scores["spec_results"][sid]
        kind_short = {KIND_SINGLE: "single", KIND_MULTI: "multi"}.get(r["shot_kind"], "?")
        print(f"  {sid:14}  kind={kind_short:<6}  "
              f"Qs={r['n_questions']}  correct={r['n_correct']}  "
              f"base={r['base_score']} bonus={r['consistency_bonus']}  "
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