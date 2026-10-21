#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_questions.py · 为 action_eval 评测点生成 questions jsonl (1-5 打分版, 2 题分两次推理).

仅生成运动类(action)题目. 题面极简, 完整的 1-5 分严格档位定义全部放在 system prompt 中,
本脚本不在 question 文本里重复 rubric.

每个 spec 的每个 shot 生成 2 道 1-5 分题 (两题分两次推理, 互不干扰):
  Q1 motion_quality_no_prompt    仅看视频, 不送 prompt
                                 题面: "Is the motion in this video reasonable?"
  Q2 motion_prompt_following     送 prompt + 视频
                                 题面: "Does the motion in this video match the description in the prompt?"

详细评分细则(看什么/忽略什么/各档位定义/5 分严格度) -> 见:
  - system_prompt_motion_quality.md      (Q1)
  - system_prompt_prompt_following.md    (Q2)

prompt_text 字段:
  Q1 -> ""              (本题不送 prompt)
  Q2 -> 非空            (single_shot 取顶层 primary_action+scene_setting; multi_shot 取 shots[i] 的)

输出文件:
  t2av-bench/questions/action_eval_llm.jsonl

每行字段:
  spec_id        : str
  shot_idx       : int  (single_shot=1; multi_shot=1..K)
  question       : str  (英文, 极简)
  question_aspect: str  ("motion_quality_no_prompt" | "motion_prompt_following")
  question_type  : str  ("score_1to5")
  prompt_text    : str  (Q1 为空; Q2 非空)
  associated_subject_id : null
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]

# 2 道运动类综合题 (英文, 1-5 分), 两题分两次推理.
# 题面极简, 严格 1-5 档评分标准全部在 system prompt 里 (system_prompt_motion_quality.md /
# system_prompt_prompt_following.md), 这里不再重复 rubric.
# (aspect, question, needs_prompt)
QUESTIONS: List[Tuple[str, str, bool]] = [
    (
        "motion_quality_no_prompt",
        "Is the motion in this video reasonable?",
        False,
    ),
    (
        "motion_prompt_following",
        "Does the motion in this video match the description in the prompt?",
        True,
    ),
]


def load_specs(specs_dir: str) -> List[dict]:
    """加载所有 specs/<category>.jsonl, 返回 list[spec_obj]. 字段附加 __category."""
    out: List[dict] = []
    for name in sorted(os.listdir(specs_dir)):
        if not name.endswith(".jsonl"):
            continue
        cat = name[: -len(".jsonl")]
        p = os.path.join(specs_dir, name)
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"[WARN] {p}:{line_no} bad json: {e}", file=sys.stderr)
                    continue
                if not obj.get("id"):
                    continue
                obj["__category"] = cat
                out.append(obj)
    return out


def shot_count_of(spec: dict) -> int:
    kind = spec.get("shot_count")
    if kind == "single_shot":
        return 1
    if kind == "multi_shot":
        shots = spec.get("shots") or []
        n = len(shots)
        return n if n > 0 else 1
    shots = spec.get("shots") or []
    return max(1, len(shots))


def _compose_prompt_text(primary_action: Optional[str], scene_setting: Optional[str]) -> str:
    """把 primary_action + scene_setting 拼成给模型看的 prompt 段."""
    pa = (primary_action or "").strip()
    ss = (scene_setting or "").strip()
    parts: List[str] = []
    if pa:
        parts.append(f"Primary action: {pa}")
    if ss:
        parts.append(f"Scene setting: {ss}")
    return "\n".join(parts)


def get_prompt_text_for_shot(spec: dict, shot_idx: int) -> str:
    """根据 spec 类型与 shot_idx, 返回对应的 prompt 文本.
    single_shot: 取 spec 顶层 primary_action + scene_setting
    multi_shot : 取 spec['shots'][shot_idx-1] 的 primary_action + scene_setting
                 (若 shot 缺该字段则回落到顶层)
    """
    kind = spec.get("shot_count")
    top_pa = spec.get("primary_action")
    top_ss = spec.get("scene_setting")

    if kind == "multi_shot":
        shots = spec.get("shots") or []
        idx = shot_idx - 1
        if 0 <= idx < len(shots):
            s = shots[idx] or {}
            pa = s.get("primary_action") or top_pa
            ss = s.get("scene_setting") or top_ss
            return _compose_prompt_text(pa, ss)
        # 越界, 回落顶层
        return _compose_prompt_text(top_pa, top_ss)

    # single_shot 或未知
    return _compose_prompt_text(top_pa, top_ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs-dir",
                    default=str(_REPO_ROOT / "t2av-bench/specs"))
    ap.add_argument("--out",
                    default=str(_REPO_ROOT / "t2av-bench/questions/action_eval_llm.jsonl"))
    args = ap.parse_args()

    specs = load_specs(args.specs_dir)
    print(f"[build_questions] loaded {len(specs)} specs from {args.specs_dir}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n_lines = 0
    n_specs = 0
    n_shots = 0
    n_prompt_nonempty = 0
    cat_count: Dict[str, int] = {}

    with open(args.out, "w", encoding="utf-8") as fout:
        for spec in specs:
            sid = spec["id"]
            cat = spec.get("__category", "")
            k = shot_count_of(spec)
            n_specs += 1
            n_shots += k
            cat_count[cat] = cat_count.get(cat, 0) + 1
            for shot_idx in range(1, k + 1):
                ptxt_for_shot = get_prompt_text_for_shot(spec, shot_idx)
                for aspect, q, needs_prompt in QUESTIONS:
                    ptxt = ptxt_for_shot if needs_prompt else ""
                    if ptxt:
                        n_prompt_nonempty += 1
                    rec = {
                        "spec_id": sid,
                        "shot_idx": shot_idx,
                        "question": q,
                        "question_aspect": aspect,
                        "question_type": "score_1to5",
                        "prompt_text": ptxt,
                        "associated_subject_id": None,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_lines += 1

    print(f"[build_questions] wrote {n_lines} lines to {args.out}")
    print(f"[build_questions] specs={n_specs}  shots={n_shots}  questions/shot={len(QUESTIONS)}")
    print(f"[build_questions] questions with non-empty prompt_text: {n_prompt_nonempty}")
    print(f"[build_questions] per-category specs: {cat_count}")


if __name__ == "__main__":
    main()
