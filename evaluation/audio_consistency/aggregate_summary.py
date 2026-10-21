#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_consistency 评测点 · 跨模型汇总

自动 listdir(<runs_dir>), 读各 <model>/summary.json + per_question.jsonl,
输出 <runs_dir>/compare_summary.csv. 列设计:

    category, score_<model>..., n_<model>...,

布局:
    第 1 行 (__ALL__):
        - score_<m>   = mean(score) over Type-2 (打分题), 跨所有 category
        - yesno_yes_rate_<m> = yes 率 over Type-1
        - n_<m>       = 该模型 n_questions
    后续每行 (按 category):
        - score_<m>   = mean(score) over Type-2 in this category
        - yesno_yes_rate_<m> = yes 率 over Type-1 in this category
        - n_<m>       = 该模型该 category 下 question 数

注意: 评分聚合规则后续可能调整 (mean 仅作占位), 这里只展示原始 mean.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _safe_load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


#: 模型展示顺序: 三家商业模型固定在最前 (seedance, wan2.7, kling-v3),
#  其余 (开源模型等) 按字母序追加.
MODEL_ORDER_PREFIX: List[str] = ["seedance", "wan2.7", "kling-v3"]

# ============================================================
# 镜头正确率惩罚 (仅 multi_shot 相关 CSV: compare_summary_multi.csv 与
# compare_summary_multi_intersect.csv 的 4 个分数列生效:
#   score_<m> / yes_rate_<m> / shot_norm_<m> / final_combined_<m>
# 计数列 (n_specs_<m>, n_<m>) 不罚, compare_summary.csv / _single.csv 不罚.
#
# 数据源: TransNetV2 的 multi_shot 镜头正确率
#   路径: <TransNetV2>/runs/compare_summary_multi.csv
#   列格式: category, <model>, ..., n_<model>, ...  (无前缀, 首列后直接就是各模型 c)
#
# 惩罚规则 (线性 · c<0.8 触发, 与 action_eval 完全一致):
#   e = 1 - c  (镜头错误率)
#   c >= 0.8  或  c 缺失     -> 不罚, score 原值
#   c <  0.8                 -> score_new = score * c  (等价 score * (1 - e))
# ============================================================
SHOT_PENALTY_CSV = str(_REPO_ROOT / "evaluation/TransNetV2/runs/compare_summary_multi.csv")
SHOT_PENALTY_THRESHOLD = 0.8


def _load_shot_correct_rates(csv_path: str) -> Dict[Tuple[str, str], float]:
    """读 TransNetV2 multi CSV, 返回 {(category, model): c} .

    - 表头首列必须是 'category', 后续列里以 'n_<model>' 出现的作为模型识别锚点;
      对应的裸列名 <model> 即镜头正确率 c 列 (取值 [0,1] 或空).
    - category 保留原样 ('action' / 'speech' / ... / '__ALL__').
    - c 为空串 -> 该键跳过 (视为未知, 后续 _apply_linear_penalty 收到 None 会不罚).
    """
    out: Dict[Tuple[str, str], float] = {}
    if not csv_path or not os.path.isfile(csv_path):
        return out
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
    except Exception:
        return out
    if not rows:
        return out
    header = rows[0]
    if not header or header[0] != "category":
        return out
    models: List[str] = []
    for col in header:
        if col.startswith("n_") and not col.startswith("n_specs_"):
            m = col[len("n_"):]
            if m and m in header:
                models.append(m)
    col_idx: Dict[str, int] = {}
    for m in models:
        try:
            col_idx[m] = header.index(m)
        except ValueError:
            continue
    for r in rows[1:]:
        if not r:
            continue
        cat = r[0]
        if not cat:
            continue
        for m, idx in col_idx.items():
            if idx >= len(r):
                continue
            v = (r[idx] or "").strip()
            if not v:
                continue
            try:
                out[(cat, m)] = float(v)
            except Exception:
                continue
    return out


def _apply_linear_penalty(score_str: str, c: Optional[float],
                          threshold: float = SHOT_PENALTY_THRESHOLD) -> str:
    """给分数字符串套线性惩罚 (c < threshold 时 score *= c, 否则原值).

    - score_str 为 '' / 非法数字 -> 原样返回.
    - c is None (查不到镜头正确率) -> 不罚, 原值.
    - c >= threshold (0.8) -> 不罚, 原值.
    - c <  threshold        -> score * c , 保留 4 位小数.
    """
    if not score_str:
        return score_str
    try:
        s = float(score_str)
    except Exception:
        return score_str
    if c is None:
        return score_str
    if c >= threshold:
        return score_str
    return f"{s * c:.4f}"


def _penalize_float(v: Optional[float], c: Optional[float],
                    threshold: float = SHOT_PENALTY_THRESHOLD) -> Optional[float]:
    """float 版本的惩罚, 语义同 _apply_linear_penalty. 供全量分支加权计算内部使用."""
    if v is None:
        return None
    if c is None or c >= threshold:
        return v
    return v * c


def discover_models(runs_dir: str) -> List[str]:
    found: List[str] = []
    for name in sorted(os.listdir(runs_dir)):
        sub = os.path.join(runs_dir, name)
        if not os.path.isdir(sub):
            continue
        if name.startswith(("_", ".")):
            continue
        if os.path.isfile(os.path.join(sub, "summary.json")):
            found.append(name)
    # 按 MODEL_ORDER_PREFIX 排序: 已知商业模型优先, 其余按字母序追加
    head = [m for m in MODEL_ORDER_PREFIX if m in found]
    tail = [m for m in found if m not in MODEL_ORDER_PREFIX]
    return head + tail


def load_per_question(runs_dir: str, model: str) -> List[dict]:
    p = os.path.join(runs_dir, model, "per_question.jsonl")
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
    """
    返回 {category_or_'__ALL__': stats_dict}.
    stats_dict keys: n_questions, n_yesno, n_score, n_parse_ok,
                     yes_count, score_sum, score_count.
    """
    bucket: Dict[str, dict] = defaultdict(lambda: {
        "n_questions": 0,
        "n_yesno": 0,
        "n_score": 0,
        "n_parse_ok": 0,
        "yes_count": 0,
        "score_sum": 0.0,
        "score_count": 0,
    })
    for r in records:
        cat = r.get("category") or ""
        for key in ("__ALL__", cat):
            if not key:
                continue
            b = bucket[key]
            b["n_questions"] += 1
            if r.get("parse_ok"):
                b["n_parse_ok"] += 1
            qt = r.get("question_type")
            if qt == "yesno":
                b["n_yesno"] += 1
                if r.get("answer") == "yes":
                    b["yes_count"] += 1
            elif qt == "score":
                b["n_score"] += 1
                sc = r.get("score")
                if isinstance(sc, (int, float)) and not isinstance(sc, bool):
                    b["score_sum"] += float(sc)
                    b["score_count"] += 1
    return bucket

# ---------------------------------------------------------------------------
# spec 级归一化打分聚合 (shot_norm 列, 本评测点的主指标)
#
# 规则:
#   * parse_ok == False 的题: 题级排除 (raw 和 max 都不计).
#   * 单道题贡献:
#       yesno: raw += 1 if answer == "yes" else 0 ; max += 1
#       score: raw += score (1..5)               ; max += 5
#   * 单镜头 = (spec_id, shot_idx) 聚合: shot_raw = Σ题raw, shot_max = Σ题max.
#   * 单 spec 聚合:
#       single_shot -> spec_total = 唯一镜头的 shot_raw,  spec_max = 唯一镜头的 shot_max
#       multi_shot  -> spec_total = mean(shot_raw),       spec_max = mean(shot_max)
#   * spec_norm = spec_total / spec_max  (∈ [0,1])
#       若 spec_max == 0 (该 spec 全部题 parse_fail) -> 整 spec 排除.
#   * category 级 shot_norm_mean = mean(该 category 下所有有效 spec 的 spec_norm).
#   * 模型整体 (__ALL__) shot_norm_mean = mean(全部有效 spec 的 spec_norm).
#
# ---------------------------------------------------------------------------
# final_combined (题级全局, 参考列; 主指标为上面的 shot_norm):
#
#   final_combined = (score_mean + yes_rate * 5) / 10, 落在 [0, 1].
#   缺失兜底:
#     * score_count == 0 (无 score 题): 退化为 yes_rate (即 yes_rate*5/5)
#     * n_yesno == 0    (无 yesno 题): 退化为 score_mean / 5
#     * 两者都缺: 返回 None
#   注: 这是题级一锅平均的指标, 题数多的镜头/spec 会占更大权重.
#       如需 spec 等权指标, 请看 shot_norm 列.
# ---------------------------------------------------------------------------

def _shot_contrib(rec: dict) -> Tuple[float, float]:
    """单道题贡献的 (raw, max). parse_fail -> (0, 0) 即题级排除."""
    if not rec.get("parse_ok"):
        return 0.0, 0.0
    qt = rec.get("question_type")
    if qt == "yesno":
        return (1.0 if rec.get("answer") == "yes" else 0.0), 1.0
    if qt == "score":
        sc = rec.get("score")
        if isinstance(sc, (int, float)) and not isinstance(sc, bool):
            return float(sc), 5.0
        return 0.0, 0.0
    return 0.0, 0.0


def aggregate_spec_norm(records: List[dict]) -> Dict[str, dict]:
    """
    返回 {category_or_'__ALL__': {
        "norm_sum": float,            # Σ spec_norm
        "n_specs":  int,              # 实际计入的 spec 数 (spec_max>0 才算)
        "n_specs_excluded": int,      # spec_max==0 被排除的 spec 数
    }}
    """
    # 第一遍: (spec_id, shot_idx) -> (raw, max), 同时记录 spec 的 category & shot_count_kind
    shot_acc: Dict[Tuple[str, int], List[float]] = defaultdict(lambda: [0.0, 0.0])
    spec_meta: Dict[str, Dict[str, str]] = {}  # spec_id -> {category, shot_count_kind}

    for r in records:
        spec_id = r.get("spec_id")
        shot_idx = r.get("shot_idx")
        if not spec_id or shot_idx is None:
            continue
        raw, mx = _shot_contrib(r)
        key = (spec_id, int(shot_idx))
        shot_acc[key][0] += raw
        shot_acc[key][1] += mx
        if spec_id not in spec_meta:
            spec_meta[spec_id] = {
                "category": r.get("category") or "",
                "shot_count_kind": r.get("shot_count_kind") or "",
            }

    # 第二遍: spec_id -> [shot_raw...], [shot_max...]
    spec_shots: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for (spec_id, _shot_idx), (raw, mx) in shot_acc.items():
        spec_shots[spec_id].append((raw, mx))

    bucket: Dict[str, dict] = defaultdict(lambda: {
        "norm_sum": 0.0,
        "n_specs": 0,
        "n_specs_excluded": 0,
    })

    for spec_id, shots in spec_shots.items():
        meta = spec_meta.get(spec_id, {})
        cat = meta.get("category") or ""
        kind = meta.get("shot_count_kind") or ""

        if not shots:
            continue

        if kind == "single_shot" or len(shots) == 1:
            spec_total, spec_max = shots[0]
        else:
            # multi_shot: 镜头维度求均值 (即使某个镜头 max=0 也参与求均值, 这样不会让坏镜头被悄悄忽略)
            n = len(shots)
            spec_total = sum(s[0] for s in shots) / n
            spec_max = sum(s[1] for s in shots) / n

        targets = [k for k in ("__ALL__", cat) if k]

        if spec_max <= 0:
            for k in targets:
                bucket[k]["n_specs_excluded"] += 1
            continue

        spec_norm = spec_total / spec_max
        for k in targets:
            bucket[k]["norm_sum"] += spec_norm
            bucket[k]["n_specs"] += 1

    return bucket


def _fmt_rate(num: int, den: int) -> str:
    if den <= 0:
        return ""
    return f"{num / den:.4f}"


def _fmt_mean(s: float, n: int) -> str:
    if n <= 0:
        return ""
    return f"{s / n:.4f}"


def _fmt_norm(norm_sum: float, n: int) -> str:
    if n <= 0:
        return ""
    return f"{norm_sum / n:.4f}"


def _calc_final_combined(b: dict):
    """
    final_combined = (score_mean + yes_rate * 5) / 10, 落在 [0, 1].
    缺失兜底:
      * score_count == 0 (无 score 题): 退化为 yes_rate (即 yes_rate*5/5)
      * n_yesno == 0    (无 yesno 题): 退化为 score_mean / 5
      * 两者都缺: 返回 None
    """
    if not b:
        return None
    sc_n = int(b.get("score_count", 0) or 0)
    yn_n = int(b.get("n_yesno", 0) or 0)
    score_mean = (b.get("score_sum", 0.0) / sc_n) if sc_n > 0 else None
    yes_rate = (b.get("yes_count", 0) / yn_n) if yn_n > 0 else None
    if score_mean is not None and yes_rate is not None:
        return (score_mean + yes_rate * 5.0) / 10.0
    if yes_rate is not None:
        return yes_rate
    if score_mean is not None:
        return score_mean / 5.0
    return None


def _fmt_final(b: dict) -> str:
    v = _calc_final_combined(b)
    if v is None:
        return ""
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# __ALL__ 总分加权规则:
#
#   __ALL__ = 0.5 * speech_cat_value  +  0.5 * mean(other_cat_values)
#
# 其中 cat_value 指该指标 (score / yes_rate / shot_norm / final_combined) 在
# 具体 category 行的数值 (由 category 级桶算出来, 与 CSV 里 category 行一致).
#
# 缺失兜底:
#   * speech 缺 (无值), 其他非空 -> mean(其他)
#   * 其他全空, speech 有值      -> speech
#   * 都空                        -> None
# 计数列 (n_specs_*, n_*) 不参与加权, 继续用 __ALL__ 桶里的原值 (全体求和).
# ---------------------------------------------------------------------------

SPEECH_CATEGORY: str = "speech"
SPEECH_WEIGHT: float = 0.5


def _weighted_all(cat_values: Dict[str, Optional[float]]) -> Optional[float]:
    """按 speech 50% + 其他 category 均值 50% 加权. cat_values 里 None 视为缺失."""
    speech_v = cat_values.get(SPEECH_CATEGORY)
    others = [v for c, v in cat_values.items()
              if c and c != SPEECH_CATEGORY and c != "__ALL__" and v is not None]
    if speech_v is not None and others:
        return SPEECH_WEIGHT * speech_v + (1.0 - SPEECH_WEIGHT) * (sum(others) / len(others))
    if others:
        return sum(others) / len(others)
    if speech_v is not None:
        return speech_v
    return None


def _fmt_weighted(cat_values: Dict[str, Optional[float]]) -> str:
    v = _weighted_all(cat_values)
    if v is None:
        return ""
    return f"{v:.4f}"


def _cat_score_mean(b: dict) -> Optional[float]:
    n = int(b.get("score_count", 0) or 0)
    if n <= 0:
        return None
    return float(b.get("score_sum", 0.0)) / n


def _cat_yes_rate(b: dict) -> Optional[float]:
    n = int(b.get("n_yesno", 0) or 0)
    if n <= 0:
        return None
    return float(b.get("yes_count", 0)) / n


def _cat_shot_norm(nb: dict) -> Optional[float]:
    n = int(nb.get("n_specs", 0) or 0)
    if n <= 0:
        return None
    return float(nb.get("norm_sum", 0.0)) / n


def _parse_float(s: str) -> Optional[float]:
    """用于追加模式: 把 CSV 里的字符串还原为 float, 空串/非数 -> None."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _update_model_summary(runs_dir: str, model: str, all_stats: dict, all_basic: dict) -> None:
    """将 spec 级归一化结果 + 题级 final_combined 写回 <model>/summary.json."""
    p = os.path.join(runs_dir, model, "summary.json")
    if not os.path.isfile(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    n_specs = int(all_stats.get("n_specs", 0))
    # __ALL__ 里传入的是加权值 (调用方传, 见 _aggregate_one_variant); 计数仍用桶里的全量
    norm_mean = all_stats.get("__ALL_WEIGHTED_SHOT_NORM__")
    data["shot_score_norm_mean"] = (round(norm_mean, 6) if norm_mean is not None else None)
    data["shot_score_norm_n_specs"] = n_specs
    data["shot_score_norm_n_specs_excluded"] = int(all_stats.get("n_specs_excluded", 0))
    data["shot_score_norm_rule"] = (
        "per-question parse-fail excluded; "
        "per-shot raw=Σyes(0/1)+Σscore(1-5), max=n_yesno+5*n_score; "
        "spec_total=mean(shot_raw) for multi_shot (single_shot uses that one shot directly); "
        "spec_norm=spec_total/spec_max in [0,1]; "
        "category shot_norm = mean(spec_norm) over specs with spec_max>0; "
        "__ALL__ = 0.5*speech_cat + 0.5*mean(other categories)"
    )
    final_v = all_basic.get("__ALL_WEIGHTED_FINAL__")
    data["final_combined"] = (round(final_v, 6) if final_v is not None else None)
    # 旧版本写过的新算法字段, 这里清理掉, 避免 summary.json 里两套字段混淆
    for k in ("final_combined_n_specs", "final_combined_n_specs_excluded"):
        data.pop(k, None)
    data["final_combined_rule"] = (
        "category final_combined = (score_mean + yes_rate*5) / 10, in [0,1]; "
        "fallback: no score-questions -> yes_rate; no yesno-questions -> score_mean/5; "
        "both missing -> null; "
        "__ALL__ = 0.5*speech_cat + 0.5*mean(other categories)"
    )
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------- 追加模式辅助函数 ----------------
# 用途: 只重算指定模型的列, 保留旧 CSV 里其他模型的数值不变.
# 触发条件: main() 里传入 --append-models "<m>[,<m>...]"

def _read_existing_csv(path: str):
    """读旧 CSV, 返回 (models_in_order, {category: {col_name: value}}).

    - models_in_order: 从表头里按 shot_norm_<m> 出现顺序抽出模型名列表.
      (shot_norm_ 前缀唯一, 不与其他 6 组列前缀冲突, 用它做锚点最稳)
    - 每个 category 行存成 {col_name: value_str}.
    - 若文件不存在或为空, 返回 ([], {}).
    """
    if not os.path.isfile(path):
        return [], {}
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], {}
    header = rows[0]
    models: List[str] = []
    for col in header:
        if col.startswith("shot_norm_"):
            models.append(col[len("shot_norm_"):])
    data: Dict[str, Dict[str, str]] = {}
    for r in rows[1:]:
        if not r:
            continue
        cat = r[0]
        row_map: Dict[str, str] = {}
        for i, col_name in enumerate(header):
            row_map[col_name] = r[i] if i < len(r) else ""
        data[cat] = row_map
    return models, data


def _compute_one_model_stats(runs_dir: str, model: str, label: str):
    """给指定 model 算某个 label(all/single/multi) 下的 6 个字段, 按 category 分组.

    返回 {category: {score_<m>: v, yes_rate_<m>: v, shot_norm_<m>: v,
                     final_combined_<m>: v, n_specs_<m>: v, n_<m>: v}}
    """
    records = load_per_question(runs_dir, model)
    if label == "single":
        records = [r for r in records if (r.get("shot_count_kind") or "") == "single_shot"]
    elif label == "multi":
        records = [r for r in records if (r.get("shot_count_kind") or "") == "multi_shot"]

    stats = aggregate_one(records)
    norm_stats = aggregate_spec_norm(records)

    out: Dict[str, Dict[str, str]] = {}
    all_cats = set(stats.keys()) | set(norm_stats.keys())
    for cat in all_cats:
        b = stats.get(cat) or {}
        nb = norm_stats.get(cat) or {}
        out[cat] = {
            f"score_{model}":          _fmt_mean(b.get("score_sum", 0.0), b.get("score_count", 0)),
            f"yes_rate_{model}":       _fmt_rate(b.get("yes_count", 0), b.get("n_yesno", 0)),
            f"shot_norm_{model}":      _fmt_norm(nb.get("norm_sum", 0.0), nb.get("n_specs", 0)),
            f"final_combined_{model}": _fmt_final(b),
            f"n_specs_{model}":        str(nb.get("n_specs", 0)),
            f"n_{model}":              str(b.get("n_questions", 0)),
        }
    return out


def _append_one_variant(runs_dir: str, out_csv: str,
                         append_models: List[str], label: str) -> None:
    """追加模式核心: 读旧 CSV -> 保留其他模型原值 -> 只重算 append_models -> 写回.

    __ALL__ 行的 4 个分数列 (score/yes_rate/shot_norm/final_combined) 会重新按
    'speech 50% + 其他类均值 50%' 重新算, category 行不变.

    对 label == 'multi' 的 CSV, 会额外对 4 个分数列 (score/yes_rate/shot_norm/
    final_combined) 应用 TransNetV2 镜头正确率惩罚: c<0.8 时 x_new = x*c.
    重要: 只对本次计算的 append_models 应用惩罚 (已有 CSV 里其他模型的值原样保留,
    无法确认它们计算时对应的 TransNetV2 结果)."""
    old_models, old_data = _read_existing_csv(out_csv)

    final_models: List[str] = list(old_models)
    for m in append_models:
        if m not in final_models:
            final_models.append(m)

    # multi CSV 才启用惩罚
    apply_penalty = (label == "multi")
    shot_rates: Dict[Tuple[str, str], float] = (
        _load_shot_correct_rates(SHOT_PENALTY_CSV) if apply_penalty else {}
    )

    new_stats: Dict[str, Dict[str, Dict[str, str]]] = {}
    for m in append_models:
        s = _compute_one_model_stats(runs_dir, m, label)
        if apply_penalty:
            # 4 个分数列就地惩罚 (category 级, 用 (cat, m) 查表)
            for cat, cols in s.items():
                c = shot_rates.get((cat, m))
                for score_col in (f"score_{m}", f"yes_rate_{m}",
                                   f"shot_norm_{m}", f"final_combined_{m}"):
                    if score_col in cols:
                        cols[score_col] = _apply_linear_penalty(cols[score_col], c)
        new_stats[m] = s

    all_cats = set(old_data.keys())
    for m in append_models:
        all_cats.update(new_stats[m].keys())
    cats_sorted = sorted(c for c in all_cats if c and c != "__ALL__")
    rows_order = cats_sorted + ["__ALL__"]

    header = ["category"]
    for m in final_models: header.append(f"score_{m}")
    for m in final_models: header.append(f"yes_rate_{m}")
    for m in final_models: header.append(f"shot_norm_{m}")
    for m in final_models: header.append(f"final_combined_{m}")
    for m in final_models: header.append(f"n_specs_{m}")
    for m in final_models: header.append(f"n_{m}")

    # 前缀识别: 更长的前缀优先, 否则 score_/n_ 会误吃 shot_norm_/n_specs_ 等
    _prefixes = ("final_combined_", "shot_norm_", "yes_rate_", "n_specs_",
                 "score_", "n_")

    # 先把非 __ALL__ 行拼好, 同时为每个 final_model 收集 4 个分数列 · 每个 category 的 float,
    # 供 __ALL__ 行加权用. 追加模型取新值, 非追加模型取旧 CSV 里的值.
    per_cat_score:  Dict[str, Dict[str, Optional[float]]] = {m: {} for m in final_models}
    per_cat_yes:    Dict[str, Dict[str, Optional[float]]] = {m: {} for m in final_models}
    per_cat_shot:   Dict[str, Dict[str, Optional[float]]] = {m: {} for m in final_models}
    per_cat_final:  Dict[str, Dict[str, Optional[float]]] = {m: {} for m in final_models}

    def _get_val(cat: str, col: str, mtag: str) -> str:
        """追加模型取新值 (新值为空时 fallback 到旧值); 非追加模型取旧值."""
        old_row = old_data.get(cat, {})
        if mtag in append_models:
            val = new_stats[mtag].get(cat, {}).get(col, "")
            if val == "" and col in old_row:
                val = old_row[col]
            return val
        return old_row.get(col, "")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        # ---- 先写非 __ALL__ 的各 category 行, 同时收集 4 个分数列 · 每 category 的 float ----
        for cat in cats_sorted:
            row = [cat]
            for col in header[1:]:
                mtag = None
                for prefix in _prefixes:
                    if col.startswith(prefix):
                        mtag = col[len(prefix):]
                        break
                val = _get_val(cat, col, mtag)
                row.append(val)
                if mtag in final_models:
                    if col.startswith("score_") and not col.startswith("score_norm_"):
                        per_cat_score[mtag][cat] = _parse_float(val)
                    elif col.startswith("yes_rate_"):
                        per_cat_yes[mtag][cat] = _parse_float(val)
                    elif col.startswith("shot_norm_"):
                        per_cat_shot[mtag][cat] = _parse_float(val)
                    elif col.startswith("final_combined_"):
                        per_cat_final[mtag][cat] = _parse_float(val)
            w.writerow(row)

        # ---- 再写 __ALL__ 行: 4 个分数列用加权, 计数列保留旧行或取新值 ----
        row = ["__ALL__"]
        for col in header[1:]:
            mtag = None
            for prefix in _prefixes:
                if col.startswith(prefix):
                    mtag = col[len(prefix):]
                    break
            if mtag in final_models and col == f"score_{mtag}":
                v = _weighted_all(per_cat_score[mtag])
                row.append("" if v is None else f"{v:.4f}")
            elif mtag in final_models and col == f"yes_rate_{mtag}":
                v = _weighted_all(per_cat_yes[mtag])
                row.append("" if v is None else f"{v:.4f}")
            elif mtag in final_models and col == f"shot_norm_{mtag}":
                v = _weighted_all(per_cat_shot[mtag])
                row.append("" if v is None else f"{v:.4f}")
            elif mtag in final_models and col == f"final_combined_{mtag}":
                v = _weighted_all(per_cat_final[mtag])
                row.append("" if v is None else f"{v:.4f}")
            else:
                # n_specs_* / n_* 继续走与 category 行一致的取值逻辑
                row.append(_get_val("__ALL__", col, mtag))
        w.writerow(row)

    kept = [m for m in old_models if m not in append_models]
    penalty_note = (
        f"  (shot-penalty applied: c<{SHOT_PENALTY_THRESHOLD} -> x*=c on "
        f"score/yes_rate/shot_norm/final_combined; source=TransNetV2 multi)"
        if apply_penalty else ""
    )
    print(f"[aggregate:append:{label}] written: {out_csv}  "
          f"kept={kept}  refreshed/added={append_models}  "
          f"(rule: __ALL__ = 0.5*speech + 0.5*mean(other cats)){penalty_note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--append-models", default="",
                    help="逗号或空格分隔的模型名; 传入则进入'追加模式': "
                         "保留旧 CSV 里其他模型的分数不变, 只重算这些模型的列; "
                         "compare_summary_multi_intersect.csv 追加模式下整张跳过")
    args = ap.parse_args()

    runs_dir = args.runs_dir
    if not os.path.isdir(runs_dir):
        print(f"[ERROR] runs dir not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    append_models: List[str] = []
    if args.append_models.strip():
        raw = args.append_models.replace(",", " ").split()
        append_models = [m for m in raw if m]

    # ---------------- 追加模式分支 ----------------
    if append_models:
        missing = [m for m in append_models
                   if not os.path.isfile(os.path.join(runs_dir, m, "summary.json"))]
        if missing:
            print(f"[ERROR] --append-models 里这些模型在 {runs_dir} 下找不到 summary.json: "
                  f"{missing}", file=sys.stderr)
            sys.exit(1)

        print(f"[aggregate] APPEND MODE  runs_dir={runs_dir}  "
              f"append_models={append_models}")

        _append_one_variant(runs_dir,
                            os.path.join(runs_dir, "compare_summary.csv"),
                            append_models, "all")
        _append_one_variant(runs_dir,
                            os.path.join(runs_dir, "compare_summary_single.csv"),
                            append_models, "single")
        _append_one_variant(runs_dir,
                            os.path.join(runs_dir, "compare_summary_multi.csv"),
                            append_models, "multi")

        intersect_csv = os.path.join(runs_dir, "compare_summary_multi_intersect.csv")
        print(f"[aggregate:append:multi_x] SKIP: {intersect_csv} (kept unchanged; "
              f"it's the three-commercial-model intersection, not affected by new-model append)")

        # 追加模式终端摘要:
        # * 主打印一行 = multi 加权 (含惩罚) 分, 作为该模型 18 项对齐用的最终分
        # * 括号里附带 all/unpenalized 加权分作参考
        multi_shot_rates = _load_shot_correct_rates(SHOT_PENALTY_CSV)
        for m in append_models:
            recs = load_per_question(runs_dir, m)
            multi_recs = [r for r in recs
                          if (r.get("shot_count_kind") or "") == "multi_shot"]
            stats_by_cat = aggregate_one(multi_recs)
            norm_by_cat = aggregate_spec_norm(multi_recs)
            cats_here = sorted({c for c in list(stats_by_cat.keys()) + list(norm_by_cat.keys())
                                if c and c != "__ALL__"})
            pcs, pcy, pcsh, pcf = {}, {}, {}, {}
            for cat in cats_here:
                b = stats_by_cat.get(cat) or {}
                nb = norm_by_cat.get(cat) or {}
                c = multi_shot_rates.get((cat, m))
                pcs[cat] = _penalize_float(_cat_score_mean(b), c)
                pcy[cat] = _penalize_float(_cat_yes_rate(b), c)
                pcsh[cat] = _penalize_float(_cat_shot_norm(nb), c)
                pcf[cat] = _penalize_float(_calc_final_combined(b), c)
            score_mean_v = _weighted_all(pcs)
            yes_rate_v = _weighted_all(pcy)
            shot_norm_v = _weighted_all(pcsh)
            final_v = _weighted_all(pcf)
            score_mean = f"{score_mean_v:.4f}" if score_mean_v is not None else "-"
            yes_rate = f"{yes_rate_v:.4f}" if yes_rate_v is not None else "-"
            shot_norm = f"{shot_norm_v:.4f}" if shot_norm_v is not None else "-"
            final_combined = f"{final_v:.4f}" if final_v is not None else "-"
            nb_all = norm_by_cat.get("__ALL__") or {}
            b_all = stats_by_cat.get("__ALL__") or {}
            n_specs = nb_all.get("n_specs", 0)
            n_q = b_all.get("n_questions", 0)
            print(f"  [append:multi+penalty] {m:<24}  n_q={n_q}  score_mean={score_mean}  "
                  f"yes_rate={yes_rate}  shot_norm={shot_norm}  "
                  f"final={final_combined}  n_specs={n_specs}")
        print(f"  (rule: __ALL__ = 0.5*speech_cat + 0.5*mean(other categories); "
              f"multi shot-penalty applied: c<{SHOT_PENALTY_THRESHOLD} -> x*=c)")
        return

    # ---------------- 原全量分支 ----------------
    models = discover_models(runs_dir)
    if not models:
        print(f"[WARN] no model summary under {runs_dir}")
        return

    # 每个模型只读一次 per_question.jsonl
    per_q_full: Dict[str, List[dict]] = {m: load_per_question(runs_dir, m) for m in models}

    def _filter_by_kind(records: List[dict], kind: Optional[str]) -> List[dict]:
        """kind in {None, 'single_shot', 'multi_shot'}; None = 不过滤 (全样本)."""
        if kind is None:
            return records
        return [r for r in records if (r.get("shot_count_kind") or "") == kind]

    def _aggregate_one_variant(per_q: Dict[str, List[dict]], out_csv: str,
                               write_back_summary: bool, label: str,
                               models_override: Optional[List[str]] = None):
        """对一组 per_q (已经按 single/multi/all 过滤好) 做完整聚合, 写一张 csv,
        必要时把 __ALL__ 结果写回 <model>/summary.json (只 all 那次写).
        models_override: 如果传入, 则只输出这些模型的列 (默认用闭包里的 models).

        对于 multi 相关 label (multi / multi_x) , 4 个分数列会套上镜头正确率惩罚
        (score_<m>/yes_rate_<m>/shot_norm_<m>/final_combined_<m>)."""
        ms: List[str] = models_override if models_override is not None else models
        stats: Dict[str, Dict[str, dict]] = {m: aggregate_one(per_q[m]) for m in ms}
        norm_stats: Dict[str, Dict[str, dict]] = {m: aggregate_spec_norm(per_q[m]) for m in ms}

        # 只有 multi 相关的 label 才启用惩罚; single/all 不加载(键为空表 -> 全不罚)
        apply_penalty = label in ("multi", "multi_x")
        shot_rates: Dict[Tuple[str, str], float] = (
            _load_shot_correct_rates(SHOT_PENALTY_CSV) if apply_penalty else {}
        )

        cats: List[str] = sorted({
            c
            for m in ms
            for c in list(stats[m].keys()) + list(norm_stats[m].keys())
            if c and c != "__ALL__"
        })
        rows_order = cats + ["__ALL__"]

        header = ["category"]
        for m in ms: header.append(f"score_{m}")
        for m in ms: header.append(f"yes_rate_{m}")
        for m in ms: header.append(f"shot_norm_{m}")
        for m in ms: header.append(f"final_combined_{m}")
        for m in ms: header.append(f"n_specs_{m}")
        for m in ms: header.append(f"n_{m}")

        # 预先把每个模型 · 每个 category 的 4 个可比分值算出来 (float 或 None),
        # 便于后面 __ALL__ 行按 speech 50% + 其他类 mean 50% 加权.
        # 若启用了 multi 惩罚, 这里存放的是 **已惩罚** 的 float, 让 __ALL__ 加权也
        # 基于惩罚后的 category 值再合并 (与 action_eval / category 行显示一致).
        per_cat_score:  Dict[str, Dict[str, Optional[float]]] = {m: {} for m in ms}
        per_cat_yes:    Dict[str, Dict[str, Optional[float]]] = {m: {} for m in ms}
        per_cat_shot:   Dict[str, Dict[str, Optional[float]]] = {m: {} for m in ms}
        per_cat_final:  Dict[str, Dict[str, Optional[float]]] = {m: {} for m in ms}
        for m in ms:
            for cat in cats:  # 只挑真实 category, 不含 __ALL__
                b = stats[m].get(cat) or {}
                nb = norm_stats[m].get(cat) or {}
                c = shot_rates.get((cat, m)) if apply_penalty else None
                per_cat_score[m][cat] = _penalize_float(_cat_score_mean(b), c)
                per_cat_yes[m][cat] = _penalize_float(_cat_yes_rate(b), c)
                per_cat_shot[m][cat] = _penalize_float(_cat_shot_norm(nb), c)
                per_cat_final[m][cat] = _penalize_float(_calc_final_combined(b), c)

        # 每个模型的 __ALL__ 加权值 (供 CSV / summary.json / 终端摘要共用)
        weighted_all: Dict[str, Dict[str, Optional[float]]] = {}
        for m in ms:
            weighted_all[m] = {
                "score":     _weighted_all(per_cat_score[m]),
                "yes_rate":  _weighted_all(per_cat_yes[m]),
                "shot_norm": _weighted_all(per_cat_shot[m]),
                "final":     _weighted_all(per_cat_final[m]),
            }

        def _fmt_opt(v: Optional[float]) -> str:
            return "" if v is None else f"{v:.4f}"

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for cat in rows_order:
                row = [cat]
                is_all = (cat == "__ALL__")
                # 该 category (或 __ALL__) 下, 每个模型的镜头正确率 c
                # __ALL__ 行用 (cat="__ALL__", model=m) 键查表
                cur_c: Dict[str, Optional[float]] = {}
                if apply_penalty:
                    for m in ms:
                        cur_c[m] = shot_rates.get((cat, m))
                for m in ms:
                    if is_all:
                        row.append(_fmt_opt(weighted_all[m]["score"]))
                    else:
                        b = stats[m].get(cat) or {}
                        raw = _fmt_mean(b.get("score_sum", 0.0), b.get("score_count", 0))
                        row.append(_apply_linear_penalty(raw, cur_c.get(m)) if apply_penalty else raw)
                for m in ms:
                    if is_all:
                        row.append(_fmt_opt(weighted_all[m]["yes_rate"]))
                    else:
                        b = stats[m].get(cat) or {}
                        raw = _fmt_rate(b.get("yes_count", 0), b.get("n_yesno", 0))
                        row.append(_apply_linear_penalty(raw, cur_c.get(m)) if apply_penalty else raw)
                for m in ms:
                    if is_all:
                        row.append(_fmt_opt(weighted_all[m]["shot_norm"]))
                    else:
                        nb = norm_stats[m].get(cat) or {}
                        raw = _fmt_norm(nb.get("norm_sum", 0.0), nb.get("n_specs", 0))
                        row.append(_apply_linear_penalty(raw, cur_c.get(m)) if apply_penalty else raw)
                for m in ms:
                    if is_all:
                        row.append(_fmt_opt(weighted_all[m]["final"]))
                    else:
                        b = stats[m].get(cat) or {}
                        raw = _fmt_final(b)
                        row.append(_apply_linear_penalty(raw, cur_c.get(m)) if apply_penalty else raw)
                for m in ms:
                    nb = norm_stats[m].get(cat) or {}
                    row.append(nb.get("n_specs", 0))
                for m in ms:
                    b = stats[m].get(cat) or {}
                    row.append(b.get("n_questions", 0))
                w.writerow(row)
        if apply_penalty:
            print(f"[aggregate:{label}] written: {out_csv}  "
                  f"(shot-penalty applied: c<{SHOT_PENALTY_THRESHOLD} -> x*=c on "
                  f"score/yes_rate/shot_norm/final_combined; source=TransNetV2 multi)")
        else:
            print(f"[aggregate:{label}] written: {out_csv}")

        if write_back_summary:
            # 仅 all 那一次把 __ALL__ (加权后) 写回 summary.json
            for m in ms:
                nb_all = dict(norm_stats[m].get("__ALL__") or {})
                b_all = dict(stats[m].get("__ALL__") or {})
                nb_all["__ALL_WEIGHTED_SHOT_NORM__"] = weighted_all[m]["shot_norm"]
                b_all["__ALL_WEIGHTED_FINAL__"] = weighted_all[m]["final"]
                _update_model_summary(runs_dir, m, nb_all, b_all)

        # 终端摘要
        print(f"---- {label} ----")
        for m in ms:
            b = stats[m].get("__ALL__") or {}
            nb = norm_stats[m].get("__ALL__") or {}
            score_mean = _fmt_opt(weighted_all[m]["score"]) or "-"
            yes_rate = _fmt_opt(weighted_all[m]["yes_rate"]) or "-"
            shot_norm = _fmt_opt(weighted_all[m]["shot_norm"]) or "-"
            final_combined = _fmt_opt(weighted_all[m]["final"]) or "-"
            n_specs = nb.get("n_specs", 0)
            n_specs_excl = nb.get("n_specs_excluded", 0)
            n_q = b.get("n_questions", 0)
            print(f"  [{label:<6}] {m:<24}  n_q={n_q}  score_mean={score_mean}  "
                  f"yes_rate={yes_rate}  shot_norm={shot_norm}  "
                  f"final={final_combined}  n_specs={n_specs}"
                  + (f"  n_specs_excl={n_specs_excl}" if n_specs_excl else ""))
        print(f"  (rule: __ALL__ = 0.5*speech_cat + 0.5*mean(other categories))")
        print()

    # ---- 三份输出: 全样本 / single / multi ----
    # 全样本 (兼容老接口, 也是 summary.json 写回的来源)
    _aggregate_one_variant(
        per_q_full,
        out_csv=os.path.join(runs_dir, "compare_summary.csv"),
        write_back_summary=True,
        label="all",
    )
    # 仅 single_shot
    per_q_single = {m: _filter_by_kind(per_q_full[m], "single_shot") for m in models}
    _aggregate_one_variant(
        per_q_single,
        out_csv=os.path.join(runs_dir, "compare_summary_single.csv"),
        write_back_summary=False,
        label="single",
    )
    # 仅 multi_shot
    per_q_multi = {m: _filter_by_kind(per_q_full[m], "multi_shot") for m in models}
    _aggregate_one_variant(
        per_q_multi,
        out_csv=os.path.join(runs_dir, "compare_summary_multi.csv"),
        write_back_summary=False,
        label="multi",
    )

    # ---- 三家商业模型 multi_shot · spec 交集 ----
    # 只在 seedance / wan2.7 / kling-v3 之间统计 (有些 spec 在切片阶段被 pass,
    # 三家通过的 spec_id 不一致). 取三家共同进入评测的 spec_id 交集,
    # 仅这部分 spec 才计入分数, 让对照公平.
    COMMERCIAL = ["seedance", "wan2.7", "kling-v3"]
    commercial_present = [m for m in COMMERCIAL if m in models]
    if len(commercial_present) >= 2:
        spec_sets = []
        for m in commercial_present:
            s = {r.get("spec_id") for r in per_q_multi[m] if r.get("spec_id")}
            spec_sets.append(s)
        intersect_specs = set.intersection(*spec_sets) if spec_sets else set()
        per_q_multi_intersect = {
            m: [r for r in per_q_multi[m] if r.get("spec_id") in intersect_specs]
            for m in commercial_present
        }
        _aggregate_one_variant(
            per_q_multi_intersect,
            out_csv=os.path.join(runs_dir, "compare_summary_multi_intersect.csv"),
            write_back_summary=False,
            label=f"multi_x",
            models_override=commercial_present,
        )
        print(f"[aggregate:multi_intersect] commercial={commercial_present}  "
              f"intersect_specs={len(intersect_specs)}")
    else:
        print(f"[aggregate:multi_intersect] skip: commercial models present={commercial_present}")


if __name__ == "__main__":
    main()
