#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
action_eval 评测点 · 跨模型汇总 (yes/no 版)

输入: <runs-dir>/<model>/per_question.jsonl  (全部 yes/no 题;
        parse_ok=False 或 answer 非 yes/no 的题会被计数排除)
输出:
    <runs-dir>/compare_summary.csv          (全样本)
    <runs-dir>/compare_summary_single.csv   (仅 single_shot specs)
    <runs-dir>/compare_summary_multi.csv    (仅 multi_shot specs)
    <runs-dir>/compare_summary_multi_intersect.csv  (三家商业模型 multi spec 交集)

聚合规则:
  * parse_ok == False 或 answer ∉ {yes, no} -> 排除 (raw 与 max 都不计).
  * 每道 yes/no 题贡献:
      raw += 1 if answer == "yes" else 0
      max += 1
  * shot_norm  = Σ题raw / Σ题max  ∈ [0,1]     (单镜头 yes 率)
  * spec_norm:
      single_shot -> 唯一 shot 的 shot_norm
      multi_shot  -> mean(各 shot 的 shot_norm)  (镜头等权)
  * category 级 / __ALL__ 级 score = mean(spec_norm) over 有效 specs
    (有效 = spec_max>0)
  * single / multi / all 三类 csv 分别只在对应子集上做上述聚合.

CSV 列 (跟 audio_consistency 等 yes/no 工程对齐):
    category,
    yes_rate_<m>...   # 题级 yes 率 (Σyes / Σ有效题)
    score_<m>...      # spec 归一化平均分 = mean(spec_norm)  ∈ [0,1]
    n_specs_<m>...    # 该模型该类下计入的 spec 数
    n_<m>...          # 该模型该类下的题数 (含 parse_fail)
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


MODEL_ORDER_PREFIX: List[str] = ["seedance", "wan2.7", "kling-v3"]


# ============================================================
# 镜头正确率惩罚 (仅 multi_shot 相关 CSV: compare_summary_multi.csv 与
# compare_summary_multi_intersect.csv 的 score_<m> 列生效)
#
# 数据源: TransNetV2 的 multi_shot 镜头正确率
#   路径: <TransNetV2>/runs/compare_summary_multi.csv
#   列格式: category, <model>, ..., n_<model>, ...  (无前缀, 首列后直接就是各模型 c)
#
# 惩罚规则 (线性 · c<0.8 触发):
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
    - category 保留原样 ('action' / 'ambient' / ... / '__ALL__').
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
    # 找出模型集合: 出现 'n_<m>' 的锚点
    models: List[str] = []
    for col in header:
        if col.startswith("n_") and not col.startswith("n_specs_"):
            m = col[len("n_"):]
            if m and m in header:
                models.append(m)
    # 索引列位置 (取模型 c 值的裸列名列位置)
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
    """给 score_<m> 列字符串套线性惩罚.

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


def _q_contrib(rec: dict) -> Tuple[float, float]:
    """单道 yes/no 题贡献 (raw, max). parse_fail / answer 非法 -> (0, 0).

    注: humanoid_integrity (score_1to5) 分项另走 _humanoid_stats, 不走这里.
    """
    if (rec.get("question_aspect") or "") == "humanoid_integrity":
        return 0.0, 0.0
    if (rec.get("question_type") or "") == "score_1to5":
        return 0.0, 0.0
    if not rec.get("parse_ok"):
        return 0.0, 0.0
    ans = rec.get("answer")
    if not isinstance(ans, str):
        return 0.0, 0.0
    a = ans.strip().lower()
    if a == "yes":
        return 1.0, 1.0
    if a == "no":
        return 0.0, 1.0
    return 0.0, 0.0


# ============================================================
# humanoid_integrity (score_1to5, spec 级) 分项聚合
# ============================================================
def _humanoid_stats(records: List[dict]) -> Dict[str, dict]:
    """聚合 humanoid_integrity 分项 (每个 spec 1 条 score_1to5).

    归一化: humanoid_norm = (score - 1) / 4 ∈ [0, 1].
    score=None (无人体 / parse_fail) 的 spec 不计入均值分母.

    返回 {category_or_'__ALL__': {
        'humanoid_score_sum': float,   # 原始 1-5 总和
        'humanoid_norm_sum':  float,   # 归一化后 [0,1] 总和
        'n_humanoid_scored':  int,     # 参与均值的 spec 数
        'n_humanoid_null':    int,     # score=None 的 spec 数 (无人体 or parse_fail)
    }}
    """
    # 先按 spec_id 取唯一 humanoid 记录 (同一 spec 重复先按后写为准)
    per_spec: Dict[str, dict] = {}
    for r in records:
        if (r.get("question_aspect") or "") != "humanoid_integrity":
            continue
        sid = r.get("spec_id")
        if not sid:
            continue
        per_spec[sid] = r  # 后者覆盖前者

    out: Dict[str, dict] = defaultdict(lambda: {
        "humanoid_score_sum": 0.0,
        "humanoid_norm_sum":  0.0,
        "n_humanoid_scored":  0,
        "n_humanoid_null":    0,
    })
    for sid, r in per_spec.items():
        cat = r.get("category") or ""
        sc = r.get("score")
        # parse_fail 或无人体 -> null
        if (not r.get("parse_ok")) or sc is None:
            for k in ("__ALL__", cat):
                if not k:
                    continue
                out[k]["n_humanoid_null"] += 1
            continue
        try:
            si = int(sc)
        except Exception:
            for k in ("__ALL__", cat):
                if not k:
                    continue
                out[k]["n_humanoid_null"] += 1
            continue
        if si not in (1, 2, 3, 4, 5):
            for k in ("__ALL__", cat):
                if not k:
                    continue
                out[k]["n_humanoid_null"] += 1
            continue
        norm = (si - 1) / 4.0
        for k in ("__ALL__", cat):
            if not k:
                continue
            out[k]["humanoid_score_sum"] += si
            out[k]["humanoid_norm_sum"] += norm
            out[k]["n_humanoid_scored"] += 1
    return dict(out)


# final_combined_v2 = 0.6 * yesno_score + 0.4 * humanoid_norm_mean
FINAL_V2_YESNO_WEIGHT = 0.6
FINAL_V2_HUMANOID_WEIGHT = 0.4


def aggregate_stats(records: List[dict]) -> Dict[str, dict]:
    """
    返回 {category_or_'__ALL__': {
        # 题级
        "n_questions":  int,     # 该桶下全部题数 (含 parse_fail; 仅统计 yes/no 题)
        "n_parse_ok":   int,     # parse_ok 数
        "yes_count":    int,     # answer=yes 数 (仅在 parse_ok 里)
        "n_effective":  int,     # 参与 yes_rate 分母的题数 = yes_count + no_count
        # spec 级
        "norm_sum":     float,   # Σ spec_norm
        "n_specs":      int,     # 实际计入的 spec 数 (spec_max>0)
        "n_specs_excluded": int, # spec_max==0 被排除的 spec 数
        # humanoid_integrity (score_1to5, spec 级) 分项 —— 附字段, 不影响旧逻辑
        "humanoid_score_sum": float,
        "humanoid_norm_sum":  float,
        "n_humanoid_scored":  int,
        "n_humanoid_null":    int,
    }}
    """
    # (spec_id, shot_idx) -> [raw_sum, max_sum]
    shot_acc: Dict[Tuple[str, int], List[float]] = defaultdict(lambda: [0.0, 0.0])
    spec_meta: Dict[str, Dict[str, str]] = {}

    # 题级粗统计
    base_counts: Dict[str, dict] = defaultdict(lambda: {
        "n_questions": 0,
        "n_parse_ok":  0,
        "yes_count":   0,
        "n_effective": 0,
    })

    for r in records:
        spec_id = r.get("spec_id")
        shot_idx = r.get("shot_idx")
        if not spec_id or shot_idx is None:
            continue
        cat = r.get("category") or ""

        # humanoid 分项不进入题级 / shot 级统计, 另走 _humanoid_stats
        if (r.get("question_aspect") or "") == "humanoid_integrity":
            continue
        if (r.get("question_type") or "") == "score_1to5":
            continue

        raw, mx = _q_contrib(r)
        # 题级累计
        ans = r.get("answer")
        for k in ("__ALL__", cat):
            if not k:
                continue
            b = base_counts[k]
            b["n_questions"] += 1
            if r.get("parse_ok"):
                b["n_parse_ok"] += 1
            if mx > 0:
                b["n_effective"] += 1
                if isinstance(ans, str) and ans.strip().lower() == "yes":
                    b["yes_count"] += 1

        key = (spec_id, int(shot_idx))
        shot_acc[key][0] += raw
        shot_acc[key][1] += mx
        if spec_id not in spec_meta:
            spec_meta[spec_id] = {
                "category": cat,
                "shot_count_kind": r.get("shot_count_kind") or "",
            }

    # spec_id -> [(shot_norm or None) ...]
    spec_shotnorms: Dict[str, List[Optional[float]]] = defaultdict(list)
    for (spec_id, _shot_idx), (raw, mx) in shot_acc.items():
        spec_shotnorms[spec_id].append((raw / mx) if mx > 0 else None)

    bucket: Dict[str, dict] = defaultdict(lambda: {
        "norm_sum": 0.0,
        "n_specs": 0,
        "n_specs_excluded": 0,
    })

    for spec_id, sn_list in spec_shotnorms.items():
        meta = spec_meta.get(spec_id, {})
        cat = meta.get("category") or ""
        kind = meta.get("shot_count_kind") or ""

        valid_sn = [v for v in sn_list if v is not None]
        targets = [k for k in ("__ALL__", cat) if k]

        if not valid_sn:
            for k in targets:
                bucket[k]["n_specs_excluded"] += 1
            continue

        if kind == "single_shot" or len(sn_list) == 1:
            spec_norm = valid_sn[0]
        else:
            # multi_shot: 镜头等权 (只在有效镜头上平均)
            spec_norm = sum(valid_sn) / len(valid_sn)

        for k in targets:
            bucket[k]["norm_sum"] += spec_norm
            bucket[k]["n_specs"] += 1

    # 合并
    out: Dict[str, dict] = {}
    keys = set(bucket.keys()) | set(base_counts.keys())
    for k in keys:
        b = dict(bucket.get(k, {"norm_sum": 0.0, "n_specs": 0, "n_specs_excluded": 0}))
        bc = base_counts.get(k, {"n_questions": 0, "n_parse_ok": 0, "yes_count": 0, "n_effective": 0})
        b.update(bc)
        out[k] = b

    # 附加 humanoid_integrity 分项统计 (不影响 yes/no 旧字段)
    hstats = _humanoid_stats(records)
    for k in set(hstats.keys()) | set(out.keys()):
        h = hstats.get(k) or {
            "humanoid_score_sum": 0.0,
            "humanoid_norm_sum":  0.0,
            "n_humanoid_scored":  0,
            "n_humanoid_null":    0,
        }
        if k not in out:
            out[k] = {
                "norm_sum": 0.0, "n_specs": 0, "n_specs_excluded": 0,
                "n_questions": 0, "n_parse_ok": 0, "yes_count": 0, "n_effective": 0,
            }
        out[k].update(h)
    return out


def _fmt_rate(num: int, den: int) -> str:
    if den <= 0:
        return ""
    return f"{num / den:.4f}"


def _fmt_norm(norm_sum: float, n: int) -> str:
    if n <= 0:
        return ""
    return f"{norm_sum / n:.4f}"


def _fmt_humanoid_raw(score_sum: float, n_scored: int) -> str:
    """humanoid_score_<m> 列: 原始 1-5 分均值 (保留 2 位小数)."""
    if n_scored <= 0:
        return ""
    return f"{score_sum / n_scored:.2f}"


def _fmt_humanoid_norm(norm_sum: float, n_scored: int) -> str:
    """humanoid_norm_<m> 列: 归一化 [0,1] 均值."""
    if n_scored <= 0:
        return ""
    return f"{norm_sum / n_scored:.4f}"


def _fmt_final_v2(norm_sum: float, n_specs: int,
                  humanoid_norm_sum: float, n_humanoid_scored: int) -> str:
    """final_combined_v2_<m> 列: 0.6*yesno_spec_norm + 0.4*humanoid_norm.

    - 如果 humanoid 未参与计分 (n_humanoid_scored==0), 此模型兼容取 yesno_score 兑底.
    - 如果 yes/no 统计也为空, 返回空串.
    """
    yesno = (norm_sum / n_specs) if n_specs > 0 else None
    humanoid = (humanoid_norm_sum / n_humanoid_scored) if n_humanoid_scored > 0 else None
    if yesno is None and humanoid is None:
        return ""
    if humanoid is None:
        return f"{yesno:.4f}"
    if yesno is None:
        return f"{humanoid:.4f}"
    v = FINAL_V2_YESNO_WEIGHT * yesno + FINAL_V2_HUMANOID_WEIGHT * humanoid
    return f"{v:.4f}"


def _update_model_summary(runs_dir: str, model: str, all_stats: dict) -> None:
    p = os.path.join(runs_dir, model, "summary.json")
    if not os.path.isfile(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    n_specs = int(all_stats.get("n_specs", 0))
    norm_mean = (all_stats.get("norm_sum", 0.0) / n_specs) if n_specs > 0 else None
    data["shot_score_norm_mean"] = (round(norm_mean, 6) if norm_mean is not None else None)
    data["shot_score_norm_n_specs"] = n_specs
    data["shot_score_norm_n_specs_excluded"] = int(all_stats.get("n_specs_excluded", 0))
    data["shot_score_norm_rule"] = (
        "yesno per-question: parse-fail / non yes-no excluded; "
        "raw = 1 if answer==yes else 0, max=1; "
        "shot_norm = sum_raw / sum_max in [0,1]; "
        "spec_norm = mean(shot_norm) for multi_shot (single_shot uses that one shot directly); "
        "category/__ALL__ = mean(spec_norm) over specs with spec_max>0"
    )
    yes_count = int(all_stats.get("yes_count", 0))
    n_eff = int(all_stats.get("n_effective", 0))
    yes_rate = (yes_count / n_eff) if n_eff > 0 else None
    data["yes_rate"] = (round(yes_rate, 6) if yes_rate is not None else None)
    data["yes_rate_n_effective"] = n_eff
    data["final_combined"] = (round(norm_mean, 6) if norm_mean is not None else None)
    data["final_combined_rule"] = "final_combined == shot_score_norm_mean (yes/no, action_eval)"

    # ----- humanoid_integrity 分项 (score_1to5, spec 级) -----
    h_score_sum = float(all_stats.get("humanoid_score_sum", 0.0))
    h_norm_sum  = float(all_stats.get("humanoid_norm_sum", 0.0))
    n_h_scored  = int(all_stats.get("n_humanoid_scored", 0))
    n_h_null    = int(all_stats.get("n_humanoid_null", 0))
    h_score_mean = (h_score_sum / n_h_scored) if n_h_scored > 0 else None
    h_norm_mean  = (h_norm_sum  / n_h_scored) if n_h_scored > 0 else None
    data["humanoid_score_mean_1to5"] = (round(h_score_mean, 4) if h_score_mean is not None else None)
    data["humanoid_score_normed_0to1"] = (round(h_norm_mean, 6) if h_norm_mean is not None else None)
    data["humanoid_score_n_specs_scored"] = n_h_scored
    data["humanoid_score_n_specs_null"] = n_h_null
    data["humanoid_score_rule"] = (
        "per-spec 1-5 integer score on FULL video (strict rubric, worst subject wins); "
        "score=null (no humanoid subject / parse_fail) excluded from mean; "
        "normed = (score - 1) / 4 in [0,1]"
    )

    # final_combined_v2 = 0.6*yesno + 0.4*humanoid_norm (没 humanoid 时兑底只用 yesno)
    if norm_mean is not None and h_norm_mean is not None:
        final_v2 = (FINAL_V2_YESNO_WEIGHT * norm_mean
                    + FINAL_V2_HUMANOID_WEIGHT * h_norm_mean)
    elif norm_mean is not None:
        final_v2 = norm_mean
    elif h_norm_mean is not None:
        final_v2 = h_norm_mean
    else:
        final_v2 = None
    data["final_combined_v2"] = (round(final_v2, 6) if final_v2 is not None else None)
    data["final_combined_v2_rule"] = (
        f"final_combined_v2 = {FINAL_V2_YESNO_WEIGHT}*shot_score_norm_mean "
        f"+ {FINAL_V2_HUMANOID_WEIGHT}*humanoid_score_normed_0to1; "
        "if humanoid missing, fall back to shot_score_norm_mean; "
        "if yes/no missing, fall back to humanoid_score_normed_0to1"
    )

    # 旧字段清理 (从 1-5 分版本迁移过来的可能残留)
    for k in ("raw_mean_score_1to5", "raw_mean_score_n"):
        data.pop(k, None)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------- 追加模式 ----------------
# 用途: 只重算指定模型的列, 保留旧 CSV 里其他模型的数值不变.

def _read_existing_csv(path: str):
    """读旧 CSV, 返回 (models_in_order, {category: {col_name: value}}).

    使用 yes_rate_ 前缀做锚点识别 (yes_rate_ 与其他列前缀无冲突).
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
        if col.startswith("yes_rate_"):
            models.append(col[len("yes_rate_"):])
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
    """给指定 model 算某个 label(all/single/multi) 下的字段, 按 category 分组.

    返回 {category: {yes_rate_<m>: v, score_<m>: v, n_specs_<m>: v, n_<m>: v,
                       humanoid_score_<m>: v, final_combined_v2_<m>: v}}

    注: humanoid_norm_<m> 和 n_humanoid_<m> 不再写入 CSV (细节存 summary.json),
        CSV 末尾只放 humanoid_score_<m> (1-5 均值) + final_combined_v2_<m>.

    label=='multi' 时对 score_<m> 列套线性惩罚 (c<0.8 时 score * c);
    label=='all'/'single' 不动.
    """
    records = load_per_question(runs_dir, model)
    if label == "single":
        records = [r for r in records if (r.get("shot_count_kind") or "") == "single_shot"]
    elif label == "multi":
        records = [r for r in records if (r.get("shot_count_kind") or "") == "multi_shot"]
    stats = aggregate_stats(records)

    shot_rates = _load_shot_correct_rates(SHOT_PENALTY_CSV) if label == "multi" else {}

    out: Dict[str, Dict[str, str]] = {}
    for cat, nb in stats.items():
        h_score_sum = float(nb.get("humanoid_score_sum", 0.0))
        h_norm_sum  = float(nb.get("humanoid_norm_sum", 0.0))
        n_h        = int(nb.get("n_humanoid_scored", 0))
        raw_score = _fmt_norm(nb.get("norm_sum", 0.0), nb.get("n_specs", 0))
        c = shot_rates.get((cat, model))
        out[cat] = {
            f"yes_rate_{model}": _fmt_rate(nb.get("yes_count", 0), nb.get("n_effective", 0)),
            f"score_{model}":    _apply_linear_penalty(raw_score, c),
            f"n_specs_{model}":  str(nb.get("n_specs", 0)),
            f"n_{model}":        str(nb.get("n_questions", 0)),
            f"humanoid_score_{model}": _fmt_humanoid_raw(h_score_sum, n_h),
            f"final_combined_v2_{model}": _fmt_final_v2(
                nb.get("norm_sum", 0.0), nb.get("n_specs", 0),
                h_norm_sum, n_h,
            ),
        }
    return out

# 追加模式下: 前缀识别顺序需要更长的优先, 否则 n_ 会误吃 n_specs_ 等
_APPEND_PREFIXES = (
    "yes_rate_",
    "n_specs_",
    "humanoid_score_",
    "final_combined_v2_",
    "score_",
    "n_",
)


def _append_one_variant(runs_dir: str, out_csv: str,
                         append_models: List[str], label: str) -> None:
    """追加模式核心: 读旧 CSV -> 保留其他模型原值 -> 只重算 append_models -> 写回.

    注意: 若 label=='multi', 需要对 kept (老) 模型的 score_<m> 列同样套线性惩罚,
    避免旧 CSV 里是 '惩罚前' 版本时新老口径不一致.
    """
    old_models, old_data = _read_existing_csv(out_csv)

    final_models: List[str] = list(old_models)
    for m in append_models:
        if m not in final_models:
            final_models.append(m)

    new_stats: Dict[str, Dict[str, Dict[str, str]]] = {}
    for m in append_models:
        new_stats[m] = _compute_one_model_stats(runs_dir, m, label)

    # kept (老) 模型的列一律沿用旧 CSV 原值, 不再重算/回补惩罚 (见下方写行逻辑).
    kept_models = [m for m in old_models if m not in append_models]

    all_cats = set(old_data.keys())
    for m in append_models:
        all_cats.update(new_stats[m].keys())
    cats_sorted = sorted(c for c in all_cats if c and c != "__ALL__")
    rows_order = cats_sorted + ["__ALL__"]

    header = ["category"]
    for m in final_models: header.append(f"yes_rate_{m}")
    for m in final_models: header.append(f"score_{m}")
    for m in final_models: header.append(f"n_specs_{m}")
    for m in final_models: header.append(f"n_{m}")
    for m in final_models: header.append(f"humanoid_score_{m}")
    for m in final_models: header.append(f"final_combined_v2_{m}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cat in rows_order:
            row = [cat]
            old_row = old_data.get(cat, {})
            for col in header[1:]:
                mtag = None
                col_prefix: Optional[str] = None
                for prefix in _APPEND_PREFIXES:
                    if col.startswith(prefix):
                        mtag = col[len(prefix):]
                        col_prefix = prefix
                        break
                if mtag in append_models:
                    val = new_stats[mtag].get(cat, {}).get(col, "")
                    if val == "" and col in old_row:
                        val = old_row[col]
                else:
                    # kept (老) 模型: 直接沿用旧 CSV 里的值, 不再二次处理.
                    # 旧 CSV 的 score_<m> 列本身已是"罚后"值 (写入时 _compute_one_model_stats
                    # 已套过一次线性惩罚), 若在此再套 _apply_linear_penalty 会导致每跑一次
                    # append 就多乘一次 c, 对 c<0.8 的模型反复衰减、最终塌陷到接近 0.
                    val = old_row.get(col, "")
                row.append(val)
            w.writerow(row)

    kept = kept_models
    print(f"[aggregate:append:{label}] written: {out_csv}  "
          f"kept={kept}  refreshed/added={append_models}")


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
        print(f"[aggregate:append:multi_x] SKIP: {intersect_csv} (kept unchanged)")

        for m in append_models:
            recs = load_per_question(runs_dir, m)
            nb = aggregate_stats(recs).get("__ALL__") or {}
            yes_rate = _fmt_rate(nb.get("yes_count", 0), nb.get("n_effective", 0)) or "-"
            score = _fmt_norm(nb.get("norm_sum", 0.0), nb.get("n_specs", 0)) or "-"
            n_specs = nb.get("n_specs", 0)
            n_q = nb.get("n_questions", 0)
            print(f"  [append] {m:<24}  n_q={n_q}  yes_rate={yes_rate}  "
                  f"score={score}  n_specs={n_specs}")
        return

    # ---------------- 原全量分支 ----------------
    models = discover_models(runs_dir)
    if not models:
        print(f"[WARN] no model summary under {runs_dir}")
        return

    per_q_full: Dict[str, List[dict]] = {m: load_per_question(runs_dir, m) for m in models}

    def _filter_by_kind(records: List[dict], kind: Optional[str]) -> List[dict]:
        if kind is None:
            return records
        return [r for r in records if (r.get("shot_count_kind") or "") == kind]

    def _aggregate_one_variant(per_q: Dict[str, List[dict]], out_csv: str,
                               write_back_summary: bool, label: str,
                               models_override: Optional[List[str]] = None):
        ms: List[str] = models_override if models_override is not None else models
        stats: Dict[str, Dict[str, dict]] = {m: aggregate_stats(per_q[m]) for m in ms}

        # 仅 multi / multi_x 需要按 (category, model) 查 c 值套线性惩罚,
        # 其他 label 传空 dict, 让下方统一逻辑走"c 缺失 -> 不罚"分支
        if label in ("multi", "multi_x"):
            shot_rates = _load_shot_correct_rates(SHOT_PENALTY_CSV)
        else:
            shot_rates = {}

        cats: List[str] = sorted({
            c
            for m in ms
            for c in stats[m].keys()
            if c and c != "__ALL__"
        })
        rows_order = cats + ["__ALL__"]

        header = ["category"]
        for m in ms: header.append(f"yes_rate_{m}")
        for m in ms: header.append(f"score_{m}")
        for m in ms: header.append(f"n_specs_{m}")
        for m in ms: header.append(f"n_{m}")
        for m in ms: header.append(f"humanoid_score_{m}")
        for m in ms: header.append(f"final_combined_v2_{m}")

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for cat in rows_order:
                row = [cat]
                for m in ms:
                    nb = stats[m].get(cat) or {}
                    row.append(_fmt_rate(nb.get("yes_count", 0), nb.get("n_effective", 0)))
                for m in ms:
                    nb = stats[m].get(cat) or {}
                    raw_score = _fmt_norm(nb.get("norm_sum", 0.0), nb.get("n_specs", 0))
                    c = shot_rates.get((cat, m))
                    row.append(_apply_linear_penalty(raw_score, c))
                for m in ms:
                    nb = stats[m].get(cat) or {}
                    row.append(nb.get("n_specs", 0))
                for m in ms:
                    nb = stats[m].get(cat) or {}
                    row.append(nb.get("n_questions", 0))
                for m in ms:
                    nb = stats[m].get(cat) or {}
                    row.append(_fmt_humanoid_raw(
                        float(nb.get("humanoid_score_sum", 0.0)),
                        int(nb.get("n_humanoid_scored", 0)),
                    ))
                for m in ms:
                    nb = stats[m].get(cat) or {}
                    row.append(_fmt_final_v2(
                        float(nb.get("norm_sum", 0.0)),
                        int(nb.get("n_specs", 0)),
                        float(nb.get("humanoid_norm_sum", 0.0)),
                        int(nb.get("n_humanoid_scored", 0)),
                    ))
                w.writerow(row)
        print(f"[aggregate:{label}] written: {out_csv}")

        if write_back_summary:
            for m in ms:
                nb_all = stats[m].get("__ALL__") or {}
                _update_model_summary(runs_dir, m, nb_all)

        print(f"---- {label} ----")
        for m in ms:
            nb = stats[m].get("__ALL__") or {}
            yes_rate = _fmt_rate(nb.get("yes_count", 0), nb.get("n_effective", 0)) or "-"
            score = _fmt_norm(nb.get("norm_sum", 0.0), nb.get("n_specs", 0)) or "-"
            n_specs = nb.get("n_specs", 0)
            n_specs_excl = nb.get("n_specs_excluded", 0)
            n_q = nb.get("n_questions", 0)
            print(f"  [{label:<6}] {m:<24}  n_q={n_q}  yes_rate={yes_rate}  "
                  f"score={score}  n_specs={n_specs}"
                  + (f"  n_specs_excl={n_specs_excl}" if n_specs_excl else ""))
        print()

    # 全样本
    _aggregate_one_variant(
        per_q_full,
        out_csv=os.path.join(runs_dir, "compare_summary.csv"),
        write_back_summary=True,
        label="all",
    )
    # single
    per_q_single = {m: _filter_by_kind(per_q_full[m], "single_shot") for m in models}
    _aggregate_one_variant(
        per_q_single,
        out_csv=os.path.join(runs_dir, "compare_summary_single.csv"),
        write_back_summary=False,
        label="single",
    )
    # multi
    per_q_multi = {m: _filter_by_kind(per_q_full[m], "multi_shot") for m in models}
    _aggregate_one_variant(
        per_q_multi,
        out_csv=os.path.join(runs_dir, "compare_summary_multi.csv"),
        write_back_summary=False,
        label="multi",
    )

    # 三家商业模型 multi 交集
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
            label="multi_x",
            models_override=commercial_present,
        )
        print(f"[aggregate:multi_intersect] commercial={commercial_present}  "
              f"intersect_specs={len(intersect_specs)}")
    else:
        print(f"[aggregate:multi_intersect] skip: commercial models present={commercial_present}")


if __name__ == "__main__":
    main()
