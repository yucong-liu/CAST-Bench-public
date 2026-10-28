#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/scoring.py
================================================================
CAST-Bench 总分计算（论文附录 D.4 口径）—— 单一真源。

被两处复用：
  * evaluation/aggregate_18_metrics.py   生成 18 指标总表 / 排行榜
  * visualization/build_index.py         渲染内网看板（不随公开仓库发布）

口径：
  * 18 个指标先按各自 range/direction 线性映射到 0-100（lower-is-better 反向）；
  * 重点指标（面部一致性 / 音频美学PQ / 音频生成能力 / 说话人ID对齐 / 台词正确性）
    各占 10%，合计 50%；
  * 其余指标平分另外 50%：单镜头档 12 项各 1/24，多镜头档 13 项各 1/26
    （CamM 镜头生成匹配只在多镜头档）；
  * 某模型某项缺数据时，在其可用指标上重新归一化权重；
  * 覆盖度门槛：三个大组任一完全没有分数 → 该模型总分记 None。

模型清单与评测根目录由调用方设置：
    import scoring
    scoring.EVAL = Path("/path/to/repo/evaluation")
    scoring.SUMMARY_SINGLE_MODELS = [...]
    scoring.SUMMARY_MULTI_MODELS  = [...]
    summary = scoring.build_score_summary()
================================================================
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# 默认工作区 = 仓库根目录（evaluation/ 的上一级），可用环境变量覆盖
WS = Path(os.environ.get("CAST_BENCH_WS", str(Path(__file__).resolve().parents[1])))
EVAL = WS / "evaluation"


# ===========================================================================
# 分数汇总（Score Summary）
# ---------------------------------------------------------------------------
# 目标：产出一份跨模型 × 跨测评点的总表，前端渲染出模板里的那张树形分数表。
#
# 数据来源全部是各评测点 runs/ 下已经算好的 compare_summary_*.csv。
# 每个测评点由 SUMMARY_METRICS 里的一条 dict 声明。
#
# 归一化：所有测评点值线性映射到 0-100，direction="down" 时反向；
# 总分：按论文附录 D.4 加权（FcC/A-PQ/AuG/SpID/ASR 各 10%，其余指标平分 50%）。
# 覆盖度门槛（方案 A）：某模型如果在任意一个大组下所有测评点都没跑，
# 则总分置 None，避免用只跑几项的数据得出误导性总分。
# ===========================================================================

# ---- 分数汇总参赛名单 ----
# 参赛名单由调用方注入（见下方 SUMMARY_*_MODELS），注入时按这里的排除规则过滤，
# 并保持调用方给出的顺序（表格列顺序）。
#
# single（单镜头档）：不包含 hunyuan_pretrain（多镜头预训练模型，不参赛单镜头档）。
# multi（多镜头档）：不包含 LTX-2.3 / Ovi / daVinci-MagiHuman（非 generation/ 布局，多镜头档
#                   历史上不参赛）。
# 说明：kling-v3-omni 已按用户要求纳入分数汇总（即使目前评测数据为空也占列，
#      单元格显示 "-"），与 Gen-4.5 / happyhorse-1.1 / veo3.1 处理方式一致。
_SUMMARY_EXCLUDE_ALL   = set()
_SUMMARY_EXCLUDE_SINGLE = _SUMMARY_EXCLUDE_ALL | {"hunyuan_pretrain"}
_SUMMARY_EXCLUDE_MULTI  = _SUMMARY_EXCLUDE_ALL | {"LTX-2.3", "Ovi", "daVinci-MagiHuman"}

# 模型清单由调用方注入（顺序即表格列顺序）：
#   build_index.py            -> 用它从 generation/ 发现的 MODELS
#   aggregate_18_metrics.py   -> 用 models.tsv（含 --models 高亮）
SUMMARY_SINGLE_MODELS: list[str] = []
SUMMARY_MULTI_MODELS: list[str] = []

SUMMARY_GROUPS = ["画面表现", "内容一致性", "音频表现和语义对齐"]


SUMMARY_METRICS: list[dict[str, Any]] = [
    # ------------------------- 画面表现 -------------------------
    {
        "g1": "画面表现", "g2": "视频表现", "metric": "视频美学",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "Q-Align/runs/compare_summary_single.csv",
        "src_multi":  "Q-Align/runs/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "",
    },
    {
        "g1": "画面表现", "g2": "运动表现", "metric": "运动人体异常检测",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "VBench2-Human_Anatomy/runs/compare_summary_single.csv",
        "src_multi":  "VBench2-Human_Anatomy/runs/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "",
    },
    {
        "g1": "画面表现", "g2": "运动表现", "metric": "运动合理性",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "action_eval/runs/qwen3_vl/compare_summary_single.csv",
        "src_multi":  "action_eval/runs/qwen3_vl/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score_",
    },
    {
        "g1": "画面表现", "g2": "文本表现", "metric": "文本渲染能力",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "ocr/score_out/reference_full_cpu/compare_summary_ocr_single.csv",
        "src_multi":  "ocr/score_out/reference_full_cpu/compare_summary_ocr_multi.csv",
        "fmt": "long", "model_col": "model", "value_col": "mean_score",
    },
    # ------------------------- 内容一致性 -------------------------
    {
        "g1": "内容一致性", "g2": "一致性保持", "metric": "背景一致性",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "consistency_segmented/runs/compare_summary_single_shot.csv",
        "src_multi":  "consistency_segmented/runs/compare_summary_multi_shot.csv",
        "fmt": "wide", "col_tpl": "bg_",
    },
    {
        "g1": "内容一致性", "g2": "一致性保持", "metric": "人物主体一致性",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "consistency_segmented/runs/compare_summary_single_shot.csv",
        "src_multi":  "consistency_segmented/runs/compare_summary_multi_shot.csv",
        "fmt": "wide", "col_tpl": "subj_",
    },
    {
        "g1": "内容一致性", "g2": "一致性保持", "metric": "面部一致性",
        "direction": "up", "range": (0.0, 100.0),
        "src_single": "facial_consistency/runs/compare_summary_single.csv",
        "src_multi":  "facial_consistency/runs/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score_",
    },
    {
        "g1": "内容一致性", "g2": "视觉语义对齐", "metric": "主体正确性",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "subject_consistency/runs/qwen3_vl/compare_summary_single.csv",
        "src_multi":  "subject_consistency/runs/qwen3_vl/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score_",
    },
    {
        "g1": "内容一致性", "g2": "视觉语义对齐", "metric": "镜头生成匹配",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": None,  # 仅多镜头
        "src_multi":  "TransNetV2/runs/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "",
    },
    {
        "g1": "内容一致性", "g2": "视觉语义对齐", "metric": "风格匹配",
        "direction": "up", "range": (0.0, 10.0),
        "src_single": "style_consistency/runs/qwen3_vl/compare_summary_single.csv",
        "src_multi":  "style_consistency/runs/qwen3_vl/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score10_",
    },
    # ------------------------- 音频表现和语义对齐 -------------------------
    {
        "g1": "音频表现和语义对齐", "g2": "音频表现", "metric": "音频美学PQ",
        "direction": "up", "range": (1.0, 10.0),
        "src_single": "audiobox-aesthetics/runs/compare_summary_PQ_single.csv",
        "src_multi":  "audiobox-aesthetics/runs/compare_summary_PQ_multi.csv",
        "fmt": "wide", "col_tpl": "",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "音画对齐", "metric": "声音-发声主体",
        "direction": "down", "range": (0.0, 1.0),
        "src_single": "Syncformer/runs/compare_summary_24-01-04T16-39-21_single.csv",
        "src_multi":  "Syncformer/runs/compare_summary_24-01-04T16-39-21_multi.csv",
        "fmt": "wide", "col_tpl": "", "col_suffix": "_cont",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "音画对齐", "metric": "语音-口型（偏移）",
        "direction": "down", "range": (0.0, 10.0),
        "src_single": "syncnet_python/runs/compare_summary_single.csv",
        "src_multi":  "syncnet_python/runs/compare_summary_multi.csv",
        "fmt": "long", "model_col": "model", "value_col": "mean_abs_offset_frames",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "音画对齐", "metric": "语音-口型（唇音匹配）",
        "direction": "up", "range": (0.0, 10.0),
        "src_single": "syncnet_python/runs/compare_summary_single.csv",
        "src_multi":  "syncnet_python/runs/compare_summary_multi.csv",
        "fmt": "long", "model_col": "model", "value_col": "mean_confidence",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "听觉语义对齐", "metric": "音频生成能力",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "audio_consistency/runs/qwen3_omni/compare_summary_single.csv",
        "src_multi":  "audio_consistency/runs/qwen3_omni/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "shot_norm_",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "听觉语义对齐", "metric": "情感匹配",
        "direction": "up", "range": (0.0, 1.0),
        # 情感匹配以 split-modality 模式为权威管线 (见 emtion_consistency/SETUP.md).
        "src_single": "emtion_consistency/runs/qwen3_omni_split/compare_summary_single.csv",
        "src_multi":  "emtion_consistency/runs/qwen3_omni_split/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score_",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "听觉语义对齐", "metric": "说话人ID对齐",
        "direction": "up", "range": (0.0, 10.0),
        "src_single": "id_consistency/score_out/judge/judge_summary_single_shot.csv",
        "src_multi":  "id_consistency/score_out/judge/judge_summary_multi_shot.csv",
        "fmt": "long", "model_col": "model", "value_col": "total_avg",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "听觉语义对齐", "metric": "台词正确性",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "asr_eval/runs/compare_summary_single.csv",
        "src_multi":  "asr_eval/runs/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score_",
    },
]


# ===========================================================================
# 总分权重（论文附录 D.4；与 evaluation/README.md 4.2 节一致）
# ---------------------------------------------------------------------------
# 5 个重点指标各 10%，其余指标平分另外 50%：
#   单镜头档：CamM（镜头生成匹配）不参与 → 12 个非重点指标各 0.5/12
#   多镜头档：13 个非重点指标各 0.5/13
# ===========================================================================
METRIC_WEIGHT_HEAVY = 0.10
METRIC_WEIGHT_PRIORITY = ("面部一致性", "音频美学PQ", "音频生成能力", "说话人ID对齐", "台词正确性")
METRIC_WEIGHT_REST_TOTAL = 1.0 - len(METRIC_WEIGHT_PRIORITY) * METRIC_WEIGHT_HEAVY


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """通用小 CSV 读取，返回 (header, rows)。空/异常返回 ([], [])。"""
    import csv as _csv
    if not path.exists():
        return [], []
    try:
        with path.open("r", encoding="utf-8") as fh:
            all_rows = list(_csv.reader(fh))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] read csv failed {path}: {exc}")
        return [], []
    if not all_rows:
        return [], []
    return all_rows[0], all_rows[1:]


def _to_float(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:  # noqa: BLE001
        return None


def _read_wide_all_row(path: Path, col_tpl: str, col_suffix: str,
                       models: list[str], category_filter: str | None = None):
    """wide 格式：默认取 category=='__ALL__' 一行；

    若指定 category_filter（如 'speech'），则取该 category 那一行，
    按 col_tpl+<M>+col_suffix 取值。
    """
    header, rows = _read_csv_rows(path)
    values = {m: None for m in models}
    if not header:
        return values
    try:
        cat_idx = header.index("category")
    except ValueError:
        return values
    target_cat = category_filter if category_filter else "__ALL__"
    all_row = None
    for r in rows:
        if len(r) > cat_idx and r[cat_idx] == target_cat:
            all_row = r
            break
    if all_row is None:
        return values
    hmap = {h: i for i, h in enumerate(header)}
    for m in models:
        col = f"{col_tpl}{m}{col_suffix}"
        if col in hmap and hmap[col] < len(all_row):
            values[m] = _to_float(all_row[hmap[col]])
    return values


def _read_long_by_model(path: Path, model_col: str, value_col: str,
                        models: list[str]):
    """long 格式：每行一个 model；按 models 取 value_col。"""
    header, rows = _read_csv_rows(path)
    values = {m: None for m in models}
    if not header:
        return values
    hmap = {h: i for i, h in enumerate(header)}
    if model_col not in hmap or value_col not in hmap:
        return values
    m_i, v_i = hmap[model_col], hmap[value_col]
    for r in rows:
        if len(r) <= max(m_i, v_i):
            continue
        m = r[m_i].strip()
        if m in values:
            values[m] = _to_float(r[v_i])
    return values


def _normalize_100(raw, lo: float, hi: float, direction: str):
    """把原始值线性归一到 0-100；direction=='down' 时反向。"""
    if raw is None:
        return None
    if hi == lo:
        return 0.0
    x = (raw - lo) / (hi - lo)
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    if direction == "down":
        x = 1.0 - x
    return x * 100.0


def build_score_summary() -> dict:
    """扫描 SUMMARY_METRICS，产出结构化的分数汇总数据。

    返回：
      {
        "single_models": [...], "multi_models": [...],
        "groups": [...],
        "rows": [
          {"g1","g2","metric","direction","range":[lo,hi],
           "single": {model: raw|None},
           "multi":  {model: raw|None},
           "single_norm": {model: 0-100|None},
           "multi_norm":  {model: 0-100|None}}
        ],
        "group_scores": {"single": {g: {m: 0-100|None}}, "multi": {...}},
        "overall": {"weighted_single": {m: v|None}, "weighted_multi": {m: v|None},
                    "flat_single": {m: v|None}, "flat_multi": {m: v|None}},  # flat_* 为兼容别名
        "weights": {"single": {metric: w}, "multi": {metric: w}},
        "medals":  {"single": {m: "gold"|...|None}, "multi": {...}}
      }
    """
    single_models = list(SUMMARY_SINGLE_MODELS)
    multi_models  = list(SUMMARY_MULTI_MODELS)

    rows: list[dict[str, Any]] = []
    for spec in SUMMARY_METRICS:
        fmt = spec["fmt"]
        lo, hi = spec["range"]
        direction = spec["direction"]

        def _load(csv_rel, models):
            if not csv_rel:
                return {m: None for m in models}
            path = EVAL / csv_rel
            if fmt == "wide":
                return _read_wide_all_row(
                    path, spec.get("col_tpl", ""),
                    spec.get("col_suffix", ""), models,
                    category_filter=spec.get("category_filter"),
                )
            if fmt == "long":
                return _read_long_by_model(
                    path, spec.get("model_col", "model"),
                    spec["value_col"], models,
                )
            return {m: None for m in models}

        s_vals = _load(spec.get("src_single"), single_models)
        m_vals = _load(spec.get("src_multi"),  multi_models)

        s_norm = {k: _normalize_100(v, lo, hi, direction) for k, v in s_vals.items()}
        m_norm = {k: _normalize_100(v, lo, hi, direction) for k, v in m_vals.items()}

        rows.append({
            "g1": spec["g1"], "g2": spec["g2"], "metric": spec["metric"],
            "direction": direction, "range": [lo, hi],
            "single": s_vals, "single_norm": s_norm,
            "multi":  m_vals, "multi_norm":  m_norm,
        })

    def _mean_or_none(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    # 大组分数（做覆盖度检查用）
    def _group_scores(kind: str, models: list[str]):
        out = {g: {} for g in SUMMARY_GROUPS}
        nk = f"{kind}_norm"
        for g in SUMMARY_GROUPS:
            g_rows = [r for r in rows if r["g1"] == g]
            for m in models:
                vals = [r[nk].get(m) for r in g_rows]
                out[g][m] = _mean_or_none([v for v in vals if v is not None])
        return out

    gs_single = _group_scores("single", single_models)
    gs_multi  = _group_scores("multi",  multi_models)

    # 方案 A：模型必须在 3 个大组都有分数，才有资格出总分
    def _eligible(gs, m):
        return all(gs.get(g, {}).get(m) is not None for g in SUMMARY_GROUPS)

    # ---- 总分：论文附录 D.4 的加权口径（不是等权平均）----
    # 重点指标 FcC / A-PQ / AuG / SpID / ASR 各 10%（合计 50%）；
    # 其余指标平分另外 50%：单镜头档 12 项（不含仅多镜头的 CamM）各 0.5/12，
    # 多镜头档 13 项各 0.5/13。
    # 某模型某项缺数据时，在其实际可用的指标上重新归一化权重（分母同步缩小），
    # 避免缺一项就被记 0 分。
    def _weights(kind: str, models: list[str]) -> dict[str, float]:
        nk = f"{kind}_norm"
        present = [r for r in rows if any(r[nk].get(m) is not None for m in models)]
        rest = [r for r in present if r["metric"] not in METRIC_WEIGHT_PRIORITY]
        w: dict[str, float] = {}
        for r in present:
            if r["metric"] in METRIC_WEIGHT_PRIORITY:
                w[r["metric"]] = METRIC_WEIGHT_HEAVY
        for r in rest:
            w[r["metric"]] = METRIC_WEIGHT_REST_TOTAL / len(rest)
        return w

    def _weighted(kind: str, models: list[str], gs) -> dict[str, float | None]:
        nk = f"{kind}_norm"
        w = _weights(kind, models)
        out: dict[str, float | None] = {}
        for m in models:
            if not _eligible(gs, m):
                out[m] = None
                continue
            num = 0.0
            den = 0.0
            for r in rows:
                wi = w.get(r["metric"])
                if wi is None:
                    continue
                v = r[nk].get(m)
                if v is None:
                    continue
                num += wi * v
                den += wi
            out[m] = (num / den) if den > 0 else None
        return out

    weights_single = _weights("single", single_models)
    weights_multi = _weights("multi", multi_models)
    weighted_single = _weighted("single", single_models, gs_single)
    weighted_multi = _weighted("multi", multi_models, gs_multi)

    # 奖牌：按加权总分排前 3
    def _medals(overall):
        ranked = sorted(
            [(m, v) for m, v in overall.items() if v is not None],
            key=lambda x: x[1], reverse=True,
        )
        names = ["gold", "silver", "bronze"]
        out = {m: None for m in overall}
        for i, (m, _v) in enumerate(ranked[:3]):
            out[m] = names[i]
        return out

    return {
        "single_models": single_models,
        "multi_models":  multi_models,
        "groups": SUMMARY_GROUPS,
        "rows": rows,
        "group_scores": {"single": gs_single, "multi": gs_multi},
        "overall": {
            "weighted_single": weighted_single,
            "weighted_multi": weighted_multi,
            # 兼容旧字段名（v1 前端/旧脚本）：值同为论文加权口径
            "flat_single": weighted_single,
            "flat_multi": weighted_multi,
        },
        "weights": {"single": weights_single, "multi": weights_multi},
        "medals":  {"single": _medals(weighted_single), "multi": _medals(weighted_multi)},
    }


