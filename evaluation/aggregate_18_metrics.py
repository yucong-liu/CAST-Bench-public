#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/aggregate_18_metrics.py
================================================================

18 项测评指标跨模型汇总脚本 (evaluation/ 侧)

跟 visualization/ 前端**共用同一份分数映射规则** (从 visualization/build_index.py
动态 import SUMMARY_METRICS + build_score_summary), 保证 CSV 里的数字和前端
渲染的树形分数表完全一致.

18 项指标分布 (与 SUMMARY_METRICS 严格对应):
   画面表现 (4):     视频美学 / 运动人体异常检测 / 运动合理性 / 文本渲染能力
   内容一致性 (6):    背景一致性 / 人物主体一致性 / 面部一致性
                     主体正确性 / 镜头生成匹配 / 风格匹配
   音频表现和语义对齐 (8): 音频美学PQ / 声音-发声主体 / 语音-口型（偏移）/ 语音-口型（唇音匹配）
                        音频生成能力 / 情感匹配 / 说话人ID对齐 / 台词正确性

产出 4 份文件到 --output-dir (默认 evaluation/runs_summary/):
   all_models_18metrics.csv              一行一个模型 × (18 raw + 18 norm100 + group + overall)
   all_models_18metrics_single.csv       同上, 但只用 single_shot 数据
   all_models_18metrics_multi.csv        同上, 只用 multi_shot 数据
   all_models_18metrics.md               Markdown 版, 贴汇报/论文用

用法:
   # 全量所有模型
   python3 aggregate_18_metrics.py
   # 追加新模型 (基线模型仍保留在表里做对比)
   python3 aggregate_18_metrics.py --models my_new_model
   # 多个新模型
   python3 aggregate_18_metrics.py --models my_new_model,another_model
   # 只输出新模型行 (不推荐)
   python3 aggregate_18_metrics.py --models my_new_model --no-include-baseline
   # 自定义输出目录
   python3 aggregate_18_metrics.py --output-dir /tmp/scores
================================================================
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
_REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# 路径常量
# -----------------------------------------------------------------------------
EVAL_ROOT   = (_REPO_ROOT / "evaluation")
VIZ_ROOT    = (_REPO_ROOT / "visualization")
BUILD_INDEX = VIZ_ROOT / "build_index.py"
MODELS_TSV  = EVAL_ROOT / "models.tsv"

# 基线模型 (若无 --no-include-baseline, 这些始终纳入汇总, 便于横向对比)
# 与 visualization/build_index.py 里的 _SUMMARY_EXCLUDE_* 语义保持一致:
#   - multi  档不含 LTX-2.3 / Ovi / daVinci-MagiHuman (非 generation/ 布局)
_EXCLUDE_SINGLE = set()
_EXCLUDE_MULTI  = {"LTX-2.3", "Ovi", "daVinci-MagiHuman"}


# -----------------------------------------------------------------------------
# 动态 import visualization/build_index.py
# -----------------------------------------------------------------------------
def _import_build_index():
    """把 visualization/build_index.py 作为模块 import 进来."""
    if not BUILD_INDEX.exists():
        raise FileNotFoundError(
            f"找不到 visualization/build_index.py: {BUILD_INDEX}\n"
            f"该脚本依赖 visualization 工程里的 SUMMARY_METRICS 定义."
        )
    # 让 build_index.py 能找到它自己的 imports (sys.path)
    if str(VIZ_ROOT) not in sys.path:
        sys.path.insert(0, str(VIZ_ROOT))
    spec = importlib.util.spec_from_file_location("viz_build_index", BUILD_INDEX)
    if spec is None or spec.loader is None:
        raise ImportError(f"importlib 无法加载 {BUILD_INDEX}")
    mod = importlib.util.module_from_spec(spec)
    # 关键: 有些顶层代码在 import 时就会跑 (比如 print), 用 stderr 收集
    _stderr_orig = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")  # 抑制 import 期间的 warnings
        spec.loader.exec_module(mod)
    finally:
        sys.stderr.close()
        sys.stderr = _stderr_orig
    return mod


# -----------------------------------------------------------------------------
# 读 models.tsv (拿全量模型清单)
# -----------------------------------------------------------------------------
def load_models_from_tsv(tsv: Path) -> list[str]:
    if not tsv.exists():
        return []
    out: list[str] = []
    with tsv.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts and parts[0]:
                out.append(parts[0])
    return out


# -----------------------------------------------------------------------------
# CSV 落盘: 一行一个模型 × (18 raw + 18 norm100 + group + overall)
# -----------------------------------------------------------------------------
def _fmt(v, digits: int = 4) -> str:
    """None → '-', float → 保留 digits 位小数, 其他 → str"""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _flatten_one_shot(
    kind: str,             # "single" | "multi"
    summary: dict,         # build_score_summary() 的返回
    models: list[str],     # 该 shot 的模型清单 (single/multi 已经排除完毕)
) -> tuple[list[str], list[list[str]]]:
    """把 summary["rows"] flatten 成 CSV 的 (header, rows).

    输出结构 (每一行 = 一个模型):
      model, <metric1_raw>, <metric2_raw>, ..., <metric18_raw>,
             <metric1_norm100>, ..., <metric18_norm100>,
             <group1_score>, <group2_score>, <group3_score>,
             <overall_flat_norm100>,
             <medal>
    """
    rows_metrics = summary["rows"]           # 18 项
    groups       = summary["groups"]         # 3 大组
    group_scores = summary["group_scores"][kind]   # {g: {m: 0-100|None}}
    flat_overall = summary["overall"][f"flat_{kind}"]  # {m: 0-100|None}
    medals       = summary["medals"][kind]   # {m: "gold"|"silver"|"bronze"|None}

    metric_names = [r["metric"] for r in rows_metrics]

    header: list[str] = ["model"]
    # 原始值
    for name in metric_names:
        header.append(f"{name}__raw")
    # 归一化 0-100
    for name in metric_names:
        header.append(f"{name}__norm100")
    # 大组分 (0-100)
    for g in groups:
        header.append(f"group_score__{g}")
    # 综合总分 (0-100)
    header.append("overall_score_norm100")
    # 奖牌
    header.append("medal")

    out_rows: list[list[str]] = []
    for m in models:
        row: list[str] = [m]
        # raw
        for r in rows_metrics:
            row.append(_fmt(r[kind].get(m)))
        # norm100
        for r in rows_metrics:
            row.append(_fmt(r[f"{kind}_norm"].get(m), digits=2))
        # group
        for g in groups:
            row.append(_fmt(group_scores.get(g, {}).get(m), digits=2))
        # overall
        row.append(_fmt(flat_overall.get(m), digits=2))
        # medal
        row.append(medals.get(m) or "-")
        out_rows.append(row)

    return header, out_rows


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"[write] {path}  ({len(rows)} rows × {len(header)} cols)")


# -----------------------------------------------------------------------------
# Markdown 表格 (只输出 norm100 + overall + medal, raw 版列名太多不方便贴)
# -----------------------------------------------------------------------------
def write_markdown(
    path: Path,
    summary: dict,
    single_models: list[str],
    multi_models: list[str],
    highlight_models: list[str],
) -> None:
    lines: list[str] = []
    metrics = summary["rows"]
    groups  = summary["groups"]

    lines.append("# 18 指标跨模型汇总")
    lines.append("")
    lines.append("> 数据来源: 各评测点 `runs/compare_summary_{single,multi}.csv` 的 `__ALL__` 行 (与 visualization 前端一致).")
    lines.append(f"> 生成时间: {os.popen('date +%Y-%m-%dT%H:%M:%S').read().strip()}")
    lines.append("")
    if highlight_models:
        lines.append(f"> **重点关注模型**: `{', '.join(highlight_models)}`")
        lines.append("")

    for kind, mtitle, models in [
        ("single", "Single-Shot 档", single_models),
        ("multi",  "Multi-Shot 档",  multi_models),
    ]:
        if not models:
            continue
        lines.append(f"## {mtitle} (n_models = {len(models)})")
        lines.append("")

        group_scores = summary["group_scores"][kind]
        flat_overall = summary["overall"][f"flat_{kind}"]
        medals       = summary["medals"][kind]

        # 表头
        cols = ["**Model**"]
        for r in metrics:
            arrow = "↑" if r["direction"] == "up" else "↓"
            cols.append(f"{r['metric']}{arrow}")
        for g in groups:
            cols.append(f"**{g}**")
        cols.append("**综合**")
        cols.append("🏅")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")

        # 每一行
        for m in models:
            row = []
            # 加粗 highlight
            if m in highlight_models:
                row.append(f"**{m}**")
            else:
                row.append(m)
            for r in metrics:
                v = r[f"{kind}_norm"].get(m)
                row.append(_fmt(v, digits=1))
            for g in groups:
                v = group_scores.get(g, {}).get(m)
                row.append(_fmt(v, digits=1))
            v = flat_overall.get(m)
            row.append(f"**{_fmt(v, digits=1)}**")
            medal = medals.get(m)
            row.append({"gold": "🥇", "silver": "🥈", "bronze": "🥉"}.get(medal or "", "-"))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("*说明: metric 后的箭头 ↑=高越好 / ↓=低越好. 所有列已线性映射到 0-100 (方向已反转), 缺失填 `-`.*")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path}  ({path.stat().st_size} bytes)")


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="扫 evaluation/ 下 18 个测评点的 compare_summary_{single,multi}.csv, 输出统一汇总 CSV/Markdown."
    )
    ap.add_argument(
        "--models",
        default="",
        help="重点关注的模型 (逗号分隔). 会被高亮; 是否保留基线看 --no-include-baseline.",
    )
    ap.add_argument(
        "--no-include-baseline",
        action="store_true",
        help="只输出 --models 指定的模型, 不带基线模型行 (默认关闭, 即保留基线).",
    )
    ap.add_argument(
        "--output-dir",
        default=str(EVAL_ROOT / "runs_summary"),
        help="产物输出目录 (默认: evaluation/runs_summary/).",
    )
    ap.add_argument(
        "--models-tsv",
        default=str(MODELS_TSV),
        help="用来发现所有可用模型的 models.tsv 路径.",
    )
    args = ap.parse_args()

    out_dir  = Path(args.output_dir)
    tsv_path = Path(args.models_tsv)

    # 1. 加载 models.tsv 得到全量模型清单
    all_models = load_models_from_tsv(tsv_path)
    if not all_models:
        print(f"[warn] {tsv_path} 里没读到任何模型, 只能用 build_index.py 内置清单")

    # 2. 解析 --models 高亮清单
    highlight_models: list[str] = []
    if args.models.strip():
        highlight_models = [
            x.strip() for x in args.models.replace(",", " ").split() if x.strip()
        ]

    # 3. 动态 import visualization/build_index.py
    print(f"[info] importing {BUILD_INDEX} ...")
    viz = _import_build_index()

    # 4. Monkey-patch: 把 SUMMARY_SINGLE_MODELS / SUMMARY_MULTI_MODELS
    #    替换为 (models.tsv 全量 ∪ --models 指定的模型), 应用 exclude 规则.
    #    保持相对顺序: models.tsv 顺序在前, 新模型追加末尾.
    seen: set[str] = set()
    union_models: list[str] = []
    for m in all_models + highlight_models:
        if m and m not in seen:
            seen.add(m)
            union_models.append(m)

    # 若 --no-include-baseline 且指定了 --models: 缩到只有 highlight
    if args.no_include_baseline and highlight_models:
        union_models = [m for m in highlight_models if m in seen]

    new_single = [m for m in union_models if m not in _EXCLUDE_SINGLE]
    new_multi  = [m for m in union_models if m not in _EXCLUDE_MULTI]

    viz.SUMMARY_SINGLE_MODELS = new_single
    viz.SUMMARY_MULTI_MODELS  = new_multi
    print(f"[info] SUMMARY_SINGLE_MODELS ({len(new_single)}): {new_single}")
    print(f"[info] SUMMARY_MULTI_MODELS  ({len(new_multi)}):  {new_multi}")

    # 5. 直接调它内置的 build_score_summary
    print(f"[info] scanning {len(viz.SUMMARY_METRICS)} metric CSVs from evaluation/ ...")
    summary: dict[str, Any] = viz.build_score_summary()

    # 6. Flatten → CSV
    #    a) all_models_18metrics.csv         = 合并 single+multi 到一份 (行=model, 加 shot 列)
    #    b) all_models_18metrics_single.csv  = 只 single 数据
    #    c) all_models_18metrics_multi.csv   = 只 multi 数据

    # ---- b) single 表 ----
    h_s, rows_s = _flatten_one_shot("single", summary, new_single)
    write_csv(out_dir / "all_models_18metrics_single.csv", h_s, rows_s)

    # ---- c) multi 表 ----
    h_m, rows_m = _flatten_one_shot("multi", summary, new_multi)
    write_csv(out_dir / "all_models_18metrics_multi.csv", h_m, rows_m)

    # ---- a) 合并表: single/multi 两组各自作为独立行, 加一个 shot 列 ----
    combined_header = ["shot"] + h_s  # h_s 和 h_m 应该 header 相同
    combined_rows: list[list[str]] = []
    for r in rows_s:
        combined_rows.append(["single"] + r)
    for r in rows_m:
        combined_rows.append(["multi"] + r)
    write_csv(out_dir / "all_models_18metrics.csv", combined_header, combined_rows)

    # ---- d) Markdown ----
    write_markdown(
        out_dir / "all_models_18metrics.md",
        summary,
        new_single, new_multi,
        highlight_models,
    )

    # ---- e) JSON (原始 dict 直接落盘, 方便下游脚本消费) ----
    #    NB: build_score_summary 的 dict 已经能直接 json.dump (只有 str/float/None)
    json_out = out_dir / "all_models_18metrics.json"
    with json_out.open("w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": os.popen('date -Iseconds').read().strip(),
            "eval_root": str(EVAL_ROOT),
            "models_tsv": str(tsv_path),
            "highlight_models": highlight_models,
            "include_baseline": (not args.no_include_baseline),
            "summary": summary,
        }, fh, ensure_ascii=False, indent=2, default=str)
    print(f"[write] {json_out}  ({json_out.stat().st_size} bytes)")

    # 6. 打印一小段人类可读的 preview
    print()
    print("=" * 70)
    print(" 综合总分 (0-100, 越高越好)  —— Single-Shot 档")
    print("=" * 70)
    flat_s = summary["overall"]["flat_single"]
    ranked_s = sorted(
        [(m, v) for m, v in flat_s.items() if v is not None],
        key=lambda x: x[1], reverse=True,
    )
    for i, (m, v) in enumerate(ranked_s, 1):
        mark = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
        arrow_hl = "  ⬅ NEW" if m in highlight_models else ""
        print(f"  {mark} {i:2d}. {m:<30s}  {v:6.2f}{arrow_hl}")
    print()
    print("=" * 70)
    print(" 综合总分 (0-100, 越高越好)  —— Multi-Shot 档")
    print("=" * 70)
    flat_m = summary["overall"]["flat_multi"]
    ranked_m = sorted(
        [(m, v) for m, v in flat_m.items() if v is not None],
        key=lambda x: x[1], reverse=True,
    )
    for i, (m, v) in enumerate(ranked_m, 1):
        mark = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
        arrow_hl = "  ⬅ NEW" if m in highlight_models else ""
        print(f"  {mark} {i:2d}. {m:<30s}  {v:6.2f}{arrow_hl}")
    print()
    print(f"[done] 4 份产物已输出到: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
