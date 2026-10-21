#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject_consistency · 跨模型汇总

自动 listdir(<runs_dir>), 读各 <model>/summary.json + per_question.jsonl,
输出 <runs_dir>/compare_summary.csv (含按 category 拆开的行).

布局 (跨全部 9 个 category):
    第 1 行 (__ALL__): 跨所有 category 的统计
    后续每行: 单 category (speech / action / object / ...)

列:
    category,
    yes_rate_<m>...      Type-1 yes/no 的 yes 占比 (该 category 内)
    score_<m>...         从 score_calculation 写入的 normalized 分 (若有)
    n_specs_<m>...       该 category 实际产出 question 的 spec 数
    n_<m>...             该 category 的 question 总数

输出三份 csv:
    compare_summary.csv               全部 spec
    compare_summary_single.csv        仅 single_shot spec
    compare_summary_multi.csv         仅 multi_shot spec
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]


def discover_models(runs_dir: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(runs_dir)):
        sub = os.path.join(runs_dir, name)
        if not os.path.isdir(sub):
            continue
        if name.startswith(("_", ".")):
            continue
        if os.path.isfile(os.path.join(sub, "summary.json")):
            out.append(name)
    return out


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
    """{category_or_'__ALL__': stats}"""
    bucket: Dict[str, dict] = defaultdict(lambda: {
        "n_questions": 0,
        "n_parse_ok": 0,
        "yes_count": 0,
        "n_yesno": 0,
        "spec_set": set(),
    })
    for r in records:
        cat = r.get("category") or "unknown"
        sid = r.get("spec_id") or ""
        for key in ("__ALL__", cat):
            if not key:
                continue
            b = bucket[key]
            b["n_questions"] += 1
            if r.get("parse_ok"):
                b["n_parse_ok"] += 1
            qt = r.get("question_type", "yesno")
            if qt == "yesno":
                b["n_yesno"] += 1
                if r.get("answer") == "yes":
                    b["yes_count"] += 1
            if sid:
                b["spec_set"].add(sid)
    return bucket


def _fmt_rate(num: int, den: int) -> str:
    if den <= 0:
        return ""
    return f"{num / den:.4f}"


def _update_model_summary(runs_dir: str, model: str, b_all: dict) -> None:
    """把 __ALL__ 的 yes_rate / parse_ok_rate / n_specs 写回 <model>/summary.json"""
    p = os.path.join(runs_dir, model, "summary.json")
    if not os.path.isfile(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    n_yesno = int(b_all.get("n_yesno", 0))
    yes_count = int(b_all.get("yes_count", 0))
    n_q = int(b_all.get("n_questions", 0))
    n_ok = int(b_all.get("n_parse_ok", 0))
    n_specs = len(b_all.get("spec_set") or set())

    yes_rate = (yes_count / n_yesno) if n_yesno > 0 else None
    parse_ok_rate = (n_ok / n_q) if n_q > 0 else None

    data["yes_rate"] = (round(yes_rate, 6) if yes_rate is not None else None)
    data["parse_ok_rate"] = (round(parse_ok_rate, 6) if parse_ok_rate is not None else None)
    data["n_specs_with_questions"] = n_specs

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

    - models_in_order: 从表头里按 yes_rate_<m> 出现顺序抽出的模型名列表.
    - 每个 category 行存成 {col_name: value_str}, 缺列存 "".
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


def _compute_one_model_stats(runs_dir: str, model: str,
                              spec_kind_map: dict, label: str):
    """给指定 model 算某个 label(all/single/multi) 下的 4 个字段, 按 category 分组.

    返回 {category: {yes_rate_<m>: v, score_<m>: v, n_specs_<m>: v, n_<m>: v}}
    """
    records = load_per_question(runs_dir, model)
    if label == "single":
        records = [r for r in records
                   if spec_kind_map.get(r.get("spec_id") or "", "") == "single_shot"]
    elif label == "multi":
        records = [r for r in records
                   if spec_kind_map.get(r.get("spec_id") or "", "") == "multi_shot"]
    stats = aggregate_one(records)

    score_key = {
        "all":    "overall_normalized_score",
        "single": "single_shot_normalized_score",
        "multi":  "multi_shot_normalized_score",
    }.get(label, "overall_normalized_score")
    score_val = ""
    summary_path = os.path.join(runs_dir, model, "summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as sf:
                sd = json.load(sf)
            sv = (sd.get("score_calculation") or {}).get(score_key, "")
            if sv not in ("", None):
                score_val = f"{sv:.4f}"
        except Exception:
            pass

    out: Dict[str, Dict[str, str]] = {}
    for cat, b in stats.items():
        out[cat] = {
            f"yes_rate_{model}": _fmt_rate(int(b.get("yes_count", 0)),
                                            int(b.get("n_yesno", 0))),
            f"score_{model}":    score_val,
            f"n_specs_{model}":  str(len(b.get("spec_set") or set())),
            f"n_{model}":        str(int(b.get("n_questions", 0))),
        }
    return out


def _append_one_variant(runs_dir: str, out_csv: str,
                         append_models: List[str],
                         spec_kind_map: dict, label: str) -> None:
    """追加模式核心: 读旧 CSV -> 保留其他模型原值 -> 只重算 append_models -> 写回."""
    old_models, old_data = _read_existing_csv(out_csv)

    final_models: List[str] = list(old_models)
    for m in append_models:
        if m not in final_models:
            final_models.append(m)

    new_stats: Dict[str, Dict[str, Dict[str, str]]] = {}
    for m in append_models:
        new_stats[m] = _compute_one_model_stats(runs_dir, m, spec_kind_map, label)

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

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cat in rows_order:
            row = [cat]
            old_row = old_data.get(cat, {})
            for col in header[1:]:
                mtag = None
                for prefix in ("yes_rate_", "score_", "n_specs_", "n_"):
                    if col.startswith(prefix):
                        mtag = col[len(prefix):]
                        break
                if mtag in append_models:
                    val = new_stats[mtag].get(cat, {}).get(col, "")
                    if val == "" and col in old_row:
                        val = old_row[col]
                else:
                    val = old_row.get(col, "")
                row.append(val)
            w.writerow(row)

    kept = [m for m in old_models if m not in append_models]
    print(f"[aggregate:append:{label}] written: {out_csv}  "
          f"kept={kept}  refreshed/added={append_models}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True,
                    help="如 evaluation/subject_consistency/runs/qwen3_vl")
    ap.add_argument("--specs-dir", default=os.environ.get(
        "T2AV_SPECS_DIR",
        str(_REPO_ROOT / "t2av-bench/specs"),
    ), help="t2av-bench specs 目录, 用于按 shot_count 拆 single/multi 子表")
    ap.add_argument("--append-models", default="",
                    help="逗号或空格分隔的模型名; 传入则进入'追加模式': "
                         "保留旧 CSV 里其他模型的分数不变, 只重算这些模型的列; "
                         "也不会回写其他模型的 summary.json")
    args = ap.parse_args()

    runs_dir = args.runs_dir
    if not os.path.isdir(runs_dir):
        print(f"[ERROR] runs dir not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    append_models: List[str] = []
    if args.append_models.strip():
        raw = args.append_models.replace(",", " ").split()
        append_models = [m for m in raw if m]

    models = discover_models(runs_dir)
    if not models and not append_models:
        print(f"[WARN] no model summary under {runs_dir}")
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
                            append_models, spec_kind_map, "all")
        if spec_kind_map:
            _append_one_variant(runs_dir,
                                os.path.join(runs_dir, "compare_summary_single.csv"),
                                append_models, spec_kind_map, "single")
            _append_one_variant(runs_dir,
                                os.path.join(runs_dir, "compare_summary_multi.csv"),
                                append_models, spec_kind_map, "multi")

        for m in append_models:
            _score_key = "overall_normalized_score"
            score_info = "-"
            summary_path = os.path.join(runs_dir, m, "summary.json")
            try:
                with open(summary_path, "r", encoding="utf-8") as sf:
                    sd = json.load(sf)
                sv = (sd.get("score_calculation") or {}).get(_score_key, None)
                if sv not in ("", None):
                    score_info = f"{sv:.4f}"
            except Exception:
                pass
            recs = load_per_question(runs_dir, m)
            b_all = aggregate_one(recs).get("__ALL__") or {}
            yes_rate = _fmt_rate(int(b_all.get("yes_count", 0)),
                                 int(b_all.get("n_yesno", 0))) or "-"
            n_q = int(b_all.get("n_questions", 0))
            n_specs = len(b_all.get("spec_set") or set())
            print(f"  [append] {m:<24}  n_q={n_q}  n_specs={n_specs}  "
                  f"yes_rate={yes_rate}  score={score_info}")
        return

    # ---------------- 原全量分支 ----------------
    per_q_full = {m: load_per_question(runs_dir, m) for m in models}

    def _filter_by_kind(records, kind):
        if kind is None:
            return records
        return [r for r in records
                if spec_kind_map.get(r.get("spec_id") or "", "") == kind]

    def _aggregate_one_variant(per_q, out_csv, write_back_summary, label):
        stats = {m: aggregate_one(per_q[m]) for m in models}

        cats = sorted({
            c
            for m in models
            for c in stats[m].keys()
            if c and c != "__ALL__"
        })
        rows_order = cats + ["__ALL__"]

        header = ["category"]
        for m in models: header.append(f"yes_rate_{m}")
        for m in models: header.append(f"score_{m}")
        for m in models: header.append(f"n_specs_{m}")
        for m in models: header.append(f"n_{m}")

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for cat in rows_order:
                row = [cat]
                for m in models:
                    b = stats[m].get(cat) or {}
                    row.append(_fmt_rate(int(b.get("yes_count", 0)),
                                         int(b.get("n_yesno", 0))))
                _score_key = {
                    "all":    "overall_normalized_score",
                    "single": "single_shot_normalized_score",
                    "multi":  "multi_shot_normalized_score",
                }.get(label, "overall_normalized_score")
                for m in models:
                    score_val = ""
                    summary_path = os.path.join(runs_dir, m, "summary.json")
                    if os.path.exists(summary_path):
                        try:
                            with open(summary_path, "r", encoding="utf-8") as sf:
                                summary_data = json.load(sf)
                            sc = summary_data.get("score_calculation") or {}
                            sv = sc.get(_score_key, "")
                            if sv not in ("", None):
                                score_val = f"{sv:.4f}"
                        except Exception:
                            pass
                    row.append(score_val)
                for m in models:
                    b = stats[m].get(cat) or {}
                    row.append(len(b.get("spec_set") or set()))
                for m in models:
                    b = stats[m].get(cat) or {}
                    row.append(int(b.get("n_questions", 0)))
                w.writerow(row)
        print(f"[aggregate:{label}] written: {out_csv}")

        if write_back_summary:
            for m in models:
                b_all = stats[m].get("__ALL__") or {}
                _update_model_summary(runs_dir, m, b_all)

        _score_key = {
            "all":    "overall_normalized_score",
            "single": "single_shot_normalized_score",
            "multi":  "multi_shot_normalized_score",
        }.get(label, "overall_normalized_score")
        for m in models:
            b = stats[m].get("__ALL__") or {}
            yes_rate = _fmt_rate(int(b.get("yes_count", 0)), int(b.get("n_yesno", 0))) or "-"
            n_specs = len(b.get("spec_set") or set())
            n_q = int(b.get("n_questions", 0))

            score_info = "-"
            summary_path = os.path.join(runs_dir, m, "summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as sf:
                        summary_data = json.load(sf)
                    sc = summary_data.get("score_calculation") or {}
                    sv = sc.get(_score_key, None)
                    if sv not in ("", None):
                        score_info = f"{sv:.4f}"
                except Exception:
                    pass

            print(f"  [{label:<6}] {m:<24}  n_q={n_q}  n_specs={n_specs}  "
                  f"yes_rate={yes_rate}  score={score_info}")
        print()

    # ---- 三份输出 ----
    _aggregate_one_variant(
        per_q_full,
        out_csv=os.path.join(runs_dir, "compare_summary.csv"),
        write_back_summary=True,
        label="all",
    )

    if spec_kind_map:
        per_q_single = {m: _filter_by_kind(per_q_full[m], "single_shot") for m in models}
        _aggregate_one_variant(
            per_q_single,
            out_csv=os.path.join(runs_dir, "compare_summary_single.csv"),
            write_back_summary=False,
            label="single",
        )
        per_q_multi = {m: _filter_by_kind(per_q_full[m], "multi_shot") for m in models}
        _aggregate_one_variant(
            per_q_multi,
            out_csv=os.path.join(runs_dir, "compare_summary_multi.csv"),
            write_back_summary=False,
            label="multi",
        )


if __name__ == "__main__":
    main()
