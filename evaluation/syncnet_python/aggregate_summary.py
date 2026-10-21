#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 SyncNet 多模型评测结果。

扫描 runs/<MODEL>/result.csv 的 TOTAL 行，
只保留 mean_abs_offset_frames / mean_min_dist / mean_confidence 三个指标，
输出到 runs/compare_summary.csv。
"""
import csv
import os
import sys
import argparse


METRICS = ["mean_abs_offset_frames", "mean_min_dist", "mean_confidence"]


def load_total_row(csv_path):
    """从单个 result.csv 中读出 TOTAL 行的指标值。"""
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("class", "").strip() == "TOTAL":
                return row
    return None


def _read_existing_summary(csv_path):
    """读旧 compare_summary*.csv, 返回 {model: {metric: value_str}} (保序 dict).

    - 老表可能不存在 (第一次跑) -> 返回空 dict.
    - 老表若列名不一致, 会尽量按 METRICS 里的字段名读, 缺列留 "".
    """
    out = {}
    if not os.path.isfile(csv_path):
        return out
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m = (row.get("model") or "").strip()
                if not m:
                    continue
                out[m] = {k: row.get(k, "") for k in METRICS}
    except Exception:
        return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs",
                    help="存放各模型 result*.csv 的根目录（默认 runs）")
    ap.add_argument("--out_csv", default="runs/compare_summary.csv",
                    help="输出汇总表路径（默认 runs/compare_summary.csv）")
    ap.add_argument("--result_basename", default="result.csv",
                    help="每个模型子目录下要读的 csv 文件名 (默认 result.csv); "
                         "传 result_single.csv / result_multi.csv 可生成 single/multi 子表")
    ap.add_argument("--append_models", default="",
                    help="逗号或空格分隔的模型名; 传入则进入'追加模式': "
                         "读旧 compare_summary*.csv, 保留其他模型的行不变, "
                         "只重算/追加这些模型的行; 新模型追加到表末尾")
    args = ap.parse_args()

    if not os.path.isdir(args.runs_dir):
        print(f"[err] runs_dir 不存在: {args.runs_dir}", file=sys.stderr)
        sys.exit(1)

    # ---------------- 追加模式 ----------------
    append_models = []
    if args.append_models.strip():
        raw = args.append_models.replace(",", " ").split()
        append_models = [m for m in raw if m]

    if append_models:
        # 保留旧行
        old_rows = _read_existing_summary(args.out_csv)  # {model: {metric: v}}
        old_order = list(old_rows.keys())

        # 只对 append_models 里的模型读取新 result*.csv
        new_rows = {}
        for m in append_models:
            csv_path = os.path.join(args.runs_dir, m, args.result_basename)
            if not os.path.isfile(csv_path):
                print(f"[warn] append: {csv_path} 不存在, "
                      f"skip -> {args.out_csv} 里保留 {m} 的旧行 (若有)")
                continue
            total = load_total_row(csv_path)
            if total is None:
                print(f"[warn] append: {csv_path} 没有 TOTAL 行, "
                      f"skip -> {args.out_csv} 里保留 {m} 的旧行 (若有)")
                continue
            new_rows[m] = {k: total.get(k, "") for k in METRICS}

        # 最终顺序: 旧模型顺序 + 新模型追加到末尾
        final_order = list(old_order)
        for m in append_models:
            if m not in final_order:
                final_order.append(m)

        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model"] + METRICS)
            w.writeheader()
            for m in final_order:
                if m in append_models and m in new_rows:
                    row = {"model": m, **new_rows[m]}
                else:
                    # 保留旧行 (含没算出新值的 append 模型 -> 旧行不动)
                    prev = old_rows.get(m, {})
                    row = {"model": m, **{k: prev.get(k, "") for k in METRICS}}
                w.writerow(row)

        kept = [m for m in old_order if m not in append_models]
        refreshed = [m for m in append_models if m in new_rows]
        added = [m for m in append_models
                 if m in new_rows and m not in old_order]
        print(f"[aggregate:append] written: {args.out_csv}  "
              f"kept={kept}  refreshed={refreshed}  added={added}")
        # 终端友好打印新行
        if new_rows:
            col_w = max(len(m) for m in list(new_rows) + ["model"])
            header = f"{'model'.ljust(col_w)}  " + "  ".join(f"{k:>22}" for k in METRICS)
            print(header)
            print("-" * len(header))
            for m in refreshed:
                r = new_rows[m]
                try:
                    line = f"{m.ljust(col_w)}  " + "  ".join(
                        f"{(float(r[k]) if r[k] != '' else float('nan')):>22.4f}" for k in METRICS
                    )
                except Exception:
                    line = f"{m.ljust(col_w)}  " + "  ".join(f"{r[k]:>22}" for k in METRICS)
                print(line)
        return

    # ---------------- 原全量分支 ----------------
    rows = []
    for name in sorted(os.listdir(args.runs_dir)):
        model_dir = os.path.join(args.runs_dir, name)
        csv_path = os.path.join(model_dir, args.result_basename)
        if not (os.path.isdir(model_dir) and os.path.isfile(csv_path)):
            continue
        total = load_total_row(csv_path)
        if total is None:
            print(f"[warn] {csv_path} 没有 TOTAL 行，跳过")
            continue
        rows.append({
            "model": name,
            **{k: total.get(k, "") for k in METRICS},
        })

    if not rows:
        # 子表场景下 (例如 result_multi.csv 不存在), 这是合法的, 不要强制 exit(2)
        print(f"[warn] 未找到任何 {args.result_basename} 的 TOTAL 行, "
              f"输出空表到 {args.out_csv}", file=sys.stderr)
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model"] + METRICS)
            w.writeheader()
        return

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model"] + METRICS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved: {args.out_csv}")
    # 终端友好打印
    col_w = max(len(r["model"]) for r in rows + [{"model": "model"}])
    header = f"{'model'.ljust(col_w)}  " + "  ".join(f"{m:>22}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['model'].ljust(col_w)}  " + "  ".join(
            f"{(float(r[m]) if r[m] != '' else float('nan')):>22.4f}" for m in METRICS
        )
        print(line)


if __name__ == "__main__":
    main()
