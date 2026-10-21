#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-model summary tool for Q-Align video scores.

读多个被评模型在 qalign_runs/<model>/results.csv 下的 per-video 打分，
按 t2av-bench specs 的 ``shot_count`` 字段把样本拆成 single / multi 两个桶，
共输出 6 张 csv：

  ./compare_summary.csv               全样本: 行=类别 × 列=模型 mean，附 n_*
  ./compare_summary_single.csv        single_shot 子集
  ./compare_summary_multi.csv         multi_shot  子集
  ./compare_per_video.csv             全样本: 行=<folder>/<filename> × 列=模型 score
  ./compare_per_video_single.csv      single_shot 子集
  ./compare_per_video_multi.csv       multi_shot  子集

results.csv 由 batch_eval.py 产出，列固定为 (folder, video_path, score)。
开源模型 (LTX-2.3 / Ovi / daVinci-MagiHuman) 不生成多镜头视频, 因此
multi 子表中它们的列会留空, 这是预期行为.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple
_REPO_ROOT = Path(__file__).resolve().parents[2]


def read_results_csv(csv_path: Path) -> List[Tuple[str, str, float]]:
    """
    返回 [(folder, filename, score), ...]
    folder 直接来自 results.csv 的 folder 列；filename 取 video_path 的 basename。
    """
    rows: List[Tuple[str, str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"folder", "video_path", "score"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{csv_path} 的列不对，期望含 {required}，实际 {reader.fieldnames}"
            )
        for row in reader:
            folder = row["folder"].strip()
            vp = row["video_path"].strip()
            score_str = row["score"].strip()
            if not vp or not score_str:
                continue
            try:
                score = float(score_str)
            except ValueError:
                continue
            filename = os.path.basename(vp)
            rows.append((folder, filename, score))
    return rows


def fmt(x: Optional[float], nd: int = 6) -> str:
    """None / NaN 留空，其它按 nd 位小数。"""
    if x is None:
        return ""
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return ""


def _stem_of(filename: str) -> str:
    name = os.path.basename(filename)
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    return name


def _load_spec_kind_map(specs_dir: str) -> Dict[str, str]:
    """优先用 evaluation/_common/spec_kind.py，失败则返回空 dict。"""
    if not specs_dir:
        return {}
    here = Path(__file__).resolve().parent
    common = here.parent / "_common"
    if str(common) not in sys.path:
        sys.path.insert(0, str(common))
    try:
        from spec_kind import load_spec_kind_map  # noqa: E402
        return load_spec_kind_map(specs_dir, verbose=True)
    except Exception as e:
        print(f"[WARN] load_spec_kind_map failed: {e}; single/multi 拆桶将被跳过")
        return {}


def _write_compare_summary(
    out_path: Path,
    models: List[str],
    folder_order: List[str],
    model_folder_scores: "OrderedDict[str, Dict[str, List[float]]]",
):
    """写 compare_summary*.csv: 行=类别(含 __ALL__) × 列=各模型 mean，附 n_*"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category"] + list(models) + [f"n_{m}" for m in models]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 每个类别一行
        for folder in folder_order:
            row = {"category": folder}
            for m in models:
                scores = model_folder_scores.get(m, {}).get(folder, [])
                row[m] = fmt(mean(scores) if scores else None)
                row[f"n_{m}"] = str(len(scores))
            writer.writerow(row)

        # __ALL__
        row = {"category": "__ALL__"}
        for m in models:
            all_scores: List[float] = []
            for sl in model_folder_scores.get(m, {}).values():
                all_scores.extend(sl)
            row[m] = fmt(mean(all_scores) if all_scores else None)
            row[f"n_{m}"] = str(len(all_scores))
        writer.writerow(row)

    print(f"[OK] 写出 {out_path}")


def _write_compare_per_video(
    out_path: Path,
    models: List[str],
    keys: List[Tuple[str, str]],
    model_data: "OrderedDict[str, Dict[Tuple[str,str], float]]",
):
    """写 compare_per_video*.csv: 行=<folder>/<filename> × 列=各模型 score，缺失留空。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["key"] + list(models)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for folder, filename in keys:
            row = {"key": f"{folder}/{filename}"}
            for m in models:
                s = model_data.get(m, {}).get((folder, filename))
                row[m] = fmt(s)
            writer.writerow(row)

    print(f"[OK] 写出 {out_path}")


def _filter_by_kind(
    full_data: "OrderedDict[str, Dict[Tuple[str,str], float]]",
    kind_map: Dict[str, str],
    target_kind: str,
):
    """
    按 spec stem 过滤 full_data 得到 (filtered_model_data, filtered_folder_scores,
    filtered_folder_order, filtered_keys)。
    缺失 spec_kind 的样本会被丢掉（既不进 single 也不进 multi）。
    """
    filtered_model_data: "OrderedDict[str, Dict[Tuple[str,str], float]]" = OrderedDict()
    filtered_folder_scores: "OrderedDict[str, Dict[str, List[float]]]" = OrderedDict()
    folder_order: "OrderedDict[str, None]" = OrderedDict()
    keys: "OrderedDict[Tuple[str,str], None]" = OrderedDict()

    for m, per_video in full_data.items():
        sub_pv: Dict[Tuple[str, str], float] = {}
        sub_fs: Dict[str, List[float]] = {}
        for (folder, filename), score in per_video.items():
            stem = _stem_of(filename)
            if kind_map.get(stem, "") != target_kind:
                continue
            sub_pv[(folder, filename)] = score
            sub_fs.setdefault(folder, []).append(score)
            folder_order.setdefault(folder, None)
            keys.setdefault((folder, filename), None)
        filtered_model_data[m] = sub_pv
        filtered_folder_scores[m] = sub_fs

    folder_list = list(folder_order.keys())
    keys_sorted = sorted(keys.keys(), key=lambda kv: (kv[0], kv[1]))
    return filtered_model_data, filtered_folder_scores, folder_list, keys_sorted


def main():
    parser = argparse.ArgumentParser(
        description="对比多个模型在 Q-Align 上的 per-video 打分（含 single/multi 拆桶）。"
    )
    parser.add_argument(
        "--runs_root", type=str, required=True,
        help="qalign_runs 根目录，下面每个被评模型一个子目录，里面有 results.csv",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="参与对比的模型名（即 runs_root 下子目录名），例：LTX-2.3 Ovi daVinci-MagiHuman",
    )
    parser.add_argument(
        "--out_summary", type=str, required=True,
        help="输出: 全样本 类别×模型 均值表 csv（同目录会再写 _single / _multi 两份）",
    )
    parser.add_argument(
        "--out_per_video", type=str, required=True,
        help="输出: 全样本 <folder>/<filename>×模型 逐条分数表 csv（同目录会再写 _single / _multi 两份）",
    )
    parser.add_argument(
        "--results_basename", type=str, default="results.csv",
        help="每个模型目录下 per-video csv 文件名，默认 results.csv",
    )
    parser.add_argument(
        "--specs_dir", type=str,
        default=str(_REPO_ROOT / "t2av-bench/specs"),
        help="t2av-bench/specs 目录, 用于把样本拆成 single_shot / multi_shot。"
             "传空字符串可禁用拆桶。",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs_root 不存在: {runs_root}")

    # 1. 加载每个模型的 results.csv
    model_data: "OrderedDict[str, Dict[Tuple[str,str], float]]" = OrderedDict()
    model_folder_scores: "OrderedDict[str, Dict[str, List[float]]]" = OrderedDict()

    all_folders: "OrderedDict[str, None]" = OrderedDict()
    all_keys: "OrderedDict[Tuple[str,str], None]" = OrderedDict()

    for m in args.models:
        csv_path = runs_root / m / args.results_basename
        if not csv_path.is_file():
            print(f"[WARN] 找不到 {csv_path}，跳过模型 {m}")
            model_data[m] = {}
            model_folder_scores[m] = {}
            continue

        rows = read_results_csv(csv_path)
        print(f"[INFO] {m}: 读到 {len(rows)} 条 from {csv_path}")

        per_video: Dict[Tuple[str, str], float] = {}
        per_folder: Dict[str, List[float]] = {}
        for folder, filename, score in rows:
            key = (folder, filename)
            per_video[key] = score
            per_folder.setdefault(folder, []).append(score)
            all_folders.setdefault(folder, None)
            all_keys.setdefault(key, None)

        model_data[m] = per_video
        model_folder_scores[m] = per_folder

    if not all_keys:
        print("[ERR] 所有模型都没有有效数据，退出。")
        return

    folder_order = sorted(all_folders.keys())
    keys_sorted = sorted(all_keys.keys(), key=lambda kv: (kv[0], kv[1]))

    # 2. 全样本: compare_summary.csv + compare_per_video.csv
    out_summary = Path(args.out_summary).expanduser().resolve()
    out_per_video = Path(args.out_per_video).expanduser().resolve()

    _write_compare_summary(out_summary, args.models, folder_order, model_folder_scores)
    _write_compare_per_video(out_per_video, args.models, keys_sorted, model_data)

    # 3. 按 shot_count 拆 single / multi
    kind_map = _load_spec_kind_map(args.specs_dir)
    if kind_map:
        for kind, suffix in (("single_shot", "_single"), ("multi_shot", "_multi")):
            sub_md, sub_fs, sub_folders, sub_keys = _filter_by_kind(
                model_data, kind_map, kind
            )
            sum_path = out_summary.with_name(out_summary.stem + suffix + out_summary.suffix)
            pv_path = out_per_video.with_name(out_per_video.stem + suffix + out_per_video.suffix)
            _write_compare_summary(sum_path, args.models, sub_folders, sub_fs)
            _write_compare_per_video(pv_path, args.models, sub_keys, sub_md)

    # 4. 在终端打印一张精简的 summary（全样本）
    print("\n========== compare_summary preview (全样本) ==========")
    header = ["category".ljust(20)] + [m.rjust(14) for m in args.models]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header))))

    def print_row(cat: str, getter):
        cells = [cat.ljust(20)]
        for m in args.models:
            val = getter(m)
            cells.append((fmt(val, nd=4) if val is not None else "-").rjust(14))
        print(" | ".join(cells))

    for folder in folder_order:
        print_row(
            folder,
            lambda m, _f=folder: (
                mean(model_folder_scores.get(m, {}).get(_f, []))
                if model_folder_scores.get(m, {}).get(_f) else None
            ),
        )
    print_row(
        "__ALL__",
        lambda m: (
            mean([s for sl in model_folder_scores.get(m, {}).values() for s in sl])
            if any(model_folder_scores.get(m, {}).values()) else None
        ),
    )


if __name__ == "__main__":
    main()
