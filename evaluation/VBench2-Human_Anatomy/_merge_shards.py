#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_merge_shards.py
================
配合 batch_eval.py --num_shards / --shard_id 用.

把 <out_dir>/results.shard*.csv 合并成单一的 <out_dir>/results.csv,
并重新计算 <out_dir>/summary.csv (folder, n, mean_human_anatomy +
__ALL__ + 可选 __FAILED__).  failed.shard*.txt 合并为 failed.txt.

合并完成后, 默认会删除分片中间文件; 加 --keep_shards 保留它们(便于排错).

用法:
    python3 _merge_shards.py --out_dir runs/seedance
    python3 _merge_shards.py --out_dir runs/seedance --keep_shards
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True,
                    help="batch_eval.py 的 --out_dir 目录")
    ap.add_argument("--keep_shards", action="store_true",
                    help="合并后保留 results.shardN.csv / failed.shardN.txt 不删除")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if not out_dir.is_dir():
        print(f"[ERROR] out_dir not found: {out_dir}", file=sys.stderr)
        return 2

    shards = sorted(out_dir.glob("results.shard*.csv"))
    if not shards:
        print(f"[ERROR] no results.shard*.csv under {out_dir}", file=sys.stderr)
        return 2
    print(f"[merge] found {len(shards)} shards under {out_dir}")
    for s in shards:
        print(f"        - {s.name}")

    # --- 合并所有分片的行 (去重 by video_path, 保序) ---
    seen = set()
    rows: list[dict[str, str]] = []
    for s in shards:
        with s.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                vp = row.get("video_path", "")
                if not vp or vp in seen:
                    continue
                seen.add(vp)
                rows.append(row)

    # 按 (folder, video_path) 字典序排, 让最终 csv 顺序稳定
    rows.sort(key=lambda r: (r.get("folder", ""), r.get("video_path", "")))

    res_csv = out_dir / "results.csv"
    with res_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["folder", "video_path", "human_anatomy"])
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {res_csv}  rows={len(rows)}")

    # --- 重算 summary.csv ---
    bucket: dict[str, list[float]] = {}
    for row in rows:
        try:
            v = float(row.get("human_anatomy", ""))
        except (TypeError, ValueError):
            continue
        bucket.setdefault(row.get("folder", ""), []).append(v)

    # 合并 failed.shard*.txt
    fail_files = sorted(out_dir.glob("failed.shard*.txt"))
    failed_lines: list[str] = []
    for ff in fail_files:
        for ln in ff.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                failed_lines.append(ln)

    sum_csv = out_dir / "summary.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder", "n", "mean_human_anatomy"])
        for folder in sorted(bucket.keys()):
            vs = bucket[folder]
            w.writerow([folder, len(vs),
                        f"{mean(vs):.6f}" if vs else ""])
        all_vs = [v for vs in bucket.values() for v in vs]
        if all_vs:
            w.writerow(["__ALL__", len(all_vs), f"{mean(all_vs):.6f}"])
        if failed_lines:
            w.writerow(["__FAILED__", len(failed_lines), ""])
    print(f"[write] {sum_csv}  folders={len(bucket)}  failed={len(failed_lines)}")

    if failed_lines:
        fail_txt = out_dir / "failed.txt"
        fail_txt.write_text("\n".join(failed_lines) + "\n", encoding="utf-8")
        print(f"[write] {fail_txt}  count={len(failed_lines)}")

    # --- 清理分片中间文件 ---
    if not args.keep_shards:
        for s in shards:
            s.unlink()
        for ff in fail_files:
            ff.unlink()
        print(f"[clean] removed {len(shards)} shard csv + {len(fail_files)} shard failed.txt")
    else:
        print("[keep] shard files retained (--keep_shards)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
