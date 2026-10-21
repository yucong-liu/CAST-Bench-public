#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
id_consistency · 合并 DP shard 的 summary.

transcribe_omni.py 在 shard 模式下会把 transcripts/<vid>.json 直接落到
同一个目录(每条视频文件名唯一, 不同 shard 处理不同子集, 不会冲突),
但 summary 是按 shard 切分写到 summary.shard{i}.json. 本脚本把它们合并成
summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


SUM_KEYS = ("n_videos", "n_ok", "n_err", "n_parse_fail", "n_skip", "n_segments_total")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="单模型 runs 目录, 含 summary.shard*.json")
    ap.add_argument("--shard-num", type=int, required=True)
    ap.add_argument("--keep-shards", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    n = args.shard_num
    if n <= 0:
        print(f"[ERROR] shard-num must be >= 1, got {n}", file=sys.stderr)
        sys.exit(1)

    summaries = []
    missing = []
    for i in range(n):
        p = os.path.join(out_dir, f"summary.shard{i}.json")
        if not os.path.isfile(p):
            missing.append(i); continue
        try:
            summaries.append(json.load(open(p, "r", encoding="utf-8")))
        except Exception as e:
            print(f"[WARN] {p}: load fail: {e}", file=sys.stderr)
    if missing:
        print(f"[WARN] missing summary shards: {missing}", file=sys.stderr)
    if not summaries:
        print("[ERROR] no shards loaded", file=sys.stderr)
        sys.exit(2)

    base = summaries[0]
    merged: Dict[str, Any] = {k: v for k, v in base.items() if k not in SUM_KEYS}
    for k in SUM_KEYS:
        merged[k] = sum(int(s.get(k, 0) or 0) for s in summaries)
    merged["shard_num"] = n
    merged["n_shards_merged"] = len(summaries)

    out = os.path.join(out_dir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[merge] wrote {out}")
    print(f"[merge] totals: n_videos={merged.get('n_videos')} ok={merged.get('n_ok')} "
          f"parse_fail={merged.get('n_parse_fail')} err={merged.get('n_err')}")

    if not args.keep_shards:
        removed = 0
        for i in range(n):
            fp = os.path.join(out_dir, f"summary.shard{i}.json")
            if os.path.isfile(fp):
                try:
                    os.remove(fp); removed += 1
                except OSError as e:
                    print(f"[WARN] failed to remove {fp}: {e}", file=sys.stderr)
        print(f"[merge] cleaned {removed} shard files")


if __name__ == "__main__":
    main()
