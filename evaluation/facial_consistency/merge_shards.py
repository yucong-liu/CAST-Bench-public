#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
facial_consistency · merge_shards.py
=====================================

把 batch_eval.py 在 DP 模式 (--shard-num >= 2) 下产生的 N 个 shard 输出
合并成最终的 per_video.jsonl + summary.json, 给下游 aggregate_summary.py 用.

输入 (在 <out-dir> 下):
    per_video.shard0.jsonl ... per_video.shard{N-1}.jsonl
    summary.shard0.json    ... summary.shard{N-1}.json

输出:
    per_video.jsonl    (全部记录, 按 spec_id 排序)
    summary.json       (各计数累加, 重新计算 mean_score)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def _load_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] {path}: bad json line skipped: {e}", file=sys.stderr)
    return out


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-num", type=int, required=True)
    ap.add_argument("--keep-shards", action="store_true",
                    help="合并完保留分片文件 (默认会删掉)")
    args = ap.parse_args()

    out_dir = args.out_dir
    n = args.shard_num
    if n <= 0:
        print(f"[ERROR] shard-num must be >= 1, got {n}", file=sys.stderr)
        sys.exit(1)

    # ---- per_video ----
    all_records: List[dict] = []
    missing_pq: List[int] = []
    for i in range(n):
        p = os.path.join(out_dir, f"per_video.shard{i}.jsonl")
        if not os.path.isfile(p):
            missing_pq.append(i)
            continue
        recs = _load_jsonl(p)
        all_records.extend(recs)
        print(f"[merge] shard{i}: per_video={len(recs)}", flush=True)
    if missing_pq:
        print(f"[WARN] missing per_video shards: {missing_pq}", file=sys.stderr)

    all_records.sort(key=lambda r: str(r.get("spec_id", "")))

    out_pv = os.path.join(out_dir, "per_video.jsonl")
    with open(out_pv, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] wrote {out_pv}  (n={len(all_records)})")

    # ---- summary ----
    summaries: List[Dict[str, Any]] = []
    missing_sm: List[int] = []
    for i in range(n):
        p = os.path.join(out_dir, f"summary.shard{i}.json")
        if not os.path.isfile(p):
            missing_sm.append(i)
            continue
        try:
            summaries.append(_load_json(p))
        except Exception as e:
            print(f"[WARN] {p}: load fail: {e}", file=sys.stderr)
    if missing_sm:
        print(f"[WARN] missing summary shards: {missing_sm}", file=sys.stderr)

    SUM_KEYS = [
        "n_specs_seen",
        "n_scored",
        "n_skipped_expected_zero",
        "n_missing_video",
        "n_no_faces_detected",
        "n_failed",
    ]
    merged: Dict[str, Any] = {}
    if summaries:
        base = summaries[0]
        for k, v in base.items():
            if k in SUM_KEYS or k in ("mean_score", "shard_idx", "shard_num"):
                continue
            merged[k] = v
        for k in SUM_KEYS:
            merged[k] = sum(int(s.get(k, 0) or 0) for s in summaries)
    else:
        print("[ERROR] no summary shards loaded, cannot write summary.json",
              file=sys.stderr)
        sys.exit(2)

    # mean_score: 用合并后的 per_video 重算 (status in ok / ok_no_faces /
    # no_faces_detected / no_tracklets / no_clusters 都参与, status=missing_video / failed 不参与)
    SCORED_STATUSES = {"ok", "ok_no_faces", "no_faces_detected", "no_tracklets", "no_clusters"}
    scores: List[float] = []
    for r in all_records:
        if r.get("status") in SCORED_STATUSES:
            try:
                scores.append(float(r.get("final_score", 0.0) or 0.0))
            except Exception:
                pass
    merged["mean_score"] = (float(sum(scores) / len(scores)) if scores else None)
    merged["shard_num"] = n
    merged["n_shards_merged"] = len(summaries)

    out_sm = os.path.join(out_dir, "summary.json")
    with open(out_sm, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[merge] wrote {out_sm}")
    print(f"[merge] totals: n_scored={merged.get('n_scored')} "
          f"mean_score={merged.get('mean_score')}")

    if not args.keep_shards:
        removed = 0
        for i in range(n):
            for fname in (f"per_video.shard{i}.jsonl",
                          f"summary.shard{i}.json"):
                fp = os.path.join(out_dir, fname)
                if os.path.isfile(fp):
                    try:
                        os.remove(fp)
                        removed += 1
                    except OSError as e:
                        print(f"[WARN] failed to remove {fp}: {e}", file=sys.stderr)
        print(f"[merge] cleaned {removed} shard files (use --keep-shards to keep them)")
    else:
        print("[merge] --keep-shards: shard files left in place")


if __name__ == "__main__":
    main()
