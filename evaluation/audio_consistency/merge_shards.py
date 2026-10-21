#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_consistency · 把 batch_eval.py 在 DP 8x1 模式下产生的 N 个 shard 输出
合并成最终结果, 给下游 aggregate_summary.py 使用.

输入 (在 <out-dir> 下):
    per_question.shard0.jsonl ... per_question.shard{N-1}.jsonl
    summary.shard0.json       ... summary.shard{N-1}.json

输出:
    per_question.jsonl   (全部记录, 按 (spec_id, shot_idx, question, question_type) 排序)
    summary.json         (各计数累加重算)
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
    ap.add_argument("--out-dir", required=True,
                    help="单模型 runs 目录, 含 per_question.shard*.jsonl / summary.shard*.json")
    ap.add_argument("--shard-num", type=int, required=True,
                    help="shard 总数 (与 batch_eval 时一致)")
    ap.add_argument("--keep-shards", action="store_true",
                    help="合并完成后保留 shard 文件 (默认: 自动删除 per_question.shard*.jsonl / summary.shard*.json, 仅保留合并后的 per_question.jsonl + summary.json)")
    args = ap.parse_args()

    out_dir = args.out_dir
    n = args.shard_num
    if n <= 0:
        print(f"[ERROR] shard-num must be >= 1, got {n}", file=sys.stderr)
        sys.exit(1)

    # ---- 收集 records ----
    all_records: List[dict] = []
    missing_pq: List[int] = []
    for i in range(n):
        p = os.path.join(out_dir, f"per_question.shard{i}.jsonl")
        if not os.path.isfile(p):
            missing_pq.append(i)
            continue
        recs = _load_jsonl(p)
        all_records.extend(recs)
        print(f"[merge] shard{i}: per_question={len(recs)}", flush=True)
    if missing_pq:
        print(f"[WARN] missing per_question shards: {missing_pq}", file=sys.stderr)

    # 排序: (spec_id, shot_idx, question_type, question)
    all_records.sort(key=lambda r: (
        str(r.get("spec_id", "")),
        int(r.get("shot_idx", 0) or 0),
        str(r.get("question_type", "")),
        str(r.get("question", "")),
    ))

    out_pq = os.path.join(out_dir, "per_question.jsonl")
    with open(out_pq, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] wrote {out_pq}  (n={len(all_records)})")

    # ---- 合并 summary ----
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

    # 数值字段累加
    SUM_KEYS = [
        "n_questions", "n_yesno_questions", "n_score_questions",
        "n_parse_ok", "n_parse_fail",
    ]
    merged: Dict[str, Any] = {}
    if summaries:
        # 取第一个 shard 的非数值字段 (tag/model_path/manifest/...)
        base = summaries[0]
        for k, v in base.items():
            if k in SUM_KEYS:
                continue
            merged[k] = v
        # n_specs_in_manifest / n_specs_passed 各 shard 都一样, 不需累加
        # (batch_eval 在切片前打印的是全局 manifest, 但写到 summary 里的是
        #  len(manifest)/len(passed_specs), 所有 shard 相同, 这里直接沿用 base)
        for k in SUM_KEYS:
            merged[k] = sum(int(s.get(k, 0) or 0) for s in summaries)
    else:
        print("[ERROR] no summary shards loaded, cannot write summary.json",
              file=sys.stderr)
        sys.exit(2)

    merged["shard_num"] = n
    merged["n_shards_merged"] = len(summaries)

    out_sm = os.path.join(out_dir, "summary.json")
    with open(out_sm, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[merge] wrote {out_sm}")
    print(f"[merge] totals: n_questions={merged.get('n_questions')} "
          f"parse_ok={merged.get('n_parse_ok')} "
          f"parse_fail={merged.get('n_parse_fail')}")

    # ---- 默认清理 shard 中间文件, 只保留合并后的 per_question.jsonl + summary.json ----
    if not args.keep_shards:
        removed = 0
        for i in range(n):
            for fname in (f"per_question.shard{i}.jsonl",
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
        print(f"[merge] --keep-shards: shard files left in place")

if __name__ == "__main__":
    main()
