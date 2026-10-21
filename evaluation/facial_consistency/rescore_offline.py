#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
facial_consistency · rescore_offline.py
=========================================

不重新检脸/聚类, 只根据 runs/<model>/per_video.jsonl 里已经存下的 metrics
按 batch_eval.py 的打分规则重新算 final_score / breakdown / status,
并重写 per_video.jsonl + summary.json. 跑完调用 aggregate_summary.py 即可
得到对应的 compare_summary*.csv.

打分规则要点:
  1) 露脸即满分 (s_count): expected>0 时, k_pred < expected 不扣分.
     k_pred==0 走 0; 1<=k_pred<=expected_max 给 1.0; k_pred>expected_max
     线性扣分 (保留对幻觉裂簇的惩罚).
  2) 整段无脸 + expected>0: status 改为 `skipped_no_faces`,
     不写 final_score / breakdown, 不参与 mean / s_count / s_stab.
     (per_video.jsonl 里 status=no_faces_detected + final=0 的记录, 此脚本会改写.)
  3) ok_no_faces (expected==0 且无脸) 仍然 final_score=100 进 mean.
  4) no_tracklets / no_clusters 仍然 final_score=0 进 mean (检到脸但
     聚不出来, 不属于"整段无脸"那一档, 保留惩罚).

无 GPU, 无网络, 无 InsightFace 依赖. 几秒内对所有模型重写完毕.

用法:
  python rescore_offline.py --runs-dir runs
  python aggregate_summary.py --runs-dir runs --specs-dir <specs>

如需保留旧文件, 加 --backup 会把原 per_video.jsonl / summary.json 备份成
<name>.bak.<ts>.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple


# ----------------------------
# 新打分规则: 与 batch_eval.py 保持一致
# ----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def pick_primary_clusters(cluster_len: Dict[int, int],
                          total_face_dets: int,
                          primary_ratio_thresh: float = 0.10,
                          min_cluster_dets: int = 30) -> List[int]:
    if total_face_dets <= 0:
        return []
    prim = []
    for cid, n in cluster_len.items():
        try:
            cid_i = int(cid)
            n_i = int(n)
        except Exception:
            continue
        if n_i >= min_cluster_dets and (n_i / total_face_dets) >= primary_ratio_thresh:
            prim.append(cid_i)
    prim.sort(key=lambda c: int(cluster_len.get(c, cluster_len.get(str(c), 0)) or 0),
              reverse=True)
    return prim


def score_count_range_new(k_pred: int, k_min: int, k_max: int) -> float:
    """新版: 露脸即满分, 仅对 k_pred>k_max 保留扣分."""
    k_min = int(k_min); k_max = int(k_max)
    if k_min < 0:
        k_min = 0
    if k_max < k_min:
        k_max = k_min
    k_pred = int(k_pred)

    if k_min == 0 and k_max == 0:
        return 1.0 if k_pred == 0 else 0.0
    if k_pred == 0:
        return 0.0
    if k_pred <= k_max:
        return 1.0
    dist = k_pred - k_max
    width = max(1, k_max - k_min)
    denom = max(1, width)
    return clamp(1.0 - dist / denom, 0.0, 1.0)


def score_stability_primary(primary_cids: List[int],
                            cluster_len: Dict[int, int],
                            cluster_stab: Dict[int, Dict[str, float]],
                            p50_good: float = 0.80,
                            p50_excellent: float = 0.92) -> float:
    if not primary_cids:
        return 0.0
    weights = []; vals = []
    for cid in primary_cids:
        # cluster_len 的 key 在 jsonl 里是 str ("0", "1", ...) 也可能是 int
        n = float(cluster_len.get(cid, cluster_len.get(str(cid), 0)) or 0.0)
        stab = cluster_stab.get(cid, cluster_stab.get(str(cid), {})) or {}
        p50 = float(stab.get("weighted_mean_of_tracklet_p50_sim", 0.0) or 0.0)
        s = clamp((p50 - p50_good) / max(1e-6, (p50_excellent - p50_good)), 0.0, 1.0)
        weights.append(n); vals.append(s)
    wsum = sum(weights)
    if wsum <= 1e-9:
        return float(sum(vals) / len(vals)) if vals else 0.0
    return float(sum(w * v for w, v in zip(weights, vals)) / wsum)


# ----------------------------
# 把单条记录按新规则重打
# ----------------------------
def _normalize_cluster_keys(d: Dict[Any, Any]) -> Dict[int, Any]:
    """把 cluster_len / cluster_stab 这种 key 可能是 str 的 dict 归一成 int."""
    out: Dict[int, Any] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        try:
            out[int(k)] = v
        except Exception:
            continue
    return out


def rescore_one(rec: Dict[str, Any],
                primary_ratio_thresh: float = 0.10,
                min_cluster_dets: int = 30) -> Dict[str, Any]:
    """返回 (按新规则重写过的 rec, 是否改动). 直接修改并返回 rec."""
    status = rec.get("status", "")
    expected = int(rec.get("expected_face_count", 0) or 0)

    # 1) missing_video / failed: 完全不动
    if status in ("missing_video", "failed"):
        return rec

    # 2) ok_no_faces: 满分语义不变
    if status == "ok_no_faces":
        return rec

    # 3) no_faces_detected (老语义) -> 改写为 skipped_no_faces, 删掉 final/breakdown
    if status == "no_faces_detected":
        rec["status"] = "skipped_no_faces"
        for k in ("final_score", "breakdown"):
            rec.pop(k, None)
        # 保留 observed / metrics 字段, 方便审计
        return rec

    # 4) skipped_no_faces: 已经是新语义, 直接保持; 但保险起见删掉残留 final/breakdown
    if status == "skipped_no_faces":
        for k in ("final_score", "breakdown"):
            rec.pop(k, None)
        return rec

    # 5) no_tracklets / no_clusters: 检到脸但聚不出来, 仍 0 分进 mean
    if status in ("no_tracklets", "no_clusters"):
        rec["final_score"] = 0.0
        rec["breakdown"] = {
            "score_count_40": 0.0,
            "score_stability_60": 0.0,
            "s_count": 0.0,
            "s_stability": 0.0,
        }
        return rec

    # 6) ok: 用新 score_count_range 重打 s_count, 重算 final
    if status != "ok":
        # 未知状态: 不动
        return rec

    metrics = rec.get("metrics") or {}
    cluster_len_raw = metrics.get("clusters") or {}
    cluster_stab_raw = metrics.get("cluster_internal_stability_from_tracklets") or {}

    cluster_len = _normalize_cluster_keys(cluster_len_raw)
    cluster_stab = _normalize_cluster_keys(cluster_stab_raw)

    total_face_dets = int(metrics.get("total_face_dets", 0) or 0)

    primary_cids = pick_primary_clusters(
        cluster_len=cluster_len,
        total_face_dets=total_face_dets,
        primary_ratio_thresh=primary_ratio_thresh,
        min_cluster_dets=min_cluster_dets,
    )
    k_pred = len(primary_cids)

    s_count = score_count_range_new(k_pred, expected, expected)
    score_count = 40.0 * s_count
    s_stab = score_stability_primary(primary_cids, cluster_len, cluster_stab)
    score_stab = 60.0 * s_stab
    final = score_count + score_stab

    rec["final_score"] = float(final)
    rec["breakdown"] = {
        "score_count_40": float(score_count),
        "score_stability_60": float(score_stab),
        "s_count": float(s_count),
        "s_stability": float(s_stab),
    }
    rec["observed"] = {
        "primary_cluster_ids": [int(c) for c in primary_cids],
        "k_primary_pred": int(k_pred),
        "cluster_len": {int(k): int(v) for k, v in cluster_len.items()},
    }
    return rec


# ----------------------------
# 一个模型: 重写 per_video.jsonl + summary.json
# ----------------------------
def rescore_model(runs_dir: str, model: str,
                  primary_ratio_thresh: float,
                  min_cluster_dets: int,
                  backup: bool) -> Dict[str, Any]:
    model_dir = os.path.join(runs_dir, model)
    pv_path = os.path.join(model_dir, "per_video.jsonl")
    sm_path = os.path.join(model_dir, "summary.json")

    if not os.path.isfile(pv_path):
        print(f"[{model}] per_video.jsonl not found, skip", file=sys.stderr)
        return {"model": model, "skipped": True}

    if backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        for p in (pv_path, sm_path):
            if os.path.isfile(p):
                shutil.copy2(p, f"{p}.bak.{ts}")

    # 读
    rows: List[Dict[str, Any]] = []
    with open(pv_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass

    # 重打
    n_skipped_zero = 0  # expected==0 的样本: jsonl 不会写, 这里算 0 也无所谓
    n_missing = 0
    n_skipped_no_faces = 0
    n_failed = 0
    counted_scores: List[float] = []
    status_counter: Dict[str, int] = {}
    new_rows: List[Dict[str, Any]] = []

    for r in rows:
        r2 = rescore_one(r,
                         primary_ratio_thresh=primary_ratio_thresh,
                         min_cluster_dets=min_cluster_dets)
        new_rows.append(r2)
        st = r2.get("status", "")
        status_counter[st] = status_counter.get(st, 0) + 1
        if st == "missing_video":
            n_missing += 1
        elif st == "skipped_no_faces":
            n_skipped_no_faces += 1
        elif st == "failed":
            n_failed += 1
        elif st in ("ok", "ok_no_faces"):
            try:
                counted_scores.append(float(r2.get("final_score", 0.0) or 0.0))
            except Exception:
                pass
        elif st in ("no_tracklets", "no_clusters"):
            counted_scores.append(0.0)

    # 写 per_video.jsonl (按 spec_id 排序, 与 merge_shards 一致)
    new_rows.sort(key=lambda r: str(r.get("spec_id", "")))
    with open(pv_path, "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 读旧 summary 保留非计数字段 (model / video_root / config 等), 然后覆盖计数
    old_sm: Dict[str, Any] = {}
    if os.path.isfile(sm_path):
        try:
            with open(sm_path, "r", encoding="utf-8") as f:
                old_sm = json.load(f)
        except Exception:
            old_sm = {}

    n_specs_seen = (n_missing + n_skipped_no_faces + n_failed
                    + len(counted_scores))
    new_sm = dict(old_sm)
    new_sm.update({
        "model": old_sm.get("model", model),
        "n_specs_seen": int(n_specs_seen),
        "n_scored": int(len(counted_scores)),
        "n_skipped_expected_zero": int(old_sm.get("n_skipped_expected_zero", 0) or 0),
        "n_missing_video": int(n_missing),
        "n_skipped_no_faces": int(n_skipped_no_faces),
        "n_failed": int(n_failed),
        "mean_score": (float(sum(counted_scores) / len(counted_scores))
                       if counted_scores else None),
        "expected_face_source": old_sm.get(
            "expected_face_source", "t2av-bench-specs/subjects[type=human]"
        ),
        "rescored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rescore_rule": "release-ge-1: 露脸即满分 (s_count); 整段无脸 -> skipped",
    })
    # 删掉旧字段 (老 batch_eval 写过的)
    new_sm.pop("n_no_faces_detected", None)

    with open(sm_path, "w", encoding="utf-8") as f:
        json.dump(new_sm, f, ensure_ascii=False, indent=2)

    return {
        "model": model,
        "n_specs_seen": n_specs_seen,
        "n_scored": len(counted_scores),
        "n_missing_video": n_missing,
        "n_skipped_no_faces": n_skipped_no_faces,
        "n_failed": n_failed,
        "mean_score": new_sm.get("mean_score"),
        "status_counter": status_counter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True,
                    help="如 evaluation/facial_consistency/runs")
    ap.add_argument("--primary-ratio-thresh", type=float, default=0.10)
    ap.add_argument("--min-primary-cluster-dets", type=int, default=30)
    ap.add_argument("--backup", action="store_true",
                    help="重写前把原 per_video.jsonl / summary.json 备份成 *.bak.<ts>")
    ap.add_argument("--only", default="",
                    help="只重打这几个模型 (逗号分隔), 默认全部")
    args = ap.parse_args()

    runs_dir = args.runs_dir
    if not os.path.isdir(runs_dir):
        print(f"[ERROR] runs dir not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    models: List[str] = []
    only_set = set([s.strip() for s in (args.only or "").split(",") if s.strip()])
    for name in sorted(os.listdir(runs_dir)):
        if name.startswith(("_", ".")):
            continue
        sub = os.path.join(runs_dir, name)
        if not os.path.isdir(sub):
            continue
        if not os.path.isfile(os.path.join(sub, "per_video.jsonl")):
            continue
        if only_set and name not in only_set:
            continue
        models.append(name)

    if not models:
        print(f"[WARN] no model under {runs_dir}")
        return

    print(f"[rescore] models = {models}")
    print(f"[rescore] primary_ratio_thresh={args.primary_ratio_thresh}  "
          f"min_primary_cluster_dets={args.min_primary_cluster_dets}  "
          f"backup={args.backup}")
    print()

    for m in models:
        st = rescore_model(
            runs_dir, m,
            primary_ratio_thresh=args.primary_ratio_thresh,
            min_cluster_dets=args.min_primary_cluster_dets,
            backup=args.backup,
        )
        sc = st.get("mean_score")
        sc_s = "None" if sc is None else f"{sc:.4f}"
        print(f"  [{m:<22}] n_seen={st.get('n_specs_seen'):>4} "
              f"n_scored={st.get('n_scored'):>4} "
              f"n_missing={st.get('n_missing_video'):>4} "
              f"n_skipped_no_faces={st.get('n_skipped_no_faces'):>4} "
              f"n_failed={st.get('n_failed'):>2}  "
              f"mean_score={sc_s}")
        sc_dist = st.get("status_counter") or {}
        if sc_dist:
            dist_s = ", ".join(f"{k}={v}" for k, v in sorted(sc_dist.items()))
            print(f"     status: {dist_s}")

    print()
    print("[rescore] DONE. 接下来跑 aggregate 重新生成 compare_summary*.csv:")
    print(f"  python aggregate_summary.py --runs-dir {runs_dir}")


if __name__ == "__main__":
    main()
