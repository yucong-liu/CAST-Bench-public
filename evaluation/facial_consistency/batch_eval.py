#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
facial_consistency · batch_eval.py
==================================

面部一致性测评 (本地 CV 模型, 无 VLM)。流水线:

  视频 -> InsightFace buffalo_l 检测 -> 贪心追踪成 tracklets ->
  DBSCAN 聚类成"身份簇" -> 选主角簇 (primary clusters) ->
  得分 = 数量匹配 * 40 + 簇内一致性 * 60

人数标签 (expected face count):
  - 不再依赖 Gemini, 直接从 t2av-bench specs (`<category>.jsonl`) 中
    读取 `subjects` 里 `type == "human"` 的元素个数, 作为期望主角脸数.
  - 期望区间 [min_ids, max_ids] = [count, count] (精确匹配).
  - count == 0 的样本会跳过 (与原有 "max_ids == 0 跳过" 行为一致).

视频组织 (与其他评测点一致):
    <video_root>/<category>/<spec_id>.mp4
  其中 category 取自 spec["category"], spec_id 取自 spec["id"].

多模型 jobs.tsv 模式:
    --jobs-tsv <path>: 每行 "<model>\t<video_root>\t<out_dir>", 进程内顺序处理.
    --shard-idx i / --shard-num N: 按 (model_idx, spec_id) hash 取模分片.

输出 (写入每个 model 的 out_dir):
    per_video.shard{i}.jsonl   (N_SHARDS>=2)  /  per_video.jsonl  (N_SHARDS=1)
    summary.shard{i}.json      (N_SHARDS>=2)  /  summary.json     (N_SHARDS=1)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------
# Utility
# ----------------------------
def l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_sim(a, b):
    return float(np.dot(a, b))


def bbox_iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    if inter <= 0:
        return 0.0
    areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = areaA + areaB - inter
    return float(inter / denom) if denom > 0 else 0.0


def weighted_mean(embs, weights):
    embs = np.asarray(embs)
    w = np.asarray(weights).reshape(-1, 1)
    s = (embs * w).sum(axis=0) / max(w.sum(), 1e-12)
    return s


def percentile(arr, q):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def tracklet_internal_stability(tracklet_rec, all_faces, use_weights=True):
    idxs = tracklet_rec['indices']
    mu = tracklet_rec['embedding']  # already l2-normalized
    sims = []
    ws = []
    for i in idxs:
        e = all_faces[i]['embedding']
        sims.append(cosine_sim(e, mu))
        if use_weights:
            f = all_faces[i]
            ws.append(f['det_score'] * math.sqrt(max(f['rel_area'], 1e-12)))
        else:
            ws.append(1.0)

    sims = np.asarray(sims, dtype=np.float64)
    ws = np.asarray(ws, dtype=np.float64)
    wsum = float(ws.sum()) if ws.size else 0.0

    if sims.size == 0:
        return {
            'tracklet_id': int(tracklet_rec['tracklet_id']),
            'n': 0,
            'mean_sim': 0.0, 'std_sim': 0.0, 'var_sim': 0.0,
            'min_sim': 0.0, 'p05_sim': 0.0, 'p50_sim': 0.0, 'p95_sim': 0.0,
        }

    if wsum > 1e-12:
        mean = float((sims * ws).sum() / wsum)
        var = float((ws * (sims - mean) ** 2).sum() / wsum)
    else:
        mean = float(np.mean(sims))
        var = float(np.var(sims))

    return {
        'tracklet_id': int(tracklet_rec['tracklet_id']),
        'n': int(sims.size),
        'mean_sim': float(mean),
        'std_sim': float(math.sqrt(max(var, 0.0))),
        'var_sim': float(var),
        'min_sim': float(np.min(sims)),
        'p05_sim': percentile(sims, 0.05),
        'p50_sim': percentile(sims, 0.50),
        'p95_sim': percentile(sims, 0.95),
    }


# ----------------------------
# Face analyzer (核心检测/追踪/聚类管线, 与原版保持一致)
# ----------------------------
class MultiFaceConsistencyV2:
    def __init__(self, model_name='buffalo_l', ctx_id=0, det_size=(640, 640)):
        import onnxruntime as ort
        print(f"[insightface] ORT providers: {ort.get_available_providers()}", flush=True)
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def process_video(self, video_path, sample_rate=1, min_face_px=30,
                      min_det_score=0.6, min_rel_area=0.0005):
        cap = cv2.VideoCapture(video_path)
        all_faces = []
        frame_idx = 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_area = max(1, width * height)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                faces = self.app.get(frame)
                for face in faces:
                    b = face.bbox.astype(float).tolist()
                    w = b[2] - b[0]; h = b[3] - b[1]
                    if w < min_face_px or h < min_face_px:
                        continue
                    if float(face.det_score) < float(min_det_score):
                        continue
                    rel_area = (w * h) / frame_area
                    if rel_area < min_rel_area:
                        continue
                    emb = np.asarray(face.embedding, dtype=np.float32)
                    emb = l2norm(emb)
                    all_faces.append({
                        'frame_idx': frame_idx,
                        'embedding': emb,
                        'bbox': b,
                        'det_score': float(face.det_score),
                        'rel_area': float(rel_area),
                    })
            frame_idx += 1
        cap.release()
        return all_faces

    def build_tracklets(self, all_faces, max_gap_frames=3, min_iou=0.1,
                        min_cos=0.35, w_cos=0.7, w_iou=0.3):
        if not all_faces:
            return []

        by_frame = defaultdict(list)
        for i, f in enumerate(all_faces):
            by_frame[f['frame_idx']].append(i)
        frames_sorted = sorted(by_frame.keys())

        tracklets = []
        active = {}

        def new_track(idx):
            tid = len(tracklets)
            tracklets.append({'tracklet_id': tid, 'indices': [idx]})
            active[tid] = {
                'last_frame': all_faces[idx]['frame_idx'],
                'last_bbox': all_faces[idx]['bbox'],
                'last_emb': all_faces[idx]['embedding'],
                'indices': [idx],
            }
            all_faces[idx]['tracklet_id'] = tid

        for idx in by_frame[frames_sorted[0]]:
            new_track(idx)

        for fno in frames_sorted[1:]:
            to_del = []
            for tid, st in active.items():
                if fno - st['last_frame'] > max_gap_frames:
                    tracklets[tid]['indices'] = st['indices']
                    to_del.append(tid)
            for tid in to_del:
                del active[tid]

            det_indices = by_frame[fno]
            if not det_indices:
                continue

            candidates = []
            for di in det_indices:
                d = all_faces[di]
                best = None
                for tid, st in active.items():
                    iou = bbox_iou(d['bbox'], st['last_bbox'])
                    if iou < min_iou:
                        continue
                    cos = cosine_sim(d['embedding'], st['last_emb'])
                    if cos < min_cos:
                        continue
                    score = w_cos * cos + w_iou * iou
                    if (best is None) or (score > best[0]):
                        best = (score, tid, cos, iou)
                if best is not None:
                    candidates.append((best[0], di, best[1], best[2], best[3]))

            candidates.sort(reverse=True, key=lambda x: x[0])
            used_dets = set()
            used_tracks = set()
            for score, di, tid, cos, iou in candidates:
                if di in used_dets or tid in used_tracks:
                    continue
                used_dets.add(di); used_tracks.add(tid)
                st = active[tid]
                st['last_frame'] = fno
                st['last_bbox'] = all_faces[di]['bbox']
                st['last_emb'] = all_faces[di]['embedding']
                st['indices'].append(di)
                all_faces[di]['tracklet_id'] = tid

            for di in det_indices:
                if di not in used_dets:
                    new_track(di)

        for tid, st in active.items():
            tracklets[tid]['indices'] = st['indices']
        return tracklets

    def tracklet_embeddings(self, all_faces, tracklets, min_len=3):
        recs = []
        for tr in tracklets:
            idxs = tr['indices']
            if len(idxs) < min_len:
                continue
            embs = [all_faces[i]['embedding'] for i in idxs]
            ws = []
            for i in idxs:
                f = all_faces[i]
                ws.append(f['det_score'] * math.sqrt(max(f['rel_area'], 1e-12)))
            mean = weighted_mean(embs, ws)
            mean = l2norm(mean)
            frames = [all_faces[i]['frame_idx'] for i in idxs]
            recs.append({
                'tracklet_id': tr['tracklet_id'],
                'n': len(idxs),
                'start_frame': int(min(frames)),
                'end_frame': int(max(frames)),
                'embedding': mean,
                'indices': idxs,
                'weights': ws,
                'mean_det_score': float(np.mean([all_faces[i]['det_score'] for i in idxs])),
            })
        return recs

    def cluster_tracklets(self, tracklet_recs, eps=0.55, min_samples=1):
        if not tracklet_recs:
            return None
        X = np.stack([r['embedding'] for r in tracklet_recs], axis=0)
        X = l2norm(X)
        clustering = DBSCAN(eps=float(eps), min_samples=int(min_samples),
                            metric='euclidean').fit(X)
        labels = clustering.labels_
        identities = defaultdict(list)
        for i, lab in enumerate(labels):
            tracklet_recs[i]['cluster_id'] = int(lab)
            if lab == -1:
                continue
            identities[int(lab)].append(tracklet_recs[i])
        return identities, tracklet_recs, labels, float(eps)

    def propagate_cluster_to_faces(self, all_faces, tracklet_recs):
        tmap = {r['tracklet_id']: r.get('cluster_id', -1) for r in tracklet_recs}
        for f in all_faces:
            tid = f.get('tracklet_id', None)
            if tid is None:
                f['cluster_id'] = -1
            else:
                f['cluster_id'] = int(tmap.get(tid, -1))

    def consistency_metrics(self, all_faces, tracklet_recs):
        if not all_faces:
            return {}
        cluster_len = defaultdict(int)
        noise_len = 0
        for r in tracklet_recs:
            lab = int(r.get('cluster_id', -1))
            if lab == -1:
                noise_len += r['n']
            else:
                cluster_len[lab] += r['n']
        total = sum(cluster_len.values()) + noise_len
        if total == 0:
            return {}
        sorted_clusters = sorted(cluster_len.items(), key=lambda x: x[1], reverse=True)
        main_cluster, main_len = (sorted_clusters[0][0], sorted_clusters[0][1]) \
            if sorted_clusters else (-1, 0)
        main_ratio = main_len / total if total else 0.0
        noise_ratio = noise_len / total if total else 0.0

        tracklet_internal = []
        for r in tracklet_recs:
            if int(r.get('cluster_id', -1)) == -1:
                continue
            tracklet_internal.append(tracklet_internal_stability(r, all_faces, use_weights=True))

        by_cluster_items = defaultdict(list)
        for st in tracklet_internal:
            tid = st['tracklet_id']
            rec = next((r for r in tracklet_recs if int(r['tracklet_id']) == int(tid)), None)
            if rec is None:
                continue
            cid = int(rec.get('cluster_id', -1))
            if cid == -1:
                continue
            by_cluster_items[cid].append((st, int(rec['n'])))

        cluster_internal_from_tracklets = {}
        for cid, items in by_cluster_items.items():
            weights = np.asarray([w for _, w in items], dtype=np.float64)
            wsum = float(weights.sum()) if weights.size else 0.0
            mean_sims = np.asarray([st['mean_sim'] for st, _ in items], dtype=np.float64)
            p05_sims = np.asarray([st['p05_sim'] for st, _ in items], dtype=np.float64)
            p50_sims = np.asarray([st['p50_sim'] for st, _ in items], dtype=np.float64)

            if wsum > 1e-12:
                w_mean = float((mean_sims * weights).sum() / wsum)
                w_p05 = float((p05_sims * weights).sum() / wsum)
                w_p50 = float((p50_sims * weights).sum() / wsum)
            else:
                w_mean = float(np.mean(mean_sims)) if mean_sims.size else 0.0
                w_p05 = float(np.mean(p05_sims)) if p05_sims.size else 0.0
                w_p50 = float(np.mean(p50_sims)) if p50_sims.size else 0.0

            cluster_internal_from_tracklets[cid] = {
                'weighted_mean_of_tracklet_mean_sim': w_mean,
                'weighted_mean_of_tracklet_p05_sim': w_p05,
                'weighted_mean_of_tracklet_p50_sim': w_p50,
            }

        return {
            'total_face_dets': int(len(all_faces)),
            'total_tracklets': int(len(tracklet_recs)),
            'clusters': {int(k): int(v) for k, v in cluster_len.items()},
            'main_cluster_id': int(main_cluster),
            'main_cluster_ratio': float(main_ratio),
            'noise_ratio': float(noise_ratio),
            'tracklet_internal_stability': tracklet_internal,
            'cluster_internal_stability_from_tracklets': cluster_internal_from_tracklets,
        }


# ----------------------------
# Scoring (主角簇选取 + 数量 40 + 稳定性 60)
# ----------------------------
def pick_primary_clusters(cluster_len: Dict[int, int],
                          total_face_dets: int,
                          primary_ratio_thresh: float = 0.10,
                          min_cluster_dets: int = 30) -> List[int]:
    if total_face_dets <= 0:
        return []
    prim = []
    for cid, n in cluster_len.items():
        if n >= min_cluster_dets and (n / total_face_dets) >= primary_ratio_thresh:
            prim.append(int(cid))
    prim.sort(key=lambda c: cluster_len.get(c, 0), reverse=True)
    return prim


def score_count_range(k_pred: int, k_min: int, k_max: int) -> float:
    k_min = int(k_min); k_max = int(k_max)
    if k_min < 0:
        k_min = 0
    if k_max < k_min:
        k_max = k_min
    k_pred = int(k_pred)

    if k_min == 0 and k_max == 0:
        return 1.0 if k_pred == 0 else 0.0
    if k_min <= k_pred <= k_max:
        return 1.0
    if k_pred < k_min:
        dist = k_min - k_pred
    else:
        dist = k_pred - k_max
    width = max(1, k_max - k_min)
    denom = max(1, width)
    return clamp(1.0 - dist / denom, 0.0, 1.0)


def score_stability_primary(primary_cids: List[int],
                            cluster_len: Dict[int, int],
                            cluster_stab: Dict[int, Dict[str, float]],
                            total_face_dets: int,
                            p50_good: float = 0.80,
                            p50_excellent: float = 0.92) -> float:
    if not primary_cids:
        return 0.0
    weights = []; vals = []
    for cid in primary_cids:
        n = float(cluster_len.get(cid, 0))
        stab = cluster_stab.get(cid, {})
        p50 = float(stab.get('weighted_mean_of_tracklet_p50_sim', 0.0))
        s = clamp((p50 - p50_good) / max(1e-6, (p50_excellent - p50_good)), 0.0, 1.0)
        weights.append(n); vals.append(s)
    wsum = sum(weights)
    if wsum <= 1e-9:
        return float(np.mean(vals)) if vals else 0.0
    return float(sum(w * v for w, v in zip(weights, vals)) / wsum)


def compute_final_score(expected_min: int,
                        expected_max: int,
                        metrics: Dict[str, Any],
                        primary_ratio_thresh: float,
                        min_cluster_dets: int) -> Dict[str, Any]:
    cluster_len = metrics.get("clusters", {}) or {}
    total_face_dets = int(metrics.get("total_face_dets", 0) or 0)
    cluster_stab = metrics.get("cluster_internal_stability_from_tracklets", {}) or {}

    primary_cids = pick_primary_clusters(
        cluster_len=cluster_len,
        total_face_dets=total_face_dets,
        primary_ratio_thresh=primary_ratio_thresh,
        min_cluster_dets=min_cluster_dets,
    )
    k_primary_pred = len(primary_cids)

    s_count = score_count_range(k_primary_pred, expected_min, expected_max)
    score_count = 40.0 * s_count
    s_stab = score_stability_primary(primary_cids, cluster_len, cluster_stab, total_face_dets)
    score_stab = 60.0 * s_stab
    final = score_count + score_stab
    return {
        "final_score": float(final),
        "breakdown": {
            "score_count_40": float(score_count),
            "score_stability_60": float(score_stab),
            "s_count": float(s_count),
            "s_stability": float(s_stab),
        },
        "observed": {
            "primary_cluster_ids": [int(c) for c in primary_cids],
            "k_primary_pred": int(k_primary_pred),
            "cluster_len": {int(k): int(v) for k, v in cluster_len.items()},
        },
    }


# ----------------------------
# specs IO: 直接从 t2av-bench specs 读人数标签
# ----------------------------
def _count_human_subjects(spec: Dict[str, Any], include_anthropomorphic: bool = False) -> int:
    """主角脸数 = subjects 里 type == 'human' 的数量.

    `anthropomorphic` 类型 (会说话的卡通动物等) 默认不算人脸:
    InsightFace 训练数据是真实人脸, 卡通脸通常检不到. 想纳入可设 True.
    """
    subs = spec.get("subjects") or []
    n = 0
    for s in subs:
        t = (s or {}).get("type", "")
        if t == "human":
            n += 1
        elif include_anthropomorphic and t == "anthropomorphic":
            n += 1
    return n


def load_specs(specs_dir: str,
               include_anthropomorphic: bool = False,
               categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """扫 <specs_dir>/*.jsonl, 返回 [{spec_id, category, expected_face_count,
    shot_count, prompt_summary, raw}, ...].

    若 ``categories`` 非空, 只保留 ``spec["category"] in categories`` 的记录
    (常见用法: 只评测有人的桶 speech / non_verbal_vocal / action).
    """
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(specs_dir):
        raise SystemExit(f"[batch_eval] specs_dir not found: {specs_dir}")

    cat_filter = set(categories) if categories else None

    for fname in sorted(os.listdir(specs_dir)):
        if not fname.endswith(".jsonl"):
            continue
        if fname.startswith("_"):
            continue
        path = os.path.join(specs_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        spec = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[batch_eval] {fname}:{line_no} json decode error: {e}",
                              file=sys.stderr)
                        continue
                    sid = spec.get("id")
                    cat = spec.get("category")
                    if not sid or not cat:
                        continue
                    if cat_filter is not None and cat not in cat_filter:
                        continue
                    cnt = _count_human_subjects(spec, include_anthropomorphic=include_anthropomorphic)
                    out.append({
                        "spec_id": sid,
                        "category": cat,
                        "expected_face_count": int(cnt),
                        "shot_count": spec.get("shot_count", ""),
                        "scene_setting": (spec.get("scene_setting") or "")[:512],
                    })
        except OSError as e:
            print(f"[batch_eval] cannot open {path}: {e}", file=sys.stderr)

    out.sort(key=lambda x: x["spec_id"])
    return out


# ----------------------------
# jobs.tsv parsing
# ----------------------------
def parse_jobs_tsv(path: str) -> List[Tuple[str, str, str]]:
    """Return list of (model_name, video_root, out_dir)."""
    jobs: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            jobs.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return jobs


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


# ----------------------------
# 单个 (model, spec) 评测
# ----------------------------
def evaluate_one_spec(
    analyzer: MultiFaceConsistencyV2,
    video_path: str,
    expected_count: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """跑完整流水线, 返回结果 dict (status / final_score / metrics 等)."""
    expected_min = int(expected_count)
    expected_max = int(expected_count)

    if not os.path.isfile(video_path):
        return {"status": "missing_video"}

    try:
        all_faces = analyzer.process_video(
            video_path,
            sample_rate=cfg["sample_rate"],
            min_face_px=cfg["min_face_px"],
            min_det_score=cfg["min_det_score"],
            min_rel_area=cfg["min_rel_area"],
        )

        if len(all_faces) == 0:
            # 没检测到任何人脸. 如果 expected==0, 直接满分; 否则 0 分.
            if expected_count == 0:
                return {
                    "status": "ok_no_faces",
                    "final_score": 100.0,
                    "breakdown": {
                        "score_count_40": 40.0,
                        "score_stability_60": 60.0,
                        "s_count": 1.0,
                        "s_stability": 1.0,
                    },
                    "observed": {"primary_cluster_ids": [], "k_primary_pred": 0, "cluster_len": {}},
                    "metrics": {"total_face_dets": 0, "total_tracklets": 0, "clusters": {}},
                }
            return {
                "status": "no_faces_detected",
                "final_score": 0.0,
                "breakdown": {
                    "score_count_40": 0.0,
                    "score_stability_60": 0.0,
                    "s_count": 0.0,
                    "s_stability": 0.0,
                },
                "observed": {"primary_cluster_ids": [], "k_primary_pred": 0, "cluster_len": {}},
                "metrics": {"total_face_dets": 0, "total_tracklets": 0, "clusters": {}},
            }

        tracklets = analyzer.build_tracklets(
            all_faces,
            max_gap_frames=cfg["max_gap_frames"],
            min_iou=cfg["min_iou"],
            min_cos=cfg["min_cos"],
        )
        tracklet_recs = analyzer.tracklet_embeddings(
            all_faces, tracklets, min_len=cfg["min_tracklet_len"],
        )
        if len(tracklet_recs) == 0:
            return {"status": "no_tracklets", "final_score": 0.0}

        out_cl = analyzer.cluster_tracklets(
            tracklet_recs,
            eps=cfg["dbscan_eps"],
            min_samples=cfg["dbscan_min_samples"],
        )
        if out_cl is None:
            return {"status": "no_clusters", "final_score": 0.0}
        identities, tracklet_recs, labels, used_eps = out_cl
        analyzer.propagate_cluster_to_faces(all_faces, tracklet_recs)
        metrics = analyzer.consistency_metrics(all_faces, tracklet_recs)

        scored = compute_final_score(
            expected_min=expected_min,
            expected_max=expected_max,
            metrics=metrics,
            primary_ratio_thresh=cfg["primary_ratio_thresh"],
            min_cluster_dets=cfg["min_primary_cluster_dets"],
        )

        # 不把 tracklet_internal_stability 完整 dump (太大), 只留聚类级别
        light_metrics = {
            "total_face_dets": metrics.get("total_face_dets", 0),
            "total_tracklets": metrics.get("total_tracklets", 0),
            "clusters": metrics.get("clusters", {}),
            "main_cluster_id": metrics.get("main_cluster_id", -1),
            "main_cluster_ratio": metrics.get("main_cluster_ratio", 0.0),
            "noise_ratio": metrics.get("noise_ratio", 0.0),
            "cluster_internal_stability_from_tracklets":
                metrics.get("cluster_internal_stability_from_tracklets", {}),
        }
        return {
            "status": "ok",
            "dbscan": {"eps": float(used_eps), "min_samples": int(cfg["dbscan_min_samples"])},
            "metrics": light_metrics,
            **scored,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


# ----------------------------
# 处理一个模型的所有 specs
# ----------------------------
def run_one_model(
    model_idx: int,
    model_name: str,
    video_root: str,
    out_dir: str,
    spec_records: List[Dict[str, Any]],
    analyzer: MultiFaceConsistencyV2,
    cfg: Dict[str, Any],
    shard_idx: int,
    shard_num: int,
    limit: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)

    # 决定 per_video 输出文件名 (sharded vs final)
    if shard_num >= 2:
        per_video_path = os.path.join(out_dir, f"per_video.shard{shard_idx}.jsonl")
        summary_path = os.path.join(out_dir, f"summary.shard{shard_idx}.json")
    else:
        per_video_path = os.path.join(out_dir, "per_video.jsonl")
        summary_path = os.path.join(out_dir, "summary.json")

    results: List[Dict[str, Any]] = []
    counted_scores: List[float] = []
    n_skipped_zero = 0
    n_missing = 0
    n_no_faces = 0
    n_failed = 0
    n_total = 0

    # 写文件: 流式 append, 避免长跑时崩溃丢数据
    with open(per_video_path, "w", encoding="utf-8") as fout:
        for rec in spec_records:
            sid = rec["spec_id"]
            cat = rec["category"]
            expected = int(rec["expected_face_count"])

            # 分片: 按 spec_id 哈希取模
            if shard_num >= 2:
                if (stable_hash(sid) % shard_num) != shard_idx:
                    continue

            # 期望人数=0: 跳过 (不参与平均). 与原 batch_eval 保持一致.
            if expected == 0:
                n_skipped_zero += 1
                continue

            n_total += 1
            if limit > 0 and n_total > limit:
                break

            video_path = os.path.join(video_root, cat, f"{sid}.mp4")
            t0 = time.time()
            print(f"[{model_name}][shard{shard_idx}] {n_total} {sid} (expected={expected}) ...",
                  flush=True)
            res = evaluate_one_spec(analyzer, video_path, expected, cfg)
            dt = time.time() - t0

            out_rec = {
                "model": model_name,
                "spec_id": sid,
                "category": cat,
                "shot_count": rec.get("shot_count", ""),
                "video_path": video_path,
                "expected_face_count": expected,
                "expected_min": expected,
                "expected_max": expected,
                "elapsed_sec": round(dt, 3),
                **res,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(out_rec)

            status = res.get("status", "")
            if status == "missing_video":
                n_missing += 1
            elif status == "no_faces_detected":
                n_no_faces += 1
                # no_faces_detected 也算 0 分参与平均 (反映模型生成了人脸缺失)
                counted_scores.append(0.0)
            elif status == "failed":
                n_failed += 1
            elif status in ("ok", "ok_no_faces"):
                counted_scores.append(float(res.get("final_score", 0.0) or 0.0))
            # no_tracklets / no_clusters 也算 0 分
            elif status in ("no_tracklets", "no_clusters"):
                counted_scores.append(0.0)

    summary = {
        "model": model_name,
        "shard_idx": int(shard_idx),
        "shard_num": int(shard_num),
        "video_root": video_root,
        "out_dir": out_dir,
        "n_specs_seen": int(n_total),
        "n_scored": int(len(counted_scores)),
        "n_skipped_expected_zero": int(n_skipped_zero),
        "n_missing_video": int(n_missing),
        "n_no_faces_detected": int(n_no_faces),
        "n_failed": int(n_failed),
        "mean_score": (float(np.mean(counted_scores)) if counted_scores else None),
        "config": dict(cfg),
        "expected_face_source": "t2av-bench-specs/subjects[type=human]",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[{model_name}][shard{shard_idx}] wrote {per_video_path} (n_scored={len(counted_scores)})",
          flush=True)
    print(f"[{model_name}][shard{shard_idx}] wrote {summary_path}", flush=True)
    return results, summary


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    # 多模型 jobs.tsv 模式 (新, 推荐)
    ap.add_argument("--jobs-tsv", type=str, default=None,
                    help="TSV 每行 '<model>\\t<video_root>\\t<out_dir>'.")
    # 兼容: 单模型直接传参 (不写 jobs.tsv 时使用)
    ap.add_argument("--model", type=str, default=None,
                    help="单模型模式: 模型名 (用作 per_video 中的 model 字段)")
    ap.add_argument("--video-root", type=str, default=None,
                    help="单模型模式: 视频根目录 (内含 <category>/<spec_id>.mp4)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="单模型模式: 输出目录")

    ap.add_argument("--specs-dir", type=str,
                    default=os.environ.get(
                        "T2AV_SPECS_DIR",
                        str(_REPO_ROOT / "t2av-bench/specs"),
                    ),
                    help="t2av-bench specs 目录")
    ap.add_argument("--include-anthropomorphic", action="store_true",
                    help="把 subjects 中 type=anthropomorphic 也算进期望人脸数 (默认不算)")
    ap.add_argument("--categories", type=str,
                    default="speech,non_verbal_vocal,action",
                    help="只评测哪些 category (英文逗号分隔). 传 'all' 表示不过滤. "
                         "默认 'speech,non_verbal_vocal,action' (有人的 3 个桶).")

    # 分片
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-num", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="每个模型只处理前 N 个 (调试用), 0=全跑")

    # InsightFace 设备 / 检测器
    ap.add_argument("--ctx-id", type=int, default=0)
    ap.add_argument("--det-size", type=str, default="640,640")

    # pipeline cfg
    ap.add_argument("--sample-rate", type=int, default=1)
    ap.add_argument("--min-face-px", type=int, default=30)
    ap.add_argument("--min-det-score", type=float, default=0.6)
    ap.add_argument("--min-rel-area", type=float, default=0.0005)
    ap.add_argument("--max-gap-frames", type=int, default=3)
    ap.add_argument("--min-iou", type=float, default=0.1)
    ap.add_argument("--min-cos", type=float, default=0.35)
    ap.add_argument("--min-tracklet-len", type=int, default=3)
    ap.add_argument("--dbscan-eps", type=float, default=0.65)
    ap.add_argument("--dbscan-min-samples", type=int, default=1)

    # primary 选择 + 评分
    ap.add_argument("--primary-ratio-thresh", type=float, default=0.10)
    ap.add_argument("--min-primary-cluster-dets", type=int, default=30)

    args = ap.parse_args()

    # 收集 jobs
    if args.jobs_tsv:
        jobs = parse_jobs_tsv(args.jobs_tsv)
        if not jobs:
            raise SystemExit(f"[batch_eval] empty jobs in {args.jobs_tsv}")
    elif args.model and args.video_root and args.out_dir:
        jobs = [(args.model, args.video_root, args.out_dir)]
    else:
        raise SystemExit("[batch_eval] must provide --jobs-tsv OR (--model + --video-root + --out-dir)")

    # specs (可选 category 过滤)
    if args.categories.strip().lower() == "all":
        cat_filter: Optional[List[str]] = None
        cat_label = "<all>"
    else:
        cat_filter = [c.strip() for c in args.categories.split(",") if c.strip()]
        cat_label = ",".join(cat_filter) if cat_filter else "<all>"
    spec_records = load_specs(args.specs_dir,
                              include_anthropomorphic=args.include_anthropomorphic,
                              categories=cat_filter)
    print(f"[batch_eval] loaded {len(spec_records)} specs from {args.specs_dir} "
          f"(category_filter={cat_label})", flush=True)
    if not spec_records:
        raise SystemExit("[batch_eval] no specs loaded (after category filter)")

    # 统计 expected 分布 + 各 category 计数 (只为日志)
    cnt_dist = defaultdict(int)
    cat_dist = defaultdict(int)
    for r in spec_records:
        cnt_dist[int(r["expected_face_count"])] += 1
        cat_dist[r["category"]] += 1
    dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(cnt_dist.items()))
    cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(cat_dist.items()))
    print(f"[batch_eval] expected_face_count distribution: {dist_str}", flush=True)
    print(f"[batch_eval] per-category counts: {cat_str}", flush=True)

    # InsightFace (整个进程内只起一份)
    det_w, det_h = [int(x) for x in args.det_size.split(",")]
    analyzer = MultiFaceConsistencyV2(
        model_name="buffalo_l", ctx_id=args.ctx_id, det_size=(det_w, det_h),
    )

    cfg = {
        "sample_rate": args.sample_rate,
        "min_face_px": args.min_face_px,
        "min_det_score": args.min_det_score,
        "min_rel_area": args.min_rel_area,
        "max_gap_frames": args.max_gap_frames,
        "min_iou": args.min_iou,
        "min_cos": args.min_cos,
        "min_tracklet_len": args.min_tracklet_len,
        "dbscan_eps": args.dbscan_eps,
        "dbscan_min_samples": args.dbscan_min_samples,
        "primary_ratio_thresh": args.primary_ratio_thresh,
        "min_primary_cluster_dets": args.min_primary_cluster_dets,
    }

    print(f"[batch_eval] jobs={len(jobs)}  shard={args.shard_idx}/{args.shard_num}", flush=True)
    for i, (m, src, out) in enumerate(jobs):
        print(f"  job[{i}] {m}  src={src}  out={out}", flush=True)

    for mi, (model_name, video_root, out_dir) in enumerate(jobs):
        if not os.path.isdir(video_root):
            print(f"[batch_eval] {model_name}: video_root not exists ({video_root}), skip",
                  file=sys.stderr)
            continue
        print("\n" + "=" * 64)
        print(f"[batch_eval] >>> START model={model_name}  shard={args.shard_idx}/{args.shard_num}")
        print("=" * 64, flush=True)
        t0 = time.time()
        run_one_model(
            model_idx=mi,
            model_name=model_name,
            video_root=video_root,
            out_dir=out_dir,
            spec_records=spec_records,
            analyzer=analyzer,
            cfg=cfg,
            shard_idx=args.shard_idx,
            shard_num=args.shard_num,
            limit=args.limit,
        )
        print(f"[batch_eval] <<< DONE  model={model_name}  elapsed={time.time() - t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
