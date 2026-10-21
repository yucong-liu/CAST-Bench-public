#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr · eval_official.py
======================

端到端"官方逻辑" OCR 评测脚本 —— 完全绕开本地
    extract_tracks.py + ocr_pipeline/*  (tracking / 过滤 / 段合并 那一整套)
直接照搬 MSAVBench 官方
    MSAVBench_eval_code/eval_pipeline/tool_servers/pp_ocrv5/server.py
的核心逻辑:

    1. PaddleOCR 参数 一字不改:
         ocr_version="PP-OCRv5", lang="ch", use_textline_orientation=True,
         text_rec_score_thresh=0.5,
         use_doc_orientation_classify=False, use_doc_unwarping=False.
    2. _read_frames: cv2 打开 → duration>30s 截前 30s → 均匀采 5 帧
                     ( idx = int((i+0.5) * total/num) ).
    3. _ocr_one_frame: engine.predict(input=[frame])
                       → res["rec_texts"] / res["rec_scores"]
                       → 只用 min_text_score=0.5 过滤 (对齐官方).
    4. 打分复用 score_reference.py 里已经原样搬过来的 4 个官方函数:
         _normalize_text / _fuzzy_ratio /
         _aggregate_across_frames / _score_match_mode.
    5. GT 从 ocr_specs.jsonl 的 text_on_screen[*].text 读.
    6. 视频路径复用 extract_tracks.py 里的 find_video_for_vid + resolve_videos_root.

用法:
    # 单卡
    CUDA_VISIBLE_DEVICES=0 python3 eval_official.py \
        --models-tsv  ../models.tsv \
        --ocr-specs   ocr_specs.jsonl \
        --only-models "LTX-2.3,Ovi,daVinci-MagiHuman,seedance,wan2.7,kling-v3" \
        --out-dir     score_out/official

    # 6 卡 DP: 由外层 shell for i in 0..5 起 6 个进程, 每个 CUDA_VISIBLE_DEVICES=$i
    #         并加 --shard-idx $i --shard-num 6

产物:
    <out-dir>/<model>/<vid>.json          单视频结果 (含 detected_texts / details / meta)
    <out-dir>/<model>/_summary.csv        每模型 per-vid + 均值一行 (仅 shard-num=1 时写)
    <out-dir>/compare_summary.csv         跨模型汇总                (仅 shard-num=1 时写)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# ------- 复用现有代码, 一字不改 -------
# extract_tracks.py: 视频路径解析 & specs 载入
from extract_tracks import (  # noqa: E402
    load_ocr_specs,
    load_models_tsv,
    find_video_for_vid,
)

# score_reference.py: 官方 pp_ocrv5/server.py 的 4 个核心评分函数 (原样搬运)
from score_reference import (  # noqa: E402
    _normalize_text,        # noqa: F401  (间接被 _aggregate_across_frames 用)
    _fuzzy_ratio,           # noqa: F401
    _aggregate_across_frames,
    _score_match_mode,
    _AggregatedText,        # noqa: F401
)


# =====================================================================
# 官方 pp_ocrv5/server.py: _read_frames (原样搬运, 只做微小 API 化封装)
# =====================================================================
def _read_frames(
    video_path: str,
    num_frames: int = 5,
    max_video_duration_sec: float = 30.0,
) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    """打开视频, 均匀采样 num_frames 帧 (BGR).

    对齐官方:
      - fps>0 且 total/fps > max_video_duration_sec  → total = int(fps*max_dur)
      - 采样索引: int((i+0.5) * total/num_frames)
    返回: (frames_bgr, sampled_idxs, meta)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_capped = total
    if fps > 0 and total > 0 and total / fps > max_video_duration_sec:
        total_capped = int(fps * max_video_duration_sec)

    if total_capped <= 0:
        cap.release()
        return [], [], {
            "fps": fps, "total_frames": total, "total_frames_capped": total_capped,
        }

    if num_frames >= total_capped:
        idxs = list(range(total_capped))
    else:
        step = total_capped / num_frames
        idxs = [int((i + 0.5) * step) for i in range(num_frames)]

    frames: List[np.ndarray] = []
    kept_idxs: List[int] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(frame)
        kept_idxs.append(fi)
    cap.release()

    return frames, kept_idxs, {
        "fps": fps,
        "total_frames": total,
        "total_frames_capped": total_capped,
    }


# =====================================================================
# 官方 pp_ocrv5/server.py: _ocr_one_frame (原样搬运)
# =====================================================================
def _ocr_one_frame(
    engine,
    frame_bgr: np.ndarray,
    min_text_score: float,
) -> List[Tuple[str, float]]:
    """单帧 OCR. 返回 [(text, conf), ...], 按 min_text_score 过滤.

    对齐 ocr_pipeline/ocr_engine.py::_parse_ocr_result 的 dict 分支
    (PaddleOCR PP-OCRv5 predict 返回的每个 res 是带 rec_texts/rec_scores 的 dict).
    """
    try:
        results = engine.predict(input=[frame_bgr])
    except Exception:
        traceback.print_exc()
        return []

    pairs: List[Tuple[str, float]] = []
    for res in results or []:
        if isinstance(res, dict) and "rec_texts" in res and "rec_scores" in res:
            texts = res.get("rec_texts", []) or []
            scores = res.get("rec_scores", []) or []
            for t, s in zip(texts, scores):
                try:
                    sc = float(s)
                except Exception:
                    continue
                if sc < min_text_score:
                    continue
                if not isinstance(t, str) or not t.strip():
                    continue
                pairs.append((t, sc))
        elif isinstance(res, list):
            # 老 API 兼容
            for line in res:
                try:
                    t = line[1][0]
                    sc = float(line[1][1])
                except Exception:
                    continue
                if sc < min_text_score:
                    continue
                if not isinstance(t, str) or not t.strip():
                    continue
                pairs.append((t, sc))
    return pairs


# =====================================================================
# 单视频 官方评测 (读帧 → OCR → 聚合 → match 打分)
# =====================================================================
def _score_expected_texts(spec: Dict[str, Any]) -> List[str]:
    tos = spec.get("text_on_screen") or []
    exps: List[str] = []
    for it in tos:
        if not isinstance(it, dict):
            continue
        t = it.get("text")
        if isinstance(t, str) and t.strip():
            exps.append(t.strip())
    return exps


def _evaluate_one_video(
    engine,
    video_path: str,
    expected_texts: List[str],
    num_frames: int,
    min_text_score: float,
    fuzzy_threshold: float,
    max_video_duration_sec: float,
) -> Dict[str, Any]:
    frames, sampled_idxs, vmeta = _read_frames(
        video_path=video_path,
        num_frames=num_frames,
        max_video_duration_sec=max_video_duration_sec,
    )

    per_frame: List[List[Tuple[str, float]]] = []
    for f in frames:
        per_frame.append(_ocr_one_frame(engine, f, min_text_score))

    aggregated = _aggregate_across_frames(per_frame, sampled_idxs)
    n_with_text = sum(1 for p in per_frame if p)

    detected_texts = [
        {
            "text": item.text,
            "confidence": round(item.confidence, 4),
            "frames": item.frames,
        }
        for item in sorted(aggregated.values(), key=lambda x: (-x.confidence, x.text))
    ]
    avg_conf = (
        round(sum(it.confidence for it in aggregated.values()) / len(aggregated), 4)
        if aggregated else 0.0
    )

    result: Dict[str, Any] = {
        "score": None,
        "sub_scores": {},
        "details": [],
        "detected_texts": detected_texts,
        "error": None,
        "meta": {
            "num_frames_sampled": len(sampled_idxs),
            "sampled_frame_ids": sampled_idxs,
            "num_frames_with_text": n_with_text,
            **vmeta,
        },
    }

    if not aggregated:
        if expected_texts:
            result["score"] = 0.0
            result["sub_scores"] = {
                "mode": "match",
                "match_rate": 0.0,
                "total_expected": len(expected_texts),
                "total_matched": 0,
                "avg_text_confidence": 0.0,
                "num_frames_with_text": 0,
                "num_frames_sampled": len(sampled_idxs),
                "num_distinct_texts": 0,
            }
            result["details"] = [
                {"expected": t, "detected": None, "matched": False,
                 "ratio": 0.0, "confidence": None}
                for t in expected_texts
            ]
            result["error"] = "text_required_but_not_detected"
            return result
        result["sub_scores"] = {
            "mode": "skip",
            "avg_text_confidence": 0.0,
            "num_frames_with_text": 0,
            "num_frames_sampled": len(sampled_idxs),
            "num_distinct_texts": 0,
        }
        result["error"] = "no_text_detected"
        return result

    if expected_texts:
        mapped, match_rate, n_matched, details = _score_match_mode(
            aggregated, expected_texts, fuzzy_threshold,
        )
        result["score"] = mapped
        result["sub_scores"] = {
            "mode": "match",
            "match_rate": match_rate,
            "total_expected": len(expected_texts),
            "total_matched": n_matched,
            "avg_text_confidence": avg_conf,
            "num_frames_with_text": n_with_text,
            "num_frames_sampled": len(sampled_idxs),
            "num_distinct_texts": len(aggregated),
        }
        result["details"] = details
        return result

    # detection_only 兜底 (filter_specs 已保证不会走到)
    result["score"] = avg_conf
    result["sub_scores"] = {
        "mode": "detection_only",
        "avg_text_confidence": avg_conf,
        "num_frames_with_text": n_with_text,
        "num_frames_sampled": len(sampled_idxs),
        "num_distinct_texts": len(aggregated),
    }
    return result


# =====================================================================
# 引擎单例 (对齐官方 pp_ocrv5/server.py 的 PaddleOCR 参数)
# =====================================================================
def _build_engine(device: str, text_rec_score_thresh: float):
    """按 官方 pp_ocrv5/server.py 的构造参数 一字不改地建 PaddleOCR."""
    import paddle
    from paddleocr import PaddleOCR

    try:
        paddle.device.set_device(device)
    except Exception:
        device = "cpu"
        paddle.device.set_device(device)

    engine = PaddleOCR(
        ocr_version="PP-OCRv5",
        lang="ch",
        use_textline_orientation=True,
        text_rec_score_thresh=text_rec_score_thresh,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        device=device,
    )
    print(f"[eval_official] PaddleOCR ready: device={device} "
          f"text_rec_score_thresh={text_rec_score_thresh}", flush=True)
    return engine


# =====================================================================
# 主流程
# =====================================================================
def _build_todo(
    model_pairs: List[Tuple[str, str]],
    specs: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str, str, str, List[str]]]:
    """构造 [(model, video_path, vid, category, model_root, expected_texts), ...].

    视频不存在的 (model, vid) 直接跳过.
    """
    todo: List[Tuple[str, str, str, str, str, List[str]]] = []
    n_missing = 0
    for name, root in model_pairs:
        for vid in sorted(specs.keys()):
            spec = specs[vid]
            expected = _score_expected_texts(spec)
            if not expected:
                continue
            cat = spec.get("category") or vid.split("_", 1)[0]
            vpath = find_video_for_vid(root, vid, cat)
            if not vpath:
                n_missing += 1
                continue
            todo.append((name, vpath, vid, cat, root, expected))
    if n_missing:
        print(f"[warn] {n_missing} (model, vid) 视频文件不存在, 跳过", file=sys.stderr)
    return todo


def _write_per_model_and_compare(
    out_dir: Path,
    per_model_rows: Dict[str, List[Dict[str, Any]]],
) -> None:
    """写 <out_dir>/<model>/_summary.csv 和 <out_dir>/compare_summary.csv."""
    from collections import defaultdict

    all_summary: List[Dict[str, Any]] = []
    all_cats_set = set()

    for model, rows in per_model_rows.items():
        model_out = out_dir / model
        model_out.mkdir(parents=True, exist_ok=True)
        csv_path = model_out / "_summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=[
                "video_id", "category", "score", "match_rate",
                "total_expected", "total_matched", "avg_text_confidence",
                "num_distinct_texts", "num_frames_with_text",
                "num_frames_sampled", "mode", "error",
            ])
            w.writeheader()
            for row in rows:
                w.writerow(row)

        scores = [float(r["score"]) for r in rows
                  if isinstance(r.get("score"), (int, float))]
        match_rates = [float(r["match_rate"]) for r in rows
                       if isinstance(r.get("match_rate"), (int, float))]
        cat_scores: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            if isinstance(r.get("score"), (int, float)):
                cat_scores[r["category"]].append(float(r["score"]))
                all_cats_set.add(r["category"])

        all_summary.append({
            "model": model,
            "n_scored": len(rows),
            "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
            "mean_match_rate": round(statistics.mean(match_rates), 4) if match_rates else 0.0,
            "per_category_score": {c: round(statistics.mean(vs), 4)
                                    for c, vs in cat_scores.items()},
        })

    all_cats = sorted(all_cats_set)
    compare_path = out_dir / "compare_summary.csv"
    with compare_path.open("w", encoding="utf-8", newline="") as fo:
        fieldnames = ["model", "n_scored", "mean_score", "mean_match_rate"] \
                     + [f"cat_{c}" for c in all_cats]
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        for s in all_summary:
            row = {
                "model": s["model"],
                "n_scored": s["n_scored"],
                "mean_score": s["mean_score"],
                "mean_match_rate": s["mean_match_rate"],
            }
            for c in all_cats:
                row[f"cat_{c}"] = s["per_category_score"].get(c, "")
            w.writerow(row)
    print(f"[eval_official] wrote compare_summary: {compare_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-tsv", required=True)
    ap.add_argument("--ocr-specs", default=str(THIS_DIR / "ocr_specs.jsonl"))
    ap.add_argument("--only-models", default="")
    ap.add_argument("--out-dir", required=True,
                    help="产物根: <out-dir>/<model>/<vid>.json")

    # 官方 pp_ocrv5 默认值
    ap.add_argument("--num-frames", type=int, default=5,
                    help="官方 default=5")
    ap.add_argument("--min-text-score", type=float, default=0.5,
                    help="官方 default=0.5")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.7,
                    help="官方 default=0.7")
    ap.add_argument("--max-video-duration-sec", type=float, default=30.0,
                    help="官方 default=30.0")

    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-num", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="per-model 限量, 0=不限")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--device", default="gpu:0",
                    help="paddle device, 通常配合外层 CUDA_VISIBLE_DEVICES=$i 用 gpu:0")
    args = ap.parse_args()

    if args.shard_num <= 0:
        ap.error("--shard-num must be >= 1")
    if not (0 <= args.shard_idx < args.shard_num):
        ap.error(f"--shard-idx must be in [0, {args.shard_num})")

    only = [m.strip() for m in args.only_models.split(",") if m.strip()] if args.only_models else None
    model_pairs = load_models_tsv(args.models_tsv, only=only)
    if not model_pairs:
        print(f"[ERROR] no models loaded from {args.models_tsv} (only={only})", file=sys.stderr)
        return 2

    specs = load_ocr_specs(Path(args.ocr_specs))
    if not specs:
        print(f"[ERROR] empty ocr_specs: {args.ocr_specs}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval_official] models: {[n for n,_ in model_pairs]}", flush=True)
    print(f"[eval_official] specs count = {len(specs)}", flush=True)
    print(f"[eval_official] out_dir = {out_dir}", flush=True)
    print(f"[eval_official] num_frames={args.num_frames} "
          f"min_text_score={args.min_text_score} "
          f"fuzzy_threshold={args.fuzzy_threshold} "
          f"max_video_duration_sec={args.max_video_duration_sec}", flush=True)

    todo = _build_todo(model_pairs, specs)
    n_global = len(todo)
    print(f"[eval_official] total (model,vid) buckets (global): {n_global}", flush=True)

    if args.limit and args.limit > 0:
        keep: List[Tuple[str, str, str, str, str, List[str]]] = []
        cnt: Dict[str, int] = {}
        for tup in todo:
            n = tup[0]
            if cnt.get(n, 0) >= args.limit:
                continue
            cnt[n] = cnt.get(n, 0) + 1
            keep.append(tup)
        todo = keep
        print(f"[eval_official] --limit {args.limit} (per-model) -> keep {len(todo)}",
              flush=True)

    sharded = args.shard_num > 1
    if sharded:
        todo = [t for i, t in enumerate(todo) if i % args.shard_num == args.shard_idx]
        print(f"[eval_official] shard {args.shard_idx}/{args.shard_num} -> "
              f"keep {len(todo)} buckets", flush=True)

    if not todo:
        print("[eval_official] nothing to run, exit.")
        return 0

    engine = _build_engine(args.device, args.min_text_score)

    per_model_rows: Dict[str, List[Dict[str, Any]]] = {}
    t_start = time.time()
    n_total = len(todo)
    n_ok = n_err = n_skip = 0

    for bi, (model, vpath, vid, cat, mroot, expected) in enumerate(todo, start=1):
        model_out = out_dir / model
        model_out.mkdir(parents=True, exist_ok=True)
        vid_json = model_out / f"{vid}.json"

        if args.skip_existing and vid_json.is_file():
            try:
                old = json.loads(vid_json.read_text(encoding="utf-8"))
                sc = old.get("score")
                sub = old.get("sub_scores") or {}
                per_model_rows.setdefault(model, []).append({
                    "video_id": vid,
                    "category": cat,
                    "score": sc,
                    "match_rate": sub.get("match_rate"),
                    "total_expected": sub.get("total_expected"),
                    "total_matched": sub.get("total_matched"),
                    "avg_text_confidence": sub.get("avg_text_confidence"),
                    "num_distinct_texts": sub.get("num_distinct_texts"),
                    "num_frames_with_text": sub.get("num_frames_with_text"),
                    "num_frames_sampled": sub.get("num_frames_sampled"),
                    "mode": sub.get("mode"),
                    "error": old.get("error"),
                })
                n_skip += 1
                elapsed = time.time() - t_start
                print(f"[{bi}/{n_total}] {model} {vid} SKIP (exists) | elapsed={elapsed:.0f}s",
                      flush=True)
                continue
            except Exception:
                pass

        t0 = time.time()
        try:
            result = _evaluate_one_video(
                engine=engine,
                video_path=vpath,
                expected_texts=expected,
                num_frames=args.num_frames,
                min_text_score=args.min_text_score,
                fuzzy_threshold=args.fuzzy_threshold,
                max_video_duration_sec=args.max_video_duration_sec,
            )
            n_ok += 1
        except Exception as exc:
            traceback.print_exc()
            result = {
                "score": None, "sub_scores": {}, "details": [], "detected_texts": [],
                "error": f"exception: {exc!r}", "meta": {},
            }
            n_err += 1
        dt = time.time() - t0

        result["video_id"] = vid
        result["category"] = cat
        result["model"] = model
        result["video_path"] = vpath
        result["expected_texts"] = expected
        result["extract_seconds"] = round(dt, 2)

        with vid_json.open("w", encoding="utf-8") as fo:
            json.dump(result, fo, ensure_ascii=False, indent=2)

        sub = result.get("sub_scores") or {}
        per_model_rows.setdefault(model, []).append({
            "video_id": vid,
            "category": cat,
            "score": result.get("score"),
            "match_rate": sub.get("match_rate"),
            "total_expected": sub.get("total_expected"),
            "total_matched": sub.get("total_matched"),
            "avg_text_confidence": sub.get("avg_text_confidence"),
            "num_distinct_texts": sub.get("num_distinct_texts"),
            "num_frames_with_text": sub.get("num_frames_with_text"),
            "num_frames_sampled": sub.get("num_frames_sampled"),
            "mode": sub.get("mode"),
            "error": result.get("error"),
        })

        elapsed = time.time() - t_start
        rate = bi / max(elapsed, 1e-6)
        eta = (n_total - bi) / max(rate, 1e-6)
        n_dist = sub.get("num_distinct_texts")
        mrate = sub.get("match_rate")
        sc = result.get("score")
        print(f"[{bi}/{n_total}] {model} {vid} "
              f"score={sc} match_rate={mrate} distinct={n_dist} dt={dt:.1f}s | "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    # 汇总 (每次都写, 方便中间查看; 多 shard 场景下每 shard 单独写自己的临时 CSV 也无害)
    if sharded:
        # shard 模式: 只写 per-shard 结果, 完整 compare 由外层合并脚本或后续 shard-num=1 补一遍
        shard_dump = out_dir / f"_shard{args.shard_idx}_of_{args.shard_num}_partial.json"
        with shard_dump.open("w", encoding="utf-8") as fo:
            json.dump({
                "shard_idx": args.shard_idx,
                "shard_num": args.shard_num,
                "n_ok": n_ok, "n_err": n_err, "n_skip": n_skip,
                "rows": per_model_rows,
            }, fo, ensure_ascii=False, indent=2)
        print(f"[eval_official] shard{args.shard_idx} partial dump: {shard_dump}",
              flush=True)
    else:
        _write_per_model_and_compare(out_dir, per_model_rows)

    print()
    print(f"[eval_official] DONE in {time.time()-t_start:.0f}s | "
          f"ok={n_ok} err={n_err} skip={n_skip} total={n_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
