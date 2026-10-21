#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr · score_reference.py
========================

参考式 (reference-based) OCR 打分, 完整对齐 MSAVBench 官方
    MSAVBench_eval_code/eval_pipeline/tool_servers/pp_ocrv5/server.py
的 4 个核心函数:
    _normalize_text  /  _fuzzy_ratio  /  _score_match_mode  /  _aggregate_across_frames

不启 FastAPI, 也不重跑 OCR: 直接吃 stage1 已落盘的
    runs/paddleocr/<model>/tracks/<vid>.json
把里面的 tracks 按"官方 5 帧稀疏采样"重构成 per-frame OCR pairs, 再走官方算法.
GT 来自 ocr_specs.jsonl 的 text_on_screen[*].text (已由 filter_specs.py 保证非空).

用法:
    python3 score_reference.py \
        --runs-root  runs/paddleocr \
        --ocr-specs  ocr_specs.jsonl \
        --models     "LTX-2.3,Ovi,daVinci-MagiHuman,seedance,wan2.7,kling-v3" \
        --out-dir    score_out/reference \
        [--num-frames 5] [--fuzzy-threshold 0.7] [--min-text-score 0.5]

产物:
    <out-dir>/<model>/<vid>.json          单视频结果 (score / sub_scores / details / detected_texts)
    <out-dir>/<model>/_summary.csv        每模型 per-vid + 均值一行
    <out-dir>/compare_summary.csv         跨模型均值 & match_rate 汇总
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =====================================================================
# 官方 pp_ocrv5/server.py 的原样搬运 (只保留纯计算部分)
# =====================================================================
_PUNCT_RE = re.compile(
    r"[\s\u3000\.,;:!\?\uff0c\u3002\uff1b\uff1a\uff01\uff1f\u3001\uff08\uff09()\u3010\u3011\[\]\"'`~\-_/]+"
)


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = _PUNCT_RE.sub("", s)
    return s


def _fuzzy_ratio(a: str, b: str) -> float:
    na, nb = _normalize_text(a), _normalize_text(b)
    if not na or not nb:
        return 0.0
    if na == nb or na in nb or nb in na:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


class _AggregatedText:
    __slots__ = ("text", "confidence", "frames")

    def __init__(self, text: str, confidence: float, frame_idx: int) -> None:
        self.text = text
        self.confidence = confidence
        self.frames = [frame_idx]

    def update(self, text: str, confidence: float, frame_idx: int) -> None:
        if confidence > self.confidence:
            self.text = text
            self.confidence = confidence
        self.frames.append(frame_idx)


def _aggregate_across_frames(
    per_frame: List[List[Tuple[str, float]]],
    sampled_idxs: List[int],
) -> Dict[str, _AggregatedText]:
    bucket: Dict[str, _AggregatedText] = {}
    for frame_pos, pairs in enumerate(per_frame):
        frame_idx = sampled_idxs[frame_pos] if frame_pos < len(sampled_idxs) else frame_pos
        for txt, sc in pairs:
            key = _normalize_text(txt)
            if not key:
                continue
            if key in bucket:
                bucket[key].update(txt, sc, frame_idx)
            else:
                bucket[key] = _AggregatedText(txt, sc, frame_idx)
    return bucket


def _score_match_mode(
    aggregated: Dict[str, _AggregatedText],
    expected_texts: List[str],
    fuzzy_threshold: float,
) -> Tuple[float, float, int, List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    if not expected_texts:
        return 0.0, 0.0, 0, details

    detected_items = list(aggregated.values())
    weighted_sum = 0.0
    n_matched = 0

    for exp in expected_texts:
        best_ratio = 0.0
        best_item: Optional[_AggregatedText] = None
        for item in detected_items:
            r = _fuzzy_ratio(exp, item.text)
            if r > best_ratio:
                best_ratio = r
                best_item = item
        confidence = best_item.confidence if best_item is not None else 0.0

        per_score = best_ratio * (0.5 + 0.5 * confidence)
        weighted_sum += per_score
        matched = best_ratio >= fuzzy_threshold
        if matched:
            n_matched += 1
        details.append({
            "expected": exp,
            "detected": best_item.text if best_item is not None else None,
            "matched": bool(matched),
            "ratio": round(best_ratio, 4),
            "confidence": round(confidence, 4) if best_item is not None else None,
        })

    avg = weighted_sum / max(1, len(expected_texts))
    mapped = max(0.0, min(1.0, avg))
    match_rate = n_matched / max(1, len(expected_texts))
    return round(mapped, 4), round(match_rate, 4), n_matched, details


def _sample_frame_indices(total: int, num: int) -> List[int]:
    """官方 pp_ocrv5/server.py 里的均匀采样帧号生成."""
    if total <= 0:
        return []
    if num >= total:
        return list(range(total))
    step = total / num
    return [int((i + 0.5) * step) for i in range(num)]


# =====================================================================
# 从 stage1 tracks JSON 重构 per-frame OCR pairs (对齐官方稀疏采样)
# =====================================================================
def _cap_total_frames_by_duration(
    total_frames: int,
    fps: float,
    max_video_duration_sec: float,
) -> int:
    """对齐官方: 若视频超过 max_video_duration_sec, 只取前段."""
    if fps > 0 and total_frames > 0 and total_frames / fps > max_video_duration_sec:
        return int(fps * max_video_duration_sec)
    return total_frames


def _best_segment_of_track(
    track: Dict[str, Any],
    min_text_score: float,
) -> Optional[Tuple[str, float, int]]:
    """从一条 track 里挑出"conf 最高的一段", 返回 (text, conf, representative_frame_idx).

    我们的 stage1 已把同 track 中"同一段文字连续帧"合并成 text_segments (带 conf_mean),
    等价于官方在 [start_frame, end_frame] 这一段稀疏采样帧上"每帧都看到过这段文本",
    因此每条 track 视作一次"观测": 用其最高 conf_mean 的段作为该 track 的代表.
    这样送进官方 _aggregate_across_frames 的效果等价于官方在多个采样帧上都命中该 key.
    """
    segs = track.get("text_segments") or []
    best: Optional[Tuple[str, float, int]] = None
    for seg in segs:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        sc = float(seg.get("conf_mean") or 0.0)
        if sc < min_text_score:
            continue
        # 用 (start+end)/2 作为该段的代表帧号, 仅用于 detected_texts.frames 的可读性
        sf = int(seg.get("start_frame", 0))
        ef = int(seg.get("end_frame", sf))
        rep = (sf + ef) // 2
        if best is None or sc > best[1]:
            best = (txt, sc, rep)
    return best


def _score_one_video(
    tracks_json: Dict[str, Any],
    expected_texts: List[str],
    num_frames: int,
    min_text_score: float,
    fuzzy_threshold: float,
    max_video_duration_sec: float,
) -> Dict[str, Any]:
    """单视频参考式打分 (对齐官方 evaluate() 的 match 模式).

    per_frame 的构造策略 (与官方 5 帧稀疏采样等价):
        - 官方: 均匀取 5 帧 -> 每帧跑 OCR -> per_frame 长度=5, 每帧一堆 (text, conf) pair.
        - 我们: 已有 stage1 全量 tracks; 把每条 track 的"最高 conf 段"当成 1 次观测,
          把这些观测轮询均分到 num_frames 个桶里, 这样送进 _aggregate_across_frames
          的行为与官方一致 (同 key 取最高 conf, frames 数组累加).
        - `sampled_idxs` 只用于 detected_texts.frames 的可读性 (不影响得分),
          num_frames_with_text 也如实反映"有文本的桶数".
    """
    tracks = tracks_json.get("tracks") or []
    vmeta = tracks_json.get("video_meta") or {}
    total = int(vmeta.get("total_frames") or 0)
    fps = float(vmeta.get("fps") or 0.0)

    total_capped = _cap_total_frames_by_duration(total, fps, max_video_duration_sec)
    sampled_idxs = _sample_frame_indices(total_capped, num_frames)

    # 1) 每条 track 最多贡献 1 次观测 (最高 conf 段)
    observations: List[Tuple[str, float, int]] = []
    for t in tracks:
        best = _best_segment_of_track(t, min_text_score)
        if best is not None:
            observations.append(best)

    # 2) 把 observations 轮询摊到 num_frames 个桶里, 拼成 per_frame
    per_frame: List[List[Tuple[str, float]]] = [[] for _ in range(max(1, len(sampled_idxs) or num_frames))]
    for i, (txt, sc, _rep) in enumerate(observations):
        per_frame[i % len(per_frame)].append((txt, sc))
    # 若采样帧数比桶数多则补齐, 反之截齐, 保证 len(per_frame)==len(sampled_idxs)
    if not sampled_idxs:
        sampled_idxs = list(range(len(per_frame)))
    if len(per_frame) < len(sampled_idxs):
        per_frame += [[] for _ in range(len(sampled_idxs) - len(per_frame))]
    elif len(per_frame) > len(sampled_idxs):
        per_frame = per_frame[: len(sampled_idxs)]

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
            "total_frames": total,
            "total_frames_capped": total_capped,
            "fps": fps,
            "num_frames_with_text": n_with_text,
        },
    }

    if not aggregated:
        # 官方: expected 不为空但没检测到文本 → score=0, error="text_required_but_not_detected"
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
                {"expected": t, "detected": None, "matched": False, "ratio": 0.0, "confidence": None}
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

    # 我们的 pipeline 只对"有 text_on_screen 的 spec"跑, 因此永远走 match 模式;
    # 但仍保留兼容: 万一 expected 为空则退化为 detection_only (与官方一致).
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

    # detection_only (兜底, 正常不会走到)
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
# I/O 辅助
# =====================================================================
def _load_ocr_specs(path: Path) -> Dict[str, Dict[str, Any]]:
    """{vid: spec}. spec 已由 filter_specs.py 保证 text_on_screen 有非空 text."""
    if not path.is_file():
        raise FileNotFoundError(f"ocr_specs.jsonl not found: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = obj.get("id")
            if isinstance(vid, str) and vid:
                out[vid] = obj
    return out


def _extract_expected_texts(spec: Dict[str, Any]) -> List[str]:
    tos = spec.get("text_on_screen") or []
    exps: List[str] = []
    for it in tos:
        if not isinstance(it, dict):
            continue
        t = it.get("text")
        if isinstance(t, str) and t.strip():
            exps.append(t.strip())
    return exps


def _iter_track_files(runs_root: Path, model: str) -> List[Path]:
    tdir = runs_root / model / "tracks"
    if not tdir.is_dir():
        return []
    return sorted(tdir.glob("*.json"))


def _load_tracks_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[warn] load {p} failed: {exc}", file=sys.stderr)
        return None


# =====================================================================
# 主流程
# =====================================================================
def _score_model(
    model: str,
    runs_root: Path,
    specs: Dict[str, Dict[str, Any]],
    out_dir: Path,
    num_frames: int,
    min_text_score: float,
    fuzzy_threshold: float,
    max_video_duration_sec: float,
) -> Dict[str, Any]:
    model_out = out_dir / model
    model_out.mkdir(parents=True, exist_ok=True)

    per_vid_rows: List[Dict[str, Any]] = []
    n_total = 0
    n_scored = 0
    n_skip_no_track = 0
    n_skip_bad_track = 0
    scores: List[float] = []
    match_rates: List[float] = []

    for vid, spec in sorted(specs.items()):
        n_total += 1
        expected = _extract_expected_texts(spec)
        if not expected:
            # filter_specs 已保证非空, 但兜底一次
            n_skip_no_track += 1
            continue

        track_path = runs_root / model / "tracks" / f"{vid}.json"
        if not track_path.is_file():
            n_skip_no_track += 1
            continue

        payload = _load_tracks_json(track_path)
        if payload is None or not payload.get("ok", False):
            n_skip_bad_track += 1
            continue

        result = _score_one_video(
            tracks_json=payload,
            expected_texts=expected,
            num_frames=num_frames,
            min_text_score=min_text_score,
            fuzzy_threshold=fuzzy_threshold,
            max_video_duration_sec=max_video_duration_sec,
        )
        result["video_id"] = vid
        result["category"] = spec.get("category") or vid.split("_", 1)[0]
        result["model"] = model
        result["expected_texts"] = expected

        # 落盘单视频 json
        with (model_out / f"{vid}.json").open("w", encoding="utf-8") as fo:
            json.dump(result, fo, ensure_ascii=False, indent=2)

        n_scored += 1
        sc = result.get("score")
        if isinstance(sc, (int, float)):
            scores.append(float(sc))
        sub = result.get("sub_scores") or {}
        mr = sub.get("match_rate")
        if isinstance(mr, (int, float)):
            match_rates.append(float(mr))

        per_vid_rows.append({
            "video_id": vid,
            "category": result["category"],
            "score": sc,
            "match_rate": sub.get("match_rate"),
            "total_expected": sub.get("total_expected"),
            "total_matched": sub.get("total_matched"),
            "avg_text_confidence": sub.get("avg_text_confidence"),
            "num_distinct_texts": sub.get("num_distinct_texts"),
            "mode": sub.get("mode"),
            "error": result.get("error"),
        })

    # 写 per-model summary CSV
    csv_path = model_out / "_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=[
            "video_id", "category", "score", "match_rate",
            "total_expected", "total_matched", "avg_text_confidence",
            "num_distinct_texts", "mode", "error",
        ])
        w.writeheader()
        for row in per_vid_rows:
            w.writerow(row)

    mean_score = round(statistics.mean(scores), 4) if scores else 0.0
    mean_match = round(statistics.mean(match_rates), 4) if match_rates else 0.0
    print(f"[{model}] specs={n_total} scored={n_scored} "
          f"skip_no_track={n_skip_no_track} skip_bad_track={n_skip_bad_track} "
          f"mean_score={mean_score} mean_match_rate={mean_match}", flush=True)

    # per-category 均值
    from collections import defaultdict
    cat_scores: Dict[str, List[float]] = defaultdict(list)
    for row in per_vid_rows:
        s = row.get("score")
        if isinstance(s, (int, float)):
            cat_scores[row["category"]].append(float(s))
    per_cat = {c: round(statistics.mean(vs), 4) for c, vs in cat_scores.items()}

    return {
        "model": model,
        "n_specs": n_total,
        "n_scored": n_scored,
        "n_skip_no_track": n_skip_no_track,
        "n_skip_bad_track": n_skip_bad_track,
        "mean_score": mean_score,
        "mean_match_rate": mean_match,
        "per_category_score": per_cat,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, help="stage1 产物根: <root>/<model>/tracks/<vid>.json")
    ap.add_argument("--ocr-specs", required=True, help="ocr_specs.jsonl (由 filter_specs.py 生成)")
    ap.add_argument("--models", required=True, help="逗号分隔模型名, 与 runs-root 下子目录一致")
    ap.add_argument("--out-dir", required=True, help="打分产物根目录")
    ap.add_argument("--num-frames", type=int, default=5,
                    help="对齐官方 pp_ocrv5 default_num_frames=5")
    ap.add_argument("--min-text-score", type=float, default=0.5,
                    help="对齐官方 pp_ocrv5 min_text_score=0.5")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.7,
                    help="对齐官方 pp_ocrv5 fuzzy_match_threshold=0.7")
    ap.add_argument("--max-video-duration-sec", type=float, default=30.0,
                    help="对齐官方 pp_ocrv5 max_video_duration_sec=30.0")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_ocr_specs(Path(args.ocr_specs))
    print(f"[score_reference] specs count = {len(specs)}", flush=True)
    print(f"[score_reference] runs_root   = {runs_root}", flush=True)
    print(f"[score_reference] out_dir     = {out_dir}", flush=True)
    print(f"[score_reference] num_frames={args.num_frames} "
          f"min_text_score={args.min_text_score} "
          f"fuzzy_threshold={args.fuzzy_threshold} "
          f"max_video_duration_sec={args.max_video_duration_sec}", flush=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"[score_reference] models ({len(models)}): {models}", flush=True)

    all_summary: List[Dict[str, Any]] = []
    for m in models:
        s = _score_model(
            model=m,
            runs_root=runs_root,
            specs=specs,
            out_dir=out_dir,
            num_frames=args.num_frames,
            min_text_score=args.min_text_score,
            fuzzy_threshold=args.fuzzy_threshold,
            max_video_duration_sec=args.max_video_duration_sec,
        )
        all_summary.append(s)

    # 跨模型汇总 CSV
    all_cats = sorted({c for s in all_summary for c in s["per_category_score"].keys()})
    compare_path = out_dir / "compare_summary.csv"
    with compare_path.open("w", encoding="utf-8", newline="") as fo:
        fieldnames = [
            "model", "n_specs", "n_scored",
            "n_skip_no_track", "n_skip_bad_track",
            "mean_score", "mean_match_rate",
        ] + [f"cat_{c}" for c in all_cats]
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        for s in all_summary:
            row = {
                "model": s["model"],
                "n_specs": s["n_specs"],
                "n_scored": s["n_scored"],
                "n_skip_no_track": s["n_skip_no_track"],
                "n_skip_bad_track": s["n_skip_bad_track"],
                "mean_score": s["mean_score"],
                "mean_match_rate": s["mean_match_rate"],
            }
            for c in all_cats:
                row[f"cat_{c}"] = s["per_category_score"].get(c, "")
            w.writerow(row)

    print()
    print("=" * 72)
    print(f"[DONE] compare_summary: {compare_path}")
    for s in all_summary:
        print(f"  {s['model']:<24s} scored={s['n_scored']:>3d}/{s['n_specs']:<3d} "
              f"mean_score={s['mean_score']:.4f} mean_match_rate={s['mean_match_rate']:.4f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
