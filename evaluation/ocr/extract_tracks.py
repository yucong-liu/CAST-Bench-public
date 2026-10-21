#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr · extract_tracks.py
=======================

Stage 1: 把每个模型的 mp4 视频跑 PaddleOCR + 同帧合并 + 跨帧追踪, 落盘
    runs/paddleocr/<model>/tracks/<vid>.json
带 DP×N 分片 (按 (model, vid) 全局序列 stride=N), 每张卡 1 个进程, 通过
CUDA_VISIBLE_DEVICES=$i 配合 PaddlePaddle 的 OCR_DEVICE='gpu:0' 实现.

输入:
  - --models-tsv  evaluation/models.tsv         模型清单 (name<TAB>video_root)
  - --runs-root   runs/paddleocr/                根产物目录
  - --ocr-specs   ocr/ocr_specs.jsonl            OCR 子集 spec (含 vid)
                  仅这些 vid 会被跑 stage1, 与 id_consistency 范式对齐
  - --only-models LTX-2.3,Ovi,...                白名单
  - --shard-idx / --shard-num  DP 切片
  - --skip-existing            断点续跑

仅生产 stage1 产物, 不调任何 LLM. Stage 2 由 judge_qwen.py 负责.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


_KNOWN_CATEGORIES = (
    "speech", "action", "ambient", "animal", "bgm",
    "instrument", "non_verbal_vocal", "object", "sfx",
)
_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


# ============================================================
# IO 工具
# ============================================================
def resolve_videos_root(root: str) -> str:
    """root 直接含 category 子目录就用它本身; 否则下钻一层 (兼容 daVinci 540p)."""
    if not root or not os.path.isdir(root):
        return root
    for cat in _KNOWN_CATEGORIES:
        if os.path.isdir(os.path.join(root, cat)):
            return root
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        for cat in _KNOWN_CATEGORIES:
            if os.path.isdir(os.path.join(sub, cat)):
                return sub
    return root


def load_ocr_specs(path: Path) -> Dict[str, Dict]:
    """读 ocr_specs.jsonl -> {vid: spec}. 用于筛 vid 白名单 + 提供 category."""
    out: Dict[str, Dict] = {}
    if not path.is_file():
        raise FileNotFoundError(f"ocr_specs.jsonl 不存在: {path}\n请先运行 filter_specs.py")
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
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


def load_models_tsv(path: str, only: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    only_set = set(only) if only else None
    out: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            root = parts[1].strip()
            if not name or not root:
                continue
            if only_set is not None and name not in only_set:
                continue
            if not os.path.isabs(root):
                repo_root = os.environ.get("REPO_ROOT") or str(Path(__file__).resolve().parents[2])
                root = os.path.join(repo_root, root)
            out.append((name, root))
    return out


def find_video_for_vid(model_root: str, vid: str, category: str) -> Optional[str]:
    """根据 vid (如 'speech_0011') 和 category 在 model_root 里定位 mp4.

    搜索顺序:
      1. <real_root>/<category>/<vid>.<ext>       (主路径)
      2. <real_root>/<vid>.<ext>                  (扁平结构兜底)
      3. <real_root>/<category>/<vid>/...         (含子目录的兜底, 罕见)
    """
    real_root = resolve_videos_root(model_root)
    cdir = os.path.join(real_root, category)
    for ext in _VIDEO_EXTS:
        p = os.path.join(cdir, f"{vid}{ext}")
        if os.path.isfile(p):
            return p
    for ext in _VIDEO_EXTS:
        p = os.path.join(real_root, f"{vid}{ext}")
        if os.path.isfile(p):
            return p
    sub = os.path.join(cdir, vid)
    if os.path.isdir(sub):
        for name in sorted(os.listdir(sub)):
            if name.lower().endswith(_VIDEO_EXTS):
                return os.path.join(sub, name)
    return None


# ============================================================
# 单条视频 stage1 推理 + 落盘
# ============================================================
def extract_one_video(
    runner_run_stage1,
    vid: str,
    category: str,
    video_path: str,
    out_dir: str,
    skip_existing: bool,
) -> Dict[str, object]:
    tracks_dir = os.path.join(out_dir, "tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    out_path = os.path.join(tracks_dir, f"{vid}.json")

    if skip_existing and os.path.isfile(out_path):
        try:
            old = json.loads(open(out_path, "r", encoding="utf-8").read())
            if old.get("ok"):
                return {
                    "video_id": vid,
                    "status": "skip",
                    "n_tracks": len((old.get("tracks") or [])),
                    "n_dets": int((old.get("pipeline_meta") or {}).get("counts", {}).get("merged_detections") or 0),
                }
        except Exception:
            pass

    t0 = time.time()
    err = ""
    payload = None
    try:
        payload = runner_run_stage1(video_path)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()
    dt = time.time() - t0

    rec = {
        "video_id": vid,
        "category": category,
        "video_path": video_path,
        "ok": err == "" and payload is not None,
        "error": err,
        "extract_seconds": round(dt, 2),
    }
    if payload is not None:
        rec.update(payload)  # 把 video_meta / pipeline_meta / tracks 平铺进 rec

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    n_tracks = len((payload or {}).get("tracks") or [])
    n_dets = int(((payload or {}).get("pipeline_meta") or {}).get("counts", {}).get("merged_detections") or 0)
    return {
        "video_id": vid,
        "status": "ok" if rec["ok"] else "err",
        "n_tracks": n_tracks,
        "n_dets": n_dets,
        "dt": dt,
    }


# ============================================================
# 主流程
# ============================================================
def build_global_todo(
    model_pairs: List[Tuple[str, str]],
    specs: Dict[str, Dict],
    runs_root: str,
) -> List[Tuple[str, str, Dict[str, str]]]:
    """构造 [(model_name, model_out_dir, item), ...].

    item = {video_id, category, video_path}; 视频不存在的 (model, vid) 直接跳过, 打 warn.
    """
    todo: List[Tuple[str, str, Dict[str, str]]] = []
    n_missing = 0
    sorted_vids = sorted(specs.keys())
    for name, root in model_pairs:
        out_dir = os.path.join(runs_root, name)
        for vid in sorted_vids:
            spec = specs[vid]
            cat = spec.get("category") or vid.split("_", 1)[0]
            vpath = find_video_for_vid(root, vid, cat)
            if not vpath:
                n_missing += 1
                continue
            todo.append((name, out_dir, {
                "video_id": vid,
                "category": cat,
                "video_path": vpath,
            }))
    if n_missing:
        print(f"[warn] {n_missing} (model, vid) 视频文件不存在, 跳过", file=sys.stderr)
    return todo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-tsv", required=True)
    ap.add_argument("--runs-root", required=True,
                    help="stage1 产物根: <runs-root>/<model>/tracks/<vid>.json")
    ap.add_argument("--ocr-specs", default=str(THIS_DIR / "ocr_specs.jsonl"))
    ap.add_argument("--only-models", default="")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-num", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="per-model: 每个模型最多 N 条, 0 = 不限制")
    ap.add_argument("--skip-existing", action="store_true")
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

    print(f"[ocr.extract] models in scope ({len(model_pairs)}):")
    for name, root in model_pairs:
        print(f"   - {name:<24s} <- {root}")
    print(f"[ocr.extract] ocr_specs vid count = {len(specs)}", flush=True)

    todo = build_global_todo(model_pairs, specs, args.runs_root)
    n_global = len(todo)
    print(f"[ocr.extract] total (model,vid) buckets (global): {n_global}", flush=True)

    if args.limit and args.limit > 0:
        keep: List[Tuple[str, str, Dict[str, str]]] = []
        cnt: Dict[str, int] = {}
        for tup in todo:
            n = tup[0]
            if cnt.get(n, 0) >= args.limit:
                continue
            cnt[n] = cnt.get(n, 0) + 1
            keep.append(tup)
        todo = keep
        print(f"[ocr.extract] --limit {args.limit} (per-model) -> keep {len(todo)}", flush=True)

    sharded = (args.shard_num > 1)
    if sharded:
        todo = [t for i, t in enumerate(todo) if i % args.shard_num == args.shard_idx]
        print(f"[ocr.extract] shard {args.shard_idx}/{args.shard_num} -> keep {len(todo)} buckets",
              flush=True)

    if not todo:
        print("[ocr.extract] nothing to run, exit.")
        return 0

    # ---- 加载 PaddleOCR (1 次, 进程级单例) ----
    # 注意: ocr_pipeline 在 import 时按 OCR_DEVICE 加载 PaddleOCR;
    # 上层脚本通过 CUDA_VISIBLE_DEVICES=$i 把卡绑到 cuda:0, 这里依然写 'gpu:0'.
    os.environ.setdefault("OCR_DEVICE", "gpu:0")
    print(f"[ocr.extract] loading PaddleOCR (OCR_DEVICE={os.environ['OCR_DEVICE']}) ...",
          flush=True)
    from ocr_pipeline import run_stage1, OCR_DEVICE, OCR_BATCH_SIZE, STRIDE, MAX_FRAMES  # noqa: WPS433
    print(f"[ocr.extract] OCR ready: device={OCR_DEVICE} batch={OCR_BATCH_SIZE} "
          f"stride={STRIDE} max_frames={MAX_FRAMES}", flush=True)

    counters: Dict[str, Dict[str, int]] = {}
    t_start = time.time()
    n_total = len(todo)

    for bi, (mname, out_dir, item) in enumerate(todo, start=1):
        c = counters.setdefault(mname, {"n": 0, "ok": 0, "err": 0, "skip": 0,
                                         "n_tracks_total": 0, "n_dets_total": 0})
        try:
            result = extract_one_video(
                run_stage1,
                vid=item["video_id"],
                category=item["category"],
                video_path=item["video_path"],
                out_dir=out_dir,
                skip_existing=args.skip_existing,
            )
        except Exception as e:
            result = {"video_id": item["video_id"], "status": "err",
                      "n_tracks": 0, "n_dets": 0, "error": repr(e)}

        c["n"] += 1
        c["n_tracks_total"] += int(result.get("n_tracks") or 0)
        c["n_dets_total"] += int(result.get("n_dets") or 0)
        st = result.get("status")
        if st == "ok":
            c["ok"] += 1
        elif st == "skip":
            c["skip"] += 1
        else:
            c["err"] += 1

        elapsed = time.time() - t_start
        rate = bi / max(elapsed, 1e-6)
        eta = (n_total - bi) / max(rate, 1e-6)
        print(f"[{bi}/{n_total}] {mname} {item['video_id']} "
              f"status={st} tracks={result.get('n_tracks')} "
              f"dets={result.get('n_dets')} dt={result.get('dt', 0.0):.1f}s | "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    # ---- 每模型 summary ----
    sharded_suffix = f".shard{args.shard_idx}" if sharded else ""
    for name, c in counters.items():
        out_dir = os.path.join(args.runs_root, name)
        os.makedirs(out_dir, exist_ok=True)
        sum_path = os.path.join(out_dir, f"summary{sharded_suffix}.json")
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump({
                "tag": name,
                "shard_idx": args.shard_idx,
                "shard_num": args.shard_num,
                "n_videos": c["n"],
                "n_ok": c["ok"],
                "n_err": c["err"],
                "n_skip": c["skip"],
                "n_tracks_total": c["n_tracks_total"],
                "n_dets_total": c["n_dets_total"],
            }, f, ensure_ascii=False, indent=2)

    print()
    print(f"[ocr.extract] DONE in {time.time()-t_start:.0f}s")
    for name, c in counters.items():
        print(f"  [{name}] n={c['n']} ok={c['ok']} err={c['err']} skip={c['skip']} "
              f"tracks={c['n_tracks_total']} dets={c['n_dets_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
