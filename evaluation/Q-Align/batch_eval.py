#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from q_align import QAlignVideoScorer, load_video


def find_mp4s_in_subfolders(root_dir: Path) -> Dict[Path, List[Path]]:
    """
    Returns: {subfolder_path: [mp4_path, ...], ...}
    Only scans direct subfolders under root_dir; each subfolder is searched
    recursively for mp4 files.
    """
    subfolders = [p for p in root_dir.iterdir() if p.is_dir()]
    mp4_map: Dict[Path, List[Path]] = {}

    for sf in sorted(subfolders):
        mp4s = sorted(sf.rglob("*.mp4"))
        if mp4s:
            mp4_map[sf] = mp4s

    return mp4_map


@torch.no_grad()
def score_one_video(scorer: QAlignVideoScorer, video_path: Path) -> float:
    video_data = load_video(str(video_path))
    scores = scorer([video_data])  # batch size = 1
    return float(scores.tolist()[0])


def safe_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate mp4 videos with Q-Align.")
    parser.add_argument("--root", type=str, required=True, help="Root directory: each subfolder contains mp4 files")
    parser.add_argument("--device", type=str, default="cuda", help='Device, e.g. "cuda" / "cuda:0" / "cpu"')
    parser.add_argument("--model_path", type=str, default="q-future/one-align",
                        help="HF model id or local model dir, e.g. $MODEL_ROOT/q-align/one-align")
    parser.add_argument("--recursive_root", action="store_true",
                        help="If set, recursively find all mp4 files and group by parent directory relative to root")
    parser.add_argument("--save_csv", type=str, default="",
                        help="Optional: save per-video results to CSV, e.g. results.csv")
    parser.add_argument("--save_summary_csv", type=str, default="",
                        help="Optional: save summary results to CSV, e.g. summary.csv")
    parser.add_argument("--specs_dir", type=str, default="",
                        help="t2av-bench/specs 目录, 提供后会按 shot_count_kind 额外输出 "
                             "summary_single.csv / summary_multi.csv 子表 (保留 summary.csv 全样本)")
    parser.add_argument("--fail_fast", action="store_true", help="Exit immediately on error; otherwise skip the failed video")
    args = parser.parse_args()

    root_dir = Path(args.root).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root dir not found: {root_dir}")

    print("Initializing scorer...")
    print(f"  model_path = {args.model_path}")
    print(f"  device     = {args.device}")
    scorer = QAlignVideoScorer(pretrained=args.model_path, device=args.device)

    if args.recursive_root:
        all_mp4s = sorted(root_dir.rglob("*.mp4"))
        folder_to_mp4s: Dict[Path, List[Path]] = {}
        for vp in all_mp4s:
            rel_parent = vp.parent.relative_to(root_dir)
            folder_to_mp4s.setdefault(rel_parent, []).append(vp)
        groups: List[Tuple[Path, List[Path]]] = sorted(folder_to_mp4s.items(), key=lambda x: str(x[0]))
    else:
        mp4_map = find_mp4s_in_subfolders(root_dir)
        groups = [(sf.relative_to(root_dir), mp4s) for sf, mp4s in mp4_map.items()]

    if not groups:
        print(f"No mp4 files found under subfolders of {root_dir}.")
        return

    per_video_rows = []  # for CSV: folder, video_path, score
    folder_scores: Dict[str, List[float]] = {}
    all_scores: List[float] = []

    total_videos = sum(len(vs) for _, vs in groups)
    done = 0

    for rel_folder, mp4s in groups:
        rel_folder_str = str(rel_folder)
        folder_scores.setdefault(rel_folder_str, [])

        print("=" * 60)
        print(f"Subfolder: {rel_folder_str} | videos: {len(mp4s)}")

        for vp in mp4s:
            done += 1
            try:
                score = score_one_video(scorer, vp)
                folder_scores[rel_folder_str].append(score)
                all_scores.append(score)

                per_video_rows.append({
                    "folder": rel_folder_str,
                    "video_path": str(vp),
                    "score": score,
                })

                print(f"[{done}/{total_videos}] {vp.name} -> {score:.6f}")
            except Exception as e:
                msg = f"Scoring failed: {vp} | error: {repr(e)}"
                if args.fail_fast:
                    raise RuntimeError(msg) from e
                else:
                    print(msg)

        f_mean = safe_mean(folder_scores[rel_folder_str])
        if f_mean is None:
            print("Subfolder mean: N/A (no valid results)")
        else:
            print(f"Subfolder mean: {f_mean:.6f}  (n={len(folder_scores[rel_folder_str])})")

    print("\n" + "#" * 60)
    overall_mean = safe_mean(all_scores)
    if overall_mean is None:
        print("Overall mean: N/A (no valid results)")
    else:
        print(f"Overall mean: {overall_mean:.6f}  (n={len(all_scores)})")

    if args.save_csv:
        out_csv = Path(args.save_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["folder", "video_path", "score"])
            writer.writeheader()
            writer.writerows(per_video_rows)
        print(f"Saved per-video results: {out_csv}")

    if args.save_summary_csv:
        out_sum = Path(args.save_summary_csv).expanduser().resolve()
        out_sum.parent.mkdir(parents=True, exist_ok=True)

        # ---- 加载 spec_kind_map (可选) ----
        spec_kind_map = {}
        if args.specs_dir:
            import sys as _sys
            _here = Path(__file__).resolve().parent
            _common = _here.parent / "_common"
            if str(_common) not in _sys.path:
                _sys.path.insert(0, str(_common))
            try:
                from spec_kind import load_spec_kind_map  # noqa: E402
                spec_kind_map = load_spec_kind_map(args.specs_dir, verbose=True)
            except Exception as e:
                print(f"[WARN] load_spec_kind_map failed: {e}; will skip single/multi split")
                spec_kind_map = {}

        def _stem_of(p: str) -> str:
            return os.path.splitext(os.path.basename(p))[0]

        def _write_summary_csv(rows, predicate, path: Path):
            f_scores: Dict[str, List[float]] = {}
            for r in rows:
                if not predicate(r):
                    continue
                folder = r["folder"]
                f_scores.setdefault(folder, []).append(float(r["score"]))
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["folder", "n", "mean_score"])
                writer.writeheader()
                all_s: List[float] = []
                for folder in sorted(f_scores.keys()):
                    sl = f_scores[folder]
                    writer.writerow({
                        "folder": folder,
                        "n": len(sl),
                        "mean_score": (safe_mean(sl) if sl else ""),
                    })
                    all_s.extend(sl)
                writer.writerow({
                    "folder": "__ALL__",
                    "n": len(all_s),
                    "mean_score": (safe_mean(all_s) if all_s else ""),
                })
            print(f"Saved summary results: {path}")

        # (1) 全样本 (向后兼容老格式)
        _write_summary_csv(per_video_rows, lambda r: True, out_sum)

        # (2) single / multi 子表 (有 spec_kind_map 才输出)
        if spec_kind_map:
            sgl = out_sum.with_name(out_sum.stem + "_single" + out_sum.suffix)
            mlt = out_sum.with_name(out_sum.stem + "_multi"  + out_sum.suffix)
            _write_summary_csv(
                per_video_rows,
                lambda r: spec_kind_map.get(_stem_of(r["video_path"]), "") == "single_shot",
                sgl,
            )
            _write_summary_csv(
                per_video_rows,
                lambda r: spec_kind_map.get(_stem_of(r["video_path"]), "") == "multi_shot",
                mlt,
            )

if __name__ == "__main__":
    main()
