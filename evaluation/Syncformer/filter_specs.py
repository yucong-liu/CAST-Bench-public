#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build whitelist / symlink subset for evaluation, based on t2av-bench specs.

Rules (hard-coded so different tasks dont drift):

  task=syncformer  (lip <-> speech sync):
      category == "speech"  AND  audio_situation != "off_screen"

  task=syncnet     (sound event <-> sounding object):
      category NOT IN {speech, ambient, bgm}  AND  audio_situation != "off_screen"

Video naming convention: {spec_id}.mp4  (e.g. speech_0001.mp4),
matching the id field in t2av-bench/specs/*.jsonl one-to-one.

Usage:
  # 1) print whitelist only ( <cat>/<id> per line )
  python filter_specs.py --task syncformer

  # 2) build symlink subset under dst, picking matching videos from src
  python filter_specs.py --task syncformer --src /path/Ovi_test/output --dst /path/runs/Ovi/_filtered
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPECS_DIR = (_REPO_ROOT / "t2av-bench/specs")

ALL_CATEGORIES = [
    "speech", "non_verbal_vocal", "object", "action",
    "ambient", "animal", "instrument", "bgm", "sfx",
]

TASK_RULES = {
    # syncformer: 通用音画同步 (含 speech)，排掉环境类 ambient/bgm
    "syncformer": {"include_categories": set(ALL_CATEGORIES) - {"ambient", "bgm"}},
    # syncnet: 人脸唇动 <-> 语音同步, 只看 speech
    "syncnet":    {"include_categories": {"speech"}},
}


def load_specs(specs_dir: Path):
    items = []
    for cat in ALL_CATEGORIES:
        f = specs_dir / f"{cat}.jsonl"
        if not f.exists():
            print(f"[warn] specs missing: {f}", file=sys.stderr); continue
        with open(f, "r", encoding="utf-8") as fp:
            for ln, line in enumerate(fp, 1):
                line = line.strip()
                if not line: continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[warn] {f}:{ln} JSON parse fail: {e}", file=sys.stderr)
    return items


def filter_for_task(specs, task: str):
    inc = TASK_RULES[task]["include_categories"]
    out = []
    for s in specs:
        if s.get("category", "") not in inc:
            continue
        if s.get("audio_situation") == "off_screen":
            continue
        out.append(s)
    return out


def find_video(src_root: Path, cat: str, spec_id: str):
    name = f"{spec_id}.mp4"
    direct = src_root / cat / name
    if direct.exists():
        return direct
    if src_root.exists():
        for sub in sorted(src_root.iterdir()):
            if sub.is_dir():
                cand = sub / cat / name
                if cand.exists():
                    return cand
    matches = list(src_root.rglob(name))
    if matches:
        prefer = [m for m in matches if f"/{cat}/" in str(m)]
        return prefer[0] if prefer else matches[0]
    return None


def emit_symlinks(specs, src_root: Path, dst_root: Path):
    if dst_root.exists():
        for p in dst_root.rglob("*"):
            if p.is_symlink() or p.is_file():
                try: p.unlink()
                except OSError: pass
    dst_root.mkdir(parents=True, exist_ok=True)

    n_ok, n_miss = 0, 0
    miss_list, by_cat = [], {}
    for s in specs:
        cat, sid = s["category"], s["id"]
        vp = find_video(src_root, cat, sid)
        if vp is None:
            n_miss += 1; miss_list.append(f"{cat}/{sid}"); continue
        link_dir = dst_root / cat
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / f"{sid}.mp4"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(os.path.realpath(vp), link)
        n_ok += 1
        by_cat[cat] = by_cat.get(cat, 0) + 1
    return n_ok, n_miss, miss_list, by_cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_RULES.keys()))
    ap.add_argument("--specs_dir", default=str(DEFAULT_SPECS_DIR))
    ap.add_argument("--src", default="", help="model output root")
    ap.add_argument("--dst", default="", help="symlink subset root")
    ap.add_argument("--list", action="store_true", help="print whitelist only")
    args = ap.parse_args()

    specs = load_specs(Path(args.specs_dir))
    picked = filter_for_task(specs, args.task)

    if args.list or not args.src:
        for s in sorted(picked, key=lambda x: (x["category"], x["id"])):
            print(f"{s['category']}/{s['id']}")
        print(f"# task={args.task} total_specs={len(specs)} picked={len(picked)}",
              file=sys.stderr)
        return

    if not args.dst:
        ap.error("--dst is required when --src is given")
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.exists():
        ap.error(f"--src not found: {src}")

    n_ok, n_miss, miss, by_cat = emit_symlinks(picked, src, dst)
    print(f"[filter_specs] task={args.task} picked={len(picked)} linked={n_ok} missing={n_miss} dst={dst}")
    print(f"  by_category: {dict(sorted(by_cat.items()))}")
    if miss:
        for m in miss[:20]:
            print(f"  [miss] {m}")
        if len(miss) > 20:
            print(f"  ... ({len(miss)-20} more)")


if __name__ == "__main__":
    main()
