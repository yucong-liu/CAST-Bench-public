#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
style_consistency 评测点 · 单模型视频清单生成脚本

本评测覆盖 t2av-bench 全部 9 个 category (speech / action / object / animal /
sfx / bgm / instrument / ambient / non_verbal_vocal), 一整段视频送进 VLM,
不切镜头.

输入:
  --video-root   某模型 output 根 (含 speech/, action/, ..., non_verbal_vocal/
                 等子目录的整段 mp4). 兼容 daVinci-MagiHuman 这种 output/540p/
                 中间一层的布局.
  --specs-dir    t2av-bench/specs (含 speech.jsonl / action.jsonl / ...)
  --questions    questions.jsonl 路径; 用于 sanity check (可选)
  --out-dir      输出根, 通常为 evaluation/style_consistency/runs/<judge>/<model>/
  --tag          打印用的模型名

行为:
  - 加载所有 category 的 spec (集合 = sum(specs))
  - 在 <video-root>/<category>/<spec_id>.mp4 (或 video-root 下任意一层中间目录
    里的 <category>/<spec_id>.mp4) 查找视频
  - 找得到 => passed=True, 视频整段路径写进 manifest
  - 找不到 => passed=False, 不进入下一步评测

输出:
  <out-dir>/manifest.jsonl   每条 spec 一行, schema:
    {
      "spec_id": "speech_0001",
      "category": "speech",
      "shot_count": "single_shot" | "multi_shot" | "",
      "passed": true,
      "video_path": "/.../speech/speech_0001.mp4",
      "note": "ok" | "video_not_found"
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


# t2av-bench 的全部 category 名 (与 specs 目录下 *.jsonl 文件名一致)
ALL_CATEGORIES = [
    "speech",
    "action",
    "object",
    "animal",
    "sfx",
    "bgm",
    "instrument",
    "ambient",
    "non_verbal_vocal",
]


# ============================================================
# specs 加载
# ============================================================
def load_specs(specs_dir: str) -> Dict[str, dict]:
    """返回 {spec_id: spec_obj}, 遍历 specs_dir/*.jsonl, 顶层文件 (不下钻 _patch)."""
    sd = Path(specs_dir)
    out: Dict[str, dict] = {}
    if not sd.is_dir():
        print(f"[WARN] specs_dir not found: {sd}", file=sys.stderr)
        return out
    for jp in sorted(sd.glob("*.jsonl")):
        if jp.name.startswith("_"):
            continue
        cat = jp.stem
        with jp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] {jp.name}:{line_no} json error: {e}", file=sys.stderr)
                    continue
                sid = obj.get("id")
                if not sid:
                    continue
                obj.setdefault("category", cat)
                out[sid] = obj
    return out


# ============================================================
# questions sanity check
# ============================================================
def load_question_spec_ids(path: Optional[str]) -> Set[str]:
    if not path or not os.path.isfile(path):
        return set()
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get("spec_id")
            if sid:
                out.add(sid)
    return out


# ============================================================
# 视频文件定位: <video_root>/<category>/<sid>.mp4 (或多一层中间目录)
# ============================================================
def _find_videos(video_root: Path) -> Dict[str, Path]:
    """递归扫 <video_root>, 找形如 .../<category>/<sid>.mp4 的文件,
    返回 {sid: path}. 同 sid 命中多次时只保留字典序首个.
    """
    out: Dict[str, Path] = {}
    if not video_root.is_dir():
        return out
    cat_set = set(ALL_CATEGORIES)
    for mp4 in sorted(video_root.rglob("*.mp4")):
        if mp4.parent.name not in cat_set:
            continue
        sid = mp4.stem
        if sid in out:
            continue
        out[sid] = mp4
    return out


# ============================================================
# 主流程
# ============================================================
def run(video_root: str, specs_dir: str, out_dir: str, tag: str,
        questions_path: Optional[str] = None) -> None:
    print("=" * 60)
    print(f"[prepare_manifest] tag={tag}")
    print(f"  video_root = {video_root}")
    print(f"  specs_dir  = {specs_dir}")
    print(f"  questions  = {questions_path}")
    print(f"  out_dir    = {out_dir}")
    print("=" * 60, flush=True)

    os.makedirs(out_dir, exist_ok=True)

    specs = load_specs(specs_dir)
    by_cat: Dict[str, int] = {}
    for s in specs.values():
        c = s.get("category") or "unknown"
        by_cat[c] = by_cat.get(c, 0) + 1
    print(f"[prepare_manifest] loaded {len(specs)} specs ({dict(sorted(by_cat.items()))})")
    if not specs:
        print("[prepare_manifest] no specs found, nothing to do")
        return

    q_sids = load_question_spec_ids(questions_path)
    if q_sids:
        missing = q_sids - set(specs.keys())
        if missing:
            ms = sorted(missing)
            print(f"[WARN] questions reference {len(missing)} spec_id(s) not present in specs: "
                  f"{ms[:5]}{'...' if len(missing)>5 else ''}",
                  file=sys.stderr)

    video_map = _find_videos(Path(video_root))
    print(f"[prepare_manifest] found {len(video_map)} mp4 files under {video_root}")

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    n_pass = 0
    n_fail = 0
    fail_by_cat: Dict[str, int] = {}
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for idx, sid in enumerate(sorted(specs.keys()), start=1):
            spec = specs[sid]
            cat = spec.get("category") or ""
            shot_count = spec.get("shot_count") or ""
            mp4 = video_map.get(sid)
            if mp4 is None:
                rec = {
                    "spec_id": sid,
                    "category": cat,
                    "shot_count": shot_count,
                    "passed": False,
                    "video_path": "",
                    "note": "video_not_found",
                }
                n_fail += 1
                fail_by_cat[cat] = fail_by_cat.get(cat, 0) + 1
            else:
                rec = {
                    "spec_id": sid,
                    "category": cat,
                    "shot_count": shot_count,
                    "passed": True,
                    "video_path": str(mp4),
                    "note": "ok",
                }
                n_pass += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(f"[prepare_manifest] DONE: pass={n_pass}, fail={n_fail}")
    if fail_by_cat:
        print(f"[prepare_manifest] fail by category: {dict(sorted(fail_by_cat.items()))}")
    print(f"[prepare_manifest] manifest: {manifest_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--specs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--questions", default="",
                    help="可选: questions.jsonl, 仅做 sanity check")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    run(
        video_root=args.video_root,
        specs_dir=args.specs_dir,
        out_dir=args.out_dir,
        tag=args.tag or os.path.basename(args.out_dir.rstrip("/")),
        questions_path=args.questions or None,
    )


if __name__ == "__main__":
    main()
