#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emtion_consistency 评测点 · 单模型视频清单生成脚本

本评测只测 speech 类, 且不需要切镜头. 视频整段直接送进模型评估.

输入:
  --video-root   某模型 output 根 (含 speech/, action/, ... 等子目录)
                 仅扫描其中的 speech/<spec_id>.mp4
  --specs-dir    t2av-bench/specs (含 speech.jsonl 等)
                 这里只关心 speech.jsonl 里的 spec
  --questions    questions.jsonl 路径; 用于校验 spec_id 完整性 (可选打印)
  --out-dir      输出根, 通常为 evaluation/emtion_consistency/runs/<model>/
  --tag          打印用的模型名

行为:
  - 加载 specs 中 category == "speech" 的全部 spec
  - 在 <video-root>/speech/<spec_id>.mp4 (或 video-root 下任意一层中间目录里的
    speech/<spec_id>.mp4, 兼容 daVinci 这种 540p/ 中间层) 查找视频
  - 找得到 => passed=True, 视频整段路径写进 manifest
  - 找不到 => passed=False, 不进入下一步评测

输出:
  <out-dir>/manifest.jsonl   每条 spec 一行, schema:
    {
      "spec_id": "speech_0001",
      "category": "speech",
      "passed": true,
      "video_path": "/.../speech/speech_0001.mp4",
      "note": "ok" | "video_not_found"
    }

注意:
  - 与 audio_consistency/pipeline/prepare_clips.py 不同, 这里不引入 TransNetV2,
    不切片, 不生成 clips/ 目录.
  - manifest 里没有 shot_idx / shots[] 字段, 因为情绪是 spec 级标签, 不分镜.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SPEECH_CATEGORY = "speech"


# ============================================================
# specs 加载 (只挑 speech 类)
# ============================================================
def load_speech_specs(specs_dir: str) -> Dict[str, dict]:
    """返回 {spec_id: spec}, 仅 category == 'speech' 的 spec.

    speech.jsonl 不存在或为空时返回 {}.
    """
    sd = Path(specs_dir)
    out: Dict[str, dict] = {}
    sp = sd / "speech.jsonl"
    if not sp.is_file():
        # 兜底: 也允许 speech 在其它文件里 (不太可能, 但保持一致)
        for jp in sorted(sd.glob("*.jsonl")):
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
                    if obj.get("category") != SPEECH_CATEGORY:
                        continue
                    sid = obj.get("id")
                    if sid:
                        out[sid] = obj
        return out

    with sp.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] speech.jsonl:{line_no} json error: {e}", file=sys.stderr)
                continue
            if obj.get("category") != SPEECH_CATEGORY:
                continue
            sid = obj.get("id")
            if sid:
                out[sid] = obj
    return out


# ============================================================
# questions 加载 (仅用于打印 / sanity check, 不影响 manifest 内容)
# ============================================================
def load_question_spec_ids(path: Optional[str]) -> set:
    if not path or not os.path.isfile(path):
        return set()
    out = set()
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
# 视频文件定位: <video_root>/speech/<sid>.mp4 (或多一层中间目录)
# ============================================================
def _find_speech_videos(video_root: Path) -> Dict[str, Path]:
    """递归扫 <video_root>, 找形如 .../speech/<sid>.mp4 的文件,
    返回 {sid: path}. 同 sid 出现多次时只保留第一个 (按字典序).
    """
    out: Dict[str, Path] = {}
    if not video_root.is_dir():
        return out
    for mp4 in sorted(video_root.rglob("*.mp4")):
        if mp4.parent.name != SPEECH_CATEGORY:
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

    specs = load_speech_specs(specs_dir)
    print(f"[prepare_manifest] loaded {len(specs)} speech specs from {specs_dir}")
    if not specs:
        print("[prepare_manifest] no speech specs found, nothing to do")
        return

    q_sids = load_question_spec_ids(questions_path)
    if q_sids:
        missing_in_specs = q_sids - set(specs.keys())
        if missing_in_specs:
            print(f"[WARN] questions reference {len(missing_in_specs)} spec_id(s) "
                  f"not present in speech specs: {sorted(missing_in_specs)[:5]}{'...' if len(missing_in_specs)>5 else ''}",
                  file=sys.stderr)

    video_map = _find_speech_videos(Path(video_root))
    print(f"[prepare_manifest] found {len(video_map)} speech mp4 files under {video_root}")

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    n_pass = 0
    n_fail = 0
    with open(manifest_path, "w", encoding="utf-8") as fout:
        # 按 spec_id 排序, 输出稳定
        for idx, sid in enumerate(sorted(specs.keys()), start=1):
            mp4 = video_map.get(sid)
            if mp4 is None:
                rec = {
                    "spec_id": sid,
                    "category": SPEECH_CATEGORY,
                    "passed": False,
                    "video_path": "",
                    "note": "video_not_found",
                }
                n_fail += 1
                print(f"  [{idx}/{len(specs)}] {sid:14s} -> fail (video_not_found)")
            else:
                rec = {
                    "spec_id": sid,
                    "category": SPEECH_CATEGORY,
                    "passed": True,
                    "video_path": str(mp4),
                    "note": "ok",
                }
                n_pass += 1
                print(f"  [{idx}/{len(specs)}] {sid:14s} -> pass ({mp4})")
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(f"[prepare_manifest] DONE: pass={n_pass}, fail={n_fail}")
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
