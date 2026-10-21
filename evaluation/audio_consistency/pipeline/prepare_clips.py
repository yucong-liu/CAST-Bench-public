#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_consistency 评测点 · 单模型 clip 准备脚本

输入:
  --video-root   某模型 output 根 (含 action/, ambient/, ..., speech/ 等子目录)
  --specs-dir    t2av-bench/specs (含 action.jsonl, ..., speech.jsonl)
  --out-dir      输出根, 通常为 evaluation/audio_consistency/runs/<model>/
  --tag          打印用的模型名

行为:
  - 扫 <video-root>/<category>/<spec_id>.mp4
  - 加载 specs (跨所有 *.jsonl), 拿 spec["shot_count"] 与 spec["shots"] 算 expected
  - shot_count == "single_shot": 不走 TransNetV2, 直接登记原 mp4 为唯一 shot
  - shot_count == "multi_shot":  调 pipeline.shotcut.process_one 跑 TransNetV2,
        detected == expected 才 pass 并切片到 <out-dir>/clips/
  - shot_count 缺失或为其它值: 跳过

输出:
  <out-dir>/clips/<spec_id>_shot<idx>.mp4   (multi_shot pass 的样本才会产生)
  <out-dir>/shotcut_manifest.jsonl          每条 spec 一行, schema:
    {
      "spec_id": "...",
      "category": "...",
      "shot_count_kind": "single_shot" | "multi_shot",
      "expected_shots": int,
      "detected_shots": int,                # single_shot 直接记 1
      "passed": bool,
      "fps": float,                         # multi_shot 才有真实 fps; single_shot 记 0.0
      "note": "ok" | "..._failed: ..." | ...,
      "shots": [                            # 只有 passed=True 才有
        {"shot_idx": 1, "clip_path": "..."},
        ...
      ]
    }

用法:
  python prepare_clips.py \\
      --video-root LTX-2.3_test/output \\
      --specs-dir  t2av-bench/specs \\
      --out-dir    evaluation/audio_consistency/runs/LTX-2.3 \\
      --tag        LTX-2.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 让 "import shotcut" 能找到同目录文件
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)

from shotcut import load_model, process_one  # noqa: E402


# ============================================================
# specs 加载
# ============================================================
def _expected_shots(spec: dict) -> Optional[int]:
    kind = spec.get("shot_count")
    if kind == "single_shot":
        return 1
    if kind == "multi_shot":
        shots = spec.get("shots") or []
        return len(shots) if shots else None
    return None


def load_specs(specs_dir: str) -> Dict[str, dict]:
    """
    返回 {spec_id: spec_obj_with_extra_fields}.
    额外字段: __category (从文件名推断), __expected_shots, __shot_count_kind.
    """
    sd = Path(specs_dir)
    out: Dict[str, dict] = {}
    for jp in sorted(sd.glob("*.jsonl")):
        cat = jp.stem  # action / ambient / ...
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
                E = _expected_shots(obj)
                obj["__category"] = cat
                obj["__expected_shots"] = E
                obj["__shot_count_kind"] = obj.get("shot_count", "")
                out[sid] = obj
    return out


# ============================================================
# 视频文件收集
# ============================================================
def collect_videos(video_root: str, specs: Dict[str, dict]) -> List[Tuple[str, str, dict]]:
    """
    返回 [(spec_id, mp4_path, spec_obj), ...].
    扫 <video_root>/<category>/<spec_id>.mp4; 同时兼容 <video_root>/<任意一层>/<category>/<spec_id>.mp4
    (例如 daVinci-MagiHuman 把视频放在 output/540p/<category>/ 下).
    """
    root = Path(video_root)
    if not root.is_dir():
        print(f"[ERROR] video_root not found: {root}", file=sys.stderr)
        return []

    # 找出所有形如 .../<cat>/<sid>.mp4 的文件
    candidates: List[Path] = []
    for mp4 in root.rglob("*.mp4"):
        candidates.append(mp4)

    out: List[Tuple[str, str, dict]] = []
    seen_sids: set = set()
    for mp4 in sorted(candidates):
        sid = mp4.stem
        if sid not in specs:
            continue
        if sid in seen_sids:
            # 同一 spec 出现在多个层级里 (例如 540p/ 和 720p/), 只取第一个
            continue
        spec = specs[sid]
        # 校验父目录与 category 一致 (允许不一致, 只警告)
        if mp4.parent.name != spec["__category"]:
            print(f"[WARN] {sid}: parent dir '{mp4.parent.name}' != category '{spec['__category']}'",
                  file=sys.stderr)
        out.append((sid, str(mp4), spec))
        seen_sids.add(sid)
    return out


# ============================================================
# 主流程
# ============================================================
def run(video_root: str, specs_dir: str, out_dir: str, tag: str,
        threshold: float = 0.5) -> None:
    print("=" * 60)
    print(f"[prepare_clips] tag={tag}")
    print(f"  video_root = {video_root}")
    print(f"  specs_dir  = {specs_dir}")
    print(f"  out_dir    = {out_dir}")
    print("=" * 60, flush=True)

    os.makedirs(out_dir, exist_ok=True)
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    specs = load_specs(specs_dir)
    print(f"[prepare_clips] loaded {len(specs)} specs from {specs_dir}")

    samples = collect_videos(video_root, specs)
    print(f"[prepare_clips] found {len(samples)} mp4 files matching specs")
    if not samples:
        print("[prepare_clips] nothing to do")
        return

    # 是否需要加载 TransNetV2 (只有存在 multi_shot 才需要)
    n_multi = sum(1 for _, _, sp in samples if sp["__shot_count_kind"] == "multi_shot")
    n_single = sum(1 for _, _, sp in samples if sp["__shot_count_kind"] == "single_shot")
    n_skip = len(samples) - n_multi - n_single
    print(f"[prepare_clips] single_shot={n_single}, multi_shot={n_multi}, skipped(no_kind)={n_skip}")

    model = None
    device = "cpu"
    if n_multi > 0:
        print("[prepare_clips] loading TransNetV2 ...")
        model, device = load_model("")  # 走 pip 包内置权重
        print(f"[prepare_clips] TransNetV2 loaded on {device}")

    manifest_path = os.path.join(out_dir, "shotcut_manifest.jsonl")
    n_pass = 0
    n_fail_mismatch = 0
    n_fail_other = 0
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for idx, (sid, mp4, spec) in enumerate(samples, start=1):
            kind = spec["__shot_count_kind"]
            E = spec["__expected_shots"]
            cat = spec["__category"]

            if kind == "single_shot":
                # 直接登记原 mp4
                rec = {
                    "spec_id": sid,
                    "category": cat,
                    "shot_count_kind": kind,
                    "expected_shots": 1,
                    "detected_shots": 1,
                    "passed": True,
                    "fps": 0.0,
                    "note": "single_shot_passthrough",
                    "shots": [{"shot_idx": 1, "clip_path": mp4}],
                }
                n_pass += 1
                print(f"  [{idx}/{len(samples)}] {sid:30s} single_shot  -> pass (passthrough)")
            elif kind == "multi_shot":
                if E is None or E <= 0:
                    rec = {
                        "spec_id": sid, "category": cat,
                        "shot_count_kind": kind, "expected_shots": 0,
                        "detected_shots": 0, "passed": False, "fps": 0.0,
                        "note": "invalid_expected_shots", "shots": [],
                    }
                    n_fail_other += 1
                    print(f"  [{idx}/{len(samples)}] {sid:30s} multi_shot   -> fail (invalid expected)")
                else:
                    res = process_one(
                        mp4_path=mp4, expected_shots=E,
                        out_clip_dir=clips_dir, spec_id=sid,
                        model=model, device=device, threshold=threshold,
                    )
                    rec = {
                        "spec_id": sid, "category": cat,
                        "shot_count_kind": kind,
                        "expected_shots": res.expected_shots,
                        "detected_shots": res.detected_shots,
                        "passed": res.passed,
                        "fps": res.fps,
                        "note": res.note,
                        "shots": [
                            {"shot_idx": i + 1, "clip_path": p}
                            for i, p in enumerate(res.clip_paths)
                        ],
                    }
                    if res.passed:
                        n_pass += 1
                        print(f"  [{idx}/{len(samples)}] {sid:30s} multi_shot   -> pass "
                              f"(E={res.expected_shots}, A={res.detected_shots})")
                    elif res.note == "shot_count_mismatch":
                        n_fail_mismatch += 1
                        print(f"  [{idx}/{len(samples)}] {sid:30s} multi_shot   -> fail "
                              f"(E={res.expected_shots}, A={res.detected_shots})")
                    else:
                        n_fail_other += 1
                        print(f"  [{idx}/{len(samples)}] {sid:30s} multi_shot   -> fail "
                              f"({res.note})")
            else:
                continue  # 不写 manifest

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(f"[prepare_clips] DONE: pass={n_pass}, fail_mismatch={n_fail_mismatch}, fail_other={n_fail_other}")
    print(f"[prepare_clips] manifest: {manifest_path}")
    print(f"[prepare_clips] clips_dir: {clips_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--specs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    run(
        video_root=args.video_root,
        specs_dir=args.specs_dir,
        out_dir=args.out_dir,
        tag=args.tag or os.path.basename(args.out_dir.rstrip("/")),
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
