#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransNetV2 切镜头工具 (audio_consistency 评测点专用)。

直接复用 evaluation/TransNetV2/batch_eval.py 中已经验证过的推理函数:
    extract_frames_27x48 / predict_frames / predictions_to_scenes / load_model
本模块在其上补充:
    1) ffprobe 取真实 fps + 总帧数, 把 scene 帧号转成秒
    2) ffmpeg 重编码切片 (libx264 + aac), 输出 <spec_id>_shot<idx>.mp4
    3) process_one(): 给定一段 mp4 + 期望镜头数, 跑切分并写出 clip 文件,
       返回 (detected_shots, passed, clip_paths)

用法 (供 prepare_clips.py 调用):
    from shotcut import load_model, process_one
    model, device = load_model(weights_path="")          # 留空走 pip 包内置权重
    info = process_one(mp4_path, expected_shots=K,
                       out_clip_dir="...", spec_id="...", model=model, device=device,
                       threshold=0.5)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
_REPO_ROOT = Path(__file__).resolve().parents[3]

# ============================================================
# 复用 TransNetV2/batch_eval.py 的核心推理函数
# ============================================================
_TNV2_DIR = str(_REPO_ROOT / "evaluation/TransNetV2")
if _TNV2_DIR not in sys.path:
    sys.path.insert(0, _TNV2_DIR)

# 只 import 我们需要的 4 个函数, 避免 import 触发 TransNetV2/batch_eval.py 的 main()
from batch_eval import (  # noqa: E402  (path injection above)
    extract_frames_27x48,
    predict_frames,
    predictions_to_scenes,
    load_model as _load_tnv2_model,
)


# ============================================================
# 模型加载
# ============================================================
def load_model(weights_path: str = "") -> Tuple[object, str]:
    """加载 TransNetV2 (PyTorch 移植版). 返回 (model, device)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_tnv2_model(weights_path, device)
    return model, device


# ============================================================
# ffprobe: 拿真实 fps / 总帧数 / 时长
# ============================================================
def _ffprobe_video_info(mp4_path: str) -> Tuple[float, int, float]:
    """返回 (fps, nb_frames, duration_seconds). 解析失败抛 RuntimeError."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        mp4_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {mp4_path}: {proc.stderr.strip()}")
    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe: no video stream in {mp4_path}")
    s = streams[0]

    # avg_frame_rate 形如 "30000/1001"
    afr = s.get("avg_frame_rate", "0/0")
    try:
        num, den = afr.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 0.0
    except Exception:
        fps = 0.0
    if fps <= 0:
        # 兜底: 用 r_frame_rate 重新探一次
        cmd2 = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1",
            mp4_path,
        ]
        p2 = subprocess.run(cmd2, capture_output=True, text=True)
        rfr = (p2.stdout or "").strip()
        try:
            num, den = rfr.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 0.0
        except Exception:
            fps = 0.0
    if fps <= 0:
        raise RuntimeError(f"ffprobe: cannot determine fps for {mp4_path}")

    nb_frames = 0
    try:
        nb_frames = int(s.get("nb_frames") or 0)
    except Exception:
        nb_frames = 0

    duration = 0.0
    try:
        duration = float(s.get("duration") or 0.0)
    except Exception:
        duration = 0.0
    if duration <= 0:
        try:
            duration = float((info.get("format") or {}).get("duration") or 0.0)
        except Exception:
            duration = 0.0

    return fps, nb_frames, duration


# ============================================================
# TransNetV2 检测 -> scenes
# ============================================================
@torch.no_grad()
def detect_scenes(mp4_path: str, model, device: str, threshold: float = 0.5) -> np.ndarray:
    """对一段 mp4 跑 TransNetV2, 返回 scenes (np.int32, shape=[N, 2], 闭区间 [start_frame, end_frame])."""
    frames = extract_frames_27x48(mp4_path)
    if len(frames) < 2:
        raise RuntimeError(f"too few frames decoded: {len(frames)} ({mp4_path})")
    preds = predict_frames(model, frames, device)
    scenes = predictions_to_scenes(preds, threshold=threshold)
    return scenes


# ============================================================
# ffmpeg: 按帧号切片 (重编码, 求稳)
# ============================================================
def _cut_clip(src_mp4: str, dst_mp4: str, t_start: float, t_end: float) -> None:
    """重编码切片 [t_start, t_end). 失败抛 RuntimeError."""
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    if os.path.exists(dst_mp4):
        os.remove(dst_mp4)
    duration = max(0.0, t_end - t_start)
    if duration <= 0:
        raise RuntimeError(f"invalid clip range [{t_start}, {t_end})")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t_start:.3f}",
        "-i", src_mp4,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        dst_mp4,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.isfile(dst_mp4):
        raise RuntimeError(
            f"ffmpeg cut failed: {src_mp4} -> {dst_mp4}\n"
            f"  range=[{t_start:.3f},{t_end:.3f}) duration={duration:.3f}s\n"
            f"  stderr={proc.stderr.strip()[:300]}"
        )


# ============================================================
# 主接口: 处理单个 spec
# ============================================================
@dataclass
class ShotcutResult:
    spec_id: str
    expected_shots: int
    detected_shots: int
    passed: bool
    clip_paths: List[str]
    fps: float
    note: str = ""


def process_one(
    mp4_path: str,
    expected_shots: int,
    out_clip_dir: str,
    spec_id: str,
    model,
    device: str,
    threshold: float = 0.5,
) -> ShotcutResult:
    """
    multi_shot 专用: 对 mp4_path 跑 TransNetV2, 仅当 detected==expected 才切片.
    若 detected != expected, 不切, 返回 passed=False.

    single_shot 不要进这个函数, 由上层 prepare_clips 直接登记原 mp4 即可.
    """
    if expected_shots <= 0:
        return ShotcutResult(spec_id, expected_shots, 0, False, [], 0.0,
                             note="invalid expected_shots")

    try:
        fps, _nb, _dur = _ffprobe_video_info(mp4_path)
    except Exception as e:
        return ShotcutResult(spec_id, expected_shots, 0, False, [], 0.0,
                             note=f"ffprobe_failed: {e}")

    try:
        scenes = detect_scenes(mp4_path, model, device, threshold=threshold)
    except Exception as e:
        return ShotcutResult(spec_id, expected_shots, 0, False, [], fps,
                             note=f"transnetv2_failed: {e}")

    detected = int(len(scenes))
    if detected != expected_shots:
        return ShotcutResult(spec_id, expected_shots, detected, False, [], fps,
                             note="shot_count_mismatch")

    # detected == expected -> 切片
    clip_paths: List[str] = []
    for i, (sf, ef) in enumerate(scenes.tolist(), start=1):
        # scenes 是闭区间 [start_frame, end_frame], 终止时间 = (end_frame + 1) / fps
        t_start = max(0.0, sf / fps)
        t_end = max(t_start + 1.0 / max(fps, 1e-6), (ef + 1) / fps)
        dst = os.path.join(out_clip_dir, f"{spec_id}_shot{i}.mp4")
        try:
            _cut_clip(mp4_path, dst, t_start, t_end)
        except Exception as e:
            return ShotcutResult(spec_id, expected_shots, detected, False, [], fps,
                                 note=f"ffmpeg_failed_shot{i}: {e}")
        clip_paths.append(dst)

    return ShotcutResult(spec_id, expected_shots, detected, True, clip_paths, fps,
                         note="ok")


# ============================================================
# 简单自检入口
# ============================================================
def _selftest():
    """python shotcut.py <mp4> <expected_shots> <out_dir>"""
    if len(sys.argv) < 4:
        print("usage: python shotcut.py <mp4> <expected_shots> <out_dir>", file=sys.stderr)
        sys.exit(2)
    mp4 = sys.argv[1]
    K = int(sys.argv[2])
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    sid = Path(mp4).stem

    model, device = load_model("")
    res = process_one(mp4, K, out_dir, sid, model, device)
    print(json.dumps(res.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _selftest()
