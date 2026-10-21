#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emtion_consistency 评测点 · split-modality 模式的音频抽取工具

功能:
  给定一条 .mp4 视频路径, 用 ffmpeg 抽出对应的单通道 16kHz .wav, 缓存在
  <cache_dir>/<spec_id>.wav; 已存在则直接复用, 不重复抽.

命令等价于:
  ffmpeg -y -i <mp4> -vn -ac 1 -ar 16000 -c:a pcm_s16le <wav>

参数:
  -vn         去掉视频流
  -ac 1       单声道 (Qwen3-Omni 音频输入统一按单声道处理)
  -ar 16000   16 kHz 采样率
  -c:a pcm_s16le  16-bit PCM wav
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


_FFMPEG_BIN: Optional[str] = None


def _resolve_ffmpeg() -> str:
    """定位 ffmpeg 可执行文件. 允许通过环境变量 FFMPEG_BIN 强制指定."""
    global _FFMPEG_BIN
    if _FFMPEG_BIN:
        return _FFMPEG_BIN
    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    if env_bin:
        if not (os.path.isfile(env_bin) and os.access(env_bin, os.X_OK)):
            raise RuntimeError(f"FFMPEG_BIN={env_bin} not executable")
        _FFMPEG_BIN = env_bin
        return _FFMPEG_BIN
    which = shutil.which("ffmpeg")
    if not which:
        raise RuntimeError(
            "ffmpeg not found in PATH; set env FFMPEG_BIN=/path/to/ffmpeg to override."
        )
    _FFMPEG_BIN = which
    return _FFMPEG_BIN


def ensure_wav(mp4_path: str, cache_dir: str, *, sr: int = 16000,
               overwrite: bool = False) -> str:
    """确保 <mp4_path> 对应的 wav 存在于 <cache_dir>, 返回 wav 路径.

    命名: 以 mp4 的 basename (去掉后缀) 作为 wav 名, 避免不同 spec 撞名.

    Args:
        mp4_path: 输入 mp4 绝对路径
        cache_dir: wav 缓存目录, 不存在会自动创建
        sr: 采样率, 默认 16000
        overwrite: 已存在时是否重抽, 默认 False (复用)

    Returns:
        wav 文件绝对路径
    """
    if not mp4_path:
        raise ValueError("empty mp4_path")
    src = Path(mp4_path)
    if not src.is_file():
        raise FileNotFoundError(f"mp4 not found: {mp4_path}")
    cd = Path(cache_dir)
    cd.mkdir(parents=True, exist_ok=True)
    wav_path = cd / f"{src.stem}.wav"

    # 缓存命中: 非空文件直接复用
    if wav_path.is_file() and wav_path.stat().st_size > 0 and not overwrite:
        return str(wav_path)

    ffmpeg = _resolve_ffmpeg()
    # tmp 后缀必须以 .wav 结尾, 否则 ffmpeg 靠后缀推断容器会报
    # "Unable to choose an output format"; 同时再加 -f wav 双保险.
    tmp_path = wav_path.with_suffix(".tmp.wav")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"failed to invoke ffmpeg: {e}") from e

    if proc.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        # 清理坏文件, 打印 ffmpeg stderr
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise RuntimeError(
            f"ffmpeg failed for {mp4_path}: rc={proc.returncode}\n"
            f"stderr:\n{proc.stderr}"
        )

    # 原子替换到最终路径
    os.replace(str(tmp_path), str(wav_path))
    return str(wav_path)


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Extract 16kHz mono wav from mp4.")
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    wav = ensure_wav(args.mp4, args.cache_dir, sr=args.sr, overwrite=args.overwrite)
    print(wav)


if __name__ == "__main__":
    _cli()
