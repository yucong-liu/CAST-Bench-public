#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emtion_consistency 评测点 · 题面 -> 模态 分类器

split-modality 模式使用: 同一个 spec 下的 N 题按题干拆成两组:
  * audio-only: 只送音频 (ffmpeg 抽出的 wav) 给 Omni
  * video-only: 只送视频 (use_audio_in_video=False) 给 Omni

分类规则 (只针对 emtion_consistency/questions.jsonl 实际出现的三类题面):

  1. audio-only (Type A, 语音情绪):
     - "Is the speaking voice in the video expressing a <emotion> emotion?"
     - "Does the speaking voice in the video start with a <emotion> emotion?"
     - "Does the speaking voice in the video transition from ..."
     关键词: 'speaking voice'

  2. video-only (Type B, 视觉情绪 / 是否有人):
     - "Do the on-screen characters visually display a <emotion> emotion?"
     - "Does the on-screen character's facial expression shift ..."
     - "Are there no people visible in the video?"
     关键词: 'on-screen character', 'facial expression', 'people visible'

其余情况: 默认判为 audio (保守 fallback, 应该不会命中).
"""
from __future__ import annotations

import re


_RE_AUDIO = re.compile(r"speaking\s+voice", re.IGNORECASE)
_RE_VIDEO = re.compile(
    r"on-screen\s+character"          # visual display 题
    r"|facial\s+expression"           # facial expression shift 题
    r"|people\s+visible",             # Are there no people visible 题
    re.IGNORECASE,
)


def classify(question: str) -> str:
    """把一条题干分类为 'audio' 或 'video'.

    Args:
        question: 题面字符串, 来自 questions.jsonl 的 question 字段.
    Returns:
        'audio' | 'video'
    """
    q = question or ""
    if _RE_VIDEO.search(q):
        return "video"
    if _RE_AUDIO.search(q):
        return "audio"
    # 兜底: 未匹配上的默认 audio (真实数据里应该不会走到)
    return "audio"


def classify_batch(questions):
    """批量分类, 返回和输入等长的 list."""
    return [classify(q) for q in questions]


if __name__ == "__main__":
    # 自测: 用真实 questions.jsonl 的样例校验
    samples = [
        ("Is the speaking voice in the video expressing a joyful emotion?", "audio"),
        ("Do the on-screen characters visually display a joyful emotion?", "video"),
        ("Does the speaking voice in the video start with a neutral emotion?", "audio"),
        ("Does the speaking voice in the video transition from a neutral emotion to a confused emotion?", "audio"),
        ("Does the on-screen character's facial expression shift in a way that matches the vocal emotional transition?", "video"),
        ("Are there no people visible in the video?", "video"),
    ]
    ok = 0
    for q, expect in samples:
        got = classify(q)
        flag = "OK " if got == expect else "FAIL"
        if got == expect:
            ok += 1
        print(f"[{flag}] expect={expect:5s} got={got:5s}  {q}")
    print(f"\n{ok}/{len(samples)} passed")
