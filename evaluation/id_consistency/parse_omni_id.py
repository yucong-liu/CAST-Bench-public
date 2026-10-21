#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
id_consistency · 解析 Qwen3-Omni 输出 JSON

期望输出格式（严格）：
    {"segments":[
       {"speaker":"S1","subject":"画外音","text":"..."},
       {"speaker":"S2","subject":"穿白色衬衫的中年男人","text":"..."}
    ]}

容错策略 (按优先级):
    1. 直接 json.loads
    2. 剥 ```json ... ``` markdown fence
    3. 用正则找最后一个完整 {...} 对象
    4. 找最后一个 '[' 后按花括号深度扫描 {...} 对象列表

返回三元组 (speaker, subject, text) —— subject 缺失时兜底为 ""，不报错。
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

# 任意一个 {...} 对象（不含嵌套）
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
# 任意一个 [{...},{...}] 数组
_JSON_ARR_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,\s*)*\{.*?\}\s*\]", re.DOTALL)


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.lstrip("json").lstrip("JSON").strip()
    return s


def _coerce_segments(arr) -> List[Tuple[str, str, str]]:
    """把 list[dict] 强转成 [(speaker, subject, text), ...]，跳过非法元素。

    subject 缺失 / 非字符串 / 空白 时兜底为 ""，不丢弃该 segment。
    """
    out: List[Tuple[str, str, str]] = []
    if not isinstance(arr, list):
        return out
    for elem in arr:
        if not isinstance(elem, dict):
            continue
        spk = elem.get("speaker")
        txt = elem.get("text")
        sub = elem.get("subject")
        if not isinstance(spk, str) or not spk.strip():
            continue
        if not isinstance(txt, str):
            # 兼容 number/null
            if txt is None:
                continue
            txt = str(txt)
        if not txt.strip():
            continue
        # subject 兜底（向后兼容旧版无 subject 的产出）
        if not isinstance(sub, str):
            sub = ""
        sub = sub.strip()
        out.append((spk.strip(), sub, txt.strip()))
    return out


def parse_omni_segments(raw: str) -> Tuple[List[Tuple[str, str, str]], bool]:
    """
    输入: Omni 模型 batch_decode 出来的原始文本
    输出: ([(speaker, subject, text), ...], parse_ok)
        parse_ok=True  表示成功解析成结构化 segments
        parse_ok=False 表示完全无法解析（segments 仍可能为空）
    """
    if not raw or not raw.strip():
        return [], False

    candidates: List[str] = []
    s = _strip_fence(raw)
    candidates.append(s)

    # 整体若是对象 → 直接试
    obj_match = re.search(r"\{[\s\S]*\}", s)
    if obj_match:
        candidates.append(obj_match.group(0))

    # 也可能模型直接返回了一个数组 [...]
    arr_match = _JSON_ARR_RE.findall(raw)
    if arr_match:
        candidates.append(arr_match[-1])

    # ---- 路径 A: 解析成 {"segments": [...]} 对象 ----
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and "segments" in obj:
            segs = _coerce_segments(obj.get("segments") or [])
            return segs, True
        if isinstance(obj, list):
            # 模型直接吐了数组，没包 segments
            segs = _coerce_segments(obj)
            return segs, True

    # ---- 路径 B: 兜底，按花括号深度扫描 ----
    lb = raw.rfind("[")
    if lb >= 0:
        tail = raw[lb + 1:]
        collected = []
        depth = 0
        start = -1
        in_str = False
        esc = False
        for k, ch in enumerate(tail):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = k
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        try:
                            d = json.loads(tail[start:k + 1])
                            if isinstance(d, dict):
                                collected.append(d)
                        except Exception:
                            pass
                        start = -1
        if collected:
            segs = _coerce_segments(collected)
            if segs:
                return segs, True

    # 全失败
    return [], False
