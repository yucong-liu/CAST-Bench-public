#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 t2av-bench/prompts/zh/*.json 里的中文提示词，
调用本地 vLLM (Qwen3-VL-235B-A22B-Instruct) 改写成 hunyuan 训练/推理格式的
CSV，与 prompts/hunyuan/prompt_trans/exp/t2va_{single,multi}_captions_300_5s_15s.csv
列结构完全对齐，仅在最前面多一列 id。

输出列（严格顺序）：
    id, index, prompt, prompt_wo_shots,
    gt_video_cos_url, gt_audio_cos_url,
    num_frames, scene_shot_detail, seed

- gt_video_cos_url / gt_audio_cos_url 留空
- num_frames 固定 241  (10s * 24fps + 1)
- scene_shot_detail:
    single_shot -> "[[0, 10]]"
    multi_shot  -> 随机切成 N 段，总时长 10s，如 "[[0, 2.3], [2.3, 5.6], ...]"
- seed 随机整数 1..9999
- 文件编码统一 utf-8-sig（Excel 中文不乱码）

用法：
    # 前置：本地 vLLM 已经起在 127.0.0.1:8080（见 t2av-bench/SETUP.md）
    export NO_PROXY=localhost,127.0.0.1,::1
    export no_proxy=$NO_PROXY

    python rewrite_zh_to_hunyuan.py                        # 全跑
    python rewrite_zh_to_hunyuan.py --categories animal    # 只跑 animal
    python rewrite_zh_to_hunyuan.py --shot-count single    # 只跑单镜头
    python rewrite_zh_to_hunyuan.py --limit 3              # 每类只跑前 3 条（调试）
    python rewrite_zh_to_hunyuan.py --concurrency 16       # 并发 16

输出路径：
    prompts/hunyuan/<category>/<category>_single_10s.csv
    prompts/hunyuan/<category>/<category>_multi_10s.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# 依赖：openai>=1.30（SETUP.md 里已经装好）
try:
    from openai import OpenAI
except ImportError as e:  # pragma: no cover
    print("[FATAL] 缺少 openai 包，请先 `pip install 'openai>=1.30'`", file=sys.stderr)
    raise

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
ZH_DIR = ROOT_DIR / "prompts" / "zh"
HUNYUAN_DIR = ROOT_DIR / "prompts" / "hunyuan"
EXAMPLE_SINGLE_CSV = HUNYUAN_DIR / "prompt_trans" / "exp" / "t2va_single_captions_300_5s_15s.csv"
EXAMPLE_MULTI_CSV = HUNYUAN_DIR / "prompt_trans" / "exp" / "t2va_multi_captions_300_5s_15s.csv"

CATEGORIES = [
    "action", "ambient", "animal", "bgm", "instrument",
    "non_verbal_vocal", "object", "sfx", "speech",
]

TOTAL_DURATION = 10.0   # 秒
FPS = 24
NUM_FRAMES = FPS * int(TOTAL_DURATION) + 1   # 241

# CSV 表头
CSV_HEADER = [
    "id", "index", "prompt", "prompt_wo_shots",
    "gt_video_cos_url", "gt_audio_cos_url",
    "num_frames", "scene_shot_detail", "seed",
]

# 默认 vLLM 配置（见 t2av-bench/SETUP.md §5.1，可用环境变量覆盖）
DEFAULT_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8080/v1")
DEFAULT_API_KEY  = os.environ.get("QWEN_API_KEY",  "EMPTY")
DEFAULT_MODEL    = os.environ.get("QWEN_MODEL",    "Qwen3-VL-235B-A22B-Instruct")

# ----------------------------------------------------------------------
# System Prompt（关键：让 VL 学会 hunyuan 的中文自然语言风格 + 内嵌音频标签）
# ----------------------------------------------------------------------
SYSTEM_PROMPT_SINGLE = """你是一名专业的文生视频/文生音视频提示词改写专家。

【任务】
把用户提供的一段"结构化中文视频提示词"（含 [OVERALL] / [SHOT k/N] / 景别 / 视角等段落式标签）改写成
"hunyuan 单镜头影视化中文提示词"格式，然后一次性返回一段 JSON。

【hunyuan 单镜头 prompt 风格约定 —— 严格遵守】
1. 总体是**一整段流畅的中文自然语言**，不再出现 [OVERALL] / [SHOT] / --- / *** 之类的分段标记，也不再用编号列表。
2. 全篇必须自然内嵌 **音频类 XML 标签**，用来标注"能听到的具体内容"：
   - `<music>...</music>`  背景音乐/器乐
   - `<sfx>...</sfx>`      环境音、拟音、动作音
   - `<speech>...</speech>` 人物对白/独白（保留原语种原文）
   - `<lyrics>...</lyrics>` 歌词
   - `<text>...</text>`     画面里出现的文字（水印、字幕、logo 等）
   凡是源 prompt 里提及了具体的对白、歌词、环境音、字幕/水印文字，都要用对应标签精确包裹，且**尽量把这些标签自然融入描写句子**，而不是罗列。
   ⚠️ **标签名必须严格使用上述英文小写形式**（`<music>` / `<sfx>` / `<speech>` / `<lyrics>` / `<text>`），**绝对不允许**使用中文标签（如 `<音乐>`、`<音效>`、`<对白>`、`<歌词>`、`<文字>`、`<字幕>`）或大写变体（如 `<Music>`、`<SFX>`）。
3. 描述要包含：景别 + 视角 + 场景/背景 + 主体外貌服饰 + 主体动作序列 + 音频（音乐/音效/人声）。写作口吻参考电视剧、纪录片、短视频写实解说，措辞客观、时序清晰。
4. 长度：**500~900 个汉字**为宜；细节丰富但不啰嗦；标点用中文全角逗号/句号。
5. 单镜头总时长 10 秒左右；如需在描述里点出时长，可用"全程约 10 秒"或不点出。
6. 同时产出一个 `prompt_wo_shots`：从改写好的 prompt 中**只保留概括性首句/几句 + 全部音频标签**（把镜头细节、演员外貌、动作步骤、画面文字等删掉），长度 60~180 字。相当于 prompt 的"简介版"。

【返回格式 —— 必须是且只是一段可 json.loads 的 JSON，不要 markdown 代码块、不要多余文字】
{
  "prompt": "…改写后的完整单镜头 hunyuan 中文提示词…",
  "prompt_wo_shots": "…简介版…"
}
"""

SYSTEM_PROMPT_MULTI = """你是一名专业的文生视频/文生音视频提示词改写专家。

【任务】
把用户提供的一段"结构化中文多镜头视频提示词"（含 [OVERALL] / [SHOT k/N] 段落）改写成
"hunyuan 多镜头影视化中文提示词"格式，并按用户在 user prompt 里**指定好的时间戳切分**分段描写，
然后一次性返回一段 JSON。

【hunyuan 多镜头 prompt 风格约定 —— 严格遵守】
1. 全篇是**一整段流畅的中文自然语言**，没有 [OVERALL]/[SHOT]/---/*** 等段落标记，没有编号列表。
2. 开头写 1~2 句"整体概述句"（谁在哪儿做什么 + 全程持续的音乐/氛围），接着按用户给的时间戳依次分段。
3. 每一段镜头以**时间戳短语**开头，用户会给你一个 shots 列表（N 段，总时长严格 10 秒），你必须按顺序为每段写一小段描写。时间戳短语请**在下列 5 种风格里随机挑选、变换使用**（不要每段都用同一种）：
   - `[0秒-1.1秒],`
   - `(0s~2.2s),`
   - `[时段:2.4秒-6.2秒],`
   - `[时间:6.6s-8.4s],`
   - `[从7.2秒到10秒],`
   括号、"秒/s"、"~/-/至/到" 都可以混用；小数点后 1 位即可；起止时间必须严格等于用户 shots 列表里的对应端点。
4. 每段镜头描写要覆盖：**景别（近景/中景/全景/特写/俯拍/仰拍…） + 视角 + 场景 + 主体外貌/服饰 + 动作 + 台词/音效/字幕/水印**（如涉及）。
5. 全篇必须自然内嵌音频类 XML 标签：
   - `<music>...</music>`   背景音乐/器乐（一般写在整体概述里，如"全程持续<music>…</music>"）
   - `<sfx>...</sfx>`       环境音、拟音、动作音
   - `<speech>...</speech>` 人物对白/独白（保留原语种原文）
   - `<lyrics>...</lyrics>` 歌词
   - `<text>...</text>`     画面里的文字（水印/字幕/logo）
   ⚠️ **标签名必须严格使用上述英文小写形式**（`<music>` / `<sfx>` / `<speech>` / `<lyrics>` / `<text>`），**绝对不允许**使用中文标签（如 `<音乐>`、`<音效>`、`<对白>`、`<歌词>`、`<文字>`、`<字幕>`）或大写变体（如 `<Music>`、`<SFX>`）。
6. 长度：**700~1400 个汉字**；措辞客观、时序清晰、有电视剧/短视频写实感。
7. 同时产出一个 `prompt_wo_shots`：从改写好的 prompt 中**只保留概述句 + 全部音频标签**（把每一段的时间戳、镜头细节、演员外貌、字幕内容都删掉），长度 60~200 字。

【返回格式 —— 必须是且只是一段可 json.loads 的 JSON，不要 markdown 代码块、不要多余文字】
{
  "prompt": "…改写后的完整多镜头 hunyuan 中文提示词，段数与用户指定的 shots 数量一致…",
  "prompt_wo_shots": "…简介版…"
}
"""


# ----------------------------------------------------------------------
# few-shot 示例（从真实 hunyuan CSV 里抽 1 条 single + 1 条 multi）
# ----------------------------------------------------------------------
def _load_first_row(csv_path: Path) -> Optional[Dict[str, str]]:
    """从 CSV 读第 1 条样例。"""
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None


_EXAMPLE_SINGLE_ROW = _load_first_row(EXAMPLE_SINGLE_CSV)
_EXAMPLE_MULTI_ROW = _load_first_row(EXAMPLE_MULTI_CSV)


def _example_block(shot_kind: str) -> str:
    """把真实 CSV 里的一条样例转成给 VL 看的示例文本。"""
    if shot_kind == "single":
        row = _EXAMPLE_SINGLE_ROW
    else:
        row = _EXAMPLE_MULTI_ROW
    if not row:
        return ""
    demo = {
        "prompt": row.get("prompt", "").strip(),
        "prompt_wo_shots": row.get("prompt_wo_shots", "").strip(),
    }
    return (
        "以下是一条真实的 hunyuan 目标风格样例，供你模仿语气与标签用法（不要复述其内容）：\n"
        + json.dumps(demo, ensure_ascii=False, indent=2)
    )


# ----------------------------------------------------------------------
# 时间戳切分：把 10s 随机切成 N 段
# ----------------------------------------------------------------------
def split_timestamps(n_shots: int,
                     total: float = TOTAL_DURATION,
                     min_seg: float = 0.8) -> List[List[float]]:
    """
    把 [0, total] 随机切成 n_shots 段，返回 [[s0,e0],[s1,e1],...]，保留 1 位小数。
    每段至少 min_seg 秒。
    """
    n_shots = max(1, int(n_shots))
    if n_shots == 1:
        return [[0.0, round(total, 1)]]

    # 保证每段至少 min_seg：先剪掉 n_shots*min_seg，剩余部分随机分
    slack = total - n_shots * min_seg
    if slack < 0:
        # 段数太多，均分即可
        step = total / n_shots
        cuts = [round(step * i, 1) for i in range(1, n_shots)]
    else:
        # 生成 n_shots-1 个随机切点（在 slack 上），加上累计 min_seg
        ratios = sorted(random.random() for _ in range(n_shots - 1))
        cuts = []
        cum = 0.0
        for i, r in enumerate(ratios, start=1):
            cum = i * min_seg + r * slack
            cuts.append(round(cum, 1))

    # 组装区间，并做保护性去重/递增修正
    boundaries = [0.0] + cuts + [round(total, 1)]
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = round(boundaries[i - 1] + 0.1, 1)
    boundaries[-1] = round(total, 1)
    segs = [[boundaries[i], boundaries[i + 1]] for i in range(len(boundaries) - 1)]
    return segs


def format_scene_shot_detail(segs: List[List[float]]) -> str:
    """按 hunyuan CSV 里的形态输出： [[0, 2.3], [2.3, 5.6], ...]
    整数保留为整数、小数保留 1 位（跟原 CSV 的样式一致）。"""
    def _fmt(x: float) -> str:
        # 与原 CSV 观感一致：0 -> "0"; 2.3 -> "2.3"; 10.0 -> "10"
        if abs(x - round(x)) < 1e-6:
            return str(int(round(x)))
        return f"{x:.1f}".rstrip("0").rstrip(".") or "0"
    return "[" + ", ".join(
        "[" + _fmt(s) + ", " + _fmt(e) + "]" for s, e in segs
    ) + "]"


# ----------------------------------------------------------------------
# 从源 prompt 里数出镜头段数 N
# ----------------------------------------------------------------------
SHOT_HEADER_RE = re.compile(r"\[SHOT\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def infer_shot_count(src_prompt: str, default: int = 4) -> int:
    """从源 prompt 里数 [SHOT k/N] 得到 N；找不到就随机 3..5。"""
    m = SHOT_HEADER_RE.search(src_prompt)
    if m:
        try:
            n = int(m.group(2))
            if 1 <= n <= 12:
                return n
        except ValueError:
            pass
    # 兜底：数 [SHOT 出现次数
    cnt = len(re.findall(r"\[SHOT\s+\d+", src_prompt, re.IGNORECASE))
    if cnt >= 2:
        return min(cnt, 8)
    return random.choice([3, 4, 4, 5])


# ----------------------------------------------------------------------
# VL 调用
# ----------------------------------------------------------------------
class VLClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.7, timeout: float = 180.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature

    def rewrite(self, system_prompt: str, user_prompt: str,
                max_retry: int = 3) -> Dict[str, str]:
        """返回 {"prompt": ..., "prompt_wo_shots": ...}；失败抛异常。"""
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retry + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    top_p=0.9,
                    max_tokens=4096,
                )
                raw = resp.choices[0].message.content or ""
                return _parse_json_result(raw)
            except Exception as e:
                last_err = e
                wait = 2 * attempt
                print(f"    [warn] VL 调用失败(第{attempt}次): {e}; {wait}s 后重试",
                      file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"VL 调用连续失败：{last_err}")


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# 中文/大小写变体 -> 目标标签名。用于 _normalize_tags 兜底修复 VL 偶发误差。
_TAG_ALIASES: Dict[str, str] = {
    # 中文
    "音乐":   "music",
    "背景音乐": "music",
    "BGM":   "music",
    "bgm":   "music",
    "音效":   "sfx",
    "拟音":   "sfx",
    "环境音": "sfx",
    "SFX":   "sfx",
    "Sfx":   "sfx",
    "对白":   "speech",
    "台词":   "speech",
    "独白":   "speech",
    "人声":   "speech",
    "Speech":"speech",
    "SPEECH":"speech",
    "歌词":   "lyrics",
    "Lyrics":"lyrics",
    "LYRICS":"lyrics",
    "文字":   "text",
    "字幕":   "text",
    "水印":   "text",
    "Text":  "text",
    "TEXT":  "text",
    # 大写英文
    "Music": "music",
    "MUSIC": "music",
}


def _normalize_tags(text: str) -> str:
    """把 VL 偶发输出的中文/大小写变体音频标签统一成 hunyuan 标准英文小写标签。

    例：<音乐>...</音乐>  ->  <music>...</music>
        <SFX>...</SFX>    ->  <sfx>...</sfx>
    仅替换标签名，不动标签内部文字。
    """
    if not text:
        return text
    for alias, canonical in _TAG_ALIASES.items():
        if alias == canonical:
            continue
        # 处理开标签 <alias>  以及 带属性的 <alias xxx=...>（保守起见也覆盖）
        text = re.sub(
            r"<" + re.escape(alias) + r"(\s[^>]*)?>",
            lambda m: f"<{canonical}{m.group(1) or ''}>",
            text,
        )
        # 处理闭标签 </alias>
        text = re.sub(
            r"</" + re.escape(alias) + r"\s*>",
            f"</{canonical}>",
            text,
        )
    return text


def _parse_json_result(raw: str) -> Dict[str, str]:
    """从 VL 返回文本里抠出 JSON，兼容 ```json ...``` 包裹的情况。"""
    text = raw.strip()
    # 去掉 markdown 代码块围栏
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # 直接尝试
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            raise ValueError(f"VL 输出不是有效 JSON：{raw[:200]}...")
        obj = json.loads(m.group(0))
    if "prompt" not in obj:
        raise ValueError(f"VL 输出缺少 prompt 字段：{raw[:200]}...")
    prompt = _normalize_tags(str(obj["prompt"]).strip())
    prompt_wo_shots = _normalize_tags(str(obj.get("prompt_wo_shots") or "").strip())
    if not prompt_wo_shots:
        # 兜底：截 prompt 前 120 字
        prompt_wo_shots = prompt[:120]
    return {"prompt": prompt, "prompt_wo_shots": prompt_wo_shots}


# ----------------------------------------------------------------------
# 构造 user prompt
# ----------------------------------------------------------------------
def build_user_prompt_single(src_prompt: str) -> str:
    return (
        _example_block("single") + "\n\n"
        + "现在，请把下面这条【源中文提示词】改写成 hunyuan 单镜头目标风格，"
          "严格按 system 里描述的 JSON 格式返回：\n\n"
        + "【源中文提示词】\n" + src_prompt.strip() + "\n"
    )


def build_user_prompt_multi(src_prompt: str, segs: List[List[float]]) -> str:
    """告诉 VL：段数 + 每段的精确起止时间。"""
    shots_desc_lines = []
    for i, (s, e) in enumerate(segs, start=1):
        shots_desc_lines.append(f"  第{i}段：{s}s ~ {e}s")
    shots_desc = "\n".join(shots_desc_lines)
    return (
        _example_block("multi") + "\n\n"
        + f"【切分要求】总时长 10 秒，共 {len(segs)} 段，起止时间**必须严格按下列列表**（可以在原文里用不同的时间戳短语风格写出，但数字必须一致）：\n"
        + shots_desc + "\n\n"
        + "现在，请把下面这条【源中文提示词】改写成 hunyuan 多镜头目标风格，"
          "严格按 system 里描述的 JSON 格式返回：\n\n"
        + "【源中文提示词】\n" + src_prompt.strip() + "\n"
    )


# ----------------------------------------------------------------------
# 加载源 JSON
# ----------------------------------------------------------------------
def load_zh_json(category: str) -> List[Dict[str, Any]]:
    fp = ZH_DIR / f"{category}.json"
    if not fp.exists():
        raise FileNotFoundError(fp)
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{fp} 顶层不是 list")
    return data


# ----------------------------------------------------------------------
# CSV IO：读已有、写更新
# ----------------------------------------------------------------------
def load_existing(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """按 id 索引已有行，用于断点续跑。"""
    if not csv_path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id", "").strip()
            if rid:
                out[rid] = row
    return out


def write_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    """按 index 升序写出。utf-8-sig 编码。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    rows_sorted = sorted(rows, key=lambda r: int(r.get("index", 0)))
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({k: r.get(k, "") for k in CSV_HEADER})
    tmp.replace(csv_path)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def process_one_item(item: Dict[str, Any],
                     shot_kind: str,
                     index: int,
                     vl: VLClient) -> Dict[str, Any]:
    """
    改写一条，返回一行 CSV dict（列结构见 CSV_HEADER）。
    shot_kind: 'single' or 'multi'
    """
    src_prompt = item["prompt"]
    item_id = item["id"]

    if shot_kind == "single":
        user_prompt = build_user_prompt_single(src_prompt)
        result = vl.rewrite(SYSTEM_PROMPT_SINGLE, user_prompt)
        segs = [[0.0, TOTAL_DURATION]]
    else:
        n = infer_shot_count(src_prompt)
        segs = split_timestamps(n)
        user_prompt = build_user_prompt_multi(src_prompt, segs)
        result = vl.rewrite(SYSTEM_PROMPT_MULTI, user_prompt)

    return {
        "id": item_id,
        "index": index,
        "prompt": result["prompt"],
        "prompt_wo_shots": result["prompt_wo_shots"],
        "gt_video_cos_url": "",
        "gt_audio_cos_url": "",
        "num_frames": NUM_FRAMES,
        "scene_shot_detail": format_scene_shot_detail(segs),
        "seed": random.randint(1, 9999),
    }


def process_category(category: str,
                     shot_kind: str,
                     vl: VLClient,
                     concurrency: int,
                     limit: Optional[int],
                     overwrite: bool) -> None:
    """
    一个类别 + 一个 shot_kind (single/multi) 一个 CSV。
    """
    all_items = load_zh_json(category)
    # 语义：每类先取 JSON 前 limit 条，再按 shot 分流；
    # 这样每类 single + multi 合计正好 = limit（例如 100）。
    if limit is not None:
        all_items = all_items[:limit]
    target_flag = "single_shot" if shot_kind == "single" else "multi_shot"
    items = [it for it in all_items if it.get("shot_count") == target_flag]

    out_path = HUNYUAN_DIR / category / f"{category}_{shot_kind}_10s.csv"
    existing = {} if overwrite else load_existing(out_path)

    todo: List[Tuple[int, Dict[str, Any]]] = []
    kept_rows: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        rid = it["id"]
        if rid in existing:
            row = existing[rid]
            # 修补一下 index，防止历史 index 与本次不一致
            row["index"] = idx
            kept_rows.append(row)
        else:
            todo.append((idx, it))

    print(f"[{category}/{shot_kind}] 待处理 {len(todo)} 条 / 已有 {len(kept_rows)} 条 -> {out_path}")
    if not todo:
        # 只是补写一次（顺便修正 index）
        write_csv(out_path, kept_rows)
        return

    new_rows: List[Dict[str, Any]] = []
    errors: List[Tuple[str, str]] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(process_one_item, it, shot_kind, idx, vl): (idx, it["id"])
            for idx, it in todo
        }
        done = 0
        total = len(futures)
        flush_every = max(5, total // 20)  # 大约 5% flush 一次
        for fut in as_completed(futures):
            idx, rid = futures[fut]
            done += 1
            try:
                row = fut.result()
                new_rows.append(row)
            except Exception as e:
                errors.append((rid, str(e)))
                print(f"    [ERR] {rid}: {e}", file=sys.stderr)

            if done % flush_every == 0 or done == total:
                # 阶段性落盘（防止长任务中途丢结果）
                write_csv(out_path, kept_rows + new_rows)
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"    [{category}/{shot_kind}] {done}/{total} "
                      f"({rate:.2f}/s, ETA {eta/60:.1f}min, err={len(errors)})")

    # 最终落盘
    write_csv(out_path, kept_rows + new_rows)
    print(f"[{category}/{shot_kind}] 完成 -> {out_path} "
          f"(共 {len(kept_rows) + len(new_rows)} 行，失败 {len(errors)})")
    if errors:
        err_log = out_path.with_suffix(".errors.log")
        with err_log.open("w", encoding="utf-8") as f:
            for rid, msg in errors:
                f.write(f"{rid}\t{msg}\n")
        print(f"    错误详情写入 {err_log}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", type=str, default="all",
                   help=f"逗号分隔的类别，或 'all'。可选：{','.join(CATEGORIES)}")
    p.add_argument("--shot-count", type=str, default="all",
                   choices=["all", "single", "multi"],
                   help="改写单镜头/多镜头/两者都改")
    p.add_argument("--limit", type=int, default=None,
                   help="每个 (category, shot_kind) 只处理前 N 条（调试用）")
    p.add_argument("--concurrency", type=int, default=8,
                   help="并发调用 VL 的线程数（vLLM 侧自然排队）")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    p.add_argument("--api-key",  type=str, default=DEFAULT_API_KEY)
    p.add_argument("--model",    type=str, default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=None,
                   help="全局随机种子（影响 scene_shot_detail 切分与 seed 列）；不设则不固定")
    p.add_argument("--overwrite", action="store_true",
                   help="忽略已有 CSV，全部重新生成")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.categories.strip().lower() == "all":
        cats = CATEGORIES
    else:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        for c in cats:
            if c not in CATEGORIES:
                print(f"[FATAL] 未知类别：{c}；支持：{CATEGORIES}", file=sys.stderr)
                sys.exit(2)

    if args.shot_count == "all":
        shot_kinds = ["single", "multi"]
    else:
        shot_kinds = [args.shot_count]

    print("=" * 70)
    print(f" 源目录  : {ZH_DIR}")
    print(f" 输出目录: {HUNYUAN_DIR}/<category>/")
    print(f" VL 接口 : {args.base_url}  model={args.model}")
    print(f" 类别    : {cats}")
    print(f" 镜头种类: {shot_kinds}")
    print(f" 并发    : {args.concurrency}   limit={args.limit}   overwrite={args.overwrite}")
    print("=" * 70)

    vl = VLClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
    )

    # 连通性快速检查
    try:
        _ = vl.client.models.list()
    except Exception as e:
        print(f"[warn] models.list() 探测失败（可能是 vLLM 版本不支持），继续尝试正常调用: {e}",
              file=sys.stderr)

    for cat in cats:
        for sk in shot_kinds:
            try:
                process_category(
                    category=cat,
                    shot_kind=sk,
                    vl=vl,
                    concurrency=args.concurrency,
                    limit=args.limit,
                    overwrite=args.overwrite,
                )
            except FileNotFoundError as e:
                print(f"[SKIP] {cat}/{sk}: 源文件缺失 {e}", file=sys.stderr)
            except Exception as e:
                print(f"[FAIL] {cat}/{sk}: {e}", file=sys.stderr)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
