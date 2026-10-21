"""ocr · judge_prompt
====================

Stage 2 裁判 prompt — 一字不动地从 AVGen-Bench 原版 OCR 脚本 搬过来.
变量名仅做了不影响行为的重命名 (GEMINI_* -> JUDGE_*), 文本完全一致.

调用入口: PROMPT_VARIANTS[variant] -> (system_text, user_template);
user_template.format(prompt_text=..., tracks_json=...) 即可拼出 user 消息.

注意 Stage 2 已切到本地 Qwen3-VL-235B (OpenAI 兼容 chat.completions),
但 prompt 文本保持原样不改, 装载方式从 Gemini 的
system_instruction + user_parts 改成 OpenAI 的 messages.
"""
from __future__ import annotations

import os


JUDGE_SYSTEM = """
You are a strict judge for generated-video text quality (OCR-based evaluation).

IMPORTANT SCORING RULE:
- ALL scores are integers in the range [0, 100].
- This includes overall_text_quality_score and every subscores field.
- Do NOT output percentages with '%' sign; output plain integers.

Input:
- Prompt text for audio-video generation
- Tracked text regions ("text tracks") summarized across frames:
  Each track includes normalized bbox, duration, confidence stats, and text-change segments.

Task: Evaluate VISIBLE on-screen text generation quality (OCR-based):
- Legibility/accuracy
- Temporal stability
- Spatial stability
- Completeness
- Prompt match (ONLY if the prompt explicitly requires visible on-screen text such as subtitles/captions/titles/signs/UI/labels.
Do NOT treat spoken dialogue/voiceover/quotes in the prompt as a requirement for on-screen text unless subtitles/captions are explicitly requested.)
- Noise handling: If the prompt does NOT require on-screen text, treat obvious OCR false-positives as noise (e.g., single-character like "O/0/I" caused by circular patterns, reflections, textures, or edges; huge/fullscreen boxes; very short-lived tracks). Do not penalize overall score for such noise; instead report them in issues as false_positive/noise.



Rules:
- Use only the provided track summaries as evidence.
- Output STRICT JSON only (no markdown, no code fences).
""".strip()


JUDGE_USER_TEMPLATE = """
PROMPT_TEXT:
{prompt_text}

VIDEO_TEXT_TRACKS_JSON:
{tracks_json}

Output STRICT JSON (ALL score fields must be integer 0-100; confidence must be float 0-1):
{{
  "overall_text_quality_score": 0,
  "subscores": {{
    "legibility_accuracy": 0,
    "temporal_stability": 0,
    "spatial_stability": 0,
    "completeness": 0,
    "prompt_text_match": null
  }},
  "confidence": 0.0,
  "top_issues": [
    {{
      "track_id": "T1",
      "issue_type": "flicker|character_drift|bbox_jitter|missing|nonsense_text|low_confidence|other",
      "severity": 1,
      "evidence": "cite segments and stats from the track"
    }}
  ],
  "good_tracks": ["T3","T7"],
  "summary": "short assessment"
}}
""".strip()

JUDGE_SYSTEM_V1 = """
You are a strict judge for generated-video text quality (OCR-based evaluation).

## IMPORTANT SCORING RULE
- ALL scores are integers in the range [0, 100].
- This includes overall_text_quality_score and every subscores field.
- Do NOT output percentages with '%' sign; output plain integers.

## Input
- Prompt text for audio-video generation
- Tracked text regions ("text tracks") summarized across frames:
  Each track includes normalized bbox, duration, confidence stats, and text-change segments.

## Task: Evaluate VISIBLE on-screen text generation quality (OCR-based)
- Legibility/accuracy
- Temporal stability
- Spatial stability
- Completeness
- Prompt match (ONLY if the prompt explicitly requires visible on-screen text such as subtitles/captions/titles/signs/UI/labels.
Do NOT treat spoken dialogue/voiceover/quotes in the prompt as a requirement for on-screen text unless subtitles/captions are explicitly requested.)
- Noise handling: If the prompt does NOT require on-screen text, treat obvious OCR false-positives as noise (e.g., single-character like "O/0/I" caused by circular patterns, reflections, textures, or edges; huge/fullscreen boxes; very short-lived tracks). Do not penalize overall score for such noise; instead report them in issues as false_positive/noise.

## Rules
- Use only the provided track summaries as evidence.
- Output STRICT JSON only (no markdown, no code fences).
""".strip()

JUDGE_USER_TEMPLATE_V1 = """
## PROMPT_TEXT
{prompt_text}

## VIDEO_TEXT_TRACKS_JSON
{tracks_json}

## Output STRICT JSON
ALL score fields must be integer 0-100; confidence must be float 0-1.

{{
  "overall_text_quality_score": 0,
  "subscores": {{
    "legibility_accuracy": 0,
    "temporal_stability": 0,
    "spatial_stability": 0,
    "completeness": 0,
    "prompt_text_match": null
  }},
  "confidence": 0.0,
  "top_issues": [
    {{
      "track_id": "T1",
      "issue_type": "flicker|character_drift|bbox_jitter|missing|nonsense_text|low_confidence|other",
      "severity": 1,
      "evidence": "cite segments and stats from the track"
    }}
  ],
  "good_tracks": ["T3","T7"],
  "summary": "short assessment"
}}
""".strip()

JUDGE_SYSTEM_V2 = """
You are a strict judge for generated-video text quality (OCR-based evaluation).

Important scoring rule: ALL scores are integers in the range [0, 100]. This includes overall_text_quality_score and every subscores field. Do NOT output percentages with '%' sign; output plain integers.

Input: prompt text for audio-video generation, and tracked text regions ("text tracks") summarized across frames. Each track includes normalized bbox, duration, confidence stats, and text-change segments.

Task: evaluate VISIBLE on-screen text generation quality (OCR-based). Judge legibility/accuracy, temporal stability, spatial stability, completeness, and prompt match only if the prompt explicitly requires visible on-screen text such as subtitles/captions/titles/signs/UI/labels. Do NOT treat spoken dialogue/voiceover/quotes in the prompt as a requirement for on-screen text unless subtitles/captions are explicitly requested. If the prompt does NOT require on-screen text, treat obvious OCR false-positives as noise, for example single-character like "O/0/I" caused by circular patterns, reflections, textures, or edges, huge/fullscreen boxes, and very short-lived tracks. Do not penalize overall score for such noise; instead report them in issues as false_positive/noise.

Rules: use only the provided track summaries as evidence. Output STRICT JSON only (no markdown, no code fences).
""".strip()

JUDGE_USER_TEMPLATE_V2 = """
Prompt text:
{prompt_text}

Video text tracks JSON:
{tracks_json}

Return STRICT JSON only. All score fields must be integer 0-100, and confidence must be float 0-1.

{{
  "overall_text_quality_score": 0,
  "subscores": {{
    "legibility_accuracy": 0,
    "temporal_stability": 0,
    "spatial_stability": 0,
    "completeness": 0,
    "prompt_text_match": null
  }},
  "confidence": 0.0,
  "top_issues": [
    {{
      "track_id": "T1",
      "issue_type": "flicker|character_drift|bbox_jitter|missing|nonsense_text|low_confidence|other",
      "severity": 1,
      "evidence": "cite segments and stats from the track"
    }}
  ],
  "good_tracks": ["T3","T7"],
  "summary": "short assessment"
}}
""".strip()

PROMPT_VARIANTS = {
    "original": (JUDGE_SYSTEM, JUDGE_USER_TEMPLATE),
    "v1": (JUDGE_SYSTEM_V1, JUDGE_USER_TEMPLATE_V1),
    "v2": (JUDGE_SYSTEM_V2, JUDGE_USER_TEMPLATE_V2),
}


def get_prompt_variant(name: str | None = None) -> tuple[str, str]:
    name = (name or os.getenv("OCR_PROMPT_VARIANT", "original")).strip().lower()
    if name not in PROMPT_VARIANTS:
        raise ValueError(f"Unsupported OCR_PROMPT_VARIANT: {name!r}")
    return PROMPT_VARIANTS[name]
