#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2AV-Bench Speech 子集 ASR 评测脚本（spec-based 版）
====================================================

输入：
  1) 生成视频目录（mp4，文件名 = speech_XXXX.mp4）
  2) t2av-bench 的 specs/speech.jsonl（结构化 GT）

流程：
  1) 用 ffmpeg 把每个 mp4 抽成 16kHz 单声道 wav
  2) 从 spec 中读取台词作为 ref：
     - 优先 shots[].dialogue[]（按 shot_idx 升序），即真正的发声时序，
       multi_shot 多说话人交错对白也能正确对齐；
     - 回退到 subjects[].speech（is_speaker=True）。
     —— 字幕 / news_banner / signage 等画面文字进入 spec.text_on_screen，
        天然不在 ref 中，自动避开"非语音文本"干扰
  3) 调 Qwen3-ASR-1.7B 转写每个 wav 得到 hyp_text
     语种从 spec.speech_attributes.language 直取（不再启发式）
  4) 用 jiwer 计算 WER（中文按字、英文按词），输出 final_score = mean(1 - clip(wer,0,1))
  5) single_shot / multi_shot 都评测

可作为模块被外层脚本调用：
    from test_wer_t2av import run_one_model, load_asr_model
    model = load_asr_model(...)
    run_one_model(model, video_dir, spec_jsonl, audio_dir, out_json)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import List, Dict, Optional

import torch
import jiwer
from tqdm import tqdm

# 复用 test_wer.py 里的归一化逻辑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_wer import normalize  # noqa: E402

from qwen_asr import Qwen3ASRModel  # noqa: E402
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[4]

# ============================================================
# 默认路径配置（可被命令行覆盖）
# ============================================================
DEFAULT_VIDEO_DIR = str(_REPO_ROOT / "LTX-2.3_test/output/speech")
DEFAULT_SPEC_JSONL = str(_REPO_ROOT / "t2av-bench/specs/speech.jsonl")
DEFAULT_AUDIO_DIR = str(_REPO_ROOT / "evaluation/asr_eval/audio_extracted_t2av")
DEFAULT_MODEL_PATH = str(_REPO_ROOT / "evaluation/asr_eval/models/Qwen3-ASR-1.7B")
DEFAULT_OUT_JSON = str(_REPO_ROOT / "evaluation/asr_eval/resource/wer_results_t2av.json")

# spec 中语种代码（小写）→ Qwen3-ASR transcribe(language=...) 期望值
LANG_MAP = {
    "chinese":  "Chinese",
    "english":  "English",
    "japanese": "Japanese",
    "french":   "French",
    "russian":  "Russian",
    "german":   "German",
    "korean":   "Korean",
    "spanish":  "Spanish",
    "arabic":   "Arabic",
}

# ============================================================
# 过滤选项
# ============================================================
# --shot-count 取值 → 在 spec 中需要匹配的 shot_count 字段值
SHOT_COUNT_FILTER = {
    "single": "single_shot",
    "multi":  "multi_shot",
    "all":    None,  # 不过滤
}

# ============================================================
# Spec 加载 + ref 构造
# ============================================================
def load_specs(spec_jsonl: str) -> Dict[str, Dict]:
    """加载 jsonl 为 {id: spec_dict}"""
    specs: Dict[str, Dict] = {}
    with open(spec_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            specs[s["id"]] = s
    return specs


def extract_ref_from_spec(spec: Dict) -> Dict:
    """从一条 spec 中抽出 ref_lines / ref_text / language / speakers。

    Ref 来源优先级：
      1) `shots[].dialogue[]`（按 shot_idx 升序，每镜头内按 dialogue 数组顺序）
         —— 这是真正的发声时序，等价于 ASR 整段转写顺序，
            multi_shot 多说话人交错对白也能正确对齐；
            还能覆盖 `is_speaker=False` 但实际有念词的角色（如旁白/合成 voiceover/学生提问）。
      2) 回退到 `subjects[].speech`（只取 is_speaker=True）
         —— 仅当 spec 没有填 shots[].dialogue 时使用。

    若两者都拿不到任何文本，返回空 dict（调用方应跳过）。

    返回的 speakers 字段是与 ref_lines 一一对应的 speaker_id 序列（按句），
    不再是"该样本里 is_speaker subjects 列表"。
    """
    ref_lines: List[str] = []
    speakers:  List[str] = []

    # 1) 优先：shots[].dialogue
    shots = sorted(
        spec.get("shots") or [],
        key=lambda x: x.get("shot_idx", 0),
    )
    for sh in shots:
        for d in (sh.get("dialogue") or []):
            text = (d.get("text") or "").strip()
            if not text:
                continue
            ref_lines.append(text)
            speakers.append(d.get("speaker_id", "?"))

    # 2) 回退：subjects[].speech
    if not ref_lines:
        for sub in spec.get("subjects", []) or []:
            if not sub.get("is_speaker"):
                continue
            text = (sub.get("speech") or "").strip()
            if not text:
                continue
            ref_lines.append(text)
            speakers.append(sub.get("id", "?"))

    if not ref_lines:
        return {}

    lang_code = (spec.get("speech_attributes", {}) or {}).get("language", "")
    return {
        "ref_lines": ref_lines,
        "ref_text":  " ".join(ref_lines),
        "speakers":  speakers,
        "language":  LANG_MAP.get(lang_code.lower()),  # 可能为 None
    }


# ============================================================
# 音频提取：mp4 -> wav (16kHz mono)
# ============================================================
def extract_audio(mp4_path: str, wav_path: str, sample_rate: int = 16000) -> bool:
    """用 ffmpeg 把 mp4 抽成单声道 wav。每次都重抽，不使用任何缓存，
    确保评测的永远是当前输入的 mp4。"""
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", mp4_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed for {mp4_path}: {e.stderr.decode(errors='ignore')[:200]}")
        return False
    except FileNotFoundError:
        print("[FATAL] ffmpeg not found. Please install ffmpeg first.")
        sys.exit(1)


# ============================================================
# 样本组装
# ============================================================
def build_samples(video_dir: str, specs: Dict[str, Dict], audio_dir: str,
                  shot_count: str = "all") -> List[Dict]:
    """扫描 video_dir 下的 mp4，与 spec 配对，构造样本列表。

    shot_count: "single" / "multi" / "all"，控制是否按 spec.shot_count 过滤。
    """
    if not os.path.isdir(video_dir):
        print(f"[ERROR] video_dir not found: {video_dir}")
        return []

    if shot_count not in SHOT_COUNT_FILTER:
        raise ValueError(f"shot_count must be one of {list(SHOT_COUNT_FILTER)}, got {shot_count!r}")
    want_sc = SHOT_COUNT_FILTER[shot_count]  # None = all

    mp4_files = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))

    samples = []
    skipped = 0
    skipped_by_filter = 0
    for fname in mp4_files:
        sid = os.path.splitext(fname)[0]  # speech_0007
        if sid not in specs:
            print(f"[WARN] {sid} not found in spec, skip")
            skipped += 1
            continue
        spec = specs[sid]
        sc = spec.get("shot_count")
        if want_sc is not None and sc != want_sc:
            skipped_by_filter += 1
            continue
        ref = extract_ref_from_spec(spec)
        if not ref:
            print(f"[WARN] {sid} has no speaker.speech in spec, skip")
            skipped += 1
            continue

        samples.append({
            "id":         sid,
            "mp4_path":   os.path.join(video_dir, fname),
            "wav_path":   os.path.join(audio_dir, sid + ".wav"),
            "ref_lines":  ref["ref_lines"],
            "ref_text":   ref["ref_text"],
            "language":   ref["language"],
            "speakers":   ref["speakers"],
            "shot_count": spec.get("shot_count"),
            "audio_situation": spec.get("audio_situation"),
        })

    print(f"  shot_count filter = {shot_count!r} (want={want_sc!r})")
    print(f"  collected {len(samples)} samples, "
          f"skipped {skipped} (no-spec/no-speech), "
          f"skipped {skipped_by_filter} (by shot_count filter)")
    return samples


# ============================================================
# 模型加载（拆出来便于外层复用同一个模型实例）
# ============================================================
def load_asr_model(model_path: str, device: str, batch_size: int,
                   max_new_tokens: int = 256):
    print(f"  loading Qwen3-ASR model ({model_path}) on {device} ...")
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    print("  model loaded.")
    return model


# ============================================================
# 单模型评测主流程
# ============================================================
def run_one_model(model, video_dir: str, spec_jsonl: str, audio_dir: str,
                  out_json: str, batch_size: int = 4,
                  auto_language: bool = False,
                  tag: str = "",
                  shot_count: str = "single") -> Optional[Dict]:
    """对某个模型生成的视频目录跑一次完整评测，结果写到 out_json。
    返回汇总 dict（或 None 表示无可评样本）。

    shot_count: "single" / "multi" / "all"，仅评测对应 shot_count 的样本。
    """
    print("=" * 60)
    print(f"[run_one_model] tag='{tag}'  video_dir={video_dir}  shot_count={shot_count}")
    print("=" * 60)

    # 1) 检 ffmpeg
    if shutil.which("ffmpeg") is None:
        print("[FATAL] ffmpeg not found in PATH")
        sys.exit(1)

    # 2) 加载 specs + 组样本
    specs = load_specs(spec_jsonl)
    samples = build_samples(video_dir, specs, audio_dir, shot_count=shot_count)
    if not samples:
        print("  [WARN] no evaluable samples, skip")
        return None

    # 3) 抽音频
    print(f"  extracting audio to {audio_dir}")
    for s in tqdm(samples, desc="    ffmpeg"):
        ok = extract_audio(s["mp4_path"], s["wav_path"])
        if not ok:
            print(f"    [ERROR] extract failed: {s['id']}")
    samples = [s for s in samples if os.path.exists(s["wav_path"])]
    print(f"  {len(samples)} audios ready")
    if not samples:
        return None

    # 4) ASR 推理
    audio_paths = [s["wav_path"] for s in samples]
    languages   = [None if auto_language else s["language"] for s in samples]

    hyps: List[str] = []
    detected_langs: List[Optional[str]] = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="    ASR"):
        batch_audio = audio_paths[i:i + batch_size]
        batch_lang  = languages[i:i + batch_size]
        results = model.transcribe(audio=batch_audio, language=batch_lang)
        for r in results:
            hyps.append(r.text or "")
            detected_langs.append(getattr(r, "language", None))

    # 5) WER 计算
    refs = [s["ref_text"] for s in samples]
    norm_refs = [normalize(r) for r in refs]
    norm_hyps = [normalize(h) for h in hyps]

    overall_wer = jiwer.wer(norm_refs, norm_hyps)

    per_sample_wer = []
    for nref, nhyp in zip(norm_refs, norm_hyps):
        if not nref.strip():
            per_sample_wer.append(float("nan"))
            continue
        try:
            w = jiwer.wer(nref, nhyp)
        except Exception:
            w = float("nan")
        per_sample_wer.append(w)

    # 综合分：mean over videos of (1 - clip(wer,0,1))，NaN→0
    per_sample_score = [
        max(0.0, 1.0 - min(w, 1.0)) if w == w else 0.0
        for w in per_sample_wer
    ]
    final_score = (sum(per_sample_score) / len(per_sample_score)) if per_sample_score else 0.0

    # 6) 打印每条
    print("-" * 60)
    for s, ref, hyp, w, dlang in zip(samples, refs, hyps, per_sample_wer, detected_langs):
        print(f"\n  [{s['id']}]  shot={s['shot_count']}  forced_lang={s['language']}  detected={dlang}  WER={w:.4f}")
        print(f"    REF: {normalize(ref, for_display=True)[:200]}")
        print(f"    HYP: {normalize(hyp, for_display=True)[:200]}")

    print()
    print("=" * 60)
    print(f"  [{tag}] overall_wer  = {overall_wer:.4f}  ({overall_wer*100:.2f}%)")
    print(f"  [{tag}] final_score  = {final_score:.4f}   (mean over videos of 1-clip(wer,0,1))")
    print(f"  [{tag}] num_samples  = {len(samples)}")
    print("=" * 60)

    # 7) 保存
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    out = {
        "tag": tag,
        "shot_count_filter": shot_count,
        "final_score": final_score,
        "overall_wer": overall_wer,
        "num_samples": len(samples),
        "video_dir":   video_dir,
        "spec_jsonl":  spec_jsonl,
        "details": [
            {
                "id":               s["id"],
                "mp4_path":         s["mp4_path"],
                "wav_path":         s["wav_path"],
                "shot_count":       s["shot_count"],
                "audio_situation":  s["audio_situation"],
                "speakers":         s["speakers"],
                "forced_language":  s["language"],
                "detected_language": detected_langs[i],
                "ref_lines":        s["ref_lines"],
                "ref_text":         refs[i],
                "hyp_text":         hyps[i],
                "wer":              per_sample_wer[i],
                "score":            per_sample_score[i],
            }
            for i, s in enumerate(samples)
        ],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  results saved to: {out_json}")
    return out


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="T2AV-Bench Speech 子集 ASR 评测（spec-based）")
    parser.add_argument("--video-dir",  default=DEFAULT_VIDEO_DIR,  help="生成视频所在目录（直接给 speech/ 子目录）")
    parser.add_argument("--spec-jsonl", default=DEFAULT_SPEC_JSONL, help="t2av-bench specs/speech.jsonl 路径")
    parser.add_argument("--audio-dir",  default=DEFAULT_AUDIO_DIR,  help="抽取后的 wav 存放目录")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Qwen3-ASR 模型路径")
    parser.add_argument("--out-json",   default=DEFAULT_OUT_JSON,   help="评测结果输出路径")
    parser.add_argument("--device",     default="cuda:0",           help="推理设备")
    parser.add_argument("--batch-size", type=int, default=4,        help="ASR batch size")
    parser.add_argument("--auto-language", action="store_true",     help="不传 language，让模型自动检测")
    parser.add_argument("--tag", default="",                        help="日志/输出标签（如模型名）")
    parser.add_argument("--shot-count", choices=["single", "multi", "all"], default="single",
                        help="只评测指定 shot_count 的样本：single（默认）/multi/all")
    args = parser.parse_args()

    model = load_asr_model(args.model_path, args.device, args.batch_size)
    run_one_model(
        model,
        video_dir=args.video_dir,
        spec_jsonl=args.spec_jsonl,
        audio_dir=args.audio_dir,
        out_json=args.out_json,
        batch_size=args.batch_size,
        auto_language=args.auto_language,
        tag=args.tag or os.path.basename(args.video_dir),
        shot_count=args.shot_count,
    )


if __name__ == "__main__":
    main()
