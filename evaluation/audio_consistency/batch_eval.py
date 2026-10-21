#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_consistency 评测点 · 单模型 Qwen3-Omni-Instruct 评测脚本 (grouped 模式)

依赖前置:
  上一步 pipeline/prepare_clips.py 已经为该模型生成了:
      <out_dir>/shotcut_manifest.jsonl
      <out_dir>/clips/<spec_id>_shot<idx>.mp4   (multi_shot pass 的样本)
      single_shot 的 clip_path 直接指向原 mp4

输入:
  --manifest        shotcut_manifest.jsonl 路径
  --questions       evaluation/audio_consistency/questions.jsonl
  --specs-dir       t2av-bench/specs (拿 scene_setting / shots[i].scene_setting)
  --system-prompt   evaluation/audio_consistency/system_prompt.md
  --model-path      Qwen3-Omni-30B-A3B-Instruct 权重目录
  --out-dir         本模型 runs 目录, 与 manifest 同根
  --tag             模型名 (打印用)
  --max-new-tokens  thinker_max_new_tokens 基础值 (默认 512). 每个 clip 实际预算
                    会按当前 bucket 题数 N 动态增加: min(4096, max_new_tokens + 256*N)
                    (Instruct 变体直接吐 JSON, 不需要为 <think> 段预留额度,
                    单题 ~80 tokens 就够; 默认 512 + 256*N 已留足余量)
  --limit           可选: 只跑前 N 个 spec, 调试用

Grouped 模式 (核心):
  同一个 (spec_id, shot_idx) 下的所有问题 (典型 1~4 题, 末尾 1 道 score + 前若干道 yesno)
  会被合并到 *一条* user 消息发给模型, 视频只编码 1 次. 模型按指令返回长度=N 的 JSON 数组,
  每个元素与 questions 列表逐位对齐. 解析失败的元素走 fallback (yesno->no, score->1).

题型:
  Type 1 (yes/no):  question 不以 "How well does the audio" 开头
  Type 2 (score):   question == "How well does the audio in the video match the prompt?"

scene_setting (供 score 题在 grouped prompt 顶部公用 1 次):
  single_shot scene_setting 取 spec["scene_setting"]
  multi_shot  scene_setting 取 spec["shots"][shot_idx-1]["scene_setting"]

输出:
  <out-dir>/per_question.jsonl   每条 question 一行 (grouped 模式下, 同一个 bucket 的 N 行
                                 共享相同的 raw_response, 但 reason/answer/score 各自解析)
  <out-dir>/summary.json         统计计数; 打分聚合先留空 (score_aggregation: null)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# 让脚本能 import 同级仓库下的 qwen35_api 包 (API backend 用)
_THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_REPO_DIR = os.path.dirname(_THIS_FILE_DIR)
if _EVAL_REPO_DIR not in sys.path:
    sys.path.insert(0, _EVAL_REPO_DIR)


# ============================================================
# 题型判定
# ============================================================
SCORE_QUESTION_PREFIX = "How well does the audio"


def is_score_question(q: str) -> bool:
    return q.strip().startswith(SCORE_QUESTION_PREFIX)


# ============================================================
# 加载: questions.jsonl / specs / system_prompt
# ============================================================
def load_questions(path: str) -> Dict[Tuple[str, int], List[dict]]:
    """
    返回 {(spec_id, shot_idx): [question_obj, ...]}, 保留 jsonl 内顺序.
    """
    out: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] questions.jsonl:{line_no} json error: {e}", file=sys.stderr)
                continue
            sid = obj.get("spec_id")
            sidx = obj.get("shot_idx")
            if not sid or sidx is None:
                continue
            out[(sid, int(sidx))].append(obj)
    return out


def load_specs(specs_dir: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name in sorted(os.listdir(specs_dir)):
        if not name.endswith(".jsonl"):
            continue
        cat = name[:-len(".jsonl")]
        with open(os.path.join(specs_dir, name), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("id")
                if sid:
                    obj["__category"] = cat
                    out[sid] = obj
    return out


def get_scene_setting(spec: dict, shot_idx: int) -> str:
    """题型 2 用. single_shot 用 spec.scene_setting, multi_shot 用 shots[idx-1].scene_setting."""
    kind = spec.get("shot_count")
    if kind == "single_shot":
        return spec.get("scene_setting") or ""
    if kind == "multi_shot":
        shots = spec.get("shots") or []
        if 1 <= shot_idx <= len(shots):
            return (shots[shot_idx - 1] or {}).get("scene_setting") or ""
        return ""
    return spec.get("scene_setting") or ""


def load_manifest(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] manifest line skip: {e}", file=sys.stderr)
    return out


def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ============================================================
# Qwen3-Omni Instruct 推理封装
# ============================================================
class OmniRunner:
    """加载 Qwen3-Omni Instruct, 单条 generate 一次.

    device:
        - "auto"   -> device_map="auto" (跨多卡自动切分, 旧行为)
        - "cuda:0" / "cuda:1" / ...   -> 强制单卡 (DP 模式: 8 进程 x 1 卡 各跑各的)
    """

    def __init__(self, model_path: str, attn_impl: str = "sdpa",
                 use_audio_in_video: bool = True, device: str = "auto"):
        import torch  # 延迟 import, 避免脚本入口时间过长
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
            AutoConfig,
        )
        from qwen_omni_utils import process_mm_info

        self.torch = torch
        self.process_mm_info = process_mm_info
        self.use_audio_in_video = use_audio_in_video

        device_map = "auto" if device == "auto" else device
        print(f"[omni] loading model from {model_path} (attn={attn_impl}, device_map={device_map}) ...", flush=True)
        t0 = time.time()
        # 仅做文本评测, 关闭 talker 以避免 transformers Code2WavConfig 缺 rope_parameters 的加载 bug
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(cfg, "enable_audio_output"):
            cfg.enable_audio_output = False
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            config=cfg,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=device_map,
        )
        self.model.eval()
        print(f"[omni]   model loaded in {time.time()-t0:.1f}s", flush=True)

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        print("[omni]   processor loaded", flush=True)

    def _build_messages(self, system_text: str, video_path: str, user_text: str) -> List[dict]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def generate(self, system_text: str, video_path: str, user_text: str,
                 max_new_tokens: int = 512) -> str:
        torch = self.torch
        messages = self._build_messages(system_text, video_path, user_text)

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = self.process_mm_info(
            messages, use_audio_in_video=self.use_audio_in_video
        )
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with torch.inference_mode():
            text_ids, _audio_out = self.model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=max_new_tokens,
                thinker_do_sample=False,
                use_audio_in_video=self.use_audio_in_video,
                return_audio=False,
            )
        response = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response


# ============================================================
# 解析 Omni 输出 -> JSON
# ============================================================
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

# ============================================================
# Grouped prompt 构造 / 解析
# ============================================================
def build_grouped_user_text(qs: List[dict], scene_setting: str) -> str:
    """把同一个 (spec_id, shot_idx) 下的 N 个 question 拼成一条 user 消息.

    qs: questions.jsonl 顺序保留. 末尾通常是 1 道 score 题, 前面是 yesno 题.
    scene_setting: 仅当至少有一道 score 题时贴在顶部, 全 bucket 公用 1 次.
    """
    has_score = any(is_score_question(q.get("question", "")) for q in qs)
    n = len(qs)

    lines: List[str] = []
    if has_score and scene_setting:
        lines.append(f"Original prompt: {scene_setting}")
        lines.append("")
    lines.append(f"You will answer the following {n} question(s) about the SAME video clip above.")
    lines.append("Each question is independent. For each one, follow its corresponding output format from the system rules:")
    lines.append("  - For yes/no questions, the answer item must be {\"reason\": ..., \"answer\": \"yes\"|\"no\"}.")
    lines.append("  - For the scoring question (\"How well does the audio in the video match the prompt?\"), the answer item must be {\"reason\": ..., \"score\": <int 1-5>}.")
    lines.append("")
    lines.append("Questions:")
    for i, q in enumerate(qs, start=1):
        qtype_tag = "score" if is_score_question(q.get("question", "")) else "yesno"
        lines.append(f"  {i}. [{qtype_tag}] {q.get('question', '').strip()}")
    lines.append("")
    lines.append(
        f"Return EXACTLY ONE JSON ARRAY of length {n}, where the i-th element is the answer to the i-th question above, "
        "in the same order. No prose, no markdown fences, no extra keys. Just the JSON array."
    )
    return "\n".join(lines)


_JSON_ARR_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,\s*)*\{.*?\}\s*\]", re.DOTALL)


def _coerce_one(elem: Any, qtype: str) -> Tuple[Optional[dict], bool]:
    """把 JSON 数组里的单个元素归一到 yesno / score 格式, 返回 (parsed, ok)."""
    if not isinstance(elem, dict):
        return None, False
    if qtype == "yesno":
        ans = elem.get("answer")
        if isinstance(ans, str) and ans.strip().lower() in ("yes", "no"):
            return {"reason": str(elem.get("reason", "")),
                    "answer": ans.strip().lower()}, True
    elif qtype == "score":
        sc = elem.get("score")
        if isinstance(sc, bool):
            return None, False
        if isinstance(sc, int) and 1 <= sc <= 5:
            return {"reason": str(elem.get("reason", "")),
                    "score": int(sc)}, True
        if isinstance(sc, str) and sc.strip().isdigit():
            v = int(sc.strip())
            if 1 <= v <= 5:
                return {"reason": str(elem.get("reason", "")),
                        "score": v}, True
    return None, False


def parse_grouped_response(raw: str, qs: List[dict]) -> Tuple[List[Optional[dict]], List[bool]]:
    """
    解析模型对 N 题的总输出 (期望一个长度=N 的 JSON 数组).
    返回 (parsed_list, ok_list), 长度均为 len(qs).
    任意位置解析失败该位置 parsed=None, ok=False (调用方写 fallback).
    """
    n = len(qs)
    qtypes = [("score" if is_score_question(q.get("question", "")) else "yesno") for q in qs]
    parsed: List[Optional[dict]] = [None] * n
    ok: List[bool] = [False] * n

    if not raw:
        return parsed, ok

    # 候选 array: 整段 strip / 末尾 ```json fence 剥离 / 正则提取最后一个 [...]
    cand_arr_strs: List[str] = []
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("json").lstrip("JSON").strip()
    cand_arr_strs.append(s)
    matches = _JSON_ARR_RE.findall(raw)
    if matches:
        cand_arr_strs.append(matches[-1])

    arr: Optional[list] = None
    for cand in cand_arr_strs:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, list):
            arr = obj
            break

    # Fallback: 找不到合法 array, 但能找到 N 个独立 JSON 对象 (按出现顺序)
    if arr is None:
        objs = _JSON_OBJ_RE.findall(raw)
        coerced: List[dict] = []
        for o in objs:
            try:
                d = json.loads(o)
                if isinstance(d, dict):
                    coerced.append(d)
            except Exception:
                continue
        if len(coerced) >= n:
            arr = coerced[:n]

    # Fallback 3: array / N-objects 都拿不到 (典型: 最后一道题被截, ']' 没输出).
    # 扫描最后一个 '[' 之后, 按花括号深度收集已完整闭合的 {...} 对象, 按位次填回.
    if arr is None:
        lb = raw.rfind("[")
        if lb >= 0:
            tail = raw[lb + 1 :]
            collected: List[dict] = []
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
                                d = json.loads(tail[start : k + 1])
                                if isinstance(d, dict):
                                    collected.append(d)
                            except Exception:
                                pass
                            start = -1
            if collected:
                arr = collected[:n]

    if arr is None:
        return parsed, ok

    # 长度对齐 (按位)
    for i in range(min(n, len(arr))):
        p, k = _coerce_one(arr[i], qtypes[i])
        if k:
            parsed[i] = p
            ok[i] = True
    return parsed, ok


def parse_response(raw: str, qtype: str) -> Tuple[Optional[dict], bool]:
    """
    qtype: "yesno" | "score"
    返回 (parsed_dict_or_None, parse_ok).
    parse_ok=True 表示找到了合法 JSON 且字段类型符合预期.
    """
    if not raw:
        return None, False

    # 1) 直接 try
    candidates: List[str] = []
    s = raw.strip()
    candidates.append(s)
    # 2) 取最后一个 {...} 块 (兼容模型在 JSON 前后带散文)
    matches = _JSON_OBJ_RE.findall(s)
    if matches:
        candidates.append(matches[-1])

    for cand in candidates:
        cand = cand.strip()
        # 去掉可能的 ```json fence
        if cand.startswith("```"):
            cand = cand.strip("`")
            cand = cand.lstrip("json").lstrip("JSON").strip()
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if qtype == "yesno":
            ans = obj.get("answer")
            if isinstance(ans, str) and ans.strip().lower() in ("yes", "no"):
                return {"reason": str(obj.get("reason", "")),
                        "answer": ans.strip().lower()}, True
        elif qtype == "score":
            sc = obj.get("score")
            if isinstance(sc, bool):
                # python json: True/False 也是 int 实例, 单独排除
                continue
            if isinstance(sc, int) and 1 <= sc <= 5:
                return {"reason": str(obj.get("reason", "")),
                        "score": int(sc)}, True
            # 兼容字符串数字
            if isinstance(sc, str) and sc.strip().isdigit():
                v = int(sc.strip())
                if 1 <= v <= 5:
                    return {"reason": str(obj.get("reason", "")),
                            "score": v}, True
    return None, False


# ============================================================
# 主流程
# ============================================================
def _build_todo(manifest_path: str, questions_path: str, specs: Dict[str, dict],
                limit: int = 0
                ) -> Tuple[List[Tuple[str, int, str, dict, List[dict]]], List[dict]]:
    """对单个 manifest 收集 todo bucket.

    todo 元素 = (spec_id, shot_idx, clip_path, spec_obj, [qobj,...])  仅含 passed
               且有 question 的 (spec, shot) 组合.
    manifest_raw = 原 manifest, 供 summary 记录 n_specs_in_manifest 使用.
    """
    questions = load_questions(questions_path)
    manifest = load_manifest(manifest_path)

    passed_specs = [m for m in manifest if m.get("passed")]
    if limit and limit > 0:
        passed_specs = passed_specs[: limit]

    todo: List[Tuple[str, int, str, dict, List[dict]]] = []
    for m in passed_specs:
        sid = m["spec_id"]
        spec = specs.get(sid)
        if spec is None:
            print(f"[WARN] spec missing for {sid}, skip", file=sys.stderr)
            continue
        for shot in m.get("shots", []):
            sidx = int(shot["shot_idx"])
            clip = shot["clip_path"]
            qs = questions.get((sid, sidx), [])
            if not qs:
                continue
            todo.append((sid, sidx, clip, spec, list(qs)))
    return todo, manifest


def _run_one_job(
    runner: "OmniRunner",
    *,
    tag: str,
    manifest_path: str,
    questions_path: str,
    specs: Dict[str, dict],
    sys_prompt: str,
    out_dir: str,
    shard_idx: int,
    shard_num: int,
    max_new_tokens_base: int,
    limit: int,
) -> None:
    """在已加载好的 runner 上, 跑一个 (manifest, out_dir) 作业."""
    print()
    print(f"############ [batch_eval] job tag={tag} (shard {shard_idx}/{shard_num}) ############")
    print(f"  manifest = {manifest_path}")
    print(f"  out_dir  = {out_dir}")
    print(f"  questions= {questions_path}", flush=True)

    todo, manifest_raw = _build_todo(manifest_path, questions_path, specs, limit=limit)

    n_total_buckets_global = len(todo)
    n_total_questions_global = sum(len(t[4]) for t in todo)
    passed_count = len([m for m in manifest_raw if m.get("passed")])
    print(f"[batch_eval] passed specs: {passed_count} / {len(manifest_raw)} -> todo {n_total_buckets_global} buckets")
    print(f"[batch_eval] total buckets (global)  : {n_total_buckets_global}")
    print(f"[batch_eval] total questions (global): {n_total_questions_global}", flush=True)

    sharded = (shard_num > 1)
    if sharded:
        todo = [t for i, t in enumerate(todo) if i % shard_num == shard_idx]
        kept_q = sum(len(t[4]) for t in todo)
        print(f"[batch_eval] shard {shard_idx}/{shard_num} -> keep {len(todo)} buckets ({kept_q} questions)",
              flush=True)

    os.makedirs(out_dir, exist_ok=True)
    if sharded:
        out_path = os.path.join(out_dir, f"per_question.shard{shard_idx}.jsonl")
        summary_path = os.path.join(out_dir, f"summary.shard{shard_idx}.json")
    else:
        out_path = os.path.join(out_dir, "per_question.jsonl")
        summary_path = os.path.join(out_dir, "summary.json")

    n_buckets = len(todo)
    n_questions_total = sum(len(t[4]) for t in todo)

    if n_buckets == 0:
        # 该 job 在本 shard 上无任务; 仍然写出空文件 + summary, 避免下游 merge 报错.
        open(out_path, "w", encoding="utf-8").close()
        summary = {
            "tag": tag,
            "model_path": getattr(runner, "model_path", ""),
            "manifest": manifest_path,
            "n_specs_in_manifest": len(manifest_raw),
            "n_specs_passed": passed_count,
            "n_buckets": 0,
            "n_questions": 0,
            "n_yesno_questions": 0,
            "n_score_questions": 0,
            "n_parse_ok": 0,
            "n_parse_fail": 0,
            "score_aggregation": None,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[batch_eval] (job={tag}) nothing to run on this shard, wrote empty {out_path}")
        return

    n_yesno = 0
    n_score = 0
    n_parse_ok = 0
    n_parse_fail = 0
    t_start = time.time()

    with open(out_path, "w", encoding="utf-8") as fout:
        for bi, (sid, sidx, clip, spec, qs) in enumerate(todo, start=1):
            n_in_bucket = len(qs)
            qtypes = [("score" if is_score_question(q.get("question", "")) else "yesno") for q in qs]
            n_yesno += sum(1 for t in qtypes if t == "yesno")
            n_score += sum(1 for t in qtypes if t == "score")

            scene_setting_used = get_scene_setting(spec, sidx)
            user_text = build_grouped_user_text(qs, scene_setting_used)

            # 动态 token 预算: base + 256*N, 上限 4096.
            mnt = min(4096, int(max_new_tokens_base) + 256 * n_in_bucket)

            t0 = time.time()
            try:
                raw = runner.generate(
                    system_text=sys_prompt,
                    video_path=clip,
                    user_text=user_text,
                    max_new_tokens=mnt,
                )
                err = ""
            except Exception as e:
                raw = ""
                err = repr(e)
            dt = time.time() - t0

            parsed_list, ok_list = parse_grouped_response(raw, qs)

            # fallback per system_prompt rule 7
            for j in range(n_in_bucket):
                if not ok_list[j]:
                    if qtypes[j] == "yesno":
                        parsed_list[j] = {"reason": "[parse_fail fallback]", "answer": "no"}
                    else:
                        parsed_list[j] = {"reason": "[parse_fail fallback]", "score": 1}
                    n_parse_fail += 1
                else:
                    n_parse_ok += 1

            for j, qobj in enumerate(qs):
                q = qobj.get("question", "").strip()
                qtype = qtypes[j]
                p = parsed_list[j] or {}
                rec = {
                    "spec_id": sid,
                    "shot_idx": sidx,
                    "category": spec.get("__category", ""),
                    "shot_count_kind": spec.get("shot_count", ""),
                    "clip_path": clip,
                    "question": q,
                    "question_type": qtype,
                    "associated_subject_id": qobj.get("associated_subject_id"),
                    "scene_setting_used": scene_setting_used if qtype == "score" else "",
                    "raw_response": raw,
                    "reason": p.get("reason", ""),
                    "answer": p.get("answer") if qtype == "yesno" else None,
                    "score":  p.get("score")  if qtype == "score" else None,
                    "parse_ok": ok_list[j],
                    "error": err,
                    "infer_seconds": round(dt, 2),
                    "bucket_size": n_in_bucket,
                    "bucket_index_in_clip": j,
                    "max_new_tokens_used": mnt,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if bi % 5 == 0 or bi == n_buckets:
                elapsed = time.time() - t_start
                rate = bi / max(elapsed, 1e-6)
                eta = (n_buckets - bi) / max(rate, 1e-6)
                ok_in_bucket = sum(1 for o in ok_list if o)
                print(
                    f"[job={tag}] [bucket {bi}/{n_buckets}] {sid} shot{sidx} N={n_in_bucket} "
                    f"ok={ok_in_bucket}/{n_in_bucket} dt={dt:.1f}s mnt={mnt} "
                    f"| elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )

    summary = {
        "tag": tag,
        "model_path": getattr(runner, "model_path", ""),
        "manifest": manifest_path,
        "n_specs_in_manifest": len(manifest_raw),
        "n_specs_passed": passed_count,
        "n_buckets": n_buckets,
        "n_questions": n_questions_total,
        "n_yesno_questions": n_yesno,
        "n_score_questions": n_score,
        "n_parse_ok": n_parse_ok,
        "n_parse_fail": n_parse_fail,
        "score_aggregation": None,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print(f"[batch_eval] job={tag} DONE in {time.time()-t_start:.0f}s")
    print(f"  per_question : {out_path}")
    print(f"  summary      : {summary_path}")
    print(f"  buckets={n_buckets}  questions={n_questions_total}  "
          f"parse_ok={n_parse_ok}/{n_questions_total}  parse_fail={n_parse_fail}")


def _load_jobs_tsv(path: str) -> List[Dict[str, str]]:
    """读全模型作业表. 每行: tag\tmanifest_path\tout_dir"""
    jobs: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) < 3:
                print(f"[WARN] jobs tsv:{line_no} bad row (need 3 cols), skip: {raw}",
                      file=sys.stderr)
                continue
            tag, manifest_path, out_dir = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not tag or not manifest_path or not out_dir:
                print(f"[WARN] jobs tsv:{line_no} empty field, skip: {raw}", file=sys.stderr)
                continue
            jobs.append({"tag": tag, "manifest": manifest_path, "out_dir": out_dir})
    return jobs


def run(args):
    sys_prompt = load_system_prompt(args.system_prompt)
    specs = load_specs(args.specs_dir)
    print(f"[batch_eval] specs loaded      : {len(specs)}")
    print(f"[batch_eval] system_prompt len : {len(sys_prompt)} chars", flush=True)

    # 收集作业列表 (一个 job = 一个视频模型的 manifest + out_dir)
    if args.jobs_tsv:
        jobs = _load_jobs_tsv(args.jobs_tsv)
        if not jobs:
            print(f"[ERROR] no valid jobs in {args.jobs_tsv}", file=sys.stderr)
            sys.exit(2)
        print(f"[batch_eval] multi-job mode: {len(jobs)} jobs from {args.jobs_tsv}")
        for j in jobs:
            print(f"  - {j['tag']:<24} manifest={j['manifest']}  out={j['out_dir']}")
    else:
        # 单 manifest 后兼容模式
        if not (args.manifest and args.out_dir):
            print("[ERROR] either --jobs-tsv OR (--manifest + --out-dir) is required",
                  file=sys.stderr)
            sys.exit(2)
        jobs = [{
            "tag": args.tag or os.path.basename(args.out_dir.rstrip("/")) or "model",
            "manifest": args.manifest,
            "out_dir": args.out_dir,
        }]
        print(f"[batch_eval] single-job mode: tag={jobs[0]['tag']}")

    # ---- 加载模型 (只加载 1 次!) ----
    if args.backend == "api":
        from qwen35_api import OmniApiRunner  # noqa: WPS433
        runner = OmniApiRunner(
            model_path=args.model_path or None,
            attn_impl=args.attn_impl,
            use_audio_in_video=True,
            device=args.device,
        )
        if args.model_path:
            runner.model_path = args.model_path
    else:
        runner = OmniRunner(args.model_path, attn_impl=args.attn_impl,
                            use_audio_in_video=True, device=args.device)
        runner.model_path = args.model_path  # 供 summary 记录使用

    # ---- 按顺序跑每个 job ----
    t_all = time.time()
    for j in jobs:
        _run_one_job(
            runner,
            tag=j["tag"],
            manifest_path=j["manifest"],
            questions_path=args.questions,
            specs=specs,
            sys_prompt=sys_prompt,
            out_dir=j["out_dir"],
            shard_idx=args.shard_idx,
            shard_num=args.shard_num,
            max_new_tokens_base=args.max_new_tokens,
            limit=args.limit,
        )
    print()
    print(f"[batch_eval] ALL JOBS DONE in {time.time()-t_all:.0f}s  (jobs={len(jobs)})")


def main():
    ap = argparse.ArgumentParser()
    # 单 job 模式 参数 (后兼容, 跟 jobs-tsv 二选一)
    ap.add_argument("--manifest", default="",
                    help="单 job 模式: shotcut_manifest.jsonl 路径")
    ap.add_argument("--out-dir", default="",
                    help="单 job 模式: 该模型的 runs/<M>/ 输出目录")
    ap.add_argument("--tag", default="",
                    help="单 job 模式: 模型名 (仅打印用)")
    # 多 job 模式: 一次加载 Qwen, 跑多个 (manifest, out_dir)
    ap.add_argument("--jobs-tsv", default="",
                    help="多 job 模式: TSV, 每行 'tag\\tmanifest_path\\tout_dir', "
                         "与 --manifest/--out-dir/--tag 互斥")
    # 公共参数
    ap.add_argument("--questions", required=True)
    ap.add_argument("--specs-dir", required=True)
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--model-path", required=False, default="",
                    help="local backend: Qwen3-Omni 权重目录 (必填); api backend: 可选, 仅用于 summary 记录")
    ap.add_argument("--backend", choices=["local", "api"], default="local",
                    help="local = 本地加载 Qwen3-Omni 30B 权重; api = 走 qwen35_api/OmniApiRunner (qwen3.5-omni-plus 平台)")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="thinker_max_new_tokens 基础值 (默认 512, Instruct 直接吐 JSON 无需思考额度). "
                         "实际预算 = min(4096, base + 256*N), N 为该 clip 的题数.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="DP shard index in [0, shard_num)")
    ap.add_argument("--shard-num", type=int, default=1,
                    help="total number of DP shards (1 = no sharding)")
    ap.add_argument("--device", type=str, default="auto",
                    help="'auto' (device_map=auto) or 'cuda:0' / 'cuda:1' for single-GPU DP")
    args = ap.parse_args()
    if args.shard_num <= 0:
        ap.error("--shard-num must be >= 1")
    if not (0 <= args.shard_idx < args.shard_num):
        ap.error(f"--shard-idx must be in [0, {args.shard_num}), got {args.shard_idx}")
    if args.jobs_tsv and (args.manifest or args.out_dir):
        ap.error("--jobs-tsv and (--manifest/--out-dir) are mutually exclusive")
    if args.backend == "local" and not args.model_path:
        ap.error("--model-path is required when --backend=local")
    run(args)


if __name__ == "__main__":
    main()
