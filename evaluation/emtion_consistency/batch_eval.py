#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emtion_consistency 评测点 · 单模型 Qwen3-Omni-Instruct 评测脚本 (grouped 模式)

依赖前置:
  上一步 prepare_manifest.py 已经为该模型生成了:
      <out_dir>/manifest.jsonl
  每行格式:
      {"spec_id":"speech_0001","category":"speech","passed":true,
       "video_path":"/.../speech/speech_0001.mp4","note":"ok"}

输入:
  --manifest        manifest.jsonl 路径
  --questions       t2av-bench/questions/emotion_consistency.jsonl
                    每行: {"spec_id":..., "question":..., "associated_subject_id":...}
                    本评测无 shot_idx, 同一 spec 下的 N 题按文件内顺序聚成 1 个 bucket
  --system-prompt   evaluation/emtion_consistency/system_prompt.md
  --model-path      Qwen3-Omni-30B-A3B-Instruct 权重目录
  --out-dir         本模型 runs 目录, 与 manifest 同根
  --tag             模型名 (打印用)
  --max-new-tokens  thinker_max_new_tokens 基础值 (默认 512)
                    每个 bucket 实际预算: min(4096, base + 256*N)
  --limit           可选: 只跑前 N 个 spec, 调试用

题型:
  情绪一致性只有 yes/no 题, 无 score 题. system_prompt.md 里区分了
  Type A (语音情绪) 与 Type B (画面情绪 / 无人可见), 但题面上都按 yesno 解析.

Grouped 模式:
  同一个 spec_id 下的所有问题 (恒为 2 题, 但脚本不强制) 合并到 *一条* user
  消息发给模型, 视频只编码 1 次. 模型按指令返回长度=N 的 JSON 数组,
  每个元素与 questions 列表逐位对齐. 解析失败的元素 fallback 为 "no".

输出:
  <out-dir>/per_question.jsonl  每条 question 一行 (同 bucket 共享 raw_response)
  <out-dir>/summary.json        统计计数
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

# split-modality 模式使用: 题面分类 + wav 抽取
from question_modality import classify as classify_modality  # noqa: E402
from audio_extract import ensure_wav  # noqa: E402


# ============================================================
# 加载: questions.jsonl / system_prompt / manifest
# ============================================================
def load_questions(path: str) -> Dict[str, List[dict]]:
    """
    返回 {spec_id: [question_obj, ...]}, 保留 jsonl 内顺序.
    本评测的 questions.jsonl 没有 shot_idx 字段.
    """
    out: Dict[str, List[dict]] = defaultdict(list)
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
            if not sid:
                continue
            out[sid].append(obj)
    return out


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
        # 仅做文本评测, 关闭 talker 以避免 transformers Code2WavConfig 加载问题
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

    def _build_messages(self, system_text: str, video_path: str, user_text: str,
                        modality: str = "both", audio_path: str = "") -> List[dict]:
        """根据 modality 构造多模态消息.

        modality:
          - 'both'  (默认, 旧行为): user content = [video(含音), text]
          - 'audio' (split-modality A 组): user content = [audio(wav), text]
          - 'video' (split-modality B 组): user content = [video(纯画面), text]

        modality=='audio' 时必须提供 audio_path (wav).
        """
        if modality == "audio":
            if not audio_path:
                raise ValueError("modality='audio' requires audio_path (wav)")
            user_content = [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": user_text},
            ]
        else:
            # both / video 都以 video 为主, 区别仅在 use_audio_in_video 开关
            user_content = [
                {"type": "video", "video": video_path},
                {"type": "text", "text": user_text},
            ]
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def generate(self, system_text: str, video_path: str, user_text: str,
                 max_new_tokens: int = 512,
                 modality: str = "both", audio_path: str = "") -> str:
        """多模态推理.

        modality:
          - 'both'  : 视频+音轨一起送 (旧行为, use_audio_in_video=True)
          - 'audio' : 只送 wav; 视频不进模型; use_audio_in_video=False
          - 'video' : 送视频但去掉音轨; use_audio_in_video=False
        """
        torch = self.torch

        # split-modality 时强制关闭 use_audio_in_video, 保证 video-only 是真的
        # 纯画面; audio-only 时视频根本不在 messages 里, 这个开关也不会影响它.
        if modality == "both":
            uaiv = self.use_audio_in_video
        else:
            uaiv = False

        messages = self._build_messages(system_text, video_path, user_text,
                                        modality=modality, audio_path=audio_path)

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = self.process_mm_info(
            messages, use_audio_in_video=uaiv
        )
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
            use_audio_in_video=uaiv,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with torch.inference_mode():
            text_ids, _audio_out = self.model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=max_new_tokens,
                thinker_do_sample=False,
                use_audio_in_video=uaiv,
                return_audio=False,
            )
        response = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response


# ============================================================
# Grouped prompt 构造 / 解析
# ============================================================
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_JSON_ARR_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,\s*)*\{.*?\}\s*\]", re.DOTALL)


def build_grouped_user_text(qs: List[dict]) -> str:
    """把同一个 spec_id 下的 N 个 yes/no question 拼成一条 user 消息.

    全部题型为 yesno, 不再区分 score.
    """
    n = len(qs)
    lines: List[str] = []
    lines.append(f"You will answer the following {n} yes/no question(s) about the SAME video clip above.")
    lines.append("Each question is independent. For each one, return an item of the form")
    lines.append('  {"reason": "<short>", "answer": "yes"|"no"}')
    lines.append("following the system rules (Type A = audio-only vocal emotion; Type B = visual-only).")
    lines.append("")
    lines.append("Questions:")
    for i, q in enumerate(qs, start=1):
        lines.append(f"  {i}. {q.get('question', '').strip()}")
    lines.append("")
    if n == 1:
        lines.append('Return EXACTLY ONE JSON OBJECT for the single question above.')
        lines.append("No prose, no markdown fences, no extra keys. Just the JSON object.")
    else:
        lines.append(
            f"Return EXACTLY ONE JSON ARRAY of length {n}, where the i-th element is the answer to the "
            "i-th question above, in the same order. No prose, no markdown fences, no extra keys. Just the JSON array."
        )
    return "\n".join(lines)


def _coerce_yesno(elem: Any) -> Tuple[Optional[dict], bool]:
    if not isinstance(elem, dict):
        return None, False
    ans = elem.get("answer")
    if isinstance(ans, str) and ans.strip().lower() in ("yes", "no"):
        return {"reason": str(elem.get("reason", "")),
                "answer": ans.strip().lower()}, True
    return None, False


def parse_grouped_response(raw: str, n: int) -> Tuple[List[Optional[dict]], List[bool]]:
    """
    解析模型对 N 题 (yes/no) 的输出.
    n == 1 时, 接受单个 JSON 对象; n >= 2 时, 期望一个长度=N 的 JSON 数组.
    返回 (parsed_list, ok_list), 长度均为 n.
    任意位置解析失败该位置 parsed=None, ok=False.
    """
    parsed: List[Optional[dict]] = [None] * n
    ok: List[bool] = [False] * n
    if not raw or n <= 0:
        return parsed, ok

    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("json").lstrip("JSON").strip()

    # ---- n == 1: 直接当对象解析 ----
    if n == 1:
        cand_objs: List[str] = [s]
        ms = _JSON_OBJ_RE.findall(raw)
        if ms:
            cand_objs.append(ms[-1])
        for cand in cand_objs:
            try:
                obj = json.loads(cand)
            except Exception:
                continue
            if isinstance(obj, dict):
                p, k = _coerce_yesno(obj)
                if k:
                    parsed[0] = p
                    ok[0] = True
                    return parsed, ok
            elif isinstance(obj, list) and obj:
                # 模型偶尔单题也吐数组, 取第一个
                p, k = _coerce_yesno(obj[0])
                if k:
                    parsed[0] = p
                    ok[0] = True
                    return parsed, ok
        return parsed, ok

    # ---- n >= 2: 期望数组 ----
    arr: Optional[list] = None

    cand_arr_strs: List[str] = [s]
    arr_ms = _JSON_ARR_RE.findall(raw)
    if arr_ms:
        cand_arr_strs.append(arr_ms[-1])

    for cand in cand_arr_strs:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, list):
            arr = obj
            break

    # Fallback A: 找不到合法 array, 但能找到 N 个独立 JSON 对象
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

    # Fallback B: 扫描最后一个 '[' 之后, 按花括号深度收集已完整闭合的 {...}
    if arr is None:
        lb = raw.rfind("[")
        if lb >= 0:
            tail = raw[lb + 1:]
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
                                d = json.loads(tail[start: k + 1])
                                if isinstance(d, dict):
                                    collected.append(d)
                            except Exception:
                                pass
                            start = -1
            if collected:
                arr = collected[:n]

    if arr is None:
        return parsed, ok

    for i in range(min(n, len(arr))):
        p, k = _coerce_yesno(arr[i])
        if k:
            parsed[i] = p
            ok[i] = True
    return parsed, ok


# ============================================================
# 主流程
# ============================================================
def _build_todo(manifest_path: str, questions_path: str, limit: int = 0
                ) -> Tuple[List[Tuple[str, str, List[dict]]], List[dict]]:
    """处理单个 manifest 路径, 返回 (todo, manifest_raw).

    todo = [(spec_id, video_path, [qobj,...]), ...]   (仅含 passed 且有 video / question 的 spec)
    manifest_raw = 原 manifest 记录, 供 summary 记录 n_specs_in_manifest 使用
    """
    questions = load_questions(questions_path)
    manifest = load_manifest(manifest_path)

    passed_specs = [m for m in manifest if m.get("passed")]
    if limit and limit > 0:
        passed_specs = passed_specs[: limit]

    todo: List[Tuple[str, str, List[dict]]] = []
    for m in passed_specs:
        sid = m["spec_id"]
        video_path = m.get("video_path") or ""
        if not video_path:
            print(f"[WARN] {sid} passed but missing video_path, skip", file=sys.stderr)
            continue
        qs = questions.get(sid, [])
        if not qs:
            print(f"[WARN] {sid}: no questions in {questions_path}, skip", file=sys.stderr)
            continue
        todo.append((sid, video_path, list(qs)))
    return todo, manifest


def _run_one_job(
    runner: "OmniRunner",
    *,
    tag: str,
    manifest_path: str,
    questions_path: str,
    sys_prompt: str,
    out_dir: str,
    shard_idx: int,
    shard_num: int,
    max_new_tokens_base: int,
    limit: int,
    split_modality: bool = False,
    wav_cache_dir: str = "",
) -> None:
    """在已加载好的 runner 上, 跑一个 (manifest, out_dir) 作业.

    split_modality=False (默认): 旧行为, 同 spec 的所有题合一次往模型送,
        视频 + 音轨一起编码 (use_audio_in_video=True).

    split_modality=True: 同 spec 的题按题干分为 audio-only / video-only
        两组, 各跑一次 generate; audio 组用 ffmpeg 抽出的 wav, video 组
        use_audio_in_video=False. 需提供 wav_cache_dir.
    """
    print()
    print(f"############ [batch_eval] job tag={tag} (shard {shard_idx}/{shard_num}) ############")
    print(f"  manifest = {manifest_path}")
    print(f"  out_dir  = {out_dir}")
    print(f"  questions= {questions_path}", flush=True)

    todo, manifest_raw = _build_todo(manifest_path, questions_path, limit=limit)

    n_total_buckets_global = len(todo)
    n_total_questions_global = sum(len(t[2]) for t in todo)
    print(f"[batch_eval] passed specs: {len(manifest_raw)} -> todo {n_total_buckets_global} buckets")
    print(f"[batch_eval] total buckets (global)  : {n_total_buckets_global}")
    print(f"[batch_eval] total questions (global): {n_total_questions_global}", flush=True)

    sharded = (shard_num > 1)
    if sharded:
        todo = [t for i, t in enumerate(todo) if i % shard_num == shard_idx]
        kept_q = sum(len(t[2]) for t in todo)
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
    n_questions_total = sum(len(t[2]) for t in todo)

    if n_buckets == 0:
        # 该 job 在本 shard 上无任务; 仍然写出空文件 + summary, 避免下游 merge 报错.
        open(out_path, "w", encoding="utf-8").close()
        summary = {
            "tag": tag,
            "model_path": getattr(runner, "model_path", ""),
            "manifest": manifest_path,
            "n_specs_in_manifest": len(manifest_raw),
            "n_specs_passed": 0,
            "n_buckets": 0,
            "n_questions": 0,
            "n_yesno_questions": 0,
            "n_parse_ok": 0,
            "n_parse_fail": 0,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[batch_eval] (job={tag}) nothing to run on this shard, wrote empty {out_path}")
        return

    n_parse_ok = 0
    n_parse_fail = 0
    t_start = time.time()

    if split_modality:
        if not wav_cache_dir:
            wav_cache_dir = os.path.join(out_dir, "audio_wav")
        os.makedirs(wav_cache_dir, exist_ok=True)
        print(f"[batch_eval] split-modality ON, wav cache = {wav_cache_dir}", flush=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for bi, (sid, video_path, qs) in enumerate(todo, start=1):
            n_in_bucket = len(qs)

            if not split_modality:
                # ---------- 旧行为: 一次 generate, 多模态混合送 ----------
                user_text = build_grouped_user_text(qs)
                mnt = min(4096, int(max_new_tokens_base) + 256 * n_in_bucket)

                t0 = time.time()
                try:
                    raw = runner.generate(
                        system_text=sys_prompt,
                        video_path=video_path,
                        user_text=user_text,
                        max_new_tokens=mnt,
                    )
                    err = ""
                except Exception as e:
                    raw = ""
                    err = repr(e)
                dt = time.time() - t0

                parsed_list, ok_list = parse_grouped_response(raw, n_in_bucket)
                for j in range(n_in_bucket):
                    if not ok_list[j]:
                        parsed_list[j] = {"reason": "[parse_fail fallback]", "answer": "no"}
                        n_parse_fail += 1
                    else:
                        n_parse_ok += 1

                for j, qobj in enumerate(qs):
                    q = qobj.get("question", "").strip()
                    p = parsed_list[j] or {}
                    rec = {
                        "spec_id": sid,
                        "category": "speech",
                        "video_path": video_path,
                        "question": q,
                        "question_type": "yesno",
                        "associated_subject_id": qobj.get("associated_subject_id"),
                        "raw_response": raw,
                        "reason": p.get("reason", ""),
                        "answer": p.get("answer"),
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
                        f"[job={tag}] [bucket {bi}/{n_buckets}] {sid} N={n_in_bucket} "
                        f"ok={ok_in_bucket}/{n_in_bucket} dt={dt:.1f}s mnt={mnt} "
                        f"| elapsed={elapsed:.0f}s eta={eta:.0f}s",
                        flush=True,
                    )
                continue

            # ---------- split-modality: 拆成 audio_qs + video_qs, 各跑一次 ----------
            mods = [classify_modality(q.get("question", "")) for q in qs]
            audio_idx = [i for i, m in enumerate(mods) if m == "audio"]
            video_idx = [i for i, m in enumerate(mods) if m == "video"]

            # 预先抽 wav (仅当本 spec 存在 audio 题时)
            audio_path = ""
            wav_err = ""
            if audio_idx:
                try:
                    audio_path = ensure_wav(video_path, wav_cache_dir)
                except Exception as e:
                    wav_err = repr(e)
                    print(f"[job={tag}] {sid} ensure_wav FAILED: {wav_err}", file=sys.stderr, flush=True)

            # 两个子题组的结果: (parsed, ok, raw, err, dt, mnt) 按 orig index 无射
            per_q_parsed: List[Optional[dict]] = [None] * n_in_bucket
            per_q_ok: List[bool] = [False] * n_in_bucket
            per_q_raw: List[str] = [""] * n_in_bucket
            per_q_err: List[str] = [""] * n_in_bucket
            per_q_dt: List[float] = [0.0] * n_in_bucket
            per_q_mnt: List[int] = [0] * n_in_bucket
            per_q_mod: List[str] = list(mods)

            def _run_subgroup(sub_idx: List[int], modality: str):
                if not sub_idx:
                    return
                sub_qs = [qs[i] for i in sub_idx]
                sub_n = len(sub_qs)
                sub_user_text = build_grouped_user_text(sub_qs)
                sub_mnt = min(4096, int(max_new_tokens_base) + 256 * sub_n)

                sub_raw = ""
                sub_err = ""
                if modality == "audio" and not audio_path:
                    # wav 抽取失败, 本组全部标为失败
                    sub_err = wav_err or "no audio_path"
                    sub_dt = 0.0
                else:
                    t0 = time.time()
                    try:
                        sub_raw = runner.generate(
                            system_text=sys_prompt,
                            video_path=video_path,
                            user_text=sub_user_text,
                            max_new_tokens=sub_mnt,
                            modality=modality,
                            audio_path=audio_path if modality == "audio" else "",
                        )
                    except Exception as e:
                        sub_raw = ""
                        sub_err = repr(e)
                    sub_dt = time.time() - t0

                sub_parsed, sub_ok = parse_grouped_response(sub_raw, sub_n)
                for k, orig_i in enumerate(sub_idx):
                    per_q_parsed[orig_i] = sub_parsed[k]
                    per_q_ok[orig_i] = sub_ok[k]
                    per_q_raw[orig_i] = sub_raw
                    per_q_err[orig_i] = sub_err
                    per_q_dt[orig_i] = sub_dt
                    per_q_mnt[orig_i] = sub_mnt

            _run_subgroup(audio_idx, "audio")
            _run_subgroup(video_idx, "video")

            for j in range(n_in_bucket):
                if not per_q_ok[j]:
                    per_q_parsed[j] = {"reason": "[parse_fail fallback]", "answer": "no"}
                    n_parse_fail += 1
                else:
                    n_parse_ok += 1

            for j, qobj in enumerate(qs):
                q = qobj.get("question", "").strip()
                p = per_q_parsed[j] or {}
                rec = {
                    "spec_id": sid,
                    "category": "speech",
                    "video_path": video_path,
                    "question": q,
                    "question_type": "yesno",
                    "associated_subject_id": qobj.get("associated_subject_id"),
                    "raw_response": per_q_raw[j],
                    "reason": p.get("reason", ""),
                    "answer": p.get("answer"),
                    "parse_ok": per_q_ok[j],
                    "error": per_q_err[j],
                    "infer_seconds": round(per_q_dt[j], 2),
                    "bucket_size": n_in_bucket,
                    "bucket_index_in_clip": j,
                    "max_new_tokens_used": per_q_mnt[j],
                    "modality": per_q_mod[j],
                    "audio_path": audio_path if per_q_mod[j] == "audio" else "",
                    "split_modality": True,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if bi % 5 == 0 or bi == n_buckets:
                elapsed = time.time() - t_start
                rate = bi / max(elapsed, 1e-6)
                eta = (n_buckets - bi) / max(rate, 1e-6)
                ok_in_bucket = sum(1 for o in per_q_ok if o)
                total_dt = sum(per_q_dt)
                print(
                    f"[job={tag}] [bucket {bi}/{n_buckets}] {sid} N={n_in_bucket} "
                    f"(A={len(audio_idx)},V={len(video_idx)}) ok={ok_in_bucket}/{n_in_bucket} "
                    f"dt={total_dt:.1f}s | elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )

    summary = {
        "tag": tag,
        "model_path": getattr(runner, "model_path", ""),
        "manifest": manifest_path,
        "n_specs_in_manifest": len(manifest_raw),
        "n_specs_passed": len([m for m in manifest_raw if m.get("passed")]),
        "n_buckets": n_buckets,
        "n_questions": n_questions_total,
        "n_yesno_questions": n_questions_total,  # 恒为 yesno
        "n_parse_ok": n_parse_ok,
        "n_parse_fail": n_parse_fail,
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

    print(f"[batch_eval] system_prompt len : {len(sys_prompt)} chars", flush=True)

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
            sys_prompt=sys_prompt,
            out_dir=j["out_dir"],
            shard_idx=args.shard_idx,
            shard_num=args.shard_num,
            max_new_tokens_base=args.max_new_tokens,
            limit=args.limit,
            split_modality=bool(args.split_modality),
            wav_cache_dir=args.wav_cache_dir,
        )
    print()
    print(f"[batch_eval] ALL JOBS DONE in {time.time()-t_all:.0f}s  (jobs={len(jobs)})")


def main():
    ap = argparse.ArgumentParser()
    # 单 job 模式 参数 (后兼容, 跟 jobs-tsv 二选一)
    ap.add_argument("--manifest", default="",
                    help="单 job 模式: manifest.jsonl 路径")
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
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--model-path", required=False, default="",
                    help="local backend: Qwen3-Omni 权重目录 (必填); api backend: 可选, 仅用于 summary 记录")
    ap.add_argument("--backend", choices=["local", "api"], default="local",
                    help="local = 本地加载 Qwen3-Omni 30B 权重; api = 走 qwen35_api/OmniApiRunner (qwen3.5-omni-plus 平台)")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="thinker_max_new_tokens 基础值; 实际 = min(4096, base + 256*N)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="DP shard index in [0, shard_num)")
    ap.add_argument("--shard-num", type=int, default=1,
                    help="total number of DP shards (1 = no sharding)")
    ap.add_argument("--split-modality", action="store_true",
                    help="开启 split-modality 模式: audio-only 题只送 wav, video-only 题只送去音视频.")
    ap.add_argument("--wav-cache-dir", default="",
                    help="split-modality 下抽 wav 的缓存目录; 为空时默认 <out_dir>/audio_wav/")
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
