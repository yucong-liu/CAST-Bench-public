#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
action_eval 评测点 · 视频运动质量评测脚本 (yes/no 单题推理版 + 多 judge backend)

依赖前置 (由 run_all.sh 阶段 1 完成):
  <out_dir>/shotcut_manifest.jsonl  (软链接到 evaluation/audio_consistency/runs/<sub>/<model>/)
  <out_dir>/clips/                  (同上, 软链接; multi_shot 切片产物)
  single_shot 的 clip_path 直接指向原 mp4.

输入:
  --manifest        shotcut_manifest.jsonl 路径
  --questions       t2av-bench/questions/action_eval_llm.jsonl  (yes/no 版, 由
                    t2av-bench/scripts/15_build_action_eval_questions_llm.py 生成)
                    每行: {"spec_id":..., "shot_idx":..., "question":...,
                            "question_aspect": subject_deformation | physics_plausibility |
                                               action_semantic_plausibility |
                                               motion_direction_plausibility,
                            "question_type":"yesno",
                            "prompt_text": "...", "associated_subject_id": null}
                    (v2 已删除 overall aspect: 出题脚本改为对描述里点名的
                     每一个动作各出 1 道 yes/no 合理性题, 不再追加 overall
                     兜底题; audio_situation=off_screen 的 spec 整条跳过.)
                    本评测把 (spec_id, shot_idx) 下的所有题聚成 1 个 bucket,
                    **每道题各自一次 VLM 推理** (单题 yes/no), 因为 grouped
                    prompt 在 Qwen3-VL 上稳定性不如单题.
  --system-prompt   evaluation/action_eval/system_prompt.md
                    (通用 yes/no rubric, 只判 motion)
  --out-dir         本 (judge, video_model) 组合的 runs 目录

  --judge-backend   "qwen_vl"
                    qwen_vl  -> 本地 vLLM OpenAI 兼容服务 (Qwen3-VL-235B 等)
  --judge-base-url  qwen_vl: 默认 http://127.0.0.1:8080/v1
  --judge-model     qwen_vl: served-model-name (默认 Qwen3-VL-235B-A22B-Instruct)
  --judge-api-key   qwen_vl: vLLM 不校验, 默认 EMPTY

注: action_eval 不使用 omni (本地 transformers Qwen3-Omni); 完全依赖外部 VLM 服务/API.
    yes/no 版下**不再** inject prompt 给 VLM (题面本身已包含该 shot 的具体动作描述),
    prompt_text 只作为 per_question.jsonl 里的复查字段保留。

输出:
  <out-dir>/per_question.jsonl   每条 question 一行, 字段包含 answer (yes/no), parse_ok, question_aspect
  <out-dir>/summary.json         统计计数
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# IO helpers
# ============================================================
def load_questions(path: str) -> Dict[Tuple[str, int], List[dict]]:
    """{(spec_id, shot_idx): [question_obj, ...]}, 保留 jsonl 顺序."""
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
    """action_eval 只需要 spec 的 category, 用于 per_question 写回."""
    out: Dict[str, dict] = {}
    if not specs_dir or not os.path.isdir(specs_dir):
        return out
    for name in sorted(os.listdir(specs_dir)):
        if not name.endswith(".jsonl"):
            continue
        cat = name[: -len(".jsonl")]
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
# Judge backend 抽象
# ============================================================
class JudgeBackend:
    """所有 judge backend 的统一接口."""

    name: str = "base"

    def generate(self, system_text: str, video_path: str, user_text: str,
                 max_new_tokens: int = 512) -> str:
        raise NotImplementedError


def _video_to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "webm": "video/webm",
        "mkv": "video/x-matroska",
    }.get(ext, "video/mp4")
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


class QwenVLBackend(JudgeBackend):
    """通过 OpenAI 兼容 API 调用 vLLM 起的 Qwen3-VL (或同款) 模型.

    与 t2av-bench/SETUP.md 描述的 vLLM 服务 (默认 127.0.0.1:8080) 对接,
    模型名默认为 Qwen3-VL-235B-A22B-Instruct.
    """

    name = "qwen_vl"

    def __init__(self,
                 base_url: str = "http://127.0.0.1:8080/v1",
                 model: str = "Qwen3-VL-235B-A22B-Instruct",
                 api_key: str = "EMPTY",
                 timeout_sec: int = 600,
                 temperature: float = 0.0,
                 top_p: float = 0.9,
                 max_retry: int = 3,
                 video_as_data_url: bool = True,
                 video_fps: float = 12.0,
                 video_max_frames: int = 96):
        try:
            from openai import OpenAI  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError("qwen_vl backend 需要 openai 库: pip install openai") from e
        self.client = OpenAI(api_key=api_key or "EMPTY",
                             base_url=base_url,
                             timeout=timeout_sec)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_retry = max_retry
        self.video_as_data_url = video_as_data_url
        # action_eval 视频抽帧策略: mm_processor_kwargs 的 key 名在 vLLM 不同版本里
        # 不完全兼容, 起手 num_frames, 失败按 num_frames -> nframes -> max_frames -> fps_only 降级.
        self.video_fps = float(video_fps)
        self.video_max_frames = int(video_max_frames)
        self._mm_kwargs_strategy = "num_frames" if self.video_max_frames > 0 else "fps_only"
        if self.video_max_frames > 0:
            _strategy_desc = (f"num_frames={self.video_max_frames} "
                              f"(fallback fps={self.video_fps} if processor rejects num_frames)")
        else:
            _strategy_desc = f"fps={self.video_fps} (no num_frames cap)"
        print(f"[judge=qwen_vl] base_url={base_url} model={model} "
              f"timeout={timeout_sec}s video_as_data_url={video_as_data_url} "
              f"video_strategy={_strategy_desc}",
              flush=True)

    def _build_messages(self, system_text: str, video_path: str, user_text: str) -> List[dict]:
        if self.video_as_data_url:
            video_url = _video_to_data_url(video_path)
        else:
            video_url = f"file://{os.path.abspath(video_path)}"
        return [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def _build_mm_kwargs(self) -> Dict[str, Any]:
        s = self._mm_kwargs_strategy
        n = self.video_max_frames
        if s == "num_frames" and n > 0:
            return {"num_frames": n}
        if s == "nframes" and n > 0:
            return {"nframes": n}
        if s == "max_frames":
            kw: Dict[str, Any] = {"fps": self.video_fps}
            if n > 0:
                kw["max_frames"] = n
            return kw
        # fps_only
        return {"fps": self.video_fps}

    def _downgrade_strategy(self, reason: str) -> bool:
        order = ["num_frames", "nframes", "max_frames", "fps_only"]
        try:
            idx = order.index(self._mm_kwargs_strategy)
        except ValueError:
            return False
        if idx + 1 >= len(order):
            return False
        old = self._mm_kwargs_strategy
        self._mm_kwargs_strategy = order[idx + 1]
        print(f"[judge=qwen_vl][WARN] mm_processor_kwargs strategy '{old}' rejected "
              f"({reason}); downgrading to '{self._mm_kwargs_strategy}'.",
              file=sys.stderr)
        return True

    def generate(self, system_text: str, video_path: str, user_text: str,
                 max_new_tokens: int = 512) -> str:
        messages = self._build_messages(system_text, video_path, user_text)
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retry + 1):
            try:
                mm_kwargs = self._build_mm_kwargs()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=max_new_tokens,
                    extra_body={
                        "mm_processor_kwargs": mm_kwargs,
                    },
                )
                try:
                    _u = getattr(resp, "usage", None)
                    _pt = getattr(_u, "prompt_tokens", None) if _u else None
                    _ct = getattr(_u, "completion_tokens", None) if _u else None
                    _vname = os.path.basename(video_path) if video_path else "?"
                    print(f"[judge=qwen_vl][TOK] video={_vname} "
                          f"mm_kwargs={mm_kwargs} "
                          f"prompt_tokens={_pt} completion_tokens={_ct}",
                          file=sys.stderr, flush=True)
                except Exception:
                    pass
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                key_rejected = (
                    "unexpected keyword argument" in msg
                    or "got an unexpected keyword" in msg
                )
                if key_rejected and self._downgrade_strategy(repr(e)):
                    continue
                wait = 2 ** attempt
                print(f"[judge=qwen_vl][WARN] attempt {attempt}/{self.max_retry} failed: {e!r}; "
                      f"sleep {wait}s", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"qwen_vl backend all retries failed: {last_err!r}")


def build_judge_backend(args) -> JudgeBackend:
    bk = args.judge_backend.lower()
    if bk == "qwen_vl":
        return QwenVLBackend(
            base_url=args.judge_base_url,
            model=args.judge_model,
            api_key=args.judge_api_key,
            timeout_sec=args.judge_timeout_sec,
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
            max_retry=args.judge_max_retry,
            video_as_data_url=not args.judge_video_local_path,
            video_fps=args.video_fps,
            video_max_frames=args.video_max_frames,
        )
    raise ValueError(f"unknown --judge-backend: {bk}")


# ============================================================
# 单题 yes/no prompt 构造 / 解析
# ============================================================
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def build_user_text_yesno(q: dict) -> str:
    """单题 yes/no user 文本. 只贴题面 + 输出 JSON 协议; 不 inject prompt_text.

    题面本身已由 LLM 出题阶段针对该 shot 定制, 已包含足够动作细节, 无需
    再把 prompt_text 送给 VLM 让它做 prompt-following 判断.
    """
    lines: List[str] = []
    lines.append("Question:")
    lines.append(f"  {q.get('question', '').strip()}")
    lines.append("")
    lines.append("Watch ONLY the on-screen motion. Ignore audio, scenery, clothing, "
                 "lighting, camera moves, and on-screen text.")
    lines.append("")
    lines.append('Return EXACTLY ONE JSON OBJECT for the question above:')
    lines.append('  {"reason": "<short, ≤ 25 words>", "answer": "yes" | "no"}')
    lines.append("No prose, no markdown fences, no extra keys. Just the JSON object.")
    return "\n".join(lines)


def _coerce_yesno(elem: Any) -> Tuple[Optional[dict], bool]:
    """归一化 yes/no 元素. 只接受 answer ∈ {yes, no} (大小写不敏感)."""
    if not isinstance(elem, dict):
        return None, False
    ans = elem.get("answer")
    if not isinstance(ans, str):
        return None, False
    a = ans.strip().lower()
    if a not in ("yes", "no"):
        return None, False
    return {"reason": str(elem.get("reason", "")),
            "answer": a}, True


def parse_yesno_response(raw: str) -> Tuple[Optional[dict], bool]:
    """单题 yes/no 解析.

    - 尝试整段 strip 后 json.loads
    - 剥离 ```json fence
    - 用 _JSON_OBJ_RE 抠出最后一个 {...} 再解析
    """
    if not raw:
        return None, False
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("json").lstrip("JSON").strip()

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
                return p, True
        elif isinstance(obj, list) and obj:
            p, k = _coerce_yesno(obj[0])
            if k:
                return p, True
    return None, False


# ============================================================
# humanoid_integrity 分项 (1-5 分, spec 级, 整段视频)
# ============================================================
HUMANOID_QUESTION_TEXT = (
    "Score the humanoid integrity of every human/humanoid subject visible in "
    "this ENTIRE video on a 1-5 integer scale, following the strict rubric in "
    "the system prompt. Judge facial five-features (unless the character wears "
    "a mask covering the face), limbs (correct count, natural proportions and "
    "joint articulation, no extra/missing/split/fused limbs) and overall body "
    "form (no melting/splitting/fusing/deformation) across the whole clip. If "
    "multiple humans appear, take the worst one. If no human/humanoid subject "
    "is visible at all, return score=null."
)


def build_user_text_humanoid() -> str:
    """整段视频 humanoid_integrity 5 分制题的 user 文本. 不 inject prompt_text."""
    lines: List[str] = []
    lines.append("Task: humanoid_integrity (1-5 strict).")
    lines.append("")
    lines.append(HUMANOID_QUESTION_TEXT)
    lines.append("")
    lines.append('Return EXACTLY ONE JSON OBJECT:')
    lines.append('  {"reason": "<short, \u2264 30 words, cite the worst issue you saw>", '
                 '"score": <1|2|3|4|5|null>}')
    lines.append("No prose, no markdown fences, no extra keys. Just the JSON object.")
    return "\n".join(lines)


def _coerce_score_1to5(elem: Any) -> Tuple[Optional[dict], bool]:
    """归一化 humanoid_integrity 5 分制元素. score \u2208 {1..5, None}."""
    if not isinstance(elem, dict):
        return None, False
    if "score" not in elem:
        return None, False
    sv = elem.get("score")
    if sv is None:
        return {"reason": str(elem.get("reason", "")), "score": None}, True
    if isinstance(sv, bool):
        # True/False 不是合法分数
        return None, False
    if isinstance(sv, (int, float)):
        try:
            si = int(round(float(sv)))
        except Exception:
            return None, False
        if si in (1, 2, 3, 4, 5):
            return {"reason": str(elem.get("reason", "")), "score": si}, True
        return None, False
    if isinstance(sv, str):
        st = sv.strip().lower()
        if st in ("null", "none", "n/a", "na"):
            return {"reason": str(elem.get("reason", "")), "score": None}, True
        try:
            si = int(round(float(st)))
        except Exception:
            return None, False
        if si in (1, 2, 3, 4, 5):
            return {"reason": str(elem.get("reason", "")), "score": si}, True
    return None, False


def parse_score_1to5_response(raw: str) -> Tuple[Optional[dict], bool]:
    """单题 5 分制解析 (humanoid_integrity)."""
    if not raw:
        return None, False
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("json").lstrip("JSON").strip()

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
            p, k = _coerce_score_1to5(obj)
            if k:
                return p, True
        elif isinstance(obj, list) and obj:
            p, k = _coerce_score_1to5(obj[0])
            if k:
                return p, True
    return None, False


def _resolve_full_video_path(m: dict) -> Optional[str]:
    """从 manifest 一条记录里解析'整段原视频路径'.

    优先级:
      1) m['original_video_path']  (推荐, 由 run_all.sh 阶段 1 写入)
      2) m['video_path']            (兼容其它 evaluation 的字段名)
      3) single_shot: 直接用 shots[0].clip_path (那本身就是原视频)
      4) multi_shot: 尝试从 shots[0].clip_path 反推:
            <root>/clips/<spec_id>_shotK.mp4  ->  <root>/<spec_id>.mp4
         若 <root> 里没有 <spec_id>.mp4, 则回落到向上一级的 output/action/<spec_id>.mp4
         等常见 layout; 找不到则返回 None.
    """
    ov = m.get("original_video_path") or m.get("video_path")
    if ov and os.path.isfile(ov):
        return ov

    shots = m.get("shots") or []
    if not shots:
        return None
    kind = (m.get("shot_count_kind") or "").strip()
    first_clip = (shots[0] or {}).get("clip_path") or ""

    if kind == "single_shot":
        if first_clip and os.path.isfile(first_clip):
            return first_clip
        return None

    # multi_shot: 反推
    sid = m.get("spec_id", "")
    if not (first_clip and sid):
        return None
    clip_dir = os.path.dirname(first_clip)                 # .../clips
    parent = os.path.dirname(clip_dir)                     # .../<model_run_dir>
    grand = os.path.dirname(parent)                        # runs/<sub>/<model>/.. 之类

    for candidate in (
        os.path.join(parent, f"{sid}.mp4"),
        os.path.join(parent, "action", f"{sid}.mp4"),
        os.path.join(grand, "action", f"{sid}.mp4"),
        os.path.join(grand, "output", "action", f"{sid}.mp4"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


# ============================================================
# 主流程
# ============================================================
def _build_todo(manifest_path: str, questions_path: str, specs: Dict[str, dict],
                limit: int = 0
                ) -> Tuple[List[Tuple[str, int, str, dict, List[dict]]], List[dict]]:
    """对单个 manifest 收集 todo bucket.

    todo 元素 = (spec_id, shot_idx, clip_path, spec_obj, [qobj,...])
                 仅含 passed 且有 question 的 (spec, shot) 组合.
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
            spec = {"__category": m.get("category", ""), "shot_count": m.get("shot_count_kind", "")}
        for shot in m.get("shots", []):
            sidx = int(shot["shot_idx"])
            clip = shot["clip_path"]
            if not clip or not os.path.isfile(clip):
                print(f"[WARN] {sid} shot{sidx}: clip not found ({clip}), skip", file=sys.stderr)
                continue
            qs = questions.get((sid, sidx), [])
            if not qs:
                continue
            todo.append((sid, sidx, clip, spec, list(qs)))
    return todo, manifest


def _run_one_job(
    backend: JudgeBackend,
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
    sys_prompt_humanoid: Optional[str] = None,
    run_yesno: bool = True,
    run_humanoid: bool = False,
) -> None:
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

    # ----- humanoid: 每个 passed spec 一条 (整段视频, 5 分制) -----
    humanoid_todo: List[Tuple[str, str, dict]] = []  # (spec_id, full_video_path, spec_obj)
    if run_humanoid:
        passed_specs_raw = [m for m in manifest_raw if m.get("passed")]
        if limit and limit > 0:
            passed_specs_raw = passed_specs_raw[: limit]
        for m in passed_specs_raw:
            sid = m.get("spec_id")
            if not sid:
                continue
            spec = specs.get(sid) or {
                "__category": m.get("category", ""),
                "shot_count": m.get("shot_count_kind", ""),
            }
            fv = _resolve_full_video_path(m)
            if not fv:
                print(f"[WARN] humanoid: {sid} full video not resolvable from manifest, skip",
                      file=sys.stderr)
                continue
            humanoid_todo.append((sid, fv, spec))
        if sharded:
            humanoid_todo = [t for i, t in enumerate(humanoid_todo) if i % shard_num == shard_idx]
            print(f"[batch_eval] humanoid shard {shard_idx}/{shard_num} -> "
                  f"keep {len(humanoid_todo)} specs", flush=True)
        else:
            print(f"[batch_eval] humanoid todo: {len(humanoid_todo)} specs", flush=True)

    n_humanoid = len(humanoid_todo)
    n_yesno_total = n_questions_total
    n_questions_total_all = n_yesno_total + n_humanoid

    if n_buckets == 0 and n_humanoid == 0:
        open(out_path, "w", encoding="utf-8").close()
        summary = {
            "tag": tag,
            "judge_backend": backend.name,
            "manifest": manifest_path,
            "n_specs_in_manifest": len(manifest_raw),
            "n_specs_passed": passed_count,
            "n_buckets": 0,
            "n_questions": 0,
            "n_yesno_questions": 0,
            "n_humanoid_questions": 0,
            "n_parse_ok": 0,
            "n_parse_fail": 0,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[batch_eval] (job={tag}) nothing to run on this shard, wrote empty {out_path}")
        return

    n_parse_ok = 0
    n_parse_fail = 0
    n_humanoid_ok = 0
    n_humanoid_fail = 0
    t_start = time.time()

    # yes/no 单题预算: 单题 reason ≤25 词 + JSON 语法, 128 tokens 就够, 但为容错留 512.
    _mnt_single = min(1024, int(max_new_tokens_base))

    # --only-humanoid 时把 yes/no 的 todo 清空, 但保留下面 for 循环结构不变.
    if not run_yesno:
        todo = []
        n_buckets = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for bi, (sid, sidx, clip, spec, qs) in enumerate(todo, start=1):
            n_in_bucket = len(qs)

            per_q_dt: List[float] = []
            per_q_ok: List[bool] = [False] * n_in_bucket
            per_q_parsed: List[Optional[dict]] = [None] * n_in_bucket
            per_q_raw: List[str] = [""] * n_in_bucket
            per_q_err: List[str] = [""] * n_in_bucket

            for j, qobj in enumerate(qs):
                user_text = build_user_text_yesno(qobj)
                t0 = time.time()
                try:
                    raw_j = backend.generate(
                        system_text=sys_prompt,
                        video_path=clip,
                        user_text=user_text,
                        max_new_tokens=_mnt_single,
                    )
                    err_j = ""
                except Exception as e:  # noqa: BLE001
                    raw_j = ""
                    err_j = repr(e)
                dt_j = time.time() - t0
                per_q_dt.append(dt_j)
                per_q_raw[j] = raw_j
                per_q_err[j] = err_j

                parsed_one, ok_one = parse_yesno_response(raw_j)
                per_q_ok[j] = bool(ok_one)
                per_q_parsed[j] = parsed_one
                if per_q_ok[j]:
                    n_parse_ok += 1
                else:
                    # fallback: 保留 raw / parse_ok=False, answer=None (聚合时按 parse_fail 剔除)
                    per_q_parsed[j] = {"reason": "[parse_fail fallback]", "answer": None}
                    n_parse_fail += 1

            for j, qobj in enumerate(qs):
                q = qobj.get("question", "").strip()
                p = per_q_parsed[j] or {}
                rec = {
                    "spec_id": sid,
                    "shot_idx": sidx,
                    "category": spec.get("__category", ""),
                    "shot_count_kind": spec.get("shot_count", ""),
                    "clip_path": clip,
                    "question": q,
                    "question_type": qobj.get("question_type", "yesno"),
                    "question_aspect": qobj.get("question_aspect", ""),
                    "prompt_text": qobj.get("prompt_text", ""),
                    "associated_subject_id": qobj.get("associated_subject_id"),
                    "raw_response": per_q_raw[j],
                    "reason": p.get("reason", ""),
                    "answer": p.get("answer"),
                    "parse_ok": per_q_ok[j],
                    "error": per_q_err[j],
                    "infer_seconds": round(per_q_dt[j], 2),
                    "bucket_size": n_in_bucket,
                    "bucket_index_in_clip": j,
                    "max_new_tokens_used": _mnt_single,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            dt_total = sum(per_q_dt)
            elapsed = time.time() - t_start
            rate = bi / max(elapsed, 1e-6)
            eta = (n_buckets - bi) / max(rate, 1e-6)
            ok_in_bucket = sum(1 for o in per_q_ok if o)
            dt_break = "+".join(f"{x:.1f}" for x in per_q_dt)
            print(
                f"[job={tag}] [bucket {bi}/{n_buckets}] {sid} shot{sidx} N={n_in_bucket} "
                f"ok={ok_in_bucket}/{n_in_bucket} dt={dt_total:.1f}s ({dt_break}) mnt={_mnt_single} "
                f"| elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

        # ----- humanoid_integrity 分项: 整段视频 -> spec 级 1-5 分 -----
        if run_humanoid and humanoid_todo and sys_prompt_humanoid:
            for hi, (sid, full_video, spec) in enumerate(humanoid_todo, start=1):
                user_text = build_user_text_humanoid()
                t0 = time.time()
                try:
                    raw_h = backend.generate(
                        system_text=sys_prompt_humanoid,
                        video_path=full_video,
                        user_text=user_text,
                        max_new_tokens=_mnt_single,
                    )
                    err_h = ""
                except Exception as e:  # noqa: BLE001
                    raw_h = ""
                    err_h = repr(e)
                dt_h = time.time() - t0

                parsed_h, ok_h = parse_score_1to5_response(raw_h)
                if ok_h:
                    n_humanoid_ok += 1
                else:
                    parsed_h = {"reason": "[parse_fail fallback]", "score": None}
                    n_humanoid_fail += 1

                rec = {
                    "spec_id": sid,
                    "shot_idx": 0,  # 0 表示整段视频, 不属于任何单一 shot
                    "category": spec.get("__category", ""),
                    "shot_count_kind": spec.get("shot_count", ""),
                    "clip_path": full_video,
                    "question": HUMANOID_QUESTION_TEXT,
                    "question_type": "score_1to5",
                    "question_aspect": "humanoid_integrity",
                    "prompt_text": "",
                    "associated_subject_id": None,
                    "raw_response": raw_h,
                    "reason": parsed_h.get("reason", ""),
                    "answer": None,           # 保持向后兼容字段
                    "score": parsed_h.get("score"),
                    "parse_ok": bool(ok_h),
                    "error": err_h,
                    "infer_seconds": round(dt_h, 2),
                    "bucket_size": 1,
                    "bucket_index_in_clip": 0,
                    "max_new_tokens_used": _mnt_single,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                elapsed = time.time() - t_start
                sc_str = str(parsed_h.get("score"))
                print(
                    f"[job={tag}] [humanoid {hi}/{n_humanoid}] {sid} score={sc_str} "
                    f"ok={int(bool(ok_h))} dt={dt_h:.1f}s | elapsed={elapsed:.0f}s",
                    flush=True,
                )
        elif run_humanoid and humanoid_todo and not sys_prompt_humanoid:
            print("[ERROR] run_humanoid=True 但 sys_prompt_humanoid 为空, 跳过 humanoid 分项",
                  file=sys.stderr)

    summary = {
        "tag": tag,
        "judge_backend": backend.name,
        "manifest": manifest_path,
        "n_specs_in_manifest": len(manifest_raw),
        "n_specs_passed": passed_count,
        "n_buckets": n_buckets,
        "n_questions": n_questions_total_all,
        "n_yesno_questions": n_yesno_total if run_yesno else 0,
        "n_humanoid_questions": n_humanoid if run_humanoid else 0,
        "n_parse_ok": n_parse_ok + n_humanoid_ok,
        "n_parse_fail": n_parse_fail + n_humanoid_fail,
        "n_humanoid_ok": n_humanoid_ok,
        "n_humanoid_fail": n_humanoid_fail,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print(f"[batch_eval] job={tag} DONE in {time.time()-t_start:.0f}s")
    print(f"  per_question : {out_path}")
    print(f"  summary      : {summary_path}")
    print(f"  buckets={n_buckets}  yesno_questions={n_yesno_total}  "
          f"parse_ok={n_parse_ok}/{n_yesno_total}  parse_fail={n_parse_fail}")
    if run_humanoid:
        print(f"  humanoid_specs={n_humanoid}  humanoid_ok={n_humanoid_ok}/{n_humanoid}  "
              f"humanoid_fail={n_humanoid_fail}")


def _load_jobs_tsv(path: str) -> List[Dict[str, str]]:
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
    specs = load_specs(args.specs_dir) if args.specs_dir else {}

    # humanoid_integrity 分项: 独立 system prompt (5 分制严格 rubric)
    sys_prompt_humanoid: Optional[str] = None
    run_humanoid = bool(args.system_prompt_humanoid)
    if run_humanoid:
        if not os.path.isfile(args.system_prompt_humanoid):
            print(f"[ERROR] --system-prompt-humanoid 文件不存在: {args.system_prompt_humanoid}",
                  file=sys.stderr)
            sys.exit(2)
        sys_prompt_humanoid = load_system_prompt(args.system_prompt_humanoid)

    run_yesno = not args.only_humanoid
    if args.only_humanoid and not run_humanoid:
        print("[ERROR] --only-humanoid 需要同时指定 --system-prompt-humanoid",
              file=sys.stderr)
        sys.exit(2)

    print(f"[batch_eval] specs loaded    : {len(specs)}")
    print(f"[batch_eval] system_prompt len: {len(sys_prompt)} chars")
    if sys_prompt_humanoid is not None:
        print(f"[batch_eval] system_prompt_humanoid len: {len(sys_prompt_humanoid)} chars")
    print(f"[batch_eval] run_yesno={run_yesno}  run_humanoid={run_humanoid}")
    print(f"[batch_eval] judge_backend    : {args.judge_backend}", flush=True)

    if args.jobs_tsv:
        jobs = _load_jobs_tsv(args.jobs_tsv)
        if not jobs:
            print(f"[ERROR] no valid jobs in {args.jobs_tsv}", file=sys.stderr)
            sys.exit(2)
        print(f"[batch_eval] multi-job mode: {len(jobs)} jobs from {args.jobs_tsv}")
        for j in jobs:
            print(f"  - {j['tag']:<24} manifest={j['manifest']}  out={j['out_dir']}")
    else:
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

    backend = build_judge_backend(args)

    t_all = time.time()
    for j in jobs:
        _run_one_job(
            backend,
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
            sys_prompt_humanoid=sys_prompt_humanoid,
            run_yesno=run_yesno,
            run_humanoid=run_humanoid,
        )
    print()
    print(f"[batch_eval] ALL JOBS DONE in {time.time()-t_all:.0f}s  (jobs={len(jobs)})")


def main():
    ap = argparse.ArgumentParser()
    # 单 job 模式
    ap.add_argument("--manifest", default="",
                    help="单 job 模式: shotcut_manifest.jsonl 路径")
    ap.add_argument("--out-dir", default="",
                    help="单 job 模式: 该模型的 runs/<judge>/<M>/ 输出目录")
    ap.add_argument("--tag", default="",
                    help="单 job 模式: 模型名 (仅打印用)")
    # 多 job 模式
    ap.add_argument("--jobs-tsv", default="",
                    help="多 job 模式: TSV, 每行 'tag\\tmanifest_path\\tout_dir'")
    # 公共参数
    ap.add_argument("--questions", required=True)
    ap.add_argument("--specs-dir", required=False, default="")
    ap.add_argument("--system-prompt", required=True,
                    help="通用 yes/no rubric (只判 motion)")
    ap.add_argument("--system-prompt-humanoid", default="",
                    help="整体人体形态 1-5 分严格 rubric md 路径; 传了就会"
                         "对每个 passed spec 额外跑一次整段视频的 humanoid_integrity "
                         "分项 (score_1to5, spec 级, shot_idx=0). 不传则跳过.")
    ap.add_argument("--only-humanoid", action="store_true",
                    help="只跑 humanoid_integrity 分项, 跳过 yes/no. "
                         "需同时指定 --system-prompt-humanoid. 用于补跑新分项 "
                         "而不覆盖旧 yes/no 结果的场景 (旧 yes/no 记录仍会被 "
                         "per_question.jsonl 覆盖丢失, 请自行 merge; "
                         "aggregate_summary.py 的 --append-models 追加模式可"
                         "避免其它模型 CSV 列被动).")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="单题 yes/no 生成预算 (默认 512, 会 clamp 到 [0, 1024])")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-num", type=int, default=1)

    # judge backend 参数
    ap.add_argument("--judge-backend", choices=["qwen_vl"], default="qwen_vl",
                    help="评测模型后端 (qwen_vl=本地 vLLM Qwen3-VL)")
    ap.add_argument("--judge-base-url", default="http://127.0.0.1:8080/v1",
                    help="qwen_vl: vLLM OpenAI 兼容入口")
    ap.add_argument("--judge-model", default="Qwen3-VL-235B-A22B-Instruct",
                    help="qwen_vl: served-model-name")
    ap.add_argument("--judge-api-key", default="EMPTY",
                    help="qwen_vl: vLLM 不校验, 默认 EMPTY")
    ap.add_argument("--judge-timeout-sec", type=int, default=600)
    ap.add_argument("--judge-temperature", type=float, default=0.0)
    ap.add_argument("--judge-top-p", type=float, default=0.9)
    ap.add_argument("--judge-max-retry", type=int, default=3)
    ap.add_argument("--judge-video-local-path", action="store_true",
                    help="qwen_vl: 传 file:// 本地路径而非 data: base64 "
                         "(需要 vLLM 端启用 allow_local_media_path)")
    # 视频采样密度 (qwen_vl backend, 透传到 vLLM Qwen3-VL processor)
    ap.add_argument("--video-fps", type=float, default=12.0,
                    help="qwen_vl: 视频抽帧 fps; 仅在 strategy 降级到 max_frames/fps_only 时生效")
    ap.add_argument("--video-max-frames", type=int, default=96,
                    help="qwen_vl: 视频固定抽帧数 (mm_processor_kwargs.num_frames)")
    args = ap.parse_args()

    if args.shard_num <= 0:
        ap.error("--shard-num must be >= 1")
    if not (0 <= args.shard_idx < args.shard_num):
        ap.error(f"--shard-idx must be in [0, {args.shard_num}), got {args.shard_idx}")
    if args.jobs_tsv and (args.manifest or args.out_dir):
        ap.error("--jobs-tsv and (--manifest/--out-dir) are mutually exclusive")
    run(args)


if __name__ == "__main__":
    main()
