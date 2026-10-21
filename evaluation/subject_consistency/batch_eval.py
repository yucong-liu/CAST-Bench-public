#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject_consistency 评测点 · 批量评测脚本 (grouped 模式 + 多 judge backend 解耦)

依赖前置:
  prepare_manifest.py 已经为该视频模型生成了:
      <out_dir>/manifest.jsonl
  每行格式:
      {"spec_id":"speech_0001","category":"speech","shot_count":"multi_shot",
       "passed":true,"video_path":"/.../speech/speech_0001.mp4","note":"ok"}

输入:
  --manifest        manifest.jsonl 路径
  --questions       t2av-bench/questions/subject_consistency_llm.jsonl
                    每行: {"spec_id":..., "shot_idx":0, "question":..., "associated_subject_id":...}
                    本评测把同一 spec_id 下的所有 question 聚成 1 个 bucket,
                    整段视频送进 VLM 一次推理 (shot_idx 仅作 fanout 计数, 不切镜头).
  --system-prompt   evaluation/subject_consistency/system_prompt.md
  --out-dir         本 (judge, video_model) 组合的 runs 目录
  --tag             视频模型名 (打印用)
  --judge-backend   "qwen_vl"
                    qwen_vl  -> 调用本地 vLLM OpenAI 兼容服务 (Qwen3-VL-235B 等),
                                与 t2av-bench 工程用同一套 base_url / model.
  --judge-base-url  qwen_vl backend: vLLM base url (默认 http://127.0.0.1:8080/v1)
  --judge-model     qwen_vl backend: served-model-name
                                     (默认 Qwen3-VL-235B-A22B-Instruct)
  --judge-api-key   qwen_vl backend: vLLM 不校验, 默认 "EMPTY"
  --max-new-tokens  生成预算基础值, 实际 = min(4096, base + 256*N)
  --limit           可选: 只跑前 N 个 spec (passed=True 的前 N 个), 调试用

Grouped 模式:
  同一个 spec_id 下的所有 question (典型 1~3 题, 全部 yes/no) 合并到 1 条 user
  消息, 视频只编码 1 次. 模型按指令返回长度 = N 的 JSON 数组, 逐位对齐
  questions 列表; 解析失败的元素 fallback 为 "no".

输出:
  <out-dir>/per_question.jsonl  每条 question 一行 (同 bucket 共享 raw_response)
  <out-dir>/summary.json        统计计数
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
def load_questions(path: str) -> Dict[str, List[dict]]:
    """返回 {spec_id: [question_obj, ...]}, 保留 jsonl 顺序.
    本评测忽略 shot_idx (整段视频送入), 同一 spec_id 下所有题合并.
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
# Judge backend 抽象 (qwen_vl via vLLM OpenAI API)
# ============================================================
class JudgeBackend:
    """所有 judge backend 的统一接口."""

    name: str = "base"

    def generate(self, system_text: str, video_path: str, user_text: str,
                 max_new_tokens: int = 512) -> str:
        raise NotImplementedError


def _video_to_data_url(path: str) -> str:
    """读视频文件并编码成 data: URL (vLLM Qwen3-VL 支持的视频输入方式之一)."""
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
                 video_as_data_url: bool = True):
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
        # vLLM 端建议把视频编为 data: URL 直接走 content; 否则要 fileserver
        self.video_as_data_url = video_as_data_url
        print(f"[judge=qwen_vl] base_url={base_url} model={model} "
              f"timeout={timeout_sec}s video_as_data_url={video_as_data_url}", flush=True)

    def _build_messages(self, system_text: str, video_path: str, user_text: str) -> List[dict]:
        if self.video_as_data_url:
            video_url = _video_to_data_url(video_path)
        else:
            # 让 vLLM 自己从本地路径读 (需要 vLLM 端启用 allow_local_media)
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

    def generate(self, system_text: str, video_path: str, user_text: str,
                 max_new_tokens: int = 512) -> str:
        messages = self._build_messages(system_text, video_path, user_text)
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retry + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=max_new_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
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
        )
    raise ValueError(f"unknown --judge-backend: {bk}")


# ============================================================
# Grouped prompt 构造 / 解析  (与 emtion_consistency 同款 yesno-only 解析器)
# ============================================================
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_JSON_ARR_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,\s*)*\{.*?\}\s*\]", re.DOTALL)


def build_grouped_user_text(qs: List[dict]) -> str:
    n = len(qs)
    lines: List[str] = []
    lines.append(f"You will answer the following {n} yes/no question(s) about the SAME video clip above.")
    lines.append("Each question is independent. For each one, return an item of the form")
    lines.append('  {"reason": "<short>", "answer": "yes"|"no"}')
    lines.append("following the system rules (subject must be visually present in the clip at any moment).")
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
    parsed: List[Optional[dict]] = [None] * n
    ok: List[bool] = [False] * n
    if not raw or n <= 0:
        return parsed, ok

    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("json").lstrip("JSON").strip()

    # n == 1: 直接当对象解析
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
                p, k = _coerce_yesno(obj[0])
                if k:
                    parsed[0] = p
                    ok[0] = True
                    return parsed, ok
        return parsed, ok

    # n >= 2: 期望数组
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

    # Fallback A: 找 N 个独立 JSON 对象
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

    # Fallback B: 末尾被截 (没有 ']'), 按花括号深度收集已闭合对象
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
                ) -> Tuple[List[Tuple[str, str, str, str, List[dict]]], List[dict]]:
    """返回 (todo, manifest_raw).

    todo = [(spec_id, category, shot_count, video_path, [qobj,...]), ...]
           仅含 passed 且有 question 的 spec.
    """
    questions = load_questions(questions_path)
    manifest = load_manifest(manifest_path)

    passed_specs = [m for m in manifest if m.get("passed")]
    if limit and limit > 0:
        passed_specs = passed_specs[: limit]

    todo: List[Tuple[str, str, str, str, List[dict]]] = []
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
        cat = m.get("category", "") or ""
        sc = m.get("shot_count", "") or ""
        todo.append((sid, cat, sc, video_path, list(qs)))
    return todo, manifest


def _run_one_job(
    backend: JudgeBackend,
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
) -> None:
    print()
    print(f"############ [batch_eval] job tag={tag} (shard {shard_idx}/{shard_num}) ############")
    print(f"  manifest = {manifest_path}")
    print(f"  out_dir  = {out_dir}")
    print(f"  questions= {questions_path}", flush=True)

    todo, manifest_raw = _build_todo(manifest_path, questions_path, limit=limit)

    n_total_buckets_global = len(todo)
    n_total_questions_global = sum(len(t[4]) for t in todo)
    print(f"[batch_eval] passed specs: {len(manifest_raw)} -> todo {n_total_buckets_global} buckets")
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
        open(out_path, "w", encoding="utf-8").close()
        summary = {
            "tag": tag,
            "judge_backend": backend.name,
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

    with open(out_path, "w", encoding="utf-8") as fout:
        for bi, (sid, cat, shot_count, video_path, qs) in enumerate(todo, start=1):
            n_in_bucket = len(qs)
            user_text = build_grouped_user_text(qs)

            # 动态 token 预算: base + 256*N, 上限 4096
            mnt = min(4096, int(max_new_tokens_base) + 256 * n_in_bucket)

            t0 = time.time()
            try:
                raw = backend.generate(
                    system_text=sys_prompt,
                    video_path=video_path,
                    user_text=user_text,
                    max_new_tokens=mnt,
                )
                err = ""
            except Exception as e:  # noqa: BLE001
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
                    "category": cat,
                    "shot_count_kind": shot_count,
                    "shot_idx": qobj.get("shot_idx"),
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

            # 进度行: 每完成 1 个 bucket 都打印一次, 方便前台 tail -f 实时观察.
            if True:
                elapsed = time.time() - t_start
                rate = bi / max(elapsed, 1e-6)
                eta = (n_buckets - bi) / max(rate, 1e-6)
                ok_in_bucket = sum(1 for o in ok_list if o)
                print(
                    f"[job={tag}] [bucket {bi}/{n_buckets}] {sid} ({cat}) N={n_in_bucket} "
                    f"ok={ok_in_bucket}/{n_in_bucket} dt={dt:.1f}s mnt={mnt} "
                    f"| elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )

    summary = {
        "tag": tag,
        "judge_backend": backend.name,
        "manifest": manifest_path,
        "n_specs_in_manifest": len(manifest_raw),
        "n_specs_passed": len([m for m in manifest_raw if m.get("passed")]),
        "n_buckets": n_buckets,
        "n_questions": n_questions_total,
        "n_yesno_questions": n_questions_total,
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

    print(f"[batch_eval] system_prompt len : {len(sys_prompt)} chars")
    print(f"[batch_eval] judge_backend     : {args.judge_backend}", flush=True)

    backend = build_judge_backend(args)

    t_all = time.time()
    for j in jobs:
        _run_one_job(
            backend,
            tag=j["tag"],
            manifest_path=j["manifest"],
            questions_path=args.questions,
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
    # 单 job 模式
    ap.add_argument("--manifest", default="",
                    help="单 job 模式: manifest.jsonl 路径")
    ap.add_argument("--out-dir", default="",
                    help="单 job 模式: 该模型的 runs/<judge>/<M>/ 输出目录")
    ap.add_argument("--tag", default="",
                    help="单 job 模式: 模型名 (仅打印用)")
    # 多 job 模式
    ap.add_argument("--jobs-tsv", default="",
                    help="多 job 模式: TSV, 每行 'tag\\tmanifest_path\\tout_dir'")
    # 公共参数
    ap.add_argument("--questions", required=True)
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="生成预算基础值; 实际 = min(4096, base + 256*N)")
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
