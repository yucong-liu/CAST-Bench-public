#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr · judge_qwen.py
===================

Stage 2: 把 stage1 产出的 tracks JSON + 视频对应的 prompt 文本喂给本地 Qwen
(OpenAI 兼容 chat.completions, 默认 Qwen3-VL-235B-A22B-Instruct vLLM 8080),
让它按 AVGen-Bench 原版 OCR 脚本 原版 prompt 给出多维 0-100 分.

裁判 prompt 文本完全沿用 AVGen-Bench 原版 OCR 脚本 的版本 (见 judge_prompt.py),
只是把"调用方"从 Gemini 换成 Qwen, 装载方式从 system_instruction+user_parts
改成 OpenAI 的 messages.

输入:
  --runs-root  runs/paddleocr/                 stage1 产物根
  --models     LTX-2.3,Ovi,...                 模型清单 (与 id_consistency 一致)
  --ocr-specs  ocr/ocr_specs.jsonl             vid 白名单 (含 category)
  --prompts-dir /.../t2av-bench/prompts        prompts/<category>.json
  config:      llm.qwen.{base_url, prompt_model, api_key, max_tokens, timeout_sec}
               读自 t2av_bench.config.load_config; CLI 可覆盖.

输出:
  score_out/qwen/<model>/<vid>.json           per-video 裁判结果 (含 raw + 解析)
  score_out/qwen/<model>/_summary.csv         per-model 每 vid 一行
  score_out/qwen/judge_summary.csv            跨 model 横向 (overall + 4 个 subscore)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from judge_prompt import get_prompt_variant  # noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_RUNS_ROOT = THIS_DIR / "runs" / "paddleocr"
DEFAULT_OUT_DIR = THIS_DIR / "score_out" / "qwen"
DEFAULT_OCR_SPECS = THIS_DIR / "ocr_specs.jsonl"
DEFAULT_PROMPTS_DIR = (_REPO_ROOT / "t2av-bench/prompts")
DEFAULT_MODELS = ["LTX-2.3", "Ovi", "daVinci-MagiHuman", "seedance", "wan2.7"]


# ============================================================
# IO
# ============================================================
def load_ocr_specs(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if not path.is_file():
        raise FileNotFoundError(f"ocr_specs.jsonl 不存在: {path} -- 请先 run filter_specs.py")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = obj.get("id")
            if isinstance(vid, str) and vid:
                out[vid] = obj
    return out


def load_prompts_for_categories(prompts_dir: Path, categories: List[str]) -> Dict[str, str]:
    """读 prompts/<cat>.json (顶层 JSON 数组, 每条 {id, prompt}) -> {vid: prompt_text}"""
    out: Dict[str, str] = {}
    for cat in categories:
        p = prompts_dir / f"{cat}.json"
        if not p.is_file():
            print(f"[warn] prompts 文件不存在: {p}", file=sys.stderr)
            continue
        try:
            arr = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 解析 prompts {p} 失败: {e!r}", file=sys.stderr)
            continue
        if not isinstance(arr, list):
            continue
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            vid = obj.get("id")
            prompt = obj.get("prompt")
            if isinstance(vid, str) and vid and isinstance(prompt, str) and prompt.strip():
                out[vid] = prompt
    return out


def load_tracks(runs_root: Path, model: str, vid: str) -> Tuple[str, Optional[Dict]]:
    """读 stage1 产物 runs/paddleocr/<model>/tracks/<vid>.json.

    返回 (status, payload)
      status ∈ {"ok", "missing", "bad", "empty"}
      payload: {video_meta, pipeline_meta, tracks} 或 None
    """
    f = runs_root / model / "tracks" / f"{vid}.json"
    if not f.is_file():
        return "missing", None
    try:
        rec = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] tracks 解析失败 {f}: {e!r}", file=sys.stderr)
        return "bad", None
    if not rec.get("ok", False):
        return "bad", None
    payload = {
        "video_meta": rec.get("video_meta") or {},
        "pipeline_meta": rec.get("pipeline_meta") or {},
        "tracks": rec.get("tracks") or [],
    }
    if not payload["tracks"]:
        return "empty", payload
    return "ok", payload


# ============================================================
# 解析 / clamp
# ============================================================
def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    return text


def _extract_json(text: str) -> str:
    text = _strip_code_fence(text)
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON in response:\n{text[:1500]}")
    return m.group(0)


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(0.0, min(1.0, v))


def _clamp100(x: Any) -> int:
    try:
        v = int(x)
    except Exception:
        try:
            v = int(round(float(x)))
        except Exception:
            v = 0
    return max(0, min(100, v))


def parse_judge_output(raw_text: str) -> Dict[str, Any]:
    """跟 原版 gemini_judge() 同源的解析 + clamp."""
    data = json.loads(_extract_json(raw_text))
    if not isinstance(data, dict):
        raise ValueError(f"裁判返回非对象: {str(data)[:200]}")
    data["overall_text_quality_score"] = _clamp100(data.get("overall_text_quality_score", 0))
    subs = data.get("subscores", {}) or {}
    data["subscores"] = {
        "legibility_accuracy": _clamp100(subs.get("legibility_accuracy", 0)),
        "temporal_stability": _clamp100(subs.get("temporal_stability", 0)),
        "spatial_stability": _clamp100(subs.get("spatial_stability", 0)),
        "completeness": _clamp100(subs.get("completeness", 0)),
        "prompt_text_match": subs.get("prompt_text_match", None),
    }
    data["confidence"] = _clamp01(data.get("confidence", 0.0))
    return data


# ============================================================
# 调用
# ============================================================
def call_judge(
    *,
    client: Any,
    model: str,
    system_text: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
    max_retry: int,
    vid: str,
) -> Tuple[Dict[str, Any], str]:
    last_err: Optional[Exception] = None
    raw_text = ""
    for attempt in range(1, max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_text = resp.choices[0].message.content or ""
            return parse_judge_output(raw_text), raw_text
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(
                f"[warn] vid={vid} 第 {attempt}/{max_retry} 次失败: {e!r}; "
                f"{wait}s 后重试  (raw[:200]={raw_text[:200]!r})",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"vid={vid} LLM 调用全部失败: {last_err!r}")


# ============================================================
# Main
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="OCR stage2 judge: tracks JSON + prompt -> Qwen3-VL 多维 0-100 分"
    )
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--ocr-specs", type=Path, default=DEFAULT_OCR_SPECS)
    ap.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                    help="逗号分隔的 model 名 (对应 runs-root/<model>/tracks)")
    ap.add_argument("--prompt-variant", type=str, default=os.getenv("OCR_PROMPT_VARIANT", "original"),
                    choices=["original", "v1", "v2"])
    ap.add_argument("--base-url", type=str, default=None)
    ap.add_argument("--model", type=str, default=None,
                    help="LLM 模型名 (服务端 served-model-name)")
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout-sec", type=int, default=None)
    ap.add_argument("--max-retry", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 个 (vid, model) 组合, 0=不限")
    ap.add_argument("--ids", type=str, default="",
                    help="逗号分隔的 vid 白名单")
    ap.add_argument("--skip-existing", action="store_true",
                    help="若 score_out/qwen/<model>/<vid>.json 已存在则跳过")
    ap.add_argument("--num-workers", type=int, default=1,
                    help="vid 维度并发 LLM 调用的线程数 (默认 1=串行); 推荐 4-8")
    args = ap.parse_args()

    # ---- 读 specs / prompts ----
    specs = load_ocr_specs(args.ocr_specs)
    print(f"[ocr.judge] ocr_specs: {args.ocr_specs} (共 {len(specs)} 条)")

    cats_in_scope = sorted({(spec.get("category") or vid.split("_", 1)[0]) for vid, spec in specs.items()})
    print(f"[ocr.judge] categories in scope: {cats_in_scope}")
    prompts_map = load_prompts_for_categories(args.prompts_dir, cats_in_scope)
    print(f"[ocr.judge] prompts loaded: {len(prompts_map)} ids")

    id_whitelist: set[str] = set()
    if args.ids.strip():
        id_whitelist = {x.strip() for x in args.ids.split(",") if x.strip()}

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"[ocr.judge] models: {models}")
    print(f"[ocr.judge] runs-root: {args.runs_root}")
    print(f"[ocr.judge] out-dir  : {args.out_dir}")

    # ---- LLM client (OpenAI 兼容) ----
    cfg_base_url = cfg_model = cfg_api_key = None
    cfg_max_tokens = cfg_timeout_sec = None
    try:
        from t2av_bench.config import load_config
        cfg = load_config()
        cfg_base_url = cfg.get("llm.qwen.base_url")
        cfg_model = cfg.get("llm.qwen.prompt_model") or cfg.get("llm.qwen.spec_model")
        cfg_api_key = cfg.get("llm.qwen.api_key")
        cfg_max_tokens = cfg.get("llm.qwen.max_tokens")
        cfg_timeout_sec = cfg.get("llm.qwen.timeout_sec")
    except Exception as e:
        print(f"[warn] 读取 t2av_bench.config 失败, 仅依赖 CLI: {e!r}", file=sys.stderr)

    max_tokens = args.max_tokens or cfg_max_tokens or 1024
    timeout_sec = args.timeout_sec or cfg_timeout_sec or 600

    try:
        from openai import OpenAI
    except ImportError as e:
        print(f"[error] 需要 openai 库: pip install openai\n  详情: {e}", file=sys.stderr)
        return 2

    base_url = args.base_url or cfg_base_url
    llm_model = args.model or cfg_model
    api_key = args.api_key or cfg_api_key or "EMPTY"
    if not base_url:
        print("[error] 缺 base_url; 用 --base-url 或 config llm.qwen.base_url", file=sys.stderr)
        return 2
    if not llm_model:
        print("[error] 缺 model; 用 --model 或 config llm.qwen.prompt_model", file=sys.stderr)
        return 2

    print(f"[ocr.judge] LLM base_url: {base_url}")
    print(f"[ocr.judge] LLM model   : {llm_model}")
    print(f"[ocr.judge] temperature={args.temperature} max_tokens={max_tokens} timeout={timeout_sec}s")
    print(f"[ocr.judge] prompt_variant: {args.prompt_variant}")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)
    system_text, user_template = get_prompt_variant(args.prompt_variant)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- vid 列表 ----
    all_vids = sorted(specs.keys())
    if id_whitelist:
        all_vids = [v for v in all_vids if v in id_whitelist]
    print(f"[ocr.judge] 待评 vid 数 = {len(all_vids)}", flush=True)

    overall_rows: List[Dict[str, Any]] = []
    n_done = 0

    for model_name in models:
        print(f"\n=== model: {model_name} ===")
        model_out = args.out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        per_rows: List[Dict[str, Any]] = []
        # 累计 (mean) 用列表
        overall_scores: List[int] = []
        sub_legi: List[int] = []
        sub_temp: List[int] = []
        sub_spat: List[int] = []
        sub_comp: List[int] = []
        n_missing = 0
        n_failed = 0

        # ---- 第一趟: cache hit / 无 prompt / tracks missing 直接处理 ----
        pending: List[Tuple[str, str, Dict, Dict[str, Any]]] = []
        # tuple = (vid, hyp_status, payload, extra)
        for vid in all_vids:
            if args.limit > 0 and (n_done + 1) > args.limit:
                print(f"[info] 命中 --limit={args.limit}, 停止提交")
                break
            n_done += 1
            spec = specs.get(vid) or {}
            cat = spec.get("category") or vid.split("_", 1)[0]
            out_file = model_out / f"{vid}.json"

            # cache hit
            if args.skip_existing and out_file.is_file():
                try:
                    cached = json.loads(out_file.read_text(encoding="utf-8"))
                    if cached.get("judge_failed"):
                        # 失败的不算 cache hit, 重跑
                        pass
                    else:
                        ov = cached.get("overall_text_quality_score")
                        subs = cached.get("subscores") or {}
                        if isinstance(ov, int):
                            per_rows.append({
                                "video_id": vid,
                                "category": cat,
                                "hyp_status": cached.get("hyp_status", "ok"),
                                "overall": ov,
                                "legibility_accuracy": subs.get("legibility_accuracy"),
                                "temporal_stability": subs.get("temporal_stability"),
                                "spatial_stability": subs.get("spatial_stability"),
                                "completeness": subs.get("completeness"),
                                "prompt_text_match": subs.get("prompt_text_match"),
                                "confidence": cached.get("confidence"),
                                "summary": cached.get("summary", ""),
                            })
                            if cached.get("hyp_status") not in ("missing", "bad"):
                                overall_scores.append(ov)
                                if isinstance(subs.get("legibility_accuracy"), int):
                                    sub_legi.append(subs["legibility_accuracy"])
                                if isinstance(subs.get("temporal_stability"), int):
                                    sub_temp.append(subs["temporal_stability"])
                                if isinstance(subs.get("spatial_stability"), int):
                                    sub_spat.append(subs["spatial_stability"])
                                if isinstance(subs.get("completeness"), int):
                                    sub_comp.append(subs["completeness"])
                            print(f"[skip] {model_name}/{vid} 已存在 overall={ov}")
                            continue
                except Exception:
                    pass

            # tracks
            hyp_status, payload = load_tracks(args.runs_root, model_name, vid)
            if hyp_status == "missing":
                n_missing += 1
                print(f"[skip] {model_name}/{vid} tracks 缺失, 视频未生成或 stage1 未跑, 跳过")
                # 落一个 missing 标记 (便于 cache)
                rec = {
                    "video_id": vid, "model": model_name, "category": cat,
                    "hyp_status": "missing", "judge_failed": False,
                    "overall_text_quality_score": None, "subscores": {},
                }
                out_file.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            if hyp_status == "bad":
                # 直接给 0 分, 不调 LLM
                rec = {
                    "video_id": vid, "model": model_name, "category": cat,
                    "hyp_status": "bad", "judge_failed": False,
                    "overall_text_quality_score": 0,
                    "subscores": {
                        "legibility_accuracy": 0, "temporal_stability": 0,
                        "spatial_stability": 0, "completeness": 0,
                        "prompt_text_match": None,
                    },
                    "confidence": 0.0,
                    "top_issues": [{"track_id": "-", "issue_type": "missing",
                                    "severity": 5, "evidence": "stage1 tracks 解析失败"}],
                    "good_tracks": [],
                    "summary": "stage1 解析失败, 视为最低分",
                }
                out_file.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                per_rows.append({
                    "video_id": vid, "category": cat, "hyp_status": "bad",
                    "overall": 0, "legibility_accuracy": 0, "temporal_stability": 0,
                    "spatial_stability": 0, "completeness": 0,
                    "prompt_text_match": None, "confidence": 0.0, "summary": rec["summary"],
                })
                overall_scores.append(0)
                sub_legi.append(0); sub_temp.append(0); sub_spat.append(0); sub_comp.append(0)
                continue

            prompt_text = prompts_map.get(vid, "")
            if not prompt_text:
                # spec 兜底: 用 spec.prompt 字段
                spec_pt = spec.get("prompt")
                if isinstance(spec_pt, str) and spec_pt.strip():
                    prompt_text = spec_pt

            extra = {"category": cat, "prompt_text": prompt_text}
            pending.append((vid, hyp_status, payload, extra))

        # ---- 第二趟: 并发调用 LLM ----
        def _do_one(vid: str, hyp_status: str, payload: Dict, extra: Dict[str, Any]):
            try:
                user_text = user_template.format(
                    prompt_text=extra.get("prompt_text") or "",
                    tracks_json=json.dumps(payload, ensure_ascii=False),
                )
                parsed_, raw_ = call_judge(
                    client=client, model=llm_model,
                    system_text=system_text, user_text=user_text,
                    temperature=args.temperature, max_tokens=max_tokens,
                    max_retry=args.max_retry, vid=vid,
                )
                return (vid, hyp_status, extra, parsed_, raw_, None)
            except Exception as e:
                return (vid, hyp_status, extra, None, "", e)

        results: List[Tuple] = []
        nw = max(1, int(args.num_workers))
        if nw == 1 or len(pending) <= 1:
            for vid, hs, pl, ex in pending:
                results.append(_do_one(vid, hs, pl, ex))
        else:
            print(f"[info] model={model_name}: 并发 LLM 调用 num_workers={nw}, pending={len(pending)}")
            with ThreadPoolExecutor(max_workers=nw) as exe:
                fut_map = {exe.submit(_do_one, v, hs, pl, ex): v
                           for v, hs, pl, ex in pending}
                tmp: Dict[str, Tuple] = {}
                for fut in as_completed(fut_map):
                    res = fut.result()
                    tmp[res[0]] = res
                for vid, _, _, _ in pending:
                    if vid in tmp:
                        results.append(tmp[vid])

        # ---- 第三趟: 落盘 + 累计 ----
        for vid, hyp_status, extra, parsed, raw, err in results:
            cat = extra.get("category", "")
            out_file = model_out / f"{vid}.json"
            if err is not None:
                n_failed += 1
                print(f"[error] {model_name}/{vid} judge 失败: {err!r}", file=sys.stderr)
                out_file.write_text(json.dumps({
                    "video_id": vid, "model": model_name, "category": cat,
                    "hyp_status": hyp_status, "judge_failed": True, "error": repr(err),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                continue

            ov = parsed["overall_text_quality_score"]
            subs = parsed["subscores"]
            print(f"[ok ] {model_name}/{vid}  overall={ov}  legi={subs['legibility_accuracy']} "
                  f"temp={subs['temporal_stability']} spat={subs['spatial_stability']} "
                  f"comp={subs['completeness']}  status={hyp_status}")

            record = {
                "video_id": vid, "model": model_name, "category": cat,
                "hyp_status": hyp_status, "judge_failed": False,
                "overall_text_quality_score": ov,
                "subscores": subs,
                "confidence": parsed.get("confidence", 0.0),
                "top_issues": parsed.get("top_issues", []),
                "good_tracks": parsed.get("good_tracks", []),
                "summary": parsed.get("summary", ""),
                "raw_response": raw,
            }
            out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

            per_rows.append({
                "video_id": vid, "category": cat, "hyp_status": hyp_status,
                "overall": ov,
                "legibility_accuracy": subs["legibility_accuracy"],
                "temporal_stability": subs["temporal_stability"],
                "spatial_stability": subs["spatial_stability"],
                "completeness": subs["completeness"],
                "prompt_text_match": subs.get("prompt_text_match"),
                "confidence": parsed.get("confidence", 0.0),
                "summary": parsed.get("summary", ""),
            })
            overall_scores.append(ov)
            sub_legi.append(subs["legibility_accuracy"])
            sub_temp.append(subs["temporal_stability"])
            sub_spat.append(subs["spatial_stability"])
            sub_comp.append(subs["completeness"])

        # ---- 写 model 级 csv ----
        per_csv = model_out / "_summary.csv"
        with per_csv.open("w", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=[
                "video_id", "category", "hyp_status",
                "overall", "legibility_accuracy", "temporal_stability",
                "spatial_stability", "completeness", "prompt_text_match",
                "confidence", "summary",
            ])
            w.writeheader()
            for r in per_rows:
                w.writerow(r)

        def _avg(lst: List[int]) -> Optional[float]:
            return round(statistics.mean(lst), 2) if lst else None

        overall_rows.append({
            "model": model_name,
            "n_all": len(overall_scores),
            "n_missing": n_missing,
            "n_failed": n_failed,
            "overall_avg": _avg(overall_scores),
            "legibility_avg": _avg(sub_legi),
            "temporal_avg": _avg(sub_temp),
            "spatial_avg": _avg(sub_spat),
            "completeness_avg": _avg(sub_comp),
        })
        print(f"[model={model_name}] overall_avg={_avg(overall_scores)} "
              f"legi={_avg(sub_legi)} temp={_avg(sub_temp)} spat={_avg(sub_spat)} "
              f"comp={_avg(sub_comp)}  (n={len(overall_scores)} missing={n_missing} "
              f"failed={n_failed})")

    # ---- 跨 model 总表 ----
    overall_csv = args.out_dir / "judge_summary.csv"
    with overall_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=[
            "model", "n_all", "n_missing", "n_failed",
            "overall_avg", "legibility_avg", "temporal_avg",
            "spatial_avg", "completeness_avg",
        ])
        w.writeheader()
        for r in overall_rows:
            w.writerow(r)

    print()
    print(f"[ocr.judge] DONE  -> {overall_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
