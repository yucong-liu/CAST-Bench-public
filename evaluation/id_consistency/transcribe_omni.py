#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
id_consistency · 用 Qwen3-Omni 端到端转写视频 → (speaker, text) segments

模型架构与 audio_consistency/batch_eval.py 完全一致:
    Qwen3OmniMoeForConditionalGeneration + Qwen3OmniMoeProcessor
    + qwen_omni_utils.process_mm_info(use_audio_in_video=True)
    + cfg.enable_audio_output = False   (关闭 talker, 规避 Code2WavConfig bug)

输入模式:
    多模型: --models-tsv + --runs-root  (推荐)
    单模型: --videos-root + --out-dir + --tag

输出: <runs-root>/<model>/transcripts/<video_id>.json
    {
      "video_id": "speech_0011",
      "category": "speech",
      "video_path": "...",
      "raw_response": "...",
      "parse_ok": true,
      "segments": [{"speaker":"S1","text":"..."}, ...],
      "infer_seconds": 12.3,
      "error": ""
    }

shard 切片 (DP N x 1): 按 (model, video) 全局序列, stride=N. 每张卡 1 个进程,
脚本内通过 device='cuda:0' 配合 CUDA_VISIBLE_DEVICES=$i 实现.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 让脚本能 import 同级仓库下的 qwen35_api 包 (API backend 用)
_EVAL_REPO_DIR = os.path.dirname(_THIS_DIR)
if _EVAL_REPO_DIR not in sys.path:
    sys.path.insert(0, _EVAL_REPO_DIR)

from parse_omni_id import parse_omni_segments  # noqa: E402

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


# ============================================================
# 视频扫描 (复用 speechID/transcribe.py 的约定)
# ============================================================
_KNOWN_CATEGORIES = (
    "speech", "action", "ambient", "animal", "bgm",
    "instrument", "non_verbal_vocal", "object", "sfx",
)
_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def resolve_videos_root(root: str) -> str:
    """root 直接含 category 子目录就用它本身; 否则下钻一层 (兼容 daVinci 的 540p 中间层)."""
    if not root or not os.path.isdir(root):
        return root
    for cat in _KNOWN_CATEGORIES:
        if os.path.isdir(os.path.join(root, cat)):
            return root
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        for cat in _KNOWN_CATEGORIES:
            if os.path.isdir(os.path.join(sub, cat)):
                return sub
    return root


def scan_videos(root: str, categories: List[str]) -> List[Dict[str, str]]:
    """扫描 <root>/<cat>/*.mp4, 返回 [{video_id, category, video_path}, ...]."""
    out: List[Dict[str, str]] = []
    real_root = resolve_videos_root(root)
    for cat in categories:
        cdir = os.path.join(real_root, cat)
        if not os.path.isdir(cdir):
            continue
        for name in sorted(os.listdir(cdir)):
            if not name.lower().endswith(_VIDEO_EXTS):
                continue
            vid = os.path.splitext(name)[0]
            out.append({
                "video_id": vid,
                "category": cat,
                "video_path": os.path.join(cdir, name),
            })
    return out


# ============================================================
# models.tsv
# ============================================================
def load_models_tsv(path: str,
                    only: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    only_set = set(only) if only else None
    out: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            root = parts[1].strip()
            if not name or not root:
                continue
            if only_set is not None and name not in only_set:
                continue
            if not os.path.isabs(root):
                repo_root = os.environ.get("REPO_ROOT") or os.path.abspath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
                root = os.path.join(repo_root, root)
            out.append((name, root))
    return out


# ============================================================
# Qwen3-Omni Instruct 推理封装 (来自 audio_consistency/batch_eval.py)
# ============================================================
class OmniRunner:
    def __init__(self, model_path: str, attn_impl: str = "sdpa",
                 use_audio_in_video: bool = True, device: str = "cuda:0"):
        import torch
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
        print(f"[omni] loading model from {model_path} (attn={attn_impl}, device_map={device_map}) ...",
              flush=True)
        t0 = time.time()
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(cfg, "enable_audio_output"):
            cfg.enable_audio_output = False  # 跳过 talker
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

    def _build_messages(self, system_text: str, video_path: str, user_text: str):
        return [
            {"role": "system",
             "content": [{"type": "text", "text": system_text}]},
            {"role": "user",
             "content": [
                 {"type": "video", "video": video_path},
                 {"type": "text",  "text": user_text},
             ]},
        ]

    def generate(self, system_text: str, video_path: str, user_text: str,
                 max_new_tokens: int = 1024) -> str:
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
# 单条视频推理 + 解析 + 落盘
# ============================================================
USER_PROMPT = (
    "请按照系统指令的规则，把视频中所有人物对白和旁白按时间顺序转录成 JSON。"
    "严格输出 {\"segments\":[{\"speaker\":\"S1\",\"text\":\"...\"}, ...]} 一行 JSON，"
    "不要任何 markdown 代码块、不要解释。"
)


def transcribe_one_video(runner: OmniRunner,
                         system_text: str,
                         video_item: Dict[str, str],
                         out_dir: str,
                         max_new_tokens: int,
                         skip_existing: bool) -> Dict[str, object]:
    """对单条视频跑 1 次 generate, 把结果写到 out_dir/transcripts/<video_id>.json."""
    vid = video_item["video_id"]
    cat = video_item["category"]
    vpath = video_item["video_path"]

    transcripts_dir = os.path.join(out_dir, "transcripts")
    os.makedirs(transcripts_dir, exist_ok=True)
    out_path = os.path.join(transcripts_dir, f"{vid}.json")

    if skip_existing and os.path.isfile(out_path):
        try:
            old = json.loads(open(out_path, "r", encoding="utf-8").read())
            if old.get("parse_ok") and old.get("segments") is not None:
                return {"video_id": vid, "status": "skip", "n_segments": len(old.get("segments") or [])}
        except Exception:
            pass

    t0 = time.time()
    err = ""
    raw = ""
    try:
        raw = runner.generate(
            system_text=system_text,
            video_path=vpath,
            user_text=USER_PROMPT,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        err = repr(e)
    dt = time.time() - t0

    segs, ok = parse_omni_segments(raw)

    rec = {
        "video_id": vid,
        "category": cat,
        "video_path": vpath,
        "raw_response": raw,
        "parse_ok": bool(ok),
        "segments": [
            {"speaker": s, "subject": sub, "text": t}
            for (s, sub, t) in segs
        ],
        "infer_seconds": round(dt, 2),
        "error": err,
        "max_new_tokens": int(max_new_tokens),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    return {"video_id": vid, "status": "ok" if ok else ("err" if err else "parse_fail"),
            "n_segments": len(rec["segments"]), "dt": dt}


# ============================================================
# 主流程
# ============================================================
def build_global_todo(model_pairs: List[Tuple[str, str]],
                      categories: List[str],
                      runs_root: str) -> List[Tuple[str, str, Dict[str, str]]]:
    """构造 [(model_name, model_out_dir, video_item), ...]."""
    todo: List[Tuple[str, str, Dict[str, str]]] = []
    for name, root in model_pairs:
        out_dir = os.path.join(runs_root, name)
        items = scan_videos(root, categories)
        for it in items:
            todo.append((name, out_dir, it))
    return todo


def run_multi(args):
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    only = [m.strip() for m in args.only_models.split(",") if m.strip()] if args.only_models else None

    model_pairs = load_models_tsv(args.models_tsv, only=only)
    if not model_pairs:
        print(f"[ERROR] no models loaded from {args.models_tsv} (only={only})", file=sys.stderr)
        sys.exit(2)

    print(f"[id_consistency] models in scope ({len(model_pairs)}):")
    for name, root in model_pairs:
        print(f"   - {name:<24s} <- {root}")
    print(f"[id_consistency] categories: {cats}", flush=True)

    todo = build_global_todo(model_pairs, cats, args.runs_root)
    n_global = len(todo)
    print(f"[id_consistency] total (model,video) buckets (global): {n_global}", flush=True)

    if args.limit and args.limit > 0:
        # per-model limit: 每个模型只保留前 N 条 (在 build 后再过滤)
        keep: List[Tuple[str, str, Dict[str, str]]] = []
        cnt: Dict[str, int] = {}
        for tup in todo:
            n = tup[0]
            if cnt.get(n, 0) >= args.limit:
                continue
            cnt[n] = cnt.get(n, 0) + 1
            keep.append(tup)
        todo = keep
        print(f"[id_consistency] --limit {args.limit} (per-model) -> keep {len(todo)}",
              flush=True)

    sharded = (args.shard_num > 1)
    if sharded:
        todo = [t for i, t in enumerate(todo) if i % args.shard_num == args.shard_idx]
        print(f"[id_consistency] shard {args.shard_idx}/{args.shard_num} -> keep {len(todo)} buckets",
              flush=True)

    if not todo:
        print("[id_consistency] nothing to run, exit.")
        return

    # ---- system prompt ----
    with open(args.system_prompt, "r", encoding="utf-8") as f:
        system_text = f.read().strip()
    print(f"[id_consistency] system_prompt len = {len(system_text)} chars", flush=True)

    # ---- 模型加载 (1 次) ----
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

    # 后续 summary 中 会用到 runner.model_path / args.model_path; api 模式没传入时
    # 用 OmniApiRunner 默认 model_path 占位 (api://...) 作为记录
    if not args.model_path:
        args.model_path = getattr(runner, "model_path", "") or ""

    # ---- 推理 ----
    counters: Dict[str, Dict[str, int]] = {}
    t_start = time.time()
    n_total = len(todo)
    for bi, (mname, out_dir, item) in enumerate(todo, start=1):
        c = counters.setdefault(mname, {"n": 0, "ok": 0, "err": 0, "skip": 0,
                                         "parse_fail": 0, "n_segments_total": 0})
        try:
            result = transcribe_one_video(
                runner, system_text, item, out_dir,
                max_new_tokens=args.max_new_tokens,
                skip_existing=args.skip_existing,
            )
        except Exception as e:
            result = {"video_id": item["video_id"], "status": "err",
                      "n_segments": 0, "error": repr(e)}
        c["n"] += 1
        c["n_segments_total"] += int(result.get("n_segments") or 0)
        st = result.get("status")
        if st == "ok":
            c["ok"] += 1
        elif st == "skip":
            c["skip"] += 1
        elif st == "parse_fail":
            c["parse_fail"] += 1
        else:
            c["err"] += 1

        if bi % 1 == 0:
            elapsed = time.time() - t_start
            rate = bi / max(elapsed, 1e-6)
            eta = (n_total - bi) / max(rate, 1e-6)
            print(f"[{bi}/{n_total}] {mname} {item['video_id']} "
                  f"status={st} segs={result.get('n_segments')} "
                  f"dt={result.get('dt', 0.0):.1f}s | "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                  flush=True)

    # ---- 每模型 summary ----
    sharded_suffix = f".shard{args.shard_idx}" if sharded else ""
    for name, c in counters.items():
        out_dir = os.path.join(args.runs_root, name)
        os.makedirs(out_dir, exist_ok=True)
        sum_path = os.path.join(out_dir, f"summary{sharded_suffix}.json")
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump({
                "tag": name,
                "model_path": args.model_path,
                "categories": cats,
                "shard_idx": args.shard_idx,
                "shard_num": args.shard_num,
                "n_videos": c["n"],
                "n_ok": c["ok"],
                "n_err": c["err"],
                "n_parse_fail": c["parse_fail"],
                "n_skip": c["skip"],
                "n_segments_total": c["n_segments_total"],
            }, f, ensure_ascii=False, indent=2)

    print()
    print(f"[id_consistency] DONE in {time.time()-t_start:.0f}s")
    for name, c in counters.items():
        print(f"  [{name}] n={c['n']} ok={c['ok']} err={c['err']} "
              f"parse_fail={c['parse_fail']} skip={c['skip']} "
              f"segs={c['n_segments_total']}")


def run_single(args):
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    items = scan_videos(args.videos_root, cats)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    sharded = (args.shard_num > 1)
    if sharded:
        items = [it for i, it in enumerate(items) if i % args.shard_num == args.shard_idx]

    if not items:
        print("[id_consistency] nothing to run, exit.")
        return

    with open(args.system_prompt, "r", encoding="utf-8") as f:
        system_text = f.read().strip()

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

    if not args.model_path:
        args.model_path = getattr(runner, "model_path", "") or ""

    c = {"n": 0, "ok": 0, "err": 0, "skip": 0, "parse_fail": 0, "n_segments_total": 0}
    t_start = time.time()
    for bi, it in enumerate(items, start=1):
        result = transcribe_one_video(
            runner, system_text, it, args.out_dir,
            max_new_tokens=args.max_new_tokens,
            skip_existing=args.skip_existing,
        )
        c["n"] += 1
        c["n_segments_total"] += int(result.get("n_segments") or 0)
        st = result.get("status")
        if st == "ok":
            c["ok"] += 1
        elif st == "skip":
            c["skip"] += 1
        elif st == "parse_fail":
            c["parse_fail"] += 1
        else:
            c["err"] += 1
        elapsed = time.time() - t_start
        rate = bi / max(elapsed, 1e-6)
        eta = (len(items) - bi) / max(rate, 1e-6)
        print(f"[{bi}/{len(items)}] {it['video_id']} status={st} segs={result.get('n_segments')} "
              f"dt={result.get('dt', 0.0):.1f}s elapsed={elapsed:.0f}s eta={eta:.0f}s",
              flush=True)

    sharded_suffix = f".shard{args.shard_idx}" if sharded else ""
    sum_path = os.path.join(args.out_dir, f"summary{sharded_suffix}.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump({
            "tag": args.tag,
            "model_path": args.model_path,
            "videos_root": args.videos_root,
            "categories": cats,
            "shard_idx": args.shard_idx,
            "shard_num": args.shard_num,
            **c,
        }, f, ensure_ascii=False, indent=2)
    print()
    print(f"[id_consistency] DONE in {time.time()-t_start:.0f}s "
          f"n={c['n']} ok={c['ok']} err={c['err']} parse_fail={c['parse_fail']} "
          f"skip={c['skip']} segs={c['n_segments_total']}")


def main():
    ap = argparse.ArgumentParser()
    # 公共
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--model-path", required=False, default="",
                    help="local backend: Qwen3-Omni-30B-A3B-Instruct 权重目录 (必填); "
                         "api backend: 可选, 仅用于 summary 记录")
    ap.add_argument("--backend", choices=["local", "api"], default="local",
                    help="local = 本地加载 Qwen3-Omni 30B 权重; "
                         "api = 走 qwen35_api/OmniApiRunner (qwen3.5-omni-plus 平台)")
    ap.add_argument("--categories", default="speech",
                    help="逗号分隔类别, 默认 speech")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="多模型: 每个模型最多 N 条; 单模型: 总共 N 条")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-num", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0",
                    help="'auto' 或 'cuda:0' (DP 默认)")
    ap.add_argument("--skip-existing", action="store_true")
    # 多模型
    ap.add_argument("--models-tsv", default="")
    ap.add_argument("--runs-root", default="")
    ap.add_argument("--only-models", default="")
    # 单模型
    ap.add_argument("--videos-root", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tag", default="")

    args = ap.parse_args()
    if args.shard_num <= 0:
        ap.error("--shard-num must be >= 1")
    if not (0 <= args.shard_idx < args.shard_num):
        ap.error(f"--shard-idx must be in [0, {args.shard_num})")

    multi = bool(args.models_tsv)
    single = bool(args.videos_root)
    if multi and single:
        ap.error("--models-tsv and --videos-root are mutually exclusive")
    if not multi and not single:
        ap.error("must specify either --models-tsv (multi) or --videos-root (single)")
    if args.backend == "local" and not args.model_path:
        ap.error("--model-path is required when --backend=local")

    if multi:
        if not args.runs_root:
            ap.error("--runs-root is required in multi-model mode")
        run_multi(args)
    else:
        if not args.out_dir:
            ap.error("--out-dir is required in single-model mode")
        if not args.tag:
            args.tag = os.path.basename(args.out_dir.rstrip("/"))
        run_single(args)


if __name__ == "__main__":
    main()
