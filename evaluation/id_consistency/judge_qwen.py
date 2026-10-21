#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_qwen.py
=============

ID 一致性评判 (per-video 整段打分)。

流程：
1. 读 GT specs (speech.jsonl), 抽出每个 vid 的判官友好视图：
   - subjects: [{id, descriptor, gender, is_speaker}]
   - dialogues: 按 shot 顺序展平的 [{shot_idx, speaker_id, text}]
   - scene_setting: 全局场景描述 (string, 截断到 600 字)
2. 对每个 (model, vid):
   - 读 HYP transcripts/<vid>.json, 取 segments[*].{speaker, subject, text}
   - 调 OpenAI 兼容 LLM (Qwen3-VL-235B 或同款) 让它按 judge_prompt.md 打分
   - 严格要求 JSON 输出, 失败重试 (指数退避)
3. 落盘:
   - score_out/judge/<model>/<vid>.json   (含 raw + 解析后字段)
   - score_out/judge/<model>/_summary.csv (该 model 每 vid 一行)
   - score_out/judge/judge_summary.csv    (跨 model 汇总: subject_avg / line_avg / N)

LLM 服务: OpenAI 兼容接口 (vllm / sglang)。base_url / model 走 CLI 或
config/local.yaml (llm.qwen.base_url / llm.qwen.prompt_model)。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any
_REPO_ROOT = Path(__file__).resolve().parents[2]

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = THIS_DIR / "judge_prompt.md"
DEFAULT_SPECS = (_REPO_ROOT / "t2av-bench/specs/speech.jsonl")
DEFAULT_PROMPTS_FILE = (_REPO_ROOT / "t2av-bench/prompts/speech.json")
DEFAULT_RUNS_ROOT = THIS_DIR / "runs"
DEFAULT_OUT_DIR = THIS_DIR / "score_out" / "judge"
DEFAULT_MODELS = ["LTX-2.3", "Ovi", "daVinci-MagiHuman", "seedance", "wan2.7"]


def _read_existing_summary_rows(csv_path: Path) -> list[dict]:
    """读旧 judge_summary*.csv, 返回原始行 list[dict] (保序).

    用于 --append-models 追加模式: 保留其他模型的行原封不动.
    文件不存在或读失败 -> 返回 []; 不抛异常, 让追加模式变成全量写入.
    """
    if not csv_path.is_file():
        return []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            return [dict(r) for r in reader]
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 读旧汇总失败 {csv_path}: {e!r}", file=sys.stderr)
        return []


def _merge_append_rows(
    old_rows: list[dict],
    new_rows: list[dict],
    append_models: list[str],
    fieldnames: list[str],
) -> list[dict]:
    """把旧行 (old_rows) 中不属于 append_models 的行保留原值,
    再把本次算出来的 new_rows (只含 append_models 的模型) 覆写/追加进来.

    最终顺序: 旧表里其他模型顺序保持不变 -> 后面追加 append_models 里在旧表**不存在**
    的新模型 (若 append_models 里的模型已在旧表, 则**原地覆盖**行, 不改变位置).

    fieldnames 用于把缺失列补 "" (兼容旧 CSV 没有某些新列的情况).
    """
    append_set = set(append_models)
    new_by_model = {r["model"]: r for r in new_rows}

    merged: list[dict] = []
    seen_models: set[str] = set()

    for row in old_rows:
        m = row.get("model", "")
        if not m:
            continue
        if m in append_set:
            # 原地覆盖: 用新算的值
            if m in new_by_model:
                merged.append({k: new_by_model[m].get(k, "") for k in fieldnames})
                seen_models.add(m)
            # 若 append_models 里有 m 但 new_by_model 里没有 (理论上不会),
            # 就跳过这一行 (相当于清掉旧值)
        else:
            # 其他模型: 保持原值不动, 只按 fieldnames 补齐缺失列
            merged.append({k: row.get(k, "") for k in fieldnames})
            seen_models.add(m)

    # append_models 里在旧表不存在的新模型 -> 追加到末尾
    for m in append_models:
        if m in seen_models:
            continue
        if m in new_by_model:
            merged.append({k: new_by_model[m].get(k, "") for k in fieldnames})

    return merged


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_specs(path: Path) -> dict[str, dict]:
    """读 speech.jsonl, 返回 {vid: spec_obj}"""
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] 解析 specs 行失败: {e!r}", file=sys.stderr)
                continue
            vid = obj.get("id")
            if isinstance(vid, str) and vid:
                out[vid] = obj
    return out


def load_prompts_text(path: Path) -> dict[str, str]:
    """读 prompts/speech.json (一个 JSON 数组), 返回 {vid: prompt_text}"""
    out: dict[str, str] = {}
    if not path.is_file():
        print(f"[warn] prompts 文件不存在: {path}", file=sys.stderr)
        return out
    try:
        arr = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 解析 prompts 文件失败 {path}: {e!r}", file=sys.stderr)
        return out
    if not isinstance(arr, list):
        print(f"[warn] prompts 文件顶层不是数组: {path}", file=sys.stderr)
        return out
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        vid = obj.get("id")
        prompt = obj.get("prompt")
        if isinstance(vid, str) and vid and isinstance(prompt, str) and prompt.strip():
            out[vid] = prompt
    return out


def build_gt_view_v2(spec: dict, prompt_text: str) -> dict:
    """
    新版 GT 视图: 直接喂提示词原文 + 精简 subjects 表 (用作 S? -> 主体描述 的对照表)。
    返回字段:
      - video_id
      - shot_count: "single_shot" | "multi_shot"
      - prompt_text: 视频生成时使用的原始提示词 (可能含 [OVERALL]/[SHOT k/N] 标签)
      - subjects: [{id, descriptor, gender}]  -- 仅作主体对照表, 不包含 is_speaker
                  (画外音/旁白也是一个 subject; 判官需要这张表把 HYP 里的 S1/S2 对回 GT)
    注意: 这里**不**展开 dialogues, 让判官从 prompt_text 自己读取 "谁说了什么".
    """
    subjects: list[dict] = []
    for s in spec.get("subjects") or []:
        if not isinstance(s, dict):
            continue
        subjects.append(
            {
                "id": s.get("id"),
                "descriptor": s.get("descriptor"),
                "gender": s.get("gender"),
            }
        )
    return {
        "video_id": spec.get("id"),
        "shot_count": spec.get("shot_count"),
        "prompt_text": prompt_text,
        "subjects": subjects,
    }


def build_gt_view(spec: dict) -> dict:
    """从 spec 抽出判官友好的精简视图。"""
    subjects: list[dict] = []
    for s in spec.get("subjects") or []:
        if not isinstance(s, dict):
            continue
        subjects.append(
            {
                "id": s.get("id"),
                "descriptor": s.get("descriptor"),
                "gender": s.get("gender"),
                "is_speaker": s.get("is_speaker"),
            }
        )
    dialogues: list[dict] = []
    for shot in spec.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        shot_idx = shot.get("shot_idx")
        for d in shot.get("dialogue") or []:
            if not isinstance(d, dict):
                continue
            dialogues.append(
                {
                    "shot_idx": shot_idx,
                    "speaker_id": d.get("speaker_id"),
                    "text": d.get("text"),
                }
            )
    # 兜底: single_shot 类型的 spec 没有 shots[].dialogue, 台词放在 subjects[i].speech
    if not dialogues:
        for s in spec.get("subjects") or []:
            if not isinstance(s, dict):
                continue
            sp = s.get("speech")
            if isinstance(sp, str) and sp.strip():
                dialogues.append(
                    {
                        "shot_idx": 1,
                        "speaker_id": s.get("id"),
                        "text": sp.strip(),
                    }
                )
    # 自洽校正: 若某 subject 在 dialogues 里实际有台词, 强制把 is_speaker 修正为 True,
    # 避免 spec 中 subjects[i].is_speaker=False 与 shots[].dialogue 冲突时误导 LLM 判官
    speaker_ids_in_dialogue = {
        d.get("speaker_id") for d in dialogues if d.get("speaker_id")
    }
    for sub in subjects:
        if sub.get("id") in speaker_ids_in_dialogue and not sub.get("is_speaker"):
            sub["is_speaker"] = True
    scene_setting = spec.get("scene_setting") or ""
    if isinstance(scene_setting, str) and len(scene_setting) > 600:
        scene_setting = scene_setting[:600] + "..."
    return {
        "video_id": spec.get("id"),
        "scene_setting": scene_setting,
        "subjects": subjects,
        "dialogues": dialogues,
    }


def load_hyp(transcripts_dir: Path, vid: str) -> tuple[str, dict | None]:
    """
    读 HYP transcripts/<vid>.json, 返回 (status, hyp_view)
    status ∈ {"ok", "missing", "empty", "bad"}
    hyp_view: {video_id, segments:[{speaker,subject,text}]} 或 None
    """
    f = transcripts_dir / f"{vid}.json"
    if not f.is_file():
        return "missing", None
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] HYP 解析失败 {f}: {e!r}", file=sys.stderr)
        return "bad", None
    segs_raw = obj.get("segments")
    if not segs_raw:
        # 仍然返回一个空视图, 让判官有机会判定 1 分
        return "empty", {"video_id": vid, "segments": []}
    segs: list[dict] = []
    for s in segs_raw:
        if not isinstance(s, dict):
            continue
        segs.append(
            {
                "speaker": s.get("speaker"),
                "subject": s.get("subject"),
                "text": s.get("text"),
            }
        )
    return "ok", {"video_id": vid, "segments": segs}


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def call_judge(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    gt_view: dict,
    hyp_view: dict | None,
    hyp_status: str,
    temperature: float,
    max_tokens: int,
    max_retry: int,
    vid: str,
    backend: str = "qwen-openai",
) -> tuple[dict, str]:
    """
    调一次 LLM, 返回 (parsed_dict, raw_text). 强制 JSON, 失败重试.
    parsed_dict keys: subject_consistency, line_assignment, errors, comment

    backend:
      - "qwen-openai": 走 OpenAI 兼容 chat.completions (vllm/sglang/平台 OpenAI 网关).
        client 必须是 openai.OpenAI.
    """
    if hyp_status == "missing":
        # 直接给 1 分, 不调 API, 节省成本
        parsed = {
            "subject_consistency": 1,
            "line_assignment": 1,
            "errors": ["HYP 缺失或无法解析"],
            "comment": "HYP 转写文件缺失, 直接判最低分",
        }
        return parsed, ""
    if hyp_status == "bad":
        parsed = {
            "subject_consistency": 1,
            "line_assignment": 1,
            "errors": ["HYP 缺失或无法解析"],
            "comment": "HYP 转写文件无法解析, 直接判最低分",
        }
        return parsed, ""

    user_payload = (
        f"# GT (ground truth)\n"
        f"```json\n{json.dumps(gt_view, ensure_ascii=False, indent=2)}\n```\n\n"
        f"# HYP (model transcription)\n"
        f"```json\n{json.dumps(hyp_view, ensure_ascii=False, indent=2)}\n```\n\n"
        f"请按 system 中规定的 schema 输出 JSON, 不要任何额外文字。"
    )

    last_err: Exception | None = None
    raw_text = ""
    for attempt in range(1, max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_text = resp.choices[0].message.content or ""
            cleaned = _strip_code_fence(raw_text)
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise ValueError(f"LLM 返回非对象: {cleaned[:200]}")
            sc = data.get("subject_consistency")
            la = data.get("line_assignment")
            if not (isinstance(sc, int) and 1 <= sc <= 5):
                raise ValueError(f"subject_consistency 非法: {sc!r}")
            if not (isinstance(la, int) and 1 <= la <= 5):
                raise ValueError(f"line_assignment 非法: {la!r}")
            errors = data.get("errors") or []
            if not isinstance(errors, list):
                errors = [str(errors)]
            comment = data.get("comment") or ""
            if not isinstance(comment, str):
                comment = str(comment)
            return (
                {
                    "subject_consistency": sc,
                    "line_assignment": la,
                    "errors": errors,
                    "comment": comment,
                },
                raw_text,
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            print(
                f"[warn] vid={vid} 第 {attempt}/{max_retry} 次失败: {e!r}; "
                f"{wait}s 后重试  (raw[:200]={raw_text[:200]!r})",
                file=sys.stderr,
            )
            time.sleep(wait)

    raise RuntimeError(f"vid={vid} LLM 调用全部失败: {last_err!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ID 一致性 judge: per-video 让 LLM 给 subject/line 各打 1-5 分"
    )
    ap.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    ap.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE,
                    help="视频生成所用的提示词原文 JSON (prompts/speech.json), "
                         "默认开启时直接把 prompt 原文喂给 judge")
    ap.add_argument("--use-prompt-text", dest="use_prompt_text",
                    action="store_true", default=True,
                    help="GT 视图喂提示词原文 (默认开启)")
    ap.add_argument("--no-use-prompt-text", dest="use_prompt_text",
                    action="store_false",
                    help="回退到旧版 dialogues 结构化 GT 视图")
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    ap.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="逗号分隔的 model 名 (对应 runs/<model>/transcripts)",
    )
    ap.add_argument(
        "--append-models",
        type=str,
        default="",
        help="逗号/空格分隔的追加模型名. 若非空, 触发追加模式:\n"
             "  * 推理阶段只跑这些模型 (--models 参数在追加模式下被忽略);\n"
             "  * 汇总阶段读旧 judge_summary*.csv, 保留其他模型行原值不动,\n"
             "    只覆写/新增这些模型的行. 避免只跑新模型时把汇总表刷成\n"
             "    '只剩新模型' 的一行. 拼错的模型名会报错退出.",
    )
    ap.add_argument(
        "--llm-backend",
        type=str,
        default="qwen-openai",
        choices=["qwen-openai"],
        help="qwen-openai=OpenAI 兼容 chat.completions (本地 vLLM Qwen3-VL)",
    )
    ap.add_argument("--base-url", type=str, default=None)
    ap.add_argument("--model", type=str, default=None, help="LLM 模型名")
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout-sec", type=int, default=None)
    ap.add_argument("--max-retry", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 (vid, model) 组合, 0=不限")
    ap.add_argument("--ids", type=str, default="", help="逗号分隔的 vid 白名单")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="若 score_out/judge/<model>/<vid>.json 已存在则跳过",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="vid 维度并发 LLM 调用的线程数 (默认 1=串行). "
             "并发仅限 LLM 调用, IO/累计仍在主线程顺序处理; "
             "model 之间仍为顺序, 避免日志交错. "
             "适合 4-8.",
    )
    args = ap.parse_args()

    # ---- 读 system prompt ----
    if not args.prompt_file.is_file():
        print(f"[error] judge_prompt.md 不存在: {args.prompt_file}", file=sys.stderr)
        return 2
    system_prompt = args.prompt_file.read_text(encoding="utf-8")

    # ---- 读 specs ----
    if not args.specs.is_file():
        print(f"[error] specs 不存在: {args.specs}", file=sys.stderr)
        return 2
    specs = load_specs(args.specs)
    print(f"[info] specs: {args.specs} (共 {len(specs)} 条)")

    id_whitelist: set[str] = set()
    if args.ids.strip():
        id_whitelist = {x.strip() for x in args.ids.split(",") if x.strip()}

    # ---- 解析 --append-models (追加模式) ----
    append_models: list[str] = []
    if args.append_models.strip():
        raw = args.append_models.replace(",", " ").split()
        append_models = [m for m in raw if m]

    if append_models:
        # 追加模式: 只以 append_models 为工作集, 忽略 --models
        models = list(append_models)
        # 校验: 每个模型必须有 transcripts 目录, 兼容两种路径规则:
        # (a) <runs_root>/qwen3_omni/<m>/transcripts (当前 transcribe 输出)
        # (b) <runs_root>/<m>/transcripts (老工程结构)
        missing = [
            m for m in append_models
            if not (args.runs_root / "qwen3_omni" / m / "transcripts").is_dir()
            and not (args.runs_root / m / "transcripts").is_dir()
        ]
        if missing:
            print(
                f"[error] --append-models 里这些模型在 {args.runs_root} 下"
                f"找不到 transcripts/: {missing}",
                file=sys.stderr,
            )
            return 2
        print(f"[info] APPEND MODE  append_models={append_models}")
        print(f"[info] models: {models}  (--models 参数在追加模式下已被忽略)")
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        print(f"[info] models: {models}")
    print(f"[info] runs-root: {args.runs_root}")
    print(f"[info] out-dir  : {args.out_dir}")

    # ---- LLM client ----
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
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 读取 t2av_bench.config 失败, 仅依赖 CLI: {e!r}", file=sys.stderr)

    max_tokens = args.max_tokens or cfg_max_tokens or 1024
    timeout_sec = args.timeout_sec or cfg_timeout_sec or 600

    print(f"[info] llm-backend: {args.llm_backend}")

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

    print(f"[info] LLM base_url: {base_url}")
    print(f"[info] LLM model   : {llm_model}")
    print(f"[info] temperature={args.temperature} max_tokens={max_tokens} timeout={timeout_sec}s")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)

    # ---- 准备 GT view ----
    if args.use_prompt_text:
        prompts_text_map = load_prompts_text(args.prompts_file)
        print(f"[info] prompts: {args.prompts_file} (共 {len(prompts_text_map)} 条) -- 使用提示词原文做 GT")
        gt_views: dict[str, dict] = {}
        for vid, spec in specs.items():
            ptxt = prompts_text_map.get(vid)
            if isinstance(ptxt, str) and ptxt.strip():
                gt_views[vid] = build_gt_view_v2(spec, ptxt)
            else:
                print(f"[warn] vid={vid} 在 prompts 文件中找不到 prompt, 回退到旧版 dialogues 视图", file=sys.stderr)
                gt_views[vid] = build_gt_view(spec)
    else:
        print("[info] 使用旧版 build_gt_view (结构化 dialogues)")
        gt_views = {vid: build_gt_view(spec) for vid, spec in specs.items()}
    all_vids = sorted(gt_views.keys())
    if id_whitelist:
        all_vids = [v for v in all_vids if v in id_whitelist]
    print(f"[info] 待评 vid 数 = {len(all_vids)}")

    # vid -> shot_count ("single_shot" / "multi_shot" / "unknown")
    shot_count_map: dict[str, str] = {}
    for vid in all_vids:
        sc_val = (specs.get(vid) or {}).get("shot_count")
        if sc_val in ("single_shot", "multi_shot"):
            shot_count_map[vid] = sc_val
        else:
            shot_count_map[vid] = "unknown"
    n_single_total = sum(1 for v in all_vids if shot_count_map[v] == "single_shot")
    n_multi_total = sum(1 for v in all_vids if shot_count_map[v] == "multi_shot")
    n_unknown_total = sum(1 for v in all_vids if shot_count_map[v] == "unknown")
    print(
        f"[info] shot_count 分布: single={n_single_total} multi={n_multi_total} unknown={n_unknown_total}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 跨 model 汇总 ----
    overall_rows: list[dict] = []
    n_done = 0

    for model_name in models:
        print(f"\n=== model: {model_name} ===")
        model_out = args.out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        # transcribe 写入路径规则: 优先 runs_root/qwen3_omni/<model>/transcripts
        # (当前 id_consistency::transcribe 实际输出位置); 兼容旧路径
        # runs_root/<model>/transcripts (老工程结构).
        _omni_dir = args.runs_root / "qwen3_omni" / model_name / "transcripts"
        _flat_dir = args.runs_root / model_name / "transcripts"
        transcripts_dir = _omni_dir if _omni_dir.is_dir() else _flat_dir

        per_rows: list[dict] = []
        sc_scores: list[int] = []
        la_scores: list[int] = []
        sc_scores_single: list[int] = []
        la_scores_single: list[int] = []
        sc_scores_multi: list[int] = []
        la_scores_multi: list[int] = []
        n_missing = 0
        n_failed = 0

        # ---- 第一趟: 主线程顺序处理 cache hit / HYP missing, 这些不需 LLM ----
        # 收集需要实际调用 LLM 的 vid 列表 (保持 vid 字典顺序).
        pending: list[tuple[str, str, str, dict | None]] = []
        # 如果命中 --limit, 需要在提交阶段提前截断
        for vid in all_vids:
            if args.limit > 0 and (n_done + 1) > args.limit:
                print(f"[info] 命中 --limit={args.limit}, 停止提交")
                break
            n_done += 1
            shot_count_val = shot_count_map.get(vid, "unknown")
            out_file = model_out / f"{vid}.json"

            # cache hit
            if args.skip_existing and out_file.is_file():
                try:
                    cached = json.loads(out_file.read_text(encoding="utf-8"))
                    sc = cached.get("subject_consistency")
                    la = cached.get("line_assignment")
                    status = cached.get("hyp_status", "ok")
                    if status == "missing":
                        n_missing += 1
                        print(f"[skip] {model_name}/{vid} 缓存为 missing, 视频未生成, 跳过 (不计入平均)")
                        continue
                    if isinstance(sc, int) and isinstance(la, int):
                        per_rows.append(
                            {
                                "video_id": vid,
                                "shot_count": shot_count_val,
                                "hyp_status": status,
                                "subject_consistency": sc,
                                "line_assignment": la,
                                "errors": "; ".join(cached.get("errors") or []),
                                "comment": cached.get("comment", ""),
                            }
                        )
                        sc_scores.append(sc)
                        la_scores.append(la)
                        if shot_count_val == "single_shot":
                            sc_scores_single.append(sc)
                            la_scores_single.append(la)
                        elif shot_count_val == "multi_shot":
                            sc_scores_multi.append(sc)
                            la_scores_multi.append(la)
                        print(
                            f"[skip] {model_name}/{vid} 已存在 sc={sc} la={la} "
                            f"shot={shot_count_val}"
                        )
                        continue
                except Exception:
                    pass

            hyp_status, hyp_view = load_hyp(transcripts_dir, vid)
            if hyp_status == "missing":
                n_missing += 1
                print(f"[skip] {model_name}/{vid} HYP 缺失, 视频未生成, 跳过 (不计入平均)")
                continue

            pending.append((vid, shot_count_val, hyp_status, hyp_view))

        # ---- 第二趟: 并发调用 LLM (num_workers=1 时为串行) ----
        def _do_one(vid: str, shot_count_val: str, hyp_status: str, hyp_view: dict | None):
            """线程体: 只调用 call_judge, 不做 IO / 累计.

            返回 tuple, 由主线程统一落盘 / 累计.
            """
            try:
                parsed_, raw_ = call_judge(
                    client=client,
                    model=llm_model,
                    system_prompt=system_prompt,
                    gt_view=gt_views[vid],
                    hyp_view=hyp_view,
                    hyp_status=hyp_status,
                    temperature=args.temperature,
                    max_tokens=max_tokens,
                    max_retry=args.max_retry,
                    vid=vid,
                    backend=args.llm_backend,
                )
                return (vid, shot_count_val, hyp_status, parsed_, raw_, None)
            except Exception as e:  # noqa: BLE001
                return (vid, shot_count_val, hyp_status, None, "", e)

        results: list[tuple] = []
        nw = max(1, int(args.num_workers))
        if nw == 1 or len(pending) <= 1:
            for vid, sc_v, hs, hv in pending:
                results.append(_do_one(vid, sc_v, hs, hv))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            print(f"[info] model={model_name}: 并发 LLM 调用 num_workers={nw}, pending={len(pending)}")
            with ThreadPoolExecutor(max_workers=nw) as ex:
                fut_map = {
                    ex.submit(_do_one, vid, sc_v, hs, hv): vid
                    for vid, sc_v, hs, hv in pending
                }
                tmp: dict[str, tuple] = {}
                for fut in as_completed(fut_map):
                    res = fut.result()
                    tmp[res[0]] = res
                # 按提交顺序 (vid 字典顺序) 重排, 保证日志可读
                for vid, _, _, _ in pending:
                    if vid in tmp:
                        results.append(tmp[vid])

        # ---- 第三趟: 主线程统一落盘 / 累计 ----
        for vid, shot_count_val, hyp_status, parsed, raw, err in results:
            out_file = model_out / f"{vid}.json"
            if err is not None:
                n_failed += 1
                print(f"[error] {model_name}/{vid} judge 失败: {err!r}", file=sys.stderr)
                err_record = {
                    "video_id": vid,
                    "model": model_name,
                    "hyp_status": hyp_status,
                    "judge_failed": True,
                    "error": repr(err),
                }
                out_file.write_text(
                    json.dumps(err_record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                continue

            sc = parsed["subject_consistency"]
            la = parsed["line_assignment"]
            errors = parsed.get("errors") or []
            comment = parsed.get("comment", "")
            print(
                f"[ok ] {model_name}/{vid}  sc={sc}  la={la}  status={hyp_status}  "
                f"errors={errors}"
            )

            record = {
                "video_id": vid,
                "model": model_name,
                "shot_count": shot_count_val,
                "hyp_status": hyp_status,
                "subject_consistency": sc,
                "line_assignment": la,
                "errors": errors,
                "comment": comment,
                "raw_response": raw,
            }
            out_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            per_rows.append(
                {
                    "video_id": vid,
                    "shot_count": shot_count_val,
                    "hyp_status": hyp_status,
                    "subject_consistency": sc,
                    "line_assignment": la,
                    "errors": "; ".join(errors),
                    "comment": comment,
                }
            )
            if hyp_status != "missing":
                sc_scores.append(sc)
                la_scores.append(la)
                if shot_count_val == "single_shot":
                    sc_scores_single.append(sc)
                    la_scores_single.append(la)
                elif shot_count_val == "multi_shot":
                    sc_scores_multi.append(sc)
                    la_scores_multi.append(la)

        # ---- 写 model 级 csv ----
        per_csv = model_out / "_summary.csv"
        with per_csv.open("w", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(
                fp,
                fieldnames=[
                    "video_id",
                    "shot_count",
                    "hyp_status",
                    "subject_consistency",
                    "line_assignment",
                    "errors",
                    "comment",
                ],
            )
            w.writeheader()
            for r in per_rows:
                w.writerow(r)

        def _avg(lst: list[int]) -> float | None:
            return round(statistics.mean(lst), 2) if lst else None

        def _sum_avg(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return round(a + b, 2)

        sc_avg = _avg(sc_scores)
        la_avg = _avg(la_scores)
        total_avg = _sum_avg(sc_avg, la_avg)

        sc_avg_single = _avg(sc_scores_single)
        la_avg_single = _avg(la_scores_single)
        total_avg_single = _sum_avg(sc_avg_single, la_avg_single)

        sc_avg_multi = _avg(sc_scores_multi)
        la_avg_multi = _avg(la_scores_multi)
        total_avg_multi = _sum_avg(sc_avg_multi, la_avg_multi)

        overall_rows.append(
            {
                "model": model_name,
                "subject_avg": sc_avg,
                "line_avg": la_avg,
                "total_avg": total_avg,
                "n_all": len(sc_scores),
                "subject_avg_single": sc_avg_single,
                "line_avg_single": la_avg_single,
                "total_avg_single": total_avg_single,
                "n_single": len(sc_scores_single),
                "subject_avg_multi": sc_avg_multi,
                "line_avg_multi": la_avg_multi,
                "total_avg_multi": total_avg_multi,
                "n_multi": len(sc_scores_multi),
            }
        )
        print(
            f"[model={model_name}] all: sc={sc_avg} la={la_avg} total={total_avg} "
            f"(n={len(sc_scores)} missing={n_missing} failed={n_failed})"
        )
        print(
            f"[model={model_name}] single_shot: sc={sc_avg_single} la={la_avg_single} "
            f"total={total_avg_single} (n={len(sc_scores_single)})"
        )
        print(
            f"[model={model_name}] multi_shot : sc={sc_avg_multi} la={la_avg_multi} "
            f"total={total_avg_multi} (n={len(sc_scores_multi)})"
        )

    # ---- 写跨 model 总表 ----
    # 1) 全量 (兼容老格式)
    overall_csv = args.out_dir / "judge_summary.csv"
    overall_fields = [
        "model",
        "subject_avg",
        "line_avg",
        "total_avg",
        "n_all",
        "subject_avg_single",
        "line_avg_single",
        "total_avg_single",
        "n_single",
        "subject_avg_multi",
        "line_avg_multi",
        "total_avg_multi",
        "n_multi",
    ]
    if append_models:
        old_overall = _read_existing_summary_rows(overall_csv)
        overall_final = _merge_append_rows(
            old_overall, overall_rows, append_models, overall_fields
        )
        kept = [r["model"] for r in old_overall if r.get("model") not in append_models]
        print(
            f"[append:overall] {overall_csv}  "
            f"kept={kept}  refreshed/added={append_models}"
        )
    else:
        overall_final = overall_rows
    with overall_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=overall_fields)
        w.writeheader()
        for r in overall_final:
            w.writerow(r)

    # 2) 单镜头
    single_csv = args.out_dir / "judge_summary_single_shot.csv"
    single_fields = ["model", "subject_avg", "line_avg", "total_avg", "n"]
    single_new_rows = [
        {
            "model": r["model"],
            "subject_avg": r["subject_avg_single"],
            "line_avg": r["line_avg_single"],
            "total_avg": r["total_avg_single"],
            "n": r["n_single"],
        }
        for r in overall_rows
    ]
    if append_models:
        old_single = _read_existing_summary_rows(single_csv)
        single_final = _merge_append_rows(
            old_single, single_new_rows, append_models, single_fields
        )
        kept = [r["model"] for r in old_single if r.get("model") not in append_models]
        print(
            f"[append:single] {single_csv}  "
            f"kept={kept}  refreshed/added={append_models}"
        )
    else:
        single_final = single_new_rows
    with single_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=single_fields)
        w.writeheader()
        for r in single_final:
            w.writerow(r)

    # 3) 多镜头
    multi_csv = args.out_dir / "judge_summary_multi_shot.csv"
    multi_fields = ["model", "subject_avg", "line_avg", "total_avg", "n"]
    multi_new_rows = [
        {
            "model": r["model"],
            "subject_avg": r["subject_avg_multi"],
            "line_avg": r["line_avg_multi"],
            "total_avg": r["total_avg_multi"],
            "n": r["n_multi"],
        }
        for r in overall_rows
    ]
    if append_models:
        old_multi = _read_existing_summary_rows(multi_csv)
        multi_final = _merge_append_rows(
            old_multi, multi_new_rows, append_models, multi_fields
        )
        kept = [r["model"] for r in old_multi if r.get("model") not in append_models]
        print(
            f"[append:multi] {multi_csv}  "
            f"kept={kept}  refreshed/added={append_models}"
        )
    else:
        multi_final = multi_new_rows
    with multi_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=multi_fields)
        w.writeheader()
        for r in multi_final:
            w.writerow(r)

    print(f"\n[done] judge_summary.csv             -> {overall_csv}")
    print(f"[done] judge_summary_single_shot.csv -> {single_csv}")
    print(f"[done] judge_summary_multi_shot.csv  -> {multi_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
