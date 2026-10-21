# Vendored scoring module for evaluation/aggregate_18_metrics.py.
# Only SUMMARY_METRICS / build_score_summary are used by the aggregator;
# the index-building entry point (main) expects the full workspace layout.
"""构建可视化索引（配置驱动版）

将以下数据合并到一份 data/index.json：
1. t2av-bench/prompts/<cat>.json   →  spec_id → prompt 文本
2. t2av-bench/specs/<cat>.jsonl    →  spec_id → spec 完整字段
3. EVAL_DIMENSIONS 中声明的所有评测维度
4. 各模型 output 视频路径   →  spec_id → 视频文件 (整段) 与 shot 切片

输出结构（裁剪示例）::
{
  "categories": [...9 类...],
  "models": [...7 个模型...],
  "dimensions": [
    {"name": "audio_consistency", "label": "音频一致性", "shot_aware": true},
    {"name": "emotion_consistency", "label": "情绪一致性", "shot_aware": false},
    {"name": "id_consistency", "label": "身份一致性", "shot_aware": false},
    ...
  ],
  "items": {                       # 按类别分组
    "speech": [
      {
        "id": "speech_0001",
        "category": "speech",
        "shot_count": "multi_shot",
        "prompt": "...",
        "spec": {...},
        "videos": {"LTX-2.3": "/abs/path.mp4", ...},
        "evaluations": {                               # 统一字段
          "audio_consistency": {                       # shot_aware
            "LTX-2.3": {
              "shots": {
                "1": {
                  "clip_path": "...",
                  "questions": [
                    {"question": "...", "type": "yesno", "answer": "yes",
                     "score": null, "reason": "..."},
                  ]
                }
              }
            }
          },
          "emotion_consistency": {                     # 非 shot_aware
            "LTX-2.3": [
              {"question": "...", "type": "yesno", "answer": "yes", "reason": "..."}
            ]
          },
          "id_consistency": {                          # 非 shot_aware
            "LTX-2.3": [
              {"question": "subject_consistency（主体身份一致性）",
               "type": "score", "score": 2, "reason": "..."},
              {"question": "line_assignment（台词归属一致性）",
               "type": "score", "score": 2, "reason": "..."}
            ]
          }
        }
      }
    ]
  }
}

要新增评测维度只需：
1. 在 EVAL_DIMENSIONS 增加一项配置；
2. 若现有 loader 满足复用就直接用；否则写一个新 loader 函数返回
   {spec_id: {model: <payload>}}。
前端会自动循环 dimensions 渲染，无需改动。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------
WS = Path(os.environ.get("CAST_BENCH_WS", str(Path(__file__).resolve().parents[1])))
BENCH = WS / "t2av-bench"
EVAL = WS / "evaluation"
OUT_FILE = WS / "visualization" / "data" / "index.json"

CATEGORIES = [
    "speech",
    "non_verbal_vocal",
    "object",
    "action",
    "ambient",
    "animal",
    "instrument",
    "bgm",
    "sfx",
]

# --- 特殊类别: hard (困难泛化子集) ---
# 与常规 9 类不同：prompts 来自 t2av-bench/prompts/hard/<sub>.json，没有 specs、没有评测。
# 前端会用一个单独的 tab 展示，并根据自动发现的子类提供下拉过滤。
HARD_CATEGORY = "hard"

def _discover_hard_subs() -> list[str]:
    """扫描 t2av-bench/prompts/hard/*.json，返回子类名列表（不含扩展名），按字典序。"""
    hard_dir = BENCH / "prompts" / "hard"
    if not hard_dir.is_dir():
        return []
    return sorted(p.stem for p in hard_dir.glob("*.json") if p.is_file())

HARD_SUBS: list[str] = _discover_hard_subs()

# 模型名 → 视频根目录（直接到 <root>/<category>/<id>.mp4）
#
# 布局说明：
#   "flat"  -> root/<cat>/<id>.mp4
#   "540p"  -> root/540p/<cat>/<id>.mp4
#
# 生成规则：
#   1) 三个非 generation/ 布局的模型（LTX-2.3 / Ovi / daVinci-MagiHuman）显式列出；
#   2) generation/*/output 下的模型自动发现（新加模型无需改代码，刷新即出现），
#      顺序为：先按 GENERATION_PRIORITY 中已存在的项排列，其余按目录名字典序追加。

# ---- 固定布局（非 generation/ 目录）----
_FIXED_ROOTS: list[tuple[str, Path, str]] = [
    ("LTX-2.3",           WS / "LTX-2.3_test/output",           "flat"),
    ("LTX-2.5",           WS / "LTX-2.5_test/output",           "flat"),
    ("Ovi",               WS / "Ovi_test/output",               "flat"),
    ("daVinci-MagiHuman", WS / "daVinci-MagiHuman_test/output", "540p"),
]

# ---- generation/*/output 自动发现，主力优先顺序 ----
GENERATION_PRIORITY = [
    "seedance", "wan2.7", "kling-v3", "kling-v3-omni", "hunyuan_pretrain",
]

def _discover_generation_models() -> list[tuple[str, Path, str]]:
    """扫描 <WS>/generation/*/output，返回 [(name, root, 'flat'), ...]。

    顺序 = GENERATION_PRIORITY 中存在的项（保序） + 其余按目录名字典序追加。
    """
    gen_dir = WS / "generation"
    if not gen_dir.is_dir():
        return []
    found: dict[str, Path] = {}
    for child in gen_dir.iterdir():
        if not child.is_dir():
            continue
        out_dir = child / "output"
        if out_dir.is_dir():
            found[child.name] = out_dir
    ordered: list[str] = [m for m in GENERATION_PRIORITY if m in found]
    rest = sorted(m for m in found if m not in ordered)
    ordered.extend(rest)
    return [(m, found[m], "flat") for m in ordered]

MODEL_VIDEO_ROOTS: list[tuple[str, Path, str]] = (
    _FIXED_ROOTS + _discover_generation_models()
)

MODELS = [m for m, *_ in MODEL_VIDEO_ROOTS]

# ---------------------------------------------------------------------------
# 通用 IO 工具
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] bad jsonl line in {path}: {exc}")
    return out

def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] bad json {path}: {exc}")
        return None

# ---------------------------------------------------------------------------
# Prompt / Spec / 视频
# ---------------------------------------------------------------------------
def load_prompts() -> dict[str, dict]:
    """spec_id → {prompt, shot_count, category, compiled_by}（英文/默认版）

    包括常规 9 类（prompts/<cat>.json）与 hard 子集（prompts/hard/<sub>.json）。
    hard 子集里条目的 `category` 字段保留为原值（emtion/style/...），用于前端
    在 hard tab 内部按子类过滤；调用方遇到这些 id 时会另行归到 items["hard"] 下。
    """
    out: dict[str, dict] = {}
    for cat in CATEGORIES:
        data = _read_json(BENCH / "prompts" / f"{cat}.json") or []
        for rec in data:
            out[rec["id"]] = rec
    # hard 子集
    for sub in HARD_SUBS:
        data = _read_json(BENCH / "prompts" / "hard" / f"{sub}.json") or []
        for rec in data:
            rid = rec.get("id")
            if rid:
                out[rid] = rec
    return out

def load_prompts_zh() -> dict[str, dict]:
    """spec_id → 中文版 prompt 记录（可选；不存在则返回空 dict）。"""
    out: dict[str, dict] = {}
    zh_dir = BENCH / "prompts" / "zh"
    if not zh_dir.is_dir():
        return out
    for cat in CATEGORIES:
        data = _read_json(zh_dir / f"{cat}.json") or []
        for rec in data:
            if rec.get("id"):
                out[rec["id"]] = rec
    # hard 子集中文版（prompts/zh/hard/<sub>.json）——若存在则加载
    zh_hard = zh_dir / "hard"
    if zh_hard.is_dir():
        for sub in HARD_SUBS:
            data = _read_json(zh_hard / f"{sub}.json") or []
            for rec in data:
                if rec.get("id"):
                    out[rec["id"]] = rec
    return out

def load_specs() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cat in CATEGORIES:
        recs = _read_jsonl(BENCH / "specs" / f"{cat}.jsonl")
        for rec in recs:
            out[rec["id"]] = rec
    return out

def video_path_for(model: str, root: Path, layout: str, category: str, spec_id: str) -> str | None:
    """匹配给定模型 root 下 <category>/<spec_id>.mp4。

    `category` 允许含 "/"（如 "hard/emtion"）以支持多级目录。
    """
    cands: list[Path] = []
    cat_path = Path(category)  # 支持多级 category（"hard/emtion" → hard/emtion）
    if layout == "flat":
        cands.append(root / cat_path / f"{spec_id}.mp4")
    elif layout == "540p":
        cands.append(root / "540p" / cat_path / f"{spec_id}.mp4")
        cands.append(root / cat_path / f"{spec_id}.mp4")  # fallback
    for p in cands:
        if p.exists():
            return str(p)
    return None

def collect_videos(spec_ids_by_cat: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    """spec_id → {model: video_path}"""
    out: dict[str, dict[str, str]] = {}
    for cat, ids in spec_ids_by_cat.items():
        for sid in ids:
            out.setdefault(sid, {})
            for model, root, layout in MODEL_VIDEO_ROOTS:
                p = video_path_for(model, root, layout, cat, sid)
                if p:
                    out[sid][model] = p
    return out

# ---------------------------------------------------------------------------
# 通用 Loader 协议
#   入参：runs_dir 路径
#   返回：{spec_id: {judge_model: {gen_model: payload}}}
#   shot_aware=True  时 payload 形如  {"shots": {"<idx>": {"clip_path", "questions"}}}
#   shot_aware=False 时 payload 形如  [{"question","type","answer","score","reason",...}]
#
# 兼容两种磁盘结构：
#   旧扁平结构: runs_dir/<gen_model>/per_question.jsonl
#   新分级结构: runs_dir/<judge_model>/<gen_model>/per_question.jsonl
# 扁平结构会被收敛到 judge=DEFAULT_JUDGE_NAME。
# ---------------------------------------------------------------------------
DEFAULT_JUDGE_NAME = "default"


def _scan_per_question_dirs(runs_dir: Path) -> list[tuple[str, str, Path]]:
    """枚举 runs_dir 下所有 per_question.jsonl，返回 [(judge, gen_model, jsonl_path)]。

    识别策略（按 lvl1 子目录逐个独立判定，允许同一 runs_dir 下混合两种结构）：
      * 若 lvl1 里**没有任何子目录**但有 per_question.jsonl  → 扁平: judge=DEFAULT, gen=lvl1
      * 若 lvl1 里**有子目录**：把 lvl1 视作 judge_model，下钻到 lvl2 找 per_question.jsonl
        - 此情形下 lvl1/per_question.jsonl 即使存在也忽略（避免歧义），并打告警
      * 若 lvl1 既无子目录又无 per_question.jsonl → 跳过并提示
    """
    out: list[tuple[str, str, Path]] = []
    if not runs_dir.exists():
        return out
    for lvl1 in sorted(runs_dir.iterdir()):
        if not lvl1.is_dir():
            continue
        sub_dirs = [p for p in lvl1.iterdir() if p.is_dir()]
        flat_pq = lvl1 / "per_question.jsonl"

        if sub_dirs:
            # 分级结构：lvl1 = judge_model
            if flat_pq.exists():
                print(
                    f"[warn] {flat_pq} 与同层子目录共存，忽略该扁平 jsonl（结构歧义），"
                    f"按分级模式下钻识别。"
                )
            found_any = False
            for lvl2 in sorted(sub_dirs):
                pq = lvl2 / "per_question.jsonl"
                if pq.exists():
                    out.append((lvl1.name, lvl2.name, pq))
                    found_any = True
            if not found_any:
                print(f"[info] {lvl1} 下未发现 per_question.jsonl，跳过该 judge 目录")
        elif flat_pq.exists():
            # 扁平结构：lvl1 = gen_model
            out.append((DEFAULT_JUDGE_NAME, lvl1.name, flat_pq))
        else:
            print(f"[info] {lvl1} 既无子目录也无 per_question.jsonl，跳过")
    return out


def load_jsonl_per_question(
    runs_dir: Path,
    shot_aware: bool,
    judge_whitelist: list[str] | None = None,
    judge_rename: dict[str, str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """扫 runs_dir 下所有 per_question.jsonl，按 spec→judge→gen_model 三层组织。

    适配 audio_consistency / emotion_consistency / 任何写成 per_question.jsonl 的维度。

    Args:
        runs_dir: 扫描根目录
        shot_aware: 是否按 shot 对齐
        judge_whitelist: 仅保留名在列表里的 judge 目录; None = 不过滤。
            例如 ["qwen3_omni_split"] 可用于情感一致性: 只给前端展示
            split-modality 模式的结果, 忽略旧的融合模式 (qwen3_omni)。
        judge_rename: judge 名重命名映射。例如
            {"qwen3_omni_split": "qwen3_omni"} 可将 split 模式的结果对前端
            伪装成 qwen3_omni, 前端切换按钮/默认选项无需改动。
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    wl_set = set(judge_whitelist) if judge_whitelist else None
    rn = judge_rename or {}
    for judge_raw, model, jsonl_path in _scan_per_question_dirs(runs_dir):
        if wl_set is not None and judge_raw not in wl_set:
            continue
        judge = rn.get(judge_raw, judge_raw)
        for rec in _read_jsonl(jsonl_path):
            sid = rec.get("spec_id")
            if not sid:
                continue
            qa = {
                "question": rec.get("question"),
                "type": rec.get("question_type") or rec.get("type") or "yesno",
                "answer": rec.get("answer"),
                "score": rec.get("score"),
                "reason": rec.get("reason"),
                "associated_subject_id": rec.get("associated_subject_id"),
                "video_path": rec.get("video_path"),
            }
            judge_block = out.setdefault(sid, {}).setdefault(judge, {})
            if shot_aware:
                shot_idx = rec.get("shot_idx") or 1
                model_block = judge_block.setdefault(model, {"shots": {}})
                shot_block = model_block["shots"].setdefault(
                    str(shot_idx),
                    {"clip_path": rec.get("clip_path"), "questions": []},
                )
                if not shot_block.get("clip_path"):
                    shot_block["clip_path"] = rec.get("clip_path")
                shot_block["questions"].append(qa)
            else:
                judge_block.setdefault(model, []).append(qa)
    return out

# ---------------------------------------------------------------------------
# 题库加载 & 与 runs 数据的对齐
#
# 设计：
#   t2av-bench/questions/<dim>.jsonl 是「权威题面」。
#   每条 schema 至少：{spec_id, question, associated_subject_id?, shot_idx?,
#                       type?}。type 缺省为 "yesno"。
#   build_index 时按题库顺序展开题目，对每个模型把 runs 中已有的
#   answer / score / reason 填进去；模型没评测的题就留 None 占位。
#   题库不存在 → 该维度 fallback 到旧逻辑（只展示 runs 里的题）。
# ---------------------------------------------------------------------------
QuestionKey = tuple  # (question_text, associated_subject_id_or_None, shot_idx_or_None)


def _qkey(question: str | None, sub: Any, shot_idx: Any) -> QuestionKey:
    """题目唯一键：题面 + 关联主体 + shot_idx（None 表示 spec 级题）。"""
    return (
        (question or "").strip(),
        sub if sub not in ("", None) else None,
        int(shot_idx) if shot_idx not in ("", None) else None,
    )


def load_question_bank(path: Path) -> dict[str, list[dict[str, Any]]]:
    """读题库 JSONL → spec_id → 有序题目列表。

    每个题目元素带：
      {question, type, associated_subject_id, shot_idx?}
    """
    out: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return out
    for rec in _read_jsonl(path):
        sid = rec.get("spec_id")
        if not sid:
            continue
        q = {
            "question": rec.get("question"),
            "type": rec.get("question_type") or rec.get("type") or "yesno",
            "associated_subject_id": rec.get("associated_subject_id"),
        }
        if "shot_idx" in rec and rec.get("shot_idx") not in (None, ""):
            q["shot_idx"] = rec.get("shot_idx")
        out.setdefault(sid, []).append(q)
    return out


def _index_runs_by_qkey_flat(
    runs_block: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[QuestionKey, dict[str, Any]]]:
    """非 shot-aware：model → qkey → run 记录。"""
    out: dict[str, dict[QuestionKey, dict[str, Any]]] = {}
    for model, arr in (runs_block or {}).items():
        idx: dict[QuestionKey, dict[str, Any]] = {}
        for rec in arr or []:
            k = _qkey(rec.get("question"), rec.get("associated_subject_id"), None)
            # 同 key 多条时保留首条；尾部冲突可考虑追加，但当前数据未观察到。
            idx.setdefault(k, rec)
        out[model] = idx
    return out


def _index_runs_by_qkey_shot(
    runs_block: dict[str, dict[str, Any]],
) -> dict[str, dict[QuestionKey, dict[str, Any]]]:
    """shot-aware：model → qkey → run 记录（含 clip_path 信息保留在 rec 中）。"""
    out: dict[str, dict[QuestionKey, dict[str, Any]]] = {}
    for model, payload in (runs_block or {}).items():
        idx: dict[QuestionKey, dict[str, Any]] = {}
        shots = (payload or {}).get("shots") or {}
        for shot_idx, sh in shots.items():
            for rec in (sh.get("questions") or []):
                k = _qkey(
                    rec.get("question"),
                    rec.get("associated_subject_id"),
                    shot_idx,
                )
                # 把所属 shot 的 clip_path 也带回去，方便对齐时还原 shots 结构
                rec_with_shot = dict(rec)
                rec_with_shot.setdefault("clip_path", sh.get("clip_path"))
                rec_with_shot["_shot_idx"] = shot_idx
                idx.setdefault(k, rec_with_shot)
        out[model] = idx
    return out


def _qa_from_run(rec: dict[str, Any] | None, fallback_type: str) -> dict[str, Any]:
    """把一条 runs 记录折叠成展示用的 qa dict；rec=None 时返回空占位。"""
    if not rec:
        return {
            "answer": None,
            "score": None,
            "reason": None,
            "type": fallback_type,
        }
    return {
        "answer": rec.get("answer"),
        "score": rec.get("score"),
        "reason": rec.get("reason"),
        "type": rec.get("question_type") or rec.get("type") or fallback_type,
        "video_path": rec.get("video_path"),
    }


def align_with_bank(
    bank: dict[str, list[dict[str, Any]]],
    runs: dict[str, dict[str, Any]],
    shot_aware: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    """以题库为唯一权威：题面只来自题库，答案只从 runs 抽取后对齐。

    新签名（与 loader 输出结构匹配）：
      runs: {spec_id: {judge_model: {gen_model: payload}}}
      返回: {spec_id: {judge_model: {gen_model: payload}}}

    其中 payload 与原本一致：
      shot_aware=False → list[qa]
      shot_aware=True  → {"shots": {shot_idx: {"clip_path", "questions":[...]}}}

    实现：对每个 judge 单独跑一次内部 align（保持原有"按题库铺、缺答案占位"的行为）。
    """
    # 收集本维度涉及的所有 judge 与 gen_model
    all_judges: set[str] = set()
    all_models: set[str] = set()
    for sb in runs.values():
        for jm, gens in (sb or {}).items():
            all_judges.add(jm)
            all_models.update((gens or {}).keys())
    if not all_judges:
        all_judges = {DEFAULT_JUDGE_NAME}
    models_default = sorted(all_models) if all_models else list(MODELS)

    out: dict[str, dict[str, dict[str, Any]]] = {}

    if not shot_aware:
        for sid, qs in bank.items():
            spec_runs = runs.get(sid) or {}
            spec_out: dict[str, dict[str, Any]] = {}
            for judge in sorted(all_judges):
                judge_runs = spec_runs.get(judge) or {}
                run_idx = _index_runs_by_qkey_flat(judge_runs)
                spec_models = sorted(set(models_default) | set(judge_runs.keys()))
                judge_payload: dict[str, Any] = {}
                for model in spec_models:
                    idx = run_idx.get(model, {})
                    qa_list: list[dict[str, Any]] = []
                    for q in qs:
                        k = _qkey(q.get("question"), q.get("associated_subject_id"), None)
                        rec = idx.get(k)
                        qa = _qa_from_run(rec, q.get("type") or "yesno")
                        qa["question"] = q.get("question")
                        qa["associated_subject_id"] = q.get("associated_subject_id")
                        qa_list.append(qa)
                    judge_payload[model] = qa_list
                spec_out[judge] = judge_payload
            out[sid] = spec_out
        return out

    # ---- shot_aware ----
    for sid, qs in bank.items():
        spec_runs = runs.get(sid) or {}
        spec_out: dict[str, dict[str, Any]] = {}
        for judge in sorted(all_judges):
            judge_runs = spec_runs.get(judge) or {}
            run_idx = _index_runs_by_qkey_shot(judge_runs)
            spec_models = sorted(set(models_default) | set(judge_runs.keys()))
            judge_payload: dict[str, Any] = {}
            for model in spec_models:
                idx = run_idx.get(model, {})
                shots_payload: dict[str, dict[str, Any]] = {}
                for q in qs:
                    sk = str(q.get("shot_idx") or 1)
                    k = _qkey(q.get("question"), q.get("associated_subject_id"), int(sk))
                    rec = idx.get(k)
                    qa = _qa_from_run(rec, q.get("type") or "yesno")
                    qa["question"] = q.get("question")
                    qa["associated_subject_id"] = q.get("associated_subject_id")
                    shot_block = shots_payload.setdefault(
                        sk, {"clip_path": (rec or {}).get("clip_path"), "questions": []}
                    )
                    if not shot_block.get("clip_path") and rec:
                        shot_block["clip_path"] = rec.get("clip_path")
                    shot_block["questions"].append(qa)
                judge_payload[model] = {"shots": shots_payload}
            spec_out[judge] = judge_payload
        out[sid] = spec_out
    return out


def load_id_judge_dir(runs_dir: Path, shot_aware: bool) -> dict[str, dict[str, dict[str, Any]]]:
    """专用：id_consistency 的 judge JSON 目录加载器

    支持两种结构（自动识别）：
      A) 扁平   : runs_dir/<gen_model>/<spec_id>.json
      B) 分级   : runs_dir/<judge_model>/judge/<gen_model>/<spec_id>.json
                  （新结构，例如 score_out/qwen3_omni/judge/LTX-2.3/speech_0001.json）

    每文件字段: subject_consistency, line_assignment, errors[], comment
    转成 2 条"打分类"问答。

    返回 {spec_id: {judge_model: {gen_model: [qa, qa]}}}。
    扁平结构会被收敛到 judge=DEFAULT_JUDGE_NAME。
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    if not runs_dir.exists():
        return out

    def _parse_one(jf: Path, sid_default: str, judge: str, gen_model: str) -> None:
        if jf.name.startswith("_"):  # 跳过 _summary.csv 之类
            return
        data = _read_json(jf)
        if not isinstance(data, dict):
            return
        sid = data.get("video_id") or sid_default
        comment = (data.get("comment") or "").strip()
        errors = data.get("errors") or []
        err_str = ""
        if isinstance(errors, list) and errors:
            err_str = "；".join(str(e) for e in errors)
        reason_full = comment
        if err_str:
            reason_full = f"{comment} | 详细错误: {err_str}" if comment else err_str

        qs = []
        if "subject_consistency" in data:
            qs.append({
                "question": "subject_consistency（主体身份一致性 1-5）",
                "type": "score",
                "answer": None,
                "score": data.get("subject_consistency"),
                "reason": reason_full,
            })
        if "line_assignment" in data:
            qs.append({
                "question": "line_assignment（台词归属一致性 1-5）",
                "type": "score",
                "answer": None,
                "score": data.get("line_assignment"),
                "reason": reason_full,
            })
        if not qs:
            return
        out.setdefault(sid, {}).setdefault(judge, {}).setdefault(gen_model, []).extend(qs)

    for lvl1 in sorted(runs_dir.iterdir()):
        if not lvl1.is_dir():
            continue
        # 判定是 A 扁平结构还是 B 分级结构：B 的特征是下面有 "judge" 子目录
        judge_subdir = lvl1 / "judge"
        if judge_subdir.is_dir():
            # 分级：lvl1=judge_model，judge_subdir 下是 <gen_model>/<spec>.json
            judge_name = lvl1.name
            for gen_dir in sorted(judge_subdir.iterdir()):
                if not gen_dir.is_dir():
                    continue
                gen_model = gen_dir.name
                for jf in sorted(gen_dir.glob("*.json")):
                    _parse_one(jf, jf.stem, judge_name, gen_model)
        else:
            # 扁平：lvl1=gen_model，下面直接是 <spec>.json
            gen_model = lvl1.name
            for jf in sorted(lvl1.glob("*.json")):
                _parse_one(jf, jf.stem, DEFAULT_JUDGE_NAME, gen_model)
    return out

# ---------------------------------------------------------------------------
# 评测维度配置表（要加新维度只需在这里追加一项 + 必要时实现新 loader）
# ---------------------------------------------------------------------------
QUESTIONS_DIR = BENCH / "questions"

EVAL_DIMENSIONS: list[dict[str, Any]] = [
    {
        "name": "audio_consistency",
        "label": "音频一致性",
        "dir": EVAL / "audio_consistency" / "runs",
        "loader": load_jsonl_per_question,
        "shot_aware": True,
        # 题库文件（若不存在则 fallback 到 runs 自身的题）
        "question_bank": QUESTIONS_DIR / "audio_consistency_llm.jsonl",
    },
    {
        "name": "emotion_consistency",
        "label": "情绪一致性",
        "dir": EVAL / "emtion_consistency" / "runs",
        "loader": load_jsonl_per_question,
        "shot_aware": False,
        "question_bank": QUESTIONS_DIR / "emotion_consistency.jsonl",
        # 情感一致性以 split-modality 模式为权威管线 (见 emtion_consistency/SETUP.md):
        # 分模态送题避免了融合模式里“看图猜语气”的作弊. 前端仅展示 split 的结果,
        # 旧的 qwen3_omni (融合) 不进 index; 同时将 split 名对前端 rename 回 qwen3_omni,
        # 前端选项/按钮无需改动.
        "loader_kwargs": {
            "judge_whitelist": ["qwen3_omni_split", "qwen3.5_omni_split"],
            "judge_rename": {
                "qwen3_omni_split": "qwen3_omni",
                "qwen3.5_omni_split": "qwen3.5_omni",
            },
        },
    },
    {
        "name": "subject_consistency",
        "label": "主题一致性",
        "dir": EVAL / "subject_consistency" / "runs",
        "loader": load_jsonl_per_question,
        "shot_aware": False,
        "question_bank": QUESTIONS_DIR / "subject_consistency_llm.jsonl",
    },
    {
        "name": "style_consistency",
        "label": "风格一致性",
        "dir": EVAL / "style_consistency" / "runs",
        "loader": load_jsonl_per_question,
        "shot_aware": False,
        "question_bank": QUESTIONS_DIR / "style_consistency.jsonl",
    },
    {
        "name": "action_eval",
        "label": "动作评测",
        "dir": EVAL / "action_eval" / "runs",
        "loader": load_jsonl_per_question,
        # 题面带 shot_idx，按 shot 对齐（与 audio_consistency 同款）
        "shot_aware": True,
        "question_bank": QUESTIONS_DIR / "action_eval_llm.jsonl",
    },
    {
        "name": "id_consistency",
        "label": "身份一致性 (Qwen Judge)",
        "dir": EVAL / "id_consistency" / "score_out",
        "loader": load_id_judge_dir,
        "shot_aware": False,
        # id_consistency 是固定 2 项打分而非问答，不走题库
        "question_bank": None,
    },
]

def load_all_dimensions() -> dict[str, dict[str, dict[str, Any]]]:
    """name → spec_id → model → payload

    若维度声明了 question_bank 且题库存在：以题库为权威题面，把各模型 runs
    数据对齐进去；模型未评测的题留 None 占位。否则 fallback 到 loader
    自然产物。
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for d in EVAL_DIMENSIONS:
        name = d["name"]
        loader: Callable = d["loader"]
        shot_aware = bool(d.get("shot_aware", False))
        loader_kwargs = d.get("loader_kwargs") or {}
        try:
            runs_data = loader(d["dir"], shot_aware, **loader_kwargs)
        except TypeError:
            # loader 不支持 kwargs (第三方 loader), 降级成旧签名
            try:
                runs_data = loader(d["dir"], shot_aware)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] dimension '{name}' loader failed: {exc}")
                runs_data = {}
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] dimension '{name}' loader failed: {exc}")
            runs_data = {}

        bank_path: Path | None = d.get("question_bank")
        if bank_path is not None:
            # 题库优先模式：题面只来自题库，runs 仅供答案对齐
            if bank_path.exists():
                try:
                    bank = load_question_bank(bank_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] dimension '{name}' bank load failed ({bank_path}): {exc}")
                    bank = {}
                if bank:
                    data = align_with_bank(bank, runs_data, shot_aware)
                    n_bank = sum(len(v) for v in bank.values())
                    print(f"[info] dim {name}: bank={len(bank)} specs / {n_bank} questions  ({bank_path.name})")
                else:
                    data = {}
                    print(f"[info] dim {name}: bank empty, no questions emitted ({bank_path.name})")
            else:
                # 题库未提供：按 "暂无题" 处理，不再回退到 runs（避免新旧题同时展示）
                data = {}
                print(f"[info] dim {name}: bank not found → no questions emitted ({bank_path})")
        else:
            # 显式声明 question_bank=None：保持 loader 原产物（如 id_consistency 的固定打分项）
            data = runs_data

        n_specs = len(data)
        all_judges_set: set[str] = set()
        all_models_set: set[str] = set()
        for sb in data.values():
            for jm, gens in (sb or {}).items():
                all_judges_set.add(jm)
                all_models_set.update((gens or {}).keys())
        print(
            f"[info] dim {name}: {n_specs} specs, "
            f"{len(all_judges_set)} judges {sorted(all_judges_set)}, "
            f"{len(all_models_set)} gen-models  ({d['dir']})"
        )
        out[name] = data
    return out

# ---------------------------------------------------------------------------
# 维度统计：把 specs 里的全局/分类轴汇总成可视化用的分布数据
# ---------------------------------------------------------------------------
# 中文映射表（仅供前端展示，统计 key 仍用英文 enum，便于稳定排序/调试）
LABELS_GLOBAL: dict[str, dict[str, str]] = {
    "shot_count": {
        "single_shot": "单镜头",
        "multi_shot": "多镜头",
    },
    "n_shots": {
        "1":   "1 镜 (单镜头)",
        "2":   "2 镜",
        "3":   "3 镜",
        "4":   "4 镜",
        "5":   "5 镜",
        "6":   "6 镜",
        "7":   "7 镜",
        "8":   "8 镜",
        "9":   "9 镜",
        "10+": ">=10 镜 (旧)",
    },
    "visual_style": {
        "sci_fi": "科幻", "fantasy": "奇幻", "realistic": "写实",
        "punk": "朋克", "historical": "历史", "xuanhuan": "玄幻",
    },
    "audio_situation": {
        "single_subject": "单主体", "multi_subject": "多主体",
        "off_screen": "画外音",
    },
    "sound_complexity": {
        "single_source": "单声源", "multi_source": "多声源",
    },
    "video_type": {
        "2d_animation": "2D 动画", "3d_animation": "3D 动画",
        "game": "游戏", "film_tv": "影视剧",
        "vlog": "Vlog", "advertisement": "广告",
    },
    "camera_movement": {
        "static": "静止", "dolly_in": "推近", "dolly_out": "拉远",
        "tracking": "横移跟拍", "follow": "跟随",
        "orbit": "环绕", "none": "无（speech/action）",
    },
    "text_on_screen": {
        "with_text": "含画面文字", "no_text": "无画面文字",
    },
}

GLOBAL_AXIS_LABELS: dict[str, str] = {
    "shot_count": "镜头数",
    "n_shots": "镜头数(细分)",
    "visual_style": "画面风格",
    "audio_situation": "发声情况",
    "sound_complexity": "声源复杂度",
    "video_type": "视频类型",
    "camera_movement": "运镜方式",
    "text_on_screen": "画面文字",
}

CATEGORY_LABEL: dict[str, str] = {
    "speech": "语音", "non_verbal_vocal": "非语言人声",
    "object": "物体声", "action": "动作声",
    "ambient": "环境声", "animal": "动物声",
    "instrument": "乐器", "bgm": "背景音乐", "sfx": "特效音",
}

# 各类别下 L2 → 中文（“发声场景”列）
L2_LABEL: dict[str, dict[str, str]] = {
    "speech": {
        "daily_dialogue": "日常对话", "broadcast": "播报",
        "commentary": "解说", "voiceover": "旁白", "singing": "唱歌",
    },
    "non_verbal_vocal": {
        "physiological": "生理音", "mimicry": "拟声", "emotional": "情感音",
    },
    "animal": {
        "pet_vocalization": "宠物叫声", "wild_beast_roar": "野兽嘶吼",
        "bird_call": "鸟鸣", "insect_sound": "昆虫",
        "amphibian": "两栖", "livestock": "家畜",
    },
    "ambient": {
        "weather": "风雨雷电", "water_body": "江河湖海",
        "forest_field": "森林田野", "urban_street": "城市街景",
        "indoor_ambience": "室内氛围", "crowd_applause": "人群掌声",
    },
    "instrument": {
        "piano": "钢琴", "string_instrument": "弦乐",
        "wind_instrument": "管乐", "percussion": "打击乐",
        "ethnic_instrument": "民族乐器", "synthesizer": "电子合成",
    },
    "object": {
        "transportation": "交通场景", "weapon": "武器", "interaction": "互动声",
    },
    "bgm": {
        "film_score": "影视配乐", "game_music": "游戏配乐",
        "vlog_bgm": "日常视频配乐",
    },
    "sfx": {
        "transition_sfx": "转场音", "ui_notification": "UI 提示音",
        "exaggerated_sfx": "夸张音效", "sci_fi_sfx": "科幻音效",
        "ambience_sfx": "氛围音效", "magic_skill_sfx": "魔法技能",
    },
    "action": {
        "sports": "体育运动", "daily_activity": "日常活动",
    },
}

# 各类别下 L3 → 中文（"种类"列）
# 严格按照需求文档里的中文表来填，缺的就让前端 fallback 到英文 key
L3_LABEL: dict[str, dict[str, str]] = {
    "speech": {
        # 日常对话
        "interview": "访谈",
        "quarrel": "吵架",
        "casual_conversation": "日常交谈",
        "meeting": "会议",
        # 播报
        "platform_livestream": "平台直播",
        "news_broadcast": "新闻播报",
        "press_conference": "发布会",
        "lecture": "授课",
        "radio_broadcast": "广播",
        # 解说
        "tour_guide": "导游",
        "film_commentary": "电影解说",
        "sports_commentary": "赛事解说",
        # 旁白
        "film_voiceover": "影视旁白",
        "documentary_voiceover": "纪录片讲解",
        "documentary_narration": "纪录片讲解",
    },
    "non_verbal_vocal": {
        # 生理音
        "sneeze": "打喷嚏",
        "yawn": "打哈欠",
        "snore": "打呼噜",
        "cough": "咳嗽",
        "sniffle": "吸鼻涕",
        "fart": "放屁",
        "burp": "打嗝",
        # 拟声
        "beatbox": "Bbox",
        "whistle": "口哨",
        "tongue_click": "弹舌",
        # 情感音
        "laugh": "大笑",
        "sigh": "叹气",
        "cry": "哭泣",
        "cheer": "欢呼",
        "scream": "尖叫",
        "moan": "呻吟",
        "smirk": "冷笑",
        "smirk_laugh": "冷笑",
        # 数据中出现但不在表里的扩展项
        "hiccup": "打嗝（呃逆）",
        "gasp": "倒吸气",
    },
    "animal": {
        # 宠物叫声
        "cat": "猫", "dog": "狗",
        # 野兽嘶吼
        "tiger": "老虎", "bear": "熊", "wolf": "狼",
        # 鸟鸣
        "crow": "乌鸦", "cuckoo": "布谷", "eagle": "老鹰",
        # 昆虫
        "mosquito": "蚊子", "bee": "蜜蜂", "cricket": "蛐蛐", "cicada": "知了",
        # 两栖
        "frog": "蛙", "snake": "蛇",
        # 家畜
        "horse": "马", "chicken": "鸡", "duck": "鸭", "pig": "猪",
        "cow": "牛",
        # 数据中出现但不在表里的扩展项
        "elephant": "大象", "fox": "狐狸", "owl": "猫头鹰", "squirrel": "松鼠",
    },
    "ambient": {
        # 风雨雷电
        "wind": "风", "rain": "雨", "thunder": "雷", "lightning": "电",
        # 江河湖海
        "river": "江河", "lake": "湖", "sea": "海", "stream": "溪流",
        # 森林田野
        "forest": "森林", "field": "田野",
        # 城市街景
        "city_street": "城市街景", "traffic_ambience": "车流",
        # 室内氛围
        "indoor_ambience": "室内氛围",
        # 人群掌声
        "crowd": "人群", "applause": "掌声",
    },
    "instrument": {
        # 类别（如果 L3 直接复用乐器子类时）
        "piano": "钢琴",
        "violin": "小提琴", "cello": "大提琴", "guitar": "吉他",
        "flute": "长笛", "trumpet": "小号", "saxophone": "萨克斯",
        "drum": "鼓", "cymbal": "镲",
        "erhu": "二胡", "guzheng": "古筝", "pipa": "琵琶",
        "synthesizer": "电子合成",
    },
    "object": {
        # 交通场景
        "train": "火车", "car": "汽车", "bicycle": "自行车",
        "motorcycle": "摩托车", "airplane": "飞机", "ship": "轮船",
        "police_car": "警车",
        # 武器
        "knife": "刀", "arrow": "箭", "dart": "飞镖",
        "pistol": "手枪", "rifle": "步枪", "bomb": "炸弹", "missile": "导弹",
        # 互动声
        "collision": "碰撞", "friction": "摩擦",
        "tearing": "撕裂", "shattering": "破碎", "liquid": "液体",
        "tear": "撕裂", "shatter": "破碎",
        # 数据中出现但不在表里的扩展项
        "hovercar": "飞行汽车", "spark": "火花",
    },
    "bgm": {
        "song": "歌曲", "instrumental": "纯音乐", "rhythmic_beat": "节奏律动",
    },
    "sfx": {
        # 大多 sfx 的 L2/L3 同义；这里给 L3 fallback 用
        "transition_sfx": "转场音", "ui_notification": "UI 提示音",
        "exaggerated_sfx": "夸张音效", "sci_fi_sfx": "科幻音效",
        "ambience_sfx": "氛围音效", "magic_skill_sfx": "魔法技能",
    },
    "action": {
        # 体育运动（旧）
        "basketball": "打篮球", "soccer": "踢足球",
        "table_tennis": "打乒乓球", "billiards": "打台球",
        "diving": "跳水", "boxing": "拳击",
        # 体育运动（新增）
        "parkour": "跑酷",
        "boxing_match": "拳击比赛",
        "basketball_layup_contest": "篮球对抗上篮",
        "soccer_dribbling": "足球带球过人",
        "fancy_diving": "花式跳水",
        "gymnastics": "体操",
        "wrestling": "摔跤比赛",
        "swimming_race": "游泳比赛",
        # 日常活动（旧）
        "walking": "走路", "running": "奔跑", "eating": "进食",
        "knocking": "敲击", "clapping": "拍手",
        "typing": "敲击键盘", "fighting": "打斗",
        # 日常活动（新增）
        "unarmed_fighting": "空手打斗",
        "stick_fighting": "棍棒打斗",
        "martial_arts_fighting": "武术打斗",
        "sword_fighting": "刀剑互砍",
        "chopping_vegetables": "切菜",
        # 数据中出现但不在表里的扩展项（兼容历史脏值）
        "cooking": "烹饪", "opening_door": "开门",
        "shouting": "呼喊", "skateboarding": "滑板", "chopping": "剁/砍",
    },
}

# speech 专属
SPEECH_VOICE_LABEL = {
    "child_voice": "儿童声", "middle_aged_voice": "中年声",
    "elderly_voice": "老年声", "broadcast_tone": "播音腔",
    "male_voice": "男声", "female_voice": "女声",
    "robotic_voice": "机器人声",
}
SPEECH_LANG_LABEL = {
    "chinese": "汉语", "english": "英语", "japanese": "日语",
    "french": "法语", "russian": "俄语", "german": "德语",
    "korean": "韩语", "spanish": "西班牙语", "arabic": "阿拉伯语",
}
SPEECH_EMOTION_LABEL = {
    "angry": "愤怒", "joyful": "欢快", "sad": "悲伤",
    "surprised": "惊讶", "confused": "疑惑", "neutral": "无情感",
    "panicked": "惊慌",
}
SPEECH_EMOTION_CHANGE_LABEL = {"true": "有情感变化", "false": "无情感变化"}

INSTRUMENT_MODE_LABEL = {"solo": "独奏", "ensemble": "合奏"}
BGM_KIND_LABEL = {
    "song": "歌曲", "instrumental": "纯音乐", "rhythmic_beat": "节奏律动",
}


def _bucketize(items: list[dict], key_fn: Callable[[dict], Any]) -> list[dict[str, Any]]:
    """通用计数：返回 [{key, count, ratio, spec_ids}] 列表，按 count 降序。

    额外产出 spec_ids：命中该桶的 spec id 列表（每个 id 每桶最多计一次，
    避免同一 spec 的多个 audio_events 映射到同一 l3_kind 时被记两遍）。
    前端"维度统计弹层 → 点击某条目跳转详情页"依赖这个字段。
    """
    from collections import Counter
    cnt: Counter = Counter()
    ids_bucket: dict[Any, list[str]] = {}
    ids_seen: dict[Any, set[str]] = {}
    total = 0
    for it in items:
        v = key_fn(it)
        if v is None:
            continue
        sid = it.get("id")
        vals = v if isinstance(v, list) else [v]
        for vv in vals:
            cnt[vv] += 1
            total += 1
            if sid:
                seen = ids_seen.setdefault(vv, set())
                if sid not in seen:
                    seen.add(sid)
                    ids_bucket.setdefault(vv, []).append(sid)
    out = []
    for k, c in sorted(cnt.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        out.append({
            "key": k,
            "count": c,
            "ratio": (c / total) if total else 0.0,
            "spec_ids": ids_bucket.get(k, []),
        })
    return out


def _attach_label(rows: list[dict[str, Any]], label_map: dict[str, str]) -> list[dict[str, Any]]:
    for r in rows:
        r["label"] = label_map.get(r["key"], str(r["key"]))
    return rows


def compute_stats(items_by_cat: dict[str, list[dict]]) -> dict[str, Any]:
    """输入只含已组装、已带 prompt 的有效 items；输出可视化用统计结果。

    结构：
      {
        "total": int,
        "per_category_total": {cat: n},
        "global": [
          {"axis": "shot_count", "label": "镜头数",
           "rows": [{"key":"multi_shot","label":"多镜头","count":..,"ratio":..}, ...]},
          ...
        ],
        "per_category": {
          cat: {
            "label": "语音",
            "total": int,
            "axes": [
              {"axis":"l2_scene","label":"发声场景",
               "rows":[{key,label,count,ratio}, ...]},
              ...
            ],
          }
        }
      }
    """
    # 全部 spec（已过滤 prompt 为非空）
    all_items: list[dict] = []
    per_cat_count: dict[str, int] = {}
    for cat, lst in items_by_cat.items():
        per_cat_count[cat] = len(lst)
        all_items.extend(lst)

    def get_global(it: dict, axis: str):
        spec = it.get("spec") or {}
        if axis == "text_on_screen":
            tos = spec.get("text_on_screen") or []
            return "with_text" if tos else "no_text"
        if axis == "n_shots":
            sc = spec.get("shot_count") or it.get("shot_count") or ""
            shots = spec.get("shots") or []
            if sc == "single_shot":
                return "1"
            # multi_shot: 按 shots 数组长度归档（旧数据有 >=10 镜的归入 10+ 桶）
            n = len(shots) if isinstance(shots, list) else 0
            if n <= 0:
                # multi_shot 但 shots 为空（异常），归空
                return None
            if n >= 10:
                return "10+"
            return str(n)
        return spec.get(axis) or it.get(axis)

    global_axes_out = []
    for axis in GLOBAL_AXIS_LABELS:
        rows = _bucketize(all_items, lambda it, a=axis: get_global(it, a))
        rows = _attach_label(rows, LABELS_GLOBAL.get(axis, {}))
        # n_shots 轴按镜头数自然序（1,2,3,...,9,10+）展示，覆盖默认按 count 降序
        if axis == "n_shots":
            def _n_shots_order(row: dict) -> int:
                k = str(row.get("key", ""))
                if k == "10+":
                    return 10
                try:
                    return int(k)
                except ValueError:
                    return 99
            rows = sorted(rows, key=_n_shots_order)
        global_axes_out.append({
            "axis": axis,
            "label": GLOBAL_AXIS_LABELS[axis],
            "rows": rows,
        })

    per_cat_out: dict[str, Any] = {}
    for cat, lst in items_by_cat.items():
        if not lst:
            continue
        axes: list[dict[str, Any]] = []

        # L2：发声场景（双重过滤：parent_l1 必须等于本类，且 l2_id 必须是本类合法 enum）
        l2_whitelist = set((L2_LABEL.get(cat) or {}).keys())
        def _l2(it, _cat=cat, _wl=l2_whitelist):
            evs = (it.get("spec") or {}).get("audio_events") or []
            return [e.get("l2_id") for e in evs
                    if e.get("l2_id")
                    and (e.get("parent_l1") == _cat)
                    and (not _wl or e.get("l2_id") in _wl)]
        rows_l2 = _bucketize(lst, _l2)
        rows_l2 = _attach_label(rows_l2, L2_LABEL.get(cat, {}))
        axes.append({"axis": "l2_scene", "label": "发声场景 (L2)", "rows": rows_l2})

        # L3：种类
        # 口径：每条 spec 只统计"主事件"——即 audio_events 中第一个 parent_l1==cat
        # 且 l3_kind 合法的事件。原因：spec 生成器对 L3 做的是硬轮转/加权担平分配
        # （见 src/t2av_bench/spec/generator.py 的 _allocate_l3 / _allocate_l3_weighted），
        # 担平的对象就是每条 spec 的"主声源"。次事件由 LLM 自由发挥用作伴随声，
        # 会高频复用 friction/crow/sigh 等通用音；若把次事件也累计进 L3，会
        # 严重失衡。这里只看主事件，使统计口径与生成层的均匀性约束保持一致。
        l3_whitelist = set((L3_LABEL.get(cat) or {}).keys())
        def _l3(it, _cat=cat, _wl=l3_whitelist):
            evs = (it.get("spec") or {}).get("audio_events") or []
            for e in evs:
                k = e.get("l3_kind")
                if not k:
                    continue
                if e.get("parent_l1") != _cat:
                    continue
                if _wl and k not in _wl:
                    continue
                return [k]  # 只取首个匹配事件，每条 spec 计 1 次
            return []
        rows_l3 = _bucketize(lst, _l3)
        # L3 中文映射；缺失项自动 fallback 到英文 key
        rows_l3 = _attach_label(rows_l3, L3_LABEL.get(cat, {}))
        if rows_l3:
            axes.append({"axis": "l3_kind", "label": "种类 (L3)", "rows": rows_l3})

        # 各桶专属属性
        if cat == "speech":
            def _speech(field):
                return lambda it: ((it.get("spec") or {}).get("speech_attributes") or {}).get(field)

            axes.append({
                "axis": "voice_timbre", "label": "声线",
                "rows": _attach_label(_bucketize(lst, _speech("voice_timbre")), SPEECH_VOICE_LABEL),
            })
            axes.append({
                "axis": "language", "label": "语种",
                "rows": _attach_label(_bucketize(lst, _speech("language")), SPEECH_LANG_LABEL),
            })
            axes.append({
                "axis": "emotion", "label": "情感",
                "rows": _attach_label(_bucketize(lst, _speech("emotion")), SPEECH_EMOTION_LABEL),
            })

            def _emo_change(it):
                v = ((it.get("spec") or {}).get("speech_attributes") or {}).get("emotion_change")
                if v is None:
                    return None
                return "true" if v else "false"
            axes.append({
                "axis": "emotion_change", "label": "情感变化",
                "rows": _attach_label(_bucketize(lst, _emo_change), SPEECH_EMOTION_CHANGE_LABEL),
            })

        elif cat == "instrument":
            def _mode(it):
                return ((it.get("spec") or {}).get("instrument_attributes") or {}).get("playing_mode")
            axes.append({
                "axis": "playing_mode", "label": "演奏方式",
                "rows": _attach_label(_bucketize(lst, _mode), INSTRUMENT_MODE_LABEL),
            })

        elif cat == "bgm":
            def _kind(it):
                return ((it.get("spec") or {}).get("bgm") or {}).get("music_kind")
            axes.append({
                "axis": "music_kind", "label": "音乐种类",
                "rows": _attach_label(_bucketize(lst, _kind), BGM_KIND_LABEL),
            })

        per_cat_out[cat] = {
            "label": CATEGORY_LABEL.get(cat, cat),
            "total": len(lst),
            "axes": axes,
        }

    return {
        "total": len(all_items),
        "per_category_total": per_cat_count,
        "global": global_axes_out,
        "per_category": per_cat_out,
    }

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    print("[info] loading prompts ...")
    prompts = load_prompts()
    print(f"[info] prompts loaded: {len(prompts)}")
    prompts_zh = load_prompts_zh()
    print(f"[info] zh prompts loaded: {len(prompts_zh)}")

    print("[info] loading specs ...")
    specs = load_specs()
    print(f"[info] specs loaded: {len(specs)}")

    spec_ids_by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for sid, spec in specs.items():
        cat = spec.get("category")
        if cat in spec_ids_by_cat:
            spec_ids_by_cat[cat].append(sid)
    for c in spec_ids_by_cat:
        spec_ids_by_cat[c].sort()

    print("[info] scanning videos ...")
    videos = collect_videos(spec_ids_by_cat)
    n_v = sum(len(v) for v in videos.values())
    print(f"[info] videos found: {n_v}")

    print("[info] loading evaluation dimensions ...")
    dim_data = load_all_dimensions()

    # 组装最终结果
    items: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for cat in CATEGORIES:
        for sid in spec_ids_by_cat[cat]:
            spec = specs.get(sid, {})
            prompt_rec = prompts.get(sid, {})
            evaluations: dict[str, Any] = {}
            for d in EVAL_DIMENSIONS:
                payload = dim_data.get(d["name"], {}).get(sid)
                if payload:
                    evaluations[d["name"]] = payload
            zh_rec = prompts_zh.get(sid, {})
            item = {
                "id": sid,
                "category": cat,
                "shot_count": spec.get("shot_count", prompt_rec.get("shot_count")),
                "prompt": prompt_rec.get("prompt", ""),
                "prompt_zh": zh_rec.get("prompt", ""),
                "compiled_by": prompt_rec.get("compiled_by", ""),
                "spec": spec,
                "videos": videos.get(sid, {}),
                "evaluations": evaluations,
            }
            items[cat].append(item)

    # ---- 追加特殊类别: hard（困难泛化子集，无 specs、无评测）----
    # 每条 item 附带 sub_category = 原 category（emtion/style/...），
    # 前端在 hard tab 内部用一个子类下拉过滤；视频匹配走 <root>/hard/<sub>/<id>.mp4。
    hard_items: list[dict] = []
    if HARD_SUBS:
        # 收集 prompts 里 category ∈ HARD_SUBS 的条目
        for pid, prec in prompts.items():
            sub = prec.get("category")
            if sub not in HARD_SUBS:
                continue
            zh_rec = prompts_zh.get(pid, {})
            # 逐模型匹配 <root>/hard/<sub>/<pid>.mp4
            vids: dict[str, str] = {}
            for m, root, layout in MODEL_VIDEO_ROOTS:
                p = video_path_for(m, root, layout, f"hard/{sub}", pid)
                if p:
                    vids[m] = p
            hard_items.append({
                "id": pid,
                "category": HARD_CATEGORY,        # 让前端 tab 归到 "hard"
                "sub_category": sub,               # 前端子类下拉过滤 & badge 显示
                "shot_count": prec.get("shot_count"),
                "prompt": prec.get("prompt", ""),
                "prompt_zh": zh_rec.get("prompt", ""),
                "compiled_by": prec.get("compiled_by", ""),
                "spec": None,                      # 无 specs → 前端隐藏 Specs 栏
                "videos": vids,
                "evaluations": {},                 # 无评测 → 前端隐藏 QA 栏
            })
        # 按 (sub_category, id) 稳定排序
        hard_items.sort(key=lambda x: (x.get("sub_category") or "", x["id"]))
    items[HARD_CATEGORY] = hard_items

    # 维度元信息（供前端循环渲染）
    # judges_by_dim: 每个维度都汇总一份"出现过的 judge 模型"列表，前端用来生成切换按钮
    judges_by_dim: dict[str, list[str]] = {}
    for d in EVAL_DIMENSIONS:
        jset: set[str] = set()
        for sb in dim_data.get(d["name"], {}).values():
            jset.update((sb or {}).keys())
        judges_by_dim[d["name"]] = sorted(jset)

    dimensions_meta = [
        {
            "name": d["name"],
            "label": d.get("label", d["name"]),
            "shot_aware": bool(d.get("shot_aware", False)),
            "judges": judges_by_dim.get(d["name"], []),
        }
        for d in EVAL_DIMENSIONS
    ]

    # 维度统计（仅统计有 prompt 的条目，与前端展示口径一致）
    items_for_stats: dict[str, list[dict]] = {
        c: [it for it in items[c] if (it.get("prompt") or "").strip()]
        for c in CATEGORIES
    }
    print("[info] computing dimension stats ...")
    stats = compute_stats(items_for_stats)
    n_axes_global = len(stats["global"])
    n_cats_with_axes = len(stats["per_category"])
    print(f"[info] stats: total={stats['total']} "
          f"global_axes={n_axes_global} per_cat_blocks={n_cats_with_axes}")

    summary = {
        "categories": CATEGORIES,
        "models": MODELS,
        "dimensions": [d["name"] for d in EVAL_DIMENSIONS],
        "counts": {c: len(items[c]) for c in CATEGORIES},
        "video_counts": {
            m: sum(1 for sid in videos if m in videos[sid])
            for m in MODELS
        },
    }
    print("[info] summary:", json.dumps(summary, ensure_ascii=False))

    # 最终暴露给前端的 categories 列表：常规 9 类 + hard（若发现了 hard 条目）
    categories_out = list(CATEGORIES)
    if hard_items:
        categories_out.append(HARD_CATEGORY)
        print(f"[info] hard items: {len(hard_items)} "
              f"(subs={HARD_SUBS})")

    out = {
        "categories": categories_out,
        "models": MODELS,
        "dimensions": dimensions_meta,
        "items": items,
        "stats": stats,
        # hard 子类元信息（前端渲染子类下拉用；无则为空数组）
        "hard_subs": HARD_SUBS if hard_items else [],
        # 中文映射，供前端在"主要维度"面板等处复用
        "label_maps": {
            "global_axis": GLOBAL_AXIS_LABELS,
            "global_value": LABELS_GLOBAL,
            "category": CATEGORY_LABEL,
            "l2": L2_LABEL,
            "l3": L3_LABEL,
            "speech_voice": SPEECH_VOICE_LABEL,
            "speech_lang": SPEECH_LANG_LABEL,
            "speech_emotion": SPEECH_EMOTION_LABEL,
            "speech_emotion_change": SPEECH_EMOTION_CHANGE_LABEL,
            "instrument_mode": INSTRUMENT_MODE_LABEL,
            "bgm_kind": BGM_KIND_LABEL,
        },
    }

    # --- 分数汇总（跨模型跨测评点的总表；出错不阻塞主索引写入）---
    try:
        out["summary"] = build_score_summary()
        n_rows = len(out["summary"].get("rows", []))
        print(f"[info] score summary built: {n_rows} metric rows")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] build score summary failed: {exc}")
        out["summary"] = None

    # --- 按镜头数聚合的总成绩（📈 镜头趋势视图使用；出错不阻塞主索引写入）---
    if out.get("summary"):
        try:
            # 收集 id → shot_count_int 映射：single_shot→1, multi_shot→len(spec.shots)
            id2shot: dict[str, int] = {}
            id2cat: dict[str, str] = {}
            id2kind: dict[str, str] = {}
            for cat, arr in out.get("items", {}).items():
                for it in arr:
                    sid = it.get("id")
                    if not sid:
                        continue
                    id2cat[sid] = cat
                    sc_kind = it.get("shot_count") or ""
                    if sc_kind in ("single_shot", "multi_shot"):
                        id2kind[sid] = sc_kind
                    spec = it.get("spec") or {}
                    shots = spec.get("shots")
                    if sc_kind == "single_shot":
                        id2shot[sid] = 1
                    elif sc_kind == "multi_shot" and isinstance(shots, list) and len(shots) >= 2:
                        id2shot[sid] = len(shots)
                    # 其它情况（None / 0 / 空）不记录，视为无有效镜头数
            by_shot = build_score_summary_by_shot(id2shot)
            out["summary"]["by_shot"] = by_shot
            n_models = len(by_shot.get("models", []))
            n_shots  = len(by_shot.get("shot_counts", []))
            print(f"[info] by-shot summary built: {n_models} models × {n_shots} shot buckets")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] build by-shot summary failed: {exc}")
            out["summary"]["by_shot"] = None

        # --- 按采样比例聚合的总成绩（📈 趋势统计-采样图使用；single/multi 分开）---
        try:
            by_samp = build_score_summary_by_sampling(id2cat, id2kind)
            out["summary"]["by_sampling"] = by_samp
            n_ratios = len(by_samp.get("ratios", []))
            print(f"[info] by-sampling summary built: {len(by_samp.get('models',[]))} models × {n_ratios} ratios (single+multi)")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] build by-sampling summary failed: {exc}")
            out["summary"]["by_sampling"] = None

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    print(f"[done] index written: {OUT_FILE} ({size_mb:.2f} MB)")


# ===========================================================================
# 分数汇总（Score Summary）
# ---------------------------------------------------------------------------
# 目标：产出一份跨模型 × 跨测评点的总表，前端渲染出模板里的那张树形分数表。
#
# 数据来源全部是各评测点 runs/ 下已经算好的 compare_summary_*.csv。
# 每个测评点由 SUMMARY_METRICS 里的一条 dict 声明。
#
# 归一化：所有测评点值线性映射到 0-100，direction="down" 时反向；
# 总分：每测评点等权平均（"flat"）。
# 覆盖度门槛（方案 A）：某模型如果在任意一个大组下所有测评点都没跑，
# 则总分置 None，避免用只跑几项的数据得出误导性总分。
# ===========================================================================

# ---- 分数汇总参赛名单 ----
# 从 MODELS 派生，保持"主力优先"顺序（即 MODELS 中的相对顺序）。
#
# single（单镜头档）：不包含 hunyuan_pretrain（多镜头预训练模型，不参赛单镜头档）。
# multi（多镜头档）：不包含 LTX-2.3 / Ovi / daVinci-MagiHuman（非 generation/ 布局，多镜头档
#                   历史上不参赛）。
# 说明：kling-v3-omni 已按用户要求纳入分数汇总（即使目前评测数据为空也占列，
#      单元格显示 "-"），与 Gen-4.5 / happyhorse-1.1 / veo3.1 处理方式一致。
_SUMMARY_EXCLUDE_ALL   = set()
_SUMMARY_EXCLUDE_SINGLE = _SUMMARY_EXCLUDE_ALL | {"hunyuan_pretrain"}
_SUMMARY_EXCLUDE_MULTI  = _SUMMARY_EXCLUDE_ALL | {"LTX-2.3", "Ovi", "daVinci-MagiHuman"}

SUMMARY_SINGLE_MODELS = [m for m in MODELS if m not in _SUMMARY_EXCLUDE_SINGLE]
SUMMARY_MULTI_MODELS  = [m for m in MODELS if m not in _SUMMARY_EXCLUDE_MULTI]

SUMMARY_GROUPS = ["画面表现", "内容一致性", "音频表现和语义对齐"]


SUMMARY_METRICS: list[dict[str, Any]] = [
    # ------------------------- 画面表现 -------------------------
    {
        "g1": "画面表现", "g2": "视频表现", "metric": "视频美学",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "Q-Align/runs/compare_summary_single.csv",
        "src_multi":  "Q-Align/runs/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "",
    },
    {
        "g1": "画面表现", "g2": "运动表现", "metric": "运动人体异常检测",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "VBench2-Human_Anatomy/runs/compare_summary_single.csv",
        "src_multi":  "VBench2-Human_Anatomy/runs/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "",
    },
    {
        "g1": "画面表现", "g2": "运动表现", "metric": "运动合理性",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "action_eval/runs/qwen3_vl/compare_summary_single.csv",
        "src_multi":  "action_eval/runs/qwen3_vl/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score_",
    },
    {
        "g1": "画面表现", "g2": "文本表现", "metric": "文本渲染能力",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "ocr/score_out/reference_full_cpu/compare_summary_ocr_single.csv",
        "src_multi":  "ocr/score_out/reference_full_cpu/compare_summary_ocr_multi.csv",
        "fmt": "long", "model_col": "model", "value_col": "mean_score",
    },
    # ------------------------- 内容一致性 -------------------------
    {
        "g1": "内容一致性", "g2": "一致性保持", "metric": "背景一致性",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "consistency_segmented/runs/compare_summary_single_shot.csv",
        "src_multi":  "consistency_segmented/runs/compare_summary_multi_shot.csv",
        "fmt": "wide", "col_tpl": "bg_",
    },
    {
        "g1": "内容一致性", "g2": "一致性保持", "metric": "人物主体一致性",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "consistency_segmented/runs/compare_summary_single_shot.csv",
        "src_multi":  "consistency_segmented/runs/compare_summary_multi_shot.csv",
        "fmt": "wide", "col_tpl": "subj_",
    },
    {
        "g1": "内容一致性", "g2": "一致性保持", "metric": "面部一致性",
        "direction": "up", "range": (0.0, 100.0),
        "src_single": "facial_consistency/runs/compare_summary_single.csv",
        "src_multi":  "facial_consistency/runs/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score_",
    },
    {
        "g1": "内容一致性", "g2": "视觉语义对齐", "metric": "主体正确性",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "subject_consistency/runs/qwen3_vl/compare_summary_single.csv",
        "src_multi":  "subject_consistency/runs/qwen3_vl/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score_",
    },
    {
        "g1": "内容一致性", "g2": "视觉语义对齐", "metric": "镜头生成匹配",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": None,  # 仅多镜头
        "src_multi":  "TransNetV2/runs/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "",
    },
    {
        "g1": "内容一致性", "g2": "视觉语义对齐", "metric": "风格匹配",
        "direction": "up", "range": (0.0, 10.0),
        "src_single": "style_consistency/runs/qwen3_vl/compare_summary_single.csv",
        "src_multi":  "style_consistency/runs/qwen3_vl/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score10_",
    },
    # ------------------------- 音频表现和语义对齐 -------------------------
    {
        "g1": "音频表现和语义对齐", "g2": "音频表现", "metric": "音频美学PQ",
        "direction": "up", "range": (1.0, 10.0),
        "src_single": "audiobox-aesthetics/runs/compare_summary_PQ_single.csv",
        "src_multi":  "audiobox-aesthetics/runs/compare_summary_PQ_multi.csv",
        "fmt": "wide", "col_tpl": "",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "音画对齐", "metric": "声音-发声主体",
        "direction": "down", "range": (0.0, 1.0),
        "src_single": "Syncformer/runs/compare_summary_24-01-04T16-39-21_single.csv",
        "src_multi":  "Syncformer/runs/compare_summary_24-01-04T16-39-21_multi.csv",
        "fmt": "wide", "col_tpl": "", "col_suffix": "_cont",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "音画对齐", "metric": "语音-口型（偏移）",
        "direction": "down", "range": (0.0, 10.0),
        "src_single": "syncnet_python/runs/compare_summary_single.csv",
        "src_multi":  "syncnet_python/runs/compare_summary_multi.csv",
        "fmt": "long", "model_col": "model", "value_col": "mean_abs_offset_frames",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "音画对齐", "metric": "语音-口型（唇音匹配）",
        "direction": "up", "range": (0.0, 10.0),
        "src_single": "syncnet_python/runs/compare_summary_single.csv",
        "src_multi":  "syncnet_python/runs/compare_summary_multi.csv",
        "fmt": "long", "model_col": "model", "value_col": "mean_confidence",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "听觉语义对齐", "metric": "音频生成能力",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "audio_consistency/runs/qwen3_omni/compare_summary_single.csv",
        "src_multi":  "audio_consistency/runs/qwen3_omni/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "shot_norm_",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "听觉语义对齐", "metric": "情感匹配",
        "direction": "up", "range": (0.0, 1.0),
        # 情感匹配以 split-modality 模式为权威管线 (见 emtion_consistency/SETUP.md).
        "src_single": "emtion_consistency/runs/qwen3_omni_split/compare_summary_single.csv",
        "src_multi":  "emtion_consistency/runs/qwen3_omni_split/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score_",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "听觉语义对齐", "metric": "说话人ID对齐",
        "direction": "up", "range": (0.0, 10.0),
        "src_single": "id_consistency/score_out/judge/judge_summary_single_shot.csv",
        "src_multi":  "id_consistency/score_out/judge/judge_summary_multi_shot.csv",
        "fmt": "long", "model_col": "model", "value_col": "total_avg",
    },
    {
        "g1": "音频表现和语义对齐", "g2": "听觉语义对齐", "metric": "台词正确性",
        "direction": "up", "range": (0.0, 1.0),
        "src_single": "asr_eval/runs/compare_summary_single.csv",
        "src_multi":  "asr_eval/runs/compare_summary_multi.csv",
        "fmt": "wide", "col_tpl": "score_",
    },
]


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """通用小 CSV 读取，返回 (header, rows)。空/异常返回 ([], [])。"""
    import csv as _csv
    if not path.exists():
        return [], []
    try:
        with path.open("r", encoding="utf-8") as fh:
            all_rows = list(_csv.reader(fh))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] read csv failed {path}: {exc}")
        return [], []
    if not all_rows:
        return [], []
    return all_rows[0], all_rows[1:]


def _to_float(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:  # noqa: BLE001
        return None


def _read_wide_all_row(path: Path, col_tpl: str, col_suffix: str,
                       models: list[str], category_filter: str | None = None):
    """wide 格式：默认取 category=='__ALL__' 一行；

    若指定 category_filter（如 'speech'），则取该 category 那一行，
    按 col_tpl+<M>+col_suffix 取值。
    """
    header, rows = _read_csv_rows(path)
    values = {m: None for m in models}
    if not header:
        return values
    try:
        cat_idx = header.index("category")
    except ValueError:
        return values
    target_cat = category_filter if category_filter else "__ALL__"
    all_row = None
    for r in rows:
        if len(r) > cat_idx and r[cat_idx] == target_cat:
            all_row = r
            break
    if all_row is None:
        return values
    hmap = {h: i for i, h in enumerate(header)}
    for m in models:
        col = f"{col_tpl}{m}{col_suffix}"
        if col in hmap and hmap[col] < len(all_row):
            values[m] = _to_float(all_row[hmap[col]])
    return values


def _read_long_by_model(path: Path, model_col: str, value_col: str,
                        models: list[str]):
    """long 格式：每行一个 model；按 models 取 value_col。"""
    header, rows = _read_csv_rows(path)
    values = {m: None for m in models}
    if not header:
        return values
    hmap = {h: i for i, h in enumerate(header)}
    if model_col not in hmap or value_col not in hmap:
        return values
    m_i, v_i = hmap[model_col], hmap[value_col]
    for r in rows:
        if len(r) <= max(m_i, v_i):
            continue
        m = r[m_i].strip()
        if m in values:
            values[m] = _to_float(r[v_i])
    return values


def _normalize_100(raw, lo: float, hi: float, direction: str):
    """把原始值线性归一到 0-100；direction=='down' 时反向。"""
    if raw is None:
        return None
    if hi == lo:
        return 0.0
    x = (raw - lo) / (hi - lo)
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    if direction == "down":
        x = 1.0 - x
    return x * 100.0


def build_score_summary() -> dict:
    """扫描 SUMMARY_METRICS，产出结构化的分数汇总数据。

    返回：
      {
        "single_models": [...], "multi_models": [...],
        "groups": [...],
        "rows": [
          {"g1","g2","metric","direction","range":[lo,hi],
           "single": {model: raw|None},
           "multi":  {model: raw|None},
           "single_norm": {model: 0-100|None},
           "multi_norm":  {model: 0-100|None}}
        ],
        "group_scores": {"single": {g: {m: 0-100|None}}, "multi": {...}},
        "overall": {"flat_single": {m: v|None}, "flat_multi": {m: v|None}},
        "medals":  {"single": {m: "gold"|...|None}, "multi": {...}}
      }
    """
    single_models = list(SUMMARY_SINGLE_MODELS)
    multi_models  = list(SUMMARY_MULTI_MODELS)

    rows: list[dict[str, Any]] = []
    for spec in SUMMARY_METRICS:
        fmt = spec["fmt"]
        lo, hi = spec["range"]
        direction = spec["direction"]

        def _load(csv_rel, models):
            if not csv_rel:
                return {m: None for m in models}
            path = EVAL / csv_rel
            if fmt == "wide":
                return _read_wide_all_row(
                    path, spec.get("col_tpl", ""),
                    spec.get("col_suffix", ""), models,
                    category_filter=spec.get("category_filter"),
                )
            if fmt == "long":
                return _read_long_by_model(
                    path, spec.get("model_col", "model"),
                    spec["value_col"], models,
                )
            return {m: None for m in models}

        s_vals = _load(spec.get("src_single"), single_models)
        m_vals = _load(spec.get("src_multi"),  multi_models)

        s_norm = {k: _normalize_100(v, lo, hi, direction) for k, v in s_vals.items()}
        m_norm = {k: _normalize_100(v, lo, hi, direction) for k, v in m_vals.items()}

        rows.append({
            "g1": spec["g1"], "g2": spec["g2"], "metric": spec["metric"],
            "direction": direction, "range": [lo, hi],
            "single": s_vals, "single_norm": s_norm,
            "multi":  m_vals, "multi_norm":  m_norm,
        })

    def _mean_or_none(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    # 大组分数（做覆盖度检查用）
    def _group_scores(kind: str, models: list[str]):
        out = {g: {} for g in SUMMARY_GROUPS}
        nk = f"{kind}_norm"
        for g in SUMMARY_GROUPS:
            g_rows = [r for r in rows if r["g1"] == g]
            for m in models:
                vals = [r[nk].get(m) for r in g_rows]
                out[g][m] = _mean_or_none([v for v in vals if v is not None])
        return out

    gs_single = _group_scores("single", single_models)
    gs_multi  = _group_scores("multi",  multi_models)

    # 方案 A：模型必须在 3 个大组都有分数，才有资格出总分
    def _eligible(gs, m):
        return all(gs.get(g, {}).get(m) is not None for g in SUMMARY_GROUPS)

    def _flat(kind: str, models: list[str], gs):
        nk = f"{kind}_norm"
        out = {}
        for m in models:
            if not _eligible(gs, m):
                out[m] = None
                continue
            out[m] = _mean_or_none([r[nk].get(m) for r in rows])
        return out

    flat_single = _flat("single", single_models, gs_single)
    flat_multi  = _flat("multi",  multi_models,  gs_multi)

    # 奖牌：按 flat 排前 3
    def _medals(overall):
        ranked = sorted(
            [(m, v) for m, v in overall.items() if v is not None],
            key=lambda x: x[1], reverse=True,
        )
        names = ["gold", "silver", "bronze"]
        out = {m: None for m in overall}
        for i, (m, _v) in enumerate(ranked[:3]):
            out[m] = names[i]
        return out

    return {
        "single_models": single_models,
        "multi_models":  multi_models,
        "groups": SUMMARY_GROUPS,
        "rows": rows,
        "group_scores": {"single": gs_single, "multi": gs_multi},
        "overall": {"flat_single": flat_single, "flat_multi": flat_multi},
        "medals":  {"single": _medals(flat_single), "multi": _medals(flat_multi)},
    }


# ===========================================================================
# 按镜头数汇总（Score Summary by Shot Count）
# ---------------------------------------------------------------------------
# 目的：为前端"📈 镜头趋势"子视图产出：
#   {
#     "models":      [...],           # 参与折线的模型（保持配置顺序）
#     "shot_counts": [1..9],
#     "flat":        {model: {shot(str): 总分0-100|None}},  # 与"总分（测评点等权）"同口径
#     "coverage":    {model: {shot(str): {"n_metrics_used": int, "n_videos": int,
#                                        "n_precise": int, "n_fallback": int}}}
#   }
#
# 计算口径（严格对齐表格"总分"那行）：
#   1) 对每个 metric，优先从 per-video 数据（pv 声明）读到 {spec_id: raw_value}，
#      按 shot_count(int) 分桶求均值；
#   2) 若该 metric 没声明 pv 或 per-video 数据缺失 → 用 compare_summary_*.csv 的
#      __ALL__ 值广播（shot=1 用 src_single，shot>=2 用 src_multi）；
#   3) 用现有 _normalize_100 归一化到 0-100，direction=down 反向；
#   4) 覆盖度门槛（方案 A）：某 (model,shot) 若某个大组下所有 metric 都没数据 → 该点 None；
#   5) 最终"总分" = 该点 18 个 metric 归一化分等权平均（"测评点等权"）。
#
# 参与"镜头趋势"折线的模型固定为 seedance / wan2.7 / kling-v3（用户拍板）。
# ===========================================================================
BY_SHOT_MODELS = ["seedance", "wan2.7", "kling-v3"]
BY_SHOT_RANGE  = list(range(1, 10))  # 1..9

# 每个 metric 声明 per-video 数据源（pv）；结构：
#   {"kind": "csv_wide", "path_tpl": "<eval-rel>/runs/{model}/results.csv",
#    "score_col": "score", "video_col": "video_path"}
#   {"kind": "jsonl",    "path_tpl": "<eval-rel>/runs/{model}/per_video.jsonl",
#    "id_field": "spec_id", "score_field": "final_score"}
# 未在下表中出现的 metric（如 consistency_segmented / TransNetV2 / Syncformer /
# syncnet_python / audio_consistency / emtion_consistency / id_consistency /
# asr_eval / audiobox-aesthetics）自动走 fallback（__ALL__ 广播）。
_PV_MAP: dict[tuple[str, str], dict[str, Any]] = {
    ("画面表现", "视频美学"): {
        "kind": "csv_wide",
        "path_tpl": "Q-Align/runs/{model}/results.csv",
        "score_col": "score", "video_col": "video_path",
    },
    ("画面表现", "运动人体异常检测"): {
        "kind": "csv_wide",
        "path_tpl": "VBench2-Human_Anatomy/runs/{model}/results.csv",
        "score_col": "human_anatomy", "video_col": "video_path",
    },
    ("画面表现", "文本表现"): {
        "kind": "csv_wide",
        "path_tpl": "ocr/score_out/reference_full_cpu/{model}/results.csv",
        "score_col": None, "video_col": "video_path",  # 分数列自动挑选（非 video/folder 的第一列）
    },
    ("内容一致性", "面部一致性"): {
        "kind": "jsonl",
        "path_tpl": "facial_consistency/runs/{model}/per_video.jsonl",
        "id_field": "spec_id", "score_field": "final_score",
    },
    # 以下三个 metric 都需要走 Qwen3-VL judge 的 per_question.jsonl 聚合成 per-spec 均分，
    # 复杂度较高；这里不做精确 loader，交给 fallback（compare_summary __ALL__ 广播），
    # 数据准确度依赖 fallback 值 —— 精度足够画趋势图。
    # ("画面表现", "运动合理性"),
    # ("内容一致性", "主体正确性"),
    # ("内容一致性", "风格匹配"),
}

def _extract_spec_id_from_path(vp: str) -> str | None:
    """从视频路径中提取 spec_id：<...>/<cat>/<spec_id>.mp4 → spec_id"""
    if not vp:
        return None
    name = os.path.basename(str(vp))
    stem = os.path.splitext(name)[0]
    return stem or None

def _read_pv_csv_wide(path: Path, video_col: str, score_col: str | None) -> dict[str, float]:
    """CSV wide 格式 per-video: 返回 {spec_id: 原始分数}"""
    header, rows = _read_csv_rows(path)
    if not header:
        return {}
    hmap = {h: i for i, h in enumerate(header)}
    if video_col not in hmap:
        return {}
    # 分数列自动挑选（用户未指定时）：非 video_col / folder 的最后一列
    if score_col is None:
        cands = [h for h in header if h not in (video_col, "folder", "category")]
        if not cands:
            return {}
        score_col = cands[-1]
    if score_col not in hmap:
        return {}
    vi, si = hmap[video_col], hmap[score_col]
    out: dict[str, float] = {}
    for r in rows:
        if len(r) <= max(vi, si):
            continue
        sid = _extract_spec_id_from_path(r[vi])
        if not sid:
            continue
        v = _to_float(r[si])
        if v is not None:
            # 同一 spec_id 若出现多次（几乎不可能），取平均
            if sid in out:
                out[sid] = (out[sid] + v) / 2.0
            else:
                out[sid] = v
    return out

def _read_pv_jsonl(path: Path, id_field: str, score_field: str) -> dict[str, float]:
    """JSONL per-video: 每行一个 dict，返回 {spec_id: 原始分数}"""
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                sid = d.get(id_field)
                if not sid:
                    continue
                v = d.get(score_field)
                v = _to_float(v) if not isinstance(v, (int, float)) else float(v)
                if v is not None:
                    out[str(sid)] = v
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] read jsonl failed {path}: {exc}")
    return out

def _load_pv_for_metric(spec: dict, model: str) -> dict[str, float]:
    """按 spec 里的 pv 声明加载 per-video 分数，返回 {spec_id: raw}；找不到返回 {}"""
    key = (spec.get("g1"), spec.get("metric"))
    pv = _PV_MAP.get(key)
    if not pv:
        return {}
    path = EVAL / pv["path_tpl"].format(model=model)
    if pv["kind"] == "csv_wide":
        return _read_pv_csv_wide(path, pv["video_col"], pv.get("score_col"))
    if pv["kind"] == "jsonl":
        return _read_pv_jsonl(path, pv["id_field"], pv["score_field"])
    return {}

def build_score_summary_by_shot(id2shot: dict[str, int]) -> dict:
    """按镜头数 (1..9) 汇总每个模型的"测评点等权"总分。

    Args:
        id2shot: {spec_id: shot_count_int}，供 per-video 数据分桶使用。
    """
    models = list(BY_SHOT_MODELS)
    shots  = list(BY_SHOT_RANGE)

    # 兜底数据：读一遍所有 metric 的 __ALL__ single/multi 值，一次性拿出来
    # {("g1","metric"): {"single": {m:v}, "multi": {m:v}}}
    fallback_all: dict[tuple[str, str], dict[str, dict[str, float | None]]] = {}
    for spec in SUMMARY_METRICS:
        key = (spec["g1"], spec["metric"])
        fmt = spec["fmt"]
        def _load_all(csv_rel):
            if not csv_rel:
                return {m: None for m in models}
            path = EVAL / csv_rel
            if fmt == "wide":
                return _read_wide_all_row(
                    path, spec.get("col_tpl", ""),
                    spec.get("col_suffix", ""), models,
                    category_filter=spec.get("category_filter"),
                )
            if fmt == "long":
                return _read_long_by_model(
                    path, spec.get("model_col", "model"),
                    spec["value_col"], models,
                )
            return {m: None for m in models}
        fallback_all[key] = {
            "single": _load_all(spec.get("src_single")),
            "multi":  _load_all(spec.get("src_multi")),
        }

    # per-video 数据：{("g1","metric"): {model: {spec_id: raw}}}
    pv_data: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for spec in SUMMARY_METRICS:
        key = (spec["g1"], spec["metric"])
        if key not in _PV_MAP:
            continue
        pv_data[key] = {}
        for m in models:
            pv_data[key][m] = _load_pv_for_metric(spec, m)

    # 汇总统计
    # metric_norm[(g1,metric)][model][shot] = 归一化后 0-100（或 None）
    # metric_meta：记录该 (model,shot) 用了精确 pv 还是 fallback，用了几条视频
    metric_norm: dict[tuple[str, str], dict[str, dict[int, float | None]]] = {}
    coverage_precise: dict[str, dict[int, int]] = {m: {s: 0 for s in shots} for m in models}
    coverage_fallback: dict[str, dict[int, int]] = {m: {s: 0 for s in shots} for m in models}
    n_videos_used: dict[str, dict[int, int]] = {m: {s: 0 for s in shots} for m in models}

    for spec in SUMMARY_METRICS:
        key = (spec["g1"], spec["metric"])
        lo, hi = spec["range"]
        direction = spec["direction"]
        metric_norm[key] = {m: {s: None for s in shots} for m in models}

        pv_bag = pv_data.get(key, {})
        fb_bag = fallback_all.get(key, {"single": {}, "multi": {}})

        for m in models:
            pv_scores = pv_bag.get(m, {})  # {spec_id: raw}
            # 按 shot_count 分桶
            per_shot_raws: dict[int, list[float]] = {s: [] for s in shots}
            for sid, raw in pv_scores.items():
                sh = id2shot.get(sid)
                if sh is None or sh not in per_shot_raws:
                    continue
                if raw is None:
                    continue
                per_shot_raws[sh].append(float(raw))

            for s in shots:
                raws = per_shot_raws[s]
                if raws:
                    mean_raw = sum(raws) / len(raws)
                    metric_norm[key][m][s] = _normalize_100(mean_raw, lo, hi, direction)
                    coverage_precise[m][s] += 1
                    n_videos_used[m][s] += len(raws)
                else:
                    # 走 fallback：shot=1 用 single，shot>=2 用 multi
                    fb_kind = "single" if s == 1 else "multi"
                    fb_raw = (fb_bag.get(fb_kind) or {}).get(m)
                    if fb_raw is not None:
                        metric_norm[key][m][s] = _normalize_100(fb_raw, lo, hi, direction)
                        coverage_fallback[m][s] += 1

    # 按大组汇总 → 覆盖度门槛 → 测评点等权总分
    groups = SUMMARY_GROUPS
    metrics_of_group = {g: [(sp["g1"], sp["metric"]) for sp in SUMMARY_METRICS if sp["g1"] == g] for g in groups}
    all_keys = [(sp["g1"], sp["metric"]) for sp in SUMMARY_METRICS]

    def _mean_or_none(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    flat: dict[str, dict[str, float | None]] = {m: {} for m in models}
    coverage: dict[str, dict[str, dict[str, int]]] = {m: {} for m in models}

    for m in models:
        for s in shots:
            # 大组分（覆盖度检查用）
            group_ok = True
            for g in groups:
                vals = [metric_norm[k][m][s] for k in metrics_of_group[g]]
                if _mean_or_none(vals) is None:
                    group_ok = False
                    break
            if not group_ok:
                flat[m][str(s)] = None
            else:
                all_vals = [metric_norm[k][m][s] for k in all_keys]
                flat[m][str(s)] = _mean_or_none(all_vals)
            n_used = sum(1 for k in all_keys if metric_norm[k][m][s] is not None)
            coverage[m][str(s)] = {
                "n_metrics_used": n_used,
                "n_precise": coverage_precise[m][s],
                "n_fallback": coverage_fallback[m][s],
                "n_videos": n_videos_used[m][s],
            }

    return {
        "models": models,
        "shot_counts": shots,
        "flat": flat,
        "coverage": coverage,
    }


# ===========================================================================
# 按采样比例汇总（Score Summary by Sampling Ratio）
# ---------------------------------------------------------------------------
# 目的：为前端"📈 趋势统计 - 采样图"产出：观察 25/50/75/100% 采样比例下总分的稳定性。
#
# 计算口径（与表格"总分（测评点等权）"严格同源，单镜头/多镜头分开算）：
#   1) 把全部 spec_id 拆成 single_shot / multi_shot 两个池；
#   2) 每个池内，按 category 独立随机采样 x% 的 spec_id（固定 seed 保可复现）；
#      100% = 全量池（无随机性），此时结果应等同于表格 overall.flat_single /
#      overall.flat_multi；
#   3) 单镜头算分：只用采样到的 single_shot 集合，per-video loader 只取这些
#      spec 求均值，fallback 只用 src_single 的 __ALL__（不是混合了）；
#      多镜头算分：对称。
#   4) 归一化 → 3 大组覆盖度门槛 → 该 kind 参与的 metric 等权平均。
#      注意：某些 metric 只在 multi/single 有 src（如"镜头生成匹配"只在多镜头）；
#      该 kind 下缺 src 的 metric 直接跳过、不参与该 kind 的等权平均。
#
# 参与折线的模型固定为 seedance / wan2.7 / kling-v3（与镜头趋势保持一致）。
# ===========================================================================
BY_SAMPLING_MODELS  = ["seedance", "wan2.7", "kling-v3"]
BY_SAMPLING_RATIOS  = [0.25, 0.50, 0.75, 1.00]
BY_SAMPLING_SEED    = 42

def build_score_summary_by_sampling(id2cat: dict[str, str], id2kind: dict[str, str]) -> dict:
    """按采样比例汇总每个模型的"测评点等权"总分，单/多镜头分开。

    Args:
        id2cat:  {spec_id: category}，用于按 category 独立采样。
        id2kind: {spec_id: "single_shot"|"multi_shot"}，用于拆两个采样池。
    """
    import random as _random

    models = list(BY_SAMPLING_MODELS)
    ratios = list(BY_SAMPLING_RATIOS)

    # --- 一次性加载：per-video 数据 & fallback __ALL__ 值（避免每个比例重复读盘）---
    fallback_all: dict[tuple[str, str], dict[str, dict[str, float | None]]] = {}
    for spec in SUMMARY_METRICS:
        key = (spec["g1"], spec["metric"])
        fmt = spec["fmt"]
        def _load_all(csv_rel):
            if not csv_rel:
                return {m: None for m in models}
            path = EVAL / csv_rel
            if fmt == "wide":
                return _read_wide_all_row(
                    path, spec.get("col_tpl", ""),
                    spec.get("col_suffix", ""), models,
                    category_filter=spec.get("category_filter"),
                )
            if fmt == "long":
                return _read_long_by_model(
                    path, spec.get("model_col", "model"),
                    spec["value_col"], models,
                )
            return {m: None for m in models}
        fallback_all[key] = {
            "single": _load_all(spec.get("src_single")),
            "multi":  _load_all(spec.get("src_multi")),
        }

    pv_data: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for spec in SUMMARY_METRICS:
        key = (spec["g1"], spec["metric"])
        if key not in _PV_MAP:
            continue
        pv_data[key] = {}
        for m in models:
            pv_data[key][m] = _load_pv_for_metric(spec, m)

    # --- 按 kind 拆两个池，池内再按 category 组织 spec_id ---
    from collections import defaultdict as _dd
    pool_ids: dict[str, dict[str, list[str]]] = {
        "single_shot": _dd(list),
        "multi_shot":  _dd(list),
    }
    for sid, cat in id2cat.items():
        kind = id2kind.get(sid)
        if kind not in pool_ids:
            continue
        pool_ids[kind][cat].append(sid)
    # 排序保证可复现
    for kind in pool_ids:
        for cat in pool_ids[kind]:
            pool_ids[kind][cat].sort()

    def _sample_ids_of(kind: str, ratio: float, rng: _random.Random) -> set[str]:
        """在指定 kind 的池内、按 category 独立采样"""
        cat2ids = pool_ids[kind]
        if ratio >= 1.0:
            return {sid for ids in cat2ids.values() for sid in ids}
        out = set()
        for cat, ids in cat2ids.items():
            if not ids:
                continue
            n = max(1, int(round(len(ids) * ratio)))
            out.update(rng.sample(ids, n))
        return out

    groups = SUMMARY_GROUPS
    metrics_of_group = {g: [(sp["g1"], sp["metric"]) for sp in SUMMARY_METRICS if sp["g1"] == g] for g in groups}
    all_keys = [(sp["g1"], sp["metric"]) for sp in SUMMARY_METRICS]
    # 每个 metric 在 single/multi 下是否有 src（无则该 kind 下跳过）
    has_src = {
        (sp["g1"], sp["metric"]): {
            "single_shot": bool(sp.get("src_single")),
            "multi_shot":  bool(sp.get("src_multi")),
        }
        for sp in SUMMARY_METRICS
    }

    def _mean_or_none(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def _score_for_sample(sample_ids: set[str], kind: str) -> dict[str, float | None]:
        """给定采样集合 + kind(single_shot/multi_shot)，返回 {model: 该 kind 下总分|None}"""
        fb_kind = "single" if kind == "single_shot" else "multi"
        metric_norm: dict[tuple[str, str], dict[str, float | None]] = {}
        for spec in SUMMARY_METRICS:
            key = (spec["g1"], spec["metric"])
            # 该 kind 无 src → 直接标记为 skip（None）
            if not has_src[key][kind]:
                metric_norm[key] = {m: None for m in models}
                continue
            lo, hi = spec["range"]
            direction = spec["direction"]
            metric_norm[key] = {}
            pv_bag = pv_data.get(key)
            fb_bag = (fallback_all.get(key) or {}).get(fb_kind) or {}
            for m in models:
                v_norm = None
                if pv_bag is not None:
                    # 精确 metric：按采样集合筛选 per-video 分数求均值
                    pv_scores = pv_bag.get(m, {})
                    raws = [v for sid, v in pv_scores.items()
                            if sid in sample_ids and v is not None]
                    if raws:
                        v_norm = _normalize_100(sum(raws) / len(raws), lo, hi, direction)
                    # 精确 metric 若采样后没数据 → 走该 kind 的 fallback 保住覆盖度
                    if v_norm is None:
                        fb = fb_bag.get(m)
                        if fb is not None:
                            v_norm = _normalize_100(fb, lo, hi, direction)
                else:
                    # fallback metric：用该 kind 的 __ALL__（对采样不敏感）
                    fb = fb_bag.get(m)
                    if fb is not None:
                        v_norm = _normalize_100(fb, lo, hi, direction)
                metric_norm[key][m] = v_norm

        result: dict[str, float | None] = {}
        for m in models:
            # 大组覆盖度门槛（该 kind 参与的 metric 才检查）
            group_ok = True
            for g in groups:
                valid_metrics = [k for k in metrics_of_group[g] if has_src[k][kind]]
                if not valid_metrics:
                    group_ok = False
                    break
                vals = [metric_norm[k][m] for k in valid_metrics]
                if _mean_or_none(vals) is None:
                    group_ok = False
                    break
            if not group_ok:
                result[m] = None
            else:
                valid_all = [k for k in all_keys if has_src[k][kind]]
                result[m] = _mean_or_none([metric_norm[k][m] for k in valid_all])
        return result

    # --- 主循环：single/multi 分开、每个比例算一次（固定 seed 可复现）---
    flat: dict[str, dict[str, dict[str, float | None]]] = {
        "single": {m: {} for m in models},
        "multi":  {m: {} for m in models},
    }
    n_sampled: dict[str, dict[str, int]] = {"single": {}, "multi": {}}

    for r in ratios:
        # 每个 (kind,ratio) 用独立 seed 派生，保证互相独立可复现
        rng_s = _random.Random(BY_SAMPLING_SEED + int(r * 1000) + 1)
        rng_m = _random.Random(BY_SAMPLING_SEED + int(r * 1000) + 2)
        ids_s = _sample_ids_of("single_shot", r, rng_s)
        ids_m = _sample_ids_of("multi_shot",  r, rng_m)
        n_sampled["single"][str(r)] = len(ids_s)
        n_sampled["multi"][str(r)]  = len(ids_m)
        sc_s = _score_for_sample(ids_s, "single_shot")
        sc_m = _score_for_sample(ids_m, "multi_shot")
        for m in models:
            flat["single"][m][str(r)] = sc_s[m]
            flat["multi"][m][str(r)]  = sc_m[m]

    return {
        "models": models,
        "ratios": ratios,
        "seed": BY_SAMPLING_SEED,
        "flat": flat,           # {"single": {model: {ratio: v}}, "multi": {model: {ratio: v}}}
        "n_sampled": n_sampled, # {"single": {ratio: n}, "multi": {ratio: n}}
    }


if __name__ == "__main__":
    main()
