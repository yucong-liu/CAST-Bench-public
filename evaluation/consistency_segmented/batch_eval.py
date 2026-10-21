#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_eval.py
=============
一次跑完所有模型的「分割增强 主体/背景 一致性」评测。

数据来源
--------
直接复用 audio_consistency 工程已经切好的镜头：
  <audio_runs_dir>/<model>/shotcut_manifest.jsonl
  <audio_runs_dir>/<model>/clips/<spec_id>_shot<k>.mp4   (multi_shot)
  对 single_shot 的 spec, manifest 里 clip_path 直接指向原始整段 mp4。

评分流程
--------
1. 扫 <audio_runs_dir> 下所有含 shotcut_manifest.jsonl 的子目录, 视为一个模型。
2. 加载 SAM2 / DINOv2 / CLIP 一次, 跨所有模型复用 (节省启动+显存)。
3. 对每个模型:
   - 解析 manifest, 跳过 passed=false / shots=[] / 文件缺失;
   - 对每个 spec 的每个 shot 独立: 帧级 SAM2 主体 mask 追踪 ->
     主体路 DINOv2 / 背景路 CLIP -> 帧间一致性 (sim, cnt);
   - spec 分 = sum(shot_sim) / sum(shot_cnt)             (按帧加权)
   - 模型分 = sum(all_shot_sim) / sum(all_shot_cnt)      (按帧加权)
   - 还会按 category 做一份聚合用于诊断。
4. 输出:
   ./runs/<model>/segmented_results.json        每个模型的明细 (含 per_spec / per_category)
   ./runs/<model>/skipped.json                  被跳过的 spec 列表
   ./runs/compare_summary_single_shot.csv       单镜头桶: 行=category, 列=各模型 subj/bg/n
   ./runs/compare_summary_multi_shot.csv        多镜头桶: 行=category, 列=各模型 subj/bg/n

用法
----
  python batch_eval.py \\
      --audio_runs_dir evaluation/audio_consistency/runs/qwen3_omni \\
      --sam2_ckpt   ./pretrained/sam2.1_hiera_large.pt \\
      --sam2_cfg    configs/sam2.1/sam2.1_hiera_l.yaml \\
      --subject_encoder dinov2 \\
      --clip_model  ViT-B/32 \\
      --device cuda \\
      --output_root ./runs

可选:
  --models LTX-2.3,Ovi   只评指定的一组模型 (逗号分隔)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import torch  # noqa: E402

from seg_consistency import (  # noqa: E402
    discover_models,
    load_manifest,
    compute_segmented_consistency_from_manifest,
    SpecsLoader,
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Batch segmentation-enhanced consistency eval over all models."
    )
    ap.add_argument(
        "--audio_runs_dir", type=str, required=True,
        help="audio_consistency runs 目录, 例如 .../audio_consistency/runs/qwen3_omni "
             "(其下每个子目录是一个模型, 含 shotcut_manifest.jsonl)",
    )
    ap.add_argument(
        "--sam2_ckpt", type=str,
        default=str(THIS_DIR / "pretrained" / "sam2.1_hiera_large.pt"),
        help="SAM 2 .pt 权重路径",
    )
    ap.add_argument(
        "--sam2_cfg", type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM 2 model config (相对 sam2 包内置 configs)",
    )
    ap.add_argument(
        "--subject_encoder", type=str, default="dinov2",
        choices=["dinov2", "dino"],
    )
    ap.add_argument(
        "--clip_model", type=str, default="ViT-B/32",
        help="CLIP backbone for background, e.g. 'ViT-B/32' (VBench parity) or 'ViT-L/14'",
    )
    ap.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
    )
    ap.add_argument(
        "--output_root", type=str, default=str(THIS_DIR / "runs"),
        help="所有模型输出的根目录",
    )
    ap.add_argument(
        "--models", type=str, default="",
        help="可选, 逗号分隔白名单, 默认评估 audio_runs_dir 下所有模型",
    )
    ap.add_argument(
        "--categories", type=str, default="speech,non_verbal_vocal",
        help="逗号分隔的 category 白名单。**默认 'speech,non_verbal_vocal'** — 只评估"
             "说话人 / 非语言人声两类, 因为它们的主体运动幅度小、外观漂移信号更纯净。"
             "action 类主体动作幅度大, 会把'生成能力强'错算成扣分, 已从默认集里剔除。"
             "如需评估其他类别, 显式传入 (例如 'speech,action'); 传 'all' 或空串 '' 表示"
             "全量。匹配 manifest 里 entry['category'] 字段, 大小写敏感。",
    )
    ap.add_argument(
        "--fail_fast", action="store_true",
        help="任一 spec 异常即中止 (默认是跳过继续跑)",
    )
    ap.add_argument(
        "--max_frames", type=int, default=32,
        help="每个 shot 最多送入 SAM2/DINOv2/CLIP 的帧数, 超过则等距抽样 (VBench 风格). "
             "0 表示关闭抽帧、用全帧. 默认 32.",
    )
    # ---- text-grounded subject seeding (GroundingDINO) ----
    ap.add_argument(
        "--specs_root", type=str,
        default=str(_REPO_ROOT / "t2av-bench/specs"),
        help="t2av-bench specs 根目录, 含 <category>.jsonl. 用于按 spec_id "
             "取出 actor 主体的 descriptor 喂给 GroundingDINO.",
    )
    ap.add_argument(
        "--grounding_ckpt", type=str,
        default=str(THIS_DIR / "pretrained" / "grounding-dino-tiny"),
        help="GroundingDINO HF 权重目录 (含 model.safetensors).",
    )
    ap.add_argument(
        "--use_text_prompt", action="store_true",
        help="开启基于 specs.subjects 的文本提示主体选择 (GroundingDINO -> SAM2 box). "
             "未开启时走原 auto-mask + 居中加权回退.",
    )
    ap.add_argument(
        "--verbose_json", action="store_true",
        help="写盘 segmented_results.json 时保留 seed_meta / per_actor / actors / clip_path "
             "等调试字段 (默认精简, 只保留得分相关).",
    )
    ap.add_argument(
        "--append_summary", action="store_true",
        help="汇总 CSV 采用追加模式: 读取已存在的 compare_summary_<kind>.csv, "
             "把本次跑到的模型作为新列合并进去; 已有列/值一律保留不改, 只追加新模型列 "
             "和新出现的 category 行。默认关闭 (全量覆盖, 保留原全量跑行为).",
    )
    return ap.parse_args()


def _filter_models(found: List[Tuple[str, str]],
                   whitelist: str) -> List[Tuple[str, str]]:
    if not whitelist.strip():
        return found
    keep = {x.strip() for x in whitelist.split(",") if x.strip()}
    out = [x for x in found if x[0] in keep]
    missing = keep - {x[0] for x in out}
    if missing:
        print(f"[WARN] --models 中以下名字未在 audio_runs_dir 下找到: "
              f"{sorted(missing)}", file=sys.stderr)
    return out


def _slim_result_for_json(res: Dict) -> Dict:
    """把 compute_segmented_consistency_from_manifest 的返回值瘦身, 只保留得分相关字段.

    删除: per_spec[*].shots[*].seed_meta / per_actor / actors / clip_path /
          n_frames / n_frames_used / mask_aggregation / per_spec[*].actors
    保留: 顶层聚合 + per_spec 的得分字段 + shots 的得分字段。
    """
    keep_top = {
        "subject_consistency_avg", "background_consistency_avg",
        "n_specs_used", "n_shots_used",
        "per_category", "per_kind", "per_category_kind",
        "model", "manifest", "elapsed_sec",
        "skipped", "runtime_skipped",
    }
    slim: Dict = {k: v for k, v in res.items() if k in keep_top}

    slim_specs: List[Dict] = []
    for sp in res.get("per_spec", []) or []:
        sp_keep = {
            "spec_id": sp.get("spec_id"),
            "category": sp.get("category"),
            "kind": sp.get("kind"),
            "subject_consistency": sp.get("subject_consistency"),
            "background_consistency": sp.get("background_consistency"),
            "n_shots_used": sp.get("n_shots_used"),
        }
        shots_slim: List[Dict] = []
        for sh in sp.get("shots", []) or []:
            shots_slim.append({
                "shot_idx": sh.get("shot_idx"),
                "ok": sh.get("ok"),
                "subject_consistency": sh.get("subject_consistency"),
                "background_consistency": sh.get("background_consistency"),
                "n_actors_in_union": sh.get("n_actors_in_union"),
            })
        sp_keep["shots"] = shots_slim
        slim_specs.append(sp_keep)
    slim["per_spec"] = slim_specs
    return slim


def _read_existing_summary_csv(csv_path: Path) -> Tuple[List[str], Dict[str, Dict[str, Tuple[str, str, str]]]]:
    """读取已存在的 compare_summary_<kind>.csv。

    返回:
        old_models:  按 header 中 subj_* 出现顺序解析出的模型列表
        old_rows:    {category: {model: (subj_str, bg_str, n_str)}}

    如果文件不存在或格式异常, 返回 ([], {}).
    """
    if not csv_path.is_file():
        return [], {}
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] failed to read existing {csv_path}: {e}", file=sys.stderr)
        return [], {}
    if not rows:
        return [], {}
    header = rows[0]
    # 从 header 里按 subj_ 前缀提取模型顺序; bg_/n_ 与 subj_ 应当一一对应
    old_models: List[str] = []
    idx_subj: Dict[str, int] = {}
    idx_bg: Dict[str, int] = {}
    idx_n: Dict[str, int] = {}
    for i, col in enumerate(header):
        if col.startswith("subj_"):
            m = col[len("subj_"):]
            old_models.append(m)
            idx_subj[m] = i
        elif col.startswith("bg_"):
            idx_bg[col[len("bg_"):]] = i
        elif col.startswith("n_"):
            idx_n[col[len("n_"):]] = i
    old_rows: Dict[str, Dict[str, Tuple[str, str, str]]] = {}
    for r in rows[1:]:
        if not r:
            continue
        cat = r[0]
        per_model: Dict[str, Tuple[str, str, str]] = {}
        for m in old_models:
            si = idx_subj.get(m); bi = idx_bg.get(m); ni = idx_n.get(m)
            s = r[si] if si is not None and si < len(r) else ""
            b = r[bi] if bi is not None and bi < len(r) else ""
            n = r[ni] if ni is not None and ni < len(r) else ""
            per_model[m] = (s, b, n)
        old_rows[cat] = per_model
    return old_models, old_rows


def _write_compare_by_kind_split(out_root: Path,
                                 model_results: List[Tuple[str, Dict]],
                                 kind: str,
                                 append: bool = False) -> None:
    """输出某一 kind (single_shot / multi_shot) 桶的横向对比表。

    行  = category (来自 per_category_kind 中 'category|<kind>' 键)
    列  = 每个模型的 subj_<model> / bg_<model> / n_<model>
    末尾追加 __ALL__ 行: 该 kind 下所有 category 的总均值, 直接取 per_kind[<kind>]。

    append 模式:
        读取磁盘已存在的 compare_summary_<kind>.csv, 保留其中所有旧列/旧值不动,
        本次跑到的模型作为新列追加; 若本次模型与旧模型同名, 保留旧值 (不覆盖).
        行 (category) 取旧行 ∪ 新行 的并集, __ALL__ 放最后。
    """
    csv_path = out_root / f"compare_summary_{kind}.csv"

    # 本次 (new) 结果
    new_cats = set()
    for _, res in model_results:
        for key in (res.get("per_category_kind", {}) or {}).keys():
            if "|" in key:
                cat, kk = key.split("|", 1)
                if kk == kind:
                    new_cats.add(cat)
    new_models = [m for m, _ in model_results]

    # ---- 防覆盖保险 ----
    # 本次 model_results 里如果没有任何模型对当前 kind 贡献了 n_specs>0 的数据,
    # 视为"这轮跑什么都没跑出来", 不要用空表覆盖磁盘上已有的历史 CSV。
    def _kind_has_any_valid() -> bool:
        for _, res in model_results:
            v = (res.get("per_kind", {}) or {}).get(kind, {}) or {}
            try:
                if int(v.get("n_specs", 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                pass
            # 兜底: per_category_kind 里存在该 kind 的行, 且 n_specs > 0
            for key, cv in (res.get("per_category_kind", {}) or {}).items():
                if "|" in key and key.split("|", 1)[1] == kind:
                    try:
                        if int((cv or {}).get("n_specs", 0) or 0) > 0:
                            return True
                    except (TypeError, ValueError):
                        pass
        return False

    if not _kind_has_any_valid():
        if csv_path.is_file():
            print(f"[skip write] {csv_path.name}: "
                  f"no valid model result this run, keep existing file untouched.")
            return
        # 磁盘上也没有旧文件时, 仍然写一个只有 header + 空 __ALL__ 的占位表,
        # 与旧行为保持一致 (至少产出文件, 方便后续检查)。
        print(f"[warn] {csv_path.name}: no valid model result and no existing file, "
              f"write empty placeholder.")

    def _fmt_score(v) -> str:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return ""
        return "" if f != f else f"{f:.6f}"  # nan -> ""

    def _new_val(res: Dict, cat: str) -> Tuple[str, str, str]:
        v = res.get("per_category_kind", {}).get(f"{cat}|{kind}", {})
        return (_fmt_score(v.get("subject_consistency", float("nan"))),
                _fmt_score(v.get("background_consistency", float("nan"))),
                str(int(v.get("n_specs", 0))))

    def _new_val_all(res: Dict) -> Tuple[str, str, str]:
        v = res.get("per_kind", {}).get(kind, {})
        return (_fmt_score(v.get("subject_consistency", float("nan"))),
                _fmt_score(v.get("background_consistency", float("nan"))),
                str(int(v.get("n_specs", 0))))

    if append:
        old_models, old_rows = _read_existing_summary_csv(csv_path)
        # 合并模型列: 旧顺序在前, 本次新出现的模型追加到后面; 同名不重复, 不覆盖旧值
        merged_models: List[str] = list(old_models)
        for m in new_models:
            if m not in merged_models:
                merged_models.append(m)
        # 合并 category 行 (不含 __ALL__): 旧行顺序在前, 新出现的追加
        old_cats_ordered = [c for c in old_rows.keys() if c != "__ALL__"]
        merged_cats: List[str] = list(old_cats_ordered)
        for c in sorted(new_cats):
            if c not in merged_cats:
                merged_cats.append(c)
        print(f"[append] {csv_path.name}: old_models={old_models} "
              f"new_models={new_models} -> merged={merged_models}")
    else:
        old_models, old_rows = [], {}
        merged_models = list(new_models)
        merged_cats = sorted(new_cats)

    # 名字 -> 本次 result 的映射, 便于按合并顺序取值
    new_res_map: Dict[str, Dict] = {m: r for m, r in model_results}

    def _cell_for(cat_or_all: str, model: str) -> Tuple[str, str, str]:
        """取某 (category, model) 的 (subj, bg, n) 三元组。

        规则 (append 模式下严格'只追加不改, 但允许补空'):
          1) 若 model 是旧模型 (在 old_models 中):
             - 默认用旧值 (不覆盖已有的非空历史数据)
             - 例外: 若旧值三元组 (subj, bg, n) 全为空/'0' 这种"占位空", 且本次这个模型
               恰好有非空的新值, 则用新值把空位补上。这样能自动修复"某模型上次跑残破产物
               留下的空 header 列, 之后每次再跑都被永久锁在空值"的历史 bug。
          2) 若 model 是本次新模型: 用本次结果。
        非 append 模式: 全走本次结果。
        """
        # 计算本次这个 model 的候选值 (可能没跑, 那就是 None)
        def _new_triple() -> Tuple[str, str, str]:
            res = new_res_map.get(model)
            if res is None:
                return ("", "", "")
            if cat_or_all == "__ALL__":
                return _new_val_all(res)
            if f"{cat_or_all}|{kind}" not in (res.get("per_category_kind", {}) or {}):
                return ("", "", "")
            return _new_val(res, cat_or_all)

        if append and model in old_models:
            row = old_rows.get(cat_or_all, {})
            old = row.get(model, ("", "", ""))
            # "占位空"判定: subj 和 bg 都为空 (n 允许是 '0' 或空, 都算残破)
            is_empty_placeholder = (
                (old[0] == "" or old[0] is None) and
                (old[1] == "" or old[1] is None)
            )
            if is_empty_placeholder:
                new = _new_triple()
                if new[0] != "" and new[1] != "":
                    print(f"[append/补空] {model} @ {cat_or_all}: 旧值空 {old} -> 用新值 {new}")
                    return new
            return old
        # new model (或非 append 模式)
        return _new_triple()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["category"]
        for m in merged_models:
            header.append(f"subj_{m}")
        for m in merged_models:
            header.append(f"bg_{m}")
        for m in merged_models:
            header.append(f"n_{m}")
        w.writerow(header)

        for cat in merged_cats:
            row = [cat]
            triples = [_cell_for(cat, m) for m in merged_models]
            row.extend(t[0] for t in triples)
            row.extend(t[1] for t in triples)
            row.extend(t[2] for t in triples)
            w.writerow(row)

        # __ALL__ 行: 旧模型用旧 __ALL__ 值, 新模型用本次 per_kind[<kind>]
        row = ["__ALL__"]
        triples = [_cell_for("__ALL__", m) for m in merged_models]
        row.extend(t[0] for t in triples)
        row.extend(t[1] for t in triples)
        row.extend(t[2] for t in triples)
        w.writerow(row)
    print(f"[write] {csv_path}{'  (append)' if append else ''}")


def main() -> int:
    args = _parse_args()

    audio_runs_dir = Path(args.audio_runs_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    sam2_ckpt = Path(args.sam2_ckpt).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not audio_runs_dir.is_dir():
        print(f"[ERROR] audio_runs_dir not found: {audio_runs_dir}", file=sys.stderr)
        return 2
    if not sam2_ckpt.is_file():
        print(f"[ERROR] SAM2 ckpt not found: {sam2_ckpt}", file=sys.stderr)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU (will be VERY slow).")
        args.device = "cpu"

    # 1) 发现模型
    found = discover_models(str(audio_runs_dir))
    found = _filter_models(found, args.models)
    if not found:
        print(f"[ERROR] no model with shotcut_manifest.jsonl under {audio_runs_dir}",
              file=sys.stderr)
        return 2
    cat_whitelist = {c.strip() for c in args.categories.split(",") if c.strip()}
    # 显式 'all' → 关闭过滤, 跑全部 category
    if cat_whitelist == {"all"}:
        cat_whitelist = set()
    print(f"[info] audio_runs_dir = {audio_runs_dir}")
    print(f"[info] models to evaluate ({len(found)}): "
          f"{[m for m, _ in found]}")
    print(f"[info] max_frames per shot = "
          f"{'<full>' if not args.max_frames or args.max_frames <= 0 else args.max_frames}")
    if cat_whitelist:
        print(f"[info] category whitelist = {sorted(cat_whitelist)}")
    print(f"[info] use_text_prompt = {args.use_text_prompt}")
    if args.use_text_prompt:
        print(f"[info] specs_root      = {args.specs_root}")
        print(f"[info] grounding_ckpt  = {args.grounding_ckpt}")
        if not Path(args.grounding_ckpt).is_dir():
            print(f"[ERROR] grounding_ckpt not a directory: {args.grounding_ckpt}",
                  file=sys.stderr)
            return 2
        if not Path(args.specs_root).is_dir():
            print(f"[ERROR] specs_root not a directory: {args.specs_root}",
                  file=sys.stderr)
            return 2

    specs_loader = SpecsLoader(args.specs_root) if args.use_text_prompt else None

    # 2) 跨模型共享 SAM2 / DINOv2 / CLIP, 由 _build_encoders 内部惰性构造
    encoder_cache: Dict = {}

    # 3) 逐模型评测
    model_results: List[Tuple[str, Dict]] = []
    for mi, (model, manifest_path) in enumerate(found, 1):
        print(f"\n========== [{mi}/{len(found)}] {model} ==========")
        entries, skipped = load_manifest(manifest_path)
        print(f"[info] manifest: {manifest_path}")
        print(f"[info] specs to score = {len(entries)}, skipped = {len(skipped)}")
        if cat_whitelist:
            before = len(entries)
            entries = [e for e in entries if e.get("category") in cat_whitelist]
            seen_cats = sorted({e.get("category") for e in entries})
            print(f"[info] category filter: kept {len(entries)}/{before} "
                  f"(categories present after filter = {seen_cats})")
            missing_cats = cat_whitelist - set(seen_cats)
            if missing_cats:
                print(f"[WARN] --categories 中以下类别在 {model} 的 manifest 里没有匹配到: "
                      f"{sorted(missing_cats)}", file=sys.stderr)
        if not entries:
            print(f"[WARN] no scorable spec for {model}, skip.")
            res = {
                "model": model,
                "manifest": str(manifest_path),
                "subject_consistency_avg": float("nan"),
                "background_consistency_avg": float("nan"),
                "n_specs_used": 0, "n_shots_used": 0,
                "per_category": {}, "per_spec": [],
                "skipped": skipped, "runtime_skipped": [],
            }
        else:
            t0 = time.time()
            try:
                res = compute_segmented_consistency_from_manifest(
                    entries,
                    device=args.device,
                    sam2_ckpt=str(sam2_ckpt),
                    sam2_cfg=args.sam2_cfg,
                    subject_encoder=args.subject_encoder,
                    clip_model=args.clip_model,
                    encoder_cache=encoder_cache,
                    progress_desc=model,
                    max_frames=(args.max_frames if args.max_frames and args.max_frames > 0 else None),
                    specs_loader=specs_loader,
                    grounding_ckpt=(args.grounding_ckpt if args.use_text_prompt else None),
                    use_text_prompt=args.use_text_prompt,
                )
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                print(f"[ERROR] model {model} failed: {msg}", file=sys.stderr)
                if args.fail_fast:
                    raise
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                continue
            dt = time.time() - t0
            res["model"] = model
            res["manifest"] = str(manifest_path)
            res["skipped"] = skipped
            res["elapsed_sec"] = round(dt, 2)
            print(f"[done] {model}: subject={res['subject_consistency_avg']:.4f} "
                  f"background={res['background_consistency_avg']:.4f} "
                  f"specs={res['n_specs_used']} shots={res['n_shots_used']} "
                  f"({dt:.1f}s)")

        # 写每模型 JSON (默认瘦身, --verbose_json 时保留完整字段)
        m_dir = output_root / model
        m_dir.mkdir(parents=True, exist_ok=True)
        out_json = m_dir / "segmented_results.json"
        to_dump = res if args.verbose_json else _slim_result_for_json(res)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(to_dump, f, indent=2, ensure_ascii=False)
        print(f"[write] {out_json}"
              f"{'  (verbose)' if args.verbose_json else '  (slim)'}")

        # 单独写 skipped (manifest + runtime), 方便后续排查
        skipped_all = {
            "manifest_skipped": res.get("skipped", []),
            "runtime_skipped": res.get("runtime_skipped", []),
        }
        with (m_dir / "skipped.json").open("w", encoding="utf-8") as f:
            json.dump(skipped_all, f, indent=2, ensure_ascii=False)

        model_results.append((model, res))

        # 跨模型回收一次显存碎片
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if not model_results:
        print("[ERROR] all models failed.", file=sys.stderr)
        return 1

    # 4) 横向对比表: 只输出 single_shot / multi_shot 两张分桶表
    _write_compare_by_kind_split(output_root, model_results, "single_shot",
                                 append=args.append_summary)
    _write_compare_by_kind_split(output_root, model_results, "multi_shot",
                                 append=args.append_summary)

    print("\n========== ALL DONE ==========")
    for model, res in model_results:
        print(f"  {model:24s}  subj={res.get('subject_consistency_avg', float('nan')):.4f}  "
              f"bg={res.get('background_consistency_avg', float('nan')):.4f}  "
              f"specs={res.get('n_specs_used', 0)}  shots={res.get('n_shots_used', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
