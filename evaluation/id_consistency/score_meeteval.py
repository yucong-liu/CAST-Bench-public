#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
id_consistency · meeteval 评分

ref : t2av-bench/specs/speech.jsonl
hyp : <runs-root>/<model>/transcripts/<spec_id>.json   (transcribe_omni.py 输出)

输出: <out-dir>/compare_summary.csv  (横轴=模型, 纵轴=__ALL__ 一行)
     <out-dir>/per_sample.csv
     <out-dir>/details.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple
_REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from meeteval.io.seglst import SegLST
    from meeteval.wer import cp_word_error_rate, greedy_orc_word_error_rate
except Exception as e:
    print(f"[FATAL] import meeteval failed: {e}", file=sys.stderr)
    sys.exit(1)

# ============================================================
# 文本归一化
# ============================================================
_PUNCT_RE = re.compile(
    r"[\s\u3000\.,!\?;:\"'`~@#\$%\^&\*\(\)\[\]\{\}<>\\/\|=\+\-_。，！？；：、""''『』「」（）《》【】…—~·]+"
)
_CN_RE = re.compile(r"[\u4e00-\u9fff]")


def _is_special_token(text: str) -> bool:
    t = text.strip()
    return t.startswith("[") and t.endswith("]")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _CN_RE.sub(lambda m: f" {m.group(0)} ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ============================================================
# Ref / Hyp
# ============================================================
def load_ref(specs_path: Path) -> Tuple[
        Dict[str, List[Tuple[str, str]]],
        Dict[str, Dict[str, str]],
        Dict[str, str],
]:
    """读取 specs。

    返回:
        ref_segs : {spec_id: [(speaker_id, text), ...]}
                   优先 shots[*].dialogue, 否则 subjects[*].speech
        ref_desc : {spec_id: {speaker_id: descriptor_str}}
                   方便 per_sample.csv 人工核对说话主体
        ref_kind : {spec_id: shot_count}  (single_shot / multi_shot / "")
                   用于在 compare_summary 中按 single/multi 拆子表
    """
    ref: Dict[str, List[Tuple[str, str]]] = {}
    desc: Dict[str, Dict[str, str]] = {}
    kind: Dict[str, str] = {}
    with open(specs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj["id"]
            segs: List[Tuple[str, str]] = []
            shots = obj.get("shots") or []
            for sh in shots:
                for d in sh.get("dialogue", []) or []:
                    spk = d.get("speaker_id") or "S?"
                    txt = d.get("text") or ""
                    if txt.strip():
                        segs.append((spk, txt))
            if not segs:
                for sub in obj.get("subjects", []) or []:
                    if sub.get("is_speaker") and sub.get("speech"):
                        segs.append((sub.get("id") or "S?", sub["speech"]))
            ref[sid] = segs

            # 收集 speaker_id -> descriptor (供 per_sample 人工核对)
            d_map: Dict[str, str] = {}
            for sub in obj.get("subjects", []) or []:
                sub_id = sub.get("id")
                if not sub_id:
                    continue
                descriptor = (sub.get("descriptor")
                              or sub.get("appearance")
                              or sub.get("role")
                              or "")
                d_map[str(sub_id)] = str(descriptor).strip()
            desc[sid] = d_map
            # shot_count: single_shot / multi_shot / "" (其它)
            sc = obj.get("shot_count", "")
            kind[sid] = sc if sc in ("single_shot", "multi_shot") else ""
    return ref, desc, kind


def load_hyp(json_path: Path) -> Tuple[List[Tuple[str, str, str]], dict]:
    """读取 transcribe_omni 单条输出, 返回 ([(speaker, subject, text), ...], meta).

    subject 旧版没有的话兜底为 ""。
    """
    meta = {"parse_ok": False, "error": "", "raw_len": 0}
    if not json_path.exists():
        meta["error"] = "missing"
        return [], meta
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        meta["error"] = f"parse_fail:{e}"
        return [], meta

    meta["parse_ok"] = bool(obj.get("parse_ok", False))
    meta["error"] = obj.get("error", "") or ""
    meta["raw_len"] = len(obj.get("raw_response", "") or "")

    segs: List[Tuple[str, str, str]] = []
    for s in obj.get("segments", []) or []:
        txt = s.get("text") or ""
        if not txt.strip():
            continue
        if _is_special_token(txt):
            continue
        spk = s.get("speaker") or "S1"
        sub = s.get("subject") or ""
        segs.append((str(spk), str(sub).strip(), txt))
    return segs, meta


def to_seglst(session_id: str, segs) -> SegLST:
    """将 [(speaker, text)] 或 [(speaker, subject, text)] 转成 SegLST。

    评分只用 speaker 与 text，subject 不参与 cpwer 计算。
    """
    rows = []
    for i, item in enumerate(segs):
        if len(item) == 3:
            spk, _sub, txt = item
        else:
            spk, txt = item
        ntxt = normalize_text(txt)
        if not ntxt:
            continue
        rows.append({
            "session_id": session_id,
            "speaker": str(spk),
            "words": ntxt,
            "start_time": float(i),
            "end_time": float(i) + 0.5,
        })
    return SegLST(rows)


# ============================================================
# 单样本评分
# ============================================================
def score_one(spec_id: str,
              ref_segs,
              hyp_segs) -> dict:
    out = {
        "status": "ok",
        "cp_errors": 0, "cp_length": 0,
        "orc_errors": 0, "orc_length": 0,
        "cpwer": None, "orcwer": None, "delta_cp": None,
    }
    ref_lst = to_seglst(spec_id, ref_segs)
    if len(ref_lst) == 0:
        out["status"] = "no_ref"
        return out

    if len(hyp_segs) == 0:
        ref_words = sum(len(r["words"].split()) for r in ref_lst)
        out["status"] = "empty_hyp"
        out["cp_errors"] = ref_words
        out["cp_length"] = ref_words
        out["orc_errors"] = ref_words
        out["orc_length"] = ref_words
        out["cpwer"] = 1.0 if ref_words > 0 else 0.0
        out["orcwer"] = 1.0 if ref_words > 0 else 0.0
        out["delta_cp"] = 0.0
        return out

    hyp_lst = to_seglst(spec_id, hyp_segs)
    cp = cp_word_error_rate(ref_lst, hyp_lst)
    orc = greedy_orc_word_error_rate(ref_lst, hyp_lst)
    out["cp_errors"] = int(cp.errors)
    out["cp_length"] = int(cp.length)
    out["orc_errors"] = int(orc.errors)
    out["orc_length"] = int(orc.length)
    out["cpwer"] = float(cp.error_rate) if cp.length > 0 else 0.0
    out["orcwer"] = float(orc.error_rate) if orc.length > 0 else 0.0
    out["delta_cp"] = out["cpwer"] - out["orcwer"]
    return out


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs",
                    default=str(_REPO_ROOT / "t2av-bench/specs/speech.jsonl"))
    ap.add_argument("--runs-root",
                    default=str(_REPO_ROOT / "evaluation/id_consistency/runs"))
    ap.add_argument("--models",
                    default="LTX-2.3,Ovi,daVinci-MagiHuman,seedance,wan2.7")
    ap.add_argument("--out-dir",
                    default=str(_REPO_ROOT / "evaluation/id_consistency/score_out"))
    ap.add_argument("--append-models", default="",
                    help="逗号或空格分隔的模型名; 传入则进入'追加模式':\n"
                         "  * 读旧 3 张 compare_summary*.csv, 保留其他模型的列不变;\n"
                         "  * 只对这些模型现场跑 meeteval, 结果追加到 CSV 列末尾\n"
                         "    (若模型名在旧表已存在, 原地覆盖 7 组指标列);\n"
                         "  * per_sample.csv / details.json 不覆盖原表, 改名写到\n"
                         "    per_sample.append.csv / details.append.json.\n"
                         "  * --models 参数在追加模式下会被忽略, 只以 --append-models 为准.")
    args = ap.parse_args()

    specs_path = Path(args.specs)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析 --append-models
    append_models: List[str] = []
    if args.append_models.strip():
        raw = args.append_models.replace(",", " ").split()
        append_models = [m for m in raw if m]

    if append_models:
        # 追加模式: 只以 append_models 为工作集
        models = list(append_models)
        print(f"[INFO] APPEND MODE  append_models={append_models}")
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"[INFO] specs   : {specs_path}")
    print(f"[INFO] runs    : {runs_root}")
    print(f"[INFO] models  : {models}")
    print(f"[INFO] out     : {out_dir}")

    ref, ref_desc, ref_kind = load_ref(specs_path)
    print(f"[INFO] loaded {len(ref)} ref specs "
          f"(single={sum(1 for v in ref_kind.values() if v == 'single_shot')}, "
          f"multi={sum(1 for v in ref_kind.values() if v == 'multi_shot')}, "
          f"unknown={sum(1 for v in ref_kind.values() if not v)})")

    per_sample_rows = []
    # 三套 agg 桶: all (全样本) / single / multi
    def _new_agg():
        return {m: {"cp_e": 0, "cp_l": 0, "orc_e": 0, "orc_l": 0,
                    "n_specs": 0, "n_ok": 0, "n_empty_hyp": 0,
                    "n_missing": 0, "n_no_ref": 0} for m in models}
    agg_all    = _new_agg()
    agg_single = _new_agg()
    agg_multi  = _new_agg()

    def _fmt_ref_speakers(sid: str, segs) -> str:
        """按 ref 中 speaker 首次出现顺序拼: 'S1=<descriptor> | S2=<descriptor>'."""
        seen = []
        for spk, _txt in segs:
            if spk not in seen:
                seen.append(spk)
        d_map = ref_desc.get(sid, {})
        return " | ".join(
            f"{spk}={d_map.get(spk, '')}".rstrip("=")
            for spk in seen
        )

    def _fmt_hyp_speakers(segs) -> str:
        """按 hyp 中 speaker 首次出现顺序拼: 'S1=<subject> | S2=<subject>'."""
        seen = {}  # spk -> first non-empty subject
        order = []
        for item in segs:
            if len(item) == 3:
                spk, sub, _txt = item
            else:
                spk, _txt = item
                sub = ""
            if spk not in seen:
                seen[spk] = sub
                order.append(spk)
            elif not seen[spk] and sub:
                seen[spk] = sub
        return " | ".join(
            f"{spk}={seen[spk]}".rstrip("=")
            for spk in order
        )

    for sid in sorted(ref.keys()):
        ref_segs = ref[sid]
        ref_speakers_str = _fmt_ref_speakers(sid, ref_segs)
        for m in models:
            hyp_path = runs_root / m / "transcripts" / f"{sid}.json"
            hyp_segs, meta = load_hyp(hyp_path)
            row = score_one(sid, ref_segs, hyp_segs)
            row["spec_id"] = sid
            row["model"] = m
            row["hyp_missing"] = (not hyp_path.exists())
            row["error"] = meta["error"]
            row["raw_len"] = meta["raw_len"]
            row["hyp_parse_ok"] = meta["parse_ok"]
            row["ref_speakers"] = ref_speakers_str
            row["hyp_speakers"] = _fmt_hyp_speakers(hyp_segs)

            if not hyp_path.exists():
                row["status"] = "missing"

            per_sample_rows.append(row)

            # 累加到 all 桶 + 对应 kind 桶 (single/multi)
            kind = ref_kind.get(sid, "")
            target_aggs = [agg_all]
            if kind == "single_shot":
                target_aggs.append(agg_single)
            elif kind == "multi_shot":
                target_aggs.append(agg_multi)

            for _agg in target_aggs:
                a = _agg[m]
                a["n_specs"] += 1
                if row["status"] == "ok":
                    a["n_ok"] += 1
                    a["cp_e"] += row["cp_errors"]
                    a["cp_l"] += row["cp_length"]
                    a["orc_e"] += row["orc_errors"]
                    a["orc_l"] += row["orc_length"]
                elif row["status"] == "empty_hyp":
                    a["n_empty_hyp"] += 1
                    a["cp_e"] += row["cp_errors"]
                    a["cp_l"] += row["cp_length"]
                    a["orc_e"] += row["orc_errors"]
                    a["orc_l"] += row["orc_length"]
                elif row["status"] == "missing":
                    a["n_missing"] += 1
                elif row["status"] == "no_ref":
                    a["n_no_ref"] += 1

    # ---- per_sample.csv ----
    #   全量模式: 写 per_sample.csv (覆盖)
    #   追加模式: 写 per_sample.append.csv (不覆盖原表, 仅含 append_models 的行)
    ps_filename = "per_sample.append.csv" if append_models else "per_sample.csv"
    ps_path = out_dir / ps_filename
    fields = ["spec_id", "model", "status",
              "cpwer", "orcwer", "delta_cp",
              "cp_errors", "cp_length", "orc_errors", "orc_length",
              "ref_speakers", "hyp_speakers",
              "hyp_missing", "hyp_parse_ok", "raw_len", "error"]
    with open(ps_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_sample_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[OK] wrote {ps_path}  ({len(per_sample_rows)} rows)")

    # ---- compare_summary*.csv (三份: all / single / multi) ----
    metrics = ["cpwer", "orcwer", "delta_cp", "n_specs", "n_ok", "n_empty_hyp", "n_missing"]

    def fmt(v, met):
        if v is None:
            return ""
        if met in ("n_specs", "n_ok", "n_empty_hyp", "n_missing"):
            return str(int(v))
        return f"{v:.4f}"

    def _cell_val(a, met):
        """给某个 model 的 agg 桶 a, 返回它在 metric met 下的原始数值 (float/int/None)."""
        if met == "cpwer":
            return (a["cp_e"] / a["cp_l"]) if a["cp_l"] > 0 else None
        if met == "orcwer":
            return (a["orc_e"] / a["orc_l"]) if a["orc_l"] > 0 else None
        if met == "delta_cp":
            cp = (a["cp_e"] / a["cp_l"]) if a["cp_l"] > 0 else None
            orc = (a["orc_e"] / a["orc_l"]) if a["orc_l"] > 0 else None
            return (cp - orc) if (cp is not None and orc is not None) else None
        return a[met]

    def _read_old_summary_csv(path: Path):
        """读旧 compare_summary*.csv, 返回 (old_models, header_list, all_row_map).

        - old_models: 从 header 里按 cpwer_<m> 出现顺序抽出的模型名列表
        - header_list: 原 header (list)
        - all_row_map: __ALL__ 那一行的 {col_name: value_str}; 若无 __ALL__ 行则 {}
        - 若文件不存在, 返回 ([], [], {})
        """
        if not path.exists():
            return [], [], {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
        except Exception:
            return [], [], {}
        if not rows:
            return [], [], {}
        h = rows[0]
        old_ms: List[str] = []
        for col in h:
            if col.startswith("cpwer_"):
                old_ms.append(col[len("cpwer_"):])
        all_row_map: Dict[str, str] = {}
        for r in rows[1:]:
            if not r:
                continue
            if r[0] == "__ALL__":
                for i, col_name in enumerate(h):
                    all_row_map[col_name] = r[i] if i < len(r) else ""
                break
        return old_ms, h, all_row_map

    VARIANTS = [
        ("compare_summary.csv",        agg_all,    "all"),
        ("compare_summary_single.csv", agg_single, "single"),
        ("compare_summary_multi.csv",  agg_multi,  "multi"),
    ]

    for fname, this_agg, label in VARIANTS:
        cs_path = out_dir / fname

        if append_models:
            # -------- 追加模式 --------
            old_models, _old_header, old_row = _read_old_summary_csv(cs_path)
            # 最终列顺序: 旧模型顺序保留在前, append_models 里未出现在旧表的追加到右
            final_models = list(old_models)
            for m in append_models:
                if m not in final_models:
                    final_models.append(m)

            new_header = ["bucket"]
            for met in metrics:
                for m in final_models:
                    new_header.append(f"{met}_{m}")

            new_row = ["__ALL__"]
            for met in metrics:
                for m in final_models:
                    if m in append_models:
                        # 用现算的值; this_agg 里必然有 m (append 模式下 models=append_models)
                        v = _cell_val(this_agg[m], met)
                        cell = fmt(v, met)
                    else:
                        # 保留旧值
                        cell = old_row.get(f"{met}_{m}", "")
                    new_row.append(cell)

            with open(cs_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(new_header)
                w.writerow(new_row)
            kept = [m for m in old_models if m not in append_models]
            print(f"[OK] APPEND wrote {cs_path}  (kind={label})  "
                  f"kept={kept}  refreshed/added={append_models}")
        else:
            # -------- 全量模式 (原逻辑, 一字不动) --------
            header = ["bucket"]
            for met in metrics:
                for m in models:
                    header.append(f"{met}_{m}")

            def _build_row(label_, this_agg_):
                row = [label_]
                for met in metrics:
                    for m in models:
                        v = _cell_val(this_agg_[m], met)
                        row.append(fmt(v, met))
                return row

            with open(cs_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerow(_build_row("__ALL__", this_agg))
            print(f"[OK] wrote {cs_path}  (kind={label})")

    # ---- details.json ----
    #   全量模式: 写 details.json (覆盖)
    #   追加模式: 写 details.append.json (不覆盖原表, 仅含 append_models 的聚合)
    dj = {
        "specs": str(specs_path),
        "runs_root": str(runs_root),
        "models": models,
        "n_ref_specs": len(ref),
        "n_ref_specs_single": sum(1 for v in ref_kind.values() if v == "single_shot"),
        "n_ref_specs_multi":  sum(1 for v in ref_kind.values() if v == "multi_shot"),
        "agg_per_model_all":    agg_all,
        "agg_per_model_single": agg_single,
        "agg_per_model_multi":  agg_multi,
        "orc_metric": "greedy_orc_word_error_rate",
    }
    if append_models:
        dj["append_mode"] = True
        dj["append_models"] = append_models
    dj_filename = "details.append.json" if append_models else "details.json"
    (out_dir / dj_filename).write_text(
        json.dumps(dj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] wrote {out_dir/dj_filename}")

    # ---- 控制台漂亮打印 ----
    for label, this_agg in [("all", agg_all), ("single", agg_single), ("multi", agg_multi)]:
        print(f"\n================ compare_summary [{label}] (__ALL__) ================")
        print(f"{'metric':<12s} | " + " | ".join(f"{m:>20s}" for m in models))
        print("-" * (14 + 23 * len(models)))
        for met in metrics:
            cells = []
            for m in models:
                a = this_agg[m]
                if met == "cpwer":
                    v = (a["cp_e"] / a["cp_l"]) if a["cp_l"] > 0 else None
                    s = f"{v:.4f}" if v is not None else "n/a"
                elif met == "orcwer":
                    v = (a["orc_e"] / a["orc_l"]) if a["orc_l"] > 0 else None
                    s = f"{v:.4f}" if v is not None else "n/a"
                elif met == "delta_cp":
                    cp = (a["cp_e"] / a["cp_l"]) if a["cp_l"] > 0 else None
                    orc = (a["orc_e"] / a["orc_l"]) if a["orc_l"] > 0 else None
                    s = f"{cp - orc:.4f}" if (cp is not None and orc is not None) else "n/a"
                else:
                    s = str(a[met])
                cells.append(f"{s:>20s}")
            print(f"{met:<12s} | " + " | ".join(cells))
    print("\n注: cpwer / orcwer / delta_cp **越小越好**;")
    print("    orcwer 用 meeteval.greedy_orc_word_error_rate (避免 hyp 多 speaker 时指数复杂度);")
    print("    开源模型 (LTX-2.3 / Ovi / daVinci-MagiHuman) 不生成多镜头视频, 因此 ")
    print("    [multi] 表里它们的 n_specs 仍按 ref 计数, 但绝大多数 spec 都会落在 missing/empty_hyp.")


if __name__ == "__main__":
    main()
