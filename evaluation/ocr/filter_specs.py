#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr · filter_specs.py

扫描 t2av-bench/specs/*.jsonl, 按 text_on_screen != [] 过滤出
"该视频应渲染屏幕文字" 的子集, 作为 OCR 评测点的 spec 子集落盘到
evaluation/ocr/ocr_specs.jsonl.

输出 jsonl 的每条记录额外加上 "_source_file" 字段, 方便回溯;
其余字段原样保留 (不丢 prompt / scene_setting / shots / text_on_screen).

用法:
    python filter_specs.py                            # 写入默认输出
    python filter_specs.py --dry-run                  # 只统计, 不落盘
    python filter_specs.py --specs-dir <dir> --out <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Tuple
_REPO_ROOT = Path(__file__).resolve().parents[2]


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SPECS_DIR = (_REPO_ROOT / "t2av-bench/specs")
DEFAULT_OUT = THIS_DIR / "ocr_specs.jsonl"


def iter_specs_files(specs_dir: Path) -> Iterator[Path]:
    """枚举 specs/*.jsonl, 跳过 _patch/ 子目录与 schema.json."""
    if not specs_dir.is_dir():
        raise FileNotFoundError(f"specs dir not found: {specs_dir}")
    for p in sorted(specs_dir.glob("*.jsonl")):
        # 顶层 jsonl 即可; _patch 下的不要 (历史快照)
        if p.is_file():
            yield p


def iter_specs_in_file(path: Path) -> Iterator[Tuple[int, dict]]:
    """逐行解析 jsonl, 返回 (lineno, spec). 跳过空行与解析失败行 (打 warning)."""
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[warn] {path.name}:L{ln} json decode failed: {e}",
                    file=sys.stderr,
                )


def has_screen_text(spec: dict) -> bool:
    """判断该 spec 是否带有有效的屏幕文字标签.

    text_on_screen 为非空 list, 且至少有 1 项含非空 text 字段.
    """
    tos = spec.get("text_on_screen")
    if not isinstance(tos, list) or not tos:
        return False
    for item in tos:
        if not isinstance(item, dict):
            continue
        txt = item.get("text")
        if isinstance(txt, str) and txt.strip():
            return True
    return False


def kind_of(item: dict) -> str:
    k = item.get("kind") if isinstance(item, dict) else None
    return k if isinstance(k, str) and k else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--specs-dir",
        type=Path,
        default=DEFAULT_SPECS_DIR,
        help=f"t2av-bench/specs 目录, 默认 {DEFAULT_SPECS_DIR}",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"输出 jsonl, 默认 {DEFAULT_OUT}",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计, 不写文件",
    )
    args = ap.parse_args()

    specs_dir: Path = args.specs_dir
    out_path: Path = args.out

    print(f"[filter_specs] specs-dir = {specs_dir}")
    print(f"[filter_specs] out       = {out_path}{' (DRY-RUN, 不会写)' if args.dry_run else ''}")
    print()

    per_file_total: dict[str, int] = {}
    per_file_hit: dict[str, int] = {}
    kind_counter: Counter[str] = Counter()
    lang_counter: Counter[str] = Counter()
    hits: list[dict] = []

    for spath in iter_specs_files(specs_dir):
        cat = spath.stem  # speech / action / ...
        n_total = 0
        n_hit = 0
        for _ln, spec in iter_specs_in_file(spath):
            n_total += 1
            if not has_screen_text(spec):
                continue
            n_hit += 1
            spec_out = dict(spec)  # 保留原字段
            spec_out["_source_file"] = spath.name
            spec_out.setdefault("category", cat)
            hits.append(spec_out)
            for tos in spec.get("text_on_screen") or []:
                kind_counter[kind_of(tos)] += 1
                lang = tos.get("language") if isinstance(tos, dict) else None
                if isinstance(lang, str) and lang:
                    lang_counter[lang] += 1
        per_file_total[cat] = n_total
        per_file_hit[cat] = n_hit

    print("=" * 64)
    print(f"{'category':<22s} {'total':>8s} {'hit':>8s} {'hit_rate':>10s}")
    print("-" * 64)
    grand_total = 0
    grand_hit = 0
    for cat in sorted(per_file_total.keys()):
        t = per_file_total[cat]
        h = per_file_hit[cat]
        grand_total += t
        grand_hit += h
        rate = (100.0 * h / t) if t > 0 else 0.0
        print(f"{cat:<22s} {t:>8d} {h:>8d} {rate:>9.1f}%")
    print("-" * 64)
    rate_g = (100.0 * grand_hit / grand_total) if grand_total > 0 else 0.0
    print(f"{'__ALL__':<22s} {grand_total:>8d} {grand_hit:>8d} {rate_g:>9.1f}%")
    print("=" * 64)
    print()

    print("[text_on_screen.kind 分布 (按文字段计)]")
    for k, v in kind_counter.most_common():
        print(f"  {k:<16s} {v:>6d}")
    print()
    print("[text_on_screen.language 分布 (按文字段计)]")
    for k, v in lang_counter.most_common():
        print(f"  {k:<16s} {v:>6d}")
    print()

    if args.dry_run:
        print("[dry-run] 不写文件, exit 0")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fo:
        for spec in hits:
            fo.write(json.dumps(spec, ensure_ascii=False) + "\n")
    print(f"[done] 写入 {out_path}  共 {len(hits)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
