#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/_common/spec_kind.py
=================================

公共 helper：把 t2av-bench specs 里的 ``shot_count`` 字段抽出来，做成
``spec_id -> 'single_shot' | 'multi_shot'`` 的映射，方便 evaluation 下所有
评测点在汇总阶段把"全样本"再切成"single 子表"和"multi 子表"。

设计原则
--------
* 只读 specs，不污染 evaluation 之外的代码；
* 不依赖 batch_eval 输出的列。即便某评测点的 results.csv 没有
  ``shot_count_kind`` 列，只要能拿到 ``spec_id`` (或视频文件 stem)，
  就可以从这里反查出 kind；
* 视频命名约定固定为 ``{spec_id}.mp4``，因此 ``stem_to_kind`` 直接吃文件名 stem。

约定
----
* 默认 specs 目录：
  ``t2av-bench/specs``
  可由环境变量 ``T2AV_SPECS_DIR`` 覆盖；
* ``shot_count`` 取值约定：``"single_shot"`` 或 ``"multi_shot"``。
  其余值 / 缺失 -> 标记为 ``""`` (即 unknown)，由调用方决定怎么处理。

典型用法
--------
.. code-block:: python

    from spec_kind import load_spec_kind_map, KIND_SINGLE, KIND_MULTI

    kind_map = load_spec_kind_map()  # {'speech_0001': 'multi_shot', ...}
    kind = kind_map.get(spec_id, "")
    if kind == KIND_SINGLE:
        ...
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

KIND_SINGLE = "single_shot"
KIND_MULTI = "multi_shot"
KIND_UNKNOWN = ""  # 既不属于 single 也不属于 multi，调用方按需处理

ALL_KINDS = (KIND_SINGLE, KIND_MULTI)

DEFAULT_SPECS_DIR = Path(
    os.environ.get(
        "T2AV_SPECS_DIR",
        str(_REPO_ROOT / "t2av-bench/specs"),
    )
)


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_spec_kind_map(specs_dir: Optional[Path | str] = None,
                       verbose: bool = False) -> Dict[str, str]:
    """扫 ``<specs_dir>/*.jsonl``，返回 ``{spec_id: shot_count_kind}``。

    * ``shot_count`` 字段只接受 ``single_shot`` / ``multi_shot``；
      其它值 / 缺失会被丢到 ``KIND_UNKNOWN`` (空串)。
    * 跳过 ``_patch`` 之类的子目录(只读顶层 jsonl)。
    * 同 spec_id 出现多次时，后写入的覆盖先写入的。
    """
    if specs_dir is None:
        specs_dir = DEFAULT_SPECS_DIR
    specs_dir = Path(specs_dir).expanduser().resolve()

    out: Dict[str, str] = {}
    if not specs_dir.is_dir():
        if verbose:
            print(f"[spec_kind] specs_dir not found: {specs_dir}", file=sys.stderr)
        return out

    n_files = 0
    for jp in sorted(specs_dir.glob("*.jsonl")):
        # 不下钻 _patch / _backup 等隐藏/补丁目录 (glob 本身只取顶层文件,
        # 这里再保险过滤一下名字以下划线开头的文件)
        if jp.name.startswith("_"):
            continue
        n_files += 1
        try:
            with jp.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        spec = json.loads(line)
                    except json.JSONDecodeError as e:
                        if verbose:
                            print(
                                f"[spec_kind] {jp.name}:{line_no} json decode error: {e}",
                                file=sys.stderr,
                            )
                        continue
                    sid = spec.get("id")
                    if not sid:
                        continue
                    raw_kind = spec.get("shot_count", "")
                    if raw_kind in ALL_KINDS:
                        out[sid] = raw_kind
                    else:
                        out[sid] = KIND_UNKNOWN
        except OSError as e:
            if verbose:
                print(f"[spec_kind] cannot open {jp}: {e}", file=sys.stderr)

    if verbose:
        n_single = sum(1 for v in out.values() if v == KIND_SINGLE)
        n_multi = sum(1 for v in out.values() if v == KIND_MULTI)
        n_unknown = sum(1 for v in out.values() if v == KIND_UNKNOWN)
        print(
            f"[spec_kind] loaded {len(out)} specs from {n_files} jsonl under {specs_dir}: "
            f"single={n_single} multi={n_multi} unknown={n_unknown}",
            file=sys.stderr,
        )
    return out


# ---------------------------------------------------------------------------
# 反查辅助
# ---------------------------------------------------------------------------

def kind_of(spec_id: str, kind_map: Dict[str, str]) -> str:
    """根据 spec_id 反查 kind；spec_id 缺失时返回 ``KIND_UNKNOWN``。"""
    if not spec_id:
        return KIND_UNKNOWN
    return kind_map.get(spec_id, KIND_UNKNOWN)


def stem_to_kind(video_path_or_stem: str, kind_map: Dict[str, str]) -> str:
    """从 video_path / 文件名 stem (eg. ``action_0001.mp4`` 或 ``action_0001``)
    反查 kind。

    适用于 results.csv 没有 spec_id 列、只有 video_path 的评测点。
    """
    if not video_path_or_stem:
        return KIND_UNKNOWN
    name = os.path.basename(str(video_path_or_stem))
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    elif name.lower().endswith(".wav"):
        name = name[:-4]
    return kind_map.get(name, KIND_UNKNOWN)


def split_spec_ids_by_kind(spec_ids: Iterable[str],
                           kind_map: Dict[str, str]):
    """把一组 spec_id 拆成 (single_set, multi_set, unknown_set)，便于做汇总过滤。"""
    s, m, u = set(), set(), set()
    for sid in spec_ids:
        k = kind_of(sid, kind_map)
        if k == KIND_SINGLE:
            s.add(sid)
        elif k == KIND_MULTI:
            m.add(sid)
        else:
            u.add(sid)
    return s, m, u


# ---------------------------------------------------------------------------
# CLI: 方便手动检查 specs 里 single/multi 的分布
# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="dump spec_id -> shot_count from t2av-bench specs")
    ap.add_argument("--specs-dir", default=str(DEFAULT_SPECS_DIR))
    ap.add_argument("--show", choices=["count", "single", "multi", "unknown", "all"],
                    default="count")
    args = ap.parse_args()

    m = load_spec_kind_map(args.specs_dir, verbose=True)
    if args.show == "count":
        # 只走 verbose 输出即可，已经打印了 single/multi/unknown 计数
        return 0

    if args.show == "all":
        for sid in sorted(m.keys()):
            print(f"{sid}\t{m[sid]}")
        return 0

    target = {
        "single": KIND_SINGLE,
        "multi": KIND_MULTI,
        "unknown": KIND_UNKNOWN,
    }[args.show]
    for sid in sorted(m.keys()):
        if m[sid] == target:
            print(sid)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
