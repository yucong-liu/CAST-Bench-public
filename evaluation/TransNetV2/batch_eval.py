#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量评测 TransNetV2 在视频生成模型 (LTX-2.3 / Ovi / daVinci-MagiHuman 等) 上
对镜头数 (shot count) 的预测正确性。

输出格式:
    results.csv : folder, video_path, spec_id, shot_count_kind,
                  expected_shots, predicted_shots, ratio, score
    summary.csv : folder, n, n_matched, mean_score   (含 __ALL__ 行)

真值来源:
    t2av-bench/specs/<category>.jsonl
        shot_count == "single_shot"   -> E = 1
        shot_count == "multi_shot"    -> E = len(spec["shots"])

预测来源:
    transnetv2_pytorch.TransNetV2 (社区 PyTorch 移植版)
        -> 输入 27x48 RGB 帧, 输出每帧 hard-cut 概率
        -> threshold=0.5 后 predictions_to_scenes, A = len(scenes)

打分公式 (线性相对误差):
    score = max(0, 1 - |A - E| / E)    (E > 0)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from transnetv2_pytorch import TransNetV2
except ImportError as e:
    raise ImportError(
        "需要先安装 transnetv2-pytorch: \n"
        "    pip install transnetv2-pytorch\n"
        "或用本仓库 inference-pytorch/ 目录中的源文件 + convert_weights.py 转换权重."
    ) from e


# ---------- spec 真值加载 ----------
def expected_shots_from_spec(spec: dict) -> Optional[int]:
    kind = spec.get("shot_count")
    if kind == "single_shot":
        return 1
    if kind == "multi_shot":
        shots = spec.get("shots") or []
        return len(shots) if len(shots) > 0 else None
    return None


def load_specs(specs_dir: Path) -> Tuple[Dict[str, int], Dict[str, str]]:
    expected_map: Dict[str, int] = {}
    kind_map: Dict[str, str] = {}
    for jp in sorted(specs_dir.glob("*.jsonl")):
        with jp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    spec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] {jp.name}:{line_no} json decode error: {e}", file=sys.stderr)
                    continue
                sid = spec.get("id")
                if not sid:
                    continue
                E = expected_shots_from_spec(spec)
                if E is None:
                    continue
                expected_map[sid] = E
                kind_map[sid] = spec.get("shot_count", "")
    return expected_map, kind_map


# ---------- mp4 收集 ----------
def find_mp4s_recursive(root_dir: Path) -> List[Path]:
    return sorted(root_dir.rglob("*.mp4"))


# ---------- 打分 ----------
def linear_score(A: int, E: int) -> float:
    if E <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(A - E) / float(E))


# ---------- ffmpeg 解码 -> 27x48x3 RGB ----------
def extract_frames_27x48(video_path: str, ffmpeg_bin: str = "ffmpeg") -> np.ndarray:
    """与 TransNetV2 官方 inference 保持一致: 48x27 rgb24 rawvideo."""
    cmd = [
        ffmpeg_bin, "-loglevel", "error", "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "48x27", "pipe:",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    frames = np.frombuffer(proc.stdout, np.uint8).reshape([-1, 27, 48, 3])
    return frames


# ---------- 滑窗推理 (复刻官方逻辑: 100帧窗口 / 步长50 / 取中间50) ----------
@torch.no_grad()
def predict_frames(model: TransNetV2, frames: np.ndarray, device: str) -> np.ndarray:
    """返回每帧的 single_frame_pred (sigmoid 后), shape=[n_frames]."""
    no_padded_start = 25
    pad_end_extra = 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
    no_padded_end = 25 + pad_end_extra

    start_frame = np.expand_dims(frames[0], 0)
    end_frame = np.expand_dims(frames[-1], 0)
    padded = np.concatenate(
        [start_frame] * no_padded_start + [frames] + [end_frame] * no_padded_end, 0
    )

    preds = []
    ptr = 0
    while ptr + 100 <= len(padded):
        chunk = padded[ptr:ptr + 100]  # [100, 27, 48, 3] uint8
        ptr += 50
        inp = torch.from_numpy(chunk).unsqueeze(0).to(device)  # [1, 100, 27, 48, 3]
        single_logits, all_dict = model(inp)
        single = torch.sigmoid(single_logits).cpu().numpy()[0, 25:75, 0]
        preds.append(single)
    single_frame_pred = np.concatenate(preds)[:len(frames)]
    return single_frame_pred


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """与 TransNetV2 官方一致: 1 = 切点帧, 把 0 段聚合成 scene."""
    p = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0
    last_i = 0
    for i, t in enumerate(p):
        last_i = i
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, last_i])
    if len(scenes) == 0:
        return np.array([[0, len(p) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


# ---------- 模型加载 ----------
def _builtin_weights_path() -> Optional[str]:
    """transnetv2-pytorch pip 包内置权重路径 (site-packages/transnetv2_pytorch/transnetv2-pytorch-weights.pth)."""
    try:
        import transnetv2_pytorch as _p
        wpath = os.path.join(os.path.dirname(_p.__file__), "transnetv2-pytorch-weights.pth")
        return wpath if os.path.isfile(wpath) else None
    except Exception:
        return None


def load_model(weights_path: str, device: str) -> TransNetV2:
    model = TransNetV2()
    if weights_path and os.path.isfile(weights_path):
        sd = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        print(f"[INFO] loaded weights from {weights_path}")
    else:
        builtin = _builtin_weights_path()
        if builtin is None:
            raise FileNotFoundError(
                "未提供 --weights 且 transnetv2-pytorch 包内未找到 "
                "transnetv2-pytorch-weights.pth, 请显式传入 --weights /path/to/.pth"
            )
        sd = torch.load(builtin, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        print(f"[INFO] loaded built-in weights from {builtin}")
    model.eval().to(device)
    return model


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(
        "Batch eval TransNetV2 shot-count accuracy on generated videos."
    )
    parser.add_argument("--root", type=str, required=True,
                        help="模型生成视频根目录, 递归扫描 *.mp4")
    parser.add_argument("--specs_dir", type=str, required=True,
                        help="t2av-bench/specs 目录")
    parser.add_argument("--weights", type=str, default="",
                        help="可选: TransNetV2 PyTorch 权重 .pth 路径; 留空则用 pip 包内置权重")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="single_frame_pred 二值化阈值, 默认 0.5 (官方默认)")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg",
                        help="ffmpeg 可执行路径, 默认走 PATH")
    parser.add_argument("--save_csv", type=str, default="",
                        help="results.csv 输出路径")
    parser.add_argument("--save_summary_csv", type=str, default="",
                        help="summary.csv 输出路径")
    parser.add_argument("--shot_count_kind", type=str, default="multi",
                        choices=["all", "multi"],
                        help="只评测指定 shot_count_kind 的样本: "
                             "multi=仅 multi_shot (默认); all=全样本. "
                             "single_shot 样本一律不参与评测. "
                             "未匹配 spec 的视频在 multi 模式下会被跳过.")
    parser.add_argument("--fail_fast", action="store_true",
                        help="出错立即抛出 (默认遇到失败视频跳过并记录)")
    args = parser.parse_args()

    kind_filter = args.shot_count_kind  # all / multi
    target_kind = {
        "all":   None,
        "multi": "multi_shot",
    }[kind_filter]

    root_dir = Path(args.root).expanduser().resolve()
    specs_dir = Path(args.specs_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root not found: {root_dir}")
    if not specs_dir.exists():
        raise FileNotFoundError(f"specs_dir not found: {specs_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device     = {device}")
    print(f"[INFO] root       = {root_dir}")
    print(f"[INFO] specs_dir  = {specs_dir}")
    print(f"[INFO] weights    = {args.weights or '(builtin)'}")
    print(f"[INFO] threshold  = {args.threshold}")

    # 1) specs
    expected_map, kind_map = load_specs(specs_dir)
    print(f"[INFO] loaded specs: {len(expected_map)} entries with valid expected_shots")

    # 2) model
    print("[INFO] loading TransNetV2 model ...")
    model = load_model(args.weights, device)
    print("[INFO] model ready.")

    # 3) videos
    mp4s = find_mp4s_recursive(root_dir)
    if not mp4s:
        print(f"[WARN] no mp4 found under {root_dir}, exit.")
        return
    print(f"[INFO] found {len(mp4s)} mp4 files under {root_dir}")

    # 4) infer
    per_video_rows: List[Dict] = []
    folder_scores: Dict[str, List[float]] = {}
    folder_total: Dict[str, int] = {}

    for idx, vp in enumerate(mp4s, 1):
        spec_id = vp.stem
        folder = vp.parent.name
        E = expected_map.get(spec_id)
        kind = kind_map.get(spec_id, "")

        # shot_count_kind 过滤: single/multi 模式下跳过不匹配 (含无 spec) 的视频
        if target_kind is not None and kind != target_kind:
            continue

        folder_total[folder] = folder_total.get(folder, 0) + 1

        try:
            frames = extract_frames_27x48(str(vp), args.ffmpeg)
            if len(frames) < 2:
                raise RuntimeError(f"too few frames decoded: {len(frames)}")
            preds = predict_frames(model, frames, device)
            scenes = predictions_to_scenes(preds, threshold=args.threshold)
            A = int(len(scenes))
        except Exception as e:
            msg = f"[ERR] inference failed: {vp} | {repr(e)}"
            if args.fail_fast:
                raise
            print(msg, file=sys.stderr)
            per_video_rows.append({
                "folder": folder, "video_path": str(vp), "spec_id": spec_id,
                "shot_count_kind": kind,
                "expected_shots": E if E is not None else "",
                "predicted_shots": "", "ratio": "", "score": "",
            })
            continue

        if E is None:
            print(f"[{idx}/{len(mp4s)}] {vp.name}: A={A} (no spec, skip in stats)")
            per_video_rows.append({
                "folder": folder, "video_path": str(vp), "spec_id": spec_id,
                "shot_count_kind": kind, "expected_shots": "",
                "predicted_shots": A, "ratio": "", "score": "",
            })
            continue

        s = linear_score(A, E)
        ratio_str = f"{A}/{E}"
        folder_scores.setdefault(folder, []).append(s)
        per_video_rows.append({
            "folder": folder, "video_path": str(vp), "spec_id": spec_id,
            "shot_count_kind": kind, "expected_shots": E,
            "predicted_shots": A, "ratio": ratio_str, "score": f"{s:.4f}",
        })
        print(f"[{idx}/{len(mp4s)}] {vp.name}: A={A} E={E} ratio={ratio_str} score={s:.4f}")

    # 5) results.csv
    if args.save_csv:
        out_csv = Path(args.save_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "folder", "video_path", "spec_id", "shot_count_kind",
            "expected_shots", "predicted_shots", "ratio", "score",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_video_rows)
        print(f"[write] {out_csv}  rows={len(per_video_rows)}")

    # 6) summary.csv 写出
    #
    # multi 模式 (默认): per_video_rows 已是 multi_shot 过滤后的子集,
    #   只写一份 summary 到用户指定路径.
    # all 模式 (向后兼容): 同时输出全样本 summary 和
    #   summary_multi (仅 multi_shot 子集) 两表. single_shot 不再单独拆表.
    if args.save_summary_csv:
        out_sum = Path(args.save_summary_csv).expanduser().resolve()
        out_sum.parent.mkdir(parents=True, exist_ok=True)

        def _write_summary_csv(rows: List[Dict],
                               total_filter,
                               score_filter,
                               path: Path):
            f_total: Dict[str, int] = {}
            f_scores: Dict[str, List[float]] = {}
            for r in rows:
                folder = r["folder"]
                if total_filter(r):
                    f_total[folder] = f_total.get(folder, 0) + 1
                if score_filter(r):
                    try:
                        s = float(r["score"])
                    except (ValueError, TypeError):
                        continue
                    f_scores.setdefault(folder, []).append(s)
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["folder", "n", "n_matched", "mean_score"])
                all_s: List[float] = []
                for folder in sorted(f_total.keys()):
                    n = f_total[folder]
                    scores = f_scores.get(folder, [])
                    n_matched = len(scores)
                    mean_s = f"{statistics.mean(scores):.4f}" if n_matched > 0 else ""
                    w.writerow([folder, n, n_matched, mean_s])
                    all_s.extend(scores)
                n_all = sum(f_total.values())
                n_all_matched = len(all_s)
                mean_all = f"{statistics.mean(all_s):.4f}" if n_all_matched > 0 else ""
                w.writerow(["__ALL__", n_all, n_all_matched, mean_all])
            print(f"[write] {path}")

        # multi 模式: per_video_rows 已是过滤后的子集, 只写一份 summary.
        if kind_filter != "all":
            _write_summary_csv(
                per_video_rows,
                total_filter=lambda r: True,
                score_filter=lambda r: r.get("score") not in ("", None),
                path=out_sum,
            )
            return

        # all 模式: 产出全样本 summary + multi 子集 summary (single_shot 不再拆)
        # (1) 全样本
        _write_summary_csv(
            per_video_rows,
            total_filter=lambda r: True,
            score_filter=lambda r: r.get("score") not in ("", None),
            path=out_sum,
        )
        # (2) multi 子集
        mlt_path = out_sum.with_name(out_sum.stem + "_multi" + out_sum.suffix)
        _write_summary_csv(
            per_video_rows,
            total_filter=lambda r: r.get("shot_count_kind") == "multi_shot",
            score_filter=lambda r: (
                r.get("shot_count_kind") == "multi_shot"
                and r.get("score") not in ("", None)
            ),
            path=mlt_path,
        )


if __name__ == "__main__":
    main()
