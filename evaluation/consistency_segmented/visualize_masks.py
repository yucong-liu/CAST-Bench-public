#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_masks.py
==================
把 SAM2 在评测时实际产出的 subject mask 序列可视化出来，方便人工检查
"主体/背景分割是否正确"。

关键设计
--------
本脚本**完全复用** seg_consistency 包里的真实评测函数：
  - common.load_frames_rgb       (帧读取)
  - metrics._subsample_frames    (VBench 风格等距抽帧, 默认 32)
  - segmenter.SAM2Segmenter      (首帧自动选主体 + propagate)
  - segmenter.apply_mask         (主体路/背景路填灰逻辑)

因此可视化看到的 mask, 就是评测时 DINOv2 / CLIP 真正"看到"的输入,
1:1 对应, 不存在二次实现导致的偏差。

输出 (每个 shot 一个目录)
-------------------------
  <out_root>/<spec_id>/shot<k>/
      info.txt                        基本信息
      grid.png                        关键 5 帧 × 3 列(原图 / 主体路 / 背景路) 拼图
      frame_<i>_triple.png            单帧三联图
      frame_<i>_overlay.png           原图 + 红色半透明 mask 边界
      overlay.mp4                     (可选, --save_video) 全部抽样帧 mask 叠加视频

用法示例
--------
1) 看一个模型 manifest 里几个 spec:
   python3 visualize_masks.py \
       --manifest  .../audio_consistency/runs/qwen3_omni/LTX-2.3/shotcut_manifest.jsonl \
       --out_root  ./runs/LTX-2.3/_viz \
       --sam2_ckpt ./pretrained/sam2.1_hiera_large.pt \
       --sam2_cfg  configs/sam2.1/sam2.1_hiera_l.yaml \
       --spec_ids action_001 speech_003

2) 直接看一段 mp4:
   python3 visualize_masks.py \
       --clip /path/to/x.mp4 \
       --out_root ./runs/_viz_one \
       --sam2_ckpt ./pretrained/sam2.1_hiera_large.pt \
       --sam2_cfg  configs/sam2.1/sam2.1_hiera_l.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import torch  # noqa: E402

from seg_consistency.common import load_frames_rgb  # noqa: E402
from seg_consistency.metrics import _subsample_frames  # noqa: E402
from seg_consistency.segmenter import SAM2Segmenter, apply_mask  # noqa: E402
from seg_consistency.specs import SpecsLoader  # noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------- 绘图工具 -----------------------

def _overlay_mask_green(np_rgb: np.ndarray, mask: np.ndarray,
                        alpha: float = 0.45) -> np.ndarray:
    """在原图上叠加绿色半透明 mask, 并描出 mask 边界(亮黄)。

    与后续 `--use_text_prompt` 划取的 actor box 保持同调 (绿色 = actor),
    既表示“这是评测以为的主体”。
    """
    out = np_rgb.astype(np.float32).copy()
    green = np.zeros_like(out)
    green[..., 1] = 255.0
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * green[m]

    # 画 mask 边界 (简易膨胀差分)
    try:
        from scipy.ndimage import binary_dilation
        edge = binary_dilation(m, iterations=2) & (~m)
    except Exception:
        # 没有 scipy 就退化成上下左右移位求差
        dil = np.zeros_like(m)
        dil[1:, :] |= m[:-1, :]
        dil[:-1, :] |= m[1:, :]
        dil[:, 1:] |= m[:, :-1]
        dil[:, :-1] |= m[:, 1:]
        edge = dil & (~m)
    out[edge] = np.array([255.0, 255.0, 0.0])  # 黄色描边
    return np.clip(out, 0, 255).astype(np.uint8)


# Backward-compat alias (旧名字被别处引用的防护)
_overlay_mask_red = _overlay_mask_green


def _draw_box(np_rgb: np.ndarray,
              box_xyxy: Tuple[int, int, int, int],
              color: Tuple[int, int, int] = (0, 255, 0),
              width: int = 3) -> np.ndarray:
    """在图上画一个实线矩形 (默认 actor 绿色)."""
    img = Image.fromarray(np_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return np.array(img)


def _label_image(np_rgb: np.ndarray, text: str) -> np.ndarray:
    """在图片左上角写一段文字, 不依赖外部字体。"""
    img = Image.fromarray(np_rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 4
    # 黑底白字
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = 8 * len(text), 12
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def _hstack(imgs: List[np.ndarray], gap: int = 4,
            bg: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    H = max(im.shape[0] for im in imgs)
    out_w = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)
    canvas = np.full((H, out_w, 3), bg, dtype=np.uint8)
    x = 0
    for im in imgs:
        h, w = im.shape[:2]
        canvas[(H - h) // 2:(H - h) // 2 + h, x:x + w] = im
        x += w + gap
    return canvas


def _vstack(imgs: List[np.ndarray], gap: int = 4,
            bg: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    W = max(im.shape[1] for im in imgs)
    out_h = sum(im.shape[0] for im in imgs) + gap * (len(imgs) - 1)
    canvas = np.full((out_h, W, 3), bg, dtype=np.uint8)
    y = 0
    for im in imgs:
        h, w = im.shape[:2]
        canvas[y:y + h, (W - w) // 2:(W - w) // 2 + w] = im
        y += h + gap
    return canvas


def _save_png(path: Path, np_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np_rgb).save(str(path))


def _save_overlay_video(path: Path, frames: List[np.ndarray],
                        masks: List[np.ndarray], fps: int = 8) -> bool:
    """把每帧叠加 mask 后写成 mp4。用 imageio 优先, 退到 OpenCV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    overlays = [_overlay_mask_green(f, m) for f, m in zip(frames, masks)]
    try:
        import imageio.v2 as imageio  # noqa
        with imageio.get_writer(str(path), fps=fps, codec="libx264",
                                quality=8, macro_block_size=1) as w:
            for im in overlays:
                w.append_data(im)
        return True
    except Exception as e1:
        try:
            import cv2
            h, w = overlays[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
            for im in overlays:
                vw.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            vw.release()
            return True
        except Exception as e2:
            print(f"[warn] save_overlay_video failed: imageio={e1}; cv2={e2}")
            return False


# ----------------------- 单 shot 可视化 -----------------------

def visualize_one_clip(
        clip_path: str,
        out_dir: Path,
        seg: SAM2Segmenter,
        max_frames: int = 32,
        n_keyframes: int = 5,
        save_video: bool = False,
        spec_id: str = "",
        shot_idx: int = 0,
        actor_info: Optional[Dict[str, Any]] = None,
        actors_info: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """对一个 clip 跑一遍评测同款的 SAM2 流水线, 把 mask 画出来。

    如果传入 ``actors_info`` (来自 specs.extract_actors) 则对**每个** actor
    各跑一次 GroundingDINO + SAM2 box prompt 路径, 各自输出一组关键帧
    overlay; 同时输出一张 union overlay 看"全主体"。

    如果只传入 ``actor_info`` (单数, 旧调用方), 等价于 actors_info=[actor_info]。
    两者都不传则按 auto-mask 走 (最大近中心 blob)。
    """
    # normalize: actors_info wins; otherwise wrap actor_info; otherwise [None]
    if actors_info is not None and len(actors_info) > 0:
        actors_list: List[Optional[Dict[str, Any]]] = list(actors_info)
    elif actor_info is not None:
        actors_list = [actor_info]
    else:
        actors_list = [None]

    info: Dict[str, Any] = {
        "spec_id": spec_id, "shot_idx": shot_idx,
        "clip_path": clip_path, "out_dir": str(out_dir),
        "ok": False,
    }
    actor_summaries = [
        {
            "actor_id": (a or {}).get("actor_id"),
            "actor_type": (a or {}).get("actor_type"),
            "phrase_short": (a or {}).get("phrase_short"),
        }
        for a in actors_list if a is not None
    ]
    if actor_summaries:
        info["actors"] = actor_summaries
    if not os.path.exists(clip_path):
        info["reason"] = "clip not found"
        return info

    try:
        frames_full = load_frames_rgb(clip_path)
    except Exception as e:
        info["reason"] = f"load_failed: {e}"
        return info
    info["n_frames_total"] = len(frames_full)

    frames = _subsample_frames(frames_full,
                               max_frames if max_frames and max_frames > 0 else None)
    info["n_frames_used"] = len(frames)
    if len(frames) < 2:
        info["reason"] = f"too_short: {len(frames_full)} frames"
        return info

    T = len(frames)
    key_idx = sorted(set(np.linspace(0, T - 1, n_keyframes).round().astype(int).tolist()))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 与评测同款: 对每个 actor 分别跑 SAM2, 各产一组 mask
    per_actor_runs: List[Dict[str, Any]] = []
    union_masks: Optional[List[np.ndarray]] = None
    for ai in actors_list:
        masks, seed_meta = seg.subject_masks_for_video(
            frames, actor_info=ai, return_meta=True)
        # actor 在首帧上的 box (仅当走了 text-grounded 路径才有)
        actor_box = None
        if seed_meta and seed_meta.get("source") == "text_grounded":
            box = seed_meta.get("box_xyxy", [])
            if len(box) == 4:
                actor_box = tuple(box)

        # actor 文字标签
        if ai is not None:
            label_base = ai.get("label_for_viz") or (
                f'actor {ai.get("actor_id", "S?")}'
                f' ({ai.get("actor_type", "?")}):'
                f' "{ai.get("phrase_short", "?")}"'
            )
            if seed_meta and seed_meta.get("score") is not None:
                label_base += f"  conf={seed_meta['score']:.2f}"
            if seed_meta and seed_meta.get("source") not in (None, "text_grounded"):
                label_base += f"  [seed={seed_meta.get('source')}]"
        else:
            label_base = ""

        actor_id = (ai or {}).get("actor_id") or "auto"
        actor_tag = str(actor_id).replace("/", "_")

        # 关键帧 overlay
        for i in key_idx:
            ov = _overlay_mask_green(frames[i], masks[i])
            if i == 0 and actor_box is not None:
                ov = _draw_box(ov, actor_box, color=(0, 255, 0), width=3)
            text = f"t={i}/{T - 1}  area={float(masks[i].mean()):.3f}"
            if label_base:
                text = label_base + "  |  " + text
            ov = _label_image(ov, text)
            # 旧文件名 frame_xxx_overlay.png 在单 actor 时仍保留, 多 actor 时
            # 用 actor_id 加后缀, 防止互相覆盖。
            if len(actors_list) == 1:
                fname = f"frame_{i:03d}_overlay.png"
            else:
                fname = f"frame_{i:03d}__{actor_tag}_overlay.png"
            _save_png(out_dir / fname, ov)

        if save_video:
            vname = "overlay.mp4" if len(actors_list) == 1 else f"overlay__{actor_tag}.mp4"
            _save_overlay_video(out_dir / vname, frames, masks, fps=8)

        # 累计 union mask
        if union_masks is None:
            union_masks = [m.astype(bool).copy() for m in masks]
        else:
            union_masks = [u | m.astype(bool) for u, m in zip(union_masks, masks)]

        areas = [float(m.mean()) for m in masks]
        per_actor_runs.append({
            "actor_id": (ai or {}).get("actor_id"),
            "actor_type": (ai or {}).get("actor_type"),
            "phrase_short": (ai or {}).get("phrase_short"),
            "seed_meta": seed_meta,
            "mask_area_min": float(min(areas)),
            "mask_area_max": float(max(areas)),
            "mask_area_mean": float(sum(areas) / len(areas)),
        })

    # 多 actor 时, 额外输出一张 union 关键帧, 帮助直观看“全主体”
    if len(actors_list) > 1 and union_masks is not None:
        for i in key_idx:
            ov = _overlay_mask_green(frames[i], union_masks[i])
            text = f"UNION  t={i}/{T - 1}  area={float(union_masks[i].mean()):.3f}"
            ov = _label_image(ov, text)
            _save_png(out_dir / f"frame_{i:03d}__union_overlay.png", ov)
        if save_video:
            _save_overlay_video(out_dir / "overlay__union.mp4",
                                frames, union_masks, fps=8)

    # 兼容字段: 第一个 actor 的 seed_meta 与面积均值
    first = per_actor_runs[0] if per_actor_runs else {}
    info.update({
        "ok": True,
        "key_frames": [int(i) for i in key_idx],
        "n_actors": len(per_actor_runs),
        "per_actor": per_actor_runs,
        "seed_meta": first.get("seed_meta"),
        "mask_area_min": first.get("mask_area_min"),
        "mask_area_max": first.get("mask_area_max"),
        "mask_area_mean": first.get("mask_area_mean"),
    })
    with (out_dir / "info.txt").open("w", encoding="utf-8") as f:
        f.write(json.dumps(info, indent=2, ensure_ascii=False))
    return info

# ----------------------- 批量入口 -----------------------

def _load_manifest_entries(manifest_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            entries.append(e)
    return entries


def _filter_entries(entries: List[Dict[str, Any]],
                    spec_ids: List[str],
                    categories: List[str],
                    limit: int) -> List[Dict[str, Any]]:
    out = entries
    if spec_ids:
        sset = set(spec_ids)
        out = [e for e in out if e.get("spec_id") in sset]
    if categories:
        cset = set(categories)
        out = [e for e in out if e.get("category") in cset]
    # 跳过 manifest 已经标记不可评测的
    out = [e for e in out if e.get("shots") and e.get("passed", True)]
    if limit and limit > 0:
        out = out[:limit]
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize SAM2 subject masks "
                                             "actually used by the eval pipeline.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=str, default=None,
                     help="shotcut_manifest.jsonl 路径 (批量模式)")
    src.add_argument("--clip", type=str, default=None,
                     help="单个 mp4 路径 (单 clip 模式)")

    ap.add_argument("--out_root", type=str, required=True,
                    help="可视化结果输出根目录")
    ap.add_argument("--sam2_ckpt", type=str,
                    default=str(THIS_DIR / "pretrained" / "sam2.1_hiera_large.pt"))
    ap.add_argument("--sam2_cfg", type=str,
                    default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--device", type=str, default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--max_frames", type=int, default=32,
                    help="每个 shot 抽多少帧 (与评测保持一致, 默认 32)")
    ap.add_argument("--n_keyframes", type=int, default=5,
                    help="每个 shot 画几张关键帧 (默认 5)")
    ap.add_argument("--save_video", action="store_true",
                    help="同时输出全部抽样帧的 mask 叠加 mp4")

    ap.add_argument("--spec_ids", type=str, nargs="*", default=[],
                    help="只可视化这些 spec_id (manifest 模式)")
    ap.add_argument("--categories", type=str, nargs="*", default=[],
                    help="只可视化这些 category (manifest 模式)")
    ap.add_argument("--limit", type=int, default=0,
                    help="最多处理多少个 spec, 0=不限")
    ap.add_argument("--max_shots_per_spec", type=int, default=0,
                    help="每个 spec 最多画前 N 个 shot, 0=全部")
    # ---- text-grounded subject seeding (GroundingDINO) ----
    ap.add_argument("--specs_root", type=str,
                    default=str(_REPO_ROOT / "t2av-bench/specs"),
                    help="t2av-bench specs 根目录, 含 <category>.jsonl")
    ap.add_argument("--grounding_ckpt", type=str,
                    default=str(THIS_DIR / "pretrained" / "grounding-dino-tiny"),
                    help="GroundingDINO HF 权重目录")
    ap.add_argument("--use_text_prompt", action="store_true",
                    help="开启基于 specs.subjects 的文本提示主体选择 "
                         "(GroundingDINO -> SAM2 box). 可视化会在首帧画一个"
                         "绿色 box + actor 文字标签。")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sam2_ckpt = Path(args.sam2_ckpt).expanduser().resolve()
    if not sam2_ckpt.is_file():
        print(f"[ERROR] SAM2 ckpt not found: {sam2_ckpt}", file=sys.stderr)
        return 2

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU (will be VERY slow).")
        device = "cpu"

    print(f"[info] device       = {device}")
    print(f"[info] sam2_ckpt    = {sam2_ckpt}")
    print(f"[info] sam2_cfg     = {args.sam2_cfg}")
    print(f"[info] max_frames   = {args.max_frames}")
    print(f"[info] n_keyframes  = {args.n_keyframes}")
    print(f"[info] save_video   = {args.save_video}")
    print(f"[info] out_root     = {out_root}")
    print(f"[info] use_text_prompt = {args.use_text_prompt}")

    text_grounder = None
    specs_loader = None
    if args.use_text_prompt:
        if not Path(args.grounding_ckpt).is_dir():
            print(f"[ERROR] grounding_ckpt not a directory: {args.grounding_ckpt}",
                  file=sys.stderr)
            return 2
        if not Path(args.specs_root).is_dir():
            print(f"[ERROR] specs_root not a directory: {args.specs_root}",
                  file=sys.stderr)
            return 2
        from seg_consistency.text_grounder import TextGrounder
        print(f"[info] grounding_ckpt = {args.grounding_ckpt}")
        print(f"[info] specs_root     = {args.specs_root}")
        text_grounder = TextGrounder(args.grounding_ckpt, device=device)
        specs_loader = SpecsLoader(args.specs_root)

    seg = SAM2Segmenter(str(sam2_ckpt), args.sam2_cfg, device,
                        text_grounder=text_grounder)

    summary: List[Dict[str, Any]] = []

    if args.clip:
        out_dir = out_root / Path(args.clip).stem
        info = visualize_one_clip(
            args.clip, out_dir, seg,
            max_frames=args.max_frames,
            n_keyframes=args.n_keyframes,
            save_video=args.save_video,
            spec_id=Path(args.clip).stem, shot_idx=0,
            actor_info=None,
        )
        summary.append(info)
        print(f"[done] {out_dir} ok={info.get('ok')} reason={info.get('reason', '')}")
    else:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.is_file():
            print(f"[ERROR] manifest not found: {manifest_path}", file=sys.stderr)
            return 2
        entries = _load_manifest_entries(manifest_path)
        entries = _filter_entries(entries, args.spec_ids,
                                  args.categories, args.limit)
        print(f"[info] specs to visualize = {len(entries)}")

        for ei, spec in enumerate(entries, 1):
            spec_id = spec.get("spec_id", f"spec{ei:03d}")
            category = spec.get("category", "")
            shots = spec.get("shots", []) or []
            if args.max_shots_per_spec and args.max_shots_per_spec > 0:
                shots = shots[:args.max_shots_per_spec]

            actors_info: List[Dict[str, Any]] = []
            if specs_loader is not None:
                try:
                    actors_info = specs_loader.get_actors(
                        spec_id, category=category) or []
                except Exception as e:  # noqa: BLE001
                    print(f"    [warn] specs.get_actors({spec_id}) failed: {e}")
            actor_label = (
                " actors=[" + ", ".join(
                    f"{a.get('actor_id')}/{a.get('actor_type')}"
                    f" '{a.get('phrase_short')}'"
                    for a in actors_info
                ) + "]"
            ) if actors_info else ""
            print(f"[{ei}/{len(entries)}] {spec_id} ({category}) "
                  f"shots={len(shots)}{actor_label}")
            for s in shots:
                shot_idx = s.get("shot_idx", 0)
                clip_path = s.get("clip_path", "")
                out_dir = out_root / spec_id / f"shot{shot_idx:02d}"
                info = visualize_one_clip(
                    clip_path, out_dir, seg,
                    max_frames=args.max_frames,
                    n_keyframes=args.n_keyframes,
                    save_video=args.save_video,
                    spec_id=spec_id, shot_idx=shot_idx,
                    actors_info=actors_info,
                )
                summary.append(info)
                tag = "OK " if info.get("ok") else "FAIL"
                area = (f"area~{info.get('mask_area_mean', float('nan')):.3f}"
                        if info.get("ok") else info.get("reason", ""))
                # 多 actor 时把每个 seed source 都打出来
                pa = info.get("per_actor") or []
                if pa:
                    seed_tag = "  seeds=[" + ", ".join(
                        f"{(p.get('actor_id') or 'auto')}:"
                        f"{(p.get('seed_meta') or {}).get('source', '?')}"
                        + (f"({(p.get('seed_meta') or {}).get('score'):.2f})"
                           if isinstance((p.get('seed_meta') or {}).get('score'), float)
                           else "")
                        for p in pa
                    ) + "]"
                else:
                    seed_src = (info.get("seed_meta") or {}).get("source", "")
                    seed_score = (info.get("seed_meta") or {}).get("score")
                    seed_tag = (
                        f"  seed={seed_src}"
                        + (f" conf={seed_score:.2f}"
                           if isinstance(seed_score, float) else "")
                    ) if seed_src else ""
                print(f"    [{tag}] shot{shot_idx:02d}  {area}{seed_tag}")

            # GPU 释放一下碎片
            if device == "cuda":
                torch.cuda.empty_cache()

    with (out_root / "_viz_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[write] {out_root / '_viz_summary.json'}")
    print("[ALL DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
