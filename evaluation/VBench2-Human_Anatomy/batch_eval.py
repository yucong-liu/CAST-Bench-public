#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_eval.py
=============
对一个模型 output 根目录下所有 mp4 计算 VBench2 Human_Anatomy 分数 (0~1, 越大越正常)。

设计要点
--------
1. **绕过上游 evaluate.py / VBench2.evaluate 的 prompt-driven 流程**:
   原版要求 videos_path 下文件名匹配 VBench2_full_info.json 里的 prompt
   (eg. "{prompt[:180]}-{0..2}.mp4"), 与本仓库 <root>/<category>/<spec_id>.mp4
   的命名约定不兼容。这里直接复用 vbench2.third_party.ViTDetector.detect
   中的 Detector + Analyzer, 自己手动循环。

2. **YOLO-World + 3 个 ViT 异常分类器只加载一次**, 后续所有视频复用模型,
   避免上游 compute_abnormality 每次都重建 model 导致的额外开销。

3. 递归扫 root 下所有 *.mp4, 把每个 mp4 的 "最后一级父目录名" 当作 folder
   (= category)。无论是 LTX-2.3 那种 <root>/<cat>/*.mp4 布局, 还是
   daVinci-MagiHuman 那种 <root>/540p/<cat>/*.mp4 布局, 都能正确归类。

4. 用 --categories 过滤只评测有人的 category (默认 speech,non_verbal_vocal,action),
   省去对纯物体/动物等 mp4 的无意义评测。

5. 单视频出错 (帧数 < 1 / 解码失败 / YOLO-World 检不到任何 human / 显存炸等)
   记录到 failed.txt, 继续跑剩下的。其中 "整段视频检不到 human" 在上游会
   触发 ZeroDivisionError, 我们把它单独捕获并记 score=NaN, 不计入均值。

6. 与 motion_smoothness / audiobox-aesthetics 输出格式对齐:
     results.csv : folder, video_path, human_anatomy
     summary.csv : folder, n, mean_human_anatomy
                   (+ __ALL__ 行, 可能 + __FAILED__ 行)

运行
----
    python3 batch_eval.py \\
        --root   /path/to/<model>/output \\
        --out_dir ./runs/<model> \\
        --device cuda \\
        --categories speech,non_verbal_vocal,action

权重就位 (默认 <工程根>/vbench2/vbench2_weights, 可由 VBENCH2_CACHE_DIR 覆盖):
    $CACHE/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth
    $CACHE/anomaly_detector/{human,face,hand}.pth
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

# --- 让 vbench2 包可被 import (这个目录就是工程根) ---
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# 抑制 TF 噪声 (与上游 detect.py 保持一致)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- compat: timm>=0.6 移除了 _pil_interp, 但上游 vbench2/third_party/ViTDetector/detect.py
# 仍在 import 它. 我们不动上游, 在入口提前把这个符号 monkey patch 回去. 这段补丁
# 在还存在 _pil_interp 的旧 timm 上不会触发, 双向兼容.
try:
    from timm.data.transforms import _pil_interp  # noqa: F401
except ImportError:
    import timm.data.transforms as _t_t  # type: ignore[no-redef]
    from PIL import Image as _PIL_Image
    _PIL_INTERP_MAP = {
        "bicubic": _PIL_Image.BICUBIC, "bilinear": _PIL_Image.BILINEAR,
        "nearest": _PIL_Image.NEAREST, "lanczos": _PIL_Image.LANCZOS,
    }
    def _pil_interp(method):  # type: ignore[no-redef]
        return _PIL_INTERP_MAP.get(str(method).lower(), _PIL_Image.BICUBIC)
    _t_t._pil_interp = _pil_interp  # type: ignore[attr-defined]
# --- end compat ---

import torch  # noqa: E402

# 默认权重缓存目录, 与 vbench2/utils.py 中的 CACHE_DIR 语义保持一致
DEFAULT_CACHE_DIR = Path(
    os.environ.get("VBENCH2_CACHE_DIR", str(Path.home() / ".cache" / "vbench2"))
).expanduser().resolve()

# Human_Anatomy 维度的 detector_config 路径是上游写死的 "vbench2/..." 相对路径,
# 需要工作目录是工程根, 否则 mmdet init_detector 找不到 config 文件。
DEFAULT_DETECTOR_CONFIG_REL = (
    "vbench2/third_party/YOLO-World/"
    "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
)
DEFAULT_ANALYZER_CFG_REL = (
    "vbench2/third_party/ViTDetector/simmim_finetune__vit_base__img224__800ep.yaml"
)


def build_submodules_dict(cache_dir: Path,
                          detector_config: str,
                          analyzer_cfg: str,
                          batch_size: int = 128) -> Dict:
    """手拼 submodules_dict, 与 vbench2.utils.init_submodules('Human_Anatomy') 等价,
    但**不触发 gdown 下载**: 缺权重时直接抛清晰的错误信息 (下载方式见 evaluation/README.md §1.3)。
    """
    yw_w = cache_dir / "YOLO-World" / \
        "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
    h_w = cache_dir / "anomaly_detector" / "human.pth"
    f_w = cache_dir / "anomaly_detector" / "face.pth"
    d_w = cache_dir / "anomaly_detector" / "hand.pth"

    missing = [str(p) for p in (yw_w, h_w, f_w, d_w) if not p.is_file()]
    if missing:
        msg = ("[ERROR] required Human_Anatomy weights not found:\n  - "
               + "\n  - ".join(missing)
               + "\n\nSee evaluation/README.md §1.3 for how to download them.\n")
        raise FileNotFoundError(msg)

    return {
        "detector_config": detector_config,
        "detector_weights": str(yw_w),
        "analyzer_configs": {
            "human": {"cfg_path": analyzer_cfg, "weight_path": str(h_w),
                      "threshold": 0.4545454545454546},
            "face":  {"cfg_path": analyzer_cfg, "weight_path": str(f_w),
                      "threshold": 0.30303030303030304},
            "hand":  {"cfg_path": analyzer_cfg, "weight_path": str(d_w),
                      "threshold": 0.3232},
        },
        "batch_size": batch_size,
    }


def collect_mp4s(root: Path, categories: List[str] | None) -> List[Tuple[str, Path]]:
    """递归找 root 下所有 mp4, 返回 [(folder, abs_path), ...]。

    - folder = 直接父目录名 (即 category, e.g. 'action' / 'speech')。
    - 若 categories 非空, 只保留 folder in categories 的视频。
    """
    cats = set(categories) if categories else None
    pairs: List[Tuple[str, Path]] = []
    for vp in sorted(root.rglob("*.mp4")):
        if not vp.is_file():
            continue
        folder = vp.parent.name
        if cats is not None and folder not in cats:
            continue
        pairs.append((folder, vp))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch VBench2 Human_Anatomy eval over all mp4 under <root>."
    )
    ap.add_argument("--root", type=str, required=True,
                    help="模型 output 根目录 (会递归找所有 mp4)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="输出目录 (会写 results.csv / summary.csv / failed.txt)")
    ap.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR),
                    help=f"VBench2 权重缓存目录 (默认 {DEFAULT_CACHE_DIR})")
    ap.add_argument("--detector_config", type=str,
                    default=str(THIS_DIR / DEFAULT_DETECTOR_CONFIG_REL),
                    help="YOLO-World detector mmdet config (.py)")
    ap.add_argument("--analyzer_cfg", type=str,
                    default=str(THIS_DIR / DEFAULT_ANALYZER_CFG_REL),
                    help="SimMIM ViT analyzer config (.yaml)")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="ViT 异常分类器 batch_size (人/face/hand 共用)")
    ap.add_argument("--categories", type=str,
                    default="speech,non_verbal_vocal,action",
                    help="只评测这些 category 子目录下的 mp4, 逗号分隔; 设为 'all' 则不过滤")
    ap.add_argument("--device", type=str, default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--fail_fast", action="store_true",
                    help="任何一段视频出错即中止 (默认是跳过继续跑)")
    # ---- 多卡分片 (run_all.sh 启动 N 个进程并行, 每个进程只跑一份切片) ----
    ap.add_argument("--num_shards", type=int, default=1,
                    help="将视频列表切成 N 份并行跑, 每个进程跑 1 份. 默认 1 = 不切片.")
    ap.add_argument("--shard_id", type=int, default=0,
                    help="当前进程负责哪一份切片 (0..num_shards-1).")
    args = ap.parse_args()

    # ---- 校验 shard 参数 ----
    if args.num_shards < 1:
        print(f"[ERROR] --num_shards must be >=1, got {args.num_shards}", file=sys.stderr)
        return 2
    if not (0 <= args.shard_id < args.num_shards):
        print(f"[ERROR] --shard_id must be in [0, {args.num_shards}), got {args.shard_id}",
              file=sys.stderr)
        return 2
    sharded = args.num_shards > 1
    shard_tag = f".shard{args.shard_id}" if sharded else ""

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    if not root.is_dir():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 2
    if not Path(args.detector_config).is_file():
        print(f"[ERROR] detector_config not found: {args.detector_config}",
              file=sys.stderr)
        return 2
    if not Path(args.analyzer_cfg).is_file():
        print(f"[ERROR] analyzer_cfg not found: {args.analyzer_cfg}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU (very slow).")
        args.device = "cpu"

    # --- 视频清单 ---
    cat_filter: List[str] | None = None
    if args.categories.strip().lower() != "all":
        cat_filter = [c.strip() for c in args.categories.split(",") if c.strip()]
        print(f"[info] category filter = {cat_filter}")
    else:
        print("[info] category filter = ALL (no filtering)")

    pairs_all = collect_mp4s(root, cat_filter)
    if not pairs_all:
        print(f"[WARN] no mp4 found under {root} (with category filter)")
        # 仍然写空表, 方便上层 compare 步骤判定
        (out_dir / f"results{shard_tag}.csv").write_text(
            "folder,video_path,human_anatomy\n", encoding="utf-8")
        if not sharded:
            (out_dir / "summary.csv").write_text(
                "folder,n,mean_human_anatomy\n", encoding="utf-8")
        return 0

    # 按 index % num_shards 切片 (轮询分配, 让各 shard 工作量更均衡)
    if sharded:
        pairs = [p for i, p in enumerate(pairs_all) if i % args.num_shards == args.shard_id]
        print(f"[info] shard {args.shard_id}/{args.num_shards}: "
              f"{len(pairs)} / {len(pairs_all)} mp4 assigned")
    else:
        pairs = pairs_all

    n_cats = len({f for f, _ in pairs})
    print(f"[info] root={root}")
    print(f"[info] cache_dir={cache_dir}")
    print(f"[info] found {len(pairs)} mp4 files in {n_cats} categories")

    # --- 权重检查 + submodules_dict ---
    submodules = build_submodules_dict(
        cache_dir=cache_dir,
        detector_config=args.detector_config,
        analyzer_cfg=args.analyzer_cfg,
        batch_size=args.batch_size,
    )

    # 上游 detector_config 在 init_detector 内部 cd 找文件时是按相对路径找的;
    # 强制工作目录切到工程根, 防止用户从别的地方调用我们时 mmdet 找不到 yolo_world cfg。
    os.chdir(THIS_DIR)

    # --- 加载模型 (只一次) ---
    print(f"[info] loading YOLO-World + 3 SimMIM-ViT detectors on {args.device} ...")
    t0 = time.time()
    # 延迟 import: 这两个 import 链路会拉起 mmcv/mmdet/mmyolo 整个全家桶, 早 import
    # 只为给个清晰的报错入口 (在 weight check 之后才 import, 让缺权重的报错先抛)。
    from vbench2.third_party.ViTDetector.detect import (  # noqa: E402
        Detector, Analyzer,
    )

    detector = Detector(
        config_file=submodules["detector_config"],
        weight_file=submodules["detector_weights"],
        device=args.device,
    )
    analyzer = Analyzer(
        model_configs=submodules["analyzer_configs"],
        device=args.device,
        batch_size=submodules["batch_size"],
        class_thresholds={k: v["threshold"]
                          for k, v in submodules["analyzer_configs"].items()},
    )
    print(f"[info] models ready ({time.time() - t0:.1f}s)")

    # --- per-video 打分 ---
    rows: List[Dict[str, str]] = []
    failed: List[Tuple[Path, str]] = []
    bucket: Dict[str, List[float]] = {}
    n_total = len(pairs)

    for idx, (folder, vp) in enumerate(pairs, 1):
        try:
            t1 = time.time()
            detections = detector.detect_video(str(vp))
            result = analyzer.analyze(
                video_path=str(vp), detection_results=detections,
            )
            score_f = float(result["video_results"])
            if math.isnan(score_f) or math.isinf(score_f):
                raise ValueError(f"non-finite score: {score_f}")
            dt = time.time() - t1
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            print(f"[{idx}/{n_total}] {folder}/{vp.name}  FAILED  ({msg})",
                  file=sys.stderr)
            failed.append((vp, msg))
            if args.fail_fast:
                raise
            if args.device == "cuda":
                torch.cuda.empty_cache()
            continue

        rows.append({
            "folder": folder,
            "video_path": str(vp),
            "human_anatomy": f"{score_f:.6f}",
        })
        bucket.setdefault(folder, []).append(score_f)
        print(f"[{idx}/{n_total}] {folder}/{vp.name}  -> {score_f:.4f}  ({dt:.1f}s)",
              flush=True)

    # --- 写 results.csv (分片模式下加 .shardN 后缀, 由 _merge_shards.py 合并) ---
    res_csv = out_dir / f"results{shard_tag}.csv"
    with res_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["folder", "video_path", "human_anatomy"])
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {res_csv}  rows={len(rows)}")

    # --- 写 summary.csv (仅非分片模式; 分片模式由 _merge_shards.py 跨分片重算) ---
    if not sharded:
        sum_csv = out_dir / "summary.csv"
        with sum_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["folder", "n", "mean_human_anatomy"])
            for folder in sorted(bucket.keys()):
                vs = bucket[folder]
                w.writerow([folder, len(vs),
                            f"{mean(vs):.6f}" if vs else ""])
            all_vs = [v for vs in bucket.values() for v in vs]
            if all_vs:
                w.writerow(["__ALL__", len(all_vs), f"{mean(all_vs):.6f}"])
            if failed:
                w.writerow(["__FAILED__", len(failed), ""])
        print(f"[write] {sum_csv}  folders={len(bucket)}  failed={len(failed)}")

    # --- failed.txt (分片模式下加 .shardN 后缀) ---
    if failed:
        fail_txt = out_dir / f"failed{shard_tag}.txt"
        with fail_txt.open("w", encoding="utf-8") as f:
            for vp, msg in failed:
                f.write(f"{vp}\t{msg}\n")
        print(f"[write] {fail_txt}  count={len(failed)}")

    print(f"\n[done] root={root}  ok={len(rows)}  failed={len(failed)}"
          + (f"  (shard {args.shard_id}/{args.num_shards})" if sharded else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
