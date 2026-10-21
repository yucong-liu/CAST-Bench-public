#!/usr/bin/env python3
"""
Segmentation-enhanced Subject / Background Consistency.

Pipeline: SAM 2 separates subject vs background per frame (temporally coherent
via video tracking), then subject region is scored with DINOv2 and background
region with CLIP. This fixes the original VBench flaw where a large/fast subject
contaminates the background score.

Usage:
    python run_segmented_consistency.py \
        --videos_path /path/to/videos \
        --sam2_ckpt   ./pretrained/sam2.1_hiera_large.pt \
        --sam2_cfg    configs/sam2.1/sam2.1_hiera_l.yaml \
        --subject_encoder dinov2 \
        --clip_model  ViT-B/32 \
        --device cuda \
        --output ./segmented_results.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from seg_consistency.common import collect_videos
from seg_consistency import compute_segmented_consistency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos_path', required=True)
    ap.add_argument('--sam2_ckpt', required=True, help='SAM 2 checkpoint .pt')
    ap.add_argument('--sam2_cfg', default='configs/sam2.1/sam2.1_hiera_l.yaml',
                    help='SAM 2 model config (relative to the installed sam2 package)')
    ap.add_argument('--subject_encoder', default='dinov2', choices=['dinov2', 'dino'])
    ap.add_argument('--clip_model', default='ViT-B/32',
                    help="CLIP for background; e.g. 'ViT-B/32' (VBench parity) or 'ViT-L/14'")
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--output', default='./segmented_results.json')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU (SAM2 will be slow).")
        args.device = 'cpu'

    assert os.path.isfile(args.sam2_ckpt), (
        f"SAM2 checkpoint not found: {args.sam2_ckpt}\n"
        f"Download e.g.:\n"
        f"  wget -P pretrained https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt")

    videos = collect_videos(args.videos_path)
    assert videos, f"No videos found under {args.videos_path}"
    print(f"Found {len(videos)} videos.")

    res = compute_segmented_consistency(
        videos, args.device, args.sam2_ckpt, args.sam2_cfg,
        subject_encoder=args.subject_encoder, clip_model=args.clip_model)

    print("\n==================================")
    print(f"Subject Consistency (masked, {args.subject_encoder}):    {res['subject_consistency_avg']:.4f}")
    print(f"Background Consistency (masked, CLIP {args.clip_model}): {res['background_consistency_avg']:.4f}")
    print("==================================")

    with open(args.output, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
