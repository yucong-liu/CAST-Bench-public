"""ocr_pipeline · 从 AVGen-Bench 原版 OCR 脚本 拆出的 OCR stage1 工具库

提供:
- OCR 引擎单例与逐帧推理 (ocr_engine.OCR_ENGINE / ocr_frames / ocr_frame)
- 同帧合并 + IoU 追踪 + summarize_tracks (tracking)
- run_stage1(video_path) -> payload (主流程, 完全照搬 原版 run 的 stage 1 段)
"""
from .ocr_engine import (  # noqa: F401
    OCR_ENGINE,
    OCR_DEVICE,
    OCR_BATCH_SIZE,
    STRIDE,
    MAX_FRAMES,
    START_FRAME,
    MIN_OCR_CONF,
    ocr_frame,
    ocr_frames,
)
from .tracking import (  # noqa: F401
    Det,
    Track,
    poly_to_aabb,
    normalize_aabb,
    merge_dets_in_frame,
    build_tracks,
    summarize_tracks,
    run_stage1,
    IOU_THRESH,
    MAX_GAP_FRAMES,
    MIN_TRACK_FRAMES,
    MAX_TRACKS_TO_SEND,
    TEXT_SIM_THRESH,
    MERGE_Y_IOU_THRESH,
    MERGE_X_GAP_NORM,
    MERGE_MAX_HEIGHT_RATIO,
)
