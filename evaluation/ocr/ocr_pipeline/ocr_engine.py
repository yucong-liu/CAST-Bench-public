"""ocr_pipeline.ocr_engine
========================

从 AVGen-Bench 原版 OCR 脚本 原样搬运的 stage1 PaddleOCR 部分:
- 全部环境变量名/默认值/合并阈值/追踪阈值与原工程完全一致
- 进程级 PaddleOCR 单例 (按 OCR_DEVICE 选 GPU/CPU)
- ocr_frames(batch) / ocr_frame(single) 推理封装

注意: 该模块在 import 时即会按 OCR_DEVICE 加载 PaddleOCR; 子进程 import 前
应先 setdefault('CUDA_VISIBLE_DEVICES', '0') 把卡绑好.
"""
from __future__ import annotations

import os
from typing import Any, List, Tuple

import numpy as np
import paddle
from paddleocr import PaddleOCR


# =========================
# Env / Config (与 AVGen-Bench 原版 OCR 脚本 一字一码对齐)
# =========================
# OCR sampling (speed vs accuracy)
STRIDE = int(os.getenv("OCR_STRIDE", "3"))
MAX_FRAMES = int(os.getenv("OCR_MAX_FRAMES", "900"))
START_FRAME = int(os.getenv("OCR_START_FRAME", "0"))

# Same-frame merge params
MERGE_Y_IOU_THRESH = float(os.getenv("OCR_MERGE_Y_IOU_THRESH", "0.5"))
MERGE_X_GAP_NORM = float(os.getenv("OCR_MERGE_X_GAP_NORM", "0.02"))
MERGE_MAX_HEIGHT_RATIO = float(os.getenv("OCR_MERGE_MAX_HEIGHT_RATIO", "2.5"))

# Tracking params
IOU_THRESH = float(os.getenv("OCR_IOU_THRESH", "0.3"))
MAX_GAP_FRAMES = int(os.getenv("OCR_MAX_GAP_FRAMES", "5"))
MIN_TRACK_FRAMES = int(os.getenv("OCR_MIN_TRACK_FRAMES", "3"))
MAX_TRACKS_TO_SEND = int(os.getenv("OCR_MAX_TRACKS_TO_SEND", "40"))
MIN_OCR_CONF = float(os.getenv("OCR_MIN_CONF", "0.2"))

# Text segmentation
TEXT_SIM_THRESH = float(os.getenv("OCR_TEXT_SIM_THRESH", "0.75"))


# OCR device selection
OCR_DEVICE = os.getenv(
    "OCR_DEVICE",
    "gpu:0" if paddle.is_compiled_with_cuda() else "cpu",
).strip()
try:
    paddle.device.set_device(OCR_DEVICE)
except Exception:
    OCR_DEVICE = "cpu"
    paddle.device.set_device(OCR_DEVICE)

OCR_BATCH_SIZE = max(
    1,
    int(os.getenv("OCR_BATCH_SIZE", "16" if OCR_DEVICE.startswith("gpu") else "1")),
)
OCR_TEXT_RECOGNITION_BATCH_SIZE = max(
    1,
    int(os.getenv("OCR_TEXT_RECOGNITION_BATCH_SIZE", str(OCR_BATCH_SIZE))),
)

# OCR init:
#   - 与 MSAVBench 官方 pp_ocrv5/server.py 完全对齐:
#       ocr_version=PP-OCRv5, lang="ch", text_rec_score_thresh=0.5, use_textline_orientation=True.
#   - 注意: PaddleOCR 里 lang="ch" 加载的是 PP-OCRv5_server_rec (多语言 server 权重, 中英日韩均能识别),
#     并不是"仅中文"; 而 lang="en" 会加载 en_PP-OCRv5_mobile_rec (英文 mobile 小模型), 对艺术字/霓虹字
#     几乎完全崩溃(实测 30 条视频全部 0 检出). 因此坚持用官方的 "ch".
OCR_ENGINE = PaddleOCR(
    ocr_version="PP-OCRv5",
    lang="ch",
    text_rec_score_thresh=0.5,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    text_recognition_batch_size=OCR_TEXT_RECOGNITION_BATCH_SIZE,
    device=OCR_DEVICE,
)
try:
    OCR_ENGINE._merged_paddlex_config["batch_size"] = OCR_BATCH_SIZE
    OCR_ENGINE.paddlex_pipeline.batch_sampler.batch_size = OCR_BATCH_SIZE
except Exception:
    pass


# =========================
# OCR per frame
# =========================
def _parse_ocr_result(res: Any) -> List[Tuple[List[List[float]], str, float]]:
    out: List[Tuple[List[List[float]], str, float]] = []
    if isinstance(res, dict) and "rec_texts" in res and "rec_scores" in res:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        polys = res.get("dt_polys", []) or []
        for text, score, poly in zip(texts, scores, polys):
            poly = poly.tolist() if hasattr(poly, "tolist") else poly
            out.append((poly, str(text), float(score)))
    elif isinstance(res, list):
        for line in res:
            poly = line[0]
            text = line[1][0]
            score = line[1][1]
            poly = poly.tolist() if hasattr(poly, "tolist") else poly
            out.append((poly, str(text), float(score)))
    return out


def ocr_frames(frame_imgs: List[np.ndarray]) -> List[List[Tuple[List[List[float]], str, float]]]:
    if not frame_imgs:
        return []
    result = OCR_ENGINE.predict(input=frame_imgs)
    return [_parse_ocr_result(res) for res in result]


def ocr_frame(frame_img: np.ndarray) -> List[Tuple[List[List[float]], str, float]]:
    outs = ocr_frames([frame_img])
    return outs[0] if outs else []
