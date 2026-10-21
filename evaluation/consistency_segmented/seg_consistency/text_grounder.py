"""
Zero-shot text-grounded box detector based on GroundingDINO (HuggingFace
``IDEA-Research/grounding-dino-tiny``).

Usage
-----
    grounder = TextGrounder(ckpt_dir="./pretrained/grounding-dino-tiny",
                            device="cuda")
    boxes = grounder.detect(rgb_image_np, "soccer player")
    # -> List[(xyxy_int_tuple, score_float)] sorted by score desc

Notes
-----
* GroundingDINO expects a *lower-cased* prompt that ends with a period.
* The custom kernel for multi-scale deformable attention is missing in our
  transformers install; we silence the warning and rely on the pure-PyTorch
  fallback (~1.5x slower but accurate).
* The model is loaded ONCE per process (~700MB on disk, ~2.5-3GB GPU mem
  in fp32). We default to fp32 because the pure-PyTorch fallback for
  multi-scale deformable attention has a fp16/fp32 dtype-mismatch bug
  ("mat1 and mat2 must have the same dtype, but got Float and Half")
  on transformers 4.40.x; fp32 is rock-solid and only ~2x slower per
  call, which is negligible for our keyframe-level usage.
"""
from __future__ import annotations

import os
import warnings
from typing import List, Optional, Tuple

import numpy as np


# noisy "Could not load the custom kernel for multi-scale deformable
# attention" prints come from python's logging, hide them
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


class TextGrounder:
    def __init__(self, ckpt_dir: str, device: str = "cuda",
                 dtype: str = "fp32") -> None:
        from transformers import (AutoProcessor,
                                  AutoModelForZeroShotObjectDetection)
        import torch

        self.device = device
        self.ckpt_dir = ckpt_dir
        self._torch = torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.processor = AutoProcessor.from_pretrained(ckpt_dir)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(ckpt_dir)

        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            if dtype == "fp16":
                model = model.half()
                self._dtype = torch.float16
            else:
                self._dtype = torch.float32
        else:
            self._dtype = torch.float32
        self.model = model.eval()

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        t = (text or "").strip().lower()
        if not t:
            return ""
        # GroundingDINO requires the prompt to end with '.'
        if not t.endswith("."):
            t = t + "."
        return t

    def detect(self, image_rgb: np.ndarray, text: str,
               box_threshold: float = 0.30,
               text_threshold: float = 0.25,
               topk: int = 5) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Run GroundingDINO on a single RGB ``HxWx3`` uint8 image with the
        given text prompt.

        Returns a list of (box_xyxy, score) sorted by descending score, with
        boxes clipped/clamped to the image. If nothing matches the prompt
        the list is empty.
        """
        from PIL import Image
        torch = self._torch

        prompt = self._normalize_prompt(text)
        if not prompt:
            return []
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        H, W = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil, text=prompt,
                                return_tensors="pt").to(self.device)
        # cast image tensor to model dtype, keep ids/mask as long
        if "pixel_values" in inputs and self._dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # post-process. NOTE: the kwarg name for the box-score cutoff differs
        # across transformers versions:
        #   * <= 4.40.x  -> ``threshold``
        #   * >= 4.42.x  -> ``box_threshold`` (renamed; old name removed)
        # We try the new name first and silently fall back to the old one on
        # TypeError so the same code works regardless of installed version.
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(H, W)],
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(H, W)],
            )[0]

        boxes = results.get("boxes")
        scores = results.get("scores")
        if boxes is None or len(boxes) == 0:
            return []

        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        order = np.argsort(-scores)[:topk]

        out: List[Tuple[Tuple[int, int, int, int], float]] = []
        for k in order:
            x1, y1, x2, y2 = boxes[k].tolist()
            x1 = max(0, min(W - 1, int(round(x1))))
            y1 = max(0, min(H - 1, int(round(y1))))
            x2 = max(0, min(W - 1, int(round(x2))))
            y2 = max(0, min(H - 1, int(round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(((x1, y1, x2, y2), float(scores[k])))
        return out

    def detect_with_fallback(
            self, image_rgb: np.ndarray,
            phrase_short: str,
            phrase_long: Optional[str] = None,
            box_threshold: float = 0.30,
            text_threshold: float = 0.25,
    ) -> Optional[Tuple[Tuple[int, int, int, int], float, str]]:
        """Detect with the short prompt; if empty, retry with the long
        descriptor. Returns the top-1 (box, score, prompt_used) or None.
        """
        for prompt in [phrase_short, phrase_long]:
            if not prompt:
                continue
            cands = self.detect(image_rgb, prompt,
                                box_threshold=box_threshold,
                                text_threshold=text_threshold,
                                topk=1)
            if cands:
                box, score = cands[0]
                return box, score, prompt
        return None


__all__ = ["TextGrounder"]
