"""
Subject / background separation using SAM 2 (Segment Anything 2),
the current strongest promptable segmentation / video tracking model.

Strategy
--------
We want, for every frame, a binary SUBJECT mask (foreground) so the
remaining pixels are BACKGROUND. To keep the subject identity stable across
the whole clip (and avoid per-frame flicker of the mask itself), we use SAM 2's
*video predictor*:

  1. On the FIRST frame, choose a SEED for the subject. Two routes:
     a. **text-grounded** (preferred when a text prompt is supplied): use
        GroundingDINO to detect the box of the named actor (e.g.
        "soccer player"), then feed that box to SAM 2 as a box prompt.
        This lets us segment the *intended* subject even when it is not
        the largest / most central object in the frame.
     b. **auto-mask fallback**: SAM 2 automatic mask generator + pick the
        most salient mask (largest area weighted toward image center).
  2. Use that seed to PROPAGATE / TRACK the same object through all frames
     via SAM 2's streaming-memory video predictor.

Fallback: if SAM 2 video predictor is unavailable, we fall back to per-frame
image segmentation with SAM2ImagePredictor seeded by the same prompt.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _largest_central_mask(masks, H, W):
    """Pick subject mask: score = area * centrality."""
    cy, cx = H / 2.0, W / 2.0
    best, best_score = None, -1
    for m in masks:
        seg = m['segmentation'] if isinstance(m, dict) else m
        area = seg.sum()
        coverage = area / float(H * W)
        # ignore tiny specks and near-full-frame masks (those are background)
        if coverage < 0.01 or coverage > 0.85:
            continue
        ys, xs = np.where(seg)
        my, mx = ys.mean(), xs.mean()
        dist = np.sqrt((my - cy) ** 2 + (mx - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
        centrality = 1.0 - dist
        score = coverage * (0.5 + 0.5 * centrality)
        if score > best_score:
            best_score, best = score, seg
    return best


class SAM2Segmenter:
    def __init__(self, ckpt, model_cfg, device, text_grounder=None):
        """``text_grounder`` is an optional ``TextGrounder`` instance. When
        present and a per-clip text prompt is supplied, we use it to seed
        SAM 2 with a box of the named actor; otherwise we fall back to the
        original auto-mask + center-crop heuristic.
        """
        self.device = device
        self.ckpt = ckpt
        self.model_cfg = model_cfg
        self.text_grounder = text_grounder
        self._build()

    def _build(self):
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        self._build_video = build_sam2_video_predictor
        img_model = build_sam2(self.model_cfg, self.ckpt, device=self.device)
        self.img_predictor = SAM2ImagePredictor(img_model)
        self.mask_gen = SAM2AutomaticMaskGenerator(img_model)

    # ------------------------------------------------------------------
    # Seed selection
    # ------------------------------------------------------------------
    def _seed_from_box(self, first_frame: np.ndarray,
                       box_xyxy: Tuple[int, int, int, int]
                       ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run SAM2 image predictor with the given box prompt and pick the
        'whole-object' mask among the 3 multimask candidates using a
        box-aware scoring rule (Plan B):

          * containment(mask, box)     >= 0.90  -- mask must lie inside box
          * coverage_in_box(mask, box) >= 0.25  -- kill part / subpart heads
          * tie-break: argmax(coverage_in_box)  -- pick the 'whole' mask

        If all 3 candidates are filtered out, fall back to a relaxed rule,
        then finally to SAM2's own argmax(score) (= old behaviour).

        Returns (seg, debug_dict). debug_dict is recorded in seed_meta for
        post-hoc inspection.
        """
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        H, W = first_frame.shape[:2]
        x1 = max(0, min(W - 1, x1)); x2 = max(x1 + 1, min(W, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
        box_area = max(1, (x2 - x1) * (y2 - y1))

        with torch.inference_mode():
            self.img_predictor.set_image(first_frame)
            masks, scores, _ = self.img_predictor.predict(
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                multimask_output=True,
            )

        cands = []
        for i, m in enumerate(masks):
            seg_i = m.astype(bool)
            area = int(seg_i.sum())
            if area == 0:
                cands.append({"idx": i, "score": float(scores[i]),
                              "area": 0, "containment": 0.0,
                              "coverage_in_box": 0.0})
                continue
            inter = int(seg_i[y1:y2, x1:x2].sum())
            cands.append({
                "idx": i,
                "score": float(scores[i]),
                "area": area,
                "containment": inter / float(area),
                "coverage_in_box": inter / float(box_area),
            })

        def _pick(c_list, cont_th, cov_th):
            ok = [c for c in c_list
                  if c["area"] > 0
                  and c["containment"] >= cont_th
                  and c["coverage_in_box"] >= cov_th]
            if not ok:
                return None
            return max(ok, key=lambda c: c["coverage_in_box"])

        chosen = _pick(cands, 0.90, 0.25)
        rule = "strict_whole"
        if chosen is None:
            chosen = _pick(cands, 0.70, 0.0)
            rule = "relaxed_contained"
        if chosen is None:
            chosen = max(cands, key=lambda c: c["score"])
            rule = "sam2_argmax_fallback"

        seg = masks[chosen["idx"]].astype(bool)
        debug = {
            "rule": rule,
            "chosen_idx": int(chosen["idx"]),
            "candidates": cands,
            "box_area_frac": box_area / float(H * W),
        }
        return seg, debug

    def _seed_subject_mask(self, first_frame, actor_info=None):
        """Return (seed_mask, seed_meta_dict).

        seed_meta_dict carries diagnostic info about how the seed was chosen
        (used by visualization). Always non-None.
        """
        H, W = first_frame.shape[:2]
        meta: Dict[str, Any] = {"source": "auto", "ok": False}

        # Whether we actually attempted Route A (text-grounded). This is
        # True iff a TextGrounder is wired in AND the manifest provided a
        # usable actor descriptor for this clip. Used below to mark the
        # fallback as a *failure* of Route A vs. simply "Route A disabled".
        text_attempted = bool(self.text_grounder is not None and actor_info)
        text_failed = False
        actor_phrase_for_log = ""

        # Route A: text-grounded box prompt
        if text_attempted:
            actor_phrase_for_log = (
                (actor_info or {}).get("phrase_short")
                or (actor_info or {}).get("phrase_long")
                or ""
            )
            try:
                hit = self.text_grounder.detect_with_fallback(
                    first_frame,
                    phrase_short=actor_info.get("phrase_short", ""),
                    phrase_long=actor_info.get("phrase_long", ""),
                )
            except Exception as e:  # noqa: BLE001
                hit = None
                meta["text_grounder_error"] = repr(e)
            if hit is not None:
                box, score, prompt_used = hit
                try:
                    seg, box_dbg = self._seed_from_box(first_frame, box)
                    if seg.any():
                        meta.update({
                            "source": "text_grounded",
                            "ok": True,
                            "box_xyxy": list(box),
                            "score": float(score),
                            "prompt_used": prompt_used,
                            "actor_id": actor_info.get("actor_id"),
                            "actor_type": actor_info.get("actor_type"),
                            "box_to_mask": box_dbg,
                        })
                        return seg, meta
                    # box ok but SAM2 produced empty mask -> count as failure
                    text_failed = True
                    meta["text_grounder_empty_mask"] = True
                    meta["box_to_mask"] = box_dbg
                except Exception as e:  # noqa: BLE001
                    text_failed = True
                    meta["sam2_box_error"] = repr(e)
            else:
                # detect returned nothing (or threw an exception above)
                text_failed = True

        # Route B: auto-mask fallback
        with torch.inference_mode():
            masks = self.mask_gen.generate(first_frame)
        seg = _largest_central_mask(masks, H, W)
        if seg is None:                  # nothing found -> center box fallback
            seg = np.zeros((H, W), bool)
            seg[H // 4:3 * H // 4, W // 4:3 * W // 4] = True
            base_source = "center_box_fallback"
        else:
            base_source = "auto_largest_central"

        if text_failed:
            # Make it visually obvious in the overlay / info.txt that the
            # text-grounded path was attempted but failed, so we don't
            # silently confuse "no text prompt configured" with "text prompt
            # provided but GroundingDINO missed the subject".
            meta["source"] = "text_failed_fallback"
            meta["fallback_strategy"] = base_source
            print(
                f"[seg] GroundingDINO failed to localize "
                f"\"{actor_phrase_for_log}\"; falling back to "
                f"{base_source}."
            )
        else:
            meta["source"] = base_source
        meta["ok"] = True
        return seg, meta

    def subject_masks_for_video(self, frames, work_dir=None,
                                actor_info: Optional[Dict[str, Any]] = None,
                                return_meta: bool = False):
        """Return list of boolean subject masks, one per frame (True = subject).

        If ``return_meta=True``, also return a dict with diagnostic info
        about how the first-frame seed was chosen (mask source, GroundingDINO
        box / score, prompt used, etc.).
        """
        H, W = frames[0].shape[:2]
        seed, meta = self._seed_subject_mask(frames[0], actor_info=actor_info)

        seed_box = meta.get("box_xyxy")  # provided iff Route A succeeded

        # Try video propagation for temporally coherent masks.
        try:
            masks = self._propagate(frames, seed, work_dir, seed_box=seed_box)
            meta["propagation"] = "video_predictor"
            meta["propagation_seed"] = getattr(
                self, "_last_propagate_seed_method", None)
        except Exception as e:
            print(f"[seg] video propagation unavailable ({e}); per-frame fallback.")
            masks = self._per_frame(frames, seed)
            meta["propagation"] = "per_frame_fallback"

        if return_meta:
            return masks, meta
        return masks

    def _propagate(self, frames, seed, work_dir, seed_box=None):
        """Track ``seed`` (a binary mask on frame 0) through ``frames`` with
        SAM 2 video predictor.

        Seeding strategy (in order of preference):
          1. ``add_new_mask(mask=seed)``  -- richest signal, keeps the
             whole-object shape we picked in `_seed_from_box`.
          2. ``add_new_points_or_box(box=seed_box)`` -- if mask API not
             available; reuse GroundingDINO's box.
          3. ``add_new_points_or_box(points=centroid)`` -- last-resort, the
             old behaviour. Note: this can collapse onto a salient sub-part
             (e.g. shorts) when the centroid lands on a strong texture;
             only used if (1) and (2) both fail.
        """
        import tempfile
        from PIL import Image
        predictor = self._build_video(self.model_cfg, self.ckpt, device=self.device)
        tmp = work_dir or tempfile.mkdtemp(prefix='sam2frames_')
        os.makedirs(tmp, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(tmp, f"{i:05d}.jpg"))
        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp)

            seeded = False
            seed_method = None
            # 1) preferred: feed the whole-object mask directly
            if hasattr(predictor, "add_new_mask"):
                try:
                    predictor.add_new_mask(
                        inference_state=state, frame_idx=0, obj_id=1,
                        mask=seed.astype(bool),
                    )
                    seeded = True
                    seed_method = "mask"
                except Exception as e:  # noqa: BLE001
                    print(f"[seg] add_new_mask failed ({e}); trying box.")

            # 2) fallback: feed GroundingDINO's box (re-segments inside it
            #    on every propagated frame, more robust than a single point)
            if not seeded and seed_box is not None:
                try:
                    x1, y1, x2, y2 = [int(v) for v in seed_box]
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=0, obj_id=1,
                        box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    )
                    seeded = True
                    seed_method = "box"
                except Exception as e:  # noqa: BLE001
                    print(f"[seg] add box failed ({e}); falling back to point.")

            # 3) last resort: centroid point (old behaviour)
            if not seeded:
                ys, xs = np.where(seed)
                if ys.size == 0:
                    raise RuntimeError("empty seed mask, cannot propagate")
                pt = np.array([[xs.mean(), ys.mean()]], dtype=np.float32)
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=pt, labels=np.array([1], np.int32))
                seed_method = "centroid_point"

            self._last_propagate_seed_method = seed_method
            out = {}
            for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                m = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                out[fidx] = m
        masks = [out.get(i, seed) for i in range(len(frames))]
        return masks

    def _per_frame(self, frames, seed):
        H, W = frames[0].shape[:2]
        ys, xs = np.where(seed)
        pt = np.array([[xs.mean(), ys.mean()]], dtype=np.float32)
        masks = []
        for f in frames:
            with torch.inference_mode():
                self.img_predictor.set_image(f)
                m, scores, _ = self.img_predictor.predict(
                    point_coords=pt, point_labels=np.array([1]), multimask_output=True)
            masks.append(m[int(np.argmax(scores))].astype(bool))
        return masks


def apply_mask(np_rgb, mask, keep='subject', fill=128):
    """Return a copy of the frame where the non-kept region is filled with a
    neutral gray, so the encoder mostly sees the region we care about.

    keep='subject' -> keep mask region, gray out background.
    keep='background' -> keep ~mask region, gray out subject.
    """
    out = np_rgb.copy()
    if keep == 'subject':
        out[~mask] = fill
    else:
        out[mask] = fill
    return out
