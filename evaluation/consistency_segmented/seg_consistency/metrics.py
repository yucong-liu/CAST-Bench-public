"""
Segmentation-enhanced Subject / Background Consistency.

For every frame we first obtain a temporally-coherent SUBJECT mask from SAM 2,
then:
  - SUBJECT consistency : encode ONLY the subject region (background grayed out)
                          with DINOv2, measure frame-to-frame feature stability.
  - BACKGROUND consistency: encode ONLY the background region (subject grayed
                          out) with CLIP, measure stability.

This removes the cross-contamination of the original VBench metric, where a
large / fast-moving subject drags down the (full-frame) background score.

Aggregation matches VBench: for each frame i (i>=1),
    cur = ( cos(feat[i-1],feat[i]) + cos(feat[0],feat[i]) ) / 2
score = mean over all frames and all videos. Range 0~1, higher = more consistent.

Multi-shot handling
-------------------
``compute_segmented_consistency_from_manifest`` takes shot-level clips coming
from the audio_consistency project's ``shotcut_manifest.jsonl`` and runs the
SAM2 + DINOv2 + CLIP pipeline INDEPENDENTLY per shot, then aggregates with
frame-weighted averaging. This avoids the well-known issue where the original
VBench metric punishes any video that contains a hard cut.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .common import (load_frames_rgb, DinoV2Encoder, DinoV1Encoder,
                     ClipEncoder, cosine)
from .segmenter import SAM2Segmenter, apply_mask
from .specs import SpecsLoader


def _consistency_from_feats(feats):
    if len(feats) < 2:
        return 0.0, 0
    video_sim, cnt = 0.0, 0
    first = feats[0]
    for i in range(1, len(feats)):
        sim_pre = cosine(feats[i - 1], feats[i])
        sim_fir = cosine(first, feats[i])
        video_sim += (sim_pre + sim_fir) / 2
        cnt += 1
    return video_sim, cnt


def _subsample_frames(frames, max_frames):
    """Return ``frames`` if ``max_frames`` is None/<=0 or already short enough.
    Otherwise pick ``max_frames`` indices uniformly via ``np.linspace`` with
    rounding+dedup, preserving order. Aligned with VBench's n_samples=16/32 style
    of uniform temporal subsampling."""
    if not max_frames or max_frames <= 0:
        return frames
    T = len(frames)
    if T <= max_frames:
        return frames
    idx = np.linspace(0, T - 1, num=max_frames).round().astype(int)
    # de-duplicate while keeping order
    seen = set()
    uniq = []
    for i in idx:
        ii = int(i)
        if ii not in seen:
            seen.add(ii)
            uniq.append(ii)
    return [frames[i] for i in uniq]


def _score_one_clip(clip_path, seg, subj_enc, bg_enc, max_frames=None,
                    actor_info=None):
    """Score a single mp4 clip; return dict with sim/cnt for both metrics or
    error info if it could not be processed.

    If ``max_frames`` (int>0) is given and the clip has more frames than that,
    we uniformly subsample to ``max_frames`` BEFORE running SAM2/DINOv2/CLIP.
    This is the standard VBench-style temporal subsampling and dramatically
    reduces SAM2 propagation cost on long single-shot clips.

    ``actor_info`` (optional) is the dict returned by
    ``seg_consistency.specs.extract_actor`` and tells the segmenter which
    subject to track via GroundingDINO text prompt.
    """
    try:
        frames = load_frames_rgb(clip_path)
    except Exception as e:
        return {"ok": False, "reason": f"load_failed: {e}"}
    n_total = len(frames)
    frames = _subsample_frames(frames, max_frames)
    n_used = len(frames)
    if n_used < 2:
        return {"ok": False, "reason": f"too_short: {n_total} frames"}

    masks, seed_meta = seg.subject_masks_for_video(
        frames, actor_info=actor_info, return_meta=True)
    subj_feats, bg_feats = [], []
    for f, m in zip(frames, masks):
        subj_feats.append(subj_enc.encode(apply_mask(f, m, keep='subject')))
        bg_feats.append(bg_enc.encode(apply_mask(f, m, keep='background')))

    s_sim, s_cnt = _consistency_from_feats(subj_feats)
    b_sim, b_cnt = _consistency_from_feats(bg_feats)
    return {
        "ok": True,
        "n_frames": n_total,
        "n_frames_used": n_used,
        "subject_sim": float(s_sim), "subject_cnt": int(s_cnt),
        "background_sim": float(b_sim), "background_cnt": int(b_cnt),
        "seed_meta": seed_meta,
    }


def _score_one_clip_union(clip_path, seg, subj_enc, bg_enc, max_frames=None,
                          actor_iter=None):
    """Score a single mp4 clip using a UNION mask aggregated over all
    candidate actors of the spec.

    Algorithm
    ---------
    1. Load + uniformly sub-sample frames once.
    2. For each candidate actor in ``actor_iter`` (a list of actor_info dicts
       or ``[None]`` for legacy auto-mask), run SAM2 to get a per-frame mask
       sequence (with seed_meta for diagnostics).
    3. Drop actors whose seed_meta.source == 'text_failed_fallback'
       (GroundingDINO miss) so their auto-fallback mask does not pollute
       the union with the wrong subject.
    4. Per-frame OR all kept actor masks -> union_masks.
    5. Encode subject (apply_mask keep='subject') with DINOv2 and background
       with CLIP exactly once on the union path; compute consistency.

    If every actor's GroundingDINO missed AND we had >0 actor candidates,
    fall back to a single ``actor_info=None`` auto-mask run so we still
    return a usable score (mirrors legacy behaviour). If actor_iter is
    [None] (no actor info available, e.g. use_text_prompt=False), we run
    auto-mask once directly.

    Returns a dict shaped like ``_score_one_clip`` plus extra union
    diagnostics:
        - per_actor: list of per-actor records {actor_id, ok, used_in_union,
          seed_meta, ...}
        - n_actors_in_union: int
        - mask_aggregation: 'union' | 'auto_fallback'
    """
    try:
        frames = load_frames_rgb(clip_path)
    except Exception as e:
        return {"ok": False, "reason": f"load_failed: {e}"}
    n_total = len(frames)
    frames = _subsample_frames(frames, max_frames)
    n_used = len(frames)
    if n_used < 2:
        return {"ok": False, "reason": f"too_short: {n_total} frames"}

    if not actor_iter:
        actor_iter = [None]

    H, W = frames[0].shape[:2]
    union_masks: Optional[List[np.ndarray]] = None
    per_actor_records: List[Dict[str, Any]] = []
    first_seed_meta: Optional[Dict[str, Any]] = None
    n_in_union = 0

    for ai in actor_iter:
        try:
            masks_i, seed_meta_i = seg.subject_masks_for_video(
                frames, actor_info=ai, return_meta=True)
        except Exception as e:  # noqa: BLE001
            per_actor_records.append({
                'actor_id': (ai or {}).get('actor_id'),
                'actor_type': (ai or {}).get('actor_type'),
                'phrase_short': (ai or {}).get('phrase_short'),
                'ok': False, 'used_in_union': False,
                'reason': f'sam2_failed: {e}',
            })
            continue

        if first_seed_meta is None:
            first_seed_meta = seed_meta_i

        # GroundingDINO miss -> the resulting mask is the auto-fallback
        # (largest near-center blob) which is *not* the actor we asked for,
        # so we must NOT merge it into the union.
        grounding_failed = (ai is not None and
                            (seed_meta_i or {}).get('source') ==
                            'text_failed_fallback')

        rec = {
            'actor_id': (ai or {}).get('actor_id'),
            'actor_type': (ai or {}).get('actor_type'),
            'phrase_short': (ai or {}).get('phrase_short'),
            'ok': True,
            'used_in_union': not grounding_failed,
            'seed_meta': seed_meta_i,
            'mask_area_frac_first_frame': float(
                masks_i[0].sum()) / float(H * W) if masks_i else 0.0,
        }
        per_actor_records.append(rec)

        if grounding_failed:
            continue

        # accumulate into union (per-frame logical OR)
        if union_masks is None:
            union_masks = [m.astype(bool).copy() for m in masks_i]
        else:
            T = min(len(union_masks), len(masks_i))
            for t in range(T):
                union_masks[t] = union_masks[t] | masks_i[t].astype(bool)
        n_in_union += 1

    mask_aggregation = 'union'
    if union_masks is None:
        # Either every actor's GroundingDINO missed, or actor_iter was empty
        # of usable entries. Re-run once with actor_info=None so we still
        # produce a number for the shot (legacy auto-mask behaviour).
        try:
            masks_fb, seed_meta_fb = seg.subject_masks_for_video(
                frames, actor_info=None, return_meta=True)
        except Exception as e:  # noqa: BLE001
            return {"ok": False,
                    "reason": f"all_actors_failed_and_fallback_failed: {e}",
                    "per_actor": per_actor_records}
        union_masks = [m.astype(bool) for m in masks_fb]
        if first_seed_meta is None:
            first_seed_meta = seed_meta_fb
        per_actor_records.append({
            'actor_id': None, 'actor_type': None, 'phrase_short': None,
            'ok': True, 'used_in_union': True,
            'seed_meta': seed_meta_fb,
            'note': 'auto_mask_fallback (all_actors_grounding_failed)',
        })
        n_in_union = 1
        mask_aggregation = 'auto_fallback'

    # Encode once on the union mask path.
    subj_feats, bg_feats = [], []
    for f, m in zip(frames, union_masks):
        subj_feats.append(subj_enc.encode(apply_mask(f, m, keep='subject')))
        bg_feats.append(bg_enc.encode(apply_mask(f, m, keep='background')))

    s_sim, s_cnt = _consistency_from_feats(subj_feats)
    b_sim, b_cnt = _consistency_from_feats(bg_feats)
    return {
        "ok": True,
        "n_frames": n_total,
        "n_frames_used": n_used,
        "subject_sim": float(s_sim), "subject_cnt": int(s_cnt),
        "background_sim": float(b_sim), "background_cnt": int(b_cnt),
        "seed_meta": first_seed_meta,
        "per_actor": per_actor_records,
        "n_actors_in_union": int(n_in_union),
        "mask_aggregation": mask_aggregation,
    }


def _build_encoders(device, sam2_ckpt, sam2_cfg, subject_encoder, clip_model,
                    cache=None, grounding_ckpt=None,
                    use_text_prompt=False):
    """Build (or reuse) SAM2/DINOv2/CLIP/GroundingDINO. ``cache`` is an
    optional dict the caller can pass to keep the same instances across
    multiple models.

    When ``use_text_prompt`` is true and ``grounding_ckpt`` is given, we also
    construct a ``TextGrounder`` and attach it to the SAM2Segmenter so the
    first-frame subject seed is chosen via text-grounded box detection.
    """
    if cache is None:
        cache = {}
    if "text_grounder" not in cache:
        if use_text_prompt and grounding_ckpt:
            from .text_grounder import TextGrounder
            cache["text_grounder"] = TextGrounder(grounding_ckpt, device=device)
        else:
            cache["text_grounder"] = None
    if "seg" not in cache:
        cache["seg"] = SAM2Segmenter(sam2_ckpt, sam2_cfg, device,
                                     text_grounder=cache["text_grounder"])
    if "subj_enc" not in cache:
        cache["subj_enc"] = (DinoV2Encoder(device) if subject_encoder == 'dinov2'
                             else DinoV1Encoder(device))
    if "bg_enc" not in cache:
        cache["bg_enc"] = ClipEncoder(device, clip_model)
    return cache["seg"], cache["subj_enc"], cache["bg_enc"]


def compute_segmented_consistency(video_list, device,
                                  sam2_ckpt, sam2_cfg,
                                  subject_encoder='dinov2',
                                  clip_model='ViT-B/32',
                                  encoder_cache=None,
                                  max_frames=None,
                                  grounding_ckpt=None,
                                  use_text_prompt=False):
    """Legacy entry: each item in ``video_list`` is treated as one whole video.
    Kept for backward compatibility / quick single-folder debugging."""
    seg, subj_enc, bg_enc = _build_encoders(
        device, sam2_ckpt, sam2_cfg, subject_encoder, clip_model, encoder_cache,
        grounding_ckpt=grounding_ckpt, use_text_prompt=use_text_prompt)

    subj_sim_tot = subj_cnt_tot = 0.0
    bg_sim_tot = bg_cnt_tot = 0.0
    per_video = []

    for video_path in tqdm(video_list):
        r = _score_one_clip(video_path, seg, subj_enc, bg_enc, max_frames=max_frames)
        if not r.get("ok"):
            print(f"[WARN] skip {video_path}: {r.get('reason')}")
            continue
        s_sim, s_cnt = r["subject_sim"], r["subject_cnt"]
        b_sim, b_cnt = r["background_sim"], r["background_cnt"]
        subj_sim_tot += s_sim; subj_cnt_tot += s_cnt
        bg_sim_tot += b_sim;  bg_cnt_tot += b_cnt

        s_per = s_sim / s_cnt if s_cnt else float('nan')
        b_per = b_sim / b_cnt if b_cnt else float('nan')
        per_video.append({'video_path': video_path,
                          'subject_consistency': float(s_per),
                          'background_consistency': float(b_per)})
        name = video_path.rstrip('/').split('/')[-1]
        print(f"  {name} -> subject {s_per:.4f} | background {b_per:.4f}")

    subj_avg = subj_sim_tot / subj_cnt_tot if subj_cnt_tot else float('nan')
    bg_avg = bg_sim_tot / bg_cnt_tot if bg_cnt_tot else float('nan')
    return {
        'subject_consistency_avg': float(subj_avg),
        'background_consistency_avg': float(bg_avg),
        'per_video': per_video,
    }


def compute_segmented_consistency_from_manifest(
        manifest_entries: List[Dict[str, Any]],
        device: str,
        sam2_ckpt: str,
        sam2_cfg: str,
        subject_encoder: str = 'dinov2',
        clip_model: str = 'ViT-B/32',
        encoder_cache: Optional[Dict[str, Any]] = None,
        progress_desc: str = '',
        max_frames: Optional[int] = None,
        specs_loader: Optional[SpecsLoader] = None,
        grounding_ckpt: Optional[str] = None,
        use_text_prompt: bool = False,
) -> Dict[str, Any]:
    """Score a list of manifest entries (one entry == one spec, possibly with
    multiple shots) using PER-SHOT segmentation+encoding, then aggregate.

    Aggregation
    -----------
    * Each shot independently produces (sim, cnt) where cnt = n_frames - 1.
    * Spec-level score = sum(shot_sim) / sum(shot_cnt)            (frame-weighted)
    * Model-level score = sum(all_shot_sim) / sum(all_shot_cnt)   (frame-weighted)
    * Per-category score is also accumulated for diagnostic reporting.

    Returns a JSON-friendly dict.
    """
    seg, subj_enc, bg_enc = _build_encoders(
        device, sam2_ckpt, sam2_cfg, subject_encoder, clip_model, encoder_cache,
        grounding_ckpt=grounding_ckpt, use_text_prompt=use_text_prompt)

    # global accumulators
    g_s_sim = g_s_cnt = 0.0
    g_b_sim = g_b_cnt = 0.0
    n_shots_used = 0
    n_specs_used = 0

    # per-category accumulators
    cat_acc: Dict[str, Dict[str, float]] = {}
    # per-kind accumulators ('single_shot' / 'multi_shot')
    kind_acc: Dict[str, Dict[str, float]] = {}
    # per (category, kind) accumulators
    catkind_acc: Dict[tuple, Dict[str, float]] = {}

    per_spec: List[Dict[str, Any]] = []
    runtime_skipped: List[Dict[str, Any]] = []  # shot-level runtime drops

    pbar = tqdm(manifest_entries, desc=progress_desc or 'specs')
    for spec in pbar:
        spec_id = spec['spec_id']
        category = spec.get('category', '')
        kind = spec.get('kind', '')
        shots = spec['shots']

        # Resolve EVERY candidate actor for this spec once, shared across
        # its shots. We score each actor independently per shot and then
        # frame-weighted-average them into a single per-shot score.
        actor_infos: List[Dict[str, Any]] = []
        if specs_loader is not None and use_text_prompt:
            try:
                actor_infos = specs_loader.get_actors(
                    spec_id, category=category) or []
            except Exception as e:  # noqa: BLE001
                print(f"[specs] failed to load actors for {spec_id}: {e}")
                actor_infos = []
        # Fallback: keep the legacy 'no actor info -> auto-mask' code path so
        # that text_prompt=False or specs that have zero subjects still run.
        actor_iter: List[Optional[Dict[str, Any]]] = (
            list(actor_infos) if actor_infos else [None])

        spec_s_sim = spec_s_cnt = 0.0
        spec_b_sim = spec_b_cnt = 0.0
        shot_records: List[Dict[str, Any]] = []

        for s in shots:
            clip_path = s['clip_path']
            shot_idx_val = s['shot_idx']

            # Build a UNION mask over all candidate actors and score the
            # shot ONCE (single DINOv2 / CLIP pass). This gives every spec
            # a single (subject_sim, bg_sim) pair derived from the same
            # union region, regardless of how many subjects the prompt has.
            r = _score_one_clip_union(
                clip_path, seg, subj_enc, bg_enc,
                max_frames=max_frames, actor_iter=actor_iter,
            )
            if not r.get('ok'):
                runtime_skipped.append({
                    'spec_id': spec_id, 'shot_idx': shot_idx_val,
                    'clip_path': clip_path, 'reason': r.get('reason'),
                })
                shot_records.append({
                    'shot_idx': shot_idx_val, 'clip_path': clip_path,
                    'ok': False, 'reason': r.get('reason'),
                    'per_actor': r.get('per_actor', []),
                })
                continue

            shot_s_sim = r['subject_sim']
            shot_s_cnt = r['subject_cnt']
            shot_b_sim = r['background_sim']
            shot_b_cnt = r['background_cnt']

            spec_s_sim += shot_s_sim
            spec_s_cnt += shot_s_cnt
            spec_b_sim += shot_b_sim
            spec_b_cnt += shot_b_cnt
            n_shots_used += 1

            shot_subj = (shot_s_sim / shot_s_cnt) if shot_s_cnt else float('nan')
            shot_bg = (shot_b_sim / shot_b_cnt) if shot_b_cnt else float('nan')
            shot_records.append({
                'shot_idx': shot_idx_val, 'clip_path': clip_path,
                'ok': True, 'n_frames': r['n_frames'],
                'n_frames_used': r.get('n_frames_used', r['n_frames']),
                'n_actors_in_union': int(r.get('n_actors_in_union', 0)),
                'mask_aggregation': r.get('mask_aggregation', 'union'),
                'subject_consistency': float(shot_subj),
                'background_consistency': float(shot_bg),
                'seed_meta': r.get('seed_meta'),
                'per_actor': r.get('per_actor', []),
            })

        actor_summary = None
        if actor_infos:
            actor_summary = [{
                'actor_id': ai.get('actor_id'),
                'actor_type': ai.get('actor_type'),
                'phrase_short': ai.get('phrase_short'),
            } for ai in actor_infos]

        if spec_s_cnt == 0 and spec_b_cnt == 0:
            # no valid shot -> spec contributes nothing; still record for transparency
            per_spec.append({
                'spec_id': spec_id, 'category': category, 'kind': kind,
                'subject_consistency': float('nan'),
                'background_consistency': float('nan'),
                'n_shots_used': 0, 'shots': shot_records,
                'actors': actor_summary,
            })
            continue

        spec_subj = (spec_s_sim / spec_s_cnt) if spec_s_cnt else float('nan')
        spec_bg = (spec_b_sim / spec_b_cnt) if spec_b_cnt else float('nan')
        per_spec.append({
            'spec_id': spec_id, 'category': category, 'kind': kind,
            'subject_consistency': float(spec_subj),
            'background_consistency': float(spec_bg),
            'n_shots_used': sum(1 for x in shot_records if x.get('ok')),
            'shots': shot_records,
            'actors': actor_summary,
        })
        n_specs_used += 1

        g_s_sim += spec_s_sim; g_s_cnt += spec_s_cnt
        g_b_sim += spec_b_sim; g_b_cnt += spec_b_cnt

        ca = cat_acc.setdefault(category, {
            's_sim': 0.0, 's_cnt': 0.0, 'b_sim': 0.0, 'b_cnt': 0.0, 'n_specs': 0})
        ca['s_sim'] += spec_s_sim; ca['s_cnt'] += spec_s_cnt
        ca['b_sim'] += spec_b_sim; ca['b_cnt'] += spec_b_cnt
        ca['n_specs'] += 1

        # per-kind accumulator (single_shot / multi_shot, taken from manifest)
        kind_key = kind or 'unknown'
        ka = kind_acc.setdefault(kind_key, {
            's_sim': 0.0, 's_cnt': 0.0, 'b_sim': 0.0, 'b_cnt': 0.0, 'n_specs': 0})
        ka['s_sim'] += spec_s_sim; ka['s_cnt'] += spec_s_cnt
        ka['b_sim'] += spec_b_sim; ka['b_cnt'] += spec_b_cnt
        ka['n_specs'] += 1

        # per (category, kind) accumulator (action|single_shot etc.)
        cka = catkind_acc.setdefault((category, kind_key), {
            's_sim': 0.0, 's_cnt': 0.0, 'b_sim': 0.0, 'b_cnt': 0.0, 'n_specs': 0})
        cka['s_sim'] += spec_s_sim; cka['s_cnt'] += spec_s_cnt
        cka['b_sim'] += spec_b_sim; cka['b_cnt'] += spec_b_cnt
        cka['n_specs'] += 1

        pbar.set_postfix(spec=spec_id, subj=f"{spec_subj:.3f}", bg=f"{spec_bg:.3f}")

    subj_avg = (g_s_sim / g_s_cnt) if g_s_cnt else float('nan')
    bg_avg = (g_b_sim / g_b_cnt) if g_b_cnt else float('nan')

    per_category: Dict[str, Dict[str, float]] = {}
    for cat, ca in sorted(cat_acc.items()):
        per_category[cat] = {
            'n_specs': int(ca['n_specs']),
            'subject_consistency': (ca['s_sim'] / ca['s_cnt']) if ca['s_cnt'] else float('nan'),
            'background_consistency': (ca['b_sim'] / ca['b_cnt']) if ca['b_cnt'] else float('nan'),
        }

    # per-kind aggregation (single_shot / multi_shot): exposes whether models
    # behave differently on single-cut vs multi-cut prompts. Frame-weighted
    # like the global average, so it is directly comparable.
    per_kind: Dict[str, Dict[str, float]] = {}
    for kk, ka in sorted(kind_acc.items()):
        per_kind[kk] = {
            'n_specs': int(ka['n_specs']),
            'subject_consistency': (ka['s_sim'] / ka['s_cnt']) if ka['s_cnt'] else float('nan'),
            'background_consistency': (ka['b_sim'] / ka['b_cnt']) if ka['b_cnt'] else float('nan'),
        }

    # per (category, kind) aggregation, e.g. 'action|single_shot'.
    per_category_kind: Dict[str, Dict[str, float]] = {}
    for (cat, kk), cka in sorted(catkind_acc.items()):
        per_category_kind[f"{cat}|{kk}"] = {
            'category': cat,
            'kind': kk,
            'n_specs': int(cka['n_specs']),
            'subject_consistency': (cka['s_sim'] / cka['s_cnt']) if cka['s_cnt'] else float('nan'),
            'background_consistency': (cka['b_sim'] / cka['b_cnt']) if cka['b_cnt'] else float('nan'),
        }

    return {
        'subject_consistency_avg': float(subj_avg),
        'background_consistency_avg': float(bg_avg),
        'n_specs_used': int(n_specs_used),
        'n_shots_used': int(n_shots_used),
        'per_category': per_category,
        'per_kind': per_kind,
        'per_category_kind': per_category_kind,
        'per_spec': per_spec,
        'runtime_skipped': runtime_skipped,
    }
