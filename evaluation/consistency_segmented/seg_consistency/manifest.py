"""
Read shotcut manifests produced by the audio_consistency project so that we
can score subject/background consistency on a *per-shot* basis.

Each line in <model_dir>/shotcut_manifest.jsonl looks like:
  {
    "spec_id": "action_0001",
    "category": "action",
    "shot_count_kind": "multi_shot" | "single_shot",
    "expected_shots": 5, "detected_shots": 5,
    "passed": true | false,
    "fps": 24.0,
    "note": "ok" | "single_shot_passthrough" | "shot_count_mismatch" | ...,
    "shots": [{"shot_idx": 1, "clip_path": "...mp4"}, ...]
  }

For single_shot entries, the single "clip_path" points back to the original
full-length mp4 in the generation/ tree (passthrough). We treat that as one
shot of length = full video, so the evaluation set is consistent across
multi-shot and single-shot specs.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple


def load_manifest(manifest_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse a shotcut_manifest.jsonl.

    Returns (entries, skipped):
      entries: list of dicts that we will score, each with keys
        - spec_id, category, kind ('single_shot' | 'multi_shot'),
          fps, note, shots: [{'shot_idx', 'clip_path'}]
        Only entries with passed==True and a non-empty 'shots' list are kept.
      skipped: list of dicts {'spec_id', 'category', 'kind', 'reason'}
        for entries that were dropped (passed=false, no shots, missing files).
    """
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    entries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                skipped.append({
                    "spec_id": f"<line {ln_no}>",
                    "category": "",
                    "kind": "",
                    "reason": f"json_decode_error: {e}",
                })
                continue

            spec_id = rec.get("spec_id", f"<line {ln_no}>")
            category = rec.get("category", "")
            kind = rec.get("shot_count_kind", "")
            note = rec.get("note", "")
            passed = bool(rec.get("passed", False))
            shots_raw = rec.get("shots") or []

            if not passed:
                skipped.append({
                    "spec_id": spec_id, "category": category,
                    "kind": kind, "reason": note or "not_passed",
                })
                continue
            if not shots_raw:
                skipped.append({
                    "spec_id": spec_id, "category": category,
                    "kind": kind, "reason": "no_shots",
                })
                continue

            # filter shots whose clip file is actually present on disk
            shots: List[Dict[str, Any]] = []
            missing: List[str] = []
            for s in shots_raw:
                cp = s.get("clip_path", "")
                if cp and os.path.isfile(cp):
                    shots.append({
                        "shot_idx": int(s.get("shot_idx", len(shots) + 1)),
                        "clip_path": cp,
                    })
                else:
                    missing.append(cp)
            if not shots:
                skipped.append({
                    "spec_id": spec_id, "category": category,
                    "kind": kind,
                    "reason": "all_clips_missing: " + ";".join(missing),
                })
                continue

            entries.append({
                "spec_id": spec_id,
                "category": category,
                "kind": kind,
                "fps": rec.get("fps", 0.0),
                "note": note,
                "shots": shots,
                "missing_clips": missing,  # informational
            })

    return entries, skipped


def discover_models(audio_runs_dir: str) -> List[Tuple[str, str]]:
    """Scan <audio_runs_dir>/*/shotcut_manifest.jsonl and return
    [(model_name, manifest_path), ...] in sorted order.
    """
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(audio_runs_dir):
        return out
    for name in sorted(os.listdir(audio_runs_dir)):
        sub = os.path.join(audio_runs_dir, name)
        man = os.path.join(sub, "shotcut_manifest.jsonl")
        if os.path.isdir(sub) and os.path.isfile(man):
            out.append((name, man))
    return out
