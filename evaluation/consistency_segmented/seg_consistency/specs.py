"""
Load t2av-bench spec records (one JSONL file per category) and extract the
"actor" subject text prompt to be fed to GroundingDINO for text-grounded
subject seeding.

Why
---
The default segmenter chooses the largest near-center mask in the first
frame, which is often wrong (e.g. picks the goalkeeper instead of the
soccer player, or the foreground prop instead of the speaker). The spec
JSON already tells us *which subject is the actor* in plain English via
`subjects[*].descriptor` + `subjects[*].role`, so we can point a zero-shot
text-grounding detector (GroundingDINO) directly at the right entity.

Spec schema (relevant subset)
-----------------------------
  {
    "id": "action_0001",
    "category": "action",
    "subjects": [
       {"id": "S1", "type": "human", "descriptor": "...", "role": "actor", ...},
       {"id": "S2", "type": "object", ..., "role": "tool_or_target"},
       ...
    ],
    ...
  }

Actor selection rule
--------------------
* keep subjects whose role contains "actor" (case-insensitive); fallback to
  type=="human"/"animal"; fallback to whole list.
* among the keepers, take the one with the smallest numeric id (S1 first).
* extract a SHORT phrase (<= 6 words) for GroundingDINO from `descriptor`,
  preferring the head noun phrase ("a young adult male soccer player ..."
  -> "soccer player"). Keep the full descriptor as a fallback prompt.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
_REPO_ROOT = Path(__file__).resolve().parents[3]


# words that typically appear before the head noun in `descriptor` and add
# noise to a text-grounded detector. We strip them when shortening.
_FILLER_PREFIX = {
    "a", "an", "the",
    "young", "old", "middle", "aged", "elderly", "adult", "teen", "teenage",
    "child", "kid", "baby", "infant", "senior", "tall", "short", "small",
    "large", "tiny", "big", "huge", "thin", "slim", "fat", "chubby",
    "male", "female",
    "asian", "european", "african", "american", "latino", "white", "black",
    # common color/material words at the very start (rare)
    "red", "blue", "green", "yellow", "white", "black", "brown", "gray",
    "grey", "metal", "wooden", "plastic",
}


_DEFAULT_SPECS_ROOT = (
    str(_REPO_ROOT / "t2av-bench/specs")
)


def _short_phrase_from_descriptor(descriptor: str, max_words: int = 6) -> str:
    """Heuristically extract a short noun-phrase prompt from a full
    `descriptor` sentence.

    Strategy: tokenize on word boundaries, drop a leading run of articles /
    age / gender / ethnicity / size / color filler words, then take up to
    ``max_words`` until we hit a clause break (',', 'wearing', 'with',
    'holding', 'in', 'on', 'who', 'that', 'while', '-').
    """
    if not descriptor:
        return ""
    # cut off after the first clause break
    text = descriptor.strip()
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # split into tokens preserving simple punctuation as breakers
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]*|,", text)
    # drop leading filler
    i = 0
    while i < len(tokens) and tokens[i].lower() in _FILLER_PREFIX:
        i += 1
    BREAKERS = {",", "wearing", "with", "holding", "in", "on", "who",
                "that", "while", "and"}
    out: List[str] = []
    for tok in tokens[i:]:
        low = tok.lower()
        if low in BREAKERS:
            break
        out.append(tok)
        if len(out) >= max_words:
            break
    short = " ".join(out).strip().strip("-")
    return short


def _actor_label_for_viz(actor_id: str, actor_type: str, phrase_short: str,
                         conf: Optional[float] = None) -> str:
    base = f'actor {actor_id} ({actor_type}): "{phrase_short}"'
    if conf is not None:
        base += f"  conf={conf:.2f}"
    return base


def _id_num(sid: str) -> int:
    m = re.match(r"^S(\d+)$", str(sid))
    return int(m.group(1)) if m else 999


def _build_actor_info(a: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single subject dict from the spec into the actor_info dict
    that GroundingDINO + SAM2 expect downstream.
    """
    desc = (a.get("descriptor") or "").strip()
    species = (a.get("species") or "").strip()
    a_type = str(a.get("type", "")).lower() or "object"

    # build short phrase
    if a_type == "animal" and species:
        phrase_short = species.replace("_", " ")
    else:
        phrase_short = _short_phrase_from_descriptor(desc)
    if not phrase_short:
        # last-ditch: first 4 words of descriptor or just type
        phrase_short = " ".join(desc.split()[:4]) or a_type or "person"

    actor_id = str(a.get("id", "S?"))
    return {
        "actor_id": actor_id,
        "actor_type": a_type,
        "phrase_short": phrase_short,
        "phrase_long": desc or phrase_short,
        "label_for_viz": _actor_label_for_viz(actor_id, a_type, phrase_short),
        "raw": a,
    }


def extract_actors(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pick *all* candidate actor subjects from a single spec dict.

    Selection rule (one tier at a time, take first non-empty tier):
      1) subjects whose ``role`` contains "actor" (case-insensitive)
      2) subjects whose ``type`` is human / animal / anthropomorphic
      3) every subject in the spec

    All keepers are returned, sorted by their numeric S-id (S1, S2, ...).
    Each item is the same dict that ``extract_actor`` (singular) returns.
    Returns an empty list if the spec has no subjects.
    """
    subjects = spec.get("subjects") or []
    if not subjects:
        return []

    # 1) prefer role == "actor"
    actors = [s for s in subjects
              if "actor" in str(s.get("role", "")).lower()]
    # 2) else prefer human / animal / anthropomorphic
    if not actors:
        actors = [s for s in subjects
                  if str(s.get("type", "")).lower() in ("human", "animal",
                                                         "anthropomorphic")]
    # 3) else everyone
    if not actors:
        actors = list(subjects)

    actors_sorted = sorted(actors, key=lambda s: _id_num(s.get("id", "S99")))
    return [_build_actor_info(a) for a in actors_sorted]


def extract_actor(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the *first* actor subject from a single spec dict (back-compat).

    Returns ``None`` if no usable subject is found, in which case the caller
    should fall back to the default auto-mask selector. New code that wants
    to evaluate every candidate actor should call ``extract_actors`` instead.
    """
    actors = extract_actors(spec)
    return actors[0] if actors else None


class SpecsLoader:
    """Lazy index over <specs_root>/<category>.jsonl.

    Builds ``id -> spec`` mapping on demand. Handles `_patch/<cat>.jsonl`
    overrides if present (later wins, matching how the dataset patches work).
    """

    def __init__(self, specs_root: Optional[str] = None,
                 categories: Optional[List[str]] = None) -> None:
        self.specs_root = Path(specs_root or _DEFAULT_SPECS_ROOT)
        self._index: Dict[str, Dict[str, Any]] = {}
        self._loaded_categories: set = set()
        self._only_cats = set(categories) if categories else None

    @property
    def root(self) -> Path:
        return self.specs_root

    def _load_jsonl(self, path: Path) -> None:
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sid = rec.get("id")
                if sid:
                    self._index[sid] = rec

    def _load_category(self, cat: str) -> None:
        if cat in self._loaded_categories:
            return
        self._load_jsonl(self.specs_root / f"{cat}.jsonl")
        # apply patch overrides if present
        self._load_jsonl(self.specs_root / "_patch" / f"{cat}.jsonl")
        self._loaded_categories.add(cat)

    def _load_all(self) -> None:
        if not self.specs_root.is_dir():
            return
        for p in sorted(self.specs_root.glob("*.jsonl")):
            cat = p.stem
            if self._only_cats and cat not in self._only_cats:
                continue
            self._load_category(cat)

    def get(self, spec_id: str,
            category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # try the targeted category first to avoid loading everything
        if category and category not in self._loaded_categories:
            self._load_category(category)
        if spec_id in self._index:
            return self._index[spec_id]
        # fallback: derive category from id pattern "<cat>_<idx>"
        m = re.match(r"^([a-z_]+)_\d+$", spec_id)
        if m:
            self._load_category(m.group(1))
            if spec_id in self._index:
                return self._index[spec_id]
        # last resort: load everything once
        self._load_all()
        return self._index.get(spec_id)

    def get_actor(self, spec_id: str,
                  category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        spec = self.get(spec_id, category=category)
        if spec is None:
            return None
        return extract_actor(spec)

    def get_actors(self, spec_id: str,
                   category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return every candidate actor for a spec (see ``extract_actors``)."""
        spec = self.get(spec_id, category=category)
        if spec is None:
            return []
        return extract_actors(spec)

__all__ = [
    "SpecsLoader",
    "extract_actor",
    "extract_actors",
]
