import os
import re
import json
import unicodedata
import yaml
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import jiwer
from qwen_asr import Qwen3ASRModel

# ============================================================
# Config
# ============================================================
YAML_PATH = os.environ.get("ASR_TEST_YAML", "resource/test.yaml")
DEVICE = "cuda:0"
BATCH_SIZE = 32
MAX_NEW_TOKENS = 256
LANGUAGE = None  # set e.g. "Chinese" to force language


# ============================================================
# Text extraction helpers
# ============================================================
_TAG_PATTERN = re.compile(r"<(?:speech|lyrics)>(.*?)</(?:speech|lyrics)>", re.DOTALL)


def extract_ref_text(caption: str) -> str:
    """Extract all <speech> and <lyrics> content from a caption string.
    Returns concatenated text (space-joined), or empty string if none found.
    """
    matches = _TAG_PATTERN.findall(caption)
    return " ".join(m.strip() for m in matches if m.strip())


# ============================================================
# WER normalization
# ============================================================
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _is_chinese(text: str) -> bool:
    """Return True if the text contains a significant proportion of CJK characters."""
    cjk_count = len(_CJK_RE.findall(text))
    return cjk_count > len(text) * 0.2


def _strip_unicode_punct(text: str) -> str:
    """剥离所有 Unicode 标点（类别以 P 开头：Pc/Pd/Pe/Pf/Pi/Po/Ps）。
    覆盖 ASCII 标点、中文标点、以及 –（en dash, U+2013）、—（em dash, U+2014）、
    « » ¿ ¡ 弯引号（“ ” ‘ ’）等多语种标点，避免漏字符。
    """
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)


def normalize(text: str, for_display: bool = False) -> str:
    """Normalize text for WER computation.
    - Strip ALL Unicode punctuation (符合 Whisper / ESPnet 等主流 ASR 评测惯例)
    - Lowercase
    - For Chinese: insert spaces between every character (char-level WER)
    - For English/其他语种: collapse whitespace (word-level WER)
    """
    text = _strip_unicode_punct(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    if not for_display and _is_chinese(text):
        # Insert space between every character for char-level WER
        text = " ".join(list(text.replace(" ", "")))
    return text

# ============================================================
# Load YAML config
# ============================================================
def load_config(yaml_path: str) -> dict:
    """Parse YAML and return {testset_name: {audio_dir, jsonl_path}}."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = {}
    # YAML is a list of single-key dicts
    for entry in raw:
        for name, paths in entry.items():
            audio_dir, jsonl_path = paths[0], paths[1]
            config[name] = {"audio_dir": audio_dir, "jsonl_path": jsonl_path}
    return config


# ============================================================
# Load and filter samples from JSONL
# ============================================================
def load_samples(sub_audio_dir: str, jsonl_path: str) -> list:
    """Load JSONL, extract ref text, filter empty cases.
    sub_audio_dir: the specific subdirectory to look for audio files (e.g. audio_dir/subdir/audios).
    Returns list of {audio_path, ref_text}.
    """
    samples = []
    skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            caption = item.get("gemini25_caption_detailed", "")
            ref_text = extract_ref_text(caption)
            if not ref_text:
                skipped += 1
                continue
            # video_path may be like "xxxx_Seg0001.wav" or a relative path
            audio_filename = os.path.basename(item["video_path"])
            audio_path = os.path.join(sub_audio_dir, audio_filename)
            if not os.path.exists(audio_path):
                print(f"[WARN] Audio file not found, skipping: {audio_path}")
                skipped += 1
                continue
            samples.append({"audio_path": audio_path, "ref_text": ref_text})
    print(f"  Loaded {len(samples)} valid samples, skipped {skipped} (no speech/lyrics text or missing audio).")
    return samples


# ============================================================
# ASR inference
# ============================================================
def run_asr(model, samples: list, batch_size: int, language) -> list:
    """Run ASR on all samples in batches, return list of hyp strings."""
    audio_paths = [s["audio_path"] for s in samples]
    hyps = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="  ASR inference"):
        batch = audio_paths[i: i + batch_size]
        results = model.transcribe(audio=batch, language=language)
        for r in results:
            hyps.append(r.text if r.text else "")
    return hyps


# ============================================================
# Compute WER
# ============================================================
def compute_wer(refs: list, hyps: list) -> dict:
    """Compute overall WER and per-sample WER.
    Chinese text is evaluated at character level (spaces inserted between chars).
    English text is evaluated at word level.
    """
    norm_refs = [normalize(r) for r in refs]
    norm_hyps = [normalize(h) for h in hyps]

    # Overall WER (corpus-level)
    overall_wer = jiwer.wer(norm_refs, norm_hyps)

    # Per-sample WER
    per_sample = []
    for ref, hyp in zip(norm_refs, norm_hyps):
        if not ref.strip():
            per_sample.append(float("nan"))
            continue
        try:
            w = jiwer.wer(ref, hyp)
        except Exception:
            w = float("nan")
        per_sample.append(w)

    return {"overall_wer": overall_wer, "per_sample_wer": per_sample}


# ============================================================
# Main
# ============================================================
def main():
    config = load_config(YAML_PATH)
    print(f"Loaded {len(config)} testset(s) from {YAML_PATH}")

    # Load ASR model once
    print(f"\nLoading ASR model on {DEVICE} ...")
    model = Qwen3ASRModel.from_pretrained(
        "../../models/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map=DEVICE,
        max_inference_batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print("Model loaded.\n")

    all_results = {}

    for testset_name, paths in config.items():
        audio_dir = paths["audio_dir"]
        jsonl_path = paths["jsonl_path"]

        # Collect all subdirectories under audio_dir
        try:
            subdirs = sorted([
                d for d in os.listdir(audio_dir)
                if os.path.isdir(os.path.join(audio_dir, d))
            ])
        except FileNotFoundError:
            print(f"[ERROR] Audio dir not found: {audio_dir}")
            continue

        if not subdirs:
            # No subdirectories, treat audio_dir itself as the audio source
            subdirs_to_process = [(testset_name, audio_dir)]
        else:
            # Each subdir is a separate model/version to evaluate
            subdirs_to_process = [
                (f"{testset_name}/{subdir}", os.path.join(audio_dir, subdir, "audios"))
                for subdir in subdirs
            ]

        for run_name, sub_audio_dir in subdirs_to_process:
            print(f"{'='*60}")
            print(f"Testset: {run_name}")
            print(f"  Audio dir : {sub_audio_dir}")
            print(f"  JSONL path: {jsonl_path}")

            if not os.path.isdir(sub_audio_dir):
                print(f"  [SKIP] Audio subdir not found: {sub_audio_dir}")
                continue

            samples = load_samples(sub_audio_dir, jsonl_path)
            if not samples:
                print("  [SKIP] No valid samples found.")
                continue

            refs = [s["ref_text"] for s in samples]
            hyps = run_asr(model, samples, BATCH_SIZE, LANGUAGE)

            wer_result = compute_wer(refs, hyps)
            overall_wer = wer_result["overall_wer"]
            per_sample_wer = wer_result["per_sample_wer"]

            print(f"\n  Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
            print(f"  Per-sample WER (top 5 worst):")

            # Show top-5 worst cases
            indexed = sorted(
                enumerate(per_sample_wer), key=lambda x: x[1] if not __import__('math').isnan(x[1]) else -1, reverse=True
            )
            for rank, (idx, w) in enumerate(indexed[:5]):
                print(f"    [{rank+1}] idx={idx:4d} | WER={w:.4f} | audio={os.path.basename(samples[idx]['audio_path'])}")
            print(f"         REF: {normalize(refs[idx], for_display=True)[:80]}")
            print(f"         HYP: {normalize(hyps[idx], for_display=True)[:80]}")

            all_results[run_name] = {
                "overall_wer": overall_wer,
                "num_samples": len(samples),
                "details": [
                    {
                        "audio": samples[i]["audio_path"],
                        "ref": refs[i],
                        "hyp": hyps[i],
                        "wer": per_sample_wer[i],
                    }
                    for i in range(len(samples))
                ],
            }

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for name, res in all_results.items():
        print(f"  {name}: WER={res['overall_wer']*100:.2f}%  (n={res['num_samples']})")

    # Save detailed results
    output_path = os.path.join(os.path.dirname(YAML_PATH), "wer_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
