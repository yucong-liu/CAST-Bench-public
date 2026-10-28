# CAST-Bench

**Tracing Every Sound to Its Source in Multi-Shot Audio-Video Generation**

CAST-Bench is a sound-source-first benchmark for joint text-to-audio-video (T2AV)
generation. It contains **900 schema-grounded prompts** in **9 sound-source
categories** (100 each), split into **439 single-shot** and **461 multi-shot**
narratives, and an evaluation suite of **18 metrics** grouped into *Visual
Performance*, *Content Consistency*, and *Audio & Semantic Alignment*.

Each prompt is derived from a structured specification that binds every audio
event to an emitter, a subject, and a temporal slot. The suite therefore reports
not only *whether* a sound is present, but *which source* produced it and whether
it stays bound to that source across shots.

> Paper: `arXiv:XXXX.XXXXX` *(update after announcement)*
> Contact: see the author list in the paper.

## Repository layout

```
cast-bench/
├── t2av-bench/
│   ├── prompts/      # 9 JSON files with the 900 prompts (100 per category)
│   │                 # plus variant sets: zh/, samp/, hunyuan/, hard/
│   ├── specs/        # machine-readable specs, 9 JSONL files (one per prompt)
│   ├── questions/    # judge questions used by the LLM-based evaluators
│   ├── categories/   # category taxonomy
│   ├── vocab/        # label vocabularies
│   ├── config/       # shared configuration
│   └── docs/         # additional documentation
├── evaluation/       # 17 evaluators + scoring.py (18-metric aggregation)
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

## Quick start

**Requirements**: Linux, Python 3.12, CUDA 12.8 (driver >= 570), 8x GPUs with
80 GB memory, `ffmpeg`/`ffprobe`, and roughly 1 TB of disk for model weights and
generated videos.

```bash
git clone <REPO_URL> cast-bench
cd cast-bench/evaluation

# 1) one-time environment setup (base env + per-expert venvs + weights)
bash setup_all.sh

# 2) generate videos with your model, then map its output directory
#    <model>_test/output/<category>/<spec_id>.mp4
vim models.tsv          # name<TAB>path/to/output

# 3) run every evaluator over every model
bash run_all_experts.sh

# 4) aggregate the 18 metrics
python3 aggregate_18_metrics.py
```

Results land in `evaluation/runs_summary/`:

```
all_models_18metrics.csv        # one row per model per shot setting
all_models_18metrics_single.csv
all_models_18metrics_multi.csv
all_models_18metrics.md
all_models_18metrics.json
```

### Adding a model

`evaluation/models.tsv` is a two-column TSV (`#` comments allowed). Append one
line per model and every evaluator picks it up:

```
my-model<TAB>my-model_test/output
```

The root must contain one sub-directory per category, with videos named
`<spec_id>.mp4` (e.g. `speech/speech_0001.mp4`).

## Data format

**Prompt** (`t2av-bench/prompts/animal.json`, list of records):

```json
{
  "id": "animal_0001",
  "category": "animal",
  "shot_count": "multi_shot",
  "prompt": "[OVERALL] ... [SHOT 1/2] ... [SHOT 2/2] ...",
  "compiled_by": "Qwen3-VL-235B-A22B-Instruct"
}
```

**Specification** (`t2av-bench/specs/animal.jsonl`, one JSON object per line):

```
id, category, version, shot_count, visual_style, audio_situation,
sound_complexity, video_type, camera_movement, scene_setting,
subjects[], audio_events[], camera, shots[], text_on_screen,
metadata, visual_substyle
```

`subjects[]` and `audio_events[]` carry the emitter / subject / temporal-slot
bindings that the metrics are computed against.

## The 18 metrics

| # | Group | Metric | Evaluator |
|---|---|---|---|
| 1 | Visual Performance | Video Aesthetics | Q-Align |
| 2 | Visual Performance | Human Anatomy | VBench-2.0 |
| 3 | Visual Performance | Motion Plausibility | Qwen3-VL judge (`action_eval`) |
| 4 | Visual Performance | Text Rendering | OCR |
| 5 | Content Consistency | Background Consistency | SAM 2 (`consistency_segmented`) |
| 6 | Content Consistency | Subject Consistency | SAM 2 (`consistency_segmented`) |
| 7 | Content Consistency | Facial Consistency | InsightFace |
| 8 | Content Consistency | Subject Correctness | Qwen3-VL judge |
| 9 | Content Consistency | Shot Consistency | TransNetV2 |
| 10 | Content Consistency | Style Match | Qwen3-VL judge |
| 11 | Audio & Semantic Alignment | Audio Aesthetics (PQ) | audiobox-aesthetics |
| 12 | Audio & Semantic Alignment | Sound-Source Match | Syncformer |
| 13 | Audio & Semantic Alignment | Lip Sync (offset) | SyncNet |
| 14 | Audio & Semantic Alignment | Lip Sync (confidence) | SyncNet |
| 15 | Audio & Semantic Alignment | Audio Generation | Qwen3-Omni judge |
| 16 | Audio & Semantic Alignment | Emotion Match | Qwen3-Omni judge |
| 17 | Audio & Semantic Alignment | Speaker ID Match | Qwen3-Omni judge |
| 18 | Audio & Semantic Alignment | Speech Accuracy | Qwen3-ASR |

All metrics are linearly mapped to 0-100 for aggregation; metrics 12 and 13 are
"lower is better" and are reversed before averaging. A model receives an overall
score only if it has non-missing values in all three groups.

## Evaluation details

`evaluation/README.md` documents the full pipeline (environment setup per
evaluator, weight download, sharding, and troubleshooting) in Chinese.
`evaluation/_common/spec_kind.py` maps every `spec_id` to its shot setting and
can be checked independently:

```bash
python3 evaluation/_common/spec_kind.py --show count
```

## What is not included

- **Model weights** (Qwen3-Omni, Qwen3-VL, VBench-2.0, SAM 2, ...). `setup_all.sh`
  downloads most of them automatically; the large judge models must be fetched
  manually (see `evaluation/README.md`).
- **Generated videos** and per-evaluator `runs/` outputs.
- **Prompt-construction service** (`qwen3omni_deploy/`). The 900 prompts and
  their specs are already included, so this is only needed to regenerate prompts.
- **Qwen3-VL serving code** for the LLM judges; point the evaluators at your own
  OpenAI-compatible / vLLM endpoint.

## License

Original code and data: **MIT** (see `LICENSE`).

Third-party components keep their own licenses; in particular **Q-Align is under
the S-Lab License 1.0 (non-commercial)** and **YOLO-World / mmyolo are under
GPL-3.0**. See `THIRD_PARTY_NOTICES.md` before redistributing or using this
repository commercially.

## Citation

```bibtex
@article{liu2026castbench,
  title   = {CAST-Bench: Tracing Every Sound to Its Source in Multi-Shot
             Audio-Video Generation},
  author  = {Liu, Yucong and Tian, Qi and Tu, Shuyuan and Pan, Kaihang and
             Kong, Weijie and Pan, Chao},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Acknowledgements

CAST-Bench builds on VBench-2.0, Q-Align, audiobox-aesthetics, Syncformer,
SyncNet, TransNetV2, Qwen3-ASR, SAM 2, Grounding DINO, InsightFace and
PaddleOCR. We thank the maintainers of these projects.
