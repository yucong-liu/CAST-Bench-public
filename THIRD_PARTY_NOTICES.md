# Third-Party Notices

CAST-Bench builds on several open-source projects. Portions of their code are
vendored under `evaluation/`.
**Each component remains under its original license**, and those licenses take
precedence over the repository's MIT license for the corresponding files.

If you redistribute this repository, keep the license files listed below.

## Components with a license file in this repository

| Path | Upstream | License |
|---|---|---|
| `evaluation/Q-Align/` | https://github.com/Q-Future/Q-Align | **S-Lab License 1.0** (non-commercial research use) |
| `evaluation/VBench2-Human_Anatomy/` | https://github.com/Vchitect/VBench (VBench-2.0) | Apache-2.0 |
| `evaluation/VBench2-Human_Anatomy/vbench2/third_party/YOLO-World/` | https://github.com/AILab-CVC/YOLO-World | **GPL-3.0** |
| `evaluation/VBench2-Human_Anatomy/vbench2/third_party/YOLO-World/mmyolo/` | https://github.com/open-mmlab/mmyolo | **GPL-3.0** |
| `evaluation/audiobox-aesthetics/` | https://github.com/facebookresearch/audiobox-aesthetics | CC-BY-4.0 (attribution required) |
| `evaluation/Syncformer/` | https://github.com/v-iashin/Syncformer | MIT |
| `evaluation/syncnet_python/` | https://github.com/joonson/syncnet_python | MIT |
| `evaluation/TransNetV2/` | https://github.com/ZFTurbo/TransNetV2 | MIT |
| `evaluation/asr_eval/scripts/Qwen3-ASR/` | https://github.com/QwenLM/Qwen3-ASR | Apache-2.0 |

## Components used as dependencies (not vendored)

| Component | Upstream | License |
|---|---|---|
| SAM 2 (`consistency_segmented`) | https://github.com/facebookresearch/sam2 | Apache-2.0 |
| Grounding DINO (`consistency_segmented`) | https://github.com/IDEA-Research/GroundingDINO | Apache-2.0 |
| InsightFace (`facial_consistency`) | https://github.com/deepinsight/insightface | MIT |
| PaddleOCR (`ocr`) | https://github.com/PaddlePaddle/PaddleOCR | Apache-2.0 |
| Qwen3-Omni / Qwen3-VL | https://github.com/QwenLM/Qwen3-Omni | Apache-2.0 |

License texts for the non-vendored dependencies are distributed with those
projects.

## Important notes

* **Q-Align (S-Lab License 1.0)** permits non-commercial research and
  educational use only. Review the full text in `evaluation/Q-Align/LICENSE`
  before any commercial use of the Q-Align-based video-aesthetics metric.
* **YOLO-World / mmyolo (GPL-3.0)** are copyleft. The GPL-3.0 text is in
  `evaluation/VBench2-Human_Anatomy/vbench2/third_party/YOLO-World/mmyolo/LICENSE`.
  The MIT license at the repository root does not apply to those directories.
* **audiobox-aesthetics (CC-BY-4.0)** requires attribution when the metric
  outputs are redistributed.

## Prompts and specifications

The prompt set (`t2av-bench/prompts/`), the structured specifications
(`t2av-bench/specs/`) and the judge questions (`t2av-bench/questions/`) are
original to this project and released under the repository's MIT license.

Parts of the prompts were compiled with Qwen3-VL; the `compiled_by` field in
each prompt record documents this.
