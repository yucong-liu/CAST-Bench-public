# OCR（文本渲染能力）测评点

流水线思路参考 microsoft/AVGen-Bench 的文本测评点：PaddleOCR 逐帧识别 → 同帧多框合并 → 跨帧追踪成文本轨迹（track）→ 打分。

## 文件清单

| 文件 | 作用 |
|------|------|
| `filter_specs.py` | 扫 `t2av-bench/specs/*.jsonl`，筛出 `text_on_screen` 非空的 spec |
| `extract_tracks.py` + `ocr_pipeline/` | stage 1：PaddleOCR 抽取文本轨迹（多进程分片） |
| `merge_shards.py` | 合并各分片的 `summary.shard*.json` |
| `score_reference.py` | 参考式打分：以 spec 中要求渲染的文本为 GT 计算匹配分 |
| `judge_qwen.py` + `judge_prompt.py` | stage 2（可选）：本地 Qwen3-VL 作为裁判给出多维 0-100 分 |
| `score_summary.py` / `summarize_by_shot.py` | 跨模型汇总、按 single / multi-shot 拆分 |
| `ocr_video.py` | 调试用：单个视频逐帧 OCR，导出 CSV |
| `run_reference_cpu_full.sh` | 全量评测入口（`evaluation/run_all_experts.sh` 默认调用） |

环境配置与运行方式见 [evaluation/README.md](../README.md)。
