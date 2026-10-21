# AVGen-Bench · 面部一致性测评点 (multi-model, specs-driven)

T2AV 评测套件中的「面部一致性 (facial consistency)」测评点。

> 本目录是 **多模型批跑** 结构, 与同目录其它评测点 (如 `subject_consistency` /
> `action_eval`) 一致, 在 `models.tsv` 加一行即可纳入新模型。

## 设计要点

| 项 | 说明 |
|----|------|
| 期望主角脸数标签 | **直接读 specs**: `subjects` 中 `type=='human'` 的数量 |
| 评测桶范围 | **默认 3 个有人的桶**: `speech` / `non_verbal_vocal` / `action`（可用 `CATEGORIES=all` 跑全量） |
| 输入数据集 | t2av-bench specs (9 个 L1 桶, 默认过滤后留 3 个) |
| 视频路径约定 | `<root>/<category>/<spec_id>.mp4` |
| 多模型支持 | `models.tsv` 一行一个模型, jobs.tsv 一次跑完所有 |
| 结果汇总 | 每模型 `runs/<m>/{per_video.jsonl, summary.json}` + 跨模型 `compare_summary*.csv` |


## 文件清单

| 文件 | 作用 |
|------|------|
| `batch_eval.py`         | **批量主入口**。InsightFace 检测 -> 贪心 tracklet -> DBSCAN 聚类 -> 主角簇 -> 数量 40 + 稳定性 60 评分。多模型 jobs.tsv + 分片支持。 |
| `merge_shards.py`       | DP 模式下把 N 个 shard 输出合并成 `per_video.jsonl + summary.json`。 |
| `aggregate_summary.py`  | 跨模型按 category 拆 `compare_summary*.csv`, 同时回写每模型 summary 的 mean_score。 |
| `run_all.sh`            | 一键流水线: jobs.tsv -> batch_eval -> (merge) -> aggregate。 |
| `facial_evaluation.py`  | 单视频/可视化版 (PCA + matplotlib), 调试用, 与批量流水线无关。 |

## 模型清单

由 `evaluation/models.tsv` 提供, 当前 6 个 (默认只评测有人的 3 个桶:
`speech / non_verbal_vocal / action`, 其余 6 个桶 (`object / animal /
ambient / instrument / bgm / sfx`) 不在默认评测范围):

| 模型 | output 根 |
|------|-----------|
| LTX-2.3            | `workspace/LTX-2.3_test/output` |
| Ovi                | `workspace/Ovi_test/output` |
| daVinci-MagiHuman  | `workspace/daVinci-MagiHuman_test/output` (实际在 `output/540p`, 自动下钻) |
| seedance           | `workspace/generation/seedance/output` |
| wan2.7             | `workspace/generation/wan2.7/output` |
| kling-v3           | `workspace/generation/kling-v3` (实际在 `kling-v3/output`, 自动下钻) |

新增模型 = 在 `evaluation/models.tsv` 加一行, 不需要改本目录代码。

## 运行前置

```bash
pip install insightface onnxruntime-gpu scikit-learn opencv-python numpy
```

首次运行 InsightFace 会自动下载 `buffalo_l` 模型包到 `~/.insightface/models/`。

## 一键跑

```bash
cd evaluation/facial_consistency
bash run_all.sh
```

可调环境变量 (常用):

| 变量 | 默认 | 说明 |
|------|------|------|
| `N_SHARDS`        | 1   | DP 客户端数。InsightFace 是 GPU 模型, 多 shard 各占一张卡 (CTX_ID=base+i)。单卡留 1。 |
| `CTX_ID`          | 0   | GPU 卡号 |
| `LIMIT`           | 0   | 每模型只跑前 N 条 spec, 调试用 |
| `CATEGORIES`      | `speech,non_verbal_vocal,action` | 逗号分隔的 category 清单, 只评这些桶. 传 `CATEGORIES=all` 取消过滤 (跑全 9 个桶)。 |
| `INCLUDE_ANTHROPOMORPHIC` | 0 | 把 `subjects[type=anthropomorphic]` 也算成期望人脸 (默认不算; InsightFace 检不到卡通脸) |
| `RUN_EVAL`        | 1   | 关掉只跑汇总 |
| `RUN_AGGREGATE`   | 1   | 关掉只跑评测 |
| `MODELS_TSV`      | `evaluation/models.tsv` | 模型清单 |
| 算法相关          | -   | `DBSCAN_EPS / MIN_DET_SCORE / PRIMARY_RATIO_THRESH / MIN_PRIMARY_CLUSTER_DETS / ...` 详见 `run_all.sh` 顶部 |

## 评分流程 & 关键参数

0. **Category 过滤** (默认开启): 只读 `speech / non_verbal_vocal / action`
   这 3 个桶的 specs. 其余桶 (object / animal / ambient / instrument /
   bgm / sfx) 主体多不含人, 跑上来也是跳过 / 零分 拉低均值, 默认不评。
   可用 `CATEGORIES=all` 或 `--categories speech,sfx,...` 重新选.
1. **检测**: InsightFace `buffalo_l`, `det_size=640x640`, 过滤
   `min_det_score=0.6` / `min_face_px=30` / `min_rel_area=0.0005`。
2. **追踪**: 贪心 IoU+余弦匹配 -> tracklet, `min_iou=0.1` + `min_cos=0.35`,
   `max_gap_frames=3`, `min_tracklet_len=3`。
3. **聚类**: DBSCAN `eps=0.65`, `min_samples=1`, 欧氏距离 -> 身份簇。
4. **挑主角簇**: `cluster_len/total_dets >= 0.10` 且 `>=30` 个检测帧。
5. **打分**: `final = 数量匹配×40 + 稳定性×60`
   - 数量: 主角簇数与 `expected_face_count` (=specs 里 type=human 的 subject 数) 严格匹配 = 1, 否则线性衰减。
   - 稳定性: 主角簇内 tracklet 加权 p50 余弦相似度, 0.80 -> 0, 0.92 -> 1。

`expected_face_count == 0` 的样本 (例如 `speech` / `action` 中一些以动
物 / 虚拟形象为主体的 spec) 会被 **跳过, 不参与该模型的 mean_score**。
这与原 batch_eval "max_ids==0 跳过" 行为一致, 避免拿"无脸 -> 检不到 -> 满分"放水。

## 输出布局

```
runs/
  _jobs.tsv                       # 上次跑的 jobs (model -> src -> out)
  eval.shard*.log                 # batch_eval 进程日志
  compare_summary.csv             # 跨模型, 全 spec
  compare_summary_single.csv      # 仅 single_shot
  compare_summary_multi.csv       # 仅 multi_shot
  <model>/
    per_video.jsonl               # 一行一个 spec 的明细 (含 final_score / breakdown / metrics)
    summary.json                  # 该模型的统计 (mean_score / 各类计数 / 用过的 cfg)
```

`compare_summary*.csv` 的列:
`category, score_<m>..., s_count_<m>..., s_stab_<m>..., n_specs_<m>..., n_total_<m>...`

末尾行 `__ALL__` 是跨 category 汇总。

## 调用 batch_eval.py 的两种模式

**多模型 (推荐, run_all.sh 走这个)**:

```bash
python3 batch_eval.py \
  --jobs-tsv runs/_jobs.tsv \
  --specs-dir /path/to/t2av-bench/specs \
  --shard-idx 0 --shard-num 1
```

**单模型 (调试方便)**:

```bash
python3 batch_eval.py \
  --model my-model \
  --video-root /path/to/output \
  --out-dir   runs/my-model \
  --specs-dir /path/to/t2av-bench/specs
```

两种模式输出格式完全一致。
