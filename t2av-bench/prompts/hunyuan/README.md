## 中文提示词 → hunyuan 格式改写脚本

用本地 vLLM 起的 **OpenAI 兼容 LLM** 服务，把
`t2av-bench/prompts/zh/*.json`（9 个类别）里的结构化中文提示词，改写成
[t2va_single_captions_300_5s_15s.csv](./prompt_trans/exp/t2va_single_captions_300_5s_15s.csv) /
[t2va_multi_captions_300_5s_15s.csv](./prompt_trans/exp/t2va_multi_captions_300_5s_15s.csv)
的 hunyuan 目标格式 CSV。

---

### 1. 前置：vLLM 服务

见 [`t2av-bench/SETUP.md`](../../SETUP.md) §3；
简言之在 8 卡机上：

```bash
cd /path/to/llm_serving
./scripts/start_llm_server.sh
./scripts/status.sh
curl -s http://127.0.0.1:8080/v1/models   # 期望看到你的模型名
```

### 2. 运行

```bash
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=$NO_PROXY

cd t2av-bench/prompts/hunyuan

# 全量：9 类 × (single + multi) = 18 个 CSV
python rewrite_zh_to_hunyuan.py --concurrency 8 2>&1 | tee rewrite.log

# 只跑 animal 类调试 3 条
python rewrite_zh_to_hunyuan.py --categories animal --limit 3

# 只重跑 speech 单镜头
python rewrite_zh_to_hunyuan.py --categories speech --shot-count single

# 强制覆盖已生成结果
python rewrite_zh_to_hunyuan.py --overwrite
```

- 默认走 `http://127.0.0.1:8080/v1`、`api_key=EMPTY`、`model=<your-model>`；可用环境变量覆盖：`LLM_BASE_URL / LLM_API_KEY / LLM_MODEL`。
- 并发 = 线程数，vLLM 侧自然排队。
- **断点续跑**：脚本会读取已有 CSV，跳过已完成的 `id`；`--overwrite` 才会重跑。
- 每完成 ~5% 阶段性 flush 一次，中途 Ctrl-C 也不会丢已完成结果。

### 3. 输出

```
prompts/hunyuan/
├── action/
│   ├── action_single_10s.csv
│   └── action_multi_10s.csv
├── ambient/
│   ├── ambient_single_10s.csv
│   └── ambient_multi_10s.csv
├── animal/ …
└── speech/ …
```

**CSV 列（与原 hunyuan 完全对齐，仅最前面多 `id`）：**

```
id, index, prompt, prompt_wo_shots,
gt_video_cos_url, gt_audio_cos_url,
num_frames, scene_shot_detail, seed
```

| 列 | 值 |
|---|---|
| `id` | 源 JSON 里的 `animal_0001` 等 |
| `index` | 每个类别 CSV 内 0 起递增 |
| `prompt` | VL 改写产物（中文，内嵌 `<music>/<sfx>/<speech>/<text>/<lyrics>` 标签；多镜头带 `[0s~2.3s]` 之类时间戳） |
| `prompt_wo_shots` | 概述版（只保留整体描述 + 音频标签） |
| `gt_video_cos_url` | **空**（改写场景无真值视频） |
| `gt_audio_cos_url` | **空**（改写场景无真值音频） |
| `num_frames` | **241**（10s × 24fps + 1） |
| `scene_shot_detail` | single: `[[0, 10]]`；multi: 随机切成 N 段，总时长严格 10s，如 `[[0, 2.3], [2.3, 5.6], [5.6, 7.4], [7.4, 10]]` |
| `seed` | 随机整数 1..9999 |

- 编码统一 **`utf-8-sig`**（Excel / WPS 打开中文不乱码）
- 多镜头段数 N 从源 prompt 的 `[SHOT k/N]` 解析；找不到则回退随机 3–5 段。
- 每段时间戳先在脚本里随机生成好，再连同"每段起止时间列表"塞给 VL，让它按同一份切分写描述，保证 `prompt` 里的时间戳与 `scene_shot_detail` 严格对齐。

### 4. 调试小贴士

- 想只测 1 条：`--categories animal --shot-count single --limit 1`
- 想固定切分与随机 seed：`--seed 42`
- 想改风格/长度：改脚本里 `SYSTEM_PROMPT_SINGLE` / `SYSTEM_PROMPT_MULTI`；few-shot 样例默认从
  `prompt_trans/exp/t2va_{single,multi}_captions_300_5s_15s.csv` 的第 1 行现取，改样例就换第 1 行即可。
- 失败会写 `<category>/<category>_<shot>_10s.errors.log`。

### 5. 与其它工程的关系

- 本脚本**不启动 LLM 服务**，只当 vLLM 的客户端；服务由你自己的 LLM 服务工程拉起（任意 OpenAI 兼容端点）。
- 与 [t2av-bench 主流程](../../SETUP.md) 使用**同一个 vLLM 进程**，无状态、互不影响。
