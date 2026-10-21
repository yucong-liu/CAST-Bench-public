# T2AV-Bench · 评测协议（喂模型规则）

本文是给模型方读的"如何用我们的 prompts 跑视频生成"。本工程只产 prompts，
真实视频生成不在本仓库内（属于下游 `inference/`）。但**生成时的协议必须在本仓库锁死**，
否则评测结果不可比。

## 1. prompts 包格式

发布形式：`prompts/{category}.json` 与 `prompts/all_prompts.csv`。

每条记录三个字段：

```json
{
  "id": "speech_0001",
  "category": "speech",
  "prompt": "An elderly man with reading glasses sits in a quiet living room ..."
}
```

模型方读 `prompt` 字段，原样喂模型即可。**不允许在 `prompt` 外另加 system message、
不允许用 prompt refiner、不允许重写**。

## 2. 视频生成约束

| 字段 | 约束 | 来源 |
|------|------|------|
| 时长 | 8 秒（默认）或 `spec.duration_sec`（如果该 spec 显式指定） | spec.duration_sec |
| 分辨率 | ≥ 480p，比例不限 | 模型方自选 |
| 帧率 | ≥ 16 fps | 模型方自选 |
| 音频 | 必须有同步音轨（mono 或 stereo），≥ 16 kHz | 必填 |
| 模型 prompt refiner | 必须**关闭** | 协议要求 |
| 多 seed | 每条 prompt 至少跑 1 个 seed，可选 N 个 seed | 模型方自选 |

## 3. 视频命名规范

`{provider}_{model}_{record_id}_{seed_idx}.mp4`

例：`openai_sora2_speech_0001_0.mp4`

## 4. 提交格式

模型方将所有视频打包后，回传到评测端。评测端的 `dispatcher/` + `evaluators/`
会按 spec.dimensions_hit 跑评测。

## 5. 用 specs/ 优化 prompt 的禁令

**`specs/` 是评测端 ground truth，禁止用于 prompt 工程优化。**

如果模型方利用 spec.subjects.descriptor / spec.audio_events.l2_id 等字段
反推真实意图、二次润色 prompt，将被视为作弊，结果不入榜单。

## 6. 例外报备

如果你的模型架构无法满足 §2 中的某项约束（如不支持 8 秒视频），
请在提交时在 `submission.metadata.deviations` 字段列出，评测端会标注但不取消评测。
