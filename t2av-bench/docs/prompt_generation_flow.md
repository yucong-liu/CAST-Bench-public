# Prompt 生成 7 阶段反向流程

## 全局视图

```
[1]  题材池（声源分桶）          ←  categories/{8}.yaml + _index.yaml      ✅ locked
[1.5] 控制词表（补丁 A）          ←  vocab/*.yaml                            ⏳ partial
                                                                                (L1 locked, L2 pending)
[2]  spec 生成                    ←  scripts/02_generate_specs.py
[3]  spec → prompt 编译           ←  scripts/03_compile_prompts.py
[3.5] prompt 校验                 ←  scripts/04_validate_prompts.py
[4]  覆盖矩阵反查（补丁 B）       ←  scripts/05_build_coverage_matrix.py
[4.5] 定向补缺                    ←  scripts/06_fill_gap_specs.py
[5]   🕒 待实施：选测评器        ←  本仓库 eval/evaluators/（spec 冻结后开工）
[6]   🕒 待实施：挂 aux_info     ←  本仓库 eval/runners/（spec 冻结后开工）
[6.5] 公开/评测端拆分             ←  scripts/07_split_prompts_specs.py
[7]   🕒 待实施：打包发布        ←  本仓库 pyproject.toml + scripts/
```

## 模块依赖关系

```
                    ┌──────────────────────────────────────┐
                    │  vocab/  + categories/               │
                    │  ↓ load_vocab_bundle()               │
                    │  src/t2av_bench/vocab/loader.py      │
                    └────────────────┬─────────────────────┘
                                     │
                       VocabBundle (in-memory)
                                     │
            ┌────────────────────────┼────────────────────────┐
            ↓                        ↓                        ↓
   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
   │ SpecGenerator│         │  CoverageMatrix         │ PromptCompiler│
   │   (阶段 2)   │         │   (阶段 4)              │   (阶段 3)    │
   └──────┬───────┘         └──────────────┘         └──────┬───────┘
          │                                                   │
       LLMClient ←─────────── factory.get_client() ──────────→ LLMClient
          │                                                   │
          ↓                                                   ↓
       QwenClient (生产) / MockClient (开发)
```

## 三个补丁在代码中的落点

### 补丁 A：spec 字段从第一天起就用控制词表
- `vocab/audio_events.yaml` 是 L1 + L2 的单一来源（命门）
- `categories/{cat}.yaml.generation_constraints` 把约束硬编码到 spec 生成
- `SpecGenerator._build_messages()` 把 L2 枚举注入 system prompt
- `SpecValidator.validate_business_rules()` 校验枚举合规

### 补丁 B：覆盖矩阵反查
- `CoverageMatrix.gaps(threshold)` 返回所有 < 阈值的格子
- `scripts/06_fill_gap_specs.py` 接收一个 dimension 名 → 定向生成
- 生成出的 spec 标记 `metadata.is_gap_fill = true` + `gap_fill_target`，便于后续审计

### 补丁 C：测评器选不到的字段必须删/转 VQA
- 本阶段尚未实施，留给后续 `eval/` 层处理（spec 冻结后开工）
- 本工程通过 `dimensions_hit` 字段提供候选列表，`eval/evaluators/` 决定保留/转 VQA/删除

## 千问 LLM 部署点

只有一个文件需要关心：`src/t2av_bench/llm/qwen_client.py`。

部署步骤：
1. 阿里云 DashScope 拿到 API key
2. `export DASHSCOPE_API_KEY=sk-xxx`
3. `cp config/default.yaml config/local.yaml` 并按需调整 model / temperature
4. `python scripts/02_generate_specs.py --category speech --n 5 --provider qwen`

如果你想换成自托管千问（vLLM/OpenAI-compatible 推理服务）：
- 修改 `config.llm.qwen.base_url` 为你的服务地址
- `qwen_client.py` 不用动（OpenAI SDK 已经处理）

## 关键约定

| 约定 | 强制位置 |
|------|---------|
| spec.id 格式 `{cat}_{idx:04d}` | `specs/schema.json` + `SpecGenerator._post_process` |
| dimensions_hit 自动派生自 auxiliary_info.keys() | `SpecGenerator._post_process` |
| spec 文件按 category 分文件 jsonl | `scripts/02_generate_specs.py` + `dump_jsonl` |
| prompt 文件按 category 分文件 json | `scripts/03_compile_prompts.py` |
| coverage 阈值 50 | `config/default.yaml.generation.coverage_threshold` |
| L1 桶集合 = 8 | `categories/_index.yaml` + `vocab/audio_events.yaml` 双校验 |

---

## 附录 A：评测层架构决策（2026-06-03 确定，待实施）

### 决策

**evaluators 放在 t2av-bench 单仓库内的 `eval/` 子目录，不拆出去。**

> 这个决策**修正**了正文 [5][6][7] 步早期标注的"⛔ 不在本工程"。
> 修正理由：开源 benchmark 必须自带官方评测代码，否则使用者无法用本数据集
> 测自己的模型。业界主流 T2V/T2AV benchmark（VBench、EvalCrafter、
> T2V-CompBench 等）均为单仓库交付。

### 目标目录结构（待实施）

```
t2av-bench/
├── data/                        # 数据集（轻量，pip 默认装）
│   ├── specs/                   # 现状
│   ├── prompts/                 # 现状
│   └── eval_questions/          # 🆕 VQA 题 + GT，由 spec 编译生成
├── eval/                        # 官方评测代码（重依赖，extras 装）
│   ├── evaluators/              # 各评测器实现（lipsync/asr/vqa/...）
│   ├── runners/                 # 一键跑全套评测
│   ├── leaderboard/             # 榜单
│   └── README.md                # "如何用本 benchmark 评自己的模型"
├── scripts/                     # 现状（数据生成脚本）
└── pyproject.toml
    [project.optional-dependencies]
    eval = ["torch", "openai-whisper", "transformers", ...]
```

### 用户使用方式（设计目标）

```bash
# 只想要数据集
pip install t2av-bench

# 想跑官方评测
pip install "t2av-bench[eval]"
python -m t2av_bench.eval.runners.run_full_eval \
    --videos /my/generated_videos/ --output results.json
```

### 实施路线（分 3 阶段）

| 阶段 | 触发条件 | 要做的事 |
|---|---|---|
| **阶段 1** | spec 词表冻结后 | 建 `eval/` 目录骨架 + 写规范 README（接口、VQA 题格式） |
| **阶段 2** | 阶段 1 完成 | 写 `scripts/09_compile_eval_questions.py`：spec → VQA 题 |
| **阶段 3** | 阶段 2 完成 | 实现具体评测器（lipsync / ASR / VQA / 音画对齐 / ...） |

### 当前状态

⏸ **暂缓实施**。需先完成：
- [ ] spec 字段定型（含 `auxiliary_info` 内的 VQA 钩子字段）
- [ ] `dimensions_hit` 真实分布观察（参考 `categories/_index.yaml` 的 `deferred_decisions.dimensions_vocab`）
- [ ] prompts 完成、用户确认提示词不再大改

满足以上后，从阶段 1 开始动工。

### 历史背景

- 原始设计（正文 [5][6] 步）按"benchmark 数据集 / evaluators 双仓库"切分，
  动机是学术 benchmark 通常希望"出题方与阅卷方解耦"。
- 经讨论修正：单仓库 + 目录隔离 + 依赖隔离（`extras_require`）即可同时
  满足"轻量数据使用"和"官方评测可用"两类需求，无需拆仓库。
- 决策时间：2026-06-03。

### 未来可能扩展：`generation/` 推理层（暂不引入）

**背景**：调研发现部分同类 benchmark 工程会在仓库内直接提供"跑各家模型生成视频"
的适配层，目录结构类似：

```
generation/
├── clients/              # 各家模型 API/本地推理客户端封装
├── batch_<model_a>.py    # 模型 A 批量推理入口
├── batch_<model_b>.py    # 模型 B 批量推理入口
└── runner.py             # 统一调度入口
third_party/              # 第三方模型源码（git submodule 或 vendored）
```

**当前决定（v1.0）：暂不引入**

理由：
1. 当前优先级是先冻结 spec / prompt / 评测协议，再讨论"端到端跑模型"。
2. 维护成本高：每出一个新模型就要写一份适配，依赖会快速膨胀。
3. 业界惯例可分两步走：先发 v1.0（数据集 + 评测脚本），再补 v1.1/v2.0
   的 reference inference adapters，分批交付更稳。

**未来引入时的约束**：
- 必须以独立 extras 形式隔离依赖（如 `pip install t2av-bench[generation]`），
  保证只用数据集 / 评测器的用户不会被推理依赖污染。
- `generation/` 与 `eval/` 解耦：评测器只接受"视频 + spec"，不假设视频来源。
- 第三方模型代码统一放 `third_party/`，优先用 git submodule 而非直接 vendored。

**触发条件**（满足任一可启动该层规划）：
- spec / prompt / 评测协议 v1.0 冻结并对外发布。
- 至少有 2 个模型需要纳入官方榜单复现链路。

---

## 附录 B：评测策略层 —— Router + Rubric 架构（2026-06-03 确定，待实施）

### 背景与动机

附录 A 解决的是"评测器代码放哪里"，本附录解决的是"**哪条 prompt 用哪些评测器、考察哪些维度**"。

实际工程中存在一个核心矛盾：
- spec 阶段的原则是 **"宁可多标，不要漏标"** —— 一条 speech prompt 的 spec 里可能同时记录了 `subjects / shots / bgm / audio_events / visual_style / camera_movement / ...` 等十几个字段。
- 但**真正参与打分**的维度往往只是其中一部分，且会随评测目标变化（例如某次实验只想看"语音-主体一致性"，下次又想加上"BGM 类别匹配"）。

如果把"考察哪些维度"硬编码到 evaluator 里，每次调整都要改代码、动 spec，非常脆弱。
解决方案是引入**策略层**，把"数据"与"考核策略"解耦。

### 三层解耦架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: 数据层  (data/specs/, data/prompts/)               │
│    - 全集标签：subjects / shots / bgm / audio_events / ...  │
│    - 原则：宁可多标，不要漏标                                 │
│    - 一旦 v1.0 冻结，不再随评测策略变动                       │
└─────────────────────────────────────────────────────────────┘
                          ↓ (read-only)
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: 策略层  (eval/rubric/*.yaml)                       │
│    - 定义"考核哪些维度、权重多少、用哪个评测器、从 spec 哪里  │
│      取真值"                                                  │
│    - 支持按 category / 条件触发选择性激活评测点               │
│    - 版本化（v1.0 / v1.1 / v2.0），可同时存在多套并行对比     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: 执行层  (eval/router.py + eval/evaluators/)        │
│    - router 读 rubric → 决定每条 prompt 跑哪些 evaluator      │
│    - 各 evaluator 独立实现：VQA / CLIP / ASR / 客观指标 / ... │
│    - 聚合时按 rubric 中的 weight 加权                         │
└─────────────────────────────────────────────────────────────┘
```

### Rubric 文件示例（设计草案，非最终）

```yaml
# eval/rubric/v1.yaml
version: "1.0"

# —— 全局：所有 prompt 都跑 —— 
global_evaluators:
  - id: visual_quality
    type: objective
    weight: 0.1

# —— 按 category 分流 —— 
by_category:
  speech:
    - id: speech_subject_alignment        # 说话主体一致性
      type: vqa
      source_field: subjects[].speech     # 从 spec 哪里取真值
      question_template: "Is subject {S_id} the one speaking '{speech}'?"
      weight: 0.3
    - id: language_match
      type: asr_lang
      source_field: speech_attributes.language
      weight: 0.2

  bgm:
    - id: bgm_kind_match
      type: audio_classifier
      source_field: bgm.bgm_music_kind
      weight: 0.4
    # 注意：speech 类的 speech_subject_alignment 这里完全不出现
    #      → 这就是"分流"的本质

# —— 按自定义条件激活（对应附录 A 中 auxiliary_info 的 VQA 钩子）—— 
conditional:
  - condition: "category == 'speech' and len(subjects | filter(is_speaker)) >= 2"
    add_evaluators:
      - id: speaker_ip_consistency        # 多说话人 IP 一致性
        type: vqa
        weight: 0.2
```

**关键约定**：
1. `source_field` 告诉 router "去 spec 的哪个字段拿真值" —— 这就实现了
   "固定提示词 → 固定字段 → 固定评测器"的精准对应。
2. **没列在 rubric 里的 spec 字段自动被忽略**（例如 `auxiliary_info.lighting`
   即便存在于 spec 也不会参与打分），从根本上解决"spec 中生成的测评点不一定
   是我需要的"这个问题。
3. rubric 是版本化的；调整考核范围 = 新发一版 rubric，spec 和 evaluator 代码
   都不用动。

### Router 伪代码

```python
# eval/router.py
def route(prompt_id: str, spec: dict, rubric: Rubric) -> list[Task]:
    """根据 rubric 决定该 prompt 要跑哪些评测任务。"""
    tasks = []

    # 1) 全局评测点
    for ev in rubric.global_evaluators:
        tasks.append(Task(ev.id, build_payload(ev, spec)))

    # 2) 按 category 分流
    for ev in rubric.by_category.get(spec["category"], []):
        tasks.append(Task(ev.id, build_payload(ev, spec)))

    # 3) 条件触发
    for rule in rubric.conditional:
        if eval_condition(rule.condition, spec):
            for ev in rule.add_evaluators:
                tasks.append(Task(ev.id, build_payload(ev, spec)))

    return tasks


def aggregate(task_results: list[TaskResult], rubric: Rubric) -> float:
    """按 rubric.weight 加权聚合，得到该 prompt 的最终分数。"""
    ...
```

### 这个架构带来的收益

| 收益 | 说明 |
|---|---|
| **消融实验友好** | 想看"去掉语言一致性后排名怎么变"？复制一份 rubric，注释掉那条评测点重跑即可，spec 和评测器代码完全不动。 |
| **支持子 benchmark** | 别人只关心 speech 部分？发布 `rubric-speech-only.yaml` 即可，无需裁剪数据集。 |
| **方便讨论审稿** | 审稿人问"为什么这条 prompt 不考察 X？" → 直接指 rubric 文件。 |
| **与 auxiliary_info 天然兼容** | spec.auxiliary_info 中预埋的"额外考察点"可直接对应到 rubric 的 conditional 规则触发。 |
| **支持多版本并存** | v1.0 / v1.1 / v2.0 rubric 可同时存在，方便对比"协议演进对榜单的影响"。 |

### 与附录 A 的关系

- 附录 A 决定了 `eval/` 目录在仓库内的形态（单仓库 + extras 隔离）。
- 附录 B 决定了 `eval/` 目录**内部**如何组织：
  ```
  eval/
  ├── rubric/              # 🆕 策略层
  │   ├── v1.yaml
  │   └── README.md
  ├── router.py            # 🆕 执行层入口
  ├── evaluators/          # 已规划：各评测器实现
  └── runners/             # 已规划：一键跑全套
  ```

### 实施触发条件

⏸ **暂缓实施**。与附录 A 共享同一组前置条件：
- [ ] spec 字段定型（含 `auxiliary_info` 内的 VQA 钩子字段）
- [ ] prompts 完成、用户确认提示词不再大改
- [ ] 用户给出初版"评测点表"（speech / bgm / sfx / ... 各类要考察哪几项、
      各自权重）—— 此表是 rubric v1.0 的直接输入

满足以上后，按"先 rubric schema → 再 router → 再 evaluators"的顺序实施。

### 当前不动的文件

- `data/specs/`：保持"全集标签"原则，**不要**为了"评测不考察 X"就从 spec 删掉 X。
- `eval/`：尚未创建，等触发条件满足。

### 决策时间

2026-06-03。本节为思路落档，后续实施时如有调整以代码与 rubric 实际行为为准。
