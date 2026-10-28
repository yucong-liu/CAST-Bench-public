# T2AV-Bench 环境搭建与运行指南（SETUP.md）

> 本文档记录从 0 把 [`t2av-bench`](./README.md) 工程跑起来的完整步骤，重点是
> **如何复用同机的 [`llm_serving`](../llm_serving/README.md) 工程
> 起的本地 vLLM LLM服务** 来做 spec 生成 / prompt 编译。
>
> 工程版本：v0.5（categories/_index.yaml 已引入 L3）

---

## 1. 工程定位

`t2av-bench` 是 T2AV 反向流程评测基准的**造数据层**，按声源 9 桶生成 330 条 spec
并编译成自然语言 prompt。LLM 调用解耦在 `src/t2av_bench/llm/`，可切换：

- `mock`  —— 不调真实 LLM，本地随机生成（开发自检）
- `openai_compat`  —— 通过 OpenAI 兼容 API 调LLM（**本文采用：指向本地 vLLM**）

---

## 2. 物理环境

| 项目 | 值 |
|------|---|
| 操作系统 | Linux |
| GPU | 8 卡（vLLM `tensor-parallel-size 8`） |
| 镜像 | `<your-cuda12.8-vllm-image>` |
| LLM权重路径 | `/path/to/models/<model>-FP8/` |
| vLLM 工程路径 | `/path/to/workspace/llm_serving/` |
| 本工程路径 | `/path/to/workspace/t2av-bench/` |
| 推理服务端口 | `8080`（OpenAI 兼容） |
| served-model-name | `<your-model>` |

---

## 3. LLM vLLM 后端（依赖 llm_serving 工程）

t2av-bench 自身不启动 LLM 服务，而是通过 OpenAI SDK 连接 `llm_serving`
工程已经搭好的 vLLM。

### 3.1 启动 vLLM（在 8 卡机器上执行）
apt-get update && apt-get install -y tmux && tmux -V
```bash
cd /path/to/workspace/llm_serving
./scripts/start_llm_server.sh        # tmux 会话 llm_serve
./scripts/status.sh               # 查看就绪情况
```

首次加载 FP8 权重约 **5–10 分钟**。

### 3.2 健康检查

```bash
curl -s http://127.0.0.1:8080/v1/models
# 期望看到 id="<your-model>"
```

### 3.3 已知坑位（启动前自查）

| 现象 | 原因 | 处理 |
|------|------|------|
| `start_llm_server.sh` 立刻报 `找不到 .../config.json` | [config.env](../llm_serving/config.env) 中 `MODEL_DIR` / `WORK_DIR` 与本机实际挂载点不一致 | 已修（2026-05-24）：两者均改为 `/path/to/...`；如未来换机器重新挂载需再核 |
| `connection refused` / `Connection reset` | 本地 HTTPS 代理拦了 127.0.0.1 | 设置 `export NO_PROXY=localhost,127.0.0.1,::1; export no_proxy=$NO_PROXY` |
| GPU OOM | `--max-model-len 128000` 太大 | 改小到 32768 / 16384 重启 |
| 长时间 `RUNNING` 但接口无响应 | tmux 会话内 vllm 还在 loading shards | `tmux attach -t llm_serve` 看进度 |

---

## 4. t2av-bench 自身依赖

### 4.1 安装

```bash
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export no_proxy="localhost,127.0.0.1,.tencent.com,.oa.com,.woa.com"
cd /path/to/workspace/t2av-bench
pip install -e .
# 如需 dev / 可视化：
# pip install -e ".[dev,viz]"
```

依赖列表（来自 [pyproject.toml](./pyproject.toml)）：
`pyyaml`, `pydantic>=2.6`, `jsonschema`, `click`, `pandas`, `tabulate`, `rich`,
`tenacity`, `requests`, `openai>=1.30`。

### 4.2 烟测

```bash
pytest tests/ -q
# 当前 4 个 smoke：vocab 一致性 / spec schema / prompt 编译 / 覆盖矩阵
```

---

## 5. LLM 对接配置

### 5.1 创建 `config/local.yaml`

t2av-bench 的 [src/t2av_bench/config.py](./src/t2av_bench/config.py) 会自动深合并
`config/default.yaml` + `config/local.yaml`，本地部署只需新增 local.yaml：

```yaml
# config/local.yaml
llm:
  provider: openai_compat
  openai_compat:
    base_url: "http://127.0.0.1:8080/v1"
    api_key: "EMPTY"
    spec_model: "<your-model>"
    prompt_model: "<your-model>"
    temperature: 0.7
    top_p: 0.9
    max_tokens: 2048
    timeout_sec: 120
  retry:
    max_attempts: 4
    initial_wait_sec: 2
    multiplier: 2
```

> ⚠️ 注意：`config/local.yaml` 已在 [.gitignore](./.gitignore) 中，请勿提交。
> default.yaml 仍保持指向 DashScope 云 API，方便后续切换。

### 5.2 验证连通

```bash
cd /path/to/workspace/t2av-bench
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=$NO_PROXY

python - <<'PY'
from t2av_bench.config import load_config
from t2av_bench.llm import get_client
from t2av_bench.llm.base import Message

cfg = load_config()
print("provider:", cfg.get("llm.provider"))
print("base_url:", cfg.get("llm.openai_compat.base_url"))
print("model:",    cfg.get("llm.openai_compat.spec_model"))

cli = get_client(config=cfg)
resp = cli.chat([Message(role="user", content="ping，请回复 pong")])
print("---")
print(resp.text)
PY
```

期望输出：`provider: openai_compat` + 模型回复一段话。

---

## 6. 7 阶段反向流程命令

### 6.1 mock 全流程冒烟（不消耗 GPU，5 秒跑完）

```bash
python scripts/01_build_vocab.py
python scripts/02_generate_specs.py --category speech --n 3 --provider mock
python scripts/03_compile_prompts.py --category speech
python scripts/04_validate_prompts.py --category speech
python scripts/05_build_coverage_matrix.py
```

### 6.2 真实LLM全流程（330 条 spec）

```bash
# 阶段 1：词表/分桶配置 sanity check
python scripts/01_build_vocab.py

# 阶段 2：每个声源桶按 target_count 生成（核心 5×50 + 边缘 4×20 = 330）
python scripts/02_generate_specs.py --category all --provider openai_compat

# 阶段 3：spec → prompt 编译
python scripts/03_compile_prompts.py --category all

# 阶段 3.5：prompt 字段召回校验（默认阈值 0.85）
python scripts/04_validate_prompts.py --category all

# 阶段 4：覆盖矩阵分析
python scripts/05_build_coverage_matrix.py

# 阶段 4.5：定向补缺（示例：lipsync 维度补到 50 条）
python scripts/06_fill_gap_specs.py --dimension lipsync --target 50

# 阶段 6.5：拆分公开/评测端
python scripts/07_split_prompts_specs.py
```

> 单条 spec 生成耗时主要看 vLLM 并发与 prompt 长度，本地 235B FP8 单请求约 5–15 s。
> 若需要大并发，建议同时启动多个脚本（不同 category），让 vLLM 自己排队。

---

## 7. 输出物清单

| 路径 | 内容 |
|------|------|
| `specs/{category}.jsonl` | 完整 spec（评测端答案册） |
| `prompts/{category}.json` | 编译后的自然语言 prompt（喂模型方） |
| `coverage_matrix.csv` / `coverage_matrix.md` | 维度覆盖矩阵 |
| `prompts/public/`、`specs/private/`（07 阶段） | 公开/评测端拆分结果 |

---

## 8. 排查清单

| 现象 | 排查 |
|------|------|
| `LLMClient 需要有效的 api_key` 报错 | local.yaml 没生效 / api_key 仍是 `${...}` 形式。检查 `python -c "from t2av_bench.config import load_config; print(load_config().data['llm']['openai_compat'])"` |
| `Connection refused 127.0.0.1:8080` | vLLM 没起或还在 loading；`curl /v1/models` 看 |
| `openai.BadRequestError: response_format=json_object 不支持` | 个别 vLLM 版本不兼容；可临时改 [llm_client.py](./src/t2av_bench/llm/llm_client.py) 的 `chat_json`，去掉 `response_format` 参数，靠 system prompt 强约束 + 客户端解析兜底 |
| spec 大量被 validator 拒（rejected）| 多半是模型选了 vocab 之外的枚举；调高 temperature 或减小 max_tokens 让回答更短，或检查 system prompt 注入是否正确 |
| `prompt char 长度不达标` | 调 `config.compile.{min_chars,max_chars}` 或 `config.validation.min_field_recall` |

---

## 9. 版本记录

| 日期 | 变更 |
|------|------|
| 2026-05-24 | 初版：接入本地 vLLM LLM（127.0.0.1:8080），新增 `config/local.yaml` |
| 2026-05-24 | 修 `llm_serving/config.env` 的 `MODEL_DIR` / `WORK_DIR`：从最初的 `/path/to` 挂载迁到当时可用的挂载点（旧挂载点不存在），后又于 2026-07-13 统一迁到当前的 `/path/to`，详见下条 |

---

## 附：与 llm_serving 工程的关系

两个工程**只共享一个 vLLM 服务进程**，没有任何代码 / 配置 / 数据耦合：

- `llm_serving` 自己用 [prompts/dimensions.yaml](../llm_serving/prompts/dimensions.yaml)
  / [meta_prompts.yaml](../llm_serving/prompts/meta_prompts.yaml) 生成 10 大类
  AV 评测 prompt
- `t2av-bench` 用自己的 `vocab/` + `categories/` + `specs/schema.json` 生成声源分桶 spec

vLLM 服务本身无状态，OpenAI Chat Completions 每次请求独立，**互不影响生成内容**。

---

## LTX-2.3 批量生成时长优化 (2026-06-06)

### 修改内容
为了提升批量生成效率和降低总耗时，对LTX-2.3批量生成脚本进行了时长优化：

**1. single_shot 时长调整**
- 原设置：15秒 (361帧)
- 新设置：10秒 (241帧)
- 修改文件：`LTX-2.3_test/scripts/run_all.sh`
- 修改内容：`SINGLE_NUM_FRAMES=361` → `SINGLE_NUM_FRAMES=241`

**2. multi_shot 逻辑优化**
- 原逻辑：解析prompt中的`[SHOT i/N, Xs, ...]`时长信息
- 新逻辑：每个shot固定为5秒 (121帧)，不再解析prompt时长
- 修改文件：`LTX-2.3_test/scripts/run_all_persistent.py`
- 修改内容：`nf = seconds_to_num_frames(s["sec"], fps)` → `nf = seconds_to_num_frames(5.0, fps)`

### 预期效果
- **single_shot**：生成时间从~280-320秒减少到~180-220秒
- **multi_shot**：每个shot生成时间约60-80秒，总时间大幅减少
- **视频质量**：保持不变，只是时长缩短
- **分辨率**：保持1280×704 (接近720P)

### 运行命令
```bash
cd /path/to/workspace/LTX-2.3_test
python3 scripts/run_all_persistent.py --top-n 10 --num-gpus 8 2>&1 | tee logs/run_all_persistent_console.log
```

### 备注
- 241帧 = 10秒 × 24fps = 240帧 → 对齐到8k+1 = 241帧
- 121帧 = 5秒 × 24fps = 120帧 → 对齐到8k+1 = 121帧
- 修改基于H800 GPU性能估算，实际效果可能因具体任务而异


---

## 2026-07-13 存储盘迁移

### 背景
原挂载盘 `/path/to/` 已下线，工程整体搬迁至新挂载盘 `/path/to/`（去掉了原来的 `nj10/` 中间目录）。

### 路径映射
```
/path/to/   →   /path/to/
```

### 变更范围
本工程 `t2av-bench/` 内所有源码/脚本此前均使用相对路径或环境变量，无硬编码旧盘串；仅本 SETUP 文档中的示例命令、路径清单、历史记录里包含旧盘绝对路径，已随本次迁移一次性全部替换到新盘。

同批迁移的兄弟工程（同样已完成绝对路径替换）：
- `daVinci-MagiHuman_test/`
- `LTX-2.3_test/`
- `Ovi_test/`
- `llm_serving/`
- `generation/`
- `evaluation/`
- `prompts/hunyuan/`（rewrite_zh_to_hunyuan.py 等）

### 复现回退（仅备查，正常无需执行）
如果未来需要回滚字符串（不建议），命令为：
```bash
grep -rl "/path/to" <target_dir> \
  | xargs sed -i 's|/path/to/|/path/to/|g'
```

---

## 2026-07-20 镜头数分布控制（multi_shot 细分）

### 背景
先前 `multi_shot` 由 LLM 自由生成 5–10 镜；本次调整为**每条 spec 显式指定期望镜头数**，
并且严格按用户约定分布：镜头数 2 / 3 / 4 / 5 / {6-9} 的权重比 **0.5 : 1 : 3 : 4.5 : 1**。

### 涉及改动
| 文件 | 改动 |
|---|---|
| [`vocab/global.yaml`](./vocab/global.yaml) | `multi_shot` 注释从 "5–10" 改为 "2–9 个镜头"，coherence_rules 同步 |
| [`src/t2av_bench/spec/validator.py`](./src/t2av_bench/spec/validator.py) | 镜头数硬约束由 `5<=n<=10` 放宽为 `2<=n<=9` |
| [`src/t2av_bench/spec/generator.py`](./src/t2av_bench/spec/generator.py) | 新增 `n_shots_seq` 参数；对每条 spec 在 system prompt 里注入 `shots[] MUST contain EXACTLY {N} items` 与叙事连贯段的动态 N |
| [`scripts/02_generate_specs.py`](./scripts/02_generate_specs.py) | 新增 `--shot-dist` CLI（默认 `0.5:1:3:4.5:1`）与 `_build_n_shots_seq()`（largest-remainder 一次分档到 2/3/4/5/6-9，二次把 6-9 桶均匀展开到 6/7/8/9），并 round-robin 打散顺序 |

### 生成阶段的约束链
1. `02_generate_specs.py` 先按 `--multi-shot-ratio` 决定 single/multi 配额；
2. 对 multi 部分按 `--shot-dist` 权重用 largest-remainder 分档；6-9 档内部再均匀展开；
3. 生成的 `n_shots_seq` 传入 `SpecGenerator.generate_for_category`；
4. `_build_messages` 把 `EXACTLY {N}` 写进 LLM 提示；
5. `SpecValidator` 硬检查 `2<=len(shots)<=9`，与 LLM 提示形成双重保险。

### 每类 150 条 · 生成 + 编译 + 翻译 完整命令

前置：确保本地 vLLM LLM服务在 `127.0.0.1:8080` 上就绪（详见本文 §3）。

```bash
cd /path/to/workspace/t2av-bench

export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=$NO_PROXY

# ── 阶段 2：生成 spec（每类 150 条；multi/single 各半；镜头数按 0.5:1:3:4.5:1）
python scripts/02_generate_specs.py \
  --category all \
  --n 150 \
  --provider openai_compat \
  --multi-shot-ratio 0.5 \
  --shot-dist "0.5:1:3:4.5:1" \
  --concurrency 8 \
  2>&1 | tee logs/02_generate_specs_all_n150.log

# ── 阶段 3：spec → prompt 编译（英文）
python scripts/03_compile_prompts.py --category all \
  2>&1 | tee logs/03_compile_prompts_all_n150.log

# ── 阶段 3.5：prompt 召回校验（可选，快速摸底质量）
python scripts/04_validate_prompts.py --category all \
  2>&1 | tee logs/04_validate_prompts_all_n150.log

# ── 阶段 3.9：中文翻译（依赖 generation/kling-v3/scripts/translator.py 同一 vLLM）
python scripts/translate_prompts.py --all \
  2>&1 | tee logs/translate_all_n150.log

# ── 阶段 4：覆盖矩阵（可选）
python scripts/05_build_coverage_matrix.py \
  2>&1 | tee logs/05_coverage_all_n150.log
```

> `--n 150` 会覆盖 `categories/_index.yaml` 中每类的 `target_count`；如果只想跑其中
> 某几类，把 `--category all` 换成 `--category speech` / `object` 即可。
>
> `--concurrency 8` 建议按 vLLM 实际并发上限调整；LLM 侧本身有 batching，串行 (1) 也能跑，
> 只是慢一些。生成阶段建议 8 卡 vLLM 起步；翻译脚本内部是单线程串行（vLLM 自己 batch）。

