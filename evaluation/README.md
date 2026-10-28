# evaluation/ — T2AV-Bench 全流程评测指南

> **一次跑完 17 个测评 + 18 项分数汇总**, 覆盖画面表现 / 内容一致性 / 音频表现和语义对齐 3 大组.
>
> 参考硬件: 8×GPU (80GB / 卡), CUDA 12.8

---

## 目录

- [快速开始 (TL;DR)](#快速开始-tldr)
- [1. 环境配置 (一次性)](#1-环境配置-一次性)
- [2. 跑步骤 (3 阶段)](#2-跑步骤-3-阶段)
- [3. 追加新模型场景](#3-追加新模型场景)
- [4. 18 项分数汇总](#4-18-项分数汇总)
- [5. 常见故障排查](#5-常见故障排查)
- [6. 目录索引](#6-目录索引)

---

## 快速开始 (TL;DR)

先准备好 [§1.1](#11-前置条件) 的前置条件和 [§1.3](#13-模型权重) 里需要手动下载的权重, 然后在 GPU 节点上前台执行 (用 `tmux new -s eval` 保 SSH 断线不掉):

```bash
tmux new -s eval
cd evaluation

# ==== Step 1: 一次性配环境 (base 环境 + 各工程依赖 + 独立 venv + 自动下载的权重) ====
mkdir -p _setup_logs
bash setup_all.sh 2>&1 | tee _setup_logs/setup_all.$(date +%Y%m%d_%H%M).log

# ==== Step 2: 一键跑全量评测 ====
mkdir -p _run_logs
bash run_all_experts.sh 2>&1 | tee _run_logs/run_all.$(date +%Y%m%d_%H%M).log
```

跑完之后:
- 各工程原生产物: `<expert>/runs/*/summary.{json,csv}` + `<expert>/runs/compare_summary*.csv`
- **18 项跨工程汇总**: `runs_summary/all_models_18metrics.{csv,md,json}`

**追加一个新模型 my_new**:
```bash
echo -e "my_new\t/path/to/my_new/output" >> models.tsv
mkdir -p _run_logs
MODELS=my_new bash run_all_experts.sh 2>&1 | tee _run_logs/run_my_new.$(date +%Y%m%d_%H%M).log
```

---

## 1. 环境配置 (一次性)

### 1.1 前置条件

| 项目 | 要求 |
|---|---|
| Python | 3.12. base 环境用 conda 环境或系统 Python, **不要用 venv**: syncnet_python / Syncformer 的独立 venv 用 `--system-site-packages` 复用 base 的 torch 等大包, venv 套 venv 继承不到 |
| GPU | 8×GPU (80GB / 卡), NVIDIA 驱动支持 CUDA 12.8 (≥ 570) |
| CUDA Toolkit | 12.8 (提供 `nvcc`, 源码编译 mmcv 2.1.0 / flash_attn 时需要) |
| 系统包 | `gcc` / `g++`, `git`, `wget`, `curl`, `ffmpeg` (含 `ffprobe`); `tmux` 可由 `setup_all.sh` 安装 |
| 网络 | PyPI, Paddle 官方源 (`paddlepaddle.org.cn`), OpenMMLab (`download.openmmlab.com`), GitHub, Hugging Face, `dl.fbaipublicfiles.com`, Google Drive (VBench2 权重) |
| 目录 | 模型权重 `$MODEL_ROOT` (默认 `<repo>/models`); 独立 venv 与源码安装的包 `$ENV_ROOT` (默认 `<repo>/envs`) |

### 1.2 一键装环境: `setup_all.sh`

base 环境的全部 pip 依赖 (torch 2.8.0 / vllm 0.11.0 / transformers 4.57.6 等 299 个精确版本) 列在 [`requirements-freeze-clean.txt`](requirements-freeze-clean.txt), `setup_all.sh` 的第一步 `base_env` 用该文件顶部注释里的三条 pip 命令安装; 后续步骤再处理源码安装、独立 venv 和权重.

**用法**:
```bash
cd evaluation

# 全量顺序装 20 步 (前台跑, 需要联网下载)
mkdir -p _setup_logs
bash setup_all.sh 2>&1 | tee _setup_logs/setup_all.$(date +%Y%m%d_%H%M).log

# 只装某几项
bash setup_all.sh --only qwen3_omni_service,action_eval

# 跳过某几项 (例如 base 环境已按 requirements-freeze-clean.txt 装好)
bash setup_all.sh --skip base_env

# 只打印计划, 不真跑
bash setup_all.sh --dry-run
```

**20 步的顺序**:

| # | 名称 | 装什么 |
|---|---|---|
| 1 | `base_env` | 按 `requirements-freeze-clean.txt` 安装 base 环境 |
| 2 | `system_tmux` | `apt` / `yum` 装 tmux (run_all_experts.sh 用它管理 Qwen3-VL 服务) |
| 3 | `qwen3_omni_service` | Qwen3-Omni 推理依赖 (`qwen-omni-utils / decord / flash-attn`, 复用 `qwen3omni_deploy/setup.sh`), 并检查 Qwen3-VL 启动脚本 |
| 4-10 | LLM 客户端 7 项 | `audio / asr / emtion / style / subject / id / action_eval` 各自补少量辅助包; `asr_eval` 自动下载 Qwen3-ASR-1.7B |
| 11 | `syncnet_python` | 独立 venv `.venv_syncnet` (scenedetect 0.5.6.1) + 下载 SyncNet / S3FD 权重 |
| 12 | `TransNetV2` | `transnetv2-pytorch` (自带权重) |
| 13 | `audiobox-aesthetics` | `pip install -e .` + 下载 checkpoint |
| 14 | `facial_consistency` | insightface + onnxruntime-gpu 1.20.2 (CUDA 12) |
| 15 | `VBench2-Human_Anatomy` | mmcv 2.1.0 / mmdet 3.3.0 / mmyolo 0.6.0 (源码, 装在 `$ENV_ROOT/src`) / lvis / CLIP |
| 16 | `Q-Align` | 独立 venv `$ENV_ROOT/qalign` (torch 2.2.2+cu121) + 下载 one-align 权重 |
| 17 | `consistency_segmented` | SAM2 (源码, 装在 `$ENV_ROOT/src`) + SAM2 / GroundingDINO 权重 |
| 18 | `ocr` | paddlepaddle-gpu 3.1.0 (cu126) + paddleocr 3.7.0 / paddlex 3.7.2 |
| 19 | `Syncformer` | 独立 venv `.venv_syncformer` (transformers 4.40 / timm 0.9.16) |
| 20 | `post_setup_fixes` | 运行 `post_setup_fixes.sh`, 修正以上步骤之间已知的包冲突 (opencv / transformers / librosa / DINOv2 权重等) |

与 base 版本冲突的三个工程 (Q-Align / syncnet_python / Syncformer) 在各自的独立 venv 里运行, 不会改动 base 环境.

**关键行为**:
- 每项独立子 shell (`bash -c '...'`), 避免环境变量污染下一步
- 单项失败不中止, 继续下一项 (continue-on-failure), 最后打印 ✅/❌ 汇总表
- 每项独立日志: `_setup_logs/<step>.<timestamp>.log`

### 1.3 模型权重

需要手动准备的只有 Qwen3-Omni、Qwen3-VL 和 VBench2 的 4 个权重, 其余由 `setup_all.sh` 或首次运行时自动下载.

| 模型 | 位置 | 获取方式 |
|---|---|---|
| Qwen3-Omni-30B-A3B-Instruct | `$MODEL_ROOT/Qwen3-Omni/Qwen3-Omni-30B-A3B-Instruct` | **手动**: `huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir <位置>` |
| Qwen3-VL-235B-A22B-Instruct-FP8 | `Qwen3-VL-235B_prompt/config.env` 里的 `$MODEL_DIR/$MODEL_NAME` | **手动**: `huggingface-cli download Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 --local-dir <位置>` |
| VBench2 Human_Anatomy (4 个) | `VBench2-Human_Anatomy/vbench2/vbench2_weights/` (可用 `VBENCH2_CACHE_DIR` 覆盖) | **手动**, 见下方 gdown 命令 |
| Qwen3-ASR-1.7B | `asr_eval/models/Qwen3-ASR-1.7B` | `setup_all.sh` 自动 (HF `Qwen/Qwen3-ASR-1.7B`) |
| Q-Align one-align | `$MODEL_ROOT/q-align/one-align` | `setup_all.sh` 自动 (HF `q-future/one-align`) |
| audiobox-aesthetics | `$MODEL_ROOT/audiobox-aesthetics/checkpoint.pt` | `setup_all.sh` 自动 (`dl.fbaipublicfiles.com`) |
| SyncNet / S3FD | `syncnet_python/data/`, `syncnet_python/detectors/s3fd/weights/` | `setup_all.sh` 自动 (`download_model.sh`) |
| SAM2.1 hiera-large / GroundingDINO-tiny | `consistency_segmented/pretrained/` | `setup_all.sh` 自动 |
| CLIP ViT-B/32 (VBench2 检测器) | Hugging Face 缓存 | `setup_all.sh` 自动 |
| Syncformer `24-01-04T16-39-21` | `Syncformer/logs/sync_models/` (若 `$MODEL_ROOT/Syncformer` 存在则自动软链过去) | 首次运行自动从官方地址下载 |
| DINOv2 ViT-B/14 / CLIP ViT-B/32 (consistency_segmented) | torch hub / CLIP 缓存 | 首次运行自动; DINOv2 由 `post_setup_fixes.sh` 校验完整性 |
| InsightFace buffalo_l | `~/.insightface/models/buffalo_l/` | 首次运行自动 |
| PaddleOCR 模型 | `~/.paddlex/official_models/` | 首次运行自动 |
| TransNetV2 | `transnetv2-pytorch` 包内 | 随包安装 |

VBench2 Human_Anatomy 的 4 个权重 (`gdown` 由 base 环境提供, 在 `evaluation/` 下执行):

```bash
CACHE=VBench2-Human_Anatomy/vbench2/vbench2_weights
mkdir -p "$CACHE/YOLO-World" "$CACHE/anomaly_detector"
gdown 'https://drive.google.com/uc?id=1qo-K1kum7yiEwIlN1TWDvABXX6qriUen' \
  -O "$CACHE/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
gdown 'https://drive.google.com/uc?id=1mlSURYOi_vN9ST1wzEhJaVO6-ZoES8UT' -O "$CACHE/anomaly_detector/human.pth"
gdown 'https://drive.google.com/uc?id=1e2qTjrtsYlkWLql0qj8DqaNZ09KuV8Qo' -O "$CACHE/anomaly_detector/face.pth"
gdown 'https://drive.google.com/uc?id=1j3QeAcAtdLe5BFgK-c33UaHV6-iUto0i' -O "$CACHE/anomaly_detector/hand.pth"
```

---

## 2. 跑步骤 (3 阶段)

`run_all_experts.sh` **自动**按 3 阶段安排 GPU 显存, 只有 Phase B 需要起服务, 由 orchestrator 自动起/关.

### 2.1 阶段划分

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Phase A — Qwen3-Omni 本地推理 (各工程进程内加载权重, 不起服务, 串行)       │
│   A1) audio_consistency                                                   │
│   A2) emtion_consistency         (split-modality)                         │
│   A3) id_consistency::transcribe (Qwen3-Omni 转写对白 + 说话人)           │
├───────────────────────────────────────────────────────────────────────────┤
│ Phase B — Qwen3-VL 服务窗口 (orchestrator 起 vLLM @ 8080, 跑完关闭)        │
│   B1) action_eval                (VL yes/no 动作评测)                     │
│   B2) style_consistency          (VL judge 风格)                          │
│   B3) subject_consistency        (VL judge 主体)                          │
│   B4) id_consistency::judge      (CPU 算 cpWER + VL judge 打分)           │
│ [orchestrator 关闭 VL 服务 → nvidia-smi 校验显存回收]                     │
├───────────────────────────────────────────────────────────────────────────┤
│ Phase C — 专家窗口 (无服务, 各工程独占 GPU/CPU)                          │
│   C1)  asr_eval                  (Qwen3-ASR 本地)                        │
│   C2)  syncnet_python            (独立 venv)                             │
│   C3)  TransNetV2                                                         │
│   C4)  audiobox-aesthetics                                                │
│   C5)  facial_consistency                                                 │
│   C6)  VBench2-Human_Anatomy                                              │
│   C7)  Q-Align                   (独立 venv)                             │
│   C8)  consistency_segmented     (依赖 audio_consistency 产物)            │
│   C9)  ocr                       (CPU 参考路线)                           │
│   C10) Syncformer                (独立 venv)                             │
├───────────────────────────────────────────────────────────────────────────┤
│ Phase D — 收尾: aggregate_18_metrics.py 生成跨工程分数汇总                │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Qwen3-VL 服务 (Phase B)

- 启动脚本: `Qwen3-VL-235B_prompt/scripts/start_qwen3vl.sh` (脚本内部自建 tmux 会话 `qwen3vl_serve`, orchestrator 前台调用一次即返回)
- 配置: 脚本从 `Qwen3-VL-235B_prompt/config.env` 读取 `MODEL_DIR` / `MODEL_NAME` 等参数
- 服务地址: `127.0.0.1:8080` (OpenAI 兼容 API), 服务名 `Qwen3-VL-235B-A22B-Instruct`
- 就绪等待: 最长 `VL_READY_TIMEOUT` 秒 (默认 900)
- 关闭方式: `tmux kill-session -t qwen3vl_serve` + `pkill -f 'vllm serve.*Qwen3-VL'`
- 手动启动说明: [../t2av-bench/SETUP.md §3](../t2av-bench/SETUP.md)

### 2.3 完整用法

```bash
cd evaluation

# 全量: A → B → C → D
mkdir -p _run_logs
bash run_all_experts.sh 2>&1 | tee _run_logs/run_all.$(date +%Y%m%d_%H%M).log

# 只跑某几个模型 (其它模型的结果保留)
MODELS=seedance,wan2.7 bash run_all_experts.sh

# 只跑某几个工程 (不涉及 B 阶段时 orchestrator 不起服务)
EXPERTS=Q-Align,Syncformer bash run_all_experts.sh

# 只跑某几个阶段
PHASES=A bash run_all_experts.sh           # 只 A 阶段 (Qwen3-Omni 本地推理)
PHASES=A,B bash run_all_experts.sh         # A + B
PHASES=C bash run_all_experts.sh           # 只 C 阶段 (无服务)
PHASES=C,D bash run_all_experts.sh         # C + 汇总

# 已自己起好 Qwen3-VL 服务 (调试排查用)
SKIP_SERVICE_MGMT=1 bash run_all_experts.sh

# 断点续跑: 某工程对所有选中模型都已有产物时整体跳过
SKIP_IF_DONE=1 MODELS=my_new bash run_all_experts.sh

# 只打印计划, 不真跑
DRY_RUN=1 bash run_all_experts.sh

# 组合: 追加新模型, 只跑 A/B
MODELS=my_new PHASES=A,B bash run_all_experts.sh

# 组合: 新模型只跑 facial_consistency (只需 C 阶段)
MODELS=my_new EXPERTS=facial_consistency bash run_all_experts.sh
```

### 2.4 智能启停

orchestrator 只在 PLAN 里有 B 阶段工程时才起 Qwen3-VL 服务:

| 用户传参 | 起 Qwen3-VL 服务? |
|---|---|
| 无 (全量) | ✅ |
| `EXPERTS=Q-Align,facial_consistency` (全在 C) | ❌ |
| `EXPERTS=audio_consistency` (只 A) | ❌ |
| `EXPERTS=action_eval` (只 B) | ✅ |
| `EXPERTS=id_consistency` (跨 A+B) | ✅ |
| `PHASES=A` 或 `PHASES=C` | ❌ |
| `SKIP_IF_DONE=1` 且 B 阶段工程全部已有产物 | ❌ |

### 2.5 关键行为

- **每工程独立子 shell**, 环境变量不污染下一步
- **continue-on-failure**: 单个工程挂了不影响后面, 最后打印 ✅/❌/⏭ 汇总表
- **每工程独立日志**: `_run_logs/<expert>.<timestamp>.log`
- **前台跑 + tee 落盘**: 只有 Qwen3-VL 服务用 tmux 后台, 评测工程本身全部前台
- **`id_consistency` 拆两段** (`transcribe` @ Phase A + `judge` @ Phase B), 用户传 `EXPERTS=id_consistency` 自动展开
- **独立 venv**: Q-Align / syncnet_python / Syncformer 在各自 venv 中运行, 不影响 base 环境

### 2.6 前置检查

启动时自动做以下检查, 任一失败会立即 `exit 2`:

1. `models.tsv` 存在, 有内容
2. `MODELS=xxx` 里每个模型都在 `models.tsv` 里
3. `EXPERTS=xxx` 里每个工程都是合法名字 (含 `id_consistency` alias)
4. `PHASES=xxx` 里每个阶段都是 `A/B/C/D`
5. `consistency_segmented` 依赖 `audio_consistency/runs/qwen3_omni/`: 若本次 PLAN 里没 audio_consistency 且产物缺失, 则从 PLAN 摘掉, 标 `SKIPPED-DEPENDENCY-MISSING`

### 2.7 修改 orchestrator

- 增删或调整工程顺序: 改 `setup_all.sh` 的 `ALL_EXPERTS` 数组, 以及 `run_all_experts.sh` 的 `PHASE_A/B/C_EXPERTS` 数组
- 改某个工程的 setup / run 命令: 改对应的 `CMD[<expert>]=...`
- skip / 模型过滤 / `compare_summary` 生成由各工程自己的 `run_all.sh` 负责, 不要在 orchestrator 里重复实现
- 两个脚本都不用 `set -e` (continue-on-failure), 每个工程必须在独立的 `bash -c` 子 shell 里跑

---

## 3. 追加新模型场景

**目标**: 只对新模型 `my_new` 跑 17 项评测, 其它模型已有的结果保留, 汇总表自动带上新模型.

### 3.1 完整命令

```bash
# Step 1: models.tsv 追加一行 (TAB 分隔: 模型名 \t 视频 output 根目录)
echo -e "my_new\tgeneration/my_new/output" \
  >> evaluation/models.tsv

# Step 2: 一键跑
cd evaluation
mkdir -p _run_logs
MODELS=my_new bash run_all_experts.sh 2>&1 | tee _run_logs/run_my_new.$(date +%Y%m%d_%H%M).log
```

### 3.2 orchestrator 内部把 `MODELS=my_new` 转换成什么

不同工程支持的 filter 参数不同, orchestrator 自动映射:

| 传给工程的参数 | 使用它的工程 |
|---|---|
| `ONLY_MODELS='my_new'` | audio / asr / emtion / style / subject / id / action_eval / syncnet / ocr |
| `MODELS='my_new'` | audiobox / facial / VBench2 / Q-Align |
| `--models 'my_new'` + `--append_summary` (CLI) | consistency_segmented |
| `MODELS_TSV=<完整 models.tsv>` + `ONLY_MODELS='my_new'` | TransNetV2 / Syncformer (保留所有模型的 compare 列) |
| `--append-models 'my_new' --skip-existing` | id_consistency 的 judge 步骤 |

### 3.3 幂等 skip 机制

各工程 `run_all.sh` 支持"已有 `summary.json` / `results.csv` 的模型自动 skip", 所以指定 `MODELS=my_new` 时:
- 不会误跑其它模型
- `compare_summary*.csv` 由各工程 aggregate 时扫全部 `runs/*/summary.csv`, 自动带上所有模型
- 18 项汇总 (Phase D) 也会包含所有模型的行

### 3.4 常见追加变种

```bash
# 追加多个新模型
MODELS=my_new1,my_new2 bash run_all_experts.sh

# 让某个工程重跑一个已有模型 (比如 seedance 某工程结果不满意)
#   要先手动删 runs/<expert>/seedance/ 才不会被 skip
rm -rf audio_consistency/runs/qwen3_omni/seedance
MODELS=seedance EXPERTS=audio_consistency bash run_all_experts.sh

# 新模型只跑不涉服务的专家
MODELS=my_new EXPERTS=Q-Align,facial_consistency,ocr bash run_all_experts.sh

# 新模型跑到某一阶段停下 (人工检查中间产物)
MODELS=my_new PHASES=A bash run_all_experts.sh
```

---

## 4. 18 项分数汇总

Phase D 由 `aggregate_18_metrics.py` 自动跑, **无需手动调用** (除非你想改参数重新汇总).

指标来源、归一化区间、方向和权重都定义在 [scoring.py](scoring.py) 的 `SUMMARY_METRICS` 与 `METRIC_WEIGHT_*` 常量里, `aggregate_18_metrics.py` 动态 import 后直接调用其中的 `build_score_summary()`.

**single / multi 两档.** 每个评测点除全样本的 `compare_summary*.csv` 外, 还按 spec 的 `shot_count` (`single_shot` / `multi_shot`) 额外输出 `<原文件名>_single.csv` / `<原文件名>_multi.csv` (已按维度拆分的表, 后缀加在维度名后, 如 `compare_summary_PQ_single.csv`). 划分由公共 helper [`_common/spec_kind.py`](_common/spec_kind.py) 完成: 它扫描 `t2av-bench/specs/*.jsonl` 构造 `spec_id → shot_count_kind`; 视频统一命名为 `{spec_id}.mp4`, 所以只记录 `video_path` 的结果也能按文件名反查. specs 目录可用 `T2AV_SPECS_DIR` 覆盖, 自检: `python3 _common/spec_kind.py --show count`. 开源模型 LTX-2.3 / Ovi / daVinci-MagiHuman 只生成单镜头视频, 它们在 multi 表中的列为空属于预期.

### 4.1 18 项指标与原始分来源

每项的原始分 (raw) 读自对应工程的汇总 CSV (路径相对 `evaluation/`, single / multi 两档各一份):
- **wide 表**: 取 `category == "__ALL__"` 那一行里 `<列前缀><模型名><列后缀>` 列 (下表写作 `score_<model>` 等);
- **long 表**: 取 `model == <模型名>` 那一行的指定列.

| 大组 | 指标 (论文缩写) | 原始分来源 | 取值列 | 原始区间 | 方向 | 权重 |
|---|---|---|---|---|---|---|
| 画面表现 | 视频美学 (V-Aes) | `Q-Align/runs/compare_summary_{single,multi}.csv` | `<model>` | [0, 1] | ↑ | 50%/13 |
| 画面表现 | 运动人体异常检测 (Anat.) | `VBench2-Human_Anatomy/runs/compare_summary_{single,multi}.csv` | `<model>` | [0, 1] | ↑ | 50%/13 |
| 画面表现 | 运动合理性 (Mot.P) | `action_eval/runs/qwen3_vl/compare_summary_{single,multi}.csv` | `score_<model>` | [0, 1] | ↑ | 50%/13 |
| 画面表现 | 文本渲染能力 (T-OCR) | `ocr/score_out/reference_full_cpu/compare_summary_ocr_{single,multi}.csv` | `mean_score` (long) | [0, 1] | ↑ | 50%/13 |
| 内容一致性 | 背景一致性 (BgC) | `consistency_segmented/runs/compare_summary_{single,multi}_shot.csv` | `bg_<model>` | [0, 1] | ↑ | 50%/13 |
| 内容一致性 | 人物主体一致性 (SbC) | 同上 | `subj_<model>` | [0, 1] | ↑ | 50%/13 |
| 内容一致性 | **面部一致性 (FcC)** | `facial_consistency/runs/compare_summary_{single,multi}.csv` | `score_<model>` | [0, 100] | ↑ | **10%** |
| 内容一致性 | 主体正确性 (SbCor) | `subject_consistency/runs/qwen3_vl/compare_summary_{single,multi}.csv` | `score_<model>` | [0, 1] | ↑ | 50%/13 |
| 内容一致性 | 镜头生成匹配 (CamM) | `TransNetV2/runs/compare_summary_multi.csv` (仅 multi 档) | `<model>` | [0, 1] | ↑ | 50%/13 |
| 内容一致性 | 风格匹配 (StyM) | `style_consistency/runs/qwen3_vl/compare_summary_{single,multi}.csv` | `score10_<model>` | [0, 10] | ↑ | 50%/13 |
| 音频表现和语义对齐 | **音频美学PQ (A-PQ)** | `audiobox-aesthetics/runs/compare_summary_PQ_{single,multi}.csv` | `<model>` | [1, 10] | ↑ | **10%** |
| 音频表现和语义对齐 | 声音-发声主体 (SSM) | `Syncformer/runs/compare_summary_24-01-04T16-39-21_{single,multi}.csv` | `<model>_cont` | [0, 1] | ↓ | 50%/13 |
| 音频表现和语义对齐 | 语音-口型（偏移） (LipO) | `syncnet_python/runs/compare_summary_{single,multi}.csv` | `mean_abs_offset_frames` (long) | [0, 10] | ↓ | 50%/13 |
| 音频表现和语义对齐 | 语音-口型（唇音匹配） (LipM) | 同上 | `mean_confidence` (long) | [0, 10] | ↑ | 50%/13 |
| 音频表现和语义对齐 | **音频生成能力 (AuG)** | `audio_consistency/runs/qwen3_omni/compare_summary_{single,multi}.csv` | `shot_norm_<model>` | [0, 1] | ↑ | **10%** |
| 音频表现和语义对齐 | 情感匹配 (Emo) | `emtion_consistency/runs/qwen3_omni_split/compare_summary_{single,multi}.csv` | `score_<model>` | [0, 1] | ↑ | 50%/13 |
| 音频表现和语义对齐 | **说话人ID对齐 (SpID)** | `id_consistency/score_out/judge/judge_summary_{single,multi}_shot.csv` | `total_avg` (long) | [0, 10] | ↑ | **10%** |
| 音频表现和语义对齐 | **台词正确性 (ASR)** | `asr_eval/runs/compare_summary_{single,multi}.csv` | `score_<model>` | [0, 1] | ↑ | **10%** |

**注**: `consistency_segmented` 产 2 项 (背景 + 人物), `syncnet_python` 产 2 项 (偏移 + 唇音), 所以 16 个工程 → 18 项. 权重列是多镜头档的值; 单镜头档没有「镜头生成匹配」, 非重点的 12 项各 50%/12.

### 4.2 总分怎么算

**第 1 步: 每项线性映射到 0-100, 统一成越高越好.** `[lo, hi]` 是上表的原始区间, 超出区间的值截断到端点:

```
x    = clip((raw - lo) / (hi - lo), 0, 1)
norm = 100 · x          # 方向 ↑
norm = 100 · (1 - x)    # 方向 ↓ (声音-发声主体, 语音-口型（偏移）)
```

例: 音频美学PQ raw = 6.4 → (6.4 − 1) / (10 − 1) × 100 = 60.0; 语音-口型（偏移） raw = 1.2 → (1 − 1.2 / 10) × 100 = 88.0.

**第 2 步: 权重.** 18 项分成两组, 各占 50%:
- 重点组 5 项: 面部一致性、说话人ID对齐、音频美学PQ、音频生成能力、台词正确性;
- 其余组: 另外 13 项.

组内在有分数的指标之间平分: 重点组 5 项各 10%; 其余组在多镜头档 13 项各 50%/13 ≈ 3.85%, 在单镜头档没有「镜头生成匹配」, 12 项各 50%/12 ≈ 4.17%.

**第 3 步: 两组各取均值, 再各按 50% 加权.**

```
overall = 0.5 · mean(重点组中有分数的 norm) + 0.5 · mean(其余组中有分数的 norm)
```

- 某个模型某项缺分 (raw 为空) 时, 只在它所在的组内重新平分, 两组各 50% 不变; 某组完全没有分数时只用另一组.
- 例: Seedance-2.5 单镜头, 重点组均值 73.389, 其余 12 项均值 82.834, overall = 0.5 × 73.389 + 0.5 × 82.834 = 78.11.
- 把论文 Table 2 的各列数值代入这套算法, 可以复现表中全部 21 个 Ovr (差值 < 0.005).

**资格与大组分.** 模型必须在 3 个大组里都至少有 1 项分数才给总分, 否则总分记为 `-`. 大组分 (`group_score__<大组>`) 是组内有分数指标的 norm 等权平均, 只用于展示和资格判断, **不参与**总分计算 (总分直接在各项指标上加权, 不是先算大组分再平均).

**参评模型与奖牌.** single 档包含 `models.tsv` 里的全部模型; multi 档排除不支持多镜头生成的 LTX-2.3 / Ovi / daVinci-MagiHuman. 两档各自按总分排序, 前 3 名记 gold / silver / bronze.

**修改规则.** 来源 / 区间 / 方向改 `scoring.py` 的 `SUMMARY_METRICS`; 权重改同文件的 `METRIC_WEIGHT_HEAVY` (每个重点指标的权重)、`METRIC_WEIGHT_PRIORITY` (重点指标名单) 和 `METRIC_WEIGHT_REST_TOTAL` (其余指标的总权重).

### 4.3 产出 (evaluation/runs_summary/)

| 文件 | 内容 |
|------|------|
| `all_models_18metrics.csv` | 合并表 (single + multi, 加 `shot` 列) |
| `all_models_18metrics_single.csv` | 只 single-shot 档 |
| `all_models_18metrics_multi.csv` | 只 multi-shot 档 |
| `all_models_18metrics.md` | Markdown 表格版 (含 ↑/↓ 方向 / norm100 / 奖牌 emoji) |
| `all_models_18metrics.json` | 结构化 JSON |

CSV 42 列结构 (每行 = 1 个模型):
```
model  ← 主键
├── 18 × <指标>__raw           # 原始值 (未归一化, 4 位小数)
├── 18 × <指标>__norm100       # 线性映射到 0-100 (方向已反向)
├── 3 × group_score__<大组>    # 大组分 (norm100 等权均值)
├── overall_score_norm100      # 综合总分
└── medal                      # gold/silver/bronze/-
```

### 4.4 手动重跑汇总

```bash
cd evaluation

# 全量重扫所有工程的 compare_summary_*.csv
python3 aggregate_18_metrics.py

# 高亮某个新模型 (其它模型保留, 新模型行加粗 + 排行榜带 "⬅ NEW")
python3 aggregate_18_metrics.py --models my_new

# 换个输出目录 (另存一份)
python3 aggregate_18_metrics.py --output-dir runs_summary/snapshot
```

---

## 5. 常见故障排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `Qwen3-Omni` 加载失败 / `from_pretrained` 找不到权重 | Qwen3-Omni 权重没放好 | 按 §1.3 放到 `$MODEL_ROOT/Qwen3-Omni/Qwen3-Omni-30B-A3B-Instruct` |
| `Qwen3-VL 服务在 900s 内未就绪` | FP8 权重加载较慢, 或 `config.env` / 权重路径不对 | `tmux attach -t qwen3vl_serve` 看 vllm 日志; 核对 `Qwen3-VL-235B_prompt/config.env` 的 `MODEL_DIR` / `MODEL_NAME`; 可调大 `VL_READY_TIMEOUT` |
| `Connection refused 127.0.0.1:8080` | HTTPS 代理拦截了 127.0.0.1 | orchestrator 已自动 `export NO_PROXY=localhost,127.0.0.1,::1`; 若手动 curl 也要 export |
| `tmux 未安装` | 系统未安装 | `bash setup_all.sh --only system_tmux` |
| `qwen_omni_utils / decord 缺失` | base 环境或 Omni 依赖没装 | `bash setup_all.sh --only base_env,qwen3_omni_service` |
| `consistency_segmented FAIL: audio_consistency/runs/qwen3_omni/ missing` | Phase A 没跑或跑挂了 | 先 `PHASES=A bash run_all_experts.sh` 单独跑完 |
| `Syncformer FAIL` (transformers / timm 版本报错) | `.venv_syncformer` 没建好 | `bash setup_all.sh --only Syncformer` |
| `syncnet_python` 全部 fail / nan (scenedetect 报错) | `.venv_syncnet` 没建好 | `bash setup_all.sh --only syncnet_python` |
| `Q-Align FAIL: torch 2.2.2 not found` | qalign venv 没建 | `bash setup_all.sh --only Q-Align` (会调用 `Q-Align/setup_env.sh` 建独立 venv) |
| `VBench2-Human_Anatomy: weight not found` | 4 个权重需要手动下载 | 按 §1.3 的 gdown 命令下载 |
| `ocr FAIL: paddlepaddle not found` | ocr 环境没装 | `bash setup_all.sh --only ocr` (会调用 `ocr/rebuild_env.sh`, paddlepaddle-gpu 3.1.0 cu126) |
| transformers / opencv / librosa 版本被某一步改掉 | 各工程 setup 之间的包冲突 | `bash post_setup_fixes.sh` |
| GPU 显存关服务后没释放 | vllm 子进程未干净退出 | 手动 `pkill -9 -f 'vllm serve'` + `sleep 15` + `nvidia-smi` 确认; orchestrator 里已有兜底 pkill |

### 5.1 断开 SSH 也不掉

两个脚本本身都不使用后台启动方式, 只有 **Qwen3-VL 服务**这类长驻进程用 `tmux -d` 管理生命周期 (否则服务无法长驻).

如果需要断开 SSH 不掉整条评测 pipeline, 建议:
```bash
tmux new -s eval             # 先进会话
cd .../evaluation
bash run_all_experts.sh ...  # 前台跑
# Ctrl-B + D 断开会话, 后续 tmux attach -t eval 重新接入
```

### 5.2 Qwen3-VL 服务手动干预 (调试用)

```bash
# 手动起服务 (脚本内部自建 tmux qwen3vl_serve)
cd Qwen3-VL-235B_prompt
bash scripts/start_qwen3vl.sh

# 探活
curl -s http://127.0.0.1:8080/v1/models

# 让 orchestrator 复用你手动起的服务
SKIP_SERVICE_MGMT=1 bash run_all_experts.sh

# 手动关服务
tmux kill-session -t qwen3vl_serve
pkill -9 -f 'vllm serve'
```

---

## 6. 目录索引

```
evaluation/
├── README.md                        本文档
├── requirements-freeze-clean.txt    base 环境的完整 pip 依赖清单 (精确版本)
├── models.tsv                       模型清单 (追加新模型改这里)
│
├── setup_all.sh                     一键环境配置 (20 步)
├── post_setup_fixes.sh              包冲突修正 (setup_all.sh 最后一步自动调用)
├── run_all_experts.sh               一键三阶段评测 (17 个测评 + 18 项汇总)
├── aggregate_18_metrics.py          跨工程分数汇总
├── _common/spec_kind.py             single / multi 划分 helper
│
├── _setup_logs/                     环境配置日志 (运行时生成)
├── _run_logs/                       评测执行日志 (运行时生成)
├── runs_summary/                    18 项汇总产物 (Phase D 输出, 运行时生成)
│
├── ── LLM 客户端工程 (7 个) ──
├── audio_consistency/               Phase A: Qwen3-Omni 本地
├── emtion_consistency/              Phase A
├── id_consistency/                  跨 A+B (transcribe @ A, judge @ B)
├── action_eval/                     Phase B: VL yes/no
├── style_consistency/               Phase B
├── subject_consistency/             Phase B
├── asr_eval/                        Phase C (本地 Qwen3-ASR)
│
├── ── 专家工程 (9 个) ──
├── syncnet_python/                  Phase C (独立 venv)
├── TransNetV2/                      Phase C
├── audiobox-aesthetics/             Phase C
├── facial_consistency/              Phase C
├── VBench2-Human_Anatomy/           Phase C
├── Q-Align/                         Phase C (独立 venv)
├── consistency_segmented/           Phase C (依赖 audio_consistency)
├── ocr/                             Phase C (CPU 参考路线)
└── Syncformer/                      Phase C (独立 venv)
```

**相关工程**:
- `../qwen3omni_deploy/` — Qwen3-Omni 推理依赖的安装脚本 (`setup_all.sh` 的 `qwen3_omni_service` 步骤调用)
- `../Qwen3-VL-235B_prompt/` — Qwen3-VL 服务侧, 提供 `scripts/start_qwen3vl.sh`
- `../t2av-bench/` — 数据侧, 提供 `specs/*.jsonl` 和 `questions/*.jsonl` 给评测用
