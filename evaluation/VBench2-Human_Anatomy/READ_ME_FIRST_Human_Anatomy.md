# VBench-2.0 — Human Anatomy（人体动作/解剖异常检测）评测点精简包

本包从 https://github.com/Vchitect/VBench/tree/master/VBench-2.0 抽取，
**仅保留运行 `Human_Anatomy` 维度所需的代码与配置**，并剔除了其它 17 个维度的脚本、
仓库配图(assets)、示例视频(exmaple)、YOLO-World 的 demo/deploy/docs 等运行非必需的大文件。
原始 171MB → 精简后约 48MB。

> 注意：**模型权重不包含在内**（在 Google Drive / HuggingFace 分发，体积大）。
> 本评测的 `batch_eval.py` / `run_all.sh` 不会自动下载，按 [evaluation/README.md §1.3](../README.md) 手动放置。

---

## 1. 这个评测点是怎么工作的

逐帧人体解剖异常检测，判断"做体操时肢体是否连贯、有没有多手指/扭曲关节/穿模"：

1. **YOLO-World**（开放词表检测器）在每帧检测 `human`，再从人体裁剪框里检测 `face,hand`，阈值 0.1，bbox 扩成正方形以满足 SimMIM 输入。
2. 裁剪出的 human / face / hand 分别送入 **3 个 ViT(SimMIM 微调) 异常检测器**，各自阈值
   `human=0.4545, face=0.3030, hand=0.3232`。
3. 每帧每个人：3 个模型只要任一判异常 → 该人记 0（异常），否则 1（正常）；对帧内多人取平均。
4. 视频得分 = 各帧平均分的平均。分数越高越"解剖正常"。

核心文件：
- `vbench2/human_anatomy.py` — 维度入口（被 `evaluate.py` 动态 import）
- `vbench2/third_party/ViTDetector/detect.py` — 推理主逻辑（`compute_abnormality`）
- `vbench2/third_party/ViTDetector/models/` — SimMIM / ViT 模型定义
- `vbench2/third_party/YOLO-World/` — YOLO-World 检测器 config（utils.py 默认指向此处）
- `vbench2/utils.py` — 子模块初始化 + 权重自动下载 + 帧读取

---

## 2. 环境安装

由 `evaluation/setup_all.sh` 的 `VBench2-Human_Anatomy` 步骤安装（内部调用本目录的 `setup_env.sh`），
见 [evaluation/README.md §1](../README.md)。`environment.yml` / `requirement.txt` 是上游的依赖清单，仅供参考。

关键依赖：torch、mmcv 2.1.0、mmdet 3.3.0、mmyolo 0.6.0、mmengine、timm、decord、opencv、gdown。

---

## 3. 下载模型权重

权重默认缓存目录由环境变量 `VBENCH2_CACHE_DIR` 决定，未设则为 `<工程根>/vbench2/vbench2_weights`。
上游 `evaluate.py` 首次运行会自动 `gdown` 下载；本评测的 `batch_eval.py` 只检查不下载，请手动放到对应路径：

```
$CACHE/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth   # YOLO-World 权重
$CACHE/anomaly_detector/human.pth     # 人体异常检测器
$CACHE/anomaly_detector/face.pth      # 人脸异常检测器
$CACHE/anomaly_detector/hand.pth      # 手部异常检测器
```
Google Drive file_id（见 `vbench2/utils.py`）：
- YOLO-World: `1qo-K1kum7yiEwIlN1TWDvABXX6qriUen`
- human: `1mlSURYOi_vN9ST1wzEhJaVO6-ZoES8UT`
- face: `1e2qTjrtsYlkWLql0qj8DqaNZ09KuV8Qo`
- hand: `1j3QeAcAtdLe5BFgK-c33UaHV6-iUto0i`

手动下载示例：
```bash
pip install gdown
CACHE=evaluation/VBench2-Human_Anatomy/vbench2/vbench2_weights
gdown https://drive.google.com/uc?id=1qo-K1kum7yiEwIlN1TWDvABXX6qriUen \
  -O "$CACHE/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
```

> SimMIM 预训练（仅训练时需要）：`1dJn6GYkwMIcoP3zqOEyW1_iQfpBi8UOw`，本包面向**推理评测**，可忽略。

---

## 4. 运行评测

把待测视频放到某目录（如 `sampled_videos/MyModel/Human_Anatomy/*.mp4`），然后：

```bash
python evaluate.py \
    --dimension "Human_Anatomy" \
    --videos_path "sampled_videos/MyModel/Human_Anatomy" \
    --mode custom_input
```

或用 CLI：
```bash
vbench2 evaluate --videos_path "sampled_videos/MyModel/Human_Anatomy" --dimension "Human_Anatomy"
```

结果 JSON 会输出每条视频分数与维度均值。多卡可参考 `evaluate.sh` 的 `--max_parallel_tasks`。

---

## 5. 本包做了哪些裁剪（如需完整功能请回原仓库）

已删除：其它 17 个维度的 `.py`、`asset/`、`sampled_videos/`、大部分 `prompts/`（仅留 `prompts/prompt`）、
ViTDetector 的 `assets/ exmaple/ lvis/ data/` 及其下的 YOLO-World 副本、YOLO-World 的 `demo/ deploy/ docs/ assets/`、
mmyolo 的文档与部署文件、上游 README / FAQ。
保留：`Human_Anatomy` 全链路 + 公共模块（utils/distributed/hack_registry/__init__）+ config 引用的 LVIS 标注。

如果你的服务器需要跑**其它维度**或要做**训练**，请直接 clone 完整仓库。
