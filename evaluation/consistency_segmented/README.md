# Segmentation-enhanced Subject / Background Consistency（改进版）

在 VBench 原版一致性指标基础上，**先用当前最强分割模型 SAM 2 把每帧分成主体 / 背景，再分别计算两个一致性**，解决原版「大主体剧烈运动污染背景分数」的缺陷。

## 改进点 vs 原版
| | 原版 (VBench) | 改进版（本工程）|
|---|---|---|
| 主体编码器 | DINO ViT-B/16 | **DINOv2 ViT-B/14**（更强，可切回 DINO）|
| 背景编码器 | CLIP ViT-B/32（整帧）| **CLIP，仅背景区域**（主体被抠掉）|
| 主体/背景分离 | ❌ 无，整帧混算 | ✅ **SAM 2 分割 + 视频追踪**，时序一致的主体掩码 |
| 主体运动污染背景 | 会 | 基本消除 |

## 处理流程
1. **SAM 2** 在首帧自动生成候选掩码，挑「面积适中且靠中心」的作为主体（无需手动点选）；
2. 用 SAM 2 **视频预测器**把该主体在整段视频中**追踪传播**，得到逐帧、时序一致的主体掩码；
3. **主体一致性**：把背景灰掉、只留主体区域 → DINOv2 提特征 → 帧间余弦相似度；
4. **背景一致性**：把主体灰掉、只留背景区域 → CLIP 提特征 → 帧间余弦相似度；
5. 聚合方式与 VBench 一致：`(与前帧 + 与首帧) / 2`，全帧全视频取平均。

## 目录结构
```
consistency_segmented/
├── run_segmented_consistency.py
├── requirements.txt
├── pretrained/                       # 放 SAM2 权重（需自行下载）
└── seg_consistency/
    ├── __init__.py
    ├── common.py                     # 帧加载 + DINOv2/DINO/CLIP 编码器
    ├── segmenter.py                  # SAM2 主体分割 + 视频追踪 + 掩码应用
    └── metrics.py                    # 分区域一致性计算与聚合
```

## 安装
```bash
cd consistency_segmented
pip install -r requirements.txt

# 安装 SAM 2（需从源码装，要求 torch>=2.5.1）
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..

# 下载 SAM 2.1 large 权重
mkdir -p pretrained
wget -P pretrained https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```
> DINOv2 / DINO 通过 `torch.hub` 自动下载；CLIP 首次运行自动下载。

## 运行
```bash
python run_segmented_consistency.py \
    --videos_path /你的模型生成视频目录 \
    --sam2_ckpt   ./pretrained/sam2.1_hiera_large.pt \
    --sam2_cfg    configs/sam2.1/sam2.1_hiera_l.yaml \
    --subject_encoder dinov2 \
    --clip_model  ViT-B/32 \
    --device cuda \
    --output ./segmented_results.json
```
参数：
- `--subject_encoder`：`dinov2`（推荐）| `dino`（与 VBench 严格对齐）
- `--clip_model`：`ViT-B/32`（VBench 对齐）| `ViT-L/14`（更强背景语义）
- `--sam2_cfg`：用别的尺寸时换成对应 config（如 `configs/sam2.1/sam2.1_hiera_b+.yaml`）

输出：终端打印「主体一致性 / 背景一致性」两个平均分，`segmented_results.json` 保存每条视频分数。

## 设计与注意事项
- **掩码应用方式**：非目标区域填中性灰（128），让编码器主要「看见」目标区域，避免硬裁剪带来的边界与尺度问题。
- **SAM2 必须 GPU**：CPU 也能跑但极慢，建议在 GPU 上用 `--device cuda`。
- **可比性**：本指标修改了 background 的计算口径，**分数不再与 VBench 官方榜单可比**，适合做你自己的模型迭代对比 / 自建评测。要对齐官方请用原版工程 `consistency_standalone`。
- **鲁棒性**：若某视频 SAM2 视频追踪不可用，会自动回退到逐帧图像分割；若首帧找不到合适主体，回退为中心框。
- 已对掩码挑选、掩码应用、聚合逻辑做单元测试验证；SAM2 推理部分需在装好 SAM2 的 GPU 机器上首跑验证。
