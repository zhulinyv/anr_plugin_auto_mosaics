# 🫧 自动打码插件 (anr_plugin_auto_mosaics)

[Auto-NovelAI-Refactor](https://github.com/zhulinyv/Auto-NovelAI-Refactor) 的自动打码插件, 自动检测图片中的隐私部位并批量打码, 支持多种检测方法与打码方式。

## ✨ 功能特性

- 🎯 **三种检测方法**:
  - **YOLO+SAM**: YOLO 检测 + SAM 精细分割, 打码区域更贴合轮廓 (推荐, 需 GPU 效果最佳)
  - **YOLO**: 快速检测, 默认模型为自带的 censor.pt
  - **NudeNet**: 备用检测方案
- 🕶️ **五种打码方法**: 像素 / 模糊 / 线条 / 纯色 / 表情
- 🧩 **四个隐私部位检测**: 欧金金 / 欧芒果 / 欧派派 / 欧西利, 可按需勾选
- 🖼️ **单张与批处理**: 支持单张图片或整个目录批量处理
- 📋 **元数据保留**: 打码后还原图片原始元数据

## 📦 依赖

- scipy
- opencv-python
- ultralytics
- segment_anything
- torch
- nudenet

## 🚀 使用方法

1. 在 [Auto-NovelAI-Refactor](https://github.com/zhulinyv/Auto-NovelAI-Refactor) 的插件商店中安装本插件
2. 打开「自动打码」→「批量处理」面板
3. 选择图片或目录, 勾选需要处理的部位, 选择打码方法与参数
4. 点击 **🛠️ 开始处理**, 结果展示在输出区

## ⚙️ 配置说明

在「配置设置」面板可切换检测方法:

| 配置项 | 说明 |
| --- | --- |
| 检测方法 | YOLO+SAM / YOLO / NudeNet |
| YOLO 模型 | 自定义 YOLO 模型路径 (留空使用默认) |
| SAM 模型 | SAM 权重路径, 可选 vit_b / vit_l / vit_h |

- **首次启动默认为 YOLO 检测**; 克隆仓库后仅包含 YOLO 模型, 不包含 SAM 模型
- 选用 **YOLO+SAM** 检测方法后, SAM 模型会自动下载
- **sam_vit_h** 模型最大、精度最高, 对 GPU/CPU 性能要求较高; **sam_vit_b** 模型最小、占用较少
- 使用 YOLO+SAM 时可手动安装 CUDA 版 PyTorch, 利用 GPU 加速检测

## 🎨 打码方法说明

| 方法 | 说明 |
| --- | --- |
| 像素 | 马赛克, 可调节像素大小 |
| 模糊 | 高斯模糊, 可调节模糊半径 |
| 线条 | 线条马赛克, 自动适应图片亮度选择黑色或白色线条, 可调节线条宽度与间隔 |
| 纯色 | 指定颜色填充, 支持十六进制 / rgb() / 数字列表等多种颜色格式 |
| 表情 | 用表情图片覆盖, 可指定表情目录 |

## 🚧 未来计划

1. 支持自定义 YOLO 模型 (已支持部分)
2. 支持视频打码
