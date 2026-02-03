# PPOCRLabel半自动标注工具使用指南

## 概述

PPOCRLabel是PaddleOCR官方提供的半自动图形化标注工具，适用于OCR领域的文本检测与识别标注任务。对于牙周图表OCR项目，可以用于：

1. **标注训练数据**：为模型微调准备标注数据
2. **评估模型效果**：可视化模型预测结果与真实标注的对比
3. **迭代优化模型**：通过标注-训练-评估的循环提升精度

## 安装PPOCRLabel

```bash
# 克隆PPOCRLabel仓库
cd C:\Users\lazymark2\Documents\perio_OCR
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/PPOCRLabel

# 安装依赖
pip install -r requirements.txt

# 启动PPOCRLabel
python PPOCRLabel.py --lang ch
```

## 使用流程

### 1. 创建项目目录

```bash
# 在PPOCRLabel目录下创建项目
mkdir perio_ocr_annotation
cd perio_ocr_annotation

# 目录结构
perio_ocr_annotation/
├── crop_img/          # 裁剪后的图像
├── Label.txt          # 标注文件
├── crop_rec_img/      # 识别裁剪图像
└── file_list.txt      # 图像列表
```

### 2. 准备图像数据

将手写牙周图表图像复制到项目目录：

```bash
# 从主目录复制图像
cp ../handwrite*.jpg ./
cp ../handwrite*.jpg ./crop_img/
```

### 3. 启动标注模式

```bash
# 方式1: 使用通用模型进行预标注
python PPOCRLabel.py --lang ch

# 方式2: 使用微调后的模型（如果有）
python PPOCRLabel.py --lang ch --det_model_dir=path/to/det_model --rec_model_dir=path/to/rec_model
```

### 4. 标注操作

#### 功能快捷键

| 功能 | 快捷键 | 说明 |
|------|--------|------|
| 标注 | W | 标注文本框 |
| 删除 | Q | 删除选中的框 |
| 旋转 | E | 旋转图像 |
| 保存 | S | 保存标注 |
| 下一张 | D | 下一张图像 |
| 上一张 | A | 上一张图像 |

#### 标注步骤

1. **导入图像**：通过界面导入手写牙周图表图像
2. **自动识别**：PPOCRLabel会自动进行OCR识别
3. **修正标注**：
   - 点击识别错误的文本框
   - 修改文本内容
   - 调整边界框位置
4. **添加缺失标注**：按W键添加新的文本框
5. **保存结果**：按S键保存到Label.txt

### 5. 标注格式

Label.txt格式示例：

```
crop_img/handwrite2_1.jpg	[][]	[[[439,308],[478,316],[475,342],[436,334]]]	3.5	0.98
crop_img/handwrite2_2.jpg	[][]	[[[512,300],[550,315],[545,348],[508,333]]]	4	0.95
crop_img/handwrite2_3.jpg	[][]	[[[600,295],[642,310],[638,350],[596,335]]]	I°	0.92
```

格式说明：
```
图像路径\t[旋转角度]\t[坐标点]\t[文本内容]\t[置信度]
```

## 针对牙周图表的特殊标注策略

### 1. 分类标注

为不同数据类型创建单独的标注文件：

```
perio_ocr_annotation/
├── pd_annotation/        # PD值标注
│   ├── crop_img/
│   └── Label.txt
├── bop_annotation/       # BOP标注
│   ├── crop_img/
│   └── Label.txt
├── pi_annotation/        # PI标注
│   ├── crop_img/
│   └── Label.txt
└── mobility_annotation/  # Mobility标注
    ├── crop_img/
    └── Label.txt
```

### 2. 数据增强

在标注时考虑以下情况：
- **手写数字**：0-9的不同写法
- **小数点**：3.5, 4.0等
- **罗马数字**：I°, II°, III°
- **符号**：+, -, ×等

### 3. 质量控制

- 每个数据类型至少标注100个样本
- 覆盖不同的手写风格
- 包含边界情况（模糊、重叠等）

## 导出标注数据

标注完成后，导出数据用于模型训练：

```bash
# 导出为PaddleOCR格式
python PPOCRLabel.py --export_format paddleocr

# 导出为其他格式（如果需要）
python PPOCRLabel.py --export_format json
```

## 与模型训练的集成

标注完成后，可以使用以下命令启动模型微调：

```bash
# 检测模型微调
cd PaddleOCR
python tools/train.py -c configs/det/ch_ppocr_v2.0/det_ch_ppocr_v2.0.yml -o Global.pretrained_model=./pretrain_models/ch_ppocr_mobile_v2.0_det_train/best_accuracy.pdparams

# 识别模型微调
python tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_ch_ppocr_v2.0.yml -o Global.pretrained_model=./pretrain_models/ch_ppocr_mobile_v2.0_rec_train/best_accuracy.pdparams
```

## 常见问题

### Q1: 如何提高标注效率？
A: 使用PPOCRLabel的自动识别功能，然后只修正错误的部分。对于牙周图表这种结构化数据，可以先识别大部分内容，然后专注于修正易混淆的符号。

### Q2: 如何处理多人协作标注？
A: 将图像分配给不同人员，每个人标注一部分，最后合并Label.txt文件。

### Q3: 如何评估标注质量？
A: 使用PPOCRLabel的验证功能，或者让不同人员交叉验证同一批数据。

## 参考链接

- PPOCRLabel官方文档：https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/PPOCRLabel/README.md
- PaddleOCR训练教程：https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/training.md
