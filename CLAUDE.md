# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

perio_OCR 是一个牙周图表 OCR 数据项目，用于数字化牙科牙周检查数据。支持使用 PaddleOCR 识别手写牙周探诊表格。

## 文件说明

| 文件 | 说明 |
|------|------|
| `periodontalchart_data.txt` | 牙周图表数据（JSON格式） |
| `handwrite.jpg` | 手写牙周图表图像（OCR处理源） |
| `electronic.pdf` | 电子版牙周图表 |
| `EN_Periodontal_Chart_Scoring_Sheet.xlsx` | 牙周评分表参考 |
| `chart-perio-tools.png` | 工具参考图 |
| `ocr_system/` | OCR识别系统代码 |

## OCR系统架构

```
ocr_system/
├── __init__.py
├── main.py           # 主程序入口
├── paddle_ocr.py     # PaddleOCR调用（优化版）
├── table_detector.py # 表格检测与定位
├── cell_extractor.py # 单元格提取
├── data_parser.py    # OCR结果解析
└── data_mapper.py    # 数据映射
```

## 启动命令

```bash
# 创建conda环境
conda create -n perio_ocr python=3.9
conda activate perio_ocr
pip install paddlepaddle paddleocr opencv-python numpy pandas pyyaml

# 运行OCR (单张)
python run_ocr.py --image handwrite.jpg

# 批量处理
python run_ocr.py --batch input_images/
```

**注意**: 使用CPU模式（AMD显卡不支持CUDA）

## 牙周数据结构

每颗牙齿 (tooth_11 - tooth_48) 包含以下测量数据：

**通用字段：**
- `tooth_XX`: 牙齿存在状态 (0=缺失, 1=存在)
- `mobility_XX`: 活动度
- `implant_XX`: 种植体标志

**6个测量面 (db/b/mb/dp/p/mp 或 dl/l/ml)：**
- `pd_XX_*`: 探诊深度 (mm)
- `gm_XX_*`: 牙龈边缘 (mm，正值=牙龈退缩，负值=增生)
- `bop_XX_*`: 探诊出血 (0/1)
- `pi_XX_*`: 菌斑指数 (0/1)

**部分磨牙特有：**
- `furcation_XX_b/dp/mp` 或 `furcation_XX_b/l`: 分叉病变 (0-3级)

## 调试与开发规则

**遇到报错或实现新功能时**:
- 第一优先: 查看本地paddlepaddle安装文件夹中的源码
- 路径: `C:\Users\lazymark2\miniforge3\envs\perio_ocr\Lib\site-packages\paddleocr\`
- 先检查paddleocr源码中的API用法，再查阅在线文档

**常见问题排查**:
- `use_gpu`参数已被移除，PaddleOCR 3.x自动检测CPU/GPU
- API参数变更频繁，以本地安装版本为准
- 查看 `__init__.py` 和 `ocr.py` 文件了解当前版本API

## OCR优化说明

针对以下场景优化：
- **手写数字识别**: 使用PP-OCRv5英文模型，预处理增强笔画
- **拥挤文本**: 增大输入尺寸(960px)，提高检测精度
- **表格重叠**: 过滤表格线误检，处理重叠区域
