# perio_OCR 主程序模块

## 概述

`main.py` 是 perio_OCR 项目的统一入口程序，整合了所有 OCR 模块，提供简洁一致的 API 和命令行接口。

## 架构设计

### 核心类：PerioOCR

统一的牙周图表 OCR 接口，支持多种 OCR 模式和灵活的配置选项。

```
PerioOCR
├── mode: OCR 模式选择
├── _ocr_engine: OCR 引擎实例
├── _preprocessor: 图像预处理器
├── _post_processor: 后处理器
└── process(): 主处理方法
```

### 支持的 OCR 模式

| 模式 | 引擎 | 特点 | 适用场景 |
|------|------|------|----------|
| `auto` | 自动选择 | 根据环境自动选择最佳引擎 | 通用场景 |
| `rapid` | RapidTableOCR | 基于 ONNXRuntime，轻量级 | 需要快速处理 |
| `enhanced` | EnhancedPerioOCR | PaddleOCR + 后处理优化 | 手写数字识别 |
| `paddlex` | PaddleXOCR | PP-Structure 表格识别 | 复杂表格结构 |
| `modular` | ModularOCR | PaddleOCR 独立模块 | 需要精细控制 |

### 数据流程

```
输入图像
  ↓
[图像预处理] (可选)
  ↓
[OCR 识别] (根据选择的模式)
  ↓
[结果标准化]
  ↓
[后处理优化] (可选)
  ↓
[统计计算]
  ↓
输出统一格式数据
```

### 输出格式

```python
{
    "teeth_data": {
        "tooth_11": {
            "tooth_num": "11",
            "tooth": 1,
            "mobility": 0,
            "furcation": 0,
            "pd": {"db": 0, "b": 0, "mb": 0, "dl": 0, "l": 0, "ml": 0},
            "gm": {"db": 0, "b": 0, "mb": 0, "dl": 0, "l": 0, "ml": 0},
            "bop": {"db": 0, "b": 0, "mb": 0, "dl": 0, "l": 0, "ml": 0},
            "pi": {"db": 0, "b": 0, "mb": 0, "dl": 0, "l": 0, "ml": 0}
        },
        ...
    },
    "statistics": {
        "total_teeth": 32,
        "teeth_with_pd": 20,
        "teeth_with_bop": 15,
        "teeth_with_pi": 10,
        "teeth_with_furcation": 5,
        "teeth_with_mobility": 3
    },
    "metadata": {
        "processing_time": 1.23,
        "method": "EnhancedPerioOCR",
        "timestamp": "2026-02-03T10:00:00",
        "image_path": "handwrite.jpg",
        "mode": "enhanced",
        "preprocessing_enabled": true,
        "postprocess_enabled": true
    }
}
```

## 使用方法

### 命令行使用

```bash
# 基本用法
python main.py --image handwrite.jpg

# 指定输出文件
python main.py --image handwrite.jpg --output result.json

# 选择 OCR 模式
python main.py --image handwrite.jpg --mode rapid
python main.py --image handwrite.jpg --mode enhanced
python main.py --image handwrite.jpg --mode paddlex

# 调试模式
python main.py --image handwrite.jpg --debug --visualize

# 禁用预处理/后处理
python main.py --image handwrite.jpg --no-preprocessing --no-postprocess

# 设置日志级别
python main.py --image handwrite.jpg --log-level DEBUG
```

### Python API 使用

```python
from main import PerioOCR

# 创建 OCR 实例
ocr = PerioOCR(
    mode='enhanced',
    enable_preprocessing=True,
    enable_postprocess=True
)

# 处理图像
result = ocr.process(
    image_path='handwrite.jpg',
    output_path='result.json',
    visualize=True
)

# 访问结果
print(f"检测到 {result['statistics']['total_teeth']} 颗牙齿")
print(f"处理方法: {result['metadata']['method']}")
print(f"处理时间: {result['metadata']['processing_time']}s")

# 访问牙齿数据
for tooth_id, data in result['teeth_data'].items():
    print(f"牙齿 {tooth_id}:")
    print(f"  PD: {data['pd']}")
    print(f"  BOP: {data['bop']}")

# 释放资源
ocr.close()
```

### 使用上下文管理器

```python
from main import PerioOCR

with PerioOCR(mode='auto') as ocr:
    result = ocr.process('handwrite.jpg')
    # 自动释放资源
```

## 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--image` | `-i` | 输入图像路径（必需） | - |
| `--output` | `-o` | 输出 JSON 文件路径 | `{image}_result.json` |
| `--mode` | `-m` | OCR 模式 | `auto` |
| `--no-preprocessing` | - | 禁用图像预处理 | False |
| `--no-postprocess` | - | 禁用后处理优化 | False |
| `--debug` | `-d` | 调试模式 | False |
| `--visualize` | `-v` | 生成可视化结果 | False |
| `--log-level` | - | 日志级别 | `INFO` |

## OCR 模式详解

### 1. auto 模式

自动选择最佳 OCR 引擎，优先级：
1. RapidTableOCR（如果已安装）
2. EnhancedPerioOCR（如果已安装）
3. 报错（无可用引擎）

```python
ocr = PerioOCR(mode='auto')
```

### 2. rapid 模式

使用 RapidTableOCR（TableStructureRec + RapidOCR）

**优点**：
- 基于 ONNXRuntime，无需 PaddlePaddle
- 轻量级，处理速度快
- 支持表格分类（有线/无线）

**安装依赖**：
```bash
pip install rapidocr wired_table_rec lineless_table_rec table_cls
```

### 3. enhanced 模式

使用 EnhancedPerioOCR（PaddleOCR + 后处理优化）

**优点**：
- 手写数字识别准确率高
- 支持罗马数字自动修正
- PD 值合理性检查

**安装依赖**：
```bash
pip install paddleocr paddlepaddle
```

### 4. paddlex 模式

使用 PaddleXOCR（PP-Structure）

**优点**：
- 支持复杂表格结构
- 提供 HTML 输出
- 单元格定位精确

**安装依赖**：
```bash
pip install paddleocr paddlepaddle
```

### 5. modular 模式

使用 ModularOCR（PaddleOCR 独立模块）

**优点**：
- 可以精细控制每个处理步骤
- 支持文档方向分类
- 支持图像矫正

**安装依赖**：
```bash
pip install paddleocr paddlepaddle
```

## 依赖安装

### 基础依赖（所有模式）

```bash
pip install opencv-python numpy
```

### PaddleOCR 模式（enhanced, paddlex, modular）

```bash
# CPU 版本
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install paddleocr

# GPU 版本（CUDA）
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
pip install paddleocr
```

### RapidOCR 模式（rapid）

```bash
pip install rapidocr wired_table_rec lineless_table_rec table_cls
```

## 故障排除

### 问题 1：ImportError: No module named 'paddleocr'

**解决方案**：
```bash
pip install paddleocr paddlepaddle
```

### 问题 2：RapidTableOCR 不可用

**解决方案**：
```bash
pip install rapidocr wired_table_rec lineless_table_rec table_cls
```

或切换到 enhanced 模式：
```bash
python main.py --image handwrite.jpg --mode enhanced
```

### 问题 3：GPU 不可用

**解决方案**：

PaddleOCR 3.x 会自动检测 CPU/GPU，无需手动配置。如果需要强制使用 CPU：
```bash
pip install paddlepaddle（而非 paddlepaddle-gpu）
```

### 问题 4：识别结果不准确

**尝试以下方法**：
1. 启用图像预处理：`--preprocessing`
2. 切换 OCR 模式：`--mode enhanced`
3. 启用调试模式查看详情：`--debug`

## 性能优化

### 1. 禁用不需要的功能

```bash
# 禁用预处理和后处理以加快速度
python main.py --image handwrite.jpg --no-preprocessing --no-postprocess
```

### 2. 选择合适的模式

- 速度优先：`rapid`
- 准确度优先：`enhanced`
- 表格结构复杂：`paddlex`

### 3. 批量处理

```python
from main import PerioOCR
from pathlib import Path

ocr = PerioOCR(mode='auto')

image_dir = Path('input_images')
for img_path in image_dir.glob('*.jpg'):
    output_path = f'output/{img_path.stem}_result.json'
    ocr.process(str(img_path), output_path=output_path)

ocr.close()
```

## 最佳实践

1. **首次使用**：从 `auto` 模式开始，让系统自动选择最佳引擎
2. **手写图表**：使用 `enhanced` 模式获得最佳识别效果
3. **快速处理**：使用 `rapid` 模式加快处理速度
4. **调试**：使用 `--debug` 查看详细日志
5. **验证结果**：使用 `--visualize` 生成可视化结果进行验证

## 更新日志

### v1.0.0 (2026-02-03)
- 初始版本
- 整合所有 OCR 模块
- 统一的命令行接口
- 统一的输出格式
- 支持多种 OCR 模式
