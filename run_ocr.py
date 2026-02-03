#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
牙周图表OCR识别系统 - 入口脚本

支持功能:
- 单张图像OCR识别
- 批量处理多张图像
- GPU加速
- 可视化调试
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from ocr_system.main import PerioChartOCR, create_default_config
import yaml


def setup_logging(debug: bool = False):
    """配置日志"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'

    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # 合并默认配置
            default = create_default_config()
            default.update(config)
            return default

    return create_default_config()


def main():
    parser = argparse.ArgumentParser(
        description='牙周图表OCR识别系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图像处理
  python run_ocr.py --image handwrite.jpg

  # 批量处理
  python run_ocr.py --batch input_images/ --output output/

  # 启用GPU加速
  python run_ocr.py --image handwrite.jpg --gpu

  # 调试模式（保存中间结果）
  python run_ocr.py --image handwrite.jpg --debug
        """
    )

    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', help='输入图像路径')
    input_group.add_argument('--batch', '-b', help='输入图像目录（批量处理）')

    # 输出选项
    parser.add_argument('--output', '-o', default='output/',
                        help='输出目录（默认: output/）')

    # 配置选项
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--gpu', action='store_true',
                        help='启用GPU加速')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                        help='禁用GPU加速')
    parser.set_defaults(gpu=None)

    # 调试选项
    parser.add_argument('--debug', '-d', action='store_true',
                        help='启用调试模式')

    # 版本信息
    parser.add_argument('--version', action='version',
                        version='%(prog)s 1.0.0')

    args = parser.parse_args()

    # 配置日志
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # 加载配置
    config = load_config(args.config)
    config['debug'] = args.debug

    if args.gpu is not None:
        config['ocr']['use_gpu'] = args.gpu

    logger.info(f"OCR配置: GPU={config['ocr']['use_gpu']}, Debug={args.debug}")

    # 初始化OCR系统
    ocr_system = PerioChartOCR(config)

    try:
        if args.image:
            # 单张图像处理
            logger.info(f"处理图像: {args.image}")

            result = ocr_system.process_image(args.image)

            # 保存结果
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)

            output_file = output_path / 'periodontalchart_data.txt'
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(f"结果已保存到: {output_file}")

        elif args.batch:
            # 批量处理
            logger.info(f"批量处理目录: {args.batch}")

            results = ocr_system.process_batch(args.batch, args.output)

            # 输出摘要
            success_count = sum(1 for r in results if r.get('success', False))
            fail_count = len(results) - success_count

            logger.info(f"处理完成: {success_count} 成功, {fail_count} 失败")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
