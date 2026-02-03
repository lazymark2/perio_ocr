"""
测试脚本 - 验证新添加的 OCR 功能

测试模块:
1. table_line_detector.py - 表格线检测
2. paddlex_ocr.py - PaddleX 表格识别

用法:
    python test_new_features.py              # 测试所有功能
    python test_new_features.py --table      # 只测试表格线检测
    python test_new_features.py --paddlex    # 只测试 PaddleX
    python test_new_features.py handwrite.jpg # 测试指定图像
"""
import os
import sys
import cv2
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class FeatureTester:
    """功能测试器"""

    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'summary': {'total': 0, 'passed': 0, 'failed': 0}
        }

    def add_result(self, test_name: str, passed: bool, details: str = "", data: Any = None):
        """添加测试结果"""
        result = {
            'name': test_name,
            'passed': passed,
            'details': details,
            'data': data
        }
        self.results['tests'].append(result)
        self.results['summary']['total'] += 1
        if passed:
            self.results['summary']['passed'] += 1
            logger.info(f"✓ {test_name}: {details}")
        else:
            self.results['summary']['failed'] += 1
            logger.error(f"✗ {test_name}: {details}")

    def test_table_line_detector(self, image_path: str):
        """测试表格线检测模块"""
        logger.info("\n" + "="*50)
        logger.info("测试表格线检测模块 (table_line_detector.py)")
        logger.info("="*50)

        try:
            from ocr_system.table_line_detector import TableLineDetector
            self.add_result("导入 TableLineDetector", True, "模块导入成功")
        except ImportError as e:
            self.add_result("导入 TableLineDetector", False, f"导入失败: {e}")
            return

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            self.add_result("读取测试图像", False, f"无法读取图像: {image_path}")
            return
        self.add_result("读取测试图像", True, f"图像尺寸: {image.shape}")

        # 创建检测器
        detector = TableLineDetector(min_line_length=50, debug=True)
        self.add_result("创建 TableLineDetector", True, "检测器初始化成功")

        # 测试预处理
        try:
            binary = detector.preprocess(image)
            self.add_result("图像预处理", True, f"二值化图像形状: {binary.shape}")
        except Exception as e:
            self.add_result("图像预处理", False, f"预处理失败: {e}")
            return

        # 测试水平线检测
        try:
            h_lines = detector.detect_horizontal_lines(image)
            self.add_result("水平线检测", True, f"检测到 {len(h_lines)} 条水平线", {
                'count': len(h_lines),
                'lines': h_lines[:3] if h_lines else []  # 只保存前3条
            })
        except Exception as e:
            self.add_result("水平线检测", False, f"检测失败: {e}")

        # 测试垂直线检测
        try:
            v_lines = detector.detect_vertical_lines(image)
            self.add_result("垂直线检测", True, f"检测到 {len(v_lines)} 条垂直线", {
                'count': len(v_lines),
                'lines': v_lines[:3] if v_lines else []
            })
        except Exception as e:
            self.add_result("垂直线检测", False, f"检测失败: {e}")
            return

        # 测试单元格边界计算
        try:
            cells = detector.get_cell_boundaries(h_lines, v_lines)
            self.add_result("单元格边界计算", True, f"生成 {len(cells)} 个单元格", {
                'cell_count': len(cells),
                'rows': max([c['row'] for c in cells]) + 1 if cells else 0,
                'cols': max([c['col'] for c in cells]) + 1 if cells else 0
            })

            # 显示前3个单元格信息
            if cells:
                logger.info("  前3个单元格:")
                for i, cell in enumerate(cells[:3]):
                    bbox = cell['bbox']
                    logger.info(f"    [{i}] 行{cell['row']} 列{cell['col']}: "
                               f"bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})")
        except Exception as e:
            self.add_result("单元格边界计算", False, f"计算失败: {e}")

        # 测试可视化
        try:
            vis_img = detector.visualize_lines(image, h_lines, v_lines)
            output_path = image_path.rsplit('.', 1)[0] + '_table_lines.jpg'
            cv2.imwrite(output_path, vis_img)
            self.add_result("可视化输出", True, f"已保存: {output_path}")
        except Exception as e:
            self.add_result("可视化输出", False, f"可视化失败: {e}")

    def test_paddlex_ocr(self, image_path: str):
        """测试 PaddleX 表格识别模块"""
        logger.info("\n" + "="*50)
        logger.info("测试 PaddleX 表格识别模块 (paddlex_ocr.py)")
        logger.info("="*50)

        try:
            from ocr_system.paddlex_ocr import PaddleXOCR
            self.add_result("导入 PaddleXOCR", True, "模块导入成功")
        except ImportError as e:
            self.add_result("导入 PaddleXOCR", False, f"导入失败: {e}")
            return

        # 创建 OCR 实例
        try:
            ocr = PaddleXOCR(use_table=True, lang='ch')
            self.add_result("创建 PaddleXOCR", True, "OCR 初始化成功")
        except Exception as e:
            self.add_result("创建 PaddleXOCR", False, f"初始化失败: {e}")
            return

        # 测试表格结构检测
        try:
            table_structure = ocr.detect_table_structure(image_path)
            rows = table_structure.get('rows', 0)
            cols = table_structure.get('cols', 0)
            cells = table_structure.get('cells', [])

            self.add_result("表格结构检测", True, f"检测到 {rows}行 x {cols}列，{len(cells)} 个单元格", {
                'rows': rows,
                'cols': cols,
                'cell_count': len(cells)
            })

            # 显示前3个单元格信息
            if cells:
                logger.info("  前3个单元格:")
                for i, cell in enumerate(cells[:3]):
                    logger.info(f"    [{i}] 行{cell.get('row')} 列{cell.get('col')}: "
                               f"text='{cell.get('text', '')[:30]}'")
        except Exception as e:
            self.add_result("表格结构检测", False, f"检测失败: {e}")
            return

        # 测试牙周图表处理
        try:
            perio_data = ocr.process_periodontal_chart(image_path)
            field_count = len(perio_data)
            self.add_result("牙周图表处理", True, f"提取 {field_count} 个数据字段", {
                'field_count': field_count,
                'processing_method': perio_data.get('processing_method', 'unknown')
            })

            # 显示部分数据字段
            sample_fields = list(perio_data.keys())[:10]
            logger.info("  数据字段示例:")
            for field in sample_fields:
                if field not in ['date_saved', 'date', 'processing_method', 'table_structure']:
                    logger.info(f"    {field}: {perio_data[field]}")
        except Exception as e:
            self.add_result("牙周图表处理", False, f"处理失败: {e}")

        # 测试可视化
        try:
            vis_path = image_path.rsplit('.', 1)[0] + '_paddlex_table.jpg'
            ocr.visualize_table_structure(image_path, vis_path)
            self.add_result("表格结构可视化", True, f"已保存: {vis_path}")
        except Exception as e:
            self.add_result("表格结构可视化", False, f"可视化失败: {e}")

        # 测试导出 JSON
        try:
            json_path = image_path.rsplit('.', 1)[0] + '_paddlex_result.json'
            ocr.export_to_json(perio_data, json_path)
            self.add_result("JSON 导出", True, f"已保存: {json_path}")
        except Exception as e:
            self.add_result("JSON 导出", False, f"导出失败: {e}")

    def test_integration(self, image_path: str):
        """测试集成功能 - 结合两个模块"""
        logger.info("\n" + "="*50)
        logger.info("测试集成功能")
        logger.info("="*50)

        try:
            from ocr_system.table_line_detector import TableLineDetector
            from ocr_system.paddlex_ocr import PaddleXOCR
        except ImportError as e:
            self.add_result("集成测试", False, f"模块导入失败: {e}")
            return

        try:
            # 使用表格线检测精确定位区域
            detector = TableLineDetector()
            image = cv2.imread(image_path)

            h_lines = detector.detect_horizontal_lines(image)
            v_lines = detector.detect_vertical_lines(image)
            cells = detector.get_cell_boundaries(h_lines, v_lines)

            # 使用 PaddleX 识别
            ocr = PaddleXOCR(use_table=True)
            table_data = ocr.detect_table_structure(image_path)

            # 结合结果
            self.add_result("集成测试", True, f"表格线检测: {len(cells)} 单元格, "
                                             f"PaddleX: {table_data.get('rows', 0)}行 x {table_data.get('cols', 0)}列", {
                'table_line_detector_cells': len(cells),
                'paddlex_rows': table_data.get('rows', 0),
                'paddlex_cols': table_data.get('cols', 0)
            })
        except Exception as e:
            self.add_result("集成测试", False, f"集成测试失败: {e}")

    def print_summary(self):
        """打印测试摘要"""
        logger.info("\n" + "="*50)
        logger.info("测试摘要")
        logger.info("="*50)

        summary = self.results['summary']
        logger.info(f"总测试数: {summary['total']}")
        logger.info(f"通过: {summary['passed']} ✓")
        logger.info(f"失败: {summary['failed']} ✗")

        if summary['failed'] > 0:
            logger.info("\n失败的测试:")
            for test in self.results['tests']:
                if not test['passed']:
                    logger.info(f"  ✗ {test['name']}: {test['details']}")

        # 保存结果到 JSON
        json_path = f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=lambda x: int(x) if hasattr(x, 'dtype') else str(x))
        logger.info(f"\n测试结果已保存到: {json_path}")

        return summary['failed'] == 0


def find_test_images():
    """查找可用的测试图像"""
    image_extensions = ['.jpg', '.jpeg', '.png']
    test_images = []

    for filename in os.listdir('.'):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            if 'handwrite' in filename.lower():
                test_images.append(filename)

    return test_images if test_images else ['handwrite.jpg']


def main():
    parser = argparse.ArgumentParser(description='测试新添加的 OCR 功能')
    parser.add_argument('image', nargs='?', default='handwrite.jpg',
                       help='测试图像路径 (默认: handwrite.jpg)')
    parser.add_argument('--table', action='store_true', help='只测试表格线检测')
    parser.add_argument('--paddlex', action='store_true', help='只测试 PaddleX')
    parser.add_argument('--all', action='store_true', help='测试所有功能（包括集成）')

    args = parser.parse_args()

    # 检查图像文件
    if not os.path.exists(args.image):
        logger.error(f"图像文件不存在: {args.image}")
        logger.info("可用的测试图像:")
        for img in find_test_images():
            logger.info(f"  - {img}")
        return 1

    tester = FeatureTester()

    if args.table:
        tester.test_table_line_detector(args.image)
    elif args.paddlex:
        tester.test_paddlex_ocr(args.image)
    elif args.all:
        tester.test_table_line_detector(args.image)
        tester.test_paddlex_ocr(args.image)
        tester.test_integration(args.image)
    else:
        # 默认运行所有测试
        tester.test_table_line_detector(args.image)
        tester.test_paddlex_ocr(args.image)

    # 打印摘要
    success = tester.print_summary()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
