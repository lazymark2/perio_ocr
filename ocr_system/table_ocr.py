"""
表格识别模块 - 使用TableRecognitionPipelineV2进行表格结构识别
专门用于牙周图表等复杂表格的OCR识别
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from paddleocr import TableRecognitionPipelineV2
import logging
import json

logger = logging.getLogger(__name__)


class TableOCR:
    """基于TableRecognitionPipelineV2的表格OCR识别器"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.lang = self.config.get('lang', 'ch')

        # 初始化TableRecognitionPipelineV2用于表格识别
        self.table_rec = TableRecognitionPipelineV2(
            lang=self.lang,
            use_doc_orientation_classify=False,   # 禁用文档方向分类
            use_doc_unwarping=False,              # 禁用文档去扭曲
            text_det_limit_type='max',           # 限制最大边
            text_det_limit_side_len=960,         # 最大边长960px
        )

        logger.info("表格OCR识别器初始化完成")

    def process(self, image_path: str) -> Dict:
        """
        使用TableRecognitionPipelineV2识别图像中的表格

        Args:
            image_path: 图像路径

        Returns:
            包含表格结构和单元格信息的字典
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 使用TableRecognitionPipelineV2进行表格识别
        result = self.table_rec.predict(
            img,
            use_e2e_wireless_table_rec_model=True,  # 使用端到端无线表格模型
            use_ocr_results_with_table_cells=True,  # 使用OCR结果结合表格单元格
        )

        if not result or len(result) == 0:
            logger.warning("未检测到表格内容")
            return {'cells': [], 'tables': [], 'total_items': 0}

        # 解析结果
        parsed = self._parse_structure_result(result[0], img.shape)

        return parsed

    def _parse_structure_result(self, result: Dict, image_shape: tuple) -> Dict:
        """
        解析TableRecognitionPipelineV2的返回结果

        Args:
            result: TableRecognitionPipelineV2返回的单个结果
            image_shape: 图像尺寸

        Returns:
            解析后的数据
        """
        cells = []
        tables = []

        # TableRecognitionPipelineV2返回格式:
        # - tables: 表格列表
        # - cells: 单元格列表

        # 获取表格结果
        table_results = result.get('tables', result.get('table_results', []))

        for table in table_results:
            table_bbox = table.get('box', [])
            html_content = table.get('html', '')

            tables.append({
                'bbox': table_bbox,
                'html': html_content
            })

            # 解析HTML中的单元格
            table_cells = self._parse_html_table(html_content, table_bbox, image_shape)
            cells.extend(table_cells)

        # 如果没有表格结果，尝试使用OCR结果
        if not tables:
            ocr_results = result.get('ocr_results', [])
            for item in ocr_results:
                bbox = item.get('box', [])
                text = item.get('text', '')
                score = item.get('score', 0.0)

                if text:
                    cells.append({
                        'bbox': self._convert_bbox(bbox),
                        'text': text,
                        'score': score,
                        'table_id': -1  # 不属于任何表格
                    })

        logger.info(f"检测到 {len(tables)} 个表格, {len(cells)} 个单元格")

        return {
            'cells': cells,
            'tables': tables,
            'total_items': len(cells),
            'image_shape': image_shape
        }

    def _parse_html_table(self, html: str, table_bbox: List, image_shape: tuple) -> List[Dict]:
        """
        解析HTML表格内容

        Args:
            html: HTML表格字符串
            table_bbox: 表格边界框
            image_shape: 图像尺寸

        Returns:
            单元格列表
        """
        cells = []

        # 简单HTML解析
        import re

        # 查找所有<td>标签
        td_pattern = r'<td[^>]*>(.*?)</td>'
        matches = re.findall(td_pattern, html, re.DOTALL)

        # 查找所有<tr>标签确定行
        tr_pattern = r'<tr[^>]*>.*?</tr>'
        rows = re.findall(tr_pattern, html, re.DOTALL)

        # 计算表格尺寸
        if table_bbox and len(table_bbox) == 4:
            table_width = table_bbox[2] - table_bbox[0]
            table_height = table_bbox[3] - table_bbox[1]

            # 估算每个单元格的位置
            cell_idx = 0
            for row_idx, row in enumerate(rows):
                # 查找该行中的td数量
                tds_in_row = re.findall(td_pattern, row, re.DOTALL)
                num_cols = len(tds_in_row)

                for col_idx in range(num_cols):
                    if cell_idx < len(matches):
                        text = matches[cell_idx].strip()
                        # 移除HTML标签
                        text = re.sub(r'<[^>]+>', '', text)

                        if text:
                            # 估算单元格位置
                            x1 = table_bbox[0] + (col_idx * table_width / num_cols)
                            y1 = table_bbox[1] + (row_idx * table_height / len(rows))
                            x2 = x1 + (table_width / num_cols)
                            y2 = y1 + (table_height / len(rows))

                            cells.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'text': text,
                                'score': 0.9,  # 默认置信度
                                'table_id': 0,
                                'row': row_idx,
                                'col': col_idx
                            })
                        cell_idx += 1

        return cells

    def _convert_bbox(self, bbox: List) -> List:
        """转换边界框格式"""
        if isinstance(bbox, list) and len(bbox) >= 4:
            if isinstance(bbox[0], (list, tuple)):
                # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] -> [x1, y1, x2, y2]
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            else:
                # 已经是 [x1, y1, x2, y2] 格式
                return bbox[:4]
        return [0, 0, 0, 0]


class PerioTableAnalyzer:
    """牙周图表专用分析器 - 结合表格结构和牙齿位置映射"""

    def __init__(self):
        self.table_ocr = TableOCR()

        # 牙周图表的牙齿布局 (FDI编号系统)
        self.QUADRANT_TEETH = {
            'upper_right': ['18', '17', '16', '15', '14', '13', '12', '11'],
            'upper_left': ['21', '22', '23', '24', '25', '26', '27', '28'],
            'lower_left': ['31', '32', '33', '34', '35', '36', '37', '38'],
            'lower_right': ['41', '42', '43', '44', '45', '46', '47', '48']
        }

    def analyze_chart(self, image_path: str) -> Dict:
        """
        分析牙周图表

        Args:
            image_path: 图像路径

        Returns:
            包含所有牙齿数据的字典
        """
        # 使用表格OCR
        result = self.table_ocr.process(image_path)

        # 解析牙齿数据
        teeth_data = self._extract_teeth_data(result)

        return {
            'teeth_data': teeth_data,
            'tables_detected': len(result.get('tables', [])),
            'cells_detected': result.get('total_items', 0)
        }

    def _extract_teeth_data(self, ocr_result: Dict) -> Dict:
        """从OCR结果中提取牙齿数据"""
        cells = ocr_result.get('cells', [])
        image_shape = ocr_result.get('image_shape', (2000, 1500))

        height, width = image_shape[:2]
        mid_x = width // 2
        mid_y = height // 2

        teeth_data = {}

        for cell in cells:
            text = cell.get('text', '')
            bbox = cell.get('bbox', [0, 0, 0, 0])

            if not text or not bbox:
                continue

            # 计算中心点
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            # 确定象限
            if center_y < mid_y:
                quadrant = 'upper_right' if center_x > mid_x else 'upper_left'
            else:
                quadrant = 'lower_right' if center_x > mid_x else 'lower_left'

            # 提取数字
            digits = self._extract_digits(text)
            if not digits:
                continue

            # 确定牙齿和测量面
            tooth_num, surface = self._locate_tooth_surface(
                center_x, center_y, mid_x, mid_y, quadrant, width, height
            )

            if not tooth_num:
                continue

            # 存储数据
            if tooth_num not in teeth_data:
                teeth_data[tooth_num] = self._create_empty_tooth_data(tooth_num)

            # 根据数字类型存储到不同字段
            if len(digits) == 1 and 0 <= digits[0] <= 10:
                # 单个数字，可能是PD值
                teeth_data[tooth_num]['pd'][surface] = digits[0]

        return teeth_data

    def _extract_digits(self, text: str) -> List[int]:
        """提取文本中的数字，改进识别逻辑"""
        import re

        # 查找所有数字
        numbers = re.findall(r'\d+', text)

        if not numbers:
            return []

        # 转换为整数
        digits = [int(n) for n in numbers]

        # 过滤合理的PD值 (0-10mm)
        valid_digits = [d for d in digits if 0 <= d <= 10]

        return valid_digits if valid_digits else digits[:1]  # 优先返回有效值

    def _locate_tooth_surface(self, x: int, y: int, mid_x: int, mid_y: int,
                              quadrant: str, width: int, height: int) -> Tuple[Optional[str], Optional[str]]:
        """根据坐标定位牙齿和测量面"""
        # 计算相对位置
        if quadrant == 'upper_right':
            rel_x = x - mid_x
            rel_y = y
        elif quadrant == 'upper_left':
            rel_x = x
            rel_y = y
        elif quadrant == 'lower_right':
            rel_x = x - mid_x
            rel_y = y - mid_y
        else:  # lower_left
            rel_x = x
            rel_y = y - mid_y

        # 计算行索引（牙齿位置）
        quadrant_height = mid_y if 'upper' in quadrant else height - mid_y
        row_height = quadrant_height / 8
        row_idx = int(rel_y / row_height)

        # 计算列索引（测量面）
        quadrant_width = mid_x if 'left' in quadrant else width - mid_x
        col_width = quadrant_width / 6
        col_idx = int(rel_x / col_width)

        # 获取牙齿编号
        teeth = self.QUADRANT_TEETH.get(quadrant, [])
        if row_idx < 0 or row_idx >= len(teeth):
            return None, None
        tooth_num = teeth[row_idx]

        # 获取测量面
        if tooth_num[1] in '345678':  # 磨牙/前磨牙
            surfaces = ['db', 'b', 'mb', 'dp', 'p', 'mp']
        else:  # 前牙
            surfaces = ['db', 'b', 'mb', 'dl', 'l', 'ml']

        if col_idx < 0 or col_idx >= len(surfaces):
            return None, None

        return tooth_num, surfaces[col_idx]

    def _create_empty_tooth_data(self, tooth_num: str) -> Dict:
        """创建空的牙齿数据结构"""
        if tooth_num[1] in '345678':  # 磨牙/前磨牙
            surfaces = ['db', 'b', 'mb', 'dp', 'p', 'mp']
        else:  # 前牙
            surfaces = ['db', 'b', 'mb', 'dl', 'l', 'ml']

        return {
            'tooth_num': tooth_num,
            'tooth': 1,
            'mobility': 0,
            'implant': 0,
            'pd': {s: 0 for s in surfaces},
            'gm': {s: 0 for s in surfaces},
            'bop': {s: 0 for s in surfaces},
            'pi': {s: 0 for s in surfaces},
        }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python table_ocr.py <图像路径>")
        sys.exit(1)

    analyzer = PerioTableAnalyzer()
    result = analyzer.analyze_chart(sys.argv[1])

    print(f"检测到表格: {result['tables_detected']}")
    print(f"检测到单元格: {result['cells_detected']}")
    print(f"提取牙齿数据: {len(result['teeth_data'])} 颗")

    # 显示结果
    for tooth_id, data in list(result['teeth_data'].items())[:5]:
        print(f"  {tooth_id}: PD={data['pd']}")
