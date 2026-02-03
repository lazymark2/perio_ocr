"""
自动表格检测模块 - 使用PaddleOCR的PP-Structure自动识别表格结构
无需手工标注，使用TableMaster模型自动检测单元格
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from paddleocr import PaddleOCR
import logging
import json

logger = logging.getLogger(__name__)


class AutoTableDetector:
    """自动表格检测器 - 使用PaddleOCR PP-Structure"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.lang = self.config.get('lang', 'ch')

        # 初始化PaddleOCR用于文字识别 (禁用所有预处理，限制尺寸以加速)
        self.ocr = PaddleOCR(
            lang=self.lang,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_type='max',
            text_det_limit_side_len=960,
        )

        logger.info("自动表格检测器初始化完成")

    def detect_table_structure(self, image_path: str) -> Dict:
        """
        自动检测表格结构

        Returns:
            {
                'cell_bboxes': [[x1, y1, x2, y2], ...],  # 所有单元格边界框
                'table_html': '...',  # 表格HTML结构
                'rec_res': [...]      # OCR识别结果
            }
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 使用PaddleOCR 3.x的表格识别功能
        result = self.ocr.predict(img)

        if not result or not result[0]:
            logger.warning("未检测到表格内容")
            return {'cell_bboxes': [], 'table_html': '', 'rec_res': []}

        # 解析结果 - PaddleOCR 3.x返回格式
        cell_bboxes = []
        rec_res = []

        ocr_data = result[0]
        rec_texts = ocr_data.get('rec_texts', [])
        rec_scores = ocr_data.get('rec_scores', [])
        rec_boxes = ocr_data.get('rec_boxes', [])  # [x1, y1, x2, y2] 格式

        for idx in range(len(rec_texts)):
            text = rec_texts[idx]
            score = rec_scores[idx] if idx < len(rec_scores) else 0.0

            # 获取边界框
            if idx < len(rec_boxes):
                box = rec_boxes[idx].tolist()
            else:
                box = [0, 0, 0, 0]

            cell_bboxes.append(box)
            rec_res.append({
                'text': text,
                'confidence': score
            })

        logger.info(f"检测到 {len(cell_bboxes)} 个单元格")

        return {
            'cell_bboxes': cell_bboxes,
            'rec_res': rec_res,
            'image_shape': img.shape
        }

    def analyze_tooth_cells(self, table_result: Dict) -> Dict[str, Dict]:
        """
        分析牙周图表的32颗牙齿数据

        Args:
            table_result: detect_table_structure的返回结果

        Returns:
            32颗牙齿的数据字典
        """
        cell_bboxes = table_result.get('cell_bboxes', [])
        rec_res = table_result.get('rec_res', [])

        # 按位置分组单元格到牙齿
        # 牙周图表结构:
        # - 4个象限，每象限8颗牙齿
        # - 每颗牙齿6个测量面
        # - 每个测量面可能有PD值(数字)和BOP/PI圈选

        # 按Y坐标分组行
        rows = self._group_cells_by_row(cell_bboxes)

        # 解析每颗牙齿
        teeth_data = {}

        # 计算象限
        img_h = table_result.get('image_shape', [1000, 800, 3])[0]
        img_w = table_result.get('image_shape', [1000, 800, 3])[1]

        mid_y = img_h // 2
        mid_x = img_w // 2

        # 处理每个区域
        for row_idx, row_cells in enumerate(rows):
            if len(row_cells) < 6:
                continue

            # 按X坐标排序
            row_cells.sort(key=lambda x: x[0])

            # 确定是哪个象限的哪颗牙齿
            for col_idx, cell in enumerate(row_cells):
                if col_idx >= 6:  # 最多6个测量面
                    break

                cell_y = (cell[1] + cell[3]) // 2
                cell_x = (cell[0] + cell[2]) // 2

                # 确定象限
                if cell_y < mid_y:
                    if cell_x > mid_x:
                        quadrant = 'upper_right'
                    else:
                        quadrant = 'upper_left'
                else:
                    if cell_x > mid_x:
                        quadrant = 'lower_right'
                    else:
                        quadrant = 'lower_left'

                # 计算牙齿编号
                tooth_num = self._get_tooth_number(quadrant, row_idx, col_idx)
                if not tooth_num:
                    continue

                # 获取OCR结果
                cell_text = ''
                if col_idx < len(rec_res):
                    cell_text = rec_res[col_idx].get('text', '')

                # 解析数值
                pd_value = self._extract_number(cell_text)

                # 存储数据
                if tooth_num not in teeth_data:
                    teeth_data[tooth_num] = {
                        'tooth': 1,
                        'mobility': 0,
                        'implant': 0,
                        'pd': {},
                        'gm': {},
                        'bop': {},
                        'pi': {},
                        'furcation': {}
                    }

                surface = self._get_surface(col_idx, quadrant)
                teeth_data[tooth_num]['pd'][surface] = pd_value

        return teeth_data

    def _group_cells_by_row(self, cell_bboxes: List[List[int]],
                            row_threshold: int = 30) -> List[List]:
        """将单元格按行分组"""
        if not cell_bboxes:
            return []

        # 计算每行的中心Y坐标
        cell_rows = [(cell[1] + cell[3]) // 2 for cell in cell_bboxes]

        # 排序
        sorted_indices = np.argsort(cell_rows)
        sorted_cells = [cell_bboxes[i] for i in sorted_indices]

        rows = []
        current_row = [sorted_cells[0]]

        for i in range(1, len(sorted_cells)):
            curr_y = (sorted_cells[i][1] + sorted_cells[i][3]) // 2
            prev_y = (current_row[-1][1] + current_row[-1][3]) // 2

            if abs(curr_y - prev_y) < row_threshold:
                current_row.append(sorted_cells[i])
            else:
                rows.append(current_row)
                current_row = [sorted_cells[i]]

        if current_row:
            rows.append(current_row)

        return rows

    def _get_tooth_number(self, quadrant: str, row_idx: int,
                          col_idx: int) -> Optional[str]:
        """根据位置计算牙齿编号"""
        quadrant_teeth = {
            'upper_right': ['18', '17', '16', '15', '14', '13', '12', '11'],
            'upper_left': ['21', '22', '23', '24', '25', '26', '27', '28'],
            'lower_left': ['31', '32', '33', '34', '35', '36', '37', '38'],
            'lower_right': ['41', '42', '43', '44', '45', '46', '47', '48']
        }

        teeth = quadrant_teeth.get(quadrant, [])
        if row_idx < len(teeth):
            return teeth[row_idx]
        return None

    def _get_surface(self, col_idx: int, quadrant: str) -> str:
        """根据列索引获取测量面"""
        upper_surfaces = ['db', 'b', 'mb', 'dl', 'l', 'ml']
        lower_surfaces = ['db', 'b', 'mb', 'dp', 'p', 'mp']

        surfaces = upper_surfaces if 'upper' in quadrant else lower_surfaces
        if col_idx < len(surfaces):
            return surfaces[col_idx]
        return 'db'

    def _extract_number(self, text: str) -> int:
        """从文本中提取数字"""
        if not text:
            return 0

        digits = ''.join(c for c in text if c.isdigit())
        if digits:
            try:
                return int(digits[:2])
            except ValueError:
                return 0
        return 0

    def process_image(self, image_path: str) -> Dict:
        """处理图像并返回牙周数据"""
        # 1. 检测表格结构
        table_result = self.detect_table_structure(image_path)

        # 2. 分析牙齿数据
        teeth_data = self.analyze_tooth_cells(table_result)

        # 3. 转换为标准格式
        return self._map_to_standard_format(teeth_data)

    def _map_to_standard_format(self, teeth_data: Dict) -> Dict:
        """映射到标准牙周数据结构"""
        from .data_mapper import DataMapper

        mapper = DataMapper()
        return mapper.map_ocr_to_structure(teeth_data)


class SimpleTableOCR:
    """简化版表格OCR - 直接使用PaddleOCR 3.x识别"""

    def __init__(self):
        # 禁用MKL-DNN和所有预处理功能，限制输入尺寸加速识别
        self.ocr = PaddleOCR(
            lang='ch',
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_type='max',       # 限制最大边
            text_det_limit_side_len=960,     # 最大边长960px
        )

    def process(self, image_path: str) -> Dict:
        """
        直接OCR识别图像中的表格内容

        Args:
            image_path: 图像路径

        Returns:
            OCR结果
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # PaddleOCR 3.x 使用 predict 方法
        # 返回格式: [{'rec_texts': [...], 'rec_scores': [...], 'rec_boxes': [...], 'dt_polys': [...]}]
        result = self.ocr.predict(img)

        # 解析结果 - PaddleOCR 3.x 返回格式
        parsed = []

        if result and len(result) > 0:
            ocr_data = result[0]

            # 获取识别结果
            rec_texts = ocr_data.get('rec_texts', [])
            rec_scores = ocr_data.get('rec_scores', [])
            rec_boxes = ocr_data.get('rec_boxes', [])  # [x1, y1, x2, y2] 格式

            for idx in range(len(rec_texts)):
                text = rec_texts[idx]
                score = rec_scores[idx] if idx < len(rec_scores) else 0.0

                # 获取边界框
                if idx < len(rec_boxes):
                    box = rec_boxes[idx].tolist()  # [x1, y1, x2, y2]
                else:
                    box = [0, 0, 0, 0]

                parsed.append({
                    'bbox': box,
                    'text': text,
                    'score': score
                })

        return {
            'image_shape': img.shape,
            'results': parsed,
            'total_items': len(parsed)
        }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python auto_table_detector.py <图像路径>")
        sys.exit(1)

    # 测试简化版
    ocr = SimpleTableOCR()
    result = ocr.process(sys.argv[1])

    print(f"检测到 {result['total_items']} 个文本区域")
    for item in result['results'][:10]:  # 显示前10个
        print(f"  文本: {item['text']}, 置信度: {item['score']:.2f}")
