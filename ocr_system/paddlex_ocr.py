"""
PaddleX OCR 模块 - 基于 PaddleOCR 的表格识别能力

支持 PP-Structure 表格识别、单元格识别和牙周图表数据映射
"""
import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Optional
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class PaddleXOCR:
    """基于 PaddleOCR 的表格识别 OCR 系统"""

    # 牙齿编号常量
    UPPER_TEETH = ['18', '17', '16', '15', '14', '13', '12', '11',
                   '28', '27', '26', '25', '24', '23', '22', '21']
    LOWER_TEETH = ['48', '47', '46', '45', '44', '43', '42', '41',
                   '38', '37', '36', '35', '34', '33', '32', '31']

    # 测量类型
    MEASUREMENT_TYPES = ['PD', 'GM', 'BOP', 'PI', 'Mobility', 'Furcation']

    def __init__(self, use_table: bool = True, lang: str = 'ch', use_angle_cls: bool = True):
        """
        初始化 PaddleX OCR 系统

        Args:
            use_table: 是否启用表格识别模式
            lang: 语言 ('ch' 中文, 'en' 英文)
            use_angle_cls: 是否使用方向分类器
        """
        self.use_table = use_table
        self.lang = lang

        # 初始化 PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
        )

        # 初始化表格识别引擎（PP-Structure）
        if self.use_table:
            try:
                from paddleocr import PPStructure
                self.table_engine = PPStructure(
                    lang=lang,
                    table=True,
                    ocr=True,
                    layout=False,
                )
                logger.info("PaddleX表格识别引擎初始化完成")
            except (ImportError, Exception):
                logger.warning("PPStructure 不可用，将使用基础OCR模式")
                self.table_engine = None
        else:
            self.table_engine = None

        logger.info(f"PaddleXOCR初始化完成 (Table: {use_table}, Lang: {lang})")

    def detect_table_structure(self, image_path: str) -> Dict:
        """
        检测表格结构并提取单元格信息

        Returns:
            {
                'cells': [{'bbox': [x1,y1,x2,y2], 'row': r, 'col': c, 'text': str}, ...],
                'rows': int,
                'cols': int,
                'html': str,
                'structure': List
            }
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.detect_table_structure_from_array(image)

    def detect_table_structure_from_array(self, image: np.ndarray) -> Dict:
        """从numpy数组检测表格结构"""
        if self.table_engine is None:
            logger.warning("表格引擎未初始化，使用基础OCR")
            return self._fallback_table_detection(image)

        try:
            result = self.table_engine(image)
            return self._parse_table_result(result, image.shape)
        except Exception as e:
            logger.error(f"表格识别失败: {e}")
            return self._fallback_table_detection(image)

    def _parse_table_result(self, result: List, image_shape: tuple) -> Dict:
        """解析PP-Structure返回的表格结果"""
        cells = []
        rows = 0
        cols = 0
        html_output = ""

        for item in result:
            if item.get('type') == 'table':
                html_output = item.get('res', {}).get('html', '')
                table_cells = item.get('res', {}).get('cells', [])
                for cell_info in table_cells:
                    bbox = cell_info.get('bbox', [])
                    if bbox:
                        cells.append({
                            'bbox': [int(x) for x in bbox],
                            'text': cell_info.get('text', ''),
                            'row': cell_info.get('row', 0),
                            'col': cell_info.get('col', 0)
                        })
                rows = max(rows, max([c.get('row', 0) for c in cells]) + 1) if cells else 0
                cols = max(cols, max([c.get('col', 0) for c in cells]) + 1) if cells else 0

        return {
            'cells': cells,
            'rows': rows,
            'cols': cols,
            'html': html_output,
            'structure': result
        }

    def _fallback_table_detection(self, image: np.ndarray) -> Dict:
        """降级方案：使用基础OCR检测表格"""
        logger.info("使用基础OCR进行表格检测")
        result = self.ocr.ocr(image)

        cells = []
        if result and result[0]:
            for idx, line in enumerate(result[0]):
                bbox = line[0]
                text_info = line[1]
                text = text_info[0] if text_info else ""
                bbox_points = np.array(bbox).reshape(-1, 2)
                x1, y1 = bbox_points.min(axis=0)
                x2, y2 = bbox_points.max(axis=0)
                cells.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'text': text,
                    'confidence': float(text_info[1]) if text_info else 0.0,
                    'row': idx // 8,
                    'col': idx % 8
                })

        return {
            'cells': cells,
            'rows': (len(cells) // 8) + 1 if cells else 0,
            'cols': 8,
            'html': '',
            'structure': result
        }

    def recognize_cell_text(self, image: np.ndarray, cell_bbox: List) -> str:
        """识别单元格内的文本"""
        x1, y1, x2, y2 = cell_bbox
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        cell_img = image[y1:y2, x1:x2]
        if cell_img.size == 0:
            return ""

        processed = self._preprocess_cell(cell_img)
        try:
            result = self.ocr.ocr(processed)
            if result and result[0]:
                texts = [line[1][0] for line in result[0]]
                return ' '.join(texts).strip()
        except Exception as e:
            logger.error(f"单元格识别错误: {e}")
        return ""

    def _preprocess_cell(self, cell_img: np.ndarray) -> np.ndarray:
        """单元格图像预处理"""
        if len(cell_img.shape) == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        scale = max(1, 64 / enhanced.shape[0])
        new_h = int(enhanced.shape[0] * scale)
        new_w = int(enhanced.shape[1] * scale)
        resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized

    def map_to_periodontal_structure(self, table_data: Dict) -> Dict:
        """将表格数据映射到牙周图表数据结构"""
        result = {}
        cells = table_data.get('cells', [])

        rows_dict = {}
        for cell in cells:
            row = cell.get('row', 0)
            if row not in rows_dict:
                rows_dict[row] = []
            rows_dict[row].append(cell)

        first_row_cells = sorted(rows_dict.get(0, []), key=lambda x: x.get('col', 0))
        tooth_numbers = self._parse_tooth_numbers(first_row_cells)

        for row_idx in range(1, max(rows_dict.keys()) + 1):
            row_cells = rows_dict.get(row_idx, [])
            measurement_type = self._identify_measurement_type(row_cells)

            for col_idx, cell in enumerate(sorted(row_cells, key=lambda x: x.get('col', 0))):
                if col_idx < len(tooth_numbers):
                    tooth_num = tooth_numbers[col_idx]
                    value = self._parse_cell_value(cell.get('text', ''), measurement_type)
                    field_name = self._get_field_name(tooth_num, measurement_type, col_idx)
                    if field_name:
                        result[field_name] = value

        return result

    def _parse_tooth_numbers(self, first_row_cells: List[Dict]) -> List[str]:
        """解析第一行的牙齿编号"""
        tooth_numbers = []
        for cell in first_row_cells:
            text = cell.get('text', '').strip()
            digits = ''.join(c for c in text if c.isdigit())
            if digits:
                tooth_numbers.append(digits)
        return tooth_numbers

    def _identify_measurement_type(self, row_cells: List[Dict]) -> str:
        """识别行的测量类型"""
        if row_cells:
            first_cell_text = row_cells[0].get('text', '').upper()
            for mtype in self.MEASUREMENT_TYPES:
                if mtype.upper() in first_cell_text:
                    return mtype
        return 'PD'

    def _parse_cell_value(self, text: str, measurement_type: str):
        """解析单元格值"""
        text = text.strip()
        if measurement_type in ['BOP', 'PI']:
            positive_marks = ['1', 'O', '●', '✓', '√', 'X']
            return 1 if any(mark in text for mark in positive_marks) else 0
        elif measurement_type in ['PD', 'GM', 'Mobility', 'Furcation']:
            digits = ''.join(c for c in text if c.isdigit() or c == '-')
            try:
                return int(digits) if digits else 0
            except ValueError:
                return 0
        return text

    def _get_field_name(self, tooth_num: str, measurement_type: str, col_idx: int) -> Optional[str]:
        """生成字段名称"""
        if not tooth_num:
            return None
        surfaces = ['db', 'b', 'mb', 'dp', 'p', 'mp']
        surface_idx = col_idx % len(surfaces)
        surface = surfaces[surface_idx]

        if measurement_type == 'PD':
            return f'pd_{tooth_num}_{surface}'
        elif measurement_type == 'GM':
            return f'gm_{tooth_num}_{surface}'
        elif measurement_type == 'BOP':
            return f'bop_{tooth_num}_{surface}'
        elif measurement_type == 'PI':
            return f'pi_{tooth_num}_{surface}'
        elif measurement_type == 'Mobility':
            return f'mobility_{tooth_num}'
        elif measurement_type == 'Furcation':
            return f'furcation_{tooth_num}_{surface}'
        return None

    def process_periodontal_chart(self, image_path: str) -> Dict:
        """完整处理牙周图表"""
        table_structure = self.detect_table_structure(image_path)
        perio_data = self.map_to_periodontal_structure(table_structure)
        from datetime import datetime
        perio_data.update({
            'date_saved': datetime.now().strftime('%Y-%m-%d'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'processing_method': 'paddlex_table_ocr'
        })
        return perio_data

    def export_to_json(self, data: Dict, output_path: str):
        """导出数据到JSON文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"数据已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            raise

    def visualize_table_structure(self, image_path: str, output_path: str = None):
        """可视化表格结构（调试用）"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        table_structure = self.detect_table_structure_from_array(image)
        for cell in table_structure.get('cells', []):
            bbox = cell.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = cell.get('text', '')
                if text:
                    cv2.putText(image, text, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"可视化结果已保存到: {output_path}")
        else:
            cv2.imshow('Table Structure', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    ocr = PaddleXOCR(use_table=True, lang='ch')

    # 示例1: 表格结构检测
    print("=== 示例1: 表格结构检测 ===")
    try:
        table_structure = ocr.detect_table_structure("handwrite.jpg")
        print(f"检测到 {table_structure['rows']} 行 x {table_structure['cols']} 列")
        print(f"单元格数量: {len(table_structure['cells'])}")
    except Exception as e:
        print(f"表格检测失败: {e}")

    # 示例2: 牙周图表处理
    print("\n=== 示例2: 牙周图表处理 ===")
    try:
        perio_data = ocr.process_periodontal_chart("handwrite.jpg")
        ocr.export_to_json(perio_data, "perio_paddlex_result.json")
        print(f"处理完成，识别到 {len(perio_data)} 个字段")
    except Exception as e:
        print(f"处理失败: {e}")
