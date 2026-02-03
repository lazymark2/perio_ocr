"""
PaddleX 表格识别模块

基于 PaddleOCR 的 TableRecognitionPipelineV2 实现牙周图表的表格识别能力。
支持有线表格和无线表格的识别、表格结构解析和单元格内容提取。

参考文档: docs/PaddleX_概述.md
PaddleOCR 源码: C:\Users\lazymark2\miniforge3\envs\perio_ocr\Lib\site-packages\paddleocr\
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class PaddleXTableOCR:
    """
    PaddleX 表格识别类

    使用 PaddleOCR 的 TableRecognitionPipelineV2 实现表格识别，
    支持:
    - 表格结构识别 (有线/无线表格)
    - 单元格检测
    - 单元格内容 OCR
    - 表格数据转换为结构化格式
    """

    def __init__(
        self,
        lang: str = 'ch',
        use_doc_orientation_classify: bool = True,
        use_doc_unwarping: bool = False,
        use_layout_detection: bool = True,
        text_det_limit_side_len: int = 960,
    ):
        """
        初始化 PaddleX 表格识别器

        Args:
            lang: 语言 ('ch'=中文, 'en'=英文)
            use_doc_orientation_classify: 启用文档方向分类
            use_doc_unwarping: 启用图像矫正
            use_layout_detection: 启用版面检测
            text_det_limit_side_len: 文本检测图像尺寸限制
        """
        self.lang = lang
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_layout_detection = use_layout_detection
        self.text_det_limit_side_len = text_det_limit_side_len

        # 延迟加载 TableRecognitionPipelineV2
        self._table_pipeline = None

    def _init_pipeline(self):
        """延迟初始化表格识别 pipeline"""
        if self._table_pipeline is not None:
            return

        try:
            from paddleocr import TableRecognitionPipelineV2
            self._table_pipeline = TableRecognitionPipelineV2(
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_doc_unwarping=self.use_doc_unwarping,
                use_layout_detection=self.use_layout_detection,
                text_det_limit_side_len=self.text_det_limit_side_len,
                lang=self.lang,
            )
            logger.info("PaddleX TableRecognitionPipelineV2 初始化成功")
        except ImportError as e:
            logger.error(f"无法导入 TableRecognitionPipelineV2: {e}")
            raise ImportError(
                "请确保安装了最新版本的 PaddleOCR: pip install --upgrade paddleocr"
            )
        except Exception as e:
            logger.error(f"初始化 TableRecognitionPipelineV2 失败: {e}")
            raise

    def process(self, image_path: str) -> Dict[str, Any]:
        """
        处理图像，返回表格结构数据

        Args:
            image_path: 图像文件路径

        Returns:
            {
                'tables': [
                    {
                        'type': 'wired' | 'wireless',  # 表格类型
                        'bbox': [x1, y1, x2, y2],      # 表格边界框
                        'cells': [                      # 单元格列表
                            {
                                'bbox': [x1, y1, x2, y2],
                                'row': int,
                                'col': int,
                                'rowspan': int,
                                'colspan': int,
                                'text': str,
                                'confidence': float
                            },
                            ...
                        ],
                        'html': str,                    # HTML 格式表示
                        'structure': {...}              # 原始结构数据
                    }
                ],
                'layout': [...],                        # 版面检测结果
                'orientation': str,                     # 文档方向
                'raw_result': {...}                     # 原始识别结果
            }
        """
        self._init_pipeline()

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        logger.info(f"开始表格识别: {image_path}")

        try:
            # 调用表格识别 pipeline
            result = self._table_pipeline.predict(
                image_path,
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_doc_unwarping=self.use_doc_unwarping,
                use_layout_detection=self.use_layout_detection,
            )

            if not result or len(result) == 0:
                logger.warning("表格识别未返回结果")
                return self._empty_result(img.shape[:2])

            # 解析结果
            parsed = self._parse_result(result[0], img.shape[:2])
            logger.info(f"表格识别完成，检测到 {len(parsed['tables'])} 个表格")

            return parsed

        except Exception as e:
            logger.error(f"表格识别失败: {e}")
            raise

    def _parse_result(self, raw_result: Dict, img_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        解析表格识别结果

        PaddleX TableRecognitionPipelineV2 返回格式可能包含:
        - layout_boxes: 版面检测结果
        - tables: 表格识别结果
        - ocr_results: OCR 识别结果
        """
        parsed = {
            'tables': [],
            'layout': [],
            'orientation': None,
            'raw_result': raw_result
        }

        # 提取文档方向
        if 'doc_orientation_classify' in raw_result:
            orientation = raw_result['doc_orientation_classify']
            parsed['orientation'] = self._parse_orientation(orientation)

        # 提取版面检测结果
        if 'layout_boxes' in raw_result:
            parsed['layout'] = raw_result['layout_boxes']

        # 提取表格数据
        tables = raw_result.get('tables', [])
        for table in tables:
            parsed_table = self._parse_table(table, img_shape)
            if parsed_table:
                parsed['tables'].append(parsed_table)

        return parsed

    def _parse_table(self, table_data: Dict, img_shape: Tuple[int, int]) -> Optional[Dict]:
        """
        解析单个表格数据

        返回格式可能包含:
        - bbox: 表格边界框
        - type: 表格类型 (wired/wireless)
        - cells: 单元格列表
        - html: HTML 表示
        """
        # 获取表格边界框
        bbox = table_data.get('bbox')
        if bbox is None:
            return None

        # 确定表格类型
        table_type = table_data.get('type', 'wired')

        # 解析单元格
        cells = []
        raw_cells = table_data.get('cells', [])

        for cell in raw_cells:
            parsed_cell = self._parse_cell(cell)
            if parsed_cell:
                cells.append(parsed_cell)

        # 生成 HTML 表示
        html = table_data.get('html', self._generate_html(cells, table_type))

        return {
            'type': table_type,
            'bbox': bbox,
            'cells': cells,
            'html': html,
            'structure': table_data
        }

    def _parse_cell(self, cell_data: Dict) -> Optional[Dict]:
        """
        解询单元格数据

        单元格数据可能包含:
        - bbox: 边界框
        - row, col: 行列位置
        - rowspan, colspan: 合并信息
        - text: 单元格内容
        - confidence: 置信度
        """
        bbox = cell_data.get('bbox')
        if bbox is None:
            return None

        # 提取文本内容
        text = cell_data.get('text', '')
        if isinstance(text, list):
            # 处理 OCR 结果格式: [{'text': ..., 'confidence': ...}, ...]
            if text:
                text = text[0].get('text', '')
            else:
                text = ''

        return {
            'bbox': bbox,
            'row': cell_data.get('row', 0),
            'col': cell_data.get('col', 0),
            'rowspan': cell_data.get('rowspan', 1),
            'colspan': cell_data.get('colspan', 1),
            'text': text,
            'confidence': cell_data.get('confidence', 0.0)
        }

    def _parse_orientation(self, orientation_data: Any) -> str:
        """解析文档方向"""
        if isinstance(orientation_data, dict):
            return orientation_data.get('label', '0')
        elif isinstance(orientation_data, (int, str)):
            return str(orientation_data)
        return '0'

    def _generate_html(self, cells: List[Dict], table_type: str) -> str:
        """从单元格列表生成 HTML 表格"""
        if not cells:
            return ''

        # 计算表格行列数
        max_row = max(c['row'] + c['rowspan'] for c in cells) if cells else 0
        max_col = max(c['col'] + c['colspan'] for c in cells) if cells else 0

        # 创建单元格位置映射
        cell_grid = [[None for _ in range(max_col)] for _ in range(max_row)]
        for cell in cells:
            for r in range(cell['row'], cell['row'] + cell['rowspan']):
                for c in range(cell['col'], cell['col'] + cell['colspan']):
                    if 0 <= r < max_row and 0 <= c < max_col:
                        cell_grid[r][c] = cell

        # 生成 HTML
        html = '<table>\n'
        for r in range(max_row):
            html += '  <tr>\n'
            for c in range(max_col):
                cell = cell_grid[r][c]
                if cell and cell['row'] == r and cell['col'] == c:
                    # 合并单元格只输出一次
                    attrs = []
                    if cell['rowspan'] > 1:
                        attrs.append(f'rowspan="{cell["rowspan"]}"')
                    if cell['colspan'] > 1:
                        attrs.append(f'colspan="{cell["colspan"]}"')
                    attr_str = ' ' + ' '.join(attrs) if attrs else ''
                    html += f'    <td{attr_str}>{cell["text"]}</td>\n'
            html += '  </tr>\n'
        html += '</table>'

        return html

    def _empty_result(self, img_shape: Tuple[int, int]) -> Dict:
        """返回空结果"""
        return {
            'tables': [],
            'layout': [],
            'orientation': '0',
            'raw_result': {}
        }

    def extract_periodontal_data(self, image_path: str) -> Dict[str, Any]:
        """
        从牙周图表中提取结构化数据

        这是一个专门针对牙周图表的数据提取方法，
        结合表格识别结果和牙周图表的特定结构。

        Returns:
            {
                'teeth_data': {...},
                'table_structure': {...},
                'raw_ocr': {...}
            }
        """
        # 首先进行表格识别
        table_result = self.process(image_path)

        # 解析牙周图表特定结构
        teeth_data = self._extract_periodontal_from_table(table_result)

        return {
            'teeth_data': teeth_data,
            'table_structure': table_result,
            'raw_ocr': table_result.get('raw_result', {})
        }

    def _extract_periodontal_from_table(self, table_result: Dict) -> Dict:
        """
        从表格识别结果中提取牙周数据

        牙周图表结构分析:
        - 通常包含上下两部分 (上颌/下颌)
        - 每部分包含: PI, Mobility, Furcation, BOP, PD
        - 每颗牙齿有多个测量面 (B/L)
        """
        teeth_data = {}

        for table in table_result.get('tables', []):
            for cell in table.get('cells', []):
                text = cell.get('text', '').strip()
                if not text:
                    continue

                # 分析单元格位置和内容
                row = cell.get('row', 0)
                col = cell.get('col', 0)
                bbox = cell.get('bbox', [])

                # 根据位置和内容识别数据类型
                # 这里需要根据实际表格结构来实现
                # ...

        return teeth_data


class PaddleXTableOCRWithRegions(PaddleXTableOCR):
    """
    带区域检测的 PaddleX 表格识别

    针对牙周图表的上下半部分结构进行优化，
    自动检测并分别处理上颌和下颌区域。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.separator_y = None

    def detect_upper_lower_regions(self, image_path: str) -> Tuple[int, int, int]:
        """
        检测上下半部分的分隔线

        Returns:
            (height, width, separator_y)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        h, w = img.shape[:2]

        # 尝试 OCR 识别找到 "牙位" 关键字
        self._init_pipeline()

        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang='ch')
            result = ocr.predict(image_path)

            if result and len(result) > 0:
                ocr_data = result[0]
                texts = ocr_data.get('rec_texts', [])
                boxes = ocr_data.get('rec_boxes', [])

                for idx, text in enumerate(texts):
                    if '牙位' in text:
                        box = boxes[idx].tolist() if idx < len(boxes) else [0, 0, 0, 0]
                        self.separator_y = int((box[1] + box[3]) // 2)
                        return h, w, self.separator_y

        except Exception as e:
            logger.warning(f"OCR 检测分隔线失败: {e}")

        # 默认使用图像中点
        self.separator_y = h // 2
        return h, w, self.separator_y

    def process_upper_lower(self, image_path: str) -> Dict[str, Any]:
        """
        分别处理上下半部分

        Returns:
            {
                'upper': {...},  # 上半部分结果
                'lower': {...},  # 下半部分结果
                'separator_y': int
            }
        """
        h, w, separator_y = self.detect_upper_lower_regions(image_path)

        # 读取并分割图像
        img = cv2.imread(image_path)
        upper_img = img[:separator_y, :]
        lower_img = img[separator_y:, :]

        # 保存临时图像
        import tempfile
        import os

        temp_dir = tempfile.gettempdir()
        upper_path = os.path.join(temp_dir, 'perio_upper.jpg')
        lower_path = os.path.join(temp_dir, 'perio_lower.jpg')

        cv2.imwrite(upper_path, upper_img)
        cv2.imwrite(lower_path, lower_img)

        # 分别处理
        upper_result = self.process(upper_path)
        lower_result = self.process(lower_path)

        # 清理临时文件
        try:
            os.remove(upper_path)
            os.remove(lower_path)
        except:
            pass

        return {
            'upper': upper_result,
            'lower': lower_result,
            'separator_y': separator_y
        }


# 便捷函数
def create_table_ocr(lang: str = 'ch', **kwargs) -> PaddleXTableOCR:
    """创建表格识别器的便捷函数"""
    return PaddleXTableOCR(lang=lang, **kwargs)


def create_region_table_ocr(lang: str = 'ch', **kwargs) -> PaddleXTableOCRWithRegions:
    """创建带区域检测的表格识别器的便捷函数"""
    return PaddleXTableOCRWithRegions(lang=lang, **kwargs)


if __name__ == '__main__':
    import sys
    import json

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("用法:")
        print("  python paddlex_ocr.py <图像路径>              # 基本表格识别")
        print("  python paddlex_ocr.py <图像路径> --regions    # 带区域检测的识别")
        sys.exit(1)

    image_path = sys.argv[1]
    use_regions = '--regions' in sys.argv

    try:
        if use_regions:
            ocr = create_region_table_ocr()
            result = ocr.process_upper_lower(image_path)

            print("=== 上半部分表格 ===")
            print(f"检测到 {len(result['upper']['tables'])} 个表格")
            for table in result['upper']['tables']:
                print(f"  表格类型: {table['type']}")
                print(f"  单元格数: {len(table['cells'])}")

            print("\n=== 下半部分表格 ===")
            print(f"检测到 {len(result['lower']['tables'])} 个表格")
            for table in result['lower']['tables']:
                print(f"  表格类型: {table['type']}")
                print(f"  单元格数: {len(table['cells'])}")

            print(f"\n分隔线 Y 坐标: {result['separator_y']}")

        else:
            ocr = create_table_ocr()
            result = ocr.process(image_path)

            print(f"=== 表格识别结果 ===")
            print(f"检测到 {len(result['tables'])} 个表格")
            print(f"文档方向: {result['orientation']}")

            for idx, table in enumerate(result['tables']):
                print(f"\n表格 {idx + 1}:")
                print(f"  类型: {table['type']}")
                print(f"  边界框: {table['bbox']}")
                print(f"  单元格数: {len(table['cells'])}")

                # 显示前 5 个单元格
                for cell in table['cells'][:5]:
                    print(f"    [{cell['row']},{cell['col']}]: {cell['text']}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
