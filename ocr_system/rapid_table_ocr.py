"""
牙周图表表格识别模块 - 基于 TableStructureRec 和 RapidOCR

使用 ONNXRuntime 推理引擎，提供轻量级、高精度的表格识别方案。
支持有线表格识别，适合牙周图表结构化数据提取。

特点:
- 使用 ONNXRuntime，无需 PaddlePaddle 框架
- 集成 RapidOCR，更轻量
- 支持 table_cls 表格分类
- 自动识别表格单元格并映射到牙周数据结构

安装依赖:
    pip install wired_table_rec lineless_table_rec table_cls rapidocr

参考: https://github.com/RapidAI/TableStructureRec
"""
import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class RapidTableOCR:
    """
    基于 RapidAI TableStructureRec 的牙周图表 OCR 系统

    使用有线表格识别模型 + RapidOCR，提供轻量级高精度表格识别
    """

    # 牙齿编号常量
    UPPER_TEETH_RIGHT = ['18', '17', '16', '15', '14', '13', '12', '11']
    UPPER_TEETH_LEFT = ['21', '22', '23', '24', '25', '26', '27', '28']
    LOWER_TEETH_LEFT = ['31', '32', '33', '34', '35', '36', '37', '38']
    LOWER_TEETH_RIGHT = ['48', '47', '46', '45', '44', '43', '42', '41']

    # 测量面
    SURFACES = ['db', 'b', 'mb', 'dp', 'p', 'mp']

    # 测量类型
    MEASUREMENT_TYPES = ['PD', 'GM', 'BOP', 'PI', 'Mobility', 'Furcation']

    def __init__(
        self,
        use_table_cls: bool = True,
        use_cuda: bool = False,
        device: str = "cpu"
    ):
        """
        初始化 RapidTableOCR 系统

        Args:
            use_table_cls: 是否使用表格分类（区分有线/无线表格）
            use_cuda: 是否使用 CUDA 加速
            device: 设备类型
        """
        self.use_table_cls = use_table_cls
        self.use_cuda = use_cuda
        self.device = device
        self.table_cls = None  # 初始化为 None

        # 初始化 OCR 引擎
        try:
            from rapidocr import RapidOCR
            self.ocr_engine = RapidOCR()
            logger.info("RapidOCR 初始化成功")
        except ImportError:
            logger.error("RapidOCR 未安装，请运行: pip install rapidocr")
            raise

        # 初始化表格识别引擎
        self._init_table_engines()

        # 初始化表格分类器
        if self.use_table_cls:
            self._init_table_cls()

    def _init_table_engines(self):
        """初始化有线和无线表格识别引擎"""
        try:
            from wired_table_rec.main import WiredTableRecognition, WiredTableInput
            from lineless_table_rec.main import LinelessTableRecognition, LinelessTableInput

            # 有线表格识别
            wired_input = WiredTableInput(
                model_type="unet",  # 或 "cycle_center_net"
                use_cuda=self.use_cuda,
                device=self.device
            )
            self.wired_engine = WiredTableRecognition(wired_input)

            # 无线表格识别
            lineless_input = LinelessTableInput(
                model_type="lore",
                use_cuda=self.use_cuda,
                device=self.device
            )
            self.lineless_engine = LinelessTableRecognition(lineless_input)

            logger.info("表格识别引擎初始化成功")
        except ImportError as e:
            logger.error(f"表格识别模块未安装: {e}")
            logger.info("请运行: pip install wired_table_rec lineless_table_rec table_cls")
            raise

    def _init_table_cls(self):
        """初始化表格分类器"""
        try:
            from table_cls import TableCls
            self.table_cls = TableCls()
            logger.info("表格分类器初始化成功")
        except ImportError:
            logger.warning("表格分类器未安装，将默认使用有线表格识别")
            self.table_cls = None

    def classify_table(self, image_path: str) -> str:
        """
        分类表格类型

        Returns:
            "wired" - 有线表格
            "lineless" - 无线表格
        """
        if self.table_cls is None:
            return "wired"

        try:
            cls_result, elasp = self.table_cls(image_path)
            return cls_result if cls_result in ["wired", "lineless"] else "wired"
        except Exception as e:
            logger.warning(f"表格分类失败，使用默认有线表格: {e}")
            return "wired"

    def recognize_table(
        self,
        image_path: str,
        table_type: Optional[str] = None,
        return_html: bool = True,
        return_cells: bool = True,
        **kwargs
    ) -> Dict:
        """
        识别表格结构

        Args:
            image_path: 图像路径
            table_type: 表格类型 ("wired" 或 "lineless")，None 则自动分类
            return_html: 是否返回 HTML 格式
            return_cells: 是否返回单元格边界框
            **kwargs: 其他参数
                - col_threshold: 列阈值（默认15）
                - row_threshold: 行阈值（默认10）
                - rotated_fix: 是否旋转矫正（默认True）

        Returns:
            {
                'table_type': str,
                'html': str,
                'cells': List[Dict],
                'logic_points': np.ndarray,
                'elapse': float
            }
        """
        # 自动分类表格类型
        if table_type is None:
            table_type = self.classify_table(image_path)

        logger.info(f"使用 {table_type} 表格识别")

        # 获取 OCR 结果
        rapid_ocr_output = self.ocr_engine(image_path, return_word_box=True)
        ocr_result = list(
            zip(rapid_ocr_output.boxes, rapid_ocr_output.txts, rapid_ocr_output.scores)
        )

        # 选择表格识别引擎
        if table_type == "wired":
            engine = self.wired_engine
        else:
            engine = self.lineless_engine

        # 表格识别
        table_results = engine(
            image_path,
            ocr_result=ocr_result,
            **kwargs
        )

        return {
            'table_type': table_type,
            'html': table_results.pred_html if return_html else None,
            'cells': self._parse_cells(table_results) if return_cells else None,
            'logic_points': table_results.logic_points,
            'elapse': table_results.elapse
        }

    def _parse_cells(self, table_results) -> List[Dict]:
        """解析单元格信息"""
        cells = []
        if table_results.cell_bboxes is None:
            return cells

        cell_bboxes = table_results.cell_bboxes
        for i, bbox in enumerate(cell_bboxes):
            cells.append({
                'index': i,
                'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox),
                'x1': int(bbox[0]),
                'y1': int(bbox[1]),
                'x2': int(bbox[2]),
                'y2': int(bbox[3])
            })
        return cells

    def parse_html_to_periodontal_data(self, html: str) -> Dict:
        """
        解析表格 HTML 为牙周图表数据结构

        Args:
            html: 表格 HTML 字符串

        Returns:
            牙周数据字典
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')

        if table is None:
            return {}

        result = {}
        rows = table.find_all('tr')

        # 解析表头（牙齿编号）
        header_row = rows[0] if rows else None
        if header_row:
            tooth_numbers = self._parse_header_cells(header_row)
        else:
            tooth_numbers = []

        # 解析数据行
        for row_idx, row in enumerate(rows[1:], start=1):
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue

            # 识别测量类型（第一个单元格）
            measurement_type = self._identify_measurement_type(cells[0].get_text())

            # 解析数据单元格
            for col_idx, cell in enumerate(cells[1:]):
                if col_idx < len(tooth_numbers):
                    tooth_num = tooth_numbers[col_idx]
                    value = self._parse_cell_value(cell.get_text(), measurement_type)
                    surface = self.SURFACES[col_idx % len(self.SURFACES)]

                    field_name = self._get_field_name(tooth_num, measurement_type, surface)
                    if field_name:
                        result[field_name] = value

        return result

    def _parse_header_cells(self, header_row) -> List[str]:
        """解析表头单元格，提取牙齿编号"""
        tooth_numbers = []
        cells = header_row.find_all(['td', 'th'])

        for cell in cells:
            text = cell.get_text().strip()
            # 提取数字
            digits = ''.join(c for c in text if c.isdigit())
            if digits:
                tooth_numbers.append(digits)

        return tooth_numbers

    def _identify_measurement_type(self, text: str) -> str:
        """识别测量类型"""
        text_upper = text.upper()
        for mtype in self.MEASUREMENT_TYPES:
            if mtype.upper() in text_upper:
                return mtype
        return 'PD'  # 默认为 PD

    def _parse_cell_value(self, text: str, measurement_type: str):
        """解析单元格值"""
        text = text.strip()

        if not text:
            return 0

        if measurement_type in ['BOP', 'PI']:
            # 出血/菌斑：检查是否有标记
            positive_marks = ['1', '+', '✓', '√', 'X', '×', '●', 'O']
            return 1 if any(mark in text for mark in positive_marks) else 0

        elif measurement_type in ['PD', 'GM']:
            # 探诊深度/龈缘：提取数字
            # 处理小数格式：3:5 -> 3.5
            text = text.replace(':', '.').replace('：', '.')
            try:
                return float(text) if '.' in text else int(text)
            except ValueError:
                # 提取所有数字
                digits = ''.join(c for c in text if c.isdigit() or c == '.')
                try:
                    return float(digits) if digits else 0
                except ValueError:
                    return 0

        elif measurement_type == 'Mobility':
            # 松动度：罗马数字或数字
            roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
            if text.upper() in roman_map:
                return roman_map[text.upper()]
            try:
                return int(text)
            except ValueError:
                return 0

        elif measurement_type == 'Furcation':
            # 分叉病变：数字或罗马数字
            roman_map = {'I': 1, 'II': 2, 'III': 3}
            if text.upper() in roman_map:
                return roman_map[text.upper()]
            try:
                return int(text)
            except ValueError:
                return 0

        return text

    def _get_field_name(self, tooth_num: str, measurement_type: str, surface: str) -> Optional[str]:
        """生成字段名称"""
        if not tooth_num:
            return None

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

    def process_periodontal_chart(
        self,
        image_path: str,
        table_type: Optional[str] = None,
        output_html: bool = False
    ) -> Dict:
        """
        完整处理牙周图表

        Args:
            image_path: 图像路径
            table_type: 表格类型（None 则自动检测）
            output_html: 是否输出 HTML

        Returns:
            牙周数据字典
        """
        from datetime import datetime

        # 表格识别
        table_result = self.recognize_table(
            image_path,
            table_type=table_type,
            return_html=True
        )

        # 解析 HTML 为牙周数据
        perio_data = self.parse_html_to_periodontal_data(table_result['html'])

        # 添加元数据
        perio_data.update({
            'date_saved': datetime.now().strftime('%Y-%m-%d'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'processing_method': 'rapid_table_ocr',
            'table_type': table_result['table_type'],
            'recognition_time': table_result['elapse']
        })

        # 保存 HTML
        if output_html:
            html_path = Path(image_path).stem + '_table.html'
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(table_result['html'])
            logger.info(f"HTML 已保存: {html_path}")

        return perio_data

    def export_to_json(self, data: Dict, output_path: str):
        """导出数据到 JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"数据已保存到: {output_path}")

    def visualize_table(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        可视化表格识别结果

        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径（可选）

        Returns:
            可视化图像
        """
        try:
            from wired_table_rec.utils.utils import VisTable
        except ImportError:
            logger.error("可视化工具未找到")
            return None

        # 获取表格识别结果
        table_result = self.recognize_table(image_path)

        # 创建输出对象
        class TableResult:
            def __init__(self, html, cell_bboxes):
                self.pred_html = html
                self.cell_bboxes = cell_bboxes

        result_obj = TableResult(table_result['html'], table_result.get('cells'))

        # 可视化
        viser = VisTable()
        vis_img = viser(
            image_path,
            result_obj,
            None,  # save_html_path
            output_path,  # save_drawed_path
            None  # save_logic_path
        )

        if output_path:
            logger.info(f"可视化结果已保存: {output_path}")

        return vis_img


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试代码
    ocr = RapidTableOCR(use_table_cls=True)

    # 测试牙周图表处理
    print("=== 测试牙周图表处理 ===")
    try:
        result = ocr.process_periodontal_chart("handwrite.jpg", output_html=True)
        print(f"处理完成，识别到 {len(result)} 个字段")
        print(f"表格类型: {result.get('table_type')}")
        print(f"处理时间: {result.get('recognition_time', 0):.2f}s")

        # 导出 JSON
        ocr.export_to_json(result, "perio_rapid_result.json")
        print("结果已保存到 perio_rapid_result.json")

        # 可视化
        ocr.visualize_table("handwrite.jpg", "handwrite_rapid_table.jpg")
        print("可视化结果已保存")

    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
