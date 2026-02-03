"""
牙周图表表格识别模块 - 基于 TableStructureRec 和 RapidOCR

使用 ONNXRuntime 推理引擎，提供轻量级、高精度的表格识别方案。
支持有线表格识别，适合牙周图表结构化数据提取。

特点:
- 使用 ONNXRuntime，无需 PaddlePaddle 框架
- 集成 RapidOCR，更轻量
- 支持 table_cls 表格分类
- 自动识别表格单元格并映射到牙周数据结构
- 延迟加载模型，支持快速模式
- 结果缓存机制

安装依赖:
    pip install wired_table_rec lineless_table_rec table_cls rapidocr

参考: https://github.com/RapidAI/TableStructureRec
"""
import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class RapidTableOCR:
    """
    基于 RapidAI TableStructureRec 的牙周图表 OCR 系统

    使用有线表格识别模型 + RapidOCR，提供轻量级高精度表格识别

    优化特性:
    - 延迟加载: 模型按需加载，减少启动时间
    - 快速模式: 跳过表格分类，直接使用有线表格识别
    - 结果缓存: 避免重复识别相同图像
    - 简化 API: 一键处理牙周图表
    """

    # 牙齿编号常量
    ALL_TEETH = (
        ['18', '17', '16', '15', '14', '13', '12', '11'] +  # 上右
        ['21', '22', '23', '24', '25', '26', '27', '28'] +  # 上左
        ['31', '32', '33', '34', '35', '36', '37', '38'] +  # 下左
        ['48', '47', '46', '45', '44', '43', '42', '41']    # 下右
    )

    # 测量面映射
    SURFACES_UPPER = ['db', 'b', 'mb', 'dp', 'p', 'mp']  # 上颌/磨牙
    SURFACES_LOWER = ['db', 'b', 'mb', 'dl', 'l', 'ml']  # 下颌

    # 测量类型识别模式
    MEASUREMENT_PATTERNS = {
        'PD': ['PD', 'PROBING', 'PROBE', 'DEPTH', '深度'],
        'GM': ['GM', 'GINGIVAL', 'MARGIN', 'MUCOGINGIVAL', '龈缘'],
        'BOP': ['BOP', 'BLEEDING', 'ON', 'PROBING', '出血'],
        'PI': ['PI', 'PLAQUE', 'INDEX', '菌斑'],
        'Mobility': ['MOBILITY', 'MOB', '松', '动度'],
        'Furcation': ['FURCATION', 'FURC', '分叉', '分叉病变']
    }

    # 正值标记（用于BOP/PI识别）
    POSITIVE_MARKS = ['1', '+', '✓', '√', 'X', '×', '●', 'O', '*']

    # 罗马数字映射
    ROMAN_NUMERALS = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}

    def __init__(
        self,
        use_table_cls: bool = False,
        use_cuda: bool = False,
        device: str = "cpu",
        enable_cache: bool = True,
        lazy_load: bool = True
    ):
        """
        初始化 RapidTableOCR 系统

        Args:
            use_table_cls: 是否使用表格分类（默认False，提升速度）
            use_cuda: 是否使用 CUDA 加速
            device: 设备类型
            enable_cache: 是否启用结果缓存
            lazy_load: 是否延迟加载模型（默认True）
        """
        self.use_table_cls = use_table_cls
        self.use_cuda = use_cuda
        self.device = device
        self.enable_cache = enable_cache
        self.lazy_load = lazy_load

        # 模型引擎（延迟加载）
        self.ocr_engine = None
        self.wired_engine = None
        self.lineless_engine = None
        self.table_cls = None

        # 缓存存储
        self._cache: Dict[str, Any] = {}

        # 立即初始化 OCR 引擎（轻量级）
        self._init_ocr_engine()

        # 如果非延迟加载，立即初始化所有模型
        if not lazy_load:
            self._load_all_models()

    def _load_all_models(self):
        """加载所有模型（非延迟模式）"""
        if self.wired_engine is None:
            self._init_table_engines()
        if self.use_table_cls and self.table_cls is None:
            self._init_table_cls()

    def _init_ocr_engine(self):
        """初始化 RapidOCR 引擎"""
        if self.ocr_engine is not None:
            return

        try:
            from rapidocr import RapidOCR
            self.ocr_engine = RapidOCR()
            logger.info("RapidOCR 初始化成功")
        except ImportError:
            logger.error("RapidOCR 未安装，请运行: pip install rapidocr")
            raise

    def _init_table_engines(self):
        """初始化有线和无线表格识别引擎"""
        if self.wired_engine is not None:
            return

        try:
            from wired_table_rec.main import WiredTableRecognition, WiredTableInput
            from lineless_table_rec.main import LinelessTableRecognition, LinelessTableInput

            # 有线表格识别
            wired_input = WiredTableInput(
                model_type="unet",
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
        if self.table_cls is not None:
            return

        try:
            from table_cls import TableCls
            self.table_cls = TableCls()
            logger.info("表格分类器初始化成功")
        except ImportError:
            logger.warning("表格分类器未安装，将默认使用有线表格识别")
            self.table_cls = None

    def _get_image_hash(self, image_path: str) -> str:
        """计算图像文件的哈希值，用于缓存"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return hashlib.md5(image_path.encode()).hexdigest()

    # ============================================================
    # 简化的 API 接口
    # ============================================================

    def process(self, image_path: str, **kwargs) -> Dict:
        """
        一键处理牙周图表（简化API）

        Args:
            image_path: 图像路径
            **kwargs: 其他参数传递给 process_periodontal_chart

        Returns:
            牙周数据字典

        Example:
            >>> ocr = RapidTableOCR()
            >>> result = ocr.process("handwrite.jpg")
        """
        return self.process_periodontal_chart(image_path, **kwargs)

    def recognize(self, image_path: str, table_type: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """
        识别表格（简化API）

        Args:
            image_path: 图像路径
            table_type: 表格类型（None=自动检测）

        Returns:
            (html, cells) 元组

        Example:
            >>> ocr = RapidTableOCR()
            >>> html, cells = ocr.recognize("handwrite.jpg")
        """
        result = self.recognize_table(image_path, table_type=table_type)
        return result['html'], result.get('cells', [])

    def parse(self, html: str) -> Dict:
        """
        解析 HTML 为牙周数据（简化API）

        Args:
            html: 表格 HTML

        Returns:
            牙周数据字典

        Example:
            >>> ocr = RapidTableOCR()
            >>> perio_data = ocr.parse(html)
        """
        return self.parse_html_to_periodontal_data(html)

    # ============================================================
    # 核心功能方法
    # ============================================================

    def classify_table(self, image_path: str) -> str:
        """
        分类表格类型

        Returns:
            "wired" - 有线表格
            "lineless" - 无线表格
        """
        # 延迟加载分类器
        if self.use_table_cls and self.table_cls is None:
            self._init_table_cls()

        if self.table_cls is None:
            return "wired"

        try:
            cls_result, _ = self.table_cls(image_path)
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
        # 检查缓存
        if self.enable_cache:
            cache_key = self._get_image_hash(image_path)
            if cache_key in self._cache:
                logger.info("使用缓存结果")
                return self._cache[cache_key]

        # 确保模型已加载
        if self.wired_engine is None:
            self._init_table_engines()

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
        engine = self.wired_engine if table_type == "wired" else self.lineless_engine

        # 表格识别
        table_results = engine(
            image_path,
            ocr_result=ocr_result,
            **kwargs
        )

        result = {
            'table_type': table_type,
            'html': table_results.pred_html if return_html else None,
            'cells': self._parse_cells(table_results) if return_cells else None,
            'logic_points': table_results.logic_points,
            'elapse': table_results.elapse
        }

        # 缓存结果
        if self.enable_cache:
            self._cache[cache_key] = result

        return result

    def _parse_cells(self, table_results) -> List[Dict]:
        """解析单元格信息"""
        cells = []
        if table_results.cell_bboxes is None:
            return cells

        for i, bbox in enumerate(table_results.cell_bboxes):
            bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
            cells.append({
                'index': i,
                'bbox': bbox_list,
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

        优化:
        - 简化HTML解析逻辑
        - 优化牙齿编号提取
        - 改进数值解析
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("BeautifulSoup 未安装，请运行: pip install beautifulsoup4")
            raise

        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')

        if table is None:
            return {}

        result = {}
        rows = table.find_all('tr')

        if not rows:
            return {}

        # 解析表头（牙齿编号）
        tooth_numbers = self._parse_header_rows(rows[0]) if rows else []

        # 解析数据行
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue

            # 识别测量类型（第一个单元格）
            measurement_type = self._identify_measurement_type(cells[0].get_text())

            # 解析数据单元格
            for col_idx, cell in enumerate(cells[1:], start=1):
                if col_idx - 1 < len(tooth_numbers):
                    tooth_num = tooth_numbers[col_idx - 1]
                    value = self._parse_cell_value(cell.get_text(), measurement_type)

                    # 根据牙齿位置选择测量面
                    surface = self._get_surface_for_tooth(tooth_num, col_idx - 1)

                    field_name = self._get_field_name(tooth_num, measurement_type, surface)
                    if field_name and value is not None:
                        result[field_name] = value

        return result

    def _parse_header_rows(self, header_row) -> List[str]:
        """解析表头单元格，提取牙齿编号"""
        tooth_numbers = []
        cells = header_row.find_all(['td', 'th'])

        for cell in cells:
            text = cell.get_text().strip()
            # 提取所有1-2位数字
            import re
            digits = re.findall(r'\d{1,2}', text)
            if digits:
                tooth_numbers.extend(digits)

        # 验证牙齿编号有效性
        valid_teeth = [t for t in tooth_numbers if t in self.ALL_TEETH]
        return valid_teeth

    def _identify_measurement_type(self, text: str) -> str:
        """识别测量类型（优化版）"""
        text_upper = text.upper().strip()

        # 按优先级匹配
        for mtype, patterns in self.MEASUREMENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_upper:
                    return mtype

        return 'PD'  # 默认为 PD

    def _get_surface_for_tooth(self, tooth_num: str, col_idx: int) -> str:
        """根据牙齿编号和列索引获取测量面"""
        # 下颌牙齿使用不同的测量面
        if tooth_num in ['31', '32', '33', '34', '35', '36', '37', '38',
                         '41', '42', '43', '44', '45', '46', '47', '48']:
            surfaces = self.SURFACES_LOWER
        else:
            surfaces = self.SURFACES_UPPER

        return surfaces[col_idx % len(surfaces)]

    def _parse_cell_value(self, text: str, measurement_type: str):
        """
        解析单元格值（优化版）

        支持更多格式:
        - 小数: 3.5, 3:5
        - 负数: -2
        - 罗马数字: I, II, III
        - 标记: +, X, ✓
        """
        text = text.strip()

        if not text:
            return 0

        # BOP/PI: 检查阳性标记
        if measurement_type in ['BOP', 'PI']:
            return 1 if any(mark in text for mark in self.POSITIVE_MARKS) else 0

        # PD/GM: 提取数值
        if measurement_type in ['PD', 'GM']:
            # 处理冒号作为小数点: 3:5 -> 3.5
            text = text.replace(':', '.').replace('：', '.')
            # 提取数字（包括负号和小数点）
            import re
            match = re.search(r'-?\d+\.?\d*', text)
            if match:
                try:
                    return float(match.group()) if '.' in match.group() else int(match.group())
                except ValueError:
                    pass
            return 0

        # Mobility/Furcation: 罗马数字或阿拉伯数字
        if measurement_type in ['Mobility', 'Furcation']:
            # 检查罗马数字
            roman = text.upper().strip()
            if roman in self.ROMAN_NUMERALS:
                return self.ROMAN_NUMERALS[roman]
            # 提取阿拉伯数字
            import re
            match = re.search(r'\d+', text)
            return int(match.group()) if match else 0

        return text

    def _get_field_name(self, tooth_num: str, measurement_type: str, surface: str) -> Optional[str]:
        """生成字段名称（优化版）"""
        if not tooth_num:
            return None

        # 统一字段命名规则
        field_map = {
            'PD': f'pd_{tooth_num}_{surface}',
            'GM': f'gm_{tooth_num}_{surface}',
            'BOP': f'bop_{tooth_num}_{surface}',
            'PI': f'pi_{tooth_num}_{surface}',
            'Mobility': f'mobility_{tooth_num}',
            'Furcation': f'furcation_{tooth_num}_{surface}'
        }

        return field_map.get(measurement_type)

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
            try:
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(table_result['html'])
                logger.info(f"HTML 已保存: {html_path}")
            except Exception as e:
                logger.warning(f"保存 HTML 失败: {e}")

        return perio_data

    def export_to_json(self, data: Dict, output_path: str):
        """导出数据到 JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"数据已保存到: {output_path}")

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        logger.info("缓存已清除")

    def visualize_table(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
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


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试简化 API
    print("=== 测试 RapidTableOCR ===")

    # 基本用法
    ocr = RapidTableOCR()

    print("\n1. 基本用法示例:")
    print("   ocr = RapidTableOCR()")
    print("   result = ocr.process('handwrite.jpg')")

    print("\n2. 高级用法示例:")
    print("   html, cells = ocr.recognize('handwrite.jpg')")
    print("   perio_data = ocr.parse(html)")

    print("\n3. 快速模式（跳过表格分类）:")
    print("   ocr = RapidTableOCR(use_table_cls=False)")

    # 测试牙周图表处理
    print("\n=== 测试牙周图表处理 ===")
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

    except FileNotFoundError:
        print("测试文件 handwrite.jpg 不存在，跳过测试")
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
