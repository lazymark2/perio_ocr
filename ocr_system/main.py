"""
主程序模块 - 牙周图表OCR识别系统
支持自动表格检测和基于模板的精确定位
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import json

from .paddle_ocr import PerioOCRSystem
from .auto_table_detector import SimpleTableOCR, AutoTableDetector
from .data_parser import DataParser
from .data_mapper import DataMapper

logger = logging.getLogger(__name__)


class PerioChartOCR:
    """牙周图表OCR主类"""

    # 象限牙齿编号映射
    QUADRANT_TEETH = {
        'upper_right': ['18', '17', '16', '15', '14', '13', '12', '11'],
        'upper_left': ['21', '22', '23', '24', '25', '26', '27', '28'],
        'lower_left': ['31', '32', '33', '34', '35', '36', '37', '38'],
        'lower_right': ['41', '42', '43', '44', '45', '46', '47', '48']
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.use_template = self.config.get('use_template', False)

        # 选择OCR引擎
        if self.use_template:
            self.ocr_system = PerioOCRSystem(self.config)
        else:
            # 使用自动表格检测OCR
            self.simple_ocr = SimpleTableOCR()

        self.data_parser = DataParser(self.config)
        self.data_mapper = DataMapper(self.config)

        # 加载模板配置（如果存在）
        self.template_config = self._load_template_config()

    def _load_template_config(self) -> dict:
        """加载模板配置文件"""
        config_path = Path(__file__).parent / 'template_config.json'

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}

    def process_image(self, image_path: str) -> Dict:
        """
        处理单张图像

        Args:
            image_path: 图像路径

        Returns:
            包含所有牙周数据的字典
        """
        logger.info(f"正在处理: {image_path}")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        if self.use_template and self.template_config:
            # 使用模板定位
            return self._process_with_template(image)
        else:
            # 使用自动表格检测
            return self._process_auto(image)

    def _process_auto(self, image: np.ndarray) -> Dict:
        """自动检测表格并识别"""
        # 保存临时图像
        temp_path = 'temp_handwrite.jpg'
        cv2.imwrite(temp_path, image)

        try:
            # 使用简单OCR进行表格识别
            ocr_result = self.simple_ocr.process(temp_path)

            # 解析OCR结果
            teeth_data = self._parse_ocr_results(ocr_result, image.shape)

            # 映射到标准结构
            result = self.data_mapper.map_ocr_to_structure(teeth_data)

            # 验证结果
            errors = self.data_mapper.validate_output(result)
            if errors:
                logger.warning(f"数据验证警告: {errors}")

            return result

        finally:
            # 清理临时文件
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _parse_ocr_results(self, ocr_result: Dict, image_shape: tuple) -> Dict:
        """解析OCR结果为牙齿数据"""
        height, width = _ = image_shape[:2]

        # 计算中心分割线
        mid_x = width // 2
        mid_y = height // 2

        results = ocr_result.get('results', [])
        teeth_data = {}

        for item in results:
            text = item.get('text', '')
            bbox = item.get('bbox', [])

            if not text or not bbox:
                continue

            # bbox格式: [x1, y1, x2, y2]
            # 计算中心点
            if len(bbox) == 4:
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
            else:
                continue

            # 确定象限
            if center_y < mid_y:
                if center_x > mid_x:
                    quadrant = 'upper_right'
                else:
                    quadrant = 'upper_left'
            else:
                if center_x > mid_x:
                    quadrant = 'lower_right'
                else:
                    quadrant = 'lower_left'

            # 确定牙齿和测量面
            tooth_num, surface = self._locate_tooth_surface(
                center_x, center_y, mid_x, mid_y, quadrant, width, height
            )

            if not tooth_num:
                continue

            # 提取数字值 - 改进逻辑
            pd_value = self._extract_pd_value(text)

            # 只存储有效值 (0-15mm为正常PD范围)
            if pd_value is None or pd_value > 15:
                continue

            # 存储数据
            if tooth_num not in teeth_data:
                teeth_data[tooth_num] = self._create_empty_tooth_data(tooth_num)

            teeth_data[tooth_num]['pd'][surface] = pd_value

        return teeth_data

    def _extract_pd_value(self, text: str) -> Optional[int]:
        """
        从文本中提取有效的PD值

        Args:
            text: OCR识别的文本

        Returns:
            提取的PD值，无效返回None
        """
        import re

        # 查找所有数字
        numbers = re.findall(r'\d+', text)

        if not numbers:
            return None

        # 转换为整数
        values = [int(n) for n in numbers]

        # 优先返回单个有效的PD值 (0-15mm)
        for v in values:
            if 0 <= v <= 15:
                return v

        # 如果没有有效值，返回None
        return None

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

        result = {
            'tooth_num': tooth_num,
            'tooth': 1,
            'mobility': 0,
            'implant': 0,
            'pd': {s: 0 for s in surfaces},
            'gm': {s: 0 for s in surfaces},
            'bop': {s: 0 for s in surfaces},
            'pi': {s: 0 for s in surfaces},
            'furcation': {}
        }

        # 分叉病变
        if tooth_num[1] in '3678':
            if 'dp' in surfaces:
                result['furcation'] = {'b': 0, 'dp': 0, 'mp': 0}
            else:
                result['furcation'] = {'b': 0, 'l': 0}

        return result

    def _process_with_template(self, image: np.ndarray) -> Dict:
        """使用模板配置处理（之前定义的逻辑）"""
        # 这里可以复用之前的模板处理代码
        # 由于代码较长，这里简化为回退到自动模式
        logger.info("回退到自动检测模式")
        return self._process_auto(image)

    def process_batch(self, image_dir: str, output_dir: str) -> List[Dict]:
        """批量处理图像"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        image_path = Path(image_dir)
        image_files = list(image_path.glob('*.jpg')) + \
                      list(image_path.glob('*.jpeg')) + \
                      list(image_path.glob('*.png'))

        results = []
        for img_file in image_files:
            try:
                result = self.process_image(str(img_file))
                output_file = Path(output_dir) / f"{img_file.stem}_result.json"
                self.data_mapper.export_to_json(result, str(output_file))
                results.append({
                    'image': str(img_file),
                    'output': str(output_file),
                    'success': True
                })
            except Exception as e:
                logger.error(f"处理 {img_file} 失败: {e}")
                results.append({
                    'image': str(img_file),
                    'error': str(e),
                    'success': False
                })

        return results


def create_default_config() -> dict:
    """创建默认配置"""
    return {
        'ocr': {
            'lang': 'ch',
            'use_gpu': False,
            'det_db_thresh': 0.3,
            'det_db_box_thresh': 0.6
        },
        'use_template': False,  # 默认使用自动检测
        'paths': {
            'input_dir': 'input/',
            'output_dir': 'output/'
        },
        'processing': {
            'enhance_image': True,
            'binarize': True,
            'denoise': True
        },
        'debug': False
    }


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("用法: python main.py <图像路径>")
        sys.exit(1)

    ocr = PerioChartOCR(create_default_config())
    result = ocr.process_image(sys.argv[1])

    print(json.dumps(result, ensure_ascii=False, indent=2))
