"""
单元格提取模块 - 从表格中提取各个测量单元格
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CellExtractor:
    """牙周图表单元格提取器"""

    # 牙齿编号映射到FDI区号
    TOOTH_MAP = {
        # 右上颌 (11-18)
        '11': {'fdi': 18, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '12': {'fdi': 17, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '13': {'fdi': 16, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '14': {'fdi': 15, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '15': {'fdi': 14, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '16': {'fdi': 13, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '17': {'fdi': 12, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '18': {'fdi': 11, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        # 左上颌 (21-28)
        '21': {'fdi': 21, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '22': {'fdi': 22, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '23': {'fdi': 23, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '24': {'fdi': 24, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '25': {'fdi': 25, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '26': {'fdi': 26, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '27': {'fdi': 27, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '28': {'fdi': 28, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        # 左下颌 (31-38)
        '31': {'fdi': 31, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '32': {'fdi': 32, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '33': {'fdi': 33, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '34': {'fdi': 34, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '35': {'fdi': 35, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '36': {'fdi': 36, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '37': {'fdi': 37, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        '38': {'fdi': 38, 'surfaces': ['db', 'b', 'mb', 'dl', 'l', 'ml']},
        # 右下颌 (41-48)
        '41': {'fdi': 48, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '42': {'fdi': 47, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '43': {'fdi': 46, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '44': {'fdi': 45, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '45': {'fdi': 44, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '46': {'fdi': 43, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '47': {'fdi': 42, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
        '48': {'fdi': 41, 'surfaces': ['db', 'b', 'mb', 'dp', 'p', 'mp']},
    }

    def __init__(self, config: dict = None):
        self.config = config or {}

    def extract_cells_from_quadrant(self, quadrant_img: np.ndarray,
                                      tooth_numbers: List[str]) -> Dict[str, Dict]:
        """
        从一个象限中提取多颗牙齿的单元格

        Args:
            quadrant_img: 象限图像
            tooth_numbers: 牙齿编号列表，如 ['11', '12', '13']

        Returns:
            包含各牙齿测量数据的字典
        """
        result = {}

        height, width = quadrant_img.shape[:2]

        # 计算每颗牙齿的垂直位置
        cell_height = height // len(tooth_numbers)

        for idx, tooth_num in enumerate(tooth_numbers):
            y_start = idx * cell_height
            y_end = (idx + 1) * cell_height

            # 提取当前牙齿区域
            tooth_region = quadrant_img[y_start:y_end, :]

            # 进一步分割为PD/GM行和BOP/PI行
            result[tooth_num] = self._extract_tooth_cells(tooth_region, tooth_num)

        return result

    def _extract_tooth_cells(self, tooth_region: np.ndarray,
                             tooth_num: str) -> Dict:
        """提取单颗牙齿的各个测量单元格"""
        tooth_info = self.TOOTH_MAP.get(tooth_num, {})
        surfaces = tooth_info.get('surfaces', ['db', 'b', 'mb', 'dp', 'p', 'mp'])

        result = {
            'tooth': 1,  # 默认存在
            'mobility': 0,
            'implant': 0,
            'furcation': {},
            'pd': {},
            'gm': {},
            'bop': {},
            'pi': {}
        }

        # 检测牙齿是否存在（通过轮廓检测）
        result['tooth'] = self._check_tooth_exists(tooth_region)

        # 检测活动度
        result['mobility'] = self._extract_mobility(tooth_region)

        # 分叉病变（仅磨牙）
        if len(surfaces) == 6 and 'dp' in surfaces:
            result['furcation'] = self._extract_furcation(tooth_region, surfaces)

        return result

    def _check_tooth_exists(self, region: np.ndarray) -> int:
        """检查牙齿是否存在"""
        # 通过检测单元格区域是否有内容来判断
        if region.size == 0:
            return 0

        # 灰度化
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()

        # 计算非零像素比例
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        non_zero = cv2.countNonZero(binary)

        if non_zero > 100:  # 有内容存在
            return 1
        return 0

    def _extract_mobility(self, region: np.ndarray) -> int:
        """提取活动度值"""
        # 活动度通常在特定位置
        # 0=正常, 1=轻微, 2=中等, 3=严重
        return 0  # 默认值

    def _extract_furcation(self, region: np.ndarray,
                           surfaces: List[str]) -> Dict[str, int]:
        """提取分叉病变等级"""
        furcation = {}
        furcation_surfaces = ['b', 'dp', 'mp'] if 'dp' in surfaces else ['b', 'l']

        for surface in furcation_surfaces:
            furcation[surface] = 0  # 默认无分叉病变

        return furcation

    def extract_all_cells(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        从完整图像中提取所有牙齿的单元格

        Returns:
            包含所有32颗牙齿测量数据的字典
        """
        height, width = image.shape[:2]
        mid_x = width // 2
        mid_y = height // 2

        # 4个象限
        quadrants = {
            'upper_right': {
                'image': image[0:mid_y, mid_x:width],
                'teeth': ['18', '17', '16', '15', '14', '13', '12', '11']
            },
            'upper_left': {
                'image': image[0:mid_y, 0:mid_x],
                'teeth': ['21', '22', '23', '24', '25', '26', '27', '28']
            },
            'lower_left': {
                'image': image[mid_y:height, 0:mid_x],
                'teeth': ['31', '32', '33', '34', '35', '36', '37', '38']
            },
            'lower_right': {
                'image': image[mid_y:height, mid_x:width],
                'teeth': ['41', '42', '43', '44', '45', '46', '47', '48']
            }
        }

        all_results = {}

        for quadrant_name, quadrant_data in quadrants.items():
            quad_img = quadrant_data['image']
            teeth = quadrant_data['teeth']

            results = self.extract_cells_from_quadrant(quad_img, teeth)
            all_results.update(results)

        return all_results

    def extract_single_surface(self, image: np.ndarray,
                                bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """提取单个单元格图像"""
        x, y, w, h = bbox
        cell = image[y:y+h, x:x+w]
        return cell


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        extractor = CellExtractor()
        results = extractor.extract_all_cells(image)
        print(f"提取了 {len(results)} 颗牙齿的数据")
