"""
表格检测模块 - 使用OpenCV检测牙周图表的网格结构
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TableDetector:
    """牙周图表表格检测器"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.debug = self.config.get('debug', False)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理：灰度化、去噪、增强对比度"""
        # 灰度化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯去噪
        if self.config.get('denoise', True):
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 自适应直方图均衡化
        if self.config.get('enhance_image', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        return gray

    def detect_grid_lines(self, binary: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """检测表格网格线"""
        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxGap=10
        )

        if lines is None:
            return [], []

        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # 水平线 (角度接近0或180)
            if abs(angle) < 10 or abs(angle) > 170:
                horizontal_lines.append(line[0])
            # 垂直线 (角度接近90)
            elif 80 < abs(angle) < 100:
                vertical_lines.append(line[0])

        # 合并相近的线
        horizontal_lines = self._merge_lines(horizontal_lines, is_horizontal=True)
        vertical_lines = self._merge_lines(vertical_lines, is_horizontal=False)

        return horizontal_lines, vertical_lines

    def _merge_lines(self, lines: List[np.ndarray], is_horizontal: bool) -> List[np.ndarray]:
        """合并相近的线条"""
        if not lines:
            return []

        merged = []
        lines = sorted(lines, key=lambda x: x[1] if is_horizontal else x[0])

        threshold = 10  # 像素距离阈值
        for line in lines:
            if not merged:
                merged.append(line.tolist())
                continue

            last = merged[-1]
            if is_horizontal:
                if abs(line[1] - last[1]) < threshold:
                    # 合并水平线
                    last[0] = min(last[0], line[0])
                    last[2] = max(last[2], line[2])
                else:
                    merged.append(line.tolist())
            else:
                if abs(line[0] - last[0]) < threshold:
                    # 合并垂直线
                    last[1] = min(last[1], line[1])
                    last[3] = max(last[3], line[3])
                else:
                    merged.append(line.tolist())

        return [np.array(l) for l in merged]

    def detect_table_regions(self, image: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """
        检测牙周图表的4个象限区域
        返回: {'upper_left': (x, y, w, h), 'upper_right': ..., 'lower_left': ..., 'lower_right': ...}
        """
        height, width = image.shape[:2]

        # 计算4个象限的中心点位置
        # 牙周图表通常分为:
        # - 右上颌 (11-18): upper_right
        # - 左上颌 (21-28): upper_left
        # - 左下颌 (31-38): lower_left
        # - 右下颌 (41-48): lower_right

        mid_x = width // 2
        mid_y = height // 2

        # 根据图像实际结构调整
        regions = {
            'upper_right': (mid_x, 0, width - mid_x, mid_y),  # 11-18
            'upper_left': (0, 0, mid_x, mid_y),               # 21-28
            'lower_left': (0, mid_y, mid_x, height - mid_y),  # 31-38
            'lower_right': (mid_x, mid_y, width - mid_x, height - mid_y)  # 41-48
        }

        return regions

    def detect_quadrant(self, image: np.ndarray, quadrant: str) -> Optional[np.ndarray]:
        """检测指定象限区域"""
        regions = self.detect_table_regions(image)

        if quadrant not in regions:
            logger.error(f"未知象限: {quadrant}")
            return None

        x, y, w, h = regions[quadrant]
        roi = image[y:y+h, x:x+w]

        return roi

    def find_tooth_cells(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        定位每颗牙齿的单元格区域
        返回: [(x, y, w, h), ...] 列表
        """
        # 预处理
        gray = self.preprocess_image(image)

        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 检测网格线
        h_lines, v_lines = self.detect_grid_lines(binary)

        # 计算交点，形成网格
        cells = []
        if len(h_lines) > 1 and len(v_lines) > 1:
            # 提取水平和垂直线的位置
            h_positions = sorted([(l[1] + l[3]) // 2 for l in h_lines])
            v_positions = sorted([(l[0] + l[2]) // 2 for l in v_lines])

            # 生成单元格区域
            for i in range(len(h_positions) - 1):
                for j in range(len(v_positions) - 1):
                    x1 = v_positions[j]
                    y1 = h_positions[i]
                    x2 = v_positions[j + 1]
                    y2 = h_positions[i + 1]
                    cells.append((x1, y1, x2 - x1, y2 - y1))

        if self.debug:
            # 保存调试图像
            debug_img = image.copy()
            for x, y, w, h in cells:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite('debug_cells.png', debug_img)
            logger.info(f"检测到 {len(cells)} 个单元格")

        return cells

    def extract_tooth_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取4个象限的牙齿区域
        返回: {'11-18': roi, '21-28': roi, '31-38': roi, '41-48': roi}
        """
        regions = {}
        quadrant_map = {
            'upper_right': '11-18',
            'upper_left': '21-28',
            'lower_left': '31-38',
            'lower_right': '41-48'
        }

        for quad_key, tooth_range in quadrant_map.items():
            roi = self.detect_quadrant(image, quad_key)
            if roi is not None and roi.size > 0:
                regions[tooth_range] = roi

        return regions


if __name__ == '__main__':
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        detector = TableDetector(debug=True)
        cells = detector.find_tooth_cells(image)
        print(f"检测到 {len(cells)} 个单元格")
