"""
表格线检测模块 - 精确定位牙周图表的行列结构

使用形态学操作和霍夫变换检测表格线，支持：
- 自适应阈值处理不同光照
- 形态学操作提取线条
- 霍夫变换精确定位
- 线段合并处理断裂情况

作者: perio_OCR 项目
版本: 1.0.0
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TableLineDetector:
    """表格线检测器 - 精确定位表格行列"""

    def __init__(
        self,
        min_line_length: int = 50,
        max_line_gap: int = 10,
        merge_threshold: int = 10,
        debug: bool = False
    ):
        """
        初始化检测器

        Args:
            min_line_length: 最小线段长度（像素）
            max_line_gap: 霍夫变换最大间隙
            merge_threshold: 线段合并距离阈值
            debug: 是否输出调试信息
        """
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.merge_threshold = merge_threshold
        self.debug = debug

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理

        Args:
            image: 输入图像（BGR或灰度）

        Returns:
            二值化图像
        """
        # 灰度化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯去噪
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 自适应阈值（处理不同光照条件）
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        return binary

    def detect_horizontal_lines(
        self,
        image: np.ndarray,
        use_morphology: bool = True
    ) -> List[Tuple[int, int, int, int]]:
        """
        检测水平表格线（行分隔）

        Args:
            image: 输入图像
            use_morphology: 是否使用形态学操作增强水平线

        Returns:
            [(x1, y1, x2, y2), ...] 线段坐标列表
        """
        binary = self.preprocess(image)

        if use_morphology:
            # 形态学操作：提取水平线
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            # 膨胀连接断裂的线段
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            morph = cv2.dilate(morph, kernel, iterations=1)
        else:
            morph = binary

        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            morph,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            if self.debug:
                logger.warning("未检测到水平线")
            return []

        # 筛选水平线（角度接近0或180度）
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) < 10 or abs(angle) > 170:
                horizontal_lines.append((x1, y1, x2, y2))

        # 合并相近的线段
        horizontal_lines = self._merge_horizontal_lines(horizontal_lines)

        if self.debug:
            logger.info(f"检测到 {len(horizontal_lines)} 条水平线")

        return horizontal_lines

    def detect_vertical_lines(
        self,
        image: np.ndarray,
        use_morphology: bool = True
    ) -> List[Tuple[int, int, int, int]]:
        """
        检测垂直表格线（列分隔）

        Args:
            image: 输入图像
            use_morphology: 是否使用形态学操作增强垂直线

        Returns:
            [(x1, y1, x2, y2), ...] 线段坐标列表
        """
        binary = self.preprocess(image)

        if use_morphology:
            # 形态学操作：提取垂直线
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            # 膨胀连接断裂的线段
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            morph = cv2.dilate(morph, kernel, iterations=1)
        else:
            morph = binary

        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            morph,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            if self.debug:
                logger.warning("未检测到垂直线")
            return []

        # 筛选垂直线（角度接近90度）
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if 80 < abs(angle) < 100:
                vertical_lines.append((x1, y1, x2, y2))

        # 合并相近的线段
        vertical_lines = self._merge_vertical_lines(vertical_lines)

        if self.debug:
            logger.info(f"检测到 {len(vertical_lines)} 条垂直线")

        return vertical_lines

    def _merge_horizontal_lines(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """合并相近的水平线"""
        if not lines:
            return []

        sorted_lines = sorted(lines, key=lambda l: l[1])
        merged = []
        current_group = [sorted_lines[0]]

        for line in sorted_lines[1:]:
            if abs(line[1] - current_group[0][1]) <= self.merge_threshold:
                current_group.append(line)
            else:
                merged.append(self._merge_line_group(current_group, is_horizontal=True))
                current_group = [line]

        if current_group:
            merged.append(self._merge_line_group(current_group, is_horizontal=True))

        return merged

    def _merge_vertical_lines(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """合并相近的垂直线"""
        if not lines:
            return []

        sorted_lines = sorted(lines, key=lambda l: l[0])
        merged = []
        current_group = [sorted_lines[0]]

        for line in sorted_lines[1:]:
            if abs(line[0] - current_group[0][0]) <= self.merge_threshold:
                current_group.append(line)
            else:
                merged.append(self._merge_line_group(current_group, is_horizontal=False))
                current_group = [line]

        if current_group:
            merged.append(self._merge_line_group(current_group, is_horizontal=False))

        return merged

    def _merge_line_group(
        self,
        lines: List[Tuple[int, int, int, int]],
        is_horizontal: bool
    ) -> Tuple[int, int, int, int]:
        """合并一组线段为一条线"""
        if is_horizontal:
            min_x = min(l[0] for l in lines)
            max_x = max(l[2] for l in lines)
            avg_y = int(np.mean([l[1] for l in lines]))
            return (min_x, avg_y, max_x, avg_y)
        else:
            avg_x = int(np.mean([l[0] for l in lines]))
            min_y = min(l[1] for l in lines)
            max_y = max(l[3] for l in lines)
            return (avg_x, min_y, avg_x, max_y)

    def get_cell_boundaries(
        self,
        h_lines: List[Tuple[int, int, int, int]],
        v_lines: List[Tuple[int, int, int, int]]
    ) -> List[Dict]:
        """
        根据线交叉点获取单元格边界

        Returns:
            [{'bbox': [x1, y1, x2, y2], 'row': r, 'col': c}, ...]
        """
        row_positions = sorted([line[1] for line in h_lines])
        col_positions = sorted([line[0] for line in v_lines])

        if len(row_positions) < 2 or len(col_positions) < 2:
            if self.debug:
                logger.warning("线条数量不足，无法形成单元格")
            return []

        cells = []
        for row_idx in range(len(row_positions) - 1):
            for col_idx in range(len(col_positions) - 1):
                cells.append({
                    'bbox': [
                        col_positions[col_idx],
                        row_positions[row_idx],
                        col_positions[col_idx + 1],
                        row_positions[row_idx + 1]
                    ],
                    'row': row_idx,
                    'col': col_idx,
                    'size': (
                        col_positions[col_idx + 1] - col_positions[col_idx],
                        row_positions[row_idx + 1] - row_positions[row_idx]
                    )
                })

        if self.debug:
            logger.info(f"生成 {len(cells)} 个单元格")

        return cells

    def visualize_lines(
        self,
        image: np.ndarray,
        h_lines: List[Tuple[int, int, int, int]],
        v_lines: List[Tuple[int, int, int, int]],
        thickness: int = 2
    ) -> np.ndarray:
        """在图像上绘制检测到的表格线"""
        vis_img = image.copy()

        for x1, y1, x2, y2 in h_lines:
            cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        for x1, y1, x2, y2 in v_lines:
            cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), thickness)

        return vis_img

    def detect_table_cells(
        self,
        image_path: str,
        output_debug: bool = False
    ) -> List[Dict]:
        """一站式检测表格单元格"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        h_lines = self.detect_horizontal_lines(image)
        v_lines = self.detect_vertical_lines(image)
        cells = self.get_cell_boundaries(h_lines, v_lines)

        if output_debug:
            vis_img = self.visualize_lines(image, h_lines, v_lines)
            debug_path = image_path.rsplit('.', 1)[0] + '_table_lines.png'
            cv2.imwrite(debug_path, vis_img)
            logger.info(f"调试图像已保存: {debug_path}")

        return cells


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("用法: python table_line_detector.py <image_path>")
        sys.exit(1)

    detector = TableLineDetector(debug=True)
    cells = detector.detect_table_cells(sys.argv[1], output_debug=True)
    print(f"检测到 {len(cells)} 个单元格")
