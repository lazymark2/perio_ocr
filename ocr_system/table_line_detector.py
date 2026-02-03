"""
表格线检测模块 - 精确定位牙周图表的行列结构

优化版本:
- 简化预处理流程，减少冗余操作
- 智能参数调整，适应不同尺寸图像
- 统一检测接口，减少代码重复
- 增强稳定性，处理边缘情况

作者: perio_OCR 项目
版本: 2.0.0
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TableLineDetector:
    """表格线检测器 - 精确定位表格行列（优化版）"""

    def __init__(
        self,
        min_line_length: Optional[int] = None,
        max_line_gap: Optional[int] = None,
        merge_threshold: int = 10,
        hough_threshold: int = 25,
        debug: bool = False
    ):
        """
        初始化检测器（参数自动适配图像尺寸）

        Args:
            min_line_length: 最小线段长度（像素），None则自动计算
            max_line_gap: 霍夫变换最大间隙，None则自动计算
            merge_threshold: 线段合并距离阈值
            hough_threshold: 霍夫变换阈值（越低检测越灵敏）
            debug: 是否输出调试信息
        """
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.merge_threshold = merge_threshold
        self.hough_threshold = hough_threshold
        self.debug = debug
        self._image_size = None

    def _auto_adjust_params(self, image: np.ndarray) -> None:
        """根据图像尺寸自动调整参数"""
        h, w = image.shape[:2]
        self._image_size = (w, h)

        if self.min_line_length is None:
            # 最小线长为图像较短边的5%
            self.min_line_length = max(50, min(w, h) // 20)

        if self.max_line_gap is None:
            # 最大间隙为图像较短边的2%
            self.max_line_gap = max(10, min(w, h) // 50)

        if self.debug:
            logger.info(f"图像尺寸: {w}x{h}, 自动参数: min_line_length={self.min_line_length}, max_line_gap={self.max_line_gap}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理（优化版 - 减少冗余操作）

        Args:
            image: 输入图像（BGR或灰度）

        Returns:
            二值化图像
        """
        # 灰度化（如需要）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 检查图像尺寸，超大图像缩放处理
        h, w = gray.shape
        max_size = 2000
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if self.debug:
                logger.info(f"图像缩放: {max(h,w)} -> {max_size)}")

        # 自适应阈值（合并去噪和二值化）
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        return binary

    def _detect_lines(
        self,
        image: np.ndarray,
        orientation: str = 'horizontal'
    ) -> List[Tuple[int, int, int, int]]:
        """
        统一的线检测方法（减少代码重复）

        Args:
            image: 输入图像
            orientation: 'horizontal' 或 'vertical'

        Returns:
            [(x1, y1, x2, y2), ...] 线段坐标列表
        """
        binary = self.preprocess(image)

        # 根据方向设置形态学核
        if orientation == 'horizontal':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            angle_range = (-10, 10)
        else:  # vertical
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            angle_range = (80, 100)

        # 形态学操作提取线条
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            morph,
            1,
            np.pi / 180,
            threshold=self.hough_threshold,  # 降低阈值提高灵敏度
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap  # 增大间隙连接断裂线段
        )

        if lines is None:
            if self.debug:
                logger.warning(f"未检测到{orientation}线")
            return []

        # 筛选符合角度要求的线
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if orientation == 'horizontal':
                # 水平线：角度接近0或180
                if angle < angle_range[1] or angle > 180 - angle_range[1]:
                    filtered_lines.append((x1, y1, x2, y2))
            else:
                # 垂直线：角度接近90
                if angle_range[0] < angle < angle_range[1]:
                    filtered_lines.append((x1, y1, x2, y2))

        # 合并相近的线段
        merged_lines = self._merge_lines(filtered_lines, orientation)

        if self.debug:
            logger.info(f"检测到 {len(merged_lines)} 条{orientation}线")

        return merged_lines

    def _merge_lines(
        self,
        lines: List[Tuple[int, int, int, int]],
        orientation: str
    ) -> List[Tuple[int, int, int, int]]:
        """
        合并相近的线段（统一方法）

        Args:
            lines: 线段列表
            orientation: 'horizontal' 或 'vertical'

        Returns:
            合并后的线段列表
        """
        if not lines:
            return []

        # 按位置排序
        if orientation == 'horizontal':
            sorted_lines = sorted(lines, key=lambda l: (l[1], l[0]))
        else:
            sorted_lines = sorted(lines, key=lambda l: (l[0], l[1]))

        merged = []
        current_group = [sorted_lines[0]]

        for line in sorted_lines[1:]:
            # 判断是否应该合并
            should_merge = False
            if orientation == 'horizontal':
                # Y坐标相近且X坐标有重叠或相连
                y_close = abs(line[1] - current_group[0][1]) <= self.merge_threshold
                x_overlap = max(0, min(line[2], current_group[0][2]) - max(line[0], current_group[0][0])) > -self.merge_threshold * 2
                should_merge = y_close and x_overlap
            else:
                # X坐标相近且Y坐标有重叠或相连
                x_close = abs(line[0] - current_group[0][0]) <= self.merge_threshold
                y_overlap = max(0, min(line[3], current_group[0][3]) - max(line[1], current_group[0][1])) > -self.merge_threshold * 2
                should_merge = x_close and y_overlap

            if should_merge:
                current_group.append(line)
            else:
                merged.append(self._merge_line_group(current_group, orientation))
                current_group = [line]

        if current_group:
            merged.append(self._merge_line_group(current_group, orientation))

        return merged

    def _merge_line_group(
        self,
        lines: List[Tuple[int, int, int, int]],
        orientation: str
    ) -> Tuple[int, int, int, int]:
        """合并一组线段为一条线"""
        if orientation == 'horizontal':
            min_x = min(l[0] for l in lines)
            max_x = max(l[2] for l in lines)
            avg_y = int(np.mean([l[1] for l in lines]))
            return (min_x, avg_y, max_x, avg_y)
        else:
            avg_x = int(np.mean([l[0] for l in lines]))
            min_y = min(l[1] for l in lines)
            max_y = max(l[3] for l in lines)
            return (avg_x, min_y, avg_x, max_y)

    def detect_horizontal_lines(
        self,
        image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """检测水平表格线（行分隔）"""
        self._auto_adjust_params(image)
        return self._detect_lines(image, 'horizontal')

    def detect_vertical_lines(
        self,
        image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """检测垂直表格线（列分隔）"""
        self._auto_adjust_params(image)
        return self._detect_lines(image, 'vertical')

    def detect_all(
        self,
        image: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], List[Dict]]:
        """
        一步检测所有线条和单元格

        Returns:
            (h_lines, v_lines, cells) 元组
        """
        self._auto_adjust_params(image)
        h_lines = self._detect_lines(image, 'horizontal')
        v_lines = self._detect_lines(image, 'vertical')
        cells = self.get_cell_boundaries(h_lines, v_lines)
        return h_lines, v_lines, cells

    def get_cell_boundaries(
        self,
        h_lines: List[Tuple[int, int, int, int]],
        v_lines: List[Tuple[int, int, int, int]]
    ) -> List[Dict]:
        """
        根据线交叉点获取单元格边界

        Returns:
            [{'bbox': [x1, y1, x2, y2], 'row': r, 'col': c, 'size': (w, h)}, ...]
        """
        row_positions = sorted(set([line[1] for line in h_lines]))
        col_positions = sorted(set([line[0] for line in v_lines]))

        if len(row_positions) < 2 or len(col_positions) < 2:
            if self.debug:
                logger.warning(f"线条数量不足（行:{len(row_positions)}, 列:{len(col_positions)}），无法形成单元格")
            return []

        cells = []
        for row_idx in range(len(row_positions) - 1):
            for col_idx in range(len(col_positions) - 1):
                x1, y1 = col_positions[col_idx], row_positions[row_idx]
                x2, y2 = col_positions[col_idx + 1], row_positions[row_idx + 1]
                cells.append({
                    'bbox': [x1, y1, x2, y2],
                    'row': row_idx,
                    'col': col_idx,
                    'size': (x2 - x1, y2 - y1)
                })

        if self.debug:
            logger.info(f"生成 {len(cells)} 个单元格 ({len(col_positions)-1}列 x {len(row_positions)-1}行)")

        return cells

    def visualize(
        self,
        image: np.ndarray,
        h_lines: Optional[List[Tuple[int, int, int, int]]] = None,
        v_lines: Optional[List[Tuple[int, int, int, int]]] = None,
        cells: Optional[List[Dict]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        可视化检测结果（支持线条和单元格）

        Args:
            image: 原始图像
            h_lines: 水平线列表（可选）
            v_lines: 垂直线列表（可选）
            cells: 单元格列表（可选）
            thickness: 线条粗细

        Returns:
            可视化图像
        """
        vis_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if h_lines:
            for x1, y1, x2, y2 in h_lines:
                cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        if v_lines:
            for x1, y1, x2, y2 in v_lines:
                cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), thickness)

        if cells:
            for cell in cells:
                x1, y1, x2, y2 = cell['bbox']
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 255), 1)

        return vis_img

    def __call__(
        self,
        image: Union[str, np.ndarray],
        return_all: bool = False,
        output_debug: Optional[str] = None
    ) -> Union[List[Dict], Tuple[List, List, List]]:
        """
        一站式检测接口（最简洁的API）

        Args:
            image: 图像路径或numpy数组
            return_all: True返回(h_lines, v_lines, cells)，False只返回cells
            output_debug: 调试图像保存路径（None则不保存）

        Returns:
            cells 或 (h_lines, v_lines, cells)
        """
        # 读取图像
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        else:
            img = image

        # 检测所有线条和单元格
        h_lines, v_lines, cells = self.detect_all(img)

        # 保存调试图像
        if output_debug:
            vis_img = self.visualize(img, h_lines, v_lines, cells)
            cv2.imwrite(output_debug, vis_img)
            if self.debug:
                logger.info(f"调试图像已保存: {output_debug}")

        if return_all:
            return h_lines, v_lines, cells
        return cells


if __name__ == '__main__':
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='表格线检测工具')
    parser.add_argument('image', help='输入图像路径')
    parser.add_argument('--output', '-o', help='输出可视化图像路径')
    parser.add_argument('--debug', '-d', action='store_true', help='启用调试模式')
    args = parser.parse_args()

    detector = TableLineDetector(debug=args.debug)

    # 简洁的API调用
    cells = detector(args.image, output_debug=args.output)

    print(f"检测到 {len(cells)} 个单元格")
    if cells:
        print(f"表格尺寸: {max(c['col'] for c in cells)+1} 列 x {max(c['row'] for c in cells)+1} 行")
