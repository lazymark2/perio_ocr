"""
表格线检测模块 - 使用形态学操作检测表格网格线

专门针对牙周图表的表格线检测，使用形态学操作比霍夫变换更稳定。
支持检测水平和垂直表格线，并计算单元格边界。
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TableLineDetector:
    """
    表格线检测器

    使用形态学操作（开运算）检测表格的水平线和垂直线。
    这种方法对噪声、断线、倾斜等因素有较好的鲁棒性。
    """

    def __init__(self, config: dict = None):
        """
        初始化检测器

        Args:
            config: 配置字典，支持以下参数:
                - kernel_size_h: 水平线检测核高度 (默认40)
                - kernel_size_v: 垂直线检测核宽度 (默认40)
                - min_line_length: 最小线长度（像素）
                - merge_threshold: 线条合并阈值（像素）
                - debug: 是否输出调试信息
        """
        self.config = config or {}
        self.kernel_size_h = self.config.get('kernel_size_h', 40)
        self.kernel_size_v = self.config.get('kernel_size_v', 40)
        self.min_line_length = self.config.get('min_line_length', 50)
        self.merge_threshold = self.config.get('merge_threshold', 10)
        self.debug = self.config.get('debug', False)

        # 创建形态学操作的结构元素
        self.kernel_h = np.ones((1, self.kernel_size_h), np.uint8)  # 水平线检测
        self.kernel_v = np.ones((self.kernel_size_v, 1), np.uint8)  # 垂直线检测

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理

        Args:
            image: 输入图像（BGR或灰度）

        Returns:
            预处理后的灰度图像
        """
        # 灰度化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 去噪
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 自适应阈值二值化（对光照变化更鲁棒）
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        return binary

    def detect_horizontal_lines(self, image_path: str = None, image: np.ndarray = None) -> List[np.ndarray]:
        """
        检测水平表格线（行分隔）

        Args:
            image_path: 图像文件路径（与image参数二选一）
            image: 输入图像（numpy数组）

        Returns:
            水平线列表，每条线格式为 [x1, y1, x2, y2]
        """
        # 获取图像
        if image is None:
            if image_path is None:
                raise ValueError("必须提供 image_path 或 image 参数")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

        # 预处理
        binary = self.preprocess_image(image)
        height, width = binary.shape

        # 形态学开运算检测水平线
        # 使用横向结构元素可以保留水平方向的长线条
        morph_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_h)

        # 再次膨胀，使线条更连续
        morph_h = cv2.dilate(morph_h, self.kernel_h, iterations=1)

        # 提取线条
        horizontal_lines = self._extract_lines_from_binary(morph_h, is_horizontal=True)

        if self.debug:
            debug_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for line in horizontal_lines:
                cv2.line(debug_img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
            cv2.imwrite('debug_horizontal_lines.png', debug_img)
            logger.info(f"检测到 {len(horizontal_lines)} 条水平线")

        return horizontal_lines

    def detect_vertical_lines(self, image_path: str = None, image: np.ndarray = None) -> List[np.ndarray]:
        """
        检测垂直表格线（列分隔）

        Args:
            image_path: 图像文件路径（与image参数二选一）
            image: 输入图像（numpy数组）

        Returns:
            垂直线列表，每条线格式为 [x1, y1, x2, y2]
        """
        # 获取图像
        if image is None:
            if image_path is None:
                raise ValueError("必须提供 image_path 或 image 参数")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

        # 预处理
        binary = self.preprocess_image(image)

        # 形态学开运算检测垂直线
        # 使用纵向结构元素可以保留垂直方向的长线条
        morph_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_v)

        # 再次膨胀，使线条更连续
        morph_v = cv2.dilate(morph_v, self.kernel_v, iterations=1)

        # 提取线条
        vertical_lines = self._extract_lines_from_binary(morph_v, is_horizontal=False)

        if self.debug:
            debug_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for line in vertical_lines:
                cv2.line(debug_img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
            cv2.imwrite('debug_vertical_lines.png', debug_img)
            logger.info(f"检测到 {len(vertical_lines)} 条垂直线")

        return vertical_lines

    def _extract_lines_from_binary(self, binary: np.ndarray, is_horizontal: bool) -> List[np.ndarray]:
        """
        从二值图像中提取线条

        Args:
            binary: 二值图像
            is_horizontal: 是否为水平线

        Returns:
            线条列表，每条线格式为 [x1, y1, x2, y2]
        """
        height, width = binary.shape
        lines = []

        if is_horizontal:
            # 检测水平线：按行统计像素
            for y in range(height):
                row = binary[y, :]
                # 找到连续的前景像素段
                segments = self._find_continuous_segments(row)
                for x1, x2 in segments:
                    length = x2 - x1
                    if length >= self.min_line_length:
                        lines.append([x1, y, x2, y])
        else:
            # 检测垂直线：按列统计像素
            for x in range(width):
                col = binary[:, x]
                # 找到连续的前景像素段
                segments = self._find_continuous_segments(col)
                for y1, y2 in segments:
                    length = y2 - y1
                    if length >= self.min_line_length:
                        lines.append([x, y1, x, y2])

        # 合并相近的线条
        lines = self._merge_lines(lines, is_horizontal=is_horizontal)

        return [np.array(line) for line in lines]

    def _find_continuous_segments(self, array: np.ndarray) -> List[Tuple[int, int]]:
        """
        在一维数组中查找连续的前景像素段

        Args:
            array: 一维二值数组

        Returns:
            连续段的起止位置列表 [(start1, end1), (start2, end2), ...]
        """
        segments = []
        in_segment = False
        start = 0

        for i, val in enumerate(array):
            if val > 0:  # 前景像素
                if not in_segment:
                    in_segment = True
                    start = i
            else:  # 背景像素
                if in_segment:
                    in_segment = False
                    segments.append((start, i))

        # 处理数组末尾的段
        if in_segment:
            segments.append((start, len(array)))

        return segments

    def _merge_lines(self, lines: List, is_horizontal: bool) -> List:
        """
        合并相近的平行线条

        Args:
            lines: 线条列表
            is_horizontal: 是否为水平线

        Returns:
            合并后的线条列表
        """
        if not lines:
            return []

        # 按位置排序
        if is_horizontal:
            lines = sorted(lines, key=lambda x: x[1])  # 按y坐标排序
        else:
            lines = sorted(lines, key=lambda x: x[0])  # 按x坐标排序

        merged = []
        for line in lines:
            if not merged:
                merged.append(line)
                continue

            last = merged[-1]
            if is_horizontal:
                # 检查y坐标是否相近
                if abs(line[1] - last[1]) < self.merge_threshold:
                    # 合并：扩展x范围
                    last[0] = min(last[0], line[0])
                    last[2] = max(last[2], line[2])
                    # 更新y坐标为平均值
                    avg_y = (last[1] + line[1]) // 2
                    last[1] = last[3] = avg_y
                else:
                    merged.append(line)
            else:
                # 检查x坐标是否相近
                if abs(line[0] - last[0]) < self.merge_threshold:
                    # 合并：扩展y范围
                    last[1] = min(last[1], line[1])
                    last[3] = max(last[3], line[3])
                    # 更新x坐标为平均值
                    avg_x = (last[0] + line[0]) // 2
                    last[0] = last[2] = avg_x
                else:
                    merged.append(line)

        return merged

    def get_cell_boundaries(self, h_lines: List, v_lines: List, image_shape: Tuple[int, int] = None) -> List[Dict]:
        """
        根据水平线和垂直线的交叉点获取单元格边界

        Args:
            h_lines: 水平线列表
            v_lines: 垂直线列表
            image_shape: 图像尺寸 (height, width)，用于添加边界单元格

        Returns:
            单元格边界列表，每个元素为:
                {
                    'bbox': (x1, y1, x2, y2),  # 单元格边界框
                    'row': int,                 # 行索引
                    'col': int,                 # 列索引
                    'center': (cx, cy)         # 中心点坐标
                }
        """
        if not h_lines or not v_lines:
            logger.warning("水平线或垂直线为空，无法计算单元格边界")
            return []

        # 提取线位置
        h_positions = sorted(set([int(line[1]) for line in h_lines]))
        v_positions = sorted(set([int(line[0]) for line in v_lines]))

        # 添加图像边界
        if image_shape:
            height, width = image_shape[:2]
            # 如果第一条线不在顶部，添加顶部边界
            if h_positions[0] > 0:
                h_positions.insert(0, 0)
            # 如果最后一条线不在底部，添加底部边界
            if h_positions[-1] < height:
                h_positions.append(height)
            # 如果第一条线不在左侧，添加左侧边界
            if v_positions[0] > 0:
                v_positions.insert(0, 0)
            # 如果最后一条线不在右侧，添加右侧边界
            if v_positions[-1] < width:
                v_positions.append(width)

        cells = []

        # 计算交叉点形成的单元格
        for row_idx in range(len(h_positions) - 1):
            for col_idx in range(len(v_positions) - 1):
                y1 = h_positions[row_idx]
                y2 = h_positions[row_idx + 1]
                x1 = v_positions[col_idx]
                x2 = v_positions[col_idx + 1]

                # 过滤太小的单元格（可能是噪声）
                cell_width = x2 - x1
                cell_height = y2 - y1
                if cell_width < 10 or cell_height < 10:
                    continue

                cell = {
                    'bbox': (x1, y1, x2, y2),
                    'row': row_idx,
                    'col': col_idx,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'width': cell_width,
                    'height': cell_height
                }
                cells.append(cell)

        logger.info(f"从 {len(h_positions)} 条水平线和 {len(v_positions)} 条垂直线中检测到 {len(cells)} 个单元格")
        return cells

    def detect_table_structure(self, image_path: str = None, image: np.ndarray = None) -> Dict:
        """
        完整的表格结构检测

        Args:
            image_path: 图像文件路径（与image参数二选一）
            image: 输入图像（numpy数组）

        Returns:
            表格结构信息:
                {
                    'horizontal_lines': List[np.ndarray],
                    'vertical_lines': List[np.ndarray],
                    'cells': List[Dict],
                    'num_rows': int,
                    'num_cols': int
                }
        """
        # 获取图像
        if image is None:
            if image_path is None:
                raise ValueError("必须提供 image_path 或 image 参数")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

        # 检测水平线和垂直线
        h_lines = self.detect_horizontal_lines(image=image)
        v_lines = self.detect_vertical_lines(image=image)

        # 计算单元格边界
        cells = self.get_cell_boundaries(h_lines, v_lines, image.shape)

        # 计算行数和列数
        num_rows = len(set([int(line[1]) for line in h_lines])) + 1 if h_lines else 0
        num_cols = len(set([int(line[0]) for line in v_lines])) + 1 if v_lines else 0

        result = {
            'horizontal_lines': h_lines,
            'vertical_lines': v_lines,
            'cells': cells,
            'num_rows': num_rows,
            'num_cols': num_cols
        }

        if self.debug:
            self._visualize_table_structure(image, result)

        return result

    def _visualize_table_structure(self, image: np.ndarray, structure: Dict, save_path: str = 'debug_table_structure.png'):
        """可视化表格结构"""
        debug_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 绘制水平线（绿色）
        for line in structure['horizontal_lines']:
            cv2.line(debug_img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)

        # 绘制垂直线（蓝色）
        for line in structure['vertical_lines']:
            cv2.line(debug_img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)

        # 绘制单元格边界（红色）
        for cell in structure['cells']:
            x1, y1, x2, y2 = cell['bbox']
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # 添加文本信息
        info_text = f"Rows: {structure['num_rows']}, Cols: {structure['num_cols']}, Cells: {len(structure['cells'])}"
        cv2.putText(debug_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imwrite(save_path, debug_img)
        logger.info(f"表格结构可视化已保存到: {save_path}")

    def get_line_intersections(self, h_lines: List[np.ndarray], v_lines: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        获取水平和垂直线的交点

        Args:
            h_lines: 水平线列表
            v_lines: 垂直线列表

        Returns:
            交点坐标列表 [(x, y), ...]
        """
        intersections = []

        for h_line in h_lines:
            h_y = h_line[1]
            for v_line in v_lines:
                v_x = v_line[0]
                intersections.append((v_x, h_y))

        return sorted(intersections)


# 便捷函数
def detect_table_lines(image_path: str = None, image: np.ndarray = None,
                      kernel_size: int = 40, debug: bool = False) -> Dict:
    """
    快速检测表格线的便捷函数

    Args:
        image_path: 图像文件路径
        image: 输入图像
        kernel_size: 形态学操作核大小
        debug: 是否输出调试信息

    Returns:
        表格结构信息字典
    """
    config = {
        'kernel_size_h': kernel_size,
        'kernel_size_v': kernel_size,
        'debug': debug
    }
    detector = TableLineDetector(config)
    return detector.detect_table_structure(image_path, image)


if __name__ == '__main__':
    # 测试代码
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("用法: python table_line_detector.py <image_path> [kernel_size]")
        sys.exit(1)

    image_path = sys.argv[1]
    kernel_size = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    # 检测表格结构
    structure = detect_table_lines(image_path=image_path, kernel_size=kernel_size, debug=True)

    # 打印结果
    print(f"\n=== 表格结构检测结果 ===")
    print(f"水平线数量: {len(structure['horizontal_lines'])}")
    print(f"垂直线数量: {len(structure['vertical_lines'])}")
    print(f"表格行数: {structure['num_rows']}")
    print(f"表格列数: {structure['num_cols']}")
    print(f"单元格数量: {len(structure['cells'])}")

    # 打印前5个单元格信息
    print(f"\n前5个单元格:")
    for i, cell in enumerate(structure['cells'][:5]):
        print(f"  单元格 {i}: bbox={cell['bbox']}, row={cell['row']}, col={cell['col']}")
