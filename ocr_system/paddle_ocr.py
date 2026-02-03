"""
PaddleOCR调用模块 - 手写数字和拥挤文本识别
针对手写数字、拥挤文本、表格重叠场景优化
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from paddleocr import PaddleOCR
import logging

logger = logging.getLogger(__name__)


class PerioOCRSystem:
    """牙周图表OCR识别系统 - 优化版"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.use_gpu = self.config.get('use_gpu', False)  # CPU模式
        self.lang = self.config.get('lang', 'ch')  # 中文牙周图表

        # 初始化PaddleOCR - 使用PP-OCRv5 CPU版本
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=self.lang,
            # CPU模式会自动检测，无需指定use_gpu
        )

        logger.info(f"PaddleOCR初始化完成 (Mode: CPU, Lang: {self.lang})")

    def preprocess_for_crowded_text(self, image: np.ndarray) -> np.ndarray:
        """
        针对拥挤文本的预处理
        - 增加对比度
        - 减少噪声
        - 增强笔画清晰度
        """
        if image is None or image.size == 0:
            return image

        # 灰度化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 自适应直方图均衡化（限制对比度）
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # 中值滤波去噪（保留边缘）
        denoised = cv2.medianBlur(enhanced, 3)

        # 形态学处理增强笔画
        kernel = np.ones((2, 2), np.uint8)
        # 闭运算填充笔画内部空洞
        closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        return closed

    def handle_overlapping_text(self, ocr_result: List) -> List:
        """
        处理表格线与文字重叠的情况
        - 过滤掉表格线误检
        - 合并相邻的检测框
        """
        if not ocr_result:
            return []

        filtered_results = []
        for line in ocr_result:
            box = line[0]  # 文本框坐标
            text = line[1][0]  # 识别文本
            score = line[1][1]  # 置信度

            # 过滤太小的检测框（可能是噪点）
            width = abs(box[2][0] - box[0][0])
            height = abs(box[2][1] - box[0][1])

            if width < 5 or height < 5:
                continue

            # 过滤太长的水平线（可能是表格线）
            if width > height * 10:
                continue

            filtered_results.append(line)

        return filtered_results

    def preprocess_cell(self, image: np.ndarray) -> np.ndarray:
        """单元格图像预处理"""
        if image is None or image.size == 0:
            return image

        # 灰度化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # 二值化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return binary

    def recognize_number(self, image: np.ndarray) -> Optional[str]:
        """识别单元格中的数字"""
        if image is None or image.size == 0:
            return None

        # 预处理
        processed = self.preprocess_cell(image)

        # 裁剪边缘空白
        coords = cv2.findNonZero(processed)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        # 保留一定边距
        margin = 2
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(processed.shape[1] - x, w + 2 * margin)
        h = min(processed.shape[0] - y, h + 2 * margin)
        cropped = processed[y:y+h, x:x+w]

        # 调整尺寸以适应OCR
        resized = cv2.resize(cropped, (64, 64))

        # 保存临时图像供调试
        # cv2.imwrite('debug_digit.png', resized)

        # OCR识别
        try:
            result = self.ocr.ocr(resized, cls=True)
            if result and result[0]:
                texts = [line[1][0] for line in result[0]]
                text = ''.join(texts)
                # 提取数字
                digits = ''.join(c for c in text if c.isdigit() or c == '-')
                return digits if digits else None
        except Exception as e:
            logger.error(f"数字识别错误: {e}")

        return None

    def recognize_circle_mark(self, image: np.ndarray) -> int:
        """
        识别BOP/PI的圈选标记
        返回: 0 或 1
        """
        if image is None or image.size == 0:
            return 0

        # 灰度化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 检测圆形轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # 计算圆度
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # 圆形或近似圆形（圆度>0.6）
            if circularity > 0.6 and area > 50:
                return 1

        # 检测勾选标记
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=10, minLineLength=5, maxGap=3)
        if lines is not None:
            return 1

        return 0

    def recognize_cell(self, image: np.ndarray, cell_type: str = 'number') -> any:
        """
        识别单元格内容
        cell_type: 'number'(PD/GM), 'circle'(BOP/PI), 'mobility'(活动度), 'furcation'(分叉)
        """
        if cell_type in ['number', 'mobility', 'furcation']:
            result = self.recognize_number(image)
            if result:
                try:
                    return int(result)
                except ValueError:
                    return result
            return None
        elif cell_type == 'circle':
            return self.recognize_circle_mark(image)
        else:
            return self.recognize_number(image)

    def recognize_tooth_area(self, image: np.ndarray) -> Dict[str, any]:
        """
        识别整个牙齿区域的多个测量值
        返回包含PD, GM, BOP, PI等数据的字典
        """
        result = {
            'pd': {},      # 探诊深度
            'gm': {},      # 牙龈边缘
            'bop': {},     # 探诊出血
            'pi': {},      # 菌斑指数
            'mobility': None,
            'furcation': {}
        }

        # 识别PD值
        # 识别GM值
        # 识别BOP/PI圈选

        return result

    def batch_recognize(self, images: List[np.ndarray]) -> List[Dict]:
        """批量识别多张图像"""
        results = []
        for idx, img in enumerate(images):
            logger.info(f"正在处理第 {idx + 1}/{len(images)} 张图像")
            result = self.recognize_tooth_area(img)
            results.append(result)
        return results


if __name__ == '__main__':
    # 测试代码
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        ocr = PerioOCRSystem()
        result = ocr.recognize_tooth_area(image)
        print(result)
