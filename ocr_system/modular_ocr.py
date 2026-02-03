"""
模块化OCR处理器 - 使用PaddleOCR独立模块

模块说明：
- DocImgOrientationClassification: 文档图像方向分类（0°/90°/180°/270°）
- TextImageUnwarping: 文本图像矫正（处理弯曲、扭曲）
- TextLineOrientationClassification: 文本行方向分类
- TextDetection: 文本检测
- TextRecognition: 文本识别
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModularOCR:
    """
    模块化OCR处理器

    使用PaddleOCR的独立模块，可以灵活控制每个处理步骤
    """

    def __init__(
        self,
        *,
        # 可选预处理模块
        enable_doc_orientation: bool = True,     # 文档方向分类
        enable_unwarping: bool = False,           # 图像矫正（较慢）
        enable_textline_orientation: bool = False, # 文本行方向分类
        # 必需模块
        enable_detection: bool = True,            # 文本检测
        enable_recognition: bool = True,          # 文本识别
        # 其他参数
        device: str = 'cpu',
        **kwargs
    ):
        """
        初始化模块化OCR处理器

        Args:
            enable_doc_orientation: 启用文档方向分类
            enable_unwarping: 启用图像矫正（处理弯曲、扭曲）
            enable_textline_orientation: 启用文本行方向分类
            enable_detection: 启用文本检测
            enable_recognition: 启用文本识别
            device: 设备 ('cpu' 或 'gpu')
        """
        self.enable_doc_orientation = enable_doc_orientation
        self.enable_unwarping = enable_unwarping
        self.enable_textline_orientation = enable_textline_orientation
        self.enable_detection = enable_detection
        self.enable_recognition = enable_recognition

        # 初始化模块
        self._init_modules(device, **kwargs)

    def _init_modules(self, device: str, **kwargs):
        """初始化所有启用的模块"""

        # 可选预处理模块
        if self.enable_doc_orientation:
            from paddleocr import DocImgOrientationClassification
            logger.info("初始化文档方向分类模块...")
            self.doc_orientation_cls = DocImgOrientationClassification(
                device=device,
                **kwargs
            )
        else:
            self.doc_orientation_cls = None

        if self.enable_unwarping:
            from paddleocr import TextImageUnwarping
            logger.info("初始化图像矫正模块...")
            self.unwarping = TextImageUnwarping(
                device=device,
                **kwargs
            )
        else:
            self.unwarping = None

        if self.enable_textline_orientation:
            from paddleocr import TextLineOrientationClassification
            logger.info("初始化文本行方向分类模块...")
            self.textline_orientation_cls = TextLineOrientationClassification(
                device=device,
                **kwargs
            )
        else:
            self.textline_orientation_cls = None

        # 必需模块
        if self.enable_detection:
            from paddleocr import TextDetection
            logger.info("初始化文本检测模块...")
            self.text_detector = TextDetection(
                device=device,
                **kwargs
            )
        else:
            self.text_detector = None

        if self.enable_recognition:
            from paddleocr import TextRecognition
            logger.info("初始化文本识别模块...")
            self.text_recognizer = TextRecognition(
                device=device,
                **kwargs
            )
        else:
            self.text_recognizer = None

    def process(
        self,
        image_path: str,
        return_raw_results: bool = False
    ) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
        """
        处理图像，执行OCR识别

        Args:
            image_path: 输入图像路径
            return_raw_results: 是否返回原始结果

        Returns:
            (texts, boxes, raw_results): 文本列表、边界框数组、原始结果字典
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        logger.info(f"处理图像: {image_path}, 尺寸: {img.shape}")

        raw_results = {
            'original_shape': img.shape,
            'preprocessing_steps': [],
            'orientation': None,
            'unwarped': False,
            'textline_orientation': None
        }

        processed_img = img.copy()

        # 步骤1: 文档方向分类（可选）
        if self.doc_orientation_cls:
            logger.info("执行文档方向分类...")
            results = self.doc_orientation_cls.predict(input=processed_img)
            if results and len(results) > 0:
                result = results[0]
                orientation = result.get('class_name', '0')
                confidence = result.get('score', 0)
                logger.info(f"文档方向: {orientation}° (置信度: {confidence:.2f})")
                raw_results['orientation'] = orientation

                # 根据方向旋转图像
                if orientation != '0':
                    processed_img = self._rotate_image(processed_img, int(orientation))
                    raw_results['preprocessing_steps'].append(f'rotate_{orientation}')

        # 步骤2: 图像矫正（可选）- 处理弯曲、扭曲
        if self.unwarping:
            logger.info("执行图像矫正...")
            results = self.unwarping.predict(input=processed_img)
            if results and len(results) > 0:
                result = results[0]
                # 获取矫正后的图像
                if 'unwarped_img' in result:
                    processed_img = result['unwarped_img']
                    raw_results['unwarped'] = True
                    raw_results['preprocessing_steps'].append('unwarping')
                    logger.info("图像矫正完成")
                elif 'unwarped_image' in result:
                    processed_img = result['unwarped_image']
                    raw_results['unwarped'] = True
                    raw_results['preprocessing_steps'].append('unwarping')
                    logger.info("图像矫正完成")

        # 步骤3: 文本行方向分类（可选）
        if self.textline_orientation_cls:
            logger.info("执行文本行方向分类...")
            results = self.textline_orientation_cls.predict(input=processed_img)
            if results and len(results) > 0:
                result = results[0]
                textline_ori = result.get('class_name', 'horizontal')
                logger.info(f"文本行方向: {textline_ori}")
                raw_results['textline_orientation'] = textline_ori

        # 步骤4: 文本检测（必需）
        if not self.text_detector:
            raise RuntimeError("文本检测模块未启用")

        logger.info("执行文本检测...")
        det_results = self.text_detector.predict(input=processed_img)

        if not det_results or len(det_results) == 0:
            logger.warning("未检测到任何文本")
            return [], np.array([]), raw_results

        # 提取检测框
        boxes = []
        for result in det_results:
            if 'bbox' in result:
                boxes.append(result['bbox'])
            elif 'coordinate' in result:
                boxes.append(result['coordinate'])

        if not boxes:
            logger.warning("未检测到任何文本框")
            return [], np.array([]), raw_results

        boxes = np.array(boxes)
        logger.info(f"检测到 {len(boxes)} 个文本框")

        raw_results['detection_results'] = det_results

        # 步骤5: 文本识别（必需）
        if not self.text_recognizer:
            raw_results['recognition_results'] = None
            return [], boxes, raw_results

        logger.info("执行文本识别...")
        # 识别每个文本框
        texts = []
        rec_results = []

        for i, box in enumerate(boxes):
            # 裁剪文本区域
            cropped_img = self._crop_text_region(processed_img, box)

            # 识别
            results = self.text_recognizer.predict(input=cropped_img)

            if results and len(results) > 0:
                result = results[0]
                text = result.get('text', '')
                confidence = result.get('rec_scores', [0])[0] if 'rec_scores' in result else 0

                texts.append(text)
                rec_results.append({
                    'text': text,
                    'confidence': confidence,
                    'box': box.tolist() if hasattr(box, 'tolist') else box
                })

            else:
                texts.append('')
                rec_results.append({
                    'text': '',
                    'confidence': 0,
                    'box': box.tolist() if hasattr(box, 'tolist') else box
                })

        logger.info(f"识别完成，共 {len(texts)} 个文本")

        raw_results['recognition_results'] = rec_results

        if return_raw_results:
            return texts, boxes, raw_results
        else:
            return texts, boxes, {}

    def _rotate_image(self, img: np.ndarray, angle: int) -> np.ndarray:
        """旋转图像"""
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return img

    def _crop_text_region(self, img: np.ndarray, box: np.ndarray) -> np.ndarray:
        """裁剪文本区域"""
        # 获取边界框的坐标
        x_coords = box[:, 0] if len(box.shape) > 1 else [box[0], box[2]]
        y_coords = box[:, 1] if len(box.shape) > 1 else [box[1], box[3]]

        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))

        # 确保坐标在图像范围内
        h, w = img.shape[:2]
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)

        # 裁剪
        cropped = img[y_min:y_max, x_min:x_max]

        return cropped

    def close(self):
        """关闭所有模块，释放资源"""
        modules = [
            self.doc_orientation_cls,
            self.unwarping,
            self.textline_orientation_cls,
            self.text_detector,
            self.text_recognizer
        ]

        for module in modules:
            if module is not None and hasattr(module, 'close'):
                try:
                    module.close()
                except Exception as e:
                    logger.warning(f"关闭模块时出错: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数
def ocr_with_modular_pipeline(
    image_path: str,
    *,
    enable_doc_orientation: bool = True,
    enable_unwarping: bool = False,
    enable_textline_orientation: bool = False,
    device: str = 'cpu'
) -> Tuple[List[str], np.ndarray]:
    """
    便捷的OCR函数

    Args:
        image_path: 图像路径
        enable_doc_orientation: 启用文档方向分类
        enable_unwarping: 启用图像矫正
        enable_textline_orientation: 启用文本行方向分类
        device: 设备

    Returns:
        (texts, boxes): 文本列表和边界框数组
    """
    with ModularOCR(
        enable_doc_orientation=enable_doc_orientation,
        enable_unwarping=enable_unwarping,
        enable_textline_orientation=enable_textline_orientation,
        device=device
    ) as ocr:
        texts, boxes, _ = ocr.process(image_path)
        return texts, boxes


# 测试代码
if __name__ == '__main__':
    import sys
    import os

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("用法: python modular_ocr.py <图像路径>")
        sys.exit(1)

    image_path = sys.argv[1]

    # 测试模块化OCR
    with ModularOCR(
        enable_doc_orientation=True,
        enable_unwarping=False,  # 较慢，测试时可禁用
        enable_textline_orientation=False
    ) as ocr:
        texts, boxes, raw_results = ocr.process(image_path, return_raw_results=True)

        print(f"\n检测到 {len(texts)} 个文本:")
        for i, (text, box) in enumerate(zip(texts, boxes)):
            print(f"{i+1}. {text}")

        print(f"\n原始结果: {raw_results}")
