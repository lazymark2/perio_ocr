"""
图像预处理模块 - 提高OCR识别一致性

功能：
- 去噪（Denoising）
- 对比度增强（CLAHE）
- 锐化（Sharpening）
- 二值化（Binarization）
- 灰度转换
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """图像预处理器 - 提高OCR识别准确性"""

    def __init__(self,
                 enable_denoise: bool = True,
                 enhance_contrast: bool = True,
                 enable_sharpen: bool = True,
                 binarize: bool = False,
                 method: str = 'otsu'):
        """
        初始化图像预处理器

        Args:
            enable_denoise: 是否启用去噪
            enhance_contrast: 是否增强对比度（CLAHE）
            enable_sharpen: 是否锐化图像
            binarize: 是否进行二值化（输出单通道图像）
            method: 二值化方法 ('otsu', 'adaptive', 'sauvola')
        """
        self.enable_denoise = enable_denoise
        self.enhance_contrast = enhance_contrast
        self.enable_sharpen = enable_sharpen
        self.binarize = binarize
        self.method = method

    def preprocess(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        对图像进行预处理

        Args:
            image_path: 输入图像路径
            output_path: 可选的输出图像路径（用于调试）

        Returns:
            预处理后的图像（numpy数组）
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        h, w = img.shape[:2]
        logger.debug(f"原始图像尺寸: {w}x{h}")

        # 1. 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logger.debug("转换为灰度图")

        # 2. 去噪
        if self.enable_denoise:
            gray = self._denoise(gray)
            logger.debug("应用去噪")

        # 3. 对比度增强
        if self.enhance_contrast:
            gray = self._enhance_contrast(gray)
            logger.debug("增强对比度")

        # 4. 锐化
        if self.enable_sharpen:
            gray = self._sharpen(gray)
            logger.debug("应用锐化")

        # 5. 二值化（可选）
        if self.binarize:
            processed = self._binarize(gray)
        else:
            # 如果不二值化，转换回BGR格式（保持3通道）
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 保存预处理后的图像（用于调试）
        if output_path:
            cv2.imwrite(output_path, processed)
            logger.debug(f"保存预处理图像到: {output_path}")

        return processed

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        """
        去噪处理

        使用fastNlMeansDenoising算法，该方法在保持细节的同时有效去噪
        """
        # 参数说明：
        # h: 决定过滤强度。h值越大，去噪效果越明显，但细节损失也越多
        #    - 10: 轻度去噪，保留更多细节
        #    - 15: 中等去噪（推荐）
        # templateWindowSize: 奇数，推荐值7
        # searchWindowSize: 奇数，推荐值21

        denoised = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=15,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        对比度增强

        使用CLAHE（限制对比度自适应直方图均衡化）
        该方法可以改善图像的对比度，同时避免过度增强噪声
        """
        # 创建CLAHE对象
        # clipLimit: 对比度限制，推荐值2.0-4.0
        # tileGridSize: 网格大小，推荐(8,8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        """
        锐化处理

        使用USM（Unsharp Masking）算法锐化图像，增强边缘
        """
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)

        # USM锐化：原始图像 - 模糊图像) * 系数 + 原始图像
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        # 确保值在有效范围内
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        二值化处理

        Args:
            gray: 灰度图像

        Returns:
            二值化图像（单通道，0或255）
        """
        if self.method == 'otsu':
            # Otsu阈值法（全局阈值）
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        elif self.method == 'adaptive':
            # 自适应阈值法（局部阈值）
            # 适合光照不均匀的图像
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # blockSize: 邻域块大小
                2     # C: 从均值减去的常数
            )

        elif self.method == 'sauvola':
            # Sauvola阈值法（适合文本图像）
            # 需要转换为浮点型
            gray_float = gray.astype(np.float64)
            binary = cv2.ximgproc.niBlackThreshold(
                gray_float,
                255,
                cv2.THRESH_BINARY_INV,
                41,  # blockSize
                0    # k
            )
            binary = binary.astype(np.uint8)
            binary = cv2.bitwise_not(binary)  # 反转颜色

        else:
            # 默认使用Otsu
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return binary

    def preprocess_batch(self, image_paths: list, output_dir: str = None) -> dict:
        """
        批量预处理图像

        Args:
            image_paths: 图像路径列表
            output_dir: 输出目录（如果为None，则不保存）

        Returns:
            {image_path: processed_image} 字典
        """
        import os

        results = {}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for img_path in image_paths:
            try:
                # 生成输出路径
                out_path = None
                if output_dir:
                    import os.path
                    filename = os.path.basename(img_path)
                    name, ext = os.path.splitext(filename)
                    out_path = os.path.join(output_dir, f"{name}_preprocessed{ext}")

                # 预处理
                processed = self.preprocess(img_path, out_path)
                results[img_path] = processed

                logger.info(f"预处理完成: {img_path}")

            except Exception as e:
                logger.error(f"预处理失败 {img_path}: {e}")
                results[img_path] = None

        return results


class AdaptivePreprocessor(ImagePreprocessor):
    """
    自适应预处理器 - 根据图像质量自动调整预处理参数

    对于质量较好的图像：使用轻度处理
    对于质量较差的图像：使用重度处理
    """

    def __init__(self):
        super().__init__()
        self.quality_threshold = 50  # 图像质量阈值

    def assess_image_quality(self, image_path: str) -> float:
        """
        评估图像质量

        基于以下指标：
        1. 拉普拉斯方差（清晰度）
        2. 对比度
        3. 亮度均匀性

        Returns:
            质量分数 (0-100)，越高表示质量越好
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0

        # 1. 清晰度（拉普拉斯方差）
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

        # 2. 对比度（标准差）
        contrast = img.std()

        # 3. 综合评分
        # 清晰度归一化（假设0-10000为有效范围）
        clarity_score = min(laplacian_var / 100, 100)
        contrast_score = min(contrast / 127, 100)

        quality = (clarity_score * 0.7 + contrast_score * 0.3)

        return quality

    def preprocess(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        自适应预处理 - 根据图像质量选择处理强度
        """
        # 评估图像质量
        quality = self.assess_image_quality(image_path)
        logger.info(f"图像质量评分: {quality:.1f}")

        # 根据质量调整参数
        if quality < 40:
            # 低质量图像：重度处理
            self.enable_denoise = True
            self.enhance_contrast = True
            self.enable_sharpen = True
            logger.debug("使用重度预处理")
        elif quality < 70:
            # 中等质量：中度处理
            self.enable_denoise = True
            self.enhance_contrast = True
            self.enable_sharpen = False
            logger.debug("使用中度预处理")
        else:
            # 高质量图像：轻度处理
            self.enable_denoise = False
            self.enhance_contrast = True
            self.enable_sharpen = False
            logger.debug("使用轻度预处理")

        # 调用父类的预处理方法
        return super().preprocess(image_path, output_path)


def create_comparison_image(original_path: str, processed_path: str, output_path: str):
    """
    创建对比图像 - 并排显示原始图像和预处理后的图像

    Args:
        original_path: 原始图像路径
        processed_path: 预处理后的图像路径
        output_path: 输出对比图像路径
    """
    # 读取图像
    original = cv2.imread(original_path)
    processed = cv2.imread(processed_path)

    if original is None or processed is None:
        raise ValueError("无法读取图像")

    # 调整尺寸使其相同
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]

    max_h = max(h1, h2)
    max_w = max(w1, w2)

    if h1 != max_h or w1 != max_w:
        original = cv2.resize(original, (max_w, max_h))
    if h2 != max_h or w2 != max_w:
        processed = cv2.resize(processed, (max_w, max_h))

    # 如果是灰度图，转换为BGR
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # 水平拼接
    comparison = np.hstack([original, processed])

    # 添加标签
    cv2.putText(comparison, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Processed", (max_w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 保存
    cv2.imwrite(output_path, comparison)
    logger.info(f"对比图像已保存到: {output_path}")


# 测试代码
if __name__ == '__main__':
    import sys
    import os

    # 设置日志
    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("用法: python image_preprocessor.py <图像路径>")
        sys.exit(1)

    image_path = sys.argv[1]

    # 创建预处理器
    preprocessor = ImagePreprocessor(
        enable_denoise=True,
        enhance_contrast=True,
        enable_sharpen=True,
        binarize=False  # 暂不二值化，保持3通道
    )

    # 预处理图像
    output_path = os.path.splitext(image_path)[0] + "_preprocessed.jpg"
    processed = preprocessor.preprocess(image_path, output_path)

    print(f"预处理完成！")
    print(f"原始图像: {image_path}")
    print(f"处理后图像: {output_path}")
    print(f"处理后尺寸: {processed.shape}")

    # 创建对比图像
    comparison_path = os.path.splitext(image_path)[0] + "_comparison.jpg"
    create_comparison_image(image_path, output_path, comparison_path)
    print(f"对比图像: {comparison_path}")
