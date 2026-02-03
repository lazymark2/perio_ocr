"""
增强型牙周图表OCR系统
支持多种数据类型的自动识别和区域划分：
- PD（探诊深度）：小数格式，B/L列
- BOP（出血指数）：数字0-4或符号+/-
- PI（菌斑指数）：罗马数字I°/II°/III°
- Furcation（根分叉病变）：罗马数字I°/II°/III°
- Mobility（松动度）：罗马数字I°/II°/III°

集成立即可做的后处理优化：
- 罗马数字自动修正（ll→II, |||→III, l→I）
- PD值合理性检查和异常值过滤
- 医学符号标准化

使用PaddleOCR，禁用自定义图像预处理（测试表明效果不佳）。
启用PaddleOCR内置的文档方向分类和矫正功能。
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from paddleocr import PaddleOCR
import logging
import re

# 导入后处理模块
from .post_processor import PostProcessor, EnhancedDataTypeExtractor

logger = logging.getLogger(__name__)


class RegionDetector:
    """区域检测器 - 自动划分不同数据类型的区域"""

    def __init__(
        self,
        enable_doc_orientation: bool = True,
        enable_unwarping: bool = False,
        enable_textline_orientation: bool = False
    ):
        """
        初始化区域检测器

        Args:
            enable_doc_orientation: 启用文档方向分类（推荐）
            enable_unwarping: 启用图像矫正（较慢）
            enable_textline_orientation: 启用文本行方向分类
        """
        self.ocr = PaddleOCR(
            lang='ch',
            enable_mkldnn=False,
            use_doc_orientation_classify=enable_doc_orientation,
            use_doc_unwarping=enable_unwarping,
            use_textline_orientation=enable_textline_orientation,
            text_det_limit_type='max',
            text_det_limit_side_len=960,
        )

    def detect_regions(self, image_path: str) -> Dict:
        """
        检测图像中的不同数据类型区域（支持上下半部分）

        Returns:
            {
                'separator_y': int,  # 上下半部分分隔线Y坐标
                'upper': {
                    'pd_region': {'bbox': [...], 'y_range': [...], 'rows': [...]},
                    'bop_region': {...},
                    'pi_region': {...},
                    'furcation_region': {...},
                    'mobility_region': {...},
                },
                'lower': {
                    'pd_region': {...},
                    'bop_region': {...},
                    'pi_region': {...},
                    'furcation_region': {...},
                    'mobility_region': {...},
                },
            }
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        h, w = img.shape[:2]

        # 使用OCR识别所有文本
        result = self.ocr.predict(img)

        if not result or len(result) == 0:
            return self._get_default_regions(w, h)

        ocr_data = result[0]
        rec_texts = ocr_data.get('rec_texts', [])
        rec_boxes = ocr_data.get('rec_boxes', [])

        # 检测上下半部分的分隔线
        separator_y = self._detect_separator(rec_texts, rec_boxes, h)

        # 分析文本内容，识别上下半部分的区域
        regions = self._analyze_layout_with_separator(
            rec_texts, rec_boxes, w, h, separator_y
        )

        # 添加分隔线信息
        regions['separator_y'] = separator_y

        return regions

    def _detect_separator(self, texts: List[str], boxes: np.ndarray, img_h: int) -> int:
        """
        检测上下半部分的分隔线（"牙位"行）

        Returns:
            分隔线Y坐标
        """
        # 查找包含"牙位"的文本
        separator_ys = []
        for idx, text in enumerate(texts):
            if '牙位' in text:
                box = boxes[idx].tolist() if idx < len(boxes) else [0, 0, 0, 0]
                y_center = int((box[1] + box[3]) // 2)
                separator_ys.append(y_center)

        if separator_ys:
            # 使用中间位置的"牙位"作为分隔线
            separator_ys.sort()
            return separator_ys[len(separator_ys) // 2]

        # 如果没有找到"牙位"，使用图像中点
        return img_h // 2

    def _analyze_layout_with_separator(self, texts: List[str], boxes: np.ndarray,
                                       img_w: int, img_h: int, separator_y: int) -> Dict:
        """
        分析表格布局，识别上下半部分的不同数据类型区域
        """
        # 初始化上下半部分的区域
        regions = {
            'upper': {
                'pd_region': {'label': 'PD', 'type': 'number', 'rows': []},
                'bop_region': {'label': 'BOP', 'type': 'symbol', 'rows': []},
                'pi_region': {'label': 'PI', 'type': 'roman', 'rows': []},
                'furcation_region': {'label': 'Furcation', 'type': 'roman', 'rows': []},
                'mobility_region': {'label': 'Mobility', 'type': 'roman', 'rows': []},
            },
            'lower': {
                'pd_region': {'label': 'PD', 'type': 'number', 'rows': []},
                'bop_region': {'label': 'BOP', 'type': 'symbol', 'rows': []},
                'pi_region': {'label': 'PI', 'type': 'roman', 'rows': []},
                'furcation_region': {'label': 'Furcation', 'type': 'roman', 'rows': []},
                'mobility_region': {'label': 'Mobility', 'type': 'roman', 'rows': []},
            },
        }

        # 分析每个文本行，识别数据类型并分配到上下半部分
        for idx, text in enumerate(texts):
            text_upper = text.upper()
            box = boxes[idx].tolist() if idx < len(boxes) else [0, 0, 0, 0]
            y_center = (box[1] + box[3]) // 2

            # 确定属于上半部分还是下半部分
            part = 'upper' if y_center < separator_y else 'lower'

            # 识别数据类型
            if 'PD' in text_upper or '探诊深度' in text:
                regions[part]['pd_region']['rows'].append(y_center)
            elif 'BOP' in text_upper or '出血指数' in text_upper or '出血' in text:
                regions[part]['bop_region']['rows'].append(y_center)
            elif 'PI' in text_upper or '菌斑指数' in text:
                regions[part]['pi_region']['rows'].append(y_center)
            elif '分叉' in text or 'FURCATION' in text_upper:
                regions[part]['furcation_region']['rows'].append(y_center)
            elif '松动' in text or 'MOBILITY' in text_upper:
                regions[part]['mobility_region']['rows'].append(y_center)

        # 如果没有检测到足够的区域，使用默认布局
        upper_has_data = any(any(r['rows']) for r in regions['upper'].values())
        lower_has_data = any(any(r['rows']) for r in regions['lower'].values())

        if not upper_has_data and not lower_has_data:
            return self._get_default_regions(img_w, img_h)

        # 为每个区域计算边界框
        for part in ['upper', 'lower']:
            for region_name, region_data in regions[part].items():
                if region_data['rows']:
                    min_y = min(region_data['rows'])
                    max_y = max(region_data['rows'])
                    row_height = (max_y - min_y) / max(1, len(region_data['rows']))

                    # 扩展Y范围以包含数据行（标题行下方约60像素）
                    #牙周图表中，数据行通常在标题行下方
                    data_row_height = 60  # 像素

                    region_data['bbox'] = [0, int(min_y - row_height), img_w, int(max_y + row_height + data_row_height)]
                    region_data['y_range'] = (int(min_y - row_height), int(max_y + row_height + data_row_height))

        return regions

    def _get_default_regions(self, w: int, h: int) -> Dict:
        """
        返回默认的区域划分（支持上下半部分）

        标准表格顺序（从上到下）：
        - 菌斑指数 (PI)
        - 松动度
        - 根分叉病变
        - 出血指数 (BOP)
        - 探诊深度 (PD)
        - --- 分隔线（牙位）---
        - 下半部分重复相同项目
        """

        separator_y = h // 2

        # 上半部分（上颌）
        upper_regions = {
            'pi_region': {
                'label': 'PI',
                'type': 'roman',
                'bbox': [0, int(h * 0.22), w, int(h * 0.26)],
                'y_range': (int(h * 0.22), int(h * 0.26))
            },
            'mobility_region': {
                'label': 'Mobility',
                'type': 'roman',
                'bbox': [0, int(h * 0.26), w, int(h * 0.29)],
                'y_range': (int(h * 0.26), int(h * 0.29))
            },
            'furcation_region': {
                'label': 'Furcation',
                'type': 'roman',
                'bbox': [0, int(h * 0.29), w, int(h * 0.34)],
                'y_range': (int(h * 0.29), int(h * 0.34))
            },
            'bop_region': {
                'label': 'BOP',
                'type': 'symbol',
                'bbox': [0, int(h * 0.34), w, int(h * 0.38)],
                'y_range': (int(h * 0.34), int(h * 0.38))
            },
            'pd_region': {
                'label': 'PD',
                'type': 'number',
                'bbox': [0, int(h * 0.38), w, int(h * 0.44)],
                'y_range': (int(h * 0.38), int(h * 0.44))
            },
        }

        # 下半部分（下颌）- 镜像上半部分的布局
        lower_regions = {}
        for region_name, region_data in upper_regions.items():
            # 计算下半部分对应的Y范围
            upper_y_min, upper_y_max = region_data['y_range']
            upper_height = upper_y_max - upper_y_min

            # 下半部分从分隔线开始向下延伸
            lower_y_min = separator_y + (upper_y_min - int(h * 0.22))
            lower_y_max = lower_y_min + upper_height

            lower_regions[region_name] = {
                'label': region_data['label'],
                'type': region_data['type'],
                'bbox': [0, lower_y_min, w, lower_y_max],
                'y_range': (lower_y_min, lower_y_max)
            }

        return {
            'separator_y': separator_y,
            'upper': upper_regions,
            'lower': lower_regions,
        }


# 使用增强型数据提取器（集成后处理优化）
extractor = EnhancedDataTypeExtractor()


class EnhancedPerioOCR:
    """增强型牙周图表OCR系统 - 集成后处理优化"""

    def __init__(self):
        self.region_detector = RegionDetector()
        self.extractor = extractor
        self.post_processor = PostProcessor()

        # 牙齿编号
        self.tooth_numbers = {
            'upper_right': ['18', '17', '16', '15', '14', '13', '12', '11'],
            'upper_left': ['21', '22', '23', '24', '25', '26', '27', '28'],
            'lower_left': ['31', '32', '33', '34', '35', '36', '37', '38'],
            'lower_right': ['41', '42', '43', '44', '45', '46', '47', '48']
        }

    def process(self, image_path: str, enable_preprocessing: bool = False) -> Dict:
        """
        处理牙周图表图像，提取所有数据类型

        Args:
            image_path: 图像路径
            enable_preprocessing: 是否启用图像预处理（默认False，效果不佳）

        Returns:
            {
                'teeth_data': {...},
                'regions': {...},
                'statistics': {...}
            }
        """
        # 图像预处理（可选，默认禁用）
        if enable_preprocessing:
            from .image_preprocessor import AdaptivePreprocessor

            logger.info("应用图像预处理...")
            preprocessor = AdaptivePreprocessor()

            # 创建临时预处理图像路径
            import os
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"preprocessed_{os.path.basename(image_path)}")

            try:
                processed_img = preprocessor.preprocess(image_path, temp_path)
                # 使用预处理后的图像进行OCR
                img = processed_img
                logger.info("图像预处理完成")
            except Exception as e:
                logger.warning(f"图像预处理失败，使用原始图像: {e}")
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"无法读取图像: {image_path}")
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")

        h, w = img.shape[:2]

        # 1. 检测不同数据类型的区域
        logger.info("检测数据类型区域...")
        regions = self.region_detector.detect_regions(image_path)

        # 2. 使用OCR识别整个图像
        logger.info("执行OCR识别...")
        ocr_result = self.region_detector.ocr.predict(img)

        if not ocr_result or len(ocr_result) == 0:
            return self._create_empty_result(w, h)

        ocr_data = ocr_result[0]
        rec_texts = ocr_data.get('rec_texts', [])
        rec_boxes = ocr_data.get('rec_boxes', [])

        # 3. 根据区域分类提取数据
        logger.info("提取结构化数据...")
        teeth_data = self._extract_by_region(rec_texts, rec_boxes, regions, w, h)

        # 4. 统计信息
        statistics = self._calculate_statistics(teeth_data)

        return {
            'teeth_data': teeth_data,
            'regions': regions,
            'statistics': statistics
        }

    def _extract_by_region(self, texts: List[str], boxes: np.ndarray,
                          regions: Dict, img_w: int, img_h: int) -> Dict:
        """根据区域分类提取数据 - 支持上下半部分"""

        teeth_data = {}
        separator_y = regions.get('separator_y', img_h // 2)

        for idx, text in enumerate(texts):
            if idx >= len(boxes):
                continue

            box = boxes[idx].tolist()
            y_center = (box[1] + box[3]) // 2
            x_center = (box[0] + box[2]) // 2

            # 确定所属区域
            region_result = self._get_region_type(y_center, regions)
            if not region_result:
                continue

            # 处理新旧两种返回格式
            if isinstance(region_result, tuple):
                part, region_type = region_result
            else:
                # 旧格式：仅返回区域类型
                region_type = region_result
                part = 'upper' if y_center < separator_y else 'lower'

            # 应用后处理修正文本
            cleaned_text = self.post_processor.clean_text(text)

            # 根据区域类型提取数据（使用增强型提取器）
            if region_type == 'pd_region':
                pd_value = self.extractor.extract_pd(cleaned_text)
                if pd_value is not None:
                    tooth_num, surface = self._locate_tooth(x_center, y_center, img_w, img_h, separator_y)
                    if tooth_num:
                        if tooth_num not in teeth_data:
                            teeth_data[tooth_num] = self._create_empty_tooth_data(tooth_num)
                        teeth_data[tooth_num]['pd'][surface] = pd_value

            elif region_type == 'bop_region':
                bop_value = self.extractor.extract_bop(cleaned_text)
                if bop_value is not None:
                    tooth_num, surface = self._locate_tooth(x_center, y_center, img_w, img_h, separator_y)
                    if tooth_num:
                        if tooth_num not in teeth_data:
                            teeth_data[tooth_num] = self._create_empty_tooth_data(tooth_num)
                        teeth_data[tooth_num]['bop'][surface] = bop_value

            elif region_type == 'pi_region':
                pi_value = self.post_processor.extract_roman_with_fix(cleaned_text)
                if pi_value is not None:
                    tooth_num, surface = self._locate_tooth(x_center, y_center, img_w, img_h, separator_y)
                    if tooth_num:
                        if tooth_num not in teeth_data:
                            teeth_data[tooth_num] = self._create_empty_tooth_data(tooth_num)
                        teeth_data[tooth_num]['pi'][surface] = pi_value

            elif region_type == 'furcation_region':
                furcation_value = self.post_processor.extract_roman_with_fix(cleaned_text)
                if furcation_value is not None:
                    tooth_num, _ = self._locate_tooth(x_center, y_center, img_w, img_h, separator_y)
                    if tooth_num and tooth_num[1] in '345678':  # 只有磨牙有分叉
                        if tooth_num not in teeth_data:
                            teeth_data[tooth_num] = self._create_empty_tooth_data(tooth_num)
                        # 分叉病变不分表面，只记录等级
                        teeth_data[tooth_num]['furcation'] = furcation_value

            elif region_type == 'mobility_region':
                mobility_value = self.post_processor.extract_roman_with_fix(cleaned_text)
                if mobility_value is not None:
                    tooth_num, _ = self._locate_tooth(x_center, y_center, img_w, img_h, separator_y)
                    if tooth_num:
                        if tooth_num not in teeth_data:
                            teeth_data[tooth_num] = self._create_empty_tooth_data(tooth_num)
                        # 松动度不分表面
                        teeth_data[tooth_num]['mobility'] = mobility_value

        return teeth_data

    def _get_region_type(self, y: int, regions: Dict) -> Optional[Tuple[str, str]]:
        """
        根据Y坐标确定所属区域和部分（上半部分/下半部分）

        Returns:
            (part, region_type) 或 None
            - part: 'upper' 或 'lower'
            - region_type: 'pd_region', 'bop_region', 等

        优先级策略：当区域重叠时，选择范围最小的区域（最精确匹配）
        """
        # 检查是否是新的区域结构（包含上下半部分）
        separator_y = regions.get('separator_y')
        if separator_y is not None:
            # 新结构：包含上下半部分
            part = 'upper' if y < separator_y else 'lower'

            if part not in regions:
                return None

            part_regions = regions[part]
        else:
            # 旧结构：单一区域（向后兼容）
            part = None
            part_regions = regions

        # 按照表格顺序定义区域优先级（从上到下）
        priority_order = [
            'pi_region',
            'mobility_region',
            'furcation_region',
            'bop_region',
            'pd_region'
        ]

        # 找出所有匹配的区域
        matching_regions = []
        for region_name in priority_order:
            if region_name not in part_regions:
                continue
            region_data = part_regions[region_name]

            if 'y_range' in region_data:
                y_min, y_max = region_data['y_range']
                if y_min <= y <= y_max:
                    range_size = y_max - y_min
                    matching_regions.append((region_name, range_size))
            elif 'bbox' in region_data:
                bbox = region_data['bbox']
                if bbox[1] <= y <= bbox[3]:
                    range_size = bbox[3] - bbox[1]
                    matching_regions.append((region_name, range_size))

        # 返回范围最小的匹配区域（最精确）
        if matching_regions:
            matching_regions.sort(key=lambda x: x[1])  # 按范围大小排序
            region_type = matching_regions[0][0]

            if part is not None:
                return (part, region_type)
            else:
                # 旧结构：返回区域类型（不包含部分）
                return region_type

        return None

    def _locate_tooth(self, x: int, y: int, w: int, h: int, separator_y: int = None) -> Tuple[Optional[str], str]:
        """
        根据坐标定位牙齿和测量面（支持上下半部分）

        Args:
            x: X坐标
            y: Y坐标
            w: 图像宽度
            h: 图像高度
            separator_y: 上下半部分分隔线Y坐标（如果为None，使用h//2）

        Returns:
            (tooth_num, surface) 或 (None, 'b')
        """
        mid_x = w // 2
        if separator_y is None:
            separator_y = h // 2

        # 确定象限 - 基于Y坐标相对于分隔线判断上颌/下颌
        # 上半部分 = 上颌牙齿 (11-28)
        # 下半部分 = 下颌牙齿 (31-48)
        if y < separator_y:
            # 上颌
            quadrant = 'upper_right' if x > mid_x else 'upper_left'
        else:
            # 下颌
            quadrant = 'lower_right' if x > mid_x else 'lower_left'

        # 确定牙齿（简化版，基于X坐标）
        teeth = self.tooth_numbers.get(quadrant, [])
        if not teeth:
            return None, 'b'

        quadrant_x_start = 0 if 'left' in quadrant else mid_x
        quadrant_width = mid_x if 'left' in quadrant else (w - mid_x)

        tooth_idx = int((x - quadrant_x_start) / quadrant_width * len(teeth))
        tooth_idx = max(0, min(tooth_idx, len(teeth) - 1))
        tooth_num = teeth[tooth_idx]

        # 确定测量面（简化为B/L）
        # 在实际表格中，可能需要根据列位置判断
        surface = 'b'  # 默认颊侧

        return tooth_num, surface

    def _create_empty_tooth_data(self, tooth_num: str) -> Dict:
        """创建空的牙齿数据结构"""
        return {
            'tooth': 1,
            'mobility': 0,
            'furcation': 0,
            'pd': {'b': 0, 'l': 0},
            'gm': {'b': 0, 'l': 0},
            'bop': {'b': 0, 'l': 0},
            'pi': {'b': 0, 'l': 0},
        }

    def _calculate_statistics(self, teeth_data: Dict) -> Dict:
        """计算统计信息"""
        stats = {
            'total_teeth': len(teeth_data),
            'teeth_with_pd': sum(1 for t in teeth_data.values() if any(t['pd'].values())),
            'teeth_with_bop': sum(1 for t in teeth_data.values() if any(t['bop'].values())),
            'teeth_with_pi': sum(1 for t in teeth_data.values() if any(t['pi'].values())),
            'teeth_with_furcation': sum(1 for t in teeth_data.values() if t['furcation'] > 0),
            'teeth_with_mobility': sum(1 for t in teeth_data.values() if t['mobility'] > 0),
        }
        return stats

    def _create_empty_result(self, w: int, h: int) -> Dict:
        """创建空结果"""
        return {
            'teeth_data': {},
            'regions': self.region_detector._get_default_regions(w, h),
            'statistics': {
                'total_teeth': 0,
                'teeth_with_pd': 0,
                'teeth_with_bop': 0,
                'teeth_with_pi': 0,
                'teeth_with_furcation': 0,
                'teeth_with_mobility': 0,
            }
        }


if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print("用法: python enhanced_periodontal_ocr.py <图像路径>")
        sys.exit(1)

    ocr = EnhancedPerioOCR()
    result = ocr.process(sys.argv[1])

    print(f"=== 牙周图表识别结果 ===")
    print(f"检测到 {result['statistics']['total_teeth']} 颗牙齿")
    print(f"  有PD数据: {result['statistics']['teeth_with_pd']} 颗")
    print(f"  有BOP数据: {result['statistics']['teeth_with_bop']} 颗")
    print(f"  有PI数据: {result['statistics']['teeth_with_pi']} 颗")
    print(f"  有分叉病变: {result['statistics']['teeth_with_furcation']} 颗")
    print(f"  有松动度: {result['statistics']['teeth_with_mobility']} 颗")

    # 显示部分结果
    for tooth_id, data in list(result['teeth_data'].items())[:5]:
        print(f"\n牙齿{tooth_id}:")
        if any(data['pd'].values()):
            print(f"  PD: {data['pd']}")
        if any(data['bop'].values()):
            print(f"  BOP: {data['bop']}")
        if any(data['pi'].values()):
            print(f"  PI: {data['pi']}")
        if data['mobility'] > 0:
            print(f"  松动度: {data['mobility']}°")
        if data['furcation'] > 0:
            print(f"  分叉病变: {data['furcation']}°")
