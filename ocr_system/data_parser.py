"""
数据解析模块 - 解析OCR结果并转换为标准数据格式
"""
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataParser:
    """牙周图表数据解析器"""

    # 有效值范围
    PD_RANGE = (1, 10)  # 探诊深度 1-10mm
    GM_RANGE = (-5, 5)  # 牙龈边缘 -5~5mm
    MOBILITY_RANGE = (0, 3)  # 活动度 0-3级
    FURCATION_RANGE = (0, 3)  # 分叉病变 0-3级

    def __init__(self, config: dict = None):
        self.config = config or {}

    def parse_number(self, text: str) -> Optional[int]:
        """解析数字字符串为整数"""
        if not text:
            return None

        # 移除非数字字符
        digits = re.sub(r'[^\d-]', '', text)

        if not digits:
            return None

        try:
            value = int(digits)
            return value
        except ValueError:
            return None

    def parse_float(self, text: str) -> Optional[float]:
        """解析数字字符串为浮点数"""
        if not text:
            return None

        # 移除非数字和点号字符
        cleaned = re.sub(r'[^\d.-]', '', text)

        if not cleaned:
            return None

        try:
            value = float(cleaned)
            return value
        except ValueError:
            return None

    def parse_bop_pi(self, text: str) -> int:
        """
        解析BOP/PI圈选标记
        识别: 1, O, ●, ✓, √, X
        """
        if not text:
            return 0

        text = text.upper()
        positive_marks = ['1', 'O', '●', '✓', '√', 'X', '●']

        for mark in positive_marks:
            if mark in text:
                return 1

        return 0

    def parse_mobility(self, text: str) -> int:
        """解析活动度值"""
        value = self.parse_number(text)
        if value is None:
            return 0

        if 0 <= value <= 3:
            return value
        return 0  # 默认值

    def parse_furcation(self, text: str) -> int:
        """解析分叉病变等级"""
        value = self.parse_number(text)
        if value is None:
            return 0

        if 0 <= value <= 3:
            return value
        return 0

    def validate_pd(self, value: any) -> int:
        """验证并规范化探诊深度"""
        if value is None:
            return 0

        try:
            v = int(value)
            if self.PD_RANGE[0] <= v <= self.PD_RANGE[1]:
                return v
            # 超出范围，截断到有效范围
            return max(self.PD_RANGE[0], min(self.PD_RANGE[1], v))
        except (ValueError, TypeError):
            return 0

    def validate_gm(self, value: any) -> int:
        """验证并规范化牙龈边缘值"""
        if value is None:
            return 0

        try:
            v = int(value)
            if self.GM_RANGE[0] <= v <= self.GM_RANGE[1]:
                return v
            return max(self.GM_RANGE[0], min(self.GM_RANGE[1], v))
        except (ValueError, TypeError):
            return 0

    def parse_ocr_result(self, ocr_data: Dict) -> Dict:
        """
        解析OCR结果为标准牙周数据结构
        """
        result = {
            'date_saved': datetime.now().strftime('%Y-%m-%d'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'url': 'www.periodontalchart-online.com',
            'copyright': 'www.perio-tools.com (2024)',
            'patient_last_name': '',
            'patient_first_name': '',
            'patient_dob': '',
            'initial_exam': '1',
            'reevaluation': '0',
            'input_reevaluation': 'Reevaluation',
            'clinician': ''
        }

        # 解析32颗牙齿的数据
        for tooth_num in self._get_all_teeth():
            tooth_data = self._parse_single_tooth(ocr_data.get(tooth_num, {}))
            result.update(tooth_data)

        return result

    def _get_all_teeth(self) -> List[str]:
        """获取所有32颗牙齿编号"""
        return [
            # 右上
            '18', '17', '16', '15', '14', '13', '12', '11',
            # 左上
            '21', '22', '23', '24', '25', '26', '27', '28',
            # 左下
            '31', '32', '33', '34', '35', '36', '37', '38',
            # 右下
            '41', '42', '43', '44', '45', '46', '47', '48'
        ]

    def _get_surfaces(self, tooth_num: str) -> List[str]:
        """获取牙齿的测量面"""
        # 前牙 (单号): db, b, mb, dl, l, ml
        # 后牙 (双号): db, b, mb, dp, p, mp
        tooth_int = int(tooth_num[1])  # 取第二位数字

        if tooth_int % 2 == 1:  # 单数（实际是前牙区，但编号不同）
            return ['db', 'b', 'mb', 'dl', 'l', 'ml']
        else:  # 双数（后牙区）
            return ['db', 'b', 'mb', 'dp', 'p', 'mp']

    def _is_molar(self, tooth_num: str) -> bool:
        """判断是否为磨牙（需要分叉病变检测）"""
        tooth_int = int(tooth_num[1])
        # 7和8是磨牙，3和4是磨牙
        return tooth_int in [3, 4, 7, 8]

    def _parse_single_tooth(self, tooth_data: Dict) -> Dict:
        """解析单颗牙齿的数据"""
        result = {}

        tooth_num = tooth_data.get('tooth_num', '')
        if not tooth_num:
            return result

        # 基础字段
        result[f'tooth_{tooth_num}'] = tooth_data.get('tooth', 1)
        result[f'mobility_{tooth_num}'] = self.parse_mobility(tooth_data.get('mobility', ''))
        result[f'implant_{tooth_num}'] = tooth_data.get('implant', 0)

        surfaces = self._get_surfaces(tooth_num)

        # 探诊深度 PD
        pd_data = tooth_data.get('pd', {})
        for surface in surfaces:
            value = self.validate_pd(pd_data.get(surface, 0))
            result[f'pd_{tooth_num}_{surface}'] = value

        # 牙龈边缘 GM
        gm_data = tooth_data.get('gm', {})
        for surface in surfaces:
            value = self.validate_gm(gm_data.get(surface, 0))
            result[f'gm_{tooth_num}_{surface}'] = value

        # 探诊出血 BOP
        bop_data = tooth_data.get('bop', {})
        for surface in surfaces:
            result[f'bop_{tooth_num}_{surface}'] = self.parse_bop_pi(bop_data.get(surface, '0'))

        # 菌斑指数 PI
        pi_data = tooth_data.get('pi', {})
        for surface in surfaces:
            result[f'pi_{tooth_num}_{surface}'] = self.parse_bop_pi(pi_data.get(surface, '0'))

        # 分叉病变（仅磨牙）
        if self._is_molar(tooth_num):
            furcation_data = tooth_data.get('furcation', {})
            if 'dp' in surfaces:  # 上下颌磨牙
                for surface in ['b', 'dp', 'mp']:
                    result[f'furcation_{tooth_num}_{surface}'] = self.parse_furcation(
                        furcation_data.get(surface, 0)
                    )
            else:  # 下颌磨牙
                for surface in ['b', 'l']:
                    result[f'furcation_{tooth_num}_{surface}'] = self.parse_furcation(
                        furcation_data.get(surface, 0)
                    )

        # 备注
        result[f'note_{tooth_num}'] = tooth_data.get('note', '')

        return result

    def merge_with_template(self, ocr_result: Dict, template_path: str = None) -> Dict:
        """
        将OCR结果与模板数据合并
        如果OCR无法识别的字段，使用默认值
        """
        from .data_mapper import DataMapper
        import json

        # 加载模板数据
        if template_path:
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    template = json.load(f)
            except Exception as e:
                logger.error(f"加载模板失败: {e}")
                template = {}
        else:
            template = DataMapper.create_default_template()

        # 合并数据
        merged = template.copy()
        merged.update(ocr_result)

        return merged


if __name__ == '__main__':
    # 测试
    parser = DataParser()

    test_data = {
        'tooth_num': '16',
        'tooth': 1,
        'mobility': '2',
        'pd': {'db': '5', 'b': '7', 'mb': '4', 'dp': '6', 'p': '5', 'mp': '6'},
        'gm': {'db': '0', 'b': '0', 'mb': '0', 'dp': '0', 'p': '0', 'mp': '0'},
        'bop': {'db': '1', 'b': '1', 'mb': '1', 'dp': '1', 'p': '1', 'mp': '1'},
        'pi': {'db': '0', 'b': '0', 'mb': '0', 'dp': '0', 'p': '0', 'mp': '0'}
    }

    result = parser._parse_single_tooth(test_data)
    for key, value in result.items():
        print(f"{key}: {value}")
