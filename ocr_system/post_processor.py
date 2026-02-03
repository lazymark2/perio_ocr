"""
后处理优化模块 - 修正常见识别错误
"""
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PostProcessor:
    """后处理优化器 - 修正OCR识别错误"""

    # 罗马数字映射表
    ROMAN_MAP = {
        # 字符替换
        'l': 'I', '|': 'I', '1': 'I', '!': 'I',
        'll': 'II', '||': 'II', '11': 'II',
        'lll': 'III', '|||': 'III', '111': 'III',
        'lV': 'IV', '1V': 'IV',
        'V': 'V', 'V1': 'VI',
        'VII': 'VII', 'VIII': 'VIII',
        'i': 'I', 'ii': 'II', 'iii': 'III',
        # 度数符号
        'o': '°', '0': '°', '*': '°'
    }

    # 医学符号映射
    MEDICAL_SYMBOLS = {
        '+': '+',  # 出血/溢脓
        '-': '-',  # 无出血/无溢脓
        '×': '×', 'x': '×', 'X': '×',  # 菌斑/否定
        '√': '✓', 'v': '✓', '∨': '✓',  # 确认
    }

    @staticmethod
    def fix_roman_numerals(text: str) -> str:
        """
        修正罗马数字识别错误

        Args:
            text: 原始OCR文本

        Returns:
            修正后的文本
        """
        if not text:
            return text

        # 移除空格
        text = text.replace(' ', '')

        # 逐个替换
        for wrong, correct in PostProcessor.ROMAN_MAP.items():
            text = text.replace(wrong, correct)

        # 处理组合情况 (如 "I I" -> "II")
        text = text.replace('I I', 'II')
        text = text.replace('II I', 'III')
        text = text.replace('I II', 'III')

        # 确保度数符号格式一致
        if re.match(r'^[IVX]+°?$', text):
            if '°' not in text:
                text = text + '°'

        return text

    @staticmethod
    def fix_punctuation(text: str) -> str:
        """修正标点符号"""
        # 冒号替换
        text = text.replace(':', '.').replace('：', '.')
        # 分号替换
        text = text.replace(';', ',').replace('；', ',')
        return text

    @staticmethod
    def fix_medical_symbols(text: str) -> str:
        """修正医学符号"""
        for wrong, correct in PostProcessor.MEDICAL_SYMBOLS.items():
            text = text.replace(wrong, correct)
        return text

    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本，综合应用所有修正"""
        if not text:
            return text

        # 应用所有修正
        text = PostProcessor.fix_roman_numerals(text)
        text = PostProcessor.fix_punctuation(text)
        text = PostProcessor.fix_medical_symbols(text)

        return text

    @staticmethod
    def validate_pd_value(value: float, neighbors: list = None) -> Optional[float]:
        """
        验证并修正PD值

        Args:
            value: 原始PD值
            neighbors: 相邻PD值列表，用于上下文验证

        Returns:
            修正后的PD值，如果无效返回None
        """
        # 基本范围检查
        if value < 0 or value > 15:
            # 如果有相邻值，使用平均值
            if neighbors and len(neighbors) > 0:
                avg = sum(neighbors) / len(neighbors)
                if 0 <= avg <= 15:
                    return round(avg, 1)
            return None

        # 检查异常精度（如2.23322222）
        str_val = str(value)
        if '.' in str_val:
            integer, decimal = str_val.split('.')
            if len(decimal) > 2:
                # 保留两位小数
                return round(value, 2)

        # 检查是否为整数但明显应该是小数
        # 例如 "3" 可能是 "3.0" 的误识别
        if value == int(value) and value > 0:
            # 如果上下文都是小数，则添加小数点
            if neighbors and all(isinstance(n, float) and n != int(n) for n in neighbors):
                avg = sum(neighbors) / len(neighbors)
                # 如果平均值的小数部分有意义，则使用
                if avg != int(avg):
                    decimal_places = len(str(avg).split('.')[-1])
                    return round(value, decimal_places if decimal_places <= 2 else 2)

        return value

    @staticmethod
    def extract_roman_with_fix(text: str) -> Optional[int]:
        """
        提取罗马数字等级（带自动修正）

        Args:
            text: 原始OCR文本

        Returns:
            提取的等级值（1-3），无效返回None
        """
        if not text:
            return None

        # 先清理文本
        text = PostProcessor.clean_text(text)

        # 匹配罗马数字 + 度数
        roman_pattern = r'[IVX]+°?'
        match = re.search(roman_pattern, text)
        if match:
            roman = match.group().replace('°', '')
            # 转换为数字
            roman_values = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
            if roman in roman_values:
                value = roman_values[roman]
                return min(value, 3)  # 限制最大值为3

        # 尝试直接匹配数字+度数
        degree_pattern = r'[0123]°'
        match = re.search(degree_pattern, text)
        if match:
            return int(match.group().replace('°', ''))

        # 尝试匹配孤立的罗马数字（无度数符号）
        if 'I' in text or 'II' in text or 'III' in text:
            if 'III' in text or 'lll' in text:
                return 3
            elif 'II' in text or 'll' in text:
                return 2
            elif 'I' in text:
                return 1

        return None


class EnhancedDataTypeExtractor:
    """增强型数据类型提取器 - 结合后处理优化"""

    @staticmethod
    def extract_pd(text: str, neighbors: list = None) -> Optional[float]:
        """
        提取PD值（探诊深度）- 带后处理修正
        格式：小数，如 3.5, 4, 5.0
        范围：0-15mm
        """
        if not text:
            return None

        # 清理文本
        text = PostProcessor.clean_text(text)

        # 匹配数字（包括小数）
        pattern = r'(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                value = float(match)
                # 验证并修正
                validated = PostProcessor.validate_pd_value(value, neighbors)
                if validated is not None:
                    return validated
            except ValueError:
                continue

        return None

    @staticmethod
    def extract_bop(text: str) -> Optional[int]:
        """
        提取BOP值（出血指数）- 带后处理修正
        格式：数字0-4，或符号 +/-
        范围：0-4
        """
        if not text:
            return None

        # 清理文本
        text = PostProcessor.clean_text(text)

        # 首先尝试数字
        for digit in '01234':
            if digit in text:
                return int(digit)

        # 符号转换
        if '+' in text or '√' in text or '✓' in text:
            return 1  # 有出血
        elif '-' in text or '×' in text:
            return 0  # 无出血

        # 尝试从罗马数字转换（有时BOP也用罗马数字）
        roman_value = PostProcessor.extract_roman_with_fix(text)
        if roman_value is not None and 0 <= roman_value <= 4:
            return roman_value

        return None

    @staticmethod
    def extract_pi(text: str) -> Optional[int]:
        """
        提取PI值（菌斑指数）- 使用罗马数字提取
        格式：罗马数字 I°-III°
        范围：0-3
        """
        return PostProcessor.extract_roman_with_fix(text)

    @staticmethod
    def extract_furcation(text: str) -> Optional[int]:
        """
        提取Furcation值（根分叉病变）- 使用罗马数字提取
        格式：罗马数字 I°-III°
        范围：0-3
        """
        return PostProcessor.extract_roman_with_fix(text)

    @staticmethod
    def extract_mobility(text: str) -> Optional[int]:
        """
        提取Mobility值（松动度）- 使用罗马数字提取
        格式：罗马数字 I°-III°
        范围：0-3
        """
        return PostProcessor.extract_roman_with_fix(text)


# 测试代码
if __name__ == '__main__':
    processor = PostProcessor()
    extractor = EnhancedDataTypeExtractor()

    # 测试用例
    test_cases = [
        # 罗马数字测试
        ("I°", "PI/Furcation/Mobility"),
        ("ll", "PI/Furcation/Mobility"),
        ("|||", "PI/Furcation/Mobility"),
        ("l°", "PI/Furcation/Mobility"),
        ("1°", "PI/Furcation/Mobility"),

        # PD值测试
        ("3.5", "PD"),
        ("2.23322222", "PD"),  # 异常精度
        ("2:3", "PD"),  # 冒号
        ("4", "PD"),

        # BOP测试
        ("+", "BOP"),
        ("-", "BOP"),
        ("2°", "BOP"),  # 可能被误识别

        # 医学符号
        ("×", "PI/通用"),
    ]

    print("=" * 60)
    print("后处理优化测试")
    print("=" * 60)

    for text, category in test_cases:
        cleaned = processor.clean_text(text)

        result = None
        if "PD" in category:
            result = extractor.extract_pd(cleaned)
        elif "BOP" in category:
            result = extractor.extract_bop(cleaned)
        elif "PI" in category or "Furcation" in category or "Mobility" in category:
            result = extractor.extract_roman_with_fix(cleaned)

        print(f"{text:15} -> {cleaned:15} -> {result} ({category})")
