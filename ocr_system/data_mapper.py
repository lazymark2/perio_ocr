"""
数据映射模块 - 映射OCR结果到标准数据结构
"""
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataMapper:
    """牙周图表数据映射器"""

    # 牙齿编号映射（FDI编号系统）
    TOOTH_NUMBERS = [
        '18', '17', '16', '15', '14', '13', '12', '11',
        '21', '22', '23', '24', '25', '26', '27', '28',
        '31', '32', '33', '34', '35', '36', '37', '38',
        '41', '42', '43', '44', '45', '46', '47', '48'
    ]

    # 前牙测量面
    ANTERIOR_SURFACES = ['db', 'b', 'mb', 'dl', 'l', 'ml']
    # 后牙测量面
    POSTERIOR_SURFACES = ['db', 'b', 'mb', 'dp', 'p', 'mp']

    # 磨牙编号（需要分叉病变检测）
    MOLAR_NUMBERS = ['16', '17', '18', '26', '27', '28', '36', '37', '38', '46', '47', '48']

    def __init__(self, config: dict = None):
        self.config = config or {}

    @staticmethod
    def create_default_template() -> Dict:
        """创建默认数据模板"""
        template = {
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

        # 添加32颗牙齿的默认字段
        for tooth_num in DataMapper.TOOTH_NUMBERS:
            # 基础字段
            template[f'tooth_{tooth_num}'] = '0'
            template[f'mobility_{tooth_num}'] = '0'
            template[f'implant_{tooth_num}'] = '0'

            # 选择测量面
            surfaces = DataMapper.ANTERIOR_SURFACES if tooth_num[1] in '12345678' else DataMapper.POSTERIOR_SURFACES

            # 分叉病变（仅磨牙）
            if tooth_num in DataMapper.MOLAR_NUMBERS:
                if 'dp' in surfaces:
                    for surf in ['b', 'dp', 'mp']:
                        template[f'furcation_{tooth_num}_{surf}'] = '0'
                else:
                    for surf in ['b', 'l']:
                        template[f'furcation_{tooth_num}_{surf}'] = '0'

            # 测量数据
            for surface in surfaces:
                template[f'bop_{tooth_num}_{surface}'] = '0'
                template[f'pi_{tooth_num}_{surface}'] = '0'
                template[f'gm_{tooth_num}_{surface}'] = '0'
                template[f'pd_{tooth_num}_{surface}'] = '0'

            # 备注
            template[f'note_{tooth_num}'] = ''

        return template

    def map_to_fdi(self, us_number: str) -> str:
        """将美国编号系统转换为FDI编号"""
        # 美国编号: 1-32, FDI: 11-48
        us_to_fdi = {
            '1': '11', '2': '12', '3': '13', '4': '14', '5': '15',
            '6': '16', '7': '17', '8': '18',
            '9': '21', '10': '22', '11': '23', '12': '24', '13': '25',
            '14': '26', '15': '27', '16': '28',
            '17': '31', '18': '32', '19': '33', '20': '34', '21': '35',
            '22': '36', '23': '37', '24': '38',
            '25': '41', '26': '42', '27': '43', '28': '44', '29': '45',
            '30': '46', '31': '47', '32': '48'
        }
        return us_to_fdi.get(us_number, us_number)

    def map_ocr_to_structure(self, ocr_data: Dict) -> Dict:
        """
        将OCR结果映射到标准数据结构
        """
        result = self.create_default_template()

        for tooth_num, tooth_data in ocr_data.items():
            self._map_single_tooth(result, tooth_num, tooth_data)

        return result

    def _map_single_tooth(self, result: Dict, tooth_num: str, tooth_data: Dict):
        """映射单颗牙齿数据到结果"""
        if tooth_num not in self.TOOTH_NUMBERS:
            logger.warning(f"未知牙齿编号: {tooth_num}")
            return

        surfaces = self.ANTERIOR_SURFACES if tooth_num[1] in '12345678' else self.POSTERIOR_SURFACES

        # 基础字段
        if 'tooth' in tooth_data:
            result[f'tooth_{tooth_num}'] = str(int(tooth_data['tooth']))

        if 'mobility' in tooth_data:
            result[f'mobility_{tooth_num}'] = str(int(tooth_data['mobility']))

        if 'implant' in tooth_data:
            result[f'implant_{tooth_num}'] = str(int(tooth_data['implant']))

        # PD, GM, BOP, PI
        for measurement in ['pd', 'gm', 'bop', 'pi']:
            if measurement in tooth_data:
                for surface in surfaces:
                    if surface in tooth_data[measurement]:
                        value = tooth_data[measurement][surface]
                        result[f'{measurement}_{tooth_num}_{surface}'] = str(int(value))

        # 分叉病变
        if tooth_num in self.MOLAR_NUMBERS and 'furcation' in tooth_data:
            if 'dp' in surfaces:
                for surf in ['b', 'dp', 'mp']:
                    if surf in tooth_data['furcation']:
                        result[f'furcation_{tooth_num}_{surf}'] = str(int(tooth_data['furcation'][surf]))
            else:
                for surf in ['b', 'l']:
                    if surf in tooth_data['furcation']:
                        result[f'furcation_{tooth_num}_{surf}'] = str(int(tooth_data['furcation'][surf]))

        # 备注
        if 'note' in tooth_data:
            result[f'note_{tooth_num}'] = tooth_data['note']

    def validate_output(self, data: Dict) -> List[str]:
        """验证输出数据的有效性"""
        errors = []

        for tooth_num in self.TOOTH_NUMBERS:
            # 检查PD范围
            surfaces = self.ANTERIOR_SURFACES if tooth_num[1] in '12345678' else self.POSTERIOR_SURFACES
            for surface in surfaces:
                pd_key = f'pd_{tooth_num}_{surface}'
                if pd_key in data:
                    try:
                        pd = int(data[pd_key])
                        if pd < 0 or pd > 15:
                            errors.append(f"{pd_key} 值 {pd} 超出有效范围 (0-15)")
                    except (ValueError, TypeError):
                        errors.append(f"{pd_key} 值无效")

        return errors

    def export_to_json(self, data: Dict, output_path: str):
        """导出数据到JSON文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"数据已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            raise


if __name__ == '__main__':
    # 测试
    mapper = DataMapper()

    # 创建模板
    template = mapper.create_default_template()
    print(f"模板包含 {len(template)} 个字段")

    # 验证
    errors = mapper.validate_output(template)
    print(f"验证错误: {errors}")
