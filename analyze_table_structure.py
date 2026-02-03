"""
分析牙周检查记录表的上下半部分结构
确定中间分隔线位置和各数据类型的上下分布
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paddleocr import PaddleOCR
import cv2


def analyze_table_structure(image_path):
    """分析表格的上下半部分结构"""

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    ocr = PaddleOCR(lang='ch', enable_mkldnn=False)
    result = ocr.predict(img)

    if not result or len(result) == 0:
        print("No OCR results!")
        return

    ocr_data = result[0]
    texts = ocr_data.get('rec_texts', [])
    boxes = ocr_data.get('rec_boxes', [])

    print(f"Image size: {w}x{h}")
    print(f"Total texts detected: {len(texts)}")

    # 查找关键分隔标识
    print("\n" + "="*60)
    print("1. 查找表格分隔线（'牙位'行）")
    print("="*60)

    separator_candidates = []
    for i, text in enumerate(texts):
        if '牙位' in text:
            box = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
            y_center = int((box[1] + box[3]) // 2)
            separator_candidates.append({
                'text': text,
                'y': y_center,
                'box': box
            })
            print(f"  Found: '{text}' at Y={y_center}")

    if not separator_candidates:
        print("  No explicit '牙位' found, searching for numeric sequences...")
        # 查找数字序列 8 7 6 5 4 3 2 1
        for i, text in enumerate(texts):
            # 检查是否包含连续数字
            digits = ''.join([c for c in text if c.isdigit()])
            if len(digits) >= 4:  # 至少4个连续数字
                box = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
                y_center = int((box[1] + box[3]) // 2)
                separator_candidates.append({
                    'text': text,
                    'y': y_center,
                    'box': box
                })
                print(f"  Candidate: '{text}' at Y={y_center}")

    # 确定分隔线Y坐标（使用中间位置的候选）
    if separator_candidates:
        separator_candidates.sort(key=lambda x: x['y'])
        mid_index = len(separator_candidates) // 2
        separator_y = separator_candidates[mid_index]['y']
        print(f"\n分隔线Y坐标（估计）: {separator_y}")
    else:
        separator_y = h // 2
        print(f"\n使用图像中点作为分隔线: {separator_y}")

    # 分析各数据类型的上下分布
    print("\n" + "="*60)
    print("2. 各数据类型的上下半部分分布")
    print("="*60)

    data_types = {
        'PI': ['菌斑指数', 'PI', 'π'],
        'Mobility': ['松动度', 'Mobility'],
        'Furcation': ['根分叉病变', '分叉', 'Furcation'],
        'BOP': ['出血指数', 'BOP', '出血'],
        'PD': ['探诊深度', 'PD'],
    }

    type_distribution = {}

    for type_name, keywords in data_types.items():
        type_distribution[type_name] = {'upper': [], 'lower': []}

        for i, text in enumerate(texts):
            for keyword in keywords:
                if keyword in text:
                    box = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
                    y_center = int((box[1] + box[3]) // 2)

                    if y_center < separator_y:
                        type_distribution[type_name]['upper'].append(y_center)
                    else:
                        type_distribution[type_name]['lower'].append(y_center)
                    break

    for type_name, distribution in type_distribution.items():
        upper_y = distribution['upper']
        lower_y = distribution['lower']

        print(f"\n{type_name}:")
        if upper_y:
            avg_upper = sum(upper_y) // len(upper_y)
            print(f"  上半部分: {len(upper_y)} 个, 平均Y={avg_upper}")
        else:
            print(f"  上半部分: 未检测到")

        if lower_y:
            avg_lower = sum(lower_y) // len(lower_y)
            print(f"  下半部分: {len(lower_y)} 个, 平均Y={avg_lower}")
        else:
            print(f"  下半部分: 未检测到")

    # 分析罗马数字的分布
    print("\n" + "="*60)
    print("3. 罗马数字的上下半部分分布")
    print("="*60)

    roman_numerals = []
    for i, text in enumerate(texts):
        t_clean = text.strip().replace(' ', '').replace('°', '')
        if t_clean in ['I', 'II', 'III', 'IV', 'V', 'l', 'll', 'lll', 'lV', '|', '||', '|||']:
            box = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
            y_center = int((box[1] + box[3]) // 2)
            part = '上半部分' if y_center < separator_y else '下半部分'
            roman_numerals.append({'text': text, 'y': y_center, 'part': part})

    # 按Y坐标排序
    roman_numerals.sort(key=lambda x: x['y'])

    for roman in roman_numerals:
        print(f"  '{roman['text']}' at Y={roman['y']:4d} ({roman['part']})")

    print(f"\n总计: {len(roman_numerals)} 个罗马数字候选")
    upper_count = sum(1 for r in roman_numerals if r['part'] == '上半部分')
    lower_count = sum(1 for r in roman_numerals if r['part'] == '下半部分')
    print(f"  上半部分: {upper_count} 个")
    print(f"  下半部分: {lower_count} 个")

    # 建议的区域划分
    print("\n" + "="*60)
    print("4. 建议的区域划分方案")
    print("="*60)

    if type_distribution.get('PI', {}).get('upper'):
        pi_upper_y = type_distribution['PI']['upper'][0] if type_distribution['PI']['upper'] else None
        print(f"上半部分 PI 区域起始: Y≈{pi_upper_y if pi_upper_y else 'N/A'}")

    if type_distribution.get('PD', {}).get('upper'):
        pd_upper_y = type_distribution['PD']['upper'][0] if type_distribution['PD']['upper'] else None
        print(f"上半部分 PD 区域起始: Y≈{pd_upper_y if pd_upper_y else 'N/A'}")

    if type_distribution.get('PI', {}).get('lower'):
        pi_lower_y = type_distribution['PI']['lower'][0] if type_distribution['PI']['lower'] else None
        print(f"下半部分 PI 区域起始: Y≈{pi_lower_y if pi_lower_y else 'N/A'}")

    if type_distribution.get('PD', {}).get('lower'):
        pd_lower_y = type_distribution['PD']['lower'][0] if type_distribution['PD']['lower'] else None
        print(f"下半部分 PD 区域起始: Y≈{pd_lower_y if pd_lower_y else 'N/A'}")

    # 输出建议
    print("\n建议实现方案:")
    print("1. 检测或估计表格中间分隔线位置（牙位行）")
    print("2. 为上下半部分分别建立区域映射")
    print("3. 在牙齿定位时，根据Y坐标判断上颌/下颌")
    print("   - Y < separator_y: 上颌牙齿 (11-28)")
    print("   - Y > separator_y: 下颌牙齿 (31-48)")

    return {
        'separator_y': separator_y,
        'type_distribution': type_distribution,
        'roman_numerals': roman_numerals
    }


if __name__ == '__main__':
    image_path = 'handwrite3.jpg'
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    analyze_table_structure(image_path)
