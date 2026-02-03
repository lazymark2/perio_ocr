"""
调试Furcation检测问题
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_system.enhanced_periodontal_ocr import RegionDetector
import cv2
import json


def debug_ocr_output(image_path):
    """调试OCR输出，分析Furcation检测问题"""

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    detector = RegionDetector()

    print(f"Image size: {w}x{h}")
    print("\n" + "="*60)
    print("Running OCR detection...")
    print("="*60)

    # 获取OCR结果
    result = detector.ocr.predict(img)

    if not result or len(result) == 0:
        print("No OCR results!")
        return

    ocr_data = result[0]
    texts = ocr_data.get('rec_texts', [])
    boxes = ocr_data.get('rec_boxes', [])

    print(f"Total texts detected: {len(texts)}")

    # 分析文本内容
    print("\n" + "="*60)
    print("1. 分析包含'分叉'或'Furcation'的文本")
    print("="*60)

    furcation_related = []
    for i, text in enumerate(texts):
        if '分叉' in text or 'FUR' in text.upper() or 'fur' in text.lower():
            box = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
            y_center = (box[1] + box[3]) // 2
            furcation_related.append({
                'text': text,
                'y': y_center,
                'box': box
            })
            print(f"  Found: '{text}' at Y={y_center}")

    print(f"\nFurcation相关文本数量: {len(furcation_related)}")

    # 分析罗马数字候选
    print("\n" + "="*60)
    print("2. 分析罗马数字候选文本")
    print("="*60)

    roman_candidates = []
    for i, text in enumerate(texts):
        # 清理文本
        t_clean = text.strip().replace(' ', '').replace('°', '').replace('o', '').replace('0', '')

        # 检查是否为罗马数字模式
        if t_clean in ['I', 'II', 'III', 'IV', 'V', 'l', 'll', 'lll', 'lV', '|', '||', '|||']:
            box = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
            y_center = (box[1] + box[3]) // 2
            roman_candidates.append({
                'text': text,
                'y': y_center,
                'clean': t_clean
            })
            print(f"  Roman candidate: '{text}' -> '{t_clean}' at Y={y_center}")

    print(f"\n罗马数字候选数量: {len(roman_candidates)}")

    # 按Y坐标分析区域分布
    print("\n" + "="*60)
    print("3. 按Y坐标分析文本分布 (每100像素)")
    print("="*60)

    y_ranges = {}
    for i, text in enumerate(texts):
        box = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
        y_center = int((box[1] + box[3]) // 2)
        range_key = (y_center // 100) * 100

        if range_key not in y_ranges:
            y_ranges[range_key] = []
        y_ranges[range_key].append(text)

    for y_range in sorted(y_ranges.keys()):
        texts_in_range = y_ranges[y_range]
        print(f"  Y {y_range:4d}-{y_range+99:4d}: {len(texts_in_range)} texts")
        # 显示关键文本
        for t in texts_in_range[:5]:
            if len(t) > 0 and t not in [' ', '']:
                print(f"    - {t}")

    # 检测区域
    print("\n" + "="*60)
    print("4. 检测到的区域")
    print("="*60)

    regions = detector.detect_regions(image_path)

    for region_name, region_data in regions.items():
        label = region_data.get('label', 'Unknown')
        y_range = region_data.get('y_range', None)
        bbox = region_data.get('bbox', None)

        if y_range:
            print(f"  {region_name} ({label}):")
            print(f"    Y range: {y_range}")
        elif bbox:
            print(f"  {region_name} ({label}):")
            print(f"    BBox: {bbox}")

    # 分析罗马数字落在哪个区域
    print("\n" + "="*60)
    print("5. 罗马数字候选的区域归属")
    print("="*60)

    # 导入修复后的区域检测方法
    from ocr_system.enhanced_periodontal_ocr import EnhancedPerioOCR
    ocr_system = EnhancedPerioOCR()

    for roman in roman_candidates:
        y = roman['y']
        found_region = ocr_system._get_region_type(y, regions)

        print(f"  '{roman['text']}' (Y={y}) -> {found_region if found_region else 'NO REGION'}")

    # 输出前30个文本及其Y坐标
    print("\n" + "="*60)
    print("6. 前30个检测文本及其Y坐标")
    print("="*60)

    for i, (text, box) in enumerate(zip(texts[:30], boxes[:30])):
        y_center = int((box[1] + box[3]) // 2)
        print(f"  {y_center:4d}: {text}")


if __name__ == '__main__':
    image_path = 'handwrite3.jpg'
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    debug_ocr_output(image_path)
