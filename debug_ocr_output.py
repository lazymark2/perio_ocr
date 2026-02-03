"""
调试脚本 - 查看PaddleOCR的原始输出
"""
from paddleocr import PaddleOCR
import cv2
import json

# 初始化OCR
ocr = PaddleOCR(lang='ch', enable_mkldnn=False, use_doc_orientation_classify=True)

# 读取图像
img = cv2.imread('handwrite.jpg')
h, w = img.shape[:2]
print(f'图像尺寸: {w}x{h}')
print('=' * 50)

# 执行OCR
result = ocr.predict(img)
if result and len(result) > 0:
    ocr_data = result[0]
    texts = ocr_data.get('rec_texts', [])
    boxes = ocr_data.get('rec_boxes', [])

    print(f'识别到 {len(texts)} 个文本')
    print('=' * 50)

    # 打印所有识别结果
    for i, (text, box) in enumerate(zip(texts, boxes)):
        box_list = box.tolist() if hasattr(box, 'tolist') else box
        x1, y1, x2, y2 = box_list
        y_center = int((y1 + y2) // 2)
        part = 'upper' if y_center < h // 2 else 'lower'
        print(f'{i+1:3d}. [{part:5s}] y={y_center:4d} : {text}')

    # 分析关键字检测
    print('\n' + '=' * 50)
    print('关键字检测分析:')
    print('=' * 50)

    keywords = {
        'PD': ['PD', '探诊深度'],
        'BOP': ['BOP', '出血指数', '出血'],
        'PI': ['PI', '菌斑指数'],
        'Furcation': ['分叉', 'FURCATION'],
        'Mobility': ['松动', 'MOBILITY'],
        '牙位': ['牙位']
    }

    found_keywords = {k: [] for k in keywords}

    for i, text in enumerate(texts):
        text_upper = text.upper()
        for key, patterns in keywords.items():
            if any(pattern in text_upper or pattern in text for pattern in patterns):
                box = boxes[i].tolist() if i < len(boxes) else [0, 0, 0, 0]
                y_center = int((box[1] + box[3]) // 2)
                part = 'upper' if y_center < h // 2 else 'lower'
                found_keywords[key].append((i, y_center, part, text))

    for key, matches in found_keywords.items():
        if matches:
            print(f'\n{key} ({len(matches)}个):')
            for idx, y, part, text in matches:
                print(f'  [{idx:3d}] y={y:4d} [{part}] {text}')
        else:
            print(f'\n{key}: 未检测到')

else:
    print("OCR识别失败，无结果")
