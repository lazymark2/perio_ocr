"""
测试图像预处理效果

对比有预处理和无预处理的OCR识别结果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_system.enhanced_periodontal_ocr import EnhancedPerioOCR
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


def test_preprocessing_effect(image_path, output_dir='output/preprocessing_test'):
    """
    测试预处理效果

    Args:
        image_path: 测试图像路径
        output_dir: 输出目录
    """
    ocr = EnhancedPerioOCR()

    print(f"\n{'='*60}")
    print(f"测试图像: {Path(image_path).name}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 无预处理的OCR
    print("1. 运行无预处理的OCR...")
    try:
        result_no_preprocess = ocr.process(image_path, enable_preprocessing=False)
        stats_no_preprocess = result_no_preprocess['statistics']

        print(f"  总牙齿数: {stats_no_preprocess['total_teeth']}")
        print(f"  PD数据: {stats_no_preprocess['teeth_with_pd']} 颗")
        print(f"  BOP数据: {stats_no_preprocess['teeth_with_bop']} 颗")
        print(f"  PI数据: {stats_no_preprocess['teeth_with_pi']} 颗")
        print(f"  分叉病变: {stats_no_preprocess['teeth_with_furcation']} 颗")
        print(f"  松动度: {stats_no_preprocess['teeth_with_mobility']} 颗")

        # 保存结果
        output_file = os.path.join(output_dir, f"{Path(image_path).stem}_no_preprocess.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_no_preprocess, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存: {output_file}")

    except Exception as e:
        print(f"  错误: {e}")
        stats_no_preprocess = None

    print()

    # 2. 有预处理的OCR
    print("2. 运行有预处理的OCR...")
    try:
        result_with_preprocess = ocr.process(image_path, enable_preprocessing=True)
        stats_with_preprocess = result_with_preprocess['statistics']

        print(f"  总牙齿数: {stats_with_preprocess['total_teeth']}")
        print(f"  PD数据: {stats_with_preprocess['teeth_with_pd']} 颗")
        print(f"  BOP数据: {stats_with_preprocess['teeth_with_bop']} 颗")
        print(f"  PI数据: {stats_with_preprocess['teeth_with_pi']} 颗")
        print(f"  分叉病变: {stats_with_preprocess['teeth_with_furcation']} 颗")
        print(f"  松动度: {stats_with_preprocess['teeth_with_mobility']} 颗")

        # 保存结果
        output_file = os.path.join(output_dir, f"{Path(image_path).stem}_with_preprocess.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_with_preprocess, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存: {output_file}")

    except Exception as e:
        print(f"  错误: {e}")
        stats_with_preprocess = None

    # 3. 对比结果
    print(f"\n{'='*60}")
    print("效果对比")
    print(f"{'='*60}\n")

    if stats_no_preprocess and stats_with_preprocess:
        metrics = ['total_teeth', 'teeth_with_pd', 'teeth_with_bop',
                   'teeth_with_pi', 'teeth_with_furcation', 'teeth_with_mobility']

        print(f"{'指标':<20} | {'无预处理':<12} | {'有预处理':<12} | {'变化':>10}")
        print('-' * 60)

        for metric in metrics:
            no_proc = stats_no_preprocess[metric]
            with_proc = stats_with_preprocess[metric]
            diff = with_proc - no_proc
            diff_str = f"+{diff}" if diff > 0 else str(diff)

            print(f"{metric:<20} | {no_proc:<12} | {with_proc:<12} | {diff_str:>10}")

        # 计算总体改进
        total_improvement = sum(
            stats_with_preprocess[m] - stats_no_preprocess[m]
            for m in ['teeth_with_pd', 'teeth_with_bop', 'teeth_with_furcation', 'teeth_with_mobility']
        )

        print('-' * 60)
        print(f"{'总体改进':<20} | {total_improvement:>38}")

    # 4. 详细牙齿数据对比
    print(f"\n{'='*60}")
    print("详细牙齿数据对比（前5颗）")
    print(f"{'='*60}\n")

    if stats_no_preprocess and stats_no_preprocess['total_teeth'] > 0:
        teeth_no_proc = result_no_preprocess['teeth_data']
        print("无预处理:")
        for i, (tooth_id, data) in enumerate(list(teeth_no_proc.items())[:5]):
            print(f"  牙齿{tooth_id}:")
            if any(data['pd'].values()):
                print(f"    PD: {data['pd']}")
            if any(data['bop'].values()):
                print(f"    BOP: {data['bop']}")
            if any(data['pi'].values()):
                print(f"    PI: {data['pi']}")
            if data['mobility'] > 0:
                print(f"    松动度: {data['mobility']}°")
            if data['furcation'] > 0:
                print(f"    分叉病变: {data['furcation']}°")

    print()

    if stats_with_preprocess and stats_with_preprocess['total_teeth'] > 0:
        teeth_with_proc = result_with_preprocess['teeth_data']
        print("有预处理:")
        for i, (tooth_id, data) in enumerate(list(teeth_with_proc.items())[:5]):
            print(f"  牙齿{tooth_id}:")
            if any(data['pd'].values()):
                print(f"    PD: {data['pd']}")
            if any(data['bop'].values()):
                print(f"    BOP: {data['bop']}")
            if any(data['pi'].values()):
                print(f"    PI: {data['pi']}")
            if data['mobility'] > 0:
                print(f"    松动度: {data['mobility']}°")
            if data['furcation'] > 0:
                print(f"    分叉病变: {data['furcation']}°")


if __name__ == '__main__':
    import glob

    # 查找手写图像
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_images = glob.glob(os.path.join(base_dir, 'handwrite*.jpg'))
    test_images.sort()

    if not test_images:
        print("未找到手写图像文件")
        sys.exit(1)

    # 选择第一张图像进行测试
    test_image = test_images[0]
    print(f"选择测试图像: {Path(test_image).name}")
    print(f"如需测试所有图像，请使用: python test_preprocessing.py all")

    if len(sys.argv) > 1 and sys.argv[1] == 'all':
        # 测试所有图像
        for img_path in test_images:
            test_preprocessing_effect(img_path)
    else:
        # 只测试第一张图像
        test_preprocessing_effect(test_image)
