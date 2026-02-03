"""
测试模块化OCR处理器

验证新的模块化OCR处理器是否正常工作
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_system.enhanced_periodontal_ocr import EnhancedPerioOCR
from ocr_system.modular_ocr import ModularOCR
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


def test_modular_ocr():
    """测试模块化OCR处理器"""
    print("\n" + "=" * 60)
    print("测试模块化OCR处理器")
    print("=" * 60 + "\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_images = [
        os.path.join(base_dir, 'handwrite.jpg'),
        os.path.join(base_dir, 'handwrite2.jpg'),
    ]

    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"跳过不存在的图像: {image_path}")
            continue

        print(f"\n测试图像: {Path(image_path).name}")
        print("-" * 60)

        try:
            with ModularOCR(
                enable_doc_orientation=True,
                enable_unwarping=False,
                enable_textline_orientation=False
            ) as ocr:
                texts, boxes, raw_results = ocr.process(image_path, return_raw_results=True)

                print(f"检测到 {len(texts)} 个文本")
                print(f"文档方向: {raw_results.get('orientation', 'N/A')}")
                print(f"图像矫正: {raw_results.get('unwarped', False)}")
                print(f"预处理步骤: {raw_results.get('preprocessing_steps', [])}")

                # 显示前5个识别结果
                print("\n前5个识别结果:")
                for i, (text, box) in enumerate(zip(texts[:5], boxes[:5])):
                    print(f"  {i+1}. {text}")

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


def test_enhanced_ocr_without_preprocessing():
    """测试增强型OCR系统（禁用预处理）"""
    print("\n" + "=" * 60)
    print("测试增强型OCR系统（禁用预处理）")
    print("=" * 60 + "\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_images = [
        os.path.join(base_dir, 'handwrite.jpg'),
    ]

    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"跳过不存在的图像: {image_path}")
            continue

        print(f"\n测试图像: {Path(image_path).name}")
        print("-" * 60)

        try:
            ocr = EnhancedPerioOCR()
            result = ocr.process(image_path, enable_preprocessing=False)

            stats = result['statistics']
            print(f"总牙齿数: {stats['total_teeth']}")
            print(f"PD数据: {stats['teeth_with_pd']} 颗")
            print(f"BOP数据: {stats['teeth_with_bop']} 颗")
            print(f"PI数据: {stats['teeth_with_pi']} 颗")
            print(f"分叉病变: {stats['teeth_with_furcation']} 颗")
            print(f"松动度: {stats['teeth_with_mobility']} 颗")

            # 保存结果
            output_dir = os.path.join(base_dir, 'output', 'modular_test')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{Path(image_path).stem}_modular.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"结果已保存: {output_file}")

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


def compare_with_old_results():
    """与旧结果对比"""
    print("\n" + "=" * 60)
    print("与旧结果对比")
    print("=" * 60 + "\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 读取旧结果（无预处理）
    old_result_file = os.path.join(base_dir, 'output', 'preprocessing_test', 'handwrite_no_preprocess.json')
    if os.path.exists(old_result_file):
        with open(old_result_file, 'r', encoding='utf-8') as f:
            old_result = json.load(f)
        old_stats = old_result['statistics']
        print("旧结果（无预处理）:")
        print(f"  总牙齿数: {old_stats['total_teeth']}")
        print(f"  PD数据: {old_stats['teeth_with_pd']} 颗")
        print(f"  BOP数据: {old_stats['teeth_with_bop']} 颗")
        print(f"  PI数据: {old_stats['teeth_with_pi']} 颗")
        print(f"  分叉病变: {old_stats['teeth_with_furcation']} 颗")
        print(f"  松动度: {old_stats['teeth_with_mobility']} 颗")

    # 读取新结果（模块化OCR）
    new_result_file = os.path.join(base_dir, 'output', 'modular_test', 'handwrite_modular.json')
    if os.path.exists(new_result_file):
        with open(new_result_file, 'r', encoding='utf-8') as f:
            new_result = json.load(f)
        new_stats = new_result['statistics']
        print("\n新结果（模块化OCR）:")
        print(f"  总牙齿数: {new_stats['total_teeth']}")
        print(f"  PD数据: {new_stats['teeth_with_pd']} 颗")
        print(f"  BOP数据: {new_stats['teeth_with_bop']} 颗")
        print(f"  PI数据: {new_stats['teeth_with_pi']} 颗")
        print(f"  分叉病变: {new_stats['teeth_with_furcation']} 颗")
        print(f"  松动度: {new_stats['teeth_with_mobility']} 颗")

        # 对比
        if os.path.exists(old_result_file):
            print("\n对比:")
            metrics = ['total_teeth', 'teeth_with_pd', 'teeth_with_bop',
                       'teeth_with_pi', 'teeth_with_furcation', 'teeth_with_mobility']
            for metric in metrics:
                old_val = old_stats[metric]
                new_val = new_stats[metric]
                diff = new_val - old_val
                diff_str = f"{'+' if diff > 0 else ''}{diff}" if diff != 0 else "0"
                print(f"  {metric}: {old_val} → {new_val} ({diff_str})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='测试模块化OCR处理器')
    parser.add_argument('--test', choices=['modular', 'enhanced', 'compare', 'all'],
                        default='all', help='测试类型')
    args = parser.parse_args()

    if args.test in ['modular', 'all']:
        test_modular_ocr()

    if args.test in ['enhanced', 'all']:
        test_enhanced_ocr_without_preprocessing()

    if args.test in ['compare', 'all']:
        compare_with_old_results()
