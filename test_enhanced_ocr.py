"""
批量测试脚本 - 在多张手写图像上测试增强型OCR系统
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_system.enhanced_periodontal_ocr import EnhancedPerioOCR
import json
from pathlib import Path


def batch_test_images(image_paths, output_dir='output'):
    """批量测试多张图像"""
    ocr = EnhancedPerioOCR()

    results = []
    total_statistics = {
        'total_teeth': 0,
        'teeth_with_pd': 0,
        'teeth_with_bop': 0,
        'teeth_with_pi': 0,
        'teeth_with_furcation': 0,
        'teeth_with_mobility': 0,
        'images_processed': 0
    }

    for img_path in image_paths:
        img_name = Path(img_path).name
        print(f"\n{'='*60}")
        print(f"处理图像: {img_name}")
        print(f"{'='*60}")

        try:
            result = ocr.process(img_path)
            stats = result['statistics']

            print(f"检测到 {stats['total_teeth']} 颗牙齿")
            print(f"  有PD数据: {stats['teeth_with_pd']} 颗")
            print(f"  有BOP数据: {stats['teeth_with_bop']} 颗")
            print(f"  有PI数据: {stats['teeth_with_pi']} 颗")
            print(f"  有分叉病变: {stats['teeth_with_furcation']} 颗")
            print(f"  有松动度: {stats['teeth_with_mobility']} 颗")

            # 显示部分牙齿数据
            teeth_data = result.get('teeth_data', {})
            if teeth_data:
                print(f"\n部分牙齿数据 (前3颗):")
                for i, (tooth_id, data) in enumerate(list(teeth_data.items())[:3]):
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

            # 累计统计
            for key in total_statistics:
                if key != 'images_processed':
                    total_statistics[key] += stats.get(key, 0)
            total_statistics['images_processed'] += 1

            # 保存单张图像结果
            output_file = os.path.join(output_dir, f"{Path(img_path).stem}_result.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            results.append({
                'image': img_name,
                'statistics': stats,
                'teeth_count': len(teeth_data)
            })

        except Exception as e:
            print(f"处理失败: {e}")

    # 输出汇总统计
    print(f"\n{'='*60}")
    print(f"批量测试汇总 (处理了 {total_statistics['images_processed']} 张图像)")
    print(f"{'='*60}")
    print(f"总共检测到 {total_statistics['total_teeth']} 颗牙齿")
    print(f"平均每张图像: {total_statistics['total_teeth'] / max(1, total_statistics['images_processed']):.1f} 颗")
    print(f"\n数据类型分布:")
    print(f"  PD数据: {total_statistics['teeth_with_pd']} 个")
    print(f"  BOP数据: {total_statistics['teeth_with_bop']} 个")
    print(f"  PI数据: {total_statistics['teeth_with_pi']} 个")
    print(f"  分叉病变: {total_statistics['teeth_with_furcation']} 个")
    print(f"  松动度: {total_statistics['teeth_with_mobility']} 个")

    # 保存汇总结果
    summary_file = os.path.join(output_dir, 'batch_test_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': total_statistics,
            'per_image': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_dir}/")
    print(f"  - 每张图像的详细结果: *_result.json")
    print(f"  - 汇总统计: batch_test_summary.json")


if __name__ == '__main__':
    import glob

    # 查找所有手写图像
    base_dir = os.path.dirname(os.path.abspath(__file__))
    handwrite_images = glob.glob(os.path.join(base_dir, 'handwrite*.jpg'))
    handwrite_images.sort()

    if not handwrite_images:
        print("未找到手写图像文件")
        print(f"搜索路径: {os.path.join(base_dir, 'handwrite*.jpg')}")
        print(f"当前目录: {os.getcwd()}")
        print(f"脚本位置: {__file__}")
        sys.exit(1)

    print(f"找到 {len(handwrite_images)} 张手写图像:")
    for img in handwrite_images:
        print(f"  - {os.path.basename(img)}")

    # 批量测试
    batch_test_images(handwrite_images, output_dir='output/batch_test')
