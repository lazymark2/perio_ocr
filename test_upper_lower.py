"""
快速测试上下半部分处理
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_system.enhanced_periodontal_ocr import RegionDetector
import cv2


def quick_test():
    """快速测试区域检测"""
    image_path = 'handwrite3.jpg'

    print(f"测试图像: {image_path}")
    print("="*60)

    detector = RegionDetector()

    # 检测区域
    print("\n1. 检测区域...")
    regions = detector.detect_regions(image_path)

    # 检查返回结构
    print("\n2. 检查返回结构:")
    print(f"  是否包含 'separator_y': {'separator_y' in regions}")
    if 'separator_y' in regions:
        print(f"  分隔线Y坐标: {regions['separator_y']}")
    print(f"  是否包含 'upper': {'upper' in regions}")
    print(f"  是否包含 'lower': {'lower' in regions}")

    # 显示上半部分区域
    if 'upper' in regions:
        print("\n3. 上半部分区域:")
        for region_name, region_data in regions['upper'].items():
            if 'y_range' in region_data:
                y_min, y_max = region_data['y_range']
                print(f"  {region_name}: Y={y_min}-{y_max}")

    # 显示下半部分区域
    if 'lower' in regions:
        print("\n4. 下半部分区域:")
        for region_name, region_data in regions['lower'].items():
            if 'y_range' in region_data:
                y_min, y_max = region_data['y_range']
                print(f"  {region_name}: Y={y_min}-{y_max}")

    print("\n5. 区域检测完成!")
    return regions


if __name__ == '__main__':
    quick_test()
