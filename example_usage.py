"""
main.py 使用示例

演示如何使用 PerioOCR 进行牙周图表 OCR 识别
"""
import logging
from main import PerioOCR

# 配置日志
logging.basicConfig(level=logging.INFO)


def example_basic():
    """示例 1: 基本用法"""
    print("\n=== 示例 1: 基本用法 ===")

    ocr = PerioOCR(mode='auto')

    result = ocr.process(
        image_path='handwrite.jpg',
        output_path='output_result.json'
    )

    print(f"处理完成，检测到 {result['statistics']['total_teeth']} 颗牙齿")

    ocr.close()


def example_enhanced_mode():
    """示例 2: 使用增强模式（适合手写数字）"""
    print("\n=== 示例 2: 增强模式 ===")

    ocr = PerioOCR(
        mode='enhanced',
        enable_preprocessing=True,
        enable_postprocess=True
    )

    result = ocr.process('handwrite.jpg')

    # 显示部分牙齿数据
    for tooth_id, data in list(result['teeth_data'].items())[:5]:
        if any(data['pd'].values()):
            print(f"牙齿 {tooth_id}: PD = {data['pd']}")

    ocr.close()


def example_with_context_manager():
    """示例 3: 使用上下文管理器"""
    print("\n=== 示例 3: 上下文管理器 ===")

    with PerioOCR(mode='auto') as ocr:
        result = ocr.process('handwrite.jpg')
        print(f"处理时间: {result['metadata']['processing_time']}s")


def example_batch_processing():
    """示例 4: 批量处理"""
    print("\n=== 示例 4: 批量处理 ===")

    from pathlib import Path

    with PerioOCR(mode='auto') as ocr:
        image_dir = Path('.')
        image_files = list(image_dir.glob('handwrite*.jpg'))

        for img_path in image_files[:3]:  # 处理前 3 张
            output_path = f'output/{img_path.stem}_result.json'
            try:
                result = ocr.process(str(img_path), output_path=output_path)
                print(f"✓ {img_path.name}: {result['statistics']['total_teeth']} 颗牙齿")
            except Exception as e:
                print(f"✗ {img_path.name}: {e}")


def example_access_teeth_data():
    """示例 5: 访问牙齿数据"""
    print("\n=== 示例 5: 访问牙齿数据 ===")

    ocr = PerioOCR(mode='enhanced')
    result = ocr.process('handwrite.jpg')

    # 统计信息
    stats = result['statistics']
    print(f"总牙齿数: {stats['total_teeth']}")
    print(f"有 PD 数据: {stats['teeth_with_pd']}")
    print(f"有 BOP 数据: {stats['teeth_with_bop']}")

    # 遍历牙齿数据
    print("\n牙齿数据详情:")
    for tooth_id, data in list(result['teeth_data'].items())[:10]:
        pd_values = [v for v in data['pd'].values() if v > 0]
        if pd_values:
            print(f"  {tooth_id}: PD = {data['pd']} (有效值: {pd_values})")

    ocr.close()


def example_different_modes():
    """示例 6: 测试不同模式"""
    print("\n=== 示例 6: 测试不同模式 ===")

    modes = ['auto', 'enhanced']  # 只测试有安装的模式

    for mode in modes:
        try:
            print(f"\n测试模式: {mode}")
            ocr = PerioOCR(mode=mode)
            result = ocr.process('handwrite.jpg')
            print(f"  ✓ 成功 - {result['metadata']['method']}")
            print(f"  检测到 {result['statistics']['total_teeth']} 颗牙齿")
            ocr.close()
        except Exception as e:
            print(f"  ✗ 失败 - {e}")


if __name__ == '__main__':
    # 运行示例

    # 1. 基本用法
    example_basic()

    # 2. 增强模式
    # example_enhanced_mode()

    # 3. 上下文管理器
    # example_with_context_manager()

    # 4. 批量处理（需要先创建 output 目录）
    # example_batch_processing()

    # 5. 访问牙齿数据
    # example_access_teeth_data()

    # 6. 测试不同模式
    # example_different_modes()
