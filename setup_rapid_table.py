"""
安装和测试 RapidTableOCR 模块

基于 TableStructureRec 的轻量级牙周图表 OCR 解决方案

安装命令:
    pip install wired_table_rec lineless_table_rec table_cls rapidocr beautifulsoup4
"""
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def install_packages():
    """安装所需的包"""
    packages = [
        "wired_table_rec",
        "lineless_table_rec",
        "table_cls",
        "rapidocr",
        "beautifulsoup4",
        "lxml"
    ]

    logger.info("=" * 50)
    logger.info("安装 RapidTableOCR 依赖包")
    logger.info("=" * 50)

    for package in packages:
        logger.info(f"安装 {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            logger.info(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {package} 安装失败: {e}")
            return False

    logger.info("\n" + "=" * 50)
    logger.info("所有依赖包安装完成！")
    logger.info("=" * 50)
    return True


def test_import():
    """测试导入"""
    logger.info("\n测试模块导入...")

    try:
        from rapidocr import RapidOCR
        logger.info("✓ RapidOCR 导入成功")
    except ImportError as e:
        logger.error(f"✗ RapidOCR 导入失败: {e}")
        return False

    try:
        from wired_table_rec import WiredTableRecognition
        logger.info("✓ wired_table_rec 导入成功")
    except ImportError as e:
        logger.error(f"✗ wired_table_rec 导入失败: {e}")
        return False

    try:
        from lineless_table_rec import LinelessTableRecognition
        logger.info("✓ lineless_table_rec 导入成功")
    except ImportError as e:
        logger.error(f"✗ lineless_table_rec 导入失败: {e}")
        return False

    try:
        from table_cls import TableCls
        logger.info("✓ table_cls 导入成功")
    except ImportError as e:
        logger.warning(f"⚠ table_cls 导入失败（可选）: {e}")

    try:
        from bs4 import BeautifulSoup
        logger.info("✓ beautifulsoup4 导入成功")
    except ImportError as e:
        logger.error(f"✗ beautifulsoup4 导入失败: {e}")
        return False

    try:
        from ocr_system.rapid_table_ocr import RapidTableOCR
        logger.info("✓ RapidTableOCR 模块导入成功")
    except ImportError as e:
        logger.error(f"✗ RapidTableOCR 模块导入失败: {e}")
        return False

    logger.info("\n所有模块导入测试通过！")
    return True


def test_basic_functionality():
    """测试基本功能"""
    logger.info("\n测试基本功能...")

    try:
        from ocr_system.rapid_table_ocr import RapidTableOCR

        # 创建 OCR 实例
        ocr = RapidTableOCR(use_table_cls=False)  # 暂时不使用表格分类
        logger.info("✓ RapidTableOCR 实例创建成功")

        # 检查测试图像
        import os
        test_images = ["handwrite.jpg", "handwrite2.jpg", "handwrite3.jpg"]
        test_image = None
        for img in test_images:
            if os.path.exists(img):
                test_image = img
                break

        if test_image:
            logger.info(f"使用测试图像: {test_image}")

            # 测试表格分类
            table_type = ocr.classify_table(test_image)
            logger.info(f"✓ 表格分类: {table_type}")

            # 测试表格识别（不含 OCR，避免下载模型）
            logger.info("测试表格识别...")
            try:
                result = ocr.recognize_table(
                    test_image,
                    need_ocr=False  # 不进行 OCR，只测试表格结构检测
                )
                logger.info(f"✓ 表格识别成功，耗时: {result['elapse']:.2f}s")
                logger.info(f"  表格类型: {result['table_type']}")
                logger.info(f"  HTML 长度: {len(result['html']) if result['html'] else 0}")
            except Exception as e:
                logger.warning(f"⚠ 表格识别测试跳过（需要下载模型）: {e}")

        logger.info("\n基本功能测试完成！")
        return True

    except Exception as e:
        logger.error(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="安装和测试 RapidTableOCR")
    parser.add_argument("--install", action="store_true", help="安装依赖包")
    parser.add_argument("--test", action="store_true", help="测试导入")
    parser.add_argument("--run", action="store_true", help="运行完整测试")
    parser.add_argument("--all", action="store_true", help="执行所有步骤")

    args = parser.parse_args()

    if args.all or args.install:
        if not install_packages():
            sys.exit(1)

    if args.all or args.test:
        if not test_import():
            sys.exit(1)

    if args.all or args.run:
        if not test_basic_functionality():
            sys.exit(1)

    if not any([args.install, args.test, args.run, args.all]):
        # 默认执行所有步骤
        if install_packages() and test_import() and test_basic_functionality():
            logger.info("\n" + "=" * 50)
            logger.info("所有测试通过！RapidTableOCR 已准备就绪。")
            logger.info("=" * 50)
            logger.info("\n使用示例:")
            logger.info("  from ocr_system.rapid_table_ocr import RapidTableOCR")
            logger.info("  ocr = RapidTableOCR()")
            logger.info("  result = ocr.process_periodontal_chart('handwrite.jpg')")
        else:
            sys.exit(1)


if __name__ == '__main__':
    main()
