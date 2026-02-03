"""
统一主程序入口 - perio_OCR 牙周图表 OCR 系统

整合所有 OCR 模块，提供统一的接口和简洁的使用方式。

支持的 OCR 模式:
- auto: 自动选择最佳 OCR 方法
- rapid: RapidTableOCR (TableStructureRec + RapidOCR)
- enhanced: EnhancedPerioOCR (PaddleOCR + 后处理优化)
- paddlex: PaddleXOCR (PaddleOCR PP-Structure)
- modular: ModularOCR (PaddleOCR 独立模块)

使用示例:
    # 基本用法
    python main.py --image handwrite.jpg

    # 指定输出文件
    python main.py --image handwrite.jpg --output result.json

    # 选择 OCR 模式
    python main.py --image handwrite.jpg --mode rapid

    # 调试模式
    python main.py --image handwrite.jpg --debug --visualize
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np


def setup_logging(level: str = 'INFO', debug: bool = False):
    """配置日志系统"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    if debug:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'perio_ocr_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


class PerioOCR:
    """
    统一的牙周图表 OCR 接口

    整合所有 OCR 模块，提供一致的 API 和数据格式
    """

    # 支持的 OCR 模式
    MODES = ['auto', 'rapid', 'enhanced', 'paddlex', 'modular']

    # 牙齿编号常量
    UPPER_TEETH_RIGHT = ['18', '17', '16', '15', '14', '13', '12', '11']
    UPPER_TEETH_LEFT = ['21', '22', '23', '24', '25', '26', '27', '28']
    LOWER_TEETH_LEFT = ['31', '32', '33', '34', '35', '36', '37', '38']
    LOWER_TEETH_RIGHT = ['48', '47', '46', '45', '44', '43', '42', '41']

    def __init__(
        self,
        mode: str = 'auto',
        enable_preprocessing: bool = True,
        enable_postprocess: bool = True,
        debug: bool = False
    ):
        """
        初始化 PerioOCR 系统

        Args:
            mode: OCR 模式 ('auto', 'rapid', 'enhanced', 'paddlex', 'modular')
            enable_preprocessing: 是否启用图像预处理
            enable_postprocess: 是否启用后处理优化
            debug: 调试模式
        """
        if mode not in self.MODES:
            raise ValueError(f"不支持的 mode: {mode}，可选: {self.MODES}")

        self.mode = mode
        self.enable_preprocessing = enable_preprocessing
        self.enable_postprocess = enable_postprocess
        self.debug = debug

        # OCR 引擎（延迟初始化）
        self._ocr_engine = None
        self._preprocessor = None
        self._post_processor = None

        logger.info(f"PerioOCR 初始化完成 (mode={mode}, preprocessing={enable_preprocessing}, postprocess={enable_postprocess})")

    def _init_ocr_engine(self):
        """延迟初始化 OCR 引擎"""
        if self._ocr_engine is not None:
            return

        mode = self.mode if self.mode != 'auto' else self._auto_select_mode()

        if mode == 'rapid':
            try:
                from ocr_system.rapid_table_ocr import RapidTableOCR
                self._ocr_engine = RapidTableOCR(use_table_cls=True, use_cuda=False, device="cpu")
                logger.info("使用 RapidTableOCR 模式")
            except ImportError as e:
                logger.warning(f"RapidTableOCR 不可用: {e}，切换到 enhanced 模式")
                self._ocr_engine = self._init_enhanced_ocr()

        elif mode == 'enhanced':
            self._ocr_engine = self._init_enhanced_ocr()

        elif mode == 'paddlex':
            try:
                from ocr_system.paddlex_ocr import PaddleXOCR
                self._ocr_engine = PaddleXOCR(use_table=True, lang='ch', use_angle_cls=True)
                logger.info("使用 PaddleXOCR 模式")
            except ImportError as e:
                logger.warning(f"PaddleXOCR 不可用: {e}，切换到 enhanced 模式")
                self._ocr_engine = self._init_enhanced_ocr()

        elif mode == 'modular':
            try:
                from ocr_system.modular_ocr import ModularOCR
                self._ocr_engine = ModularOCR(
                    enable_doc_orientation=True,
                    enable_unwarping=False,
                    enable_textline_orientation=False
                )
                logger.info("使用 ModularOCR 模式")
            except ImportError as e:
                logger.warning(f"ModularOCR 不可用: {e}，切换到 enhanced 模式")
                self._ocr_engine = self._init_enhanced_ocr()

    def _init_enhanced_ocr(self):
        """初始化增强型 OCR"""
        from ocr_system.enhanced_periodontal_ocr import EnhancedPerioOCR
        logger.info("使用 EnhancedPerioOCR 模式")
        return EnhancedPerioOCR()

    def _auto_select_mode(self) -> str:
        """自动选择最佳 OCR 模式"""
        # 优先级: rapid > enhanced > paddlex
        try:
            import rapidocr
            import wired_table_rec
            import lineless_table_rec
            return 'rapid'
        except ImportError:
            pass

        try:
            from paddleocr import PaddleOCR
            return 'enhanced'
        except ImportError:
            pass

        raise RuntimeError("没有可用的 OCR 引擎，请安装依赖: pip install paddleocr 或 pip install rapidocr wired_table_rec lineless_table_rec")

    def _init_preprocessor(self):
        """初始化预处理器"""
        if self._preprocessor is not None:
            return

        if self.enable_preprocessing:
            try:
                from ocr_system.image_preprocessor import AdaptivePreprocessor
                self._preprocessor = AdaptivePreprocessor()
                logger.info("图像预处理器已启用")
            except ImportError:
                logger.warning("图像预处理器不可用")
                self._preprocessor = None

    def _init_post_processor(self):
        """初始化后处理器"""
        if self._post_processor is not None:
            return

        if self.enable_postprocess:
            try:
                from ocr_system.post_processor import PostProcessor
                self._post_processor = PostProcessor()
                logger.info("后处理器已启用")
            except ImportError:
                logger.warning("后处理器不可用")
                self._post_processor = None

    def process(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        visualize: bool = False
    ) -> Dict:
        """
        处理牙周图表图像

        Args:
            image_path: 输入图像路径
            output_path: 输出 JSON 文件路径（可选）
            visualize: 是否生成可视化图像

        Returns:
            {
                'teeth_data': Dict,      # 牙齿数据
                'statistics': Dict,      # 统计信息
                'metadata': Dict         # 元数据
            }
        """
        start_time = time.time()

        logger.info(f"开始处理: {image_path}")

        # 验证图像文件
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 初始化组件
        self._init_ocr_engine()
        if self.enable_preprocessing:
            self._init_preprocessor()
        if self.enable_postprocess:
            self._init_post_processor()

        # 图像预处理（可选）
        processed_image_path = image_path
        if self._preprocessor:
            try:
                import tempfile
                import os

                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"preprocessed_{Path(image_path).name}")

                processed_img = self._preprocessor.preprocess(image_path, temp_path)
                processed_image_path = temp_path
                logger.info("图像预处理完成")
            except Exception as e:
                logger.warning(f"图像预处理失败，使用原始图像: {e}")

        # 执行 OCR 识别
        try:
            if self.mode == 'rapid' or (self.mode == 'auto' and hasattr(self._ocr_engine, 'process_periodontal_chart')):
                raw_result = self._ocr_engine.process_periodontal_chart(processed_image_path)
                result = self._normalize_rapid_result(raw_result)

            elif self.mode == 'enhanced' or (self.mode == 'auto' and hasattr(self._ocr_engine, 'process')):
                raw_result = self._ocr_engine.process(processed_image_path, enable_preprocessing=False)
                result = self._normalize_enhanced_result(raw_result)

            elif self.mode == 'paddlex' or hasattr(self._ocr_engine, 'process_periodontal_chart'):
                raw_result = self._ocr_engine.process_periodontal_chart(processed_image_path)
                result = self._normalize_paddlex_result(raw_result)

            elif self.mode == 'modular' or hasattr(self._ocr_engine, 'process'):
                texts, boxes, raw_results = self._ocr_engine.process(processed_image_path, return_raw_results=True)
                result = self._normalize_modular_result(texts, boxes, raw_results, processed_image_path)

            else:
                raise RuntimeError(f"未知的 OCR 引擎类型")

        except Exception as e:
            logger.error(f"OCR 识别失败: {e}")
            raise

        # 后处理优化
        if self._post_processor and result.get('teeth_data'):
            result['teeth_data'] = self._apply_postprocessing(result['teeth_data'])
            logger.info("后处理优化完成")

        # 添加元数据
        processing_time = time.time() - start_time
        result['metadata'] = {
            'processing_time': round(processing_time, 2),
            'method': self._ocr_engine.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'image_path': str(image_path),
            'mode': self.mode,
            'preprocessing_enabled': self.enable_preprocessing,
            'postprocess_enabled': self.enable_postprocess
        }

        # 可视化
        if visualize:
            self._visualize_result(image_path, result)

        # 保存结果
        if output_path:
            self._save_result(result, output_path)
            logger.info(f"结果已保存: {output_path}")

        logger.info(f"处理完成，耗时: {processing_time:.2f}s")
        return result

    def _normalize_rapid_result(self, raw_result: Dict) -> Dict:
        """标准化 RapidTableOCR 结果"""
        teeth_data = {}

        # 从 raw_result 中提取牙齿数据
        for key, value in raw_result.items():
            if key.startswith(('pd_', 'gm_', 'bop_', 'pi_', 'mobility_', 'furcation_')):
                # 解析键名，如: pd_11_b -> tooth=11, field=pd, surface=b
                parts = key.split('_')
                if len(parts) >= 3:
                    tooth_num = parts[1]
                    field_name = parts[0]

                    if tooth_num not in teeth_data:
                        teeth_data[tooth_num] = self._create_empty_tooth_data(tooth_num)

                    # 设置值
                    if field_name in ['pd', 'gm', 'bop', 'pi']:
                        surface = parts[2] if len(parts) > 2 else 'b'
                        if surface in teeth_data[tooth_num][field_name]:
                            teeth_data[tooth_num][field_name][surface] = value
                    elif field_name == 'mobility':
                        teeth_data[tooth_num]['mobility'] = value
                    elif field_name == 'furcation':
                        teeth_data[tooth_num]['furcation'] = value

        statistics = self._calculate_statistics(teeth_data)

        return {
            'teeth_data': teeth_data,
            'statistics': statistics,
            'raw_result': raw_result
        }

    def _normalize_enhanced_result(self, raw_result: Dict) -> Dict:
        """标准化 EnhancedPerioOCR 结果"""
        # EnhancedPerioOCR 已经返回标准格式
        return raw_result

    def _normalize_paddlex_result(self, raw_result: Dict) -> Dict:
        """标准化 PaddleXOCR 结果"""
        teeth_data = {}

        for key, value in raw_result.items():
            if key.startswith(('pd_', 'gm_', 'bop_', 'pi_', 'mobility_', 'furcation_')):
                parts = key.split('_')
                if len(parts) >= 3:
                    tooth_num = parts[1]
                    field_name = parts[0]

                    if tooth_num not in teeth_data:
                        teeth_data[tooth_num] = self._create_empty_tooth_data(tooth_num)

                    if field_name in ['pd', 'gm', 'bop', 'pi']:
                        surface = parts[2] if len(parts) > 2 else 'b'
                        if surface in teeth_data[tooth_num][field_name]:
                            teeth_data[tooth_num][field_name][surface] = value
                    elif field_name == 'mobility':
                        teeth_data[tooth_num]['mobility'] = value
                    elif field_name == 'furcation':
                        teeth_data[tooth_num]['furcation'] = value

        statistics = self._calculate_statistics(teeth_data)

        return {
            'teeth_data': teeth_data,
            'statistics': statistics,
            'raw_result': raw_result
        }

    def _normalize_modular_result(
        self,
        texts: list,
        boxes: np.ndarray,
        raw_results: Dict,
        image_path: str
    ) -> Dict:
        """标准化 ModularOCR 结果"""
        # 读取图像获取尺寸
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        teeth_data = {}
        mid_x, mid_y = w // 2, h // 2

        for idx, text in enumerate(texts):
            if idx >= len(boxes):
                continue

            box = boxes[idx]
            if len(box.shape) == 2:
                x_coords = box[:, 0]
                y_coords = box[:, 1]
            else:
                x_coords = [box[0], box[2]]
                y_coords = [box[1], box[3]]

            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))

            # 定位牙齿
            tooth_num, surface = self._locate_tooth(center_x, center_y, w, h)

            if tooth_num and text.strip().isdigit():
                value = int(text.strip())
                if 0 <= value <= 15:  # 合理的 PD 范围
                    if tooth_num not in teeth_data:
                        teeth_data[tooth_num] = self._create_empty_tooth_data(tooth_num)
                    teeth_data[tooth_num]['pd'][surface] = value

        statistics = self._calculate_statistics(teeth_data)

        return {
            'teeth_data': teeth_data,
            'statistics': statistics,
            'raw_results': raw_results
        }

    def _locate_tooth(self, x: int, y: int, w: int, h: int) -> tuple:
        """根据坐标定位牙齿和测量面"""
        mid_x, mid_y = w // 2, h // 2

        # 确定象限
        if y < mid_y:
            quadrant = self.UPPER_TEETH_RIGHT if x > mid_x else self.UPPER_TEETH_LEFT
        else:
            quadrant = self.LOWER_TEETH_RIGHT if x > mid_x else self.LOWER_TEETH_LEFT

        # 计算相对位置
        x_start = mid_x if x > mid_x else 0
        x_end = w if x > mid_x else mid_x
        quadrant_width = x_end - x_start

        tooth_idx = int((x - x_start) / quadrant_width * len(quadrant))
        tooth_idx = max(0, min(tooth_idx, len(quadrant) - 1))

        tooth_num = quadrant[tooth_idx]
        surface = 'b'  # 简化

        return tooth_num, surface

    def _create_empty_tooth_data(self, tooth_num: str) -> Dict:
        """创建空的牙齿数据结构"""
        is_molar = tooth_num[1] in '345678'

        if is_molar:
            surfaces = ['db', 'b', 'mb', 'dp', 'p', 'mp']
        else:
            surfaces = ['db', 'b', 'mb', 'dl', 'l', 'ml']

        return {
            'tooth_num': tooth_num,
            'tooth': 1,
            'mobility': 0,
            'furcation': 0,
            'pd': {s: 0 for s in surfaces},
            'gm': {s: 0 for s in surfaces},
            'bop': {s: 0 for s in surfaces},
            'pi': {s: 0 for s in surfaces}
        }

    def _calculate_statistics(self, teeth_data: Dict) -> Dict:
        """计算统计信息"""
        return {
            'total_teeth': len(teeth_data),
            'teeth_with_pd': sum(1 for t in teeth_data.values() if any(t['pd'].values())),
            'teeth_with_bop': sum(1 for t in teeth_data.values() if any(t['bop'].values())),
            'teeth_with_pi': sum(1 for t in teeth_data.values() if any(t['pi'].values())),
            'teeth_with_furcation': sum(1 for t in teeth_data.values() if t['furcation'] > 0),
            'teeth_with_mobility': sum(1 for t in teeth_data.values() if t['mobility'] > 0)
        }

    def _apply_postprocessing(self, teeth_data: Dict) -> Dict:
        """应用后处理优化"""
        if not self._post_processor:
            return teeth_data

        # 这里可以添加各种后处理逻辑
        # 例如：异常值修正、医学规则验证等

        return teeth_data

    def _visualize_result(self, image_path: str, result: Dict):
        """生成可视化结果"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return

            # 在图像上绘制识别结果（简化版）
            vis_path = Path(image_path).stem + '_visualized.jpg'
            cv2.imwrite(vis_path, img)
            logger.info(f"可视化结果已保存: {vis_path}")
        except Exception as e:
            logger.warning(f"可视化失败: {e}")

    def _save_result(self, result: Dict, output_path: str):
        """保存结果到 JSON 文件"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def close(self):
        """释放资源"""
        if self._ocr_engine and hasattr(self._ocr_engine, 'close'):
            try:
                self._ocr_engine.close()
            except Exception as e:
                logger.warning(f"关闭 OCR 引擎时出错: {e}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='perio_OCR - 牙周图表 OCR 识别系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --image handwrite.jpg
  %(prog)s --image handwrite.jpg --output result.json
  %(prog)s --image handwrite.jpg --mode rapid
  %(prog)s --image handwrite.jpg --debug --visualize
        """
    )

    parser.add_argument(
        '--image', '-i',
        required=True,
        help='输入图像路径'
    )

    parser.add_argument(
        '--output', '-o',
        help='输出 JSON 文件路径 (默认: image_result.json)'
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['auto', 'rapid', 'enhanced', 'paddlex', 'modular'],
        default='auto',
        help='OCR 模式 (默认: auto)'
    )

    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='禁用图像预处理'
    )

    parser.add_argument(
        '--no-postprocess',
        action='store_true',
        help='禁用后处理优化'
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='调试模式'
    )

    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='生成可视化结果'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(level=args.log_level, debug=args.debug)

    # 设置默认输出路径
    if args.output is None:
        args.output = Path(args.image).stem + '_result.json'

    # 创建 OCR 实例
    try:
        ocr = PerioOCR(
            mode=args.mode,
            enable_preprocessing=not args.no_preprocessing,
            enable_postprocess=not args.no_postprocess,
            debug=args.debug
        )

        # 处理图像
        result = ocr.process(
            image_path=args.image,
            output_path=args.output,
            visualize=args.visualize
        )

        # 打印摘要
        print("\n" + "=" * 50)
        print("牙周图表 OCR 识别结果")
        print("=" * 50)
        print(f"处理方法: {result['metadata']['method']}")
        print(f"处理时间: {result['metadata']['processing_time']}s")
        print(f"\n统计信息:")
        stats = result['statistics']
        print(f"  检测到牙齿: {stats['total_teeth']} 颗")
        print(f"  有 PD 数据: {stats['teeth_with_pd']} 颗")
        print(f"  有 BOP 数据: {stats['teeth_with_bop']} 颗")
        print(f"  有 PI 数据: {stats['teeth_with_pi']} 颗")
        print(f"  有分叉病变: {stats['teeth_with_furcation']} 颗")
        print(f"  有松动度: {stats['teeth_with_mobility']} 颗")
        print(f"\n结果已保存到: {args.output}")
        print("=" * 50)

        # 释放资源
        ocr.close()

    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
