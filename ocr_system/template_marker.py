"""
牙周图表标注工具 - 用于标注表格单元格位置
"""
import cv2
import numpy as np
import json
from pathlib import Path


class TemplateMarker:
    """牙周图表模板标注工具"""

    def __init__(self, template_image_path: str):
        self.image = cv2.imread(template_image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {template_image_path}")

        self.height, self.width = self.image.shape[:2]
        self.clicks = []
        self.annotations = {}

        # 窗口名
        self.window_name = 'Perio Chart Template Marker'

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x, y))
            print(f"点击位置 ({x}, {y})")

            # 绘制标记
            img_copy = self.image.copy()
            for i, (cx, cy) in enumerate(self.clicks):
                color = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)
                cv2.circle(img_copy, (cx, cy), 5, color, -1)
                cv2.putText(img_copy, str(i+1), (cx+10, cy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow(self.window_name, img_copy)

    def annotate_table_bounds(self) -> dict:
        """标注表格整体边界"""
        print("\n=== 步骤1: 标注表格边界 ===")
        print("请按顺序点击表格的4个角点:")
        print("  1. 左上角")
        print("  2. 右上角")
        print("  3. 右下角")
        print("  4. 左下角")

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while len(self.clicks) < 4:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC退出
                break

        cv2.destroyAllWindows()

        if len(self.clicks) == 4:
            corners = self.clicks
            self.annotations['table'] = {
                'corners': corners,
                'x': min(c[0] for c in corners),
                'y': min(c[1] for c in corners),
                'width': max(c[0] for c in corners) - min(c[0] for c in corners),
                'height': max(c[1] for c in corners) - min(c[1] for c in corners)
            }
            print(f"表格边界: {self.annotations['table']}")

        return self.annotations.get('table', {})

    def annotate_quadrants(self, table_bounds: dict):
        """标注4个象限"""
        print("\n=== 步骤2: 标注象限分割线 ===")
        print("请点击水平分割线和垂直分割线的交点:")
        print("  1. 中心点")

        self.clicks = []
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while len(self.clicks) < 1:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        cv2.destroyAllWindows()

        if len(self.clicks) == 1:
            cx, cy = self.clicks[0]
            self.annotations['quadrants'] = {
                'center_x': cx,
                'center_y': cy,
                'upper_left': (0, 0, cx, cy),
                'upper_right': (cx, 0, self.width - cx, cy),
                'lower_left': (0, cy, cx, self.height - cy),
                'lower_right': (cx, cy, self.width - cx, self.height - cy)
            }
            print(f"象限中心: ({cx}, {cy})")

        return self.annotations.get('quadrants', {})

    def annotate_tooth_rows(self, quadrant: str, quadrant_bounds: tuple, tooth_numbers: list):
        """标注每颗牙齿的起始Y坐标"""
        print(f"\n=== 步骤3: 标注 {quadrant} 区域的牙齿行 ===")
        print(f"请点击 {len(tooth_numbers)} 颗牙齿的起始位置（从上到下）:")
        for i, tn in enumerate(tooth_numbers):
            print(f"  {i+1}. 牙齿 {tn}")

        # 显示象限图像
        x, y, w, h = quadrant_bounds
        quad_img = self.image[y:y+h, x:x+w]

        cv2.namedWindow(f"{self.window_name} - {quadrant}")
        cv2.setMouseCallback(f"{self.window_name} - {quadrant}", self.mouse_callback)

        self.clicks = []
        while len(self.clicks) < len(tooth_numbers):
            cv2.imshow(f"{self.window_name} - {quadrant}", quad_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        cv2.destroyAllWindows()

        if len(self.clicks) == len(tooth_numbers):
            self.annotations[f'{quadrant}_teeth'] = {
                tooth_numbers[i]: self.clicks[i][1] for i in range(len(tooth_numbers))
            }
            print(f"牙齿Y坐标: {self.annotations[f'{quadrant}_teeth']}")

        return self.annotations.get(f'{quadrant}_teeth', {})

    def annotate_surface_columns(self, quadrant: str, quadrant_bounds: tuple):
        """标注6个测量面的列X坐标"""
        print(f"\n=== 步骤4: 标注 {quadrant} 区域的测量面列 ===")
        print("请点击6个测量面的列位置（从左到右）:")
        print("  1. db (远中颊)")
        print("  2. b (颊侧)")
        print("  3. mb (近中颊)")
        print("  4. dl/l (远中舌/腭)")
        print("  5. l (舌/腭侧)")
        print("  6. ml (近中舌/腭)")

        x, y, w, h = quadrant_bounds
        quad_img = self.image[y:y+h, x:x+w]

        cv2.namedWindow(f"{self.window_name} - {quadrant} surfaces")
        cv2.setMouseCallback(f"{self.window_name} - {quadrant} surfaces", self.mouse_callback)

        self.clicks = []
        while len(self.clicks) < 6:
            cv2.imshow(f"{self.window_name} - {quadrant} surfaces", quad_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        cv2.destroyAllWindows()

        if len(self.clicks) == 6:
            surfaces = ['db', 'b', 'mb', 'dl', 'l', 'ml']
            self.annotations[f'{quadrant}_surfaces'] = {
                surfaces[i]: self.clicks[i][0] for i in range(6)
            }
            print(f"测量面X坐标: {self.annotations[f'{quadrant}_surfaces']}")

        return self.annotations.get(f'{quadrant}_surfaces', {})

    def save_annotations(self, output_path: str):
        """保存标注结果到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        print(f"标注结果已保存到: {output_path}")

    def load_annotations(self, input_path: str) -> dict:
        """加载已有标注"""
        with open(input_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        return self.annotations


def main():
    """主函数 - 交互式标注"""
    import sys

    if len(sys.argv) < 2:
        print("用法: python template_marker.py <图像路径>")
        print("示例: python template_marker.py electronic.pdf")
        sys.exit(1)

    image_path = sys.argv[1]
    marker = TemplateMarker(image_path)

    # 标注表格边界
    table_bounds = marker.annotate_table_bounds()

    # 标注象限
    quadrants = marker.annotate_quadrants(table_bounds)

    # 定义4个象限的牙齿编号
    quadrant_teeth = {
        'upper_right': ['18', '17', '16', '15', '14', '13', '12', '11'],
        'upper_left': ['21', '22', '23', '24', '25', '26', '27', '28'],
        'lower_left': ['31', '32', '33', '34', '35', '36', '37', '38'],
        'lower_right': ['41', '42', '43', '44', '45', '46', '47', '48']
    }

    # 标注每个象限的牙齿行
    for quad_name, teeth in quadrant_teeth.items():
        if quad_name in quadrants:
            marker.annotate_tooth_rows(quad_name, quadrants[quad_name], teeth)

    # 标注每个象限的测量面
    for quad_name, teeth in quadrant_teeth.items():
        if quad_name in quadrants:
            marker.annotate_surface_columns(quad_name, quadrants[quad_name])

    # 保存标注结果
    output_path = 'ocr_system/template_config.json'
    marker.save_annotations(output_path)
    print("\n标注完成！生成的文件可用于OCR识别。")


if __name__ == '__main__':
    main()
