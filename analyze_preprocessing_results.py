"""
汇总预处理测试结果
"""
import json
import os
from pathlib import Path

output_dir = 'output/preprocessing_test'
results = {}

# 读取所有结果文件
for json_file in os.listdir(output_dir):
    if not json_file.endswith('.json'):
        continue

    filepath = os.path.join(output_dir, json_file)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取图像名称
    if '_no_preprocess.json' in json_file:
        img_name = json_file.replace('_no_preprocess.json', '')
        if img_name not in results:
            results[img_name] = {}
        results[img_name]['no'] = data['statistics']
    elif '_with_preprocess.json' in json_file:
        img_name = json_file.replace('_with_preprocess.json', '')
        if img_name not in results:
            results[img_name] = {}
        results[img_name]['with'] = data['statistics']

# 打印汇总表
print("=" * 80)
print("预处理效果批量测试汇总")
print("=" * 80)
print()

# 表头
print(f"{'图像':<25} | {'总牙数':^8} | {'PD':^6} | {'BOP':^6} | {'PI':^6} | {'分叉':^6} | {'松动':^6} | {'总改进':^6}")
print("-" * 80)

total_improvement = 0
for img_name in sorted(results.keys()):
    if 'no' not in results[img_name] or 'with' not in results[img_name]:
        continue

    no_stats = results[img_name]['no']
    with_stats = results[img_name]['with']

    # 计算差异
    teeth_diff = with_stats['total_teeth'] - no_stats['total_teeth']
    pd_diff = with_stats['teeth_with_pd'] - no_stats['teeth_with_pd']
    bop_diff = with_stats['teeth_with_bop'] - no_stats['teeth_with_bop']
    pi_diff = with_stats['teeth_with_pi'] - no_stats['teeth_with_pi']
    furc_diff = with_stats['teeth_with_furcation'] - no_stats['teeth_with_furcation']
    mob_diff = with_stats['teeth_with_mobility'] - no_stats['teeth_with_mobility']

    # 计算总体改进（PD+BOP+分叉+松动）
    improvement = pd_diff + bop_diff + furc_diff + mob_diff
    total_improvement += improvement

    # 格式化差异
    def fmt_diff(val):
        return f"{'+' if val > 0 else ''}{val}" if val != 0 else "0"

    print(f"{img_name:<25} | {teeth_diff:^8} | {fmt_diff(pd_diff):^6} | {fmt_diff(bop_diff):^6} | {fmt_diff(pi_diff):^6} | {fmt_diff(furc_diff):^6} | {fmt_diff(mob_diff):^6} | {fmt_diff(improvement):^6}")

print("-" * 80)
print(f"{'总计':<25} | {'':^8} | {'':^6} | {'':^6} | {'':^6} | {'':^6} | {'':^6} | {fmt_diff(total_improvement):^6}")
print()

# 结论
if total_improvement > 0:
    print("结论: 预处理提升了识别效果")
else:
    print("结论: 预处理降低了识别效果，建议调整参数或禁用预处理")

print()
print("=" * 80)
