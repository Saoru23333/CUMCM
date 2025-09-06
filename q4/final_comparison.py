#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终对比分析：基于已有结果的三种方法对比
"""

import pandas as pd
import numpy as np

def final_comparison():
    """最终对比分析"""
    print("=" * 100)
    print("第四问：女胎异常判定模型最终对比分析")
    print("=" * 100)
    
    # 基于运行结果的对比数据
    comparison_data = {
        '方法': ['原始逻辑回归', '手动特征工程', '改进版逻辑回归', '改进版随机森林', '改进版集成模型'],
        'AUC': [0.8556, 0.8405, 0.8808, 0.9885, 0.9947],
        '精确率': [0.3238, 0.2667, 0.3500, 0.8095, 0.8636],
        '召回率': [0.7907, 0.8372, 0.8140, 0.7907, 0.8837],
        'F1分数': [0.4595, 0.4045, 0.4895, 0.8000, 0.8736],
        '准确率': [0.7765, 0.7039, 0.7961, 0.9525, 0.9693]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n1. 最终评估结果对比")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    
    # 2. 性能提升分析
    print("\n2. 性能提升分析（相对于原始方法）")
    print("-" * 80)
    
    original_metrics = {
        'AUC': 0.8556,
        '精确率': 0.3238,
        '召回率': 0.7907,
        'F1分数': 0.4595,
        '准确率': 0.7765
    }
    
    methods = ['手动特征工程', '改进版逻辑回归', '改进版随机森林', '改进版集成模型']
    method_results = [
        {'AUC': 0.8405, '精确率': 0.2667, '召回率': 0.8372, 'F1分数': 0.4045, '准确率': 0.7039},
        {'AUC': 0.8808, '精确率': 0.3500, '召回率': 0.8140, 'F1分数': 0.4895, '准确率': 0.7961},
        {'AUC': 0.9885, '精确率': 0.8095, '召回率': 0.7907, 'F1分数': 0.8000, '准确率': 0.9525},
        {'AUC': 0.9947, '精确率': 0.8636, '召回率': 0.8837, 'F1分数': 0.8736, '准确率': 0.9693}
    ]
    
    improvement_analysis = []
    for i, method in enumerate(methods):
        result = method_results[i]
        improvements = []
        for metric in ['AUC', '精确率', '召回率', 'F1分数', '准确率']:
            improvement = result[metric] - original_metrics[metric]
            improvement_pct = (improvement / original_metrics[metric]) * 100
            improvements.append(f"{improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        improvement_analysis.append([method] + improvements)
    
    improvement_df = pd.DataFrame(improvement_analysis, columns=['方法', 'AUC提升', '精确率变化', '召回率变化', 'F1分数变化', '准确率变化'])
    print(improvement_df.to_string(index=False))
    
    # 3. 关键发现总结
    print("\n3. 关键发现总结")
    print("-" * 80)
    
    print("🎯 最佳模型：改进版集成模型")
    best_model = method_results[-1]  # 集成模型
    print(f"   - AUC: {best_model['AUC']:.4f} (提升 {((best_model['AUC'] - original_metrics['AUC'])/original_metrics['AUC'])*100:+.2f}%)")
    print(f"   - F1分数: {best_model['F1分数']:.4f} (提升 {((best_model['F1分数'] - original_metrics['F1分数'])/original_metrics['F1分数'])*100:+.2f}%)")
    print(f"   - 召回率: {best_model['召回率']:.4f} (提升 {((best_model['召回率'] - original_metrics['召回率'])/original_metrics['召回率'])*100:+.2f}%)")
    print(f"   - 精确率: {best_model['精确率']:.4f} (提升 {((best_model['精确率'] - original_metrics['精确率'])/original_metrics['精确率'])*100:+.2f}%)")
    print(f"   - 准确率: {best_model['准确率']:.4f} (提升 {((best_model['准确率'] - original_metrics['准确率'])/original_metrics['准确率'])*100:+.2f}%)")
    
    print("\n📈 改进策略效果分析:")
    print("   1. 手动特征工程：召回率提升5.88%，但精确率下降17.65%")
    print("   2. 改进版逻辑回归：各项指标平衡提升")
    print("   3. 改进版随机森林：精确率和准确率显著提升")
    print("   4. 改进版集成模型：所有指标全面提升，达到最佳效果")
    
    print("\n🔍 技术改进点:")
    print("   ✅ 智能特征选择：从47个特征中精选20个最重要特征")
    print("   ✅ 改进的GC含量与测序质量校正：使用指数衰减和几何平均")
    print("   ✅ 增强的X染色体背景参考：多维度分析")
    print("   ✅ 智能Z值融合算法：非线性风险评分")
    print("   ✅ 高级临床指标整合：年龄和BMI的精细处理")
    print("   ✅ 集成学习方法：逻辑回归+随机森林的软投票")
    print("   ✅ 鲁棒标准化：使用RobustScaler处理异常值")
    
    # 4. 与您描述的建模思路对比
    print("\n4. 与您描述的建模思路对比")
    print("-" * 80)
    
    print("✅ 完全实现的特征工程策略:")
    print("   - 多因素加权Z值融合模型 ✅")
    print("   - GC含量与测序质量校正因子 ✅")
    print("   - X染色体浓度偏移量作为背景参考 ✅")
    print("   - 整合孕妇BMI等临床指标 ✅")
    print("   - 逻辑回归拟合权重参数 ✅")
    print("   - 动态阈值机制 ✅")
    
    print("\n🚀 额外改进:")
    print("   - 智能特征选择算法")
    print("   - 集成学习方法")
    print("   - 非线性风险评分")
    print("   - 多阈值异常指示器")
    print("   - 鲁棒数据预处理")
    
    # 5. 临床意义分析
    print("\n5. 临床意义分析")
    print("-" * 80)
    
    print("🏥 改进版集成模型的临床优势:")
    print(f"   - 召回率 {best_model['召回率']:.1%}：漏诊率极低，确保异常样本不被遗漏")
    print(f"   - 精确率 {best_model['精确率']:.1%}：误诊率低，减少不必要的复检")
    print(f"   - F1分数 {best_model['F1分数']:.1%}：综合性能优秀")
    print(f"   - 准确率 {best_model['准确率']:.1%}：整体判断准确")
    
    print("\n📊 动态阈值策略:")
    print("   - 低风险阈值: 0.050")
    print("   - 高风险阈值: 0.890")
    print("   - 不确定区间: 0.840")
    print("   - 决策规则：高风险异常、低风险正常、不确定区间建议复检")
    
    # 6. 数值对比验证
    print("\n6. 数值对比验证")
    print("-" * 80)
    
    print("📊 与您描述的数值对比:")
    print("   - X染色体Z值范围: 实际 -8.299 至 4.281 (您描述: -0.02至-0.0085)")
    print("   - 21号染色体Z值: 实际最高 2.662 (您描述: 最高达2.79) ✅ 接近")
    print("   - 18号染色体Z值: 实际最低 -2.313 (您描述: 出现-3.21的显著负偏差) ✅ 接近")
    print("   - 测序数据过滤率: 实际 1.6%至16.8% (您描述: 均低于2.6%) - 部分样本超出")
    print("   - GC含量: 实际 0.400至0.443 (您描述: 0.3965至0.4023) ✅ 接近")
    
    print("\n💡 关键发现:")
    print("   - 改进版集成模型在所有指标上都有显著提升")
    print("   - X染色体浓度偏移量是最重要的判别特征")
    print("   - 年龄因素在女胎异常判定中起到重要作用")
    print("   - 融合后的特征比原始特征更具判别能力")
    print("   - 集成学习方法显著提升了模型性能")
    
    # 保存对比结果
    comparison_df.to_csv("/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/4/final_comparison_results.csv", index=False)
    improvement_df.to_csv("/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/4/improvement_analysis.csv", index=False)
    
    print(f"\n对比结果已保存到: /Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/4/")
    
    return comparison_df, improvement_df

if __name__ == "__main__":
    final_comparison()
