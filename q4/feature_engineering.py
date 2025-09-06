#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：女胎异常判定 - 特征工程
根据Instruction.md的思路，实现特征标准化和质量校正的Z值指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FemaleFetusFeatureEngineer:
    """女胎特征工程类"""
    
    def __init__(self, features, target):
        """
        初始化特征工程器
        
        Args:
            features: 特征数据 (DataFrame)
            target: 标签数据 (Series)
        """
        self.features = features.copy()
        self.target = target.copy()
        self.scaler = StandardScaler()
        self.healthy_stats = {}
        self.corrected_features = None
        self.final_features = None
        
    def standardize_features(self):
        """
        步骤3.1: 特征标准化 (Z-score Normalization)
        根据Instruction.md，使用健康孕妇群体的均值和标准差进行标准化
        """
        print("=== 步骤3.1: 特征标准化 ===")
        
        # 识别健康群体（标签为0的样本）
        healthy_mask = self.target == 0
        healthy_features = self.features[healthy_mask]
        
        print(f"健康样本数量：{sum(healthy_mask)}")
        print(f"异常样本数量：{sum(~healthy_mask)}")
        
        # 计算健康群体的统计量
        self.healthy_stats = {
            'mean': healthy_features.mean(),
            'std': healthy_features.std()
        }
        
        # 使用健康群体的统计量进行标准化
        standardized_features = (self.features - self.healthy_stats['mean']) / self.healthy_stats['std']
        
        # 处理无穷大和NaN值
        standardized_features = standardized_features.replace([np.inf, -np.inf], np.nan)
        standardized_features = standardized_features.fillna(0)
        
        print("特征标准化完成")
        print(f"标准化后特征统计：")
        print(standardized_features.describe())
        
        return standardized_features
    
    def build_quality_corrected_z_values(self, standardized_features):
        """
        步骤3.2: 构建经质量校正的Z值指标 (δ)
        根据Instruction.md，引入GC含量和测序质量校正因子
        """
        print("\n=== 步骤3.2: 构建质量校正的Z值指标 ===")
        
        corrected_features = standardized_features.copy()
        
        # 识别Z值列
        z_value_cols = [col for col in standardized_features.columns if 'Z值' in col]
        gc_cols = [col for col in standardized_features.columns if 'GC含量' in col]
        quality_cols = [col for col in standardized_features.columns if '被过滤' in col or '比例' in col]
        
        print(f"Z值列：{z_value_cols}")
        print(f"GC含量列：{gc_cols}")
        print(f"质量指标列：{quality_cols}")
        
        # 为每个Z值构建校正因子
        for z_col in z_value_cols:
            if 'X染色体' in z_col:
                continue  # 跳过X染色体Z值，它作为参考
            
            print(f"\n处理 {z_col}...")
            
            # 1. 构建GC含量校正因子
            gc_correction = 1.0
            if gc_cols:
                # 找到对应的GC含量列
                target_gc_col = None
                for gc_col in gc_cols:
                    if z_col.split('号')[0] in gc_col:
                        target_gc_col = gc_col
                        break
                
                if target_gc_col is not None:
                    # GC含量偏离正常范围(40%-60%)时降低权重
                    gc_content = standardized_features[target_gc_col]
                    gc_deviation = np.abs(gc_content - 0.5)  # 0.5对应50%GC含量
                    gc_correction = np.exp(-gc_deviation * 2)  # 指数衰减
                    print(f"  GC含量校正因子范围：{gc_correction.min():.3f} - {gc_correction.max():.3f}")
            
            # 2. 构建测序质量校正因子
            quality_correction = 1.0
            if quality_cols:
                # 使用被过滤读段比例作为质量指标
                for quality_col in quality_cols:
                    if '被过滤' in quality_col:
                        filtered_ratio = standardized_features[quality_col]
                        # 过滤比例越高，质量越差，权重越低
                        quality_correction = np.exp(-filtered_ratio * 3)
                        print(f"  测序质量校正因子范围：{quality_correction.min():.3f} - {quality_correction.max():.3f}")
                        break
            
            # 3. 应用校正因子
            original_z = standardized_features[z_col]
            corrected_z = original_z * gc_correction * quality_correction
            
            # 更新特征
            corrected_features[z_col] = corrected_z
            
            print(f"  原始Z值范围：{original_z.min():.3f} - {original_z.max():.3f}")
            print(f"  校正后Z值范围：{corrected_z.min():.3f} - {corrected_z.max():.3f}")
        
        self.corrected_features = corrected_features
        print("\n质量校正完成")
        
        return corrected_features
    
    def build_comprehensive_anomaly_signal(self, corrected_features):
        """
        构建综合异常信号 (δ)
        根据Instruction.md，将校正后的Z值与X染色体浓度、孕妇BMI等多因素进行融合
        """
        print("\n=== 构建综合异常信号 ===")
        
        # 识别关键特征
        z_value_cols = [col for col in corrected_features.columns if 'Z值' in col and 'X染色体' not in col]
        x_chrom_col = [col for col in corrected_features.columns if 'X染色体' in col and 'Z值' in col]
        bmi_col = [col for col in corrected_features.columns if 'BMI' in col]
        age_col = [col for col in corrected_features.columns if '年龄' in col]
        
        print(f"目标Z值列：{z_value_cols}")
        print(f"X染色体Z值列：{x_chrom_col}")
        print(f"BMI列：{bmi_col}")
        print(f"年龄列：{age_col}")
        
        # 为每个目标染色体构建综合异常信号
        anomaly_signals = pd.DataFrame(index=corrected_features.index)
        
        for z_col in z_value_cols:
            # 提取染色体号
            chrom_num = z_col.split('号')[0]
            signal_col = f'{chrom_num}号染色体综合异常信号'
            
            # 基础Z值
            base_z = corrected_features[z_col]
            
            # X染色体参考信号
            x_signal = corrected_features[x_chrom_col[0]] if x_chrom_col else 0
            
            # BMI影响
            bmi_signal = corrected_features[bmi_col[0]] if bmi_col else 0
            
            # 年龄影响
            age_signal = corrected_features[age_col[0]] if age_col else 0
            
            # 构建综合异常信号
            # δ = w1*Z' + w2*X_chrom + w3*BMI + w4*Age
            # 这里使用简单的线性组合，权重将在后续学习中优化
            anomaly_signal = (
                0.5 * base_z +           # 主要Z值信号
                0.2 * x_signal +         # X染色体参考
                0.15 * bmi_signal +      # BMI影响
                0.15 * age_signal        # 年龄影响
            )
            
            anomaly_signals[signal_col] = anomaly_signal
            
            print(f"{signal_col} 范围：{anomaly_signal.min():.3f} - {anomaly_signal.max():.3f}")
        
        # 合并所有特征
        final_features = pd.concat([
            corrected_features,
            anomaly_signals
        ], axis=1)
        
        self.final_features = final_features
        print(f"\n综合异常信号构建完成，最终特征数量：{final_features.shape[1]}")
        
        return final_features
    
    def visualize_feature_transformation(self):
        """可视化特征变换过程"""
        print("\n=== 可视化特征变换过程 ===")
        
        if self.corrected_features is None or self.final_features is None:
            print("请先完成特征变换")
            return
        
        # 选择几个关键特征进行可视化
        z_value_cols = [col for col in self.features.columns if 'Z值' in col and 'X染色体' not in col][:3]
        
        if not z_value_cols:
            print("未找到合适的Z值列进行可视化")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('特征变换过程可视化', fontsize=16, fontweight='bold')
        
        for i, z_col in enumerate(z_value_cols):
            if i >= 3:
                break
            
            # 原始特征
            axes[0, i].hist(self.features[z_col], bins=30, alpha=0.7, color='lightblue', label='原始')
            axes[0, i].set_title(f'{z_col} - 原始')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # 校正后特征
            axes[1, i].hist(self.corrected_features[z_col], bins=30, alpha=0.7, color='lightgreen', label='校正后')
            axes[1, i].set_title(f'{z_col} - 质量校正后')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/feature_transformation.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 可视化综合异常信号
        anomaly_cols = [col for col in self.final_features.columns if '综合异常信号' in col]
        if anomaly_cols:
            fig, axes = plt.subplots(1, len(anomaly_cols), figsize=(5*len(anomaly_cols), 5))
            if len(anomaly_cols) == 1:
                axes = [axes]
            
            fig.suptitle('综合异常信号分布', fontsize=16, fontweight='bold')
            
            for i, col in enumerate(anomaly_cols):
                # 按标签分组显示
                normal_data = self.final_features.loc[self.target == 0, col]
                abnormal_data = self.final_features.loc[self.target == 1, col]
                
                axes[i].hist(normal_data, bins=20, alpha=0.7, label='正常', color='lightblue')
                axes[i].hist(abnormal_data, bins=20, alpha=0.7, label='异常', color='lightcoral')
                axes[i].set_title(col)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/anomaly_signals.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_processed_features(self):
        """保存处理后的特征"""
        if self.final_features is not None:
            # 保存最终特征
            self.final_features.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/processed_features.csv')
            
            # 保存标签
            pd.Series(self.target).to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/processed_targets.csv', 
                                         header=['target'])
            
            # 保存健康群体统计量
            healthy_stats_df = pd.DataFrame({
                'mean': self.healthy_stats['mean'],
                'std': self.healthy_stats['std']
            })
            healthy_stats_df.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/healthy_stats.csv')
            
            print("处理后的特征已保存")
            print(f"特征文件：processed_features.csv")
            print(f"标签文件：processed_targets.csv")
            print(f"统计量文件：healthy_stats.csv")
    
    def run_feature_engineering(self):
        """运行完整的特征工程流程"""
        print("=== 女胎特征工程开始 ===")
        
        # 1. 特征标准化
        standardized_features = self.standardize_features()
        
        # 2. 构建质量校正的Z值指标
        corrected_features = self.build_quality_corrected_z_values(standardized_features)
        
        # 3. 构建综合异常信号
        final_features = self.build_comprehensive_anomaly_signal(corrected_features)
        
        # 4. 可视化变换过程
        self.visualize_feature_transformation()
        
        # 5. 保存处理后的特征
        self.save_processed_features()
        
        print("\n=== 特征工程完成 ===")
        
        return final_features, self.target

def main():
    """主函数"""
    # 这里需要先运行数据探索获得特征和标签
    # 为了演示，我们创建一个简单的示例
    print("请先运行 data_exploration.py 获得特征和标签数据")
    return None, None

if __name__ == "__main__":
    main()
