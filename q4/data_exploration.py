#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：女胎异常判定 - 数据探索性分析
根据Instruction.md的思路，对女胎数据进行探索性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FemaleFetusDataExplorer:
    """女胎数据探索性分析类"""
    
    def __init__(self, data_path):
        """
        初始化数据探索器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.female_data = None
        self.features = None
        self.target = None
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成，共{len(self.data)}条记录")
        print(f"数据列名：{list(self.data.columns)}")
        
    def filter_female_fetuses(self):
        """
        筛选女胎样本
        根据Instruction.md，女胎样本的"Y染色体浓度"列为空白
        """
        print("\n正在筛选女胎样本...")
        
        # 检查Y染色体浓度列
        y_chrom_col = None
        for col in self.data.columns:
            if 'Y染色体' in col or 'Y染色体浓度' in col:
                y_chrom_col = col
                break
        
        if y_chrom_col is None:
            print("警告：未找到Y染色体浓度列，使用所有数据")
            self.female_data = self.data.copy()
        else:
            # 筛选Y染色体浓度为空白（NaN或空字符串）的样本
            self.female_data = self.data[
                (self.data[y_chrom_col].isna()) | 
                (self.data[y_chrom_col] == '') |
                (self.data[y_chrom_col] == ' ')
            ].copy()
        
        print(f"女胎样本数量：{len(self.female_data)}")
        return self.female_data
    
    def identify_features_and_target(self):
        """
        识别特征和标签
        根据Instruction.md的定义：
        - 特征：Z值、GC含量、测序质量、孕妇生理指标等
        - 标签：判定结果（AB列），空白为正常，有内容为异常
        """
        print("\n正在识别特征和标签...")
        
        # 识别标签列（判定结果）
        target_col = None
        for col in self.data.columns:
            if '染色体的非整倍体' in col or 'AB' in col:
                target_col = col
                break
        
        if target_col is None:
            print("警告：未找到判定结果列，使用胎儿是否健康列")
            target_col = '胎儿是否健康'
        
        # 创建二分类标签：空白/是 = 正常(0)，有内容/否 = 异常(1)
        if target_col in self.female_data.columns:
            self.target = (self.female_data[target_col].fillna('').astype(str) != '是').astype(int)
            print(f"标签分布：正常={sum(self.target==0)}，异常={sum(self.target==1)}")
        else:
            print("警告：未找到标签列，创建虚拟标签")
            self.target = np.zeros(len(self.female_data))
        
        # 识别特征列
        feature_cols = []
        
        # Z值特征
        z_value_cols = []
        for col in self.female_data.columns:
            if 'Z值' in col and ('13号' in col or '18号' in col or '21号' in col or 'X染色体' in col):
                z_value_cols.append(col)
        
        # GC含量特征
        gc_cols = []
        for col in self.female_data.columns:
            if 'GC含量' in col:
                gc_cols.append(col)
        
        # 孕妇生理指标
        bmi_col = None
        age_col = None
        for col in self.female_data.columns:
            if 'BMI' in col or '孕妇BMI' in col:
                bmi_col = col
            elif '年龄' in col:
                age_col = col
        
        # 测序质量指标
        quality_cols = []
        for col in self.female_data.columns:
            if '被过滤' in col or '比例' in col:
                quality_cols.append(col)
        
        # 组合所有特征
        feature_cols.extend(z_value_cols)
        feature_cols.extend(gc_cols)
        if bmi_col:
            feature_cols.append(bmi_col)
        if age_col:
            feature_cols.append(age_col)
        feature_cols.extend(quality_cols)
        
        # 过滤掉不存在的列
        feature_cols = [col for col in feature_cols if col in self.female_data.columns]
        
        print(f"识别到的特征列：{feature_cols}")
        
        # 提取特征数据
        self.features = self.female_data[feature_cols].copy()
        
        # 处理缺失值
        self.features = self.features.fillna(self.features.median())
        
        print(f"特征矩阵形状：{self.features.shape}")
        print(f"特征列：{list(self.features.columns)}")
        
        return self.features, self.target
    
    def explore_data_distribution(self):
        """探索数据分布"""
        print("\n正在分析数据分布...")
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('女胎数据分布分析', fontsize=16, fontweight='bold')
        
        # 1. 标签分布
        target_counts = pd.Series(self.target).value_counts()
        axes[0, 0].pie(target_counts.values, labels=['正常', '异常'], autopct='%1.1f%%', 
                      colors=['lightgreen', 'lightcoral'])
        axes[0, 0].set_title('标签分布')
        
        # 2. Z值分布（如果有的话）
        z_cols = [col for col in self.features.columns if 'Z值' in col]
        if z_cols:
            for col in z_cols[:3]:  # 最多显示3个Z值
                axes[0, 1].hist(self.features[col], alpha=0.7, bins=30, label=col)
            axes[0, 1].set_title('Z值分布')
            axes[0, 1].legend()
            axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        else:
            axes[0, 1].text(0.5, 0.5, '无Z值数据', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Z值分布')
        
        # 3. BMI分布
        bmi_col = [col for col in self.features.columns if 'BMI' in col]
        if bmi_col:
            axes[1, 0].hist(self.features[bmi_col[0]], bins=30, alpha=0.7, color='skyblue')
            axes[1, 0].set_title('BMI分布')
            axes[1, 0].set_xlabel('BMI')
            axes[1, 0].set_ylabel('频次')
        else:
            axes[1, 0].text(0.5, 0.5, '无BMI数据', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('BMI分布')
        
        # 4. 年龄分布
        age_col = [col for col in self.features.columns if '年龄' in col]
        if age_col:
            axes[1, 1].hist(self.features[age_col[0]], bins=20, alpha=0.7, color='lightgreen')
            axes[1, 1].set_title('年龄分布')
            axes[1, 1].set_xlabel('年龄')
            axes[1, 1].set_ylabel('频次')
        else:
            axes[1, 1].text(0.5, 0.5, '无年龄数据', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('年龄分布')
        
        plt.tight_layout()
        plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/data_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印基本统计信息
        print("\n=== 数据基本统计信息 ===")
        print(f"样本总数：{len(self.female_data)}")
        print(f"特征数量：{self.features.shape[1]}")
        print(f"正常样本：{sum(self.target==0)} ({sum(self.target==0)/len(self.target)*100:.1f}%)")
        print(f"异常样本：{sum(self.target==1)} ({sum(self.target==1)/len(self.target)*100:.1f}%)")
        
        print("\n=== 特征统计信息 ===")
        print(self.features.describe())
    
    def analyze_z_values_by_gestational_week(self):
        """
        分析Z值随孕周的变化
        根据Instruction.md，绘制正常样本与异常样本的Z值随孕周变化的散点图
        """
        print("\n正在分析Z值随孕周的变化...")
        
        # 查找孕周列
        week_col = None
        for col in self.female_data.columns:
            if '孕周' in col or '检测孕周' in col:
                week_col = col
                break
        
        if week_col is None:
            print("未找到孕周列，跳过Z值-孕周分析")
            return
        
        # 查找Z值列
        z_cols = [col for col in self.features.columns if 'Z值' in col]
        if not z_cols:
            print("未找到Z值列，跳过Z值-孕周分析")
            return
        
        # 创建图形
        n_z_cols = len(z_cols)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        fig.suptitle('Z值随孕周变化分析', fontsize=16, fontweight='bold')
        
        for i, z_col in enumerate(z_cols[:4]):  # 最多显示4个Z值
            if i >= 4:
                break
                
            # 正常样本
            normal_mask = self.target == 0
            axes[i].scatter(self.female_data.loc[normal_mask, week_col], 
                          self.features.loc[normal_mask, z_col], 
                          alpha=0.6, label='正常', color='blue', s=30)
            
            # 异常样本
            abnormal_mask = self.target == 1
            axes[i].scatter(self.female_data.loc[abnormal_mask, week_col], 
                          self.features.loc[abnormal_mask, z_col], 
                          alpha=0.6, label='异常', color='red', s=30)
            
            axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[i].set_xlabel('孕周')
            axes[i].set_ylabel('Z值')
            axes[i].set_title(f'{z_col}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(z_cols), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/z_values_by_gestational_week.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_correlations(self):
        """分析特征间的相关性"""
        print("\n正在分析特征相关性...")
        
        # 计算相关性矩阵
        corr_matrix = self.features.corr()
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('特征相关性热力图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/feature_correlations.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存相关性数据
        corr_matrix.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/feature_correlations.csv')
        print("相关性分析完成，结果已保存")
    
    def run_exploration(self):
        """运行完整的探索性分析"""
        print("=== 女胎数据探索性分析开始 ===")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 筛选女胎样本
        self.filter_female_fetuses()
        
        # 3. 识别特征和标签
        self.identify_features_and_target()
        
        # 4. 探索数据分布
        self.explore_data_distribution()
        
        # 5. 分析Z值随孕周变化
        self.analyze_z_values_by_gestational_week()
        
        # 6. 分析特征相关性
        self.analyze_correlations()
        
        print("\n=== 探索性分析完成 ===")
        
        return self.features, self.target

def main():
    """主函数"""
    # 创建数据探索器
    explorer = FemaleFetusDataExplorer('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/girl_output.csv')
    
    # 运行探索性分析
    features, target = explorer.run_exploration()
    
    return features, target

if __name__ == "__main__":
    features, target = main()
