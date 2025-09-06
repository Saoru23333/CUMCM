#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三问：多目标优化检测时点
基于风险驱动的BMI分组，采用多目标优化方法确定各组的最优检测时点
与第二问保持一致的多目标优化框架
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import os
from multivar_optimizer import MultivariateRiskOptimizer
from multivar_model import MultivariateRiskModel

class DetectionTimeOptimizer:
    """多目标优化检测时点优化器"""
    
    def __init__(self, risk_scores_path, save_dir):
        self.risk_scores_path = risk_scores_path
        self.save_dir = save_dir
        self.risk_scores = None
        self.risk_groups = None
        self.bmi_groups = None
        self.optimization_results = None
        self.multivariate_model = None
        self.optimizer = None
    
    def load_risk_scores(self):
        """加载个体风险评分数据"""
        print("正在加载个体风险评分数据...")
        self.risk_scores = pd.read_csv(self.risk_scores_path)
        print(f"加载完成，共{len(self.risk_scores)}位孕妇")
        return self.risk_scores
    
    def perform_risk_clustering(self, n_clusters=3):
        """对风险评分进行聚类（风险分层）"""
        print(f"正在进行风险分层聚类（{n_clusters}个组）...")
        
        # 使用K-means对达标时间进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        risk_clusters = kmeans.fit_predict(self.risk_scores[['achievement_time']].values)
        
        # 添加风险分层标签
        self.risk_scores['risk_group'] = risk_clusters
        
        # 按风险评分排序，确保标签含义一致
        group_means = self.risk_scores.groupby('risk_group')['achievement_time'].mean().sort_values()
        group_mapping = {old_label: new_label for new_label, old_label in enumerate(group_means.index)}
        self.risk_scores['risk_group'] = self.risk_scores['risk_group'].map(group_mapping)
        
        print("风险分层聚类完成")
        return self.risk_scores
    
    def map_risk_to_bmi_groups(self):
        """将风险分层映射回BMI分组"""
        print("正在将风险分层映射回BMI分组...")
        
        # 使用决策树将风险分层映射到BMI分组
        X = self.risk_scores[['avg_bmi']].values
        y = self.risk_scores['risk_group'].values
        
        # 训练决策树
        tree = DecisionTreeClassifier(max_depth=2, random_state=42)
        tree.fit(X, y)
        
        # 获取BMI切割点
        bmi_thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
        bmi_thresholds = sorted(bmi_thresholds)
        
        # 创建BMI分组
        def assign_bmi_group(bmi):
            if bmi <= bmi_thresholds[0]:
                return 0
            elif bmi <= bmi_thresholds[1]:
                return 1
            else:
                return 2
        
        self.risk_scores['bmi_group'] = self.risk_scores['avg_bmi'].apply(assign_bmi_group)
        self.risk_scores['BMI分组ID'] = self.risk_scores['bmi_group']  # 添加BMI分组ID列
        
        print(f"BMI切割点: {bmi_thresholds}")
        print("BMI分组映射完成")
        return bmi_thresholds
    
    def initialize_multivariate_model(self):
        """初始化多维混合效应模型和多目标优化器"""
        print("正在初始化多维混合效应模型...")
        
        # 初始化多维混合效应模型
        self.multivariate_model = MultivariateRiskModel('./data/output.csv', self.save_dir, c_min=0.04)
        
        # 加载和准备数据
        data = self.multivariate_model.load_and_prepare_data()
        
        # 拟合扩展模型
        result, df = self.multivariate_model.fit_multivariate_model()
        
        if result is not None:
            # 初始化多目标优化器
            self.optimizer = MultivariateRiskOptimizer(
                self.multivariate_model,
                strategy='multiobj',  # 使用多目标优化
                mode='chebyshev',     # 使用Chebyshev标量化方法
                alpha=0.5,           # Alpha方法的权重参数
                delta_t_retest=14,   # 重测延迟时间
                enforce_min_gap=True,
                min_gap_days=6.0,    # 最小组间间隔
                t_range=(70, 175),
            )
            print("多维混合效应模型和多目标优化器初始化完成")
            return True
        else:
            print("模型拟合失败！")
            return False
    
    def calculate_optimal_detection_times(self, use_multiobj=True):
        """计算各组的最优检测时点（多目标优化方法）"""
        if use_multiobj:
            print("正在使用多目标优化方法计算最优检测时点...")
            
            if self.optimizer is None:
                if not self.initialize_multivariate_model():
                    return None
            
            # 使用多目标优化器
            results = self.optimizer.optimize_all_groups(self.risk_scores)
            
            # 转换为DataFrame格式
            optimization_results = []
            for group_id, res in results.items():
                group_data = self.risk_scores[self.risk_scores['BMI分组ID'] == group_id]
                
                group_stats = {
                    'bmi_group': group_id,
                    'optimal_detection_time': res['optimal_t'] / 7,  # 转换为孕周
                    'optimal_detection_time_days': res['optimal_t'],  # 保持天数
                    'sample_count': res['group_size'],
                    'mean_bmi': res['avg_bmi'],
                    'bmi_range': f"{group_data['avg_bmi'].min():.1f}-{group_data['avg_bmi'].max():.1f}",
                    'strategy': res['strategy'],
                    'min_risk': res['min_risk'],
                    'p_target': res['p_target'],
                    'coverage_quantile': res['coverage_quantile'],
                    'min_gap_days': res['min_gap_days']
                }
                optimization_results.append(group_stats)
            
            self.optimization_results = pd.DataFrame(optimization_results)
            print("多目标优化检测时点计算完成")
            
        else:
            # 保留原有的分位数方法作为备选
            print("正在使用分位数方法计算最晚安全检测时点...")
            q = 0.75
            results = []
            
            for group_id in sorted(self.risk_scores['BMI分组ID'].unique()):
                group_data = self.risk_scores[self.risk_scores['BMI分组ID'] == group_id]
                
                # 计算该组的q分位数作为最晚安全检测时点
                optimal_time = np.percentile(group_data['achievement_time'], q * 100)
                
                # 计算组内统计信息
                group_stats = {
                    'bmi_group': group_id,
                    'optimal_detection_time': optimal_time,
                    'sample_count': len(group_data),
                    'mean_achievement_time': group_data['achievement_time'].mean(),
                    'std_achievement_time': group_data['achievement_time'].std(),
                    'mean_bmi': group_data['avg_bmi'].mean(),
                    'bmi_range': f"{group_data['avg_bmi'].min():.1f}-{group_data['avg_bmi'].max():.1f}",
                    'mean_success_prob': group_data['current_success_prob'].mean()
                }
                
                results.append(group_stats)
            
            self.optimization_results = pd.DataFrame(results)
            print("分位数方法检测时点计算完成")
        
        return self.optimization_results
    
    def save_results(self):
        """保存所有结果"""
        print("正在保存结果...")
        
        # 保存风险驱动分组结果
        self.risk_scores.to_csv(
            os.path.join(self.save_dir, 'q3_risk_driven_groups.csv'),
            index=False, encoding='utf-8-sig'
        )
        
        # 保存优化结果
        self.optimization_results.to_csv(
            os.path.join(self.save_dir, 'q3_optimization_results.csv'),
            index=False, encoding='utf-8-sig'
        )
        
        print("结果保存完成")
    
    def print_summary(self):
        """打印结果摘要"""
        print("\n=== 多目标优化检测时点结果 ===")
        
        for _, row in self.optimization_results.iterrows():
            print(f"\nBMI组 {row['bmi_group']}:")
            print(f"  样本数: {row['sample_count']}")
            print(f"  BMI范围: {row['bmi_range']}")
            print(f"  平均BMI: {row['mean_bmi']:.2f}")
            
            if 'optimal_detection_time_days' in row:
                print(f"  最优检测时点: {row['optimal_detection_time']:.1f} 孕周 ({row['optimal_detection_time_days']:.1f} 天)")
                print(f"  优化策略: {row['strategy']}")
                if not pd.isna(row['min_risk']):
                    print(f"  最小风险: {row['min_risk']:.4f}")
                if not pd.isna(row['p_target']):
                    print(f"  目标成功概率: {row['p_target']:.2f}")
                if not pd.isna(row['coverage_quantile']):
                    print(f"  覆盖分位数: {row['coverage_quantile']:.2f}")
                if not pd.isna(row['min_gap_days']):
                    print(f"  最小组间间隔: {row['min_gap_days']:.1f} 天")
            else:
                # 兼容原有的分位数方法结果
                print(f"  平均达标时间: {row['mean_achievement_time']:.1f} 孕周")
                print(f"  最晚安全检测时点: {row['optimal_detection_time']:.1f} 孕周")
                print(f"  平均成功概率: {row['mean_success_prob']:.4f}")

def main():
    """主函数"""
    # 初始化优化器
    optimizer = DetectionTimeOptimizer('./3/q3_individual_risk_scores.csv', './3')
    
    # 执行优化流程
    optimizer.load_risk_scores()
    optimizer.perform_risk_clustering(n_clusters=3)
    optimizer.map_risk_to_bmi_groups()
    
    # 使用多目标优化方法（默认）
    optimizer.calculate_optimal_detection_times(use_multiobj=True)
    
    optimizer.save_results()
    optimizer.print_summary()
    
    print("\n多目标优化检测时点计算完成！")

if __name__ == "__main__":
    main()
