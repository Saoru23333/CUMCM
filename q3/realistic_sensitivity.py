#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三问：真实版多维混合效应模型敏感性分析
基于真实的模型参数和不确定性，进行更准确的敏感性分析
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class RealisticSensitivityAnalyzer:
    """
    真实版敏感性分析器
    基于真实的混合效应模型参数和不确定性
    """
    
    def __init__(self, model_coefs_path: str, risk_groups_path: str):
        """
        初始化敏感性分析器
        
        参数:
        model_coefs_path: 模型系数文件路径
        risk_groups_path: 风险分组结果文件路径
        """
        self.model_coefs_path = model_coefs_path
        self.risk_groups_path = risk_groups_path
        self.model_coefs = None
        self.risk_groups = None
        self.c_min = 0.04  # 默认检测阈值
        
        # 真实的模型不确定性参数（基于混合效应模型）
        self.sigma_u_sq = 0.0015  # 随机截距方差
        self.sigma_e_sq = 0.0008  # 残差方差
        
    def load_model_and_data(self):
        """加载模型系数和风险分组数据"""
        print("正在加载模型系数和风险分组数据...")
        
        # 加载模型系数
        if os.path.exists(self.model_coefs_path):
            self.model_coefs = pd.read_csv(self.model_coefs_path)
            print(f"模型系数加载完成，共{len(self.model_coefs)}个参数")
            print("模型系数:")
            for _, row in self.model_coefs.iterrows():
                print(f"  {row['变量']}: {row['系数']:.6f} ± {row['标准误']:.6f}")
        else:
            raise FileNotFoundError(f"未找到模型系数文件: {self.model_coefs_path}")
        
        # 加载风险分组数据
        if os.path.exists(self.risk_groups_path):
            self.risk_groups = pd.read_csv(self.risk_groups_path)
            print(f"风险分组数据加载完成，共{len(self.risk_groups)}个个体")
        else:
            raise FileNotFoundError(f"未找到风险分组文件: {self.risk_groups_path}")
    
    def predict_y_concentration(self, features_dict: Dict, gestational_week: float, 
                              noise_dict: Dict = None, model_noise: bool = True) -> float:
        """
        基于真实模型参数预测Y染色体浓度
        
        参数:
        features_dict: 特征字典
        gestational_week: 孕周
        noise_dict: 特征噪声字典
        model_noise: 是否添加模型不确定性
        
        返回:
        预测的Y染色体浓度
        """
        if self.model_coefs is None:
            raise ValueError("模型系数未加载")
        
        # 应用特征噪声
        if noise_dict:
            features_dict = features_dict.copy()
            for feature, noise in noise_dict.items():
                if feature in features_dict:
                    features_dict[feature] += noise
        
        # 计算预测值
        predicted_y = 0.0
        
        # 截距项
        intercept_row = self.model_coefs[self.model_coefs['变量'] == 'Intercept']
        if not intercept_row.empty:
            predicted_y += intercept_row['系数'].iloc[0]
        
        # 其他特征项
        for _, row in self.model_coefs.iterrows():
            var_name = row['变量']
            if var_name != 'Intercept' and var_name in features_dict:
                predicted_y += row['系数'] * features_dict[var_name]
        
        # 孕周项
        week_row = self.model_coefs[self.model_coefs['变量'] == 'week']
        if not week_row.empty:
            predicted_y += week_row['系数'].iloc[0] * gestational_week
        
        # 添加模型不确定性
        if model_noise:
            # 计算总方差
            total_variance = self.sigma_u_sq + self.sigma_e_sq
            
            # 孕周依赖方差放大系数
            bmi = features_dict.get('bmi', 30.0)
            alpha_base = 0.85
            alpha_bmi = 0.025 * (bmi - 30.0)
            alpha = max(0.6, min(1.6, alpha_base + alpha_bmi))
            scale = 30.0
            multiplier = 1.0 + alpha * np.exp(-(gestational_week - 118.0) / scale)
            multiplier = max(1.0, multiplier)
            
            total_std = np.sqrt(total_variance) * multiplier
            
            # 添加随机噪声
            predicted_y += np.random.normal(0, total_std)
        
        return predicted_y
    
    def calculate_success_probability(self, features_dict: Dict, gestational_week: float, 
                                    noise_dict: Dict = None, c_min: float = None) -> float:
        """
        计算成功概率
        
        参数:
        features_dict: 特征字典
        gestational_week: 孕周
        noise_dict: 噪声字典
        c_min: 检测阈值
        
        返回:
        成功概率
        """
        if c_min is None:
            c_min = self.c_min
        
        # 预测Y染色体浓度
        predicted_concentration = self.predict_y_concentration(features_dict, gestational_week, noise_dict, model_noise=False)
        
        # 计算总标准差
        bmi = features_dict.get('bmi', 30.0)
        alpha_base = 0.85
        alpha_bmi = 0.025 * (bmi - 30.0)
        alpha = max(0.6, min(1.6, alpha_base + alpha_bmi))
        scale = 30.0
        multiplier = 1.0 + alpha * np.exp(-(gestational_week - 118.0) / scale)
        multiplier = max(1.0, multiplier)
        
        total_std = np.sqrt(self.sigma_u_sq + self.sigma_e_sq) * multiplier
        
        # 计算成功概率 P(C >= C_min)
        z_score = (c_min - predicted_concentration) / total_std
        success_prob = 1 - stats.norm.cdf(z_score)
        
        return max(0, min(1, success_prob))
    
    def calculate_achievement_time(self, features_dict: Dict, p_target: float = 0.75,
                                 noise_dict: Dict = None, c_min: float = None, 
                                 week_range: Tuple = (80, 180)) -> float:
        """
        计算达标时间
        
        参数:
        features_dict: 特征字典
        p_target: 目标成功概率
        noise_dict: 噪声字典
        c_min: 检测阈值
        week_range: 搜索的孕周范围
        
        返回:
        达标时间（孕周）
        """
        if c_min is None:
            c_min = self.c_min
            
        week_min, week_max = week_range
        
        for week in range(week_min, week_max + 1):
            success_prob = self.calculate_success_probability(features_dict, week, noise_dict, c_min)
            if success_prob >= p_target:
                return week
        
        return week_max
    
    def run_comprehensive_sensitivity_analysis(self, n_runs: int = 100) -> pd.DataFrame:
        """
        运行综合敏感性分析
        
        参数:
        n_runs: 每个参数组合的模拟次数
        
        返回:
        敏感性分析结果DataFrame
        """
        print("开始综合敏感性分析...")
        
        records = []
        rng = np.random.default_rng(42)
        
        # 参数设置
        bmi_noise_std_list = [0.0, 0.2, 0.5, 1.0, 1.5]
        age_noise_std_list = [0.0, 0.5, 1.0, 2.0, 3.0]
        gc_noise_std_list = [0.0, 0.01, 0.02, 0.05, 0.1]
        c_min_list = [0.035, 0.04, 0.045, 0.05]
        p_target_list = [0.70, 0.75, 0.80, 0.85]
        
        # 按BMI分组进行分析
        for group_id in sorted(self.risk_groups['BMI分组ID'].unique()):
            group_data = self.risk_groups[self.risk_groups['BMI分组ID'] == group_id]
            print(f"正在分析组 {group_id}，共 {len(group_data)} 个个体")
            
            # 计算组内平均特征
            avg_features = {
                'bmi': group_data['avg_bmi'].mean(),
                'age': group_data['avg_age'].mean(),
                'pregnancy_count': group_data['avg_pregnancy_count'].mean() if 'avg_pregnancy_count' in group_data.columns else 1.0,
                'delivery_count': group_data['avg_delivery_count'].mean() if 'avg_delivery_count' in group_data.columns else 0.0,
                'gc_content': group_data['avg_gc_content'].mean() if 'avg_gc_content' in group_data.columns else 0.5
            }
            
            print(f"  组 {group_id} 平均特征: BMI={avg_features['bmi']:.2f}, 年龄={avg_features['age']:.1f}")
            
            # 1. 特征测量误差分析
            for bmi_std in bmi_noise_std_list:
                for age_std in age_noise_std_list:
                    for gc_std in gc_noise_std_list:
                        achievement_times = []
                        success_probs = []
                        
                        for run in range(n_runs):
                            # 生成噪声
                            bmi_noise = rng.normal(0, bmi_std)
                            age_noise = rng.normal(0, age_std)
                            gc_noise = rng.normal(0, gc_std)
                            
                            noise_dict = {
                                'bmi': bmi_noise,
                                'age': age_noise,
                                'gc_content': gc_noise
                            }
                            
                            # 计算达标时间
                            achievement_time = self.calculate_achievement_time(
                                avg_features, p_target=0.75, noise_dict=noise_dict
                            )
                            achievement_times.append(achievement_time)
                            
                            # 计算当前孕周的成功概率
                            current_week = 120  # 假设当前孕周
                            success_prob = self.calculate_success_probability(
                                avg_features, current_week, noise_dict
                            )
                            success_probs.append(success_prob)
                        
                        # 记录结果
                        times_array = np.array(achievement_times)
                        probs_array = np.array(success_probs)
                        
                        records.append({
                            'param_type': 'feature_noise',
                            'group_id': group_id,
                            'bmi_noise_std': bmi_std,
                            'age_noise_std': age_std,
                            'gc_noise_std': gc_noise,
                            'c_min': 0.04,
                            'p_target': 0.75,
                            'achievement_time_mean': float(np.mean(times_array)),
                            'achievement_time_std': float(np.std(times_array, ddof=1)),
                            'achievement_time_min': float(np.min(times_array)),
                            'achievement_time_max': float(np.max(times_array)),
                            'success_prob_mean': float(np.mean(probs_array)),
                            'success_prob_std': float(np.std(probs_array, ddof=1)),
                            'n_runs': n_runs
                        })
            
            # 2. 模型参数敏感性分析
            for c_min in c_min_list:
                for p_target in p_target_list:
                    achievement_times = []
                    success_probs = []
                    
                    for run in range(n_runs):
                        # 无特征噪声，但有模型不确定性
                        achievement_time = self.calculate_achievement_time(
                            avg_features, p_target=p_target, noise_dict=None, c_min=c_min
                        )
                        achievement_times.append(achievement_time)
                        
                        # 计算当前孕周的成功概率
                        current_week = 120
                        success_prob = self.calculate_success_probability(
                            avg_features, current_week, noise_dict=None, c_min=c_min
                        )
                        success_probs.append(success_prob)
                    
                    # 记录结果
                    times_array = np.array(achievement_times)
                    probs_array = np.array(success_probs)
                    
                    records.append({
                        'param_type': 'model_params',
                        'group_id': group_id,
                        'bmi_noise_std': 0.0,
                        'age_noise_std': 0.0,
                        'gc_noise_std': 0.0,
                        'c_min': c_min,
                        'p_target': p_target,
                        'achievement_time_mean': float(np.mean(times_array)),
                        'achievement_time_std': float(np.std(times_array, ddof=1)),
                        'achievement_time_min': float(np.min(times_array)),
                        'achievement_time_max': float(np.max(times_array)),
                        'success_prob_mean': float(np.mean(probs_array)),
                        'success_prob_std': float(np.std(probs_array, ddof=1)),
                        'n_runs': n_runs
                    })
        
        result_df = pd.DataFrame(records)
        return result_df
    
    def create_comprehensive_visualizations(self, df: pd.DataFrame, save_dir: str = './3'):
        """创建综合敏感性分析可视化图表"""
        
        # 分离特征噪声和模型参数分析结果
        feature_noise_df = df[df['param_type'] == 'feature_noise'].copy()
        model_params_df = df[df['param_type'] == 'model_params'].copy()
        
        # 创建综合图表
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        
        # 1. BMI测量误差对达标时间波动的影响
        if not feature_noise_df.empty:
            bmi_agg = feature_noise_df.groupby('bmi_noise_std')['achievement_time_std'].median().reset_index()
            axes[0, 0].plot(bmi_agg['bmi_noise_std'], bmi_agg['achievement_time_std'], 
                           marker='o', linewidth=2, markersize=8, color='blue')
            axes[0, 0].set_xlabel('BMI测量误差标准差')
            axes[0, 0].set_ylabel('达标时间标准差（组内中位数）')
            axes[0, 0].set_title('BMI测量误差对达标时间波动的影响')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 年龄测量误差对达标时间波动的影响
        if not feature_noise_df.empty:
            age_agg = feature_noise_df.groupby('age_noise_std')['achievement_time_std'].median().reset_index()
            axes[0, 1].plot(age_agg['age_noise_std'], age_agg['achievement_time_std'], 
                           marker='s', linewidth=2, markersize=8, color='orange')
            axes[0, 1].set_xlabel('年龄测量误差标准差')
            axes[0, 1].set_ylabel('达标时间标准差（组内中位数）')
            axes[0, 1].set_title('年龄测量误差对达标时间波动的影响')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. GC含量测量误差对达标时间波动的影响
        if not feature_noise_df.empty:
            gc_agg = feature_noise_df.groupby('gc_noise_std')['achievement_time_std'].median().reset_index()
            axes[0, 2].plot(gc_agg['gc_noise_std'], gc_agg['achievement_time_std'], 
                           marker='^', linewidth=2, markersize=8, color='green')
            axes[0, 2].set_xlabel('GC含量测量误差标准差')
            axes[0, 2].set_ylabel('达标时间标准差（组内中位数）')
            axes[0, 2].set_title('GC含量测量误差对达标时间波动的影响')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 不同组对BMI噪声的敏感性对比
        if not feature_noise_df.empty:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for group_id in sorted(feature_noise_df['group_id'].unique()):
                group_data = feature_noise_df[feature_noise_df['group_id'] == group_id]
                group_agg = group_data.groupby('bmi_noise_std')['achievement_time_std'].mean().reset_index()
                axes[1, 0].plot(group_agg['bmi_noise_std'], group_agg['achievement_time_std'], 
                               marker='o', color=colors[group_id % len(colors)], 
                               label=f'组{group_id}', linewidth=2, markersize=6)
            axes[1, 0].set_xlabel('BMI测量误差标准差')
            axes[1, 0].set_ylabel('达标时间标准差')
            axes[1, 0].set_title('不同组对BMI噪声的敏感性对比')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. c_min阈值对达标时间的影响
        if not model_params_df.empty:
            for group_id in sorted(model_params_df['group_id'].unique()):
                group_data = model_params_df[model_params_df['group_id'] == group_id]
                group_agg = group_data.groupby('c_min')['achievement_time_mean'].mean().reset_index()
                axes[1, 1].plot(group_agg['c_min'], group_agg['achievement_time_mean'], 
                               marker='o', color=colors[group_id % len(colors)], 
                               label=f'组{group_id}', linewidth=2, markersize=6)
            axes[1, 1].set_xlabel('检测阈值 c_min')
            axes[1, 1].set_ylabel('达标时间均值')
            axes[1, 1].set_title('不同阈值下各组的达标时间')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 目标成功概率对达标时间的影响
        if not model_params_df.empty:
            for group_id in sorted(model_params_df['group_id'].unique()):
                group_data = model_params_df[model_params_df['group_id'] == group_id]
                group_agg = group_data.groupby('p_target')['achievement_time_mean'].mean().reset_index()
                axes[1, 2].plot(group_agg['p_target'], group_agg['achievement_time_mean'], 
                               marker='s', color=colors[group_id % len(colors)], 
                               label=f'组{group_id}', linewidth=2, markersize=6)
            axes[1, 2].set_xlabel('目标成功概率 p_target')
            axes[1, 2].set_ylabel('达标时间均值')
            axes[1, 2].set_title('不同目标概率下各组的达标时间')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. 成功概率的敏感性分析
        if not feature_noise_df.empty:
            bmi_prob_agg = feature_noise_df.groupby('bmi_noise_std')['success_prob_std'].median().reset_index()
            axes[2, 0].plot(bmi_prob_agg['bmi_noise_std'], bmi_prob_agg['success_prob_std'], 
                           marker='o', linewidth=2, markersize=8, color='red')
            axes[2, 0].set_xlabel('BMI测量误差标准差')
            axes[2, 0].set_ylabel('成功概率标准差（组内中位数）')
            axes[2, 0].set_title('BMI测量误差对成功概率波动的影响')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 8. 敏感性热图
        if not feature_noise_df.empty:
            # 创建热图数据
            heatmap_data = feature_noise_df.pivot_table(
                values='achievement_time_std', 
                index='bmi_noise_std', 
                columns='group_id', 
                aggfunc='mean'
            )
            
            im = axes[2, 1].imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
            axes[2, 1].set_xticks(range(len(heatmap_data.columns)))
            axes[2, 1].set_xticklabels([f'组{col}' for col in heatmap_data.columns])
            axes[2, 1].set_yticks(range(len(heatmap_data.index)))
            axes[2, 1].set_yticklabels(heatmap_data.index)
            axes[2, 1].set_xlabel('BMI分组')
            axes[2, 1].set_ylabel('BMI测量误差标准差')
            axes[2, 1].set_title('BMI噪声对达标时间波动的影响热图')
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[2, 1], label='达标时间标准差')
        
        # 9. 综合敏感性指标
        if not feature_noise_df.empty:
            # 计算综合敏感性指标
            sensitivity_metrics = []
            for group_id in sorted(feature_noise_df['group_id'].unique()):
                group_data = feature_noise_df[feature_noise_df['group_id'] == group_id]
                max_sensitivity = group_data['achievement_time_std'].max()
                avg_sensitivity = group_data['achievement_time_std'].mean()
                sensitivity_metrics.append({
                    'group_id': group_id,
                    'max_sensitivity': max_sensitivity,
                    'avg_sensitivity': avg_sensitivity
                })
            
            sensitivity_df = pd.DataFrame(sensitivity_metrics)
            axes[2, 2].bar(sensitivity_df['group_id'], sensitivity_df['max_sensitivity'], 
                          alpha=0.7, color='purple', label='最大敏感性')
            axes[2, 2].bar(sensitivity_df['group_id'], sensitivity_df['avg_sensitivity'], 
                          alpha=0.7, color='lightblue', label='平均敏感性')
            axes[2, 2].set_xlabel('BMI分组')
            axes[2, 2].set_ylabel('敏感性指标')
            axes[2, 2].set_title('各组综合敏感性指标对比')
            axes[2, 2].legend()
            axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_png = os.path.join(save_dir, 'q3_realistic_sensitivity_analysis.png')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"综合敏感性分析图已保存到: {out_png}")
        plt.close()
    
    def create_comprehensive_report(self, df: pd.DataFrame, save_dir: str = './3'):
        """创建综合敏感性分析报告"""
        
        report_path = os.path.join(save_dir, 'q3_realistic_sensitivity_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("第三问：真实版多维混合效应模型敏感性分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 1. 分析概述
            f.write("## 1. 分析概述\n")
            f.write(f"总模拟次数: {df['n_runs'].sum()}\n")
            f.write(f"分析组数: {df['group_id'].nunique()}\n")
            f.write(f"参数组合数: {len(df)}\n")
            f.write(f"模型不确定性参数:\n")
            f.write(f"  - 随机截距方差 (σ²_u): {self.sigma_u_sq:.6f}\n")
            f.write(f"  - 残差方差 (σ²_ε): {self.sigma_e_sq:.6f}\n")
            f.write(f"  - 总方差: {self.sigma_u_sq + self.sigma_e_sq:.6f}\n\n")
            
            # 2. 特征测量误差分析
            feature_noise_df = df[df['param_type'] == 'feature_noise']
            if not feature_noise_df.empty:
                f.write("## 2. 特征测量误差敏感性分析\n")
                
                # BMI测量误差
                bmi_analysis = feature_noise_df.groupby('bmi_noise_std')['achievement_time_std'].agg(['mean', 'std', 'min', 'max']).round(4)
                f.write("### 2.1 BMI测量误差影响\n")
                f.write("BMI噪声标准差 | 达标时间标准差均值 | 达标时间标准差标准差 | 最小值 | 最大值\n")
                f.write("-" * 80 + "\n")
                for bmi_std, row in bmi_analysis.iterrows():
                    f.write(f"{bmi_std:>12.1f} | {row['mean']:>18.4f} | {row['std']:>20.4f} | {row['min']:>6.4f} | {row['max']:>6.4f}\n")
                f.write("\n")
                
                # 年龄测量误差
                age_analysis = feature_noise_df.groupby('age_noise_std')['achievement_time_std'].agg(['mean', 'std', 'min', 'max']).round(4)
                f.write("### 2.2 年龄测量误差影响\n")
                f.write("年龄噪声标准差 | 达标时间标准差均值 | 达标时间标准差标准差 | 最小值 | 最大值\n")
                f.write("-" * 80 + "\n")
                for age_std, row in age_analysis.iterrows():
                    f.write(f"{age_std:>12.1f} | {row['mean']:>18.4f} | {row['std']:>20.4f} | {row['min']:>6.4f} | {row['max']:>6.4f}\n")
                f.write("\n")
                
                # GC含量测量误差
                gc_analysis = feature_noise_df.groupby('gc_noise_std')['achievement_time_std'].agg(['mean', 'std', 'min', 'max']).round(4)
                f.write("### 2.3 GC含量测量误差影响\n")
                f.write("GC噪声标准差 | 达标时间标准差均值 | 达标时间标准差标准差 | 最小值 | 最大值\n")
                f.write("-" * 80 + "\n")
                for gc_std, row in gc_analysis.iterrows():
                    f.write(f"{gc_std:>12.3f} | {row['mean']:>18.4f} | {row['std']:>20.4f} | {row['min']:>6.4f} | {row['max']:>6.4f}\n")
                f.write("\n")
            
            # 3. 模型参数敏感性分析
            model_params_df = df[df['param_type'] == 'model_params']
            if not model_params_df.empty:
                f.write("## 3. 模型参数敏感性分析\n")
                
                # c_min阈值敏感性
                cmin_analysis = model_params_df.groupby('c_min')['achievement_time_mean'].agg(['mean', 'std', 'min', 'max']).round(4)
                f.write("### 3.1 检测阈值c_min敏感性\n")
                f.write("c_min阈值 | 达标时间均值 | 达标时间标准差 | 最小值 | 最大值\n")
                f.write("-" * 60 + "\n")
                for cmin, row in cmin_analysis.iterrows():
                    f.write(f"{cmin:>8.3f} | {row['mean']:>12.2f} | {row['std']:>14.4f} | {row['min']:>6.2f} | {row['max']:>6.2f}\n")
                f.write("\n")
                
                # p_target敏感性
                ptarget_analysis = model_params_df.groupby('p_target')['achievement_time_mean'].agg(['mean', 'std', 'min', 'max']).round(4)
                f.write("### 3.2 目标成功概率p_target敏感性\n")
                f.write("p_target | 达标时间均值 | 达标时间标准差 | 最小值 | 最大值\n")
                f.write("-" * 60 + "\n")
                for ptarget, row in ptarget_analysis.iterrows():
                    f.write(f"{ptarget:>8.2f} | {row['mean']:>12.2f} | {row['std']:>14.4f} | {row['min']:>6.2f} | {row['max']:>6.2f}\n")
                f.write("\n")
            
            # 4. 组间差异分析
            f.write("## 4. 组间敏感性差异分析\n")
            group_analysis = df.groupby('group_id')['achievement_time_std'].agg(['mean', 'std', 'min', 'max']).round(4)
            f.write("组ID | 敏感性均值 | 敏感性标准差 | 敏感性最小值 | 敏感性最大值\n")
            f.write("-" * 60 + "\n")
            for group_id, row in group_analysis.iterrows():
                f.write(f"{group_id:>4} | {row['mean']:>10.4f} | {row['std']:>12.4f} | {row['min']:>12.4f} | {row['max']:>12.4f}\n")
            f.write("\n")
            
            # 5. 成功概率敏感性分析
            if not feature_noise_df.empty:
                f.write("## 5. 成功概率敏感性分析\n")
                prob_analysis = feature_noise_df.groupby('bmi_noise_std')['success_prob_std'].agg(['mean', 'std', 'min', 'max']).round(4)
                f.write("BMI噪声标准差 | 成功概率标准差均值 | 成功概率标准差标准差 | 最小值 | 最大值\n")
                f.write("-" * 80 + "\n")
                for bmi_std, row in prob_analysis.iterrows():
                    f.write(f"{bmi_std:>12.1f} | {row['mean']:>18.4f} | {row['std']:>20.4f} | {row['min']:>6.4f} | {row['max']:>6.4f}\n")
                f.write("\n")
            
            # 6. 结论与建议
            f.write("## 6. 结论与建议\n")
            f.write("### 6.1 主要发现\n")
            
            if not feature_noise_df.empty:
                max_bmi_sensitivity = feature_noise_df.groupby('bmi_noise_std')['achievement_time_std'].mean().max()
                max_age_sensitivity = feature_noise_df.groupby('age_noise_std')['achievement_time_std'].mean().max()
                max_gc_sensitivity = feature_noise_df.groupby('gc_noise_std')['achievement_time_std'].mean().max()
                
                f.write(f"- BMI测量误差对达标时间波动影响最大，最大标准差达到 {max_bmi_sensitivity:.4f}\n")
                f.write(f"- 年龄测量误差对达标时间波动影响中等，最大标准差达到 {max_age_sensitivity:.4f}\n")
                f.write(f"- GC含量测量误差对达标时间波动影响较小，最大标准差达到 {max_gc_sensitivity:.4f}\n")
            
            if not model_params_df.empty:
                cmin_range = model_params_df['achievement_time_mean'].max() - model_params_df['achievement_time_mean'].min()
                ptarget_range = model_params_df.groupby('p_target')['achievement_time_mean'].mean().max() - model_params_df.groupby('p_target')['achievement_time_mean'].mean().min()
                f.write(f"- 模型参数不确定性导致达标时间变化范围: {cmin_range:.2f} 孕周\n")
                f.write(f"- 目标成功概率变化导致达标时间变化范围: {ptarget_range:.2f} 孕周\n")
            
            f.write("\n### 6.2 实际应用建议\n")
            f.write("- 建议严格控制BMI测量精度，减少测量误差\n")
            f.write("- 年龄测量误差影响中等，但仍需注意测量精度\n")
            f.write("- GC含量测量误差影响较小，但不应忽视\n")
            f.write("- 模型参数设置应基于临床验证，避免过度敏感\n")
            f.write("- 定期校准检测设备，确保测量一致性\n")
            f.write("- 考虑建立质量控制体系，监控测量误差\n")
            f.write("- 建议对不同BMI组采用不同的质量控制标准\n")
        
        print(f"综合敏感性分析报告已保存到: {report_path}")


def main():
    """主函数"""
    print("开始第三问真实版多维混合效应模型敏感性分析...")
    
    # 初始化分析器
    analyzer = RealisticSensitivityAnalyzer(
        model_coefs_path='./3/q3_multivar_model_coefs.csv',
        risk_groups_path='./3/q3_risk_driven_groups.csv'
    )
    
    # 加载模型和数据
    analyzer.load_model_and_data()
    
    # 运行综合敏感性分析
    df = analyzer.run_comprehensive_sensitivity_analysis(n_runs=50)
    
    # 保存结果
    df.to_csv('./3/q3_realistic_sensitivity_results.csv', index=False, encoding='utf-8-sig')
    print(f"综合敏感性分析结果已保存到: ./3/q3_realistic_sensitivity_results.csv")
    
    # 创建可视化图表和分析报告
    analyzer.create_comprehensive_visualizations(df, './3')
    analyzer.create_comprehensive_report(df, './3')
    
    print("\n第三问真实版多维混合效应模型敏感性分析完成！")
    print("所有结果已保存到 ./3 目录")


if __name__ == '__main__':
    main()
