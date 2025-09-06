#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：女胎异常判定的多因素加权Z值融合模型（手动特征工程版）
实现您描述的具体建模思路：
1. GC含量与测序质量校正因子消除数据偏差
2. X染色体浓度偏移量作为背景参考
3. 多因素加权Z值融合模型
4. 整合孕妇BMI等临床指标建立综合判别得分
5. 动态阈值机制提高判别稳健性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                           classification_report, confusion_matrix, f1_score,
                           precision_score, recall_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ManualFeatureEngineeringModel:
    """手动特征工程的女胎异常判定模型"""
    
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.lr_model = None
        self.cv_results = {}
        self.feature_engineering_results = {}
        
    def load_and_preprocess_data(self):
        """数据加载与预处理"""
        print("正在加载数据...")
        self.data = pd.read_csv(self.data_path)
        
        # 数据基本信息
        print(f"数据形状: {self.data.shape}")
        
        # 创建标签：染色体的非整倍体 -> 0(正常) 1(异常)
        self.data['label'] = self.data['染色体的非整倍体'].fillna('').astype(str)
        self.data['label'] = (self.data['label'] != '').astype(int)
        
        print(f"标签分布: {self.data['label'].value_counts()}")
        print(f"异常样本比例: {self.data['label'].mean():.3f}")
        
        return self.data
    
    def analyze_data_characteristics(self):
        """分析数据特征，为特征工程提供依据"""
        print("\n分析数据特征...")
        
        # 分析Z值分布
        z_columns = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
        z_stats = {}
        
        for col in z_columns:
            if col in self.data.columns:
                z_stats[col] = {
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'range': self.data[col].max() - self.data[col].min()
                }
        
        # 分析GC含量分布
        gc_columns = ['GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
        gc_stats = {}
        
        for col in gc_columns:
            if col in self.data.columns:
                gc_stats[col] = {
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max()
                }
        
        # 分析测序质量指标
        quality_columns = ['被过滤掉读段数的比例', '重复读段的比例', '在参考基因组上比对的比例']
        quality_stats = {}
        
        for col in quality_columns:
            if col in self.data.columns:
                quality_stats[col] = {
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max()
                }
        
        # 分析X染色体浓度
        if 'X染色体浓度' in self.data.columns:
            x_conc_stats = {
                'mean': self.data['X染色体浓度'].mean(),
                'std': self.data['X染色体浓度'].std(),
                'min': self.data['X染色体浓度'].min(),
                'max': self.data['X染色体浓度'].max()
            }
        
        # 保存分析结果
        analysis_results = {
            'z_values': z_stats,
            'gc_content': gc_stats,
            'quality_metrics': quality_stats,
            'x_chromosome_concentration': x_conc_stats if 'X染色体浓度' in self.data.columns else None
        }
        
        self.feature_engineering_results['data_analysis'] = analysis_results
        
        print("数据特征分析完成")
        print(f"X染色体Z值范围: {z_stats.get('X染色体的Z值', {}).get('min', 'N/A'):.3f} 至 {z_stats.get('X染色体的Z值', {}).get('max', 'N/A'):.3f}")
        print(f"21号染色体Z值范围: {z_stats.get('21号染色体的Z值', {}).get('min', 'N/A'):.3f} 至 {z_stats.get('21号染色体的Z值', {}).get('max', 'N/A'):.3f}")
        print(f"18号染色体Z值范围: {z_stats.get('18号染色体的Z值', {}).get('min', 'N/A'):.3f} 至 {z_stats.get('18号染色体的Z值', {}).get('max', 'N/A'):.3f}")
        print(f"测序数据过滤率范围: {quality_stats.get('被过滤掉读段数的比例', {}).get('min', 'N/A'):.3f} 至 {quality_stats.get('被过滤掉读段数的比例', {}).get('max', 'N/A'):.3f}")
        print(f"GC含量范围: {gc_stats.get('GC含量', {}).get('min', 'N/A'):.3f} 至 {gc_stats.get('GC含量', {}).get('max', 'N/A'):.3f}")
        
        return analysis_results
    
    def create_gc_quality_correction_factors(self):
        """创建GC含量与测序质量校正因子"""
        print("\n创建GC含量与测序质量校正因子...")
        
        # 1. GC含量偏差校正因子
        # 计算各染色体GC含量与整体GC含量的偏差
        if 'GC含量' in self.data.columns:
            gc_mean = self.data['GC含量'].mean()
            
            for chrom in ['13', '18', '21']:
                gc_col = f'{chrom}号染色体的GC含量'
                if gc_col in self.data.columns:
                    # GC含量偏差校正因子
                    self.data[f'{chrom}_gc_correction'] = (self.data[gc_col] - gc_mean) / gc_mean
                    # 标准化GC含量偏差
                    self.data[f'{chrom}_gc_normalized'] = (self.data[gc_col] - self.data[gc_col].mean()) / self.data[gc_col].std()
        
        # 2. 测序质量校正因子
        # 综合测序质量指标
        quality_metrics = []
        if '被过滤掉读段数的比例' in self.data.columns:
            # 过滤率越低，质量越好
            self.data['filter_quality_score'] = 1 - self.data['被过滤掉读段数的比例']
            quality_metrics.append('filter_quality_score')
        
        if '重复读段的比例' in self.data.columns:
            # 重复率越低，质量越好
            self.data['duplicate_quality_score'] = 1 - self.data['重复读段的比例']
            quality_metrics.append('duplicate_quality_score')
        
        if '在参考基因组上比对的比例' in self.data.columns:
            # 比对率越高，质量越好
            self.data['mapping_quality_score'] = self.data['在参考基因组上比对的比例']
            quality_metrics.append('mapping_quality_score')
        
        # 综合测序质量得分
        if quality_metrics:
            self.data['overall_quality_score'] = self.data[quality_metrics].mean(axis=1)
            # 标准化质量得分
            self.data['quality_correction_factor'] = (self.data['overall_quality_score'] - self.data['overall_quality_score'].mean()) / self.data['overall_quality_score'].std()
        
        print("GC含量与测序质量校正因子创建完成")
        
        return self.data
    
    def create_x_chromosome_reference(self):
        """创建X染色体浓度偏移量作为背景参考"""
        print("\n创建X染色体浓度偏移量作为背景参考...")
        
        # 1. X染色体Z值作为背景参考
        if 'X染色体的Z值' in self.data.columns:
            x_z_mean = self.data['X染色体的Z值'].mean()
            x_z_std = self.data['X染色体的Z值'].std()
            
            # X染色体Z值标准化
            self.data['x_z_normalized'] = (self.data['X染色体的Z值'] - x_z_mean) / x_z_std
            
            # X染色体浓度偏移量（如果存在X染色体浓度列）
            if 'X染色体浓度' in self.data.columns:
                x_conc_mean = self.data['X染色体浓度'].mean()
                self.data['x_concentration_offset'] = self.data['X染色体浓度'] - x_conc_mean
                # 标准化X染色体浓度偏移量
                self.data['x_concentration_normalized'] = (self.data['X染色体浓度'] - x_conc_mean) / self.data['X染色体浓度'].std()
        
        # 2. 创建X染色体稳定性指标
        # X染色体Z值应该相对稳定，作为质量控制指标
        if 'X染色体的Z值' in self.data.columns:
            # X染色体Z值的稳定性（与正常范围的偏差）
            x_z_normal_range = [-2, 2]  # 正常Z值范围
            self.data['x_z_stability'] = np.where(
                (self.data['X染色体的Z值'] >= x_z_normal_range[0]) & 
                (self.data['X染色体的Z值'] <= x_z_normal_range[1]), 
                1, 0
            )
        
        print("X染色体浓度偏移量背景参考创建完成")
        
        return self.data
    
    def create_weighted_z_value_fusion(self):
        """创建多因素加权Z值融合模型"""
        print("\n创建多因素加权Z值融合模型...")
        
        # 目标染色体
        target_chromosomes = ['13', '18', '21']
        
        for chrom in target_chromosomes:
            z_col = f'{chrom}号染色体的Z值'
            if z_col in self.data.columns:
                
                # 1. 基础Z值
                self.data[f'{chrom}_z_raw'] = self.data[z_col]
                
                # 2. GC含量校正的Z值
                gc_correction_col = f'{chrom}_gc_correction'
                if gc_correction_col in self.data.columns:
                    # Z值经过GC含量校正
                    self.data[f'{chrom}_z_gc_corrected'] = self.data[z_col] - 0.5 * self.data[gc_correction_col]
                else:
                    self.data[f'{chrom}_z_gc_corrected'] = self.data[z_col]
                
                # 3. 测序质量校正的Z值
                if 'quality_correction_factor' in self.data.columns:
                    # Z值经过测序质量校正
                    self.data[f'{chrom}_z_quality_corrected'] = self.data[z_col] - 0.3 * self.data['quality_correction_factor']
                else:
                    self.data[f'{chrom}_z_quality_corrected'] = self.data[z_col]
                
                # 4. X染色体背景校正的Z值
                if 'x_z_normalized' in self.data.columns:
                    # 使用X染色体Z值作为背景校正
                    self.data[f'{chrom}_z_x_corrected'] = self.data[z_col] - 0.2 * self.data['x_z_normalized']
                else:
                    self.data[f'{chrom}_z_x_corrected'] = self.data[z_col]
                
                # 5. 综合校正的Z值（多因素融合）
                correction_components = []
                if f'{chrom}_z_gc_corrected' in self.data.columns:
                    correction_components.append(self.data[f'{chrom}_z_gc_corrected'])
                if f'{chrom}_z_quality_corrected' in self.data.columns:
                    correction_components.append(self.data[f'{chrom}_z_quality_corrected'])
                if f'{chrom}_z_x_corrected' in self.data.columns:
                    correction_components.append(self.data[f'{chrom}_z_x_corrected'])
                
                if correction_components:
                    # 加权平均融合
                    weights = [0.4, 0.3, 0.3]  # GC校正、质量校正、X染色体校正的权重
                    self.data[f'{chrom}_z_fused'] = np.average(correction_components, axis=0, weights=weights[:len(correction_components)])
                else:
                    self.data[f'{chrom}_z_fused'] = self.data[z_col]
                
                # 6. 创建Z值的风险评分
                # 基于Z值的绝对值创建风险评分
                self.data[f'{chrom}_risk_score'] = np.abs(self.data[f'{chrom}_z_fused'])
                
                # 7. 创建Z值的异常指示器
                # Z值超出正常范围（|Z| > 2）的指示器
                self.data[f'{chrom}_abnormal_indicator'] = (np.abs(self.data[f'{chrom}_z_fused']) > 2).astype(int)
        
        print("多因素加权Z值融合模型创建完成")
        
        return self.data
    
    def integrate_clinical_indicators(self):
        """整合孕妇BMI等临床指标建立综合判别得分"""
        print("\n整合孕妇BMI等临床指标建立综合判别得分...")
        
        # 1. 孕妇生理指标处理
        if '年龄' in self.data.columns:
            # 年龄标准化
            self.data['age_normalized'] = (self.data['年龄'] - self.data['年龄'].mean()) / self.data['年龄'].std()
            # 年龄风险评分（高龄风险）
            self.data['age_risk_score'] = np.where(self.data['年龄'] > 35, 1, 0)
        
        if '孕妇BMI' in self.data.columns:
            # BMI标准化
            self.data['bmi_normalized'] = (self.data['孕妇BMI'] - self.data['孕妇BMI'].mean()) / self.data['孕妇BMI'].std()
            # BMI风险评分（肥胖风险）
            self.data['bmi_risk_score'] = np.where(self.data['孕妇BMI'] > 30, 1, 0)
        
        # 2. 创建综合临床风险评分
        clinical_risk_components = []
        if 'age_risk_score' in self.data.columns:
            clinical_risk_components.append(self.data['age_risk_score'])
        if 'bmi_risk_score' in self.data.columns:
            clinical_risk_components.append(self.data['bmi_risk_score'])
        
        if clinical_risk_components:
            self.data['clinical_risk_score'] = np.mean(clinical_risk_components, axis=0)
        else:
            self.data['clinical_risk_score'] = 0
        
        # 3. 创建综合判别得分
        # 整合染色体风险评分和临床风险评分
        chromosome_risk_components = []
        for chrom in ['13', '18', '21']:
            risk_col = f'{chrom}_risk_score'
            if risk_col in self.data.columns:
                chromosome_risk_components.append(self.data[risk_col])
        
        if chromosome_risk_components:
            # 染色体风险得分的加权平均
            chromosome_weights = [0.3, 0.3, 0.4]  # 13号、18号、21号染色体的权重
            self.data['chromosome_risk_score'] = np.average(chromosome_risk_components, axis=0, weights=chromosome_weights[:len(chromosome_risk_components)])
        else:
            self.data['chromosome_risk_score'] = 0
        
        # 综合判别得分 = 染色体风险得分 + 临床风险得分
        self.data['comprehensive_risk_score'] = 0.8 * self.data['chromosome_risk_score'] + 0.2 * self.data['clinical_risk_score']
        
        # 4. 创建最终的综合判别得分（标准化）
        self.data['final_discriminant_score'] = (self.data['comprehensive_risk_score'] - self.data['comprehensive_risk_score'].mean()) / self.data['comprehensive_risk_score'].std()
        
        print("孕妇BMI等临床指标整合完成")
        
        return self.data
    
    def prepare_final_features(self):
        """准备最终的特征集"""
        print("\n准备最终的特征集...")
        
        # 选择最终特征
        final_features = []
        
        # 1. 融合后的Z值
        for chrom in ['13', '18', '21']:
            fused_z_col = f'{chrom}_z_fused'
            if fused_z_col in self.data.columns:
                final_features.append(fused_z_col)
        
        # 2. 风险评分
        final_features.extend(['chromosome_risk_score', 'clinical_risk_score', 'comprehensive_risk_score', 'final_discriminant_score'])
        
        # 3. 校正因子
        correction_features = ['quality_correction_factor']
        if 'x_z_normalized' in self.data.columns:
            correction_features.append('x_z_normalized')
        if 'x_concentration_normalized' in self.data.columns:
            correction_features.append('x_concentration_normalized')
        
        final_features.extend(correction_features)
        
        # 4. 临床指标
        clinical_features = []
        if 'age_normalized' in self.data.columns:
            clinical_features.append('age_normalized')
        if 'bmi_normalized' in self.data.columns:
            clinical_features.append('bmi_normalized')
        
        final_features.extend(clinical_features)
        
        # 5. 异常指示器
        for chrom in ['13', '18', '21']:
            indicator_col = f'{chrom}_abnormal_indicator'
            if indicator_col in self.data.columns:
                final_features.append(indicator_col)
        
        # 检查特征是否存在
        available_features = [col for col in final_features if col in self.data.columns]
        print(f"最终特征集: {available_features}")
        
        # 准备特征矩阵和标签
        self.X = self.data[available_features].copy()
        self.y = self.data['label'].copy()
        self.feature_names = available_features
        
        # 处理缺失值
        self.X = self.X.fillna(self.X.median())
        
        print(f"特征矩阵形状: {self.X.shape}")
        print(f"标签分布: {self.y.value_counts()}")
        
        return self.X, self.y
    
    def train_logistic_regression_model(self):
        """训练逻辑回归模型"""
        print("\n开始训练逻辑回归模型...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(self.X)
        
        # 逻辑回归模型
        self.lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # 超参数网格
        lr_params = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        # 分层K折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 逻辑回归网格搜索
        print("逻辑回归超参数调优...")
        lr_grid = GridSearchCV(
            self.lr_model, lr_params,
            cv=skf, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        lr_grid.fit(X_scaled, self.y)
        self.lr_model = lr_grid.best_estimator_
        
        print(f"逻辑回归最佳参数: {lr_grid.best_params_}")
        print(f"逻辑回归最佳交叉验证AUC: {lr_grid.best_score_:.4f}")
        
        # 保存最佳参数
        best_params = {
            'lr_best_params': lr_grid.best_params_,
            'lr_best_score': lr_grid.best_score_
        }
        
        pd.DataFrame([best_params]).to_csv(f"{self.output_path}/manual_lr_best_parameters.csv", index=False)
        
        return self.lr_model
    
    def cross_validation_evaluation(self):
        """交叉验证评估"""
        print("\n进行交叉验证评估...")
        
        X_scaled = self.scaler.transform(self.X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        auc_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []
        
        print("评估逻辑回归模型...")
        
        for train_idx, val_idx in skf.split(X_scaled, self.y):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # 训练模型
            self.lr_model.fit(X_train, y_train)
            
            # 预测
            y_pred_proba = self.lr_model.predict_proba(X_val)[:, 1]
            y_pred = self.lr_model.predict(X_val)
            
            # 计算指标
            auc_scores.append(roc_auc_score(y_val, y_pred_proba))
            f1_scores.append(f1_score(y_val, y_pred))
            precision_scores.append(precision_score(y_val, y_pred))
            recall_scores.append(recall_score(y_val, y_pred))
            accuracy_scores.append(accuracy_score(y_val, y_pred))
        
        cv_results = {
            'AUC_mean': np.mean(auc_scores),
            'AUC_std': np.std(auc_scores),
            'F1_mean': np.mean(f1_scores),
            'F1_std': np.std(f1_scores),
            'Precision_mean': np.mean(precision_scores),
            'Precision_std': np.std(precision_scores),
            'Recall_mean': np.mean(recall_scores),
            'Recall_std': np.std(recall_scores),
            'Accuracy_mean': np.mean(accuracy_scores),
            'Accuracy_std': np.std(accuracy_scores)
        }
        
        # 保存交叉验证结果
        cv_df = pd.DataFrame([cv_results], index=['Manual Feature Engineering LR'])
        cv_df.to_csv(f"{self.output_path}/manual_lr_cross_validation_results.csv")
        
        print("交叉验证结果:")
        print(cv_df)
        
        self.cv_results = cv_results
        return cv_results
    
    def implement_dynamic_threshold_mechanism(self, pred_proba):
        """实现动态阈值机制提高判别稳健性"""
        print("\n实现动态阈值机制...")
        
        # 计算不同阈值下的性能
        thresholds = np.arange(0.05, 0.95, 0.01)
        results = []
        
        for threshold in thresholds:
            y_pred = (pred_proba >= threshold).astype(int)
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(self.y, y_pred).ravel()
            
            # 计算指标
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{self.output_path}/manual_lr_threshold_analysis.csv", index=False)
        
        # 动态阈值确定策略
        # 1. 基于临床需求的阈值
        # 低风险阈值：95%灵敏度
        high_sensitivity_idx = results_df[results_df['sensitivity'] >= 0.95].index
        if len(high_sensitivity_idx) > 0:
            dl = results_df.loc[high_sensitivity_idx[0], 'threshold']
        else:
            dl = 0.1
        
        # 高风险阈值：95%精确率
        high_precision_idx = results_df[results_df['precision'] >= 0.95].index
        if len(high_precision_idx) > 0:
            dh = results_df.loc[high_precision_idx[-1], 'threshold']
        else:
            dh = 0.8
        
        # 2. 基于F1分数的最优阈值
        best_f1_idx = results_df['f1_score'].idxmax()
        optimal_f1_threshold = results_df.loc[best_f1_idx, 'threshold']
        
        # 3. 基于Youden指数的阈值
        youden_scores = results_df['sensitivity'] + results_df['specificity'] - 1
        best_youden_idx = youden_scores.idxmax()
        youden_threshold = results_df.loc[best_youden_idx, 'threshold']
        
        print(f"动态阈值确定结果:")
        print(f"低风险阈值 (DL): {dl:.3f}")
        print(f"高风险阈值 (DH): {dh:.3f}")
        print(f"最优F1阈值: {optimal_f1_threshold:.3f}")
        print(f"Youden指数最优阈值: {youden_threshold:.3f}")
        print(f"不确定区间: {dh-dl:.3f} ({(dh-dl)*100:.1f}%)")
        
        # 保存动态阈值决策规则
        dynamic_threshold_rules = {
            'low_risk_threshold': dl,
            'high_risk_threshold': dh,
            'optimal_f1_threshold': optimal_f1_threshold,
            'youden_threshold': youden_threshold,
            'uncertain_interval': dh - dl,
            'decision_rules': {
                'high_risk': f'D >= {dh:.3f} -> 高风险异常',
                'low_risk': f'D < {dl:.3f} -> 低风险正常',
                'uncertain': f'{dl:.3f} <= D < {dh:.3f} -> 结果不确定，建议复检'
            }
        }
        
        pd.DataFrame([dynamic_threshold_rules]).to_csv(f"{self.output_path}/manual_lr_dynamic_threshold_rules.csv", index=False)
        
        return dl, dh, optimal_f1_threshold, youden_threshold
    
    def final_model_evaluation(self):
        """最终模型评估"""
        print("\n进行最终模型评估...")
        
        # 在全部数据上重新训练最终模型
        X_scaled = self.scaler.fit_transform(self.X)
        
        self.lr_model.fit(X_scaled, self.y)
        
        # 预测
        lr_pred_proba = self.lr_model.predict_proba(X_scaled)[:, 1]
        lr_pred = (lr_pred_proba >= 0.5).astype(int)
        
        # 计算评估指标
        auc = roc_auc_score(self.y, lr_pred_proba)
        precision = precision_score(self.y, lr_pred)
        recall = recall_score(self.y, lr_pred)
        f1 = f1_score(self.y, lr_pred)
        accuracy = accuracy_score(self.y, lr_pred)
        
        print("=" * 60)
        print("手动特征工程逻辑回归最终模型评估结果 (阈值=0.5)")
        print("=" * 60)
        print(f"AUC: {auc:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"准确率: {accuracy:.4f}")
        print("=" * 60)
        
        # 保存最终评估结果
        final_results = {
            'Model': 'Manual Feature Engineering LR',
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Accuracy': accuracy
        }
        
        pd.DataFrame([final_results]).to_csv(f"{self.output_path}/manual_lr_final_evaluation.csv", index=False)
        
        # 实现动态阈值机制
        dl, dh, optimal_f1_threshold, youden_threshold = self.implement_dynamic_threshold_mechanism(lr_pred_proba)
        
        return lr_pred_proba, dl, dh, optimal_f1_threshold, youden_threshold
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n分析特征重要性...")
        
        # 获取逻辑回归系数
        coefficients = self.lr_model.coef_[0]
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # 保存特征重要性
        feature_importance.to_csv(f"{self.output_path}/manual_lr_feature_importance.csv", index=False)
        
        print("特征重要性分析完成")
        print("前10个重要特征:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        return feature_importance
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n生成综合报告...")
        
        report = f"""
# 第四问：女胎异常判定模型报告（手动特征工程版）

## 模型概述
- 模型类型: 多因素加权Z值融合模型 + 逻辑回归
- 数据样本数: {len(self.y)}
- 异常样本比例: {self.y.mean():.3f}
- 特征数量: {len(self.feature_names)}

## 手动特征工程策略
1. **GC含量与测序质量校正因子**: 消除数据偏差
2. **X染色体浓度偏移量**: 作为背景参考
3. **多因素加权Z值融合**: 整合多种校正因子
4. **临床指标整合**: 孕妇BMI等生理指标
5. **动态阈值机制**: 提高判别稳健性

## 交叉验证结果
- AUC: {self.cv_results['AUC_mean']:.4f} ± {self.cv_results['AUC_std']:.4f}
- F1分数: {self.cv_results['F1_mean']:.4f} ± {self.cv_results['F1_std']:.4f}
- 精确率: {self.cv_results['Precision_mean']:.4f} ± {self.cv_results['Precision_std']:.4f}
- 召回率: {self.cv_results['Recall_mean']:.4f} ± {self.cv_results['Recall_std']:.4f}
- 准确率: {self.cv_results['Accuracy_mean']:.4f} ± {self.cv_results['Accuracy_std']:.4f}

## 特征重要性分析
"""
        
        # 读取特征重要性
        try:
            feature_importance = pd.read_csv(f"{self.output_path}/manual_lr_feature_importance.csv")
            report += "\n".join([f"- {row['feature']}: {row['coefficient']:.4f}" 
                               for _, row in feature_importance.head(10).iterrows()])
        except:
            report += "特征重要性分析未完成"
        
        # 保存报告
        with open(f"{self.output_path}/manual_lr_model_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("综合报告已生成")
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("开始第四问手动特征工程分析...")
        
        # 1. 数据加载与预处理
        self.load_and_preprocess_data()
        
        # 2. 分析数据特征
        self.analyze_data_characteristics()
        
        # 3. 创建GC含量与测序质量校正因子
        self.create_gc_quality_correction_factors()
        
        # 4. 创建X染色体浓度偏移量作为背景参考
        self.create_x_chromosome_reference()
        
        # 5. 创建多因素加权Z值融合模型
        self.create_weighted_z_value_fusion()
        
        # 6. 整合孕妇BMI等临床指标建立综合判别得分
        self.integrate_clinical_indicators()
        
        # 7. 准备最终特征集
        self.prepare_final_features()
        
        # 8. 训练逻辑回归模型
        self.train_logistic_regression_model()
        
        # 9. 交叉验证评估
        self.cross_validation_evaluation()
        
        # 10. 最终模型评估
        pred_proba, dl, dh, optimal_f1_threshold, youden_threshold = self.final_model_evaluation()
        
        # 11. 分析特征重要性
        self.analyze_feature_importance()
        
        # 12. 生成综合报告
        self.generate_comprehensive_report()
        
        print("\n第四问手动特征工程分析完成！")
        print(f"所有结果已保存到: {self.output_path}")


def main():
    """主函数"""
    # 设置路径
    data_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/girl_output.csv"
    output_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/4"
    
    # 创建手动特征工程模型实例
    model = ManualFeatureEngineeringModel(data_path, output_path)
    
    # 运行完整分析
    model.run_complete_analysis()


if __name__ == "__main__":
    main()
