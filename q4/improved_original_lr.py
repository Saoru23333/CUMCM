#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：改进版原始逻辑回归模型
在保持良好泛化能力的前提下，优化原始逻辑回归模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                           classification_report, confusion_matrix, f1_score,
                           precision_score, recall_score, accuracy_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedOriginalLR:
    """改进版原始逻辑回归模型"""
    
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = RobustScaler()  # 使用RobustScaler，对异常值更鲁棒
        self.lr_model = None
        self.cv_results = {}
        
    def load_and_preprocess_data(self):
        """数据加载与预处理"""
        print("正在加载数据...")
        self.data = pd.read_csv(self.data_path)
        
        # 数据基本信息
        print(f"数据形状: {self.data.shape}")
        
        # 创建标签
        self.data['label'] = self.data['染色体的非整倍体'].fillna('').astype(str)
        self.data['label'] = (self.data['label'] != '').astype(int)
        
        print(f"标签分布: {self.data['label'].value_counts()}")
        print(f"异常样本比例: {self.data['label'].mean():.3f}")
        
        return self.data
    
    def create_enhanced_features(self):
        """创建增强特征，但保持简洁"""
        print("\n创建增强特征...")
        
        # 1. 基础特征
        base_features = [
            '年龄', '孕妇BMI',
            '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
            'GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
            '被过滤掉读段数的比例', '重复读段的比例', '在参考基因组上比对的比例'
        ]
        
        # 检查可用特征
        available_features = [col for col in base_features if col in self.data.columns]
        print(f"基础特征数量: {len(available_features)}")
        
        # 2. 创建关键衍生特征
        enhanced_features = available_features.copy()
        
        # Z值的绝对值（异常程度）
        for chrom in ['13', '18', '21']:
            z_col = f'{chrom}号染色体的Z值'
            if z_col in self.data.columns:
                self.data[f'{chrom}_z_abs'] = np.abs(self.data[z_col])
                enhanced_features.append(f'{chrom}_z_abs')
        
        # X染色体Z值的稳定性
        if 'X染色体的Z值' in self.data.columns:
            self.data['x_z_stability'] = np.where(
                (self.data['X染色体的Z值'] >= -2) & (self.data['X染色体的Z值'] <= 2), 1, 0
            )
            enhanced_features.append('x_z_stability')
        
        # GC含量偏差
        if 'GC含量' in self.data.columns:
            gc_mean = self.data['GC含量'].mean()
            for chrom in ['13', '18', '21']:
                gc_col = f'{chrom}号染色体的GC含量'
                if gc_col in self.data.columns:
                    self.data[f'{chrom}_gc_deviation'] = self.data[gc_col] - gc_mean
                    enhanced_features.append(f'{chrom}_gc_deviation')
        
        # 测序质量综合评分
        quality_components = []
        if '被过滤掉读段数的比例' in self.data.columns:
            self.data['filter_quality'] = 1 - self.data['被过滤掉读段数的比例']
            quality_components.append('filter_quality')
        
        if '重复读段的比例' in self.data.columns:
            self.data['duplicate_quality'] = 1 - self.data['重复读段的比例']
            quality_components.append('duplicate_quality')
        
        if '在参考基因组上比对的比例' in self.data.columns:
            self.data['mapping_quality'] = self.data['在参考基因组上比对的比例']
            quality_components.append('mapping_quality')
        
        if quality_components:
            self.data['overall_quality'] = self.data[quality_components].mean(axis=1)
            enhanced_features.append('overall_quality')
        
        # 临床风险评分
        if '年龄' in self.data.columns:
            self.data['age_risk'] = np.where(self.data['年龄'] > 35, 1, 0)
            enhanced_features.append('age_risk')
        
        if '孕妇BMI' in self.data.columns:
            self.data['bmi_risk'] = np.where(self.data['孕妇BMI'] > 30, 1, 0)
            enhanced_features.append('bmi_risk')
        
        # 3. 特征交互项（选择性添加）
        # 年龄与BMI的交互
        if '年龄' in self.data.columns and '孕妇BMI' in self.data.columns:
            self.data['age_bmi_interaction'] = self.data['年龄'] * self.data['孕妇BMI']
            enhanced_features.append('age_bmi_interaction')
        
        # Z值与质量的交互
        for chrom in ['13', '18', '21']:
            z_col = f'{chrom}号染色体的Z值'
            if z_col in self.data.columns and 'overall_quality' in self.data.columns:
                self.data[f'{chrom}_z_quality_interaction'] = self.data[z_col] * self.data['overall_quality']
                enhanced_features.append(f'{chrom}_z_quality_interaction')
        
        print(f"增强特征数量: {len(enhanced_features)}")
        
        # 准备特征矩阵
        self.X = self.data[enhanced_features].copy()
        self.y = self.data['label'].copy()
        self.feature_names = enhanced_features
        
        # 处理缺失值
        self.X = self.X.fillna(self.X.median())
        
        print(f"特征矩阵形状: {self.X.shape}")
        
        return self.X, self.y
    
    def intelligent_feature_selection(self):
        """智能特征选择"""
        print("\n进行智能特征选择...")
        
        # 1. 基于统计检验的特征选择
        selector_f = SelectKBest(score_func=f_classif, k=min(15, self.X.shape[1]))
        X_selected_f = selector_f.fit_transform(self.X, self.y)
        selected_features_f = [self.feature_names[i] for i in selector_f.get_support(indices=True)]
        
        print(f"统计检验选择的特征数量: {len(selected_features_f)}")
        
        # 2. 基于递归特征消除的特征选择
        lr_temp = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        selector_rfe = RFE(estimator=lr_temp, n_features_to_select=min(12, self.X.shape[1]))
        X_selected_rfe = selector_rfe.fit_transform(self.X, self.y)
        selected_features_rfe = [self.feature_names[i] for i in selector_rfe.get_support(indices=True)]
        
        print(f"递归特征消除选择的特征数量: {len(selected_features_rfe)}")
        
        # 3. 综合特征选择（取交集，确保核心特征）
        core_features = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
        available_core_features = [f for f in core_features if f in self.feature_names]
        
        # 合并所有选择的特征
        all_selected = list(set(selected_features_f + selected_features_rfe + available_core_features))
        
        # 如果特征太多，按重要性排序选择前12个
        if len(all_selected) > 12:
            # 使用统计检验的分数作为重要性
            scores = selector_f.scores_
            feature_scores = dict(zip(self.feature_names, scores))
            all_selected = sorted(all_selected, key=lambda x: feature_scores.get(x, 0), reverse=True)[:12]
        
        print(f"最终选择的特征数量: {len(all_selected)}")
        print(f"选择的特征: {all_selected}")
        
        # 更新特征矩阵
        self.X = self.X[all_selected]
        self.feature_names = all_selected
        
        print(f"特征选择后的矩阵形状: {self.X.shape}")
        
        return self.X, self.y
    
    def train_optimized_lr_model(self):
        """训练优化的逻辑回归模型"""
        print("\n训练优化的逻辑回归模型...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(self.X)
        
        # 逻辑回归模型
        self.lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=2000  # 增加迭代次数
        )
        
        # 扩展的超参数网格
        lr_params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # 用于elasticnet
        }
        
        # 分层K折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 网格搜索
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
        
        pd.DataFrame([best_params]).to_csv(f"{self.output_path}/improved_lr_best_parameters.csv", index=False)
        
        return self.lr_model
    
    def comprehensive_cross_validation(self):
        """全面的交叉验证评估"""
        print("\n进行全面交叉验证评估...")
        
        X_scaled = self.scaler.transform(self.X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        auc_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []
        
        print("评估改进版逻辑回归模型...")
        
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
        cv_df = pd.DataFrame([cv_results], index=['Improved Logistic Regression'])
        cv_df.to_csv(f"{self.output_path}/improved_lr_cross_validation_results.csv")
        
        print("改进版逻辑回归交叉验证结果:")
        print(cv_df)
        
        self.cv_results = cv_results
        return cv_results
    
    def advanced_threshold_optimization(self):
        """高级阈值优化"""
        print("\n进行高级阈值优化...")
        
        # 在全部数据上重新训练最终模型
        X_scaled = self.scaler.fit_transform(self.X)
        
        self.lr_model.fit(X_scaled, self.y)
        
        # 预测
        lr_pred_proba = self.lr_model.predict_proba(X_scaled)[:, 1]
        
        # 计算不同阈值下的性能
        thresholds = np.arange(0.01, 0.99, 0.01)
        results = []
        
        for threshold in thresholds:
            y_pred = (lr_pred_proba >= threshold).astype(int)
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(self.y, y_pred).ravel()
            
            # 计算指标
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            # 计算Youden指数
            youden = sensitivity + specificity - 1
            
            results.append({
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1,
                'youden_index': youden
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{self.output_path}/improved_lr_threshold_analysis.csv", index=False)
        
        # 多种阈值选择策略
        # 1. F1分数最高的阈值
        best_f1_idx = results_df['f1_score'].idxmax()
        optimal_f1_threshold = results_df.loc[best_f1_idx, 'threshold']
        
        # 2. Youden指数最高的阈值
        best_youden_idx = results_df['youden_index'].idxmax()
        youden_threshold = results_df.loc[best_youden_idx, 'threshold']
        
        # 3. 基于临床需求的阈值
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
        
        # 4. 平衡阈值：精确率和召回率最接近的阈值
        precision_recall_diff = np.abs(results_df['precision'] - results_df['sensitivity'])
        balanced_idx = precision_recall_diff.idxmin()
        balanced_threshold = results_df.loc[balanced_idx, 'threshold']
        
        print(f"阈值优化结果:")
        print(f"  最优F1阈值: {optimal_f1_threshold:.3f}")
        print(f"  Youden指数最优阈值: {youden_threshold:.3f}")
        print(f"  低风险阈值 (DL): {dl:.3f}")
        print(f"  高风险阈值 (DH): {dh:.3f}")
        print(f"  平衡阈值: {balanced_threshold:.3f}")
        print(f"  不确定区间: {dh-dl:.3f} ({(dh-dl)*100:.1f}%)")
        
        # 保存阈值决策规则
        threshold_rules = {
            'optimal_f1_threshold': optimal_f1_threshold,
            'youden_threshold': youden_threshold,
            'low_risk_threshold': dl,
            'high_risk_threshold': dh,
            'balanced_threshold': balanced_threshold,
            'uncertain_interval': dh - dl,
            'decision_rules': {
                'high_risk': f'D >= {dh:.3f} -> 高风险异常',
                'low_risk': f'D < {dl:.3f} -> 低风险正常',
                'uncertain': f'{dl:.3f} <= D < {dh:.3f} -> 结果不确定，建议复检'
            }
        }
        
        pd.DataFrame([threshold_rules]).to_csv(f"{self.output_path}/improved_lr_threshold_rules.csv", index=False)
        
        return optimal_f1_threshold, youden_threshold, dl, dh, balanced_threshold
    
    def final_evaluation(self):
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
        print("改进版逻辑回归最终模型评估结果 (阈值=0.5)")
        print("=" * 60)
        print(f"AUC: {auc:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"准确率: {accuracy:.4f}")
        print("=" * 60)
        
        # 保存最终评估结果
        final_results = {
            'Model': 'Improved Logistic Regression',
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Accuracy': accuracy
        }
        
        pd.DataFrame([final_results]).to_csv(f"{self.output_path}/improved_lr_final_evaluation.csv", index=False)
        
        return lr_pred_proba
    
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
        feature_importance.to_csv(f"{self.output_path}/improved_lr_feature_importance.csv", index=False)
        
        print("特征重要性分析完成")
        print("前10个重要特征:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        return feature_importance
    
    def run_improved_analysis(self):
        """运行改进版分析流程"""
        print("开始改进版原始逻辑回归分析...")
        
        # 1. 数据加载与预处理
        self.load_and_preprocess_data()
        
        # 2. 创建增强特征
        self.create_enhanced_features()
        
        # 3. 智能特征选择
        self.intelligent_feature_selection()
        
        # 4. 训练优化的逻辑回归模型
        self.train_optimized_lr_model()
        
        # 5. 全面交叉验证评估
        self.comprehensive_cross_validation()
        
        # 6. 高级阈值优化
        optimal_f1_threshold, youden_threshold, dl, dh, balanced_threshold = self.advanced_threshold_optimization()
        
        # 7. 最终模型评估
        pred_proba = self.final_evaluation()
        
        # 8. 分析特征重要性
        self.analyze_feature_importance()
        
        print("\n改进版原始逻辑回归分析完成！")
        print(f"所有结果已保存到: {self.output_path}")
        
        return pred_proba


def main():
    """主函数"""
    # 设置路径
    data_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/girl_output.csv"
    output_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/4"
    
    # 创建改进版模型实例
    model = ImprovedOriginalLR(data_path, output_path)
    
    # 运行改进版分析
    model.run_improved_analysis()


if __name__ == "__main__":
    main()
