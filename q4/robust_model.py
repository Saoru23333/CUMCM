#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：防过拟合的鲁棒女胎异常判定模型
重点解决过拟合问题，提高泛化能力
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                           classification_report, confusion_matrix, f1_score,
                           precision_score, recall_score, accuracy_score)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class RobustAntiOverfittingModel:
    """防过拟合的鲁棒女胎异常判定模型"""
    
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.models = {}
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
    
    def create_essential_features(self):
        """创建核心特征，避免过度特征工程"""
        print("\n创建核心特征...")
        
        # 1. 核心Z值特征
        z_features = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
        available_z_features = [f for f in z_features if f in self.data.columns]
        
        for feature in available_z_features:
            self.data[f'{feature}_abs'] = np.abs(self.data[feature])
            self.data[f'{feature}_squared'] = self.data[feature] ** 2
        
        # 2. GC含量特征
        gc_features = ['GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
        available_gc_features = [f for f in gc_features if f in self.data.columns]
        
        if 'GC含量' in self.data.columns:
            gc_mean = self.data['GC含量'].mean()
            for feature in available_gc_features:
                if feature != 'GC含量':
                    self.data[f'{feature}_deviation'] = self.data[feature] - gc_mean
        
        # 3. 测序质量特征
        quality_features = []
        if '被过滤掉读段数的比例' in self.data.columns:
            self.data['filter_quality'] = 1 - self.data['被过滤掉读段数的比例']
            quality_features.append('filter_quality')
        
        if '重复读段的比例' in self.data.columns:
            self.data['duplicate_quality'] = 1 - self.data['重复读段的比例']
            quality_features.append('duplicate_quality')
        
        if '在参考基因组上比对的比例' in self.data.columns:
            self.data['mapping_quality'] = self.data['在参考基因组上比对的比例']
            quality_features.append('mapping_quality')
        
        # 综合质量得分
        if quality_features:
            self.data['overall_quality'] = self.data[quality_features].mean(axis=1)
        
        # 4. 临床特征
        if '年龄' in self.data.columns:
            self.data['age_risk'] = np.where(self.data['年龄'] > 35, 1, 0)
        
        if '孕妇BMI' in self.data.columns:
            self.data['bmi_risk'] = np.where(self.data['孕妇BMI'] > 30, 1, 0)
        
        # 5. X染色体特征
        if 'X染色体的Z值' in self.data.columns:
            self.data['x_z_stability'] = np.where(
                (self.data['X染色体的Z值'] >= -2) & (self.data['X染色体的Z值'] <= 2), 1, 0
            )
        
        print("核心特征创建完成")
        
        return self.data
    
    def select_robust_features(self):
        """选择鲁棒特征，避免过拟合"""
        print("\n选择鲁棒特征...")
        
        # 基础特征集
        base_features = [
            '年龄', '孕妇BMI',
            '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
            'GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
            '被过滤掉读段数的比例', '重复读段的比例', '在参考基因组上比对的比例'
        ]
        
        # 检查可用特征
        available_base_features = [f for f in base_features if f in self.data.columns]
        
        # 添加工程特征
        engineered_features = []
        for feature in available_base_features:
            if 'Z值' in feature:
                engineered_features.extend([f'{feature}_abs', f'{feature}_squared'])
            elif 'GC含量' in feature and feature != 'GC含量':
                engineered_features.append(f'{feature}_deviation')
        
        # 添加其他工程特征
        other_features = ['filter_quality', 'duplicate_quality', 'mapping_quality', 
                         'overall_quality', 'age_risk', 'bmi_risk', 'x_z_stability']
        available_other_features = [f for f in other_features if f in self.data.columns]
        
        # 合并所有特征
        all_features = available_base_features + engineered_features + available_other_features
        
        # 检查特征是否存在
        available_features = [col for col in all_features if col in self.data.columns]
        
        print(f"候选特征数量: {len(available_features)}")
        
        # 准备特征矩阵
        X_temp = self.data[available_features].copy()
        y_temp = self.data['label'].copy()
        
        # 处理缺失值
        X_temp = X_temp.fillna(X_temp.median())
        
        # 特征选择：选择最重要的10个特征
        selector = SelectKBest(score_func=f_classif, k=min(10, len(available_features)))
        X_selected = selector.fit_transform(X_temp, y_temp)
        selected_indices = selector.get_support(indices=True)
        selected_features = [available_features[i] for i in selected_indices]
        
        print(f"选择的特征数量: {len(selected_features)}")
        print(f"选择的特征: {selected_features}")
        
        # 更新特征矩阵
        self.X = X_temp[selected_features]
        self.y = y_temp
        self.feature_names = selected_features
        
        print(f"最终特征矩阵形状: {self.X.shape}")
        
        return self.X, self.y
    
    def train_robust_models(self):
        """训练鲁棒模型，防止过拟合"""
        print("\n训练鲁棒模型...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(self.X)
        
        # 1. 强正则化逻辑回归
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # 强正则化参数
        lr_params = {
            'C': [0.001, 0.01, 0.1, 1],  # 更强的正则化
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        # 2. 保守的随机森林
        rf_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_estimators=50,  # 减少树的数量
            max_depth=3,      # 限制深度
            min_samples_split=20,  # 增加分割要求
            min_samples_leaf=10,   # 增加叶子节点要求
            max_features='sqrt'    # 限制特征数量
        )
        
        # 保守的随机森林参数
        rf_params = {
            'max_depth': [2, 3, 4],
            'min_samples_split': [15, 20, 25],
            'min_samples_leaf': [8, 10, 12],
            'max_features': ['sqrt', 'log2']
        }
        
        # 分层K折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 网格搜索
        print("逻辑回归超参数调优...")
        lr_grid = GridSearchCV(
            lr_model, lr_params,
            cv=skf, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        lr_grid.fit(X_scaled, self.y)
        self.models['lr'] = lr_grid.best_estimator_
        
        print("随机森林超参数调优...")
        rf_grid = GridSearchCV(
            rf_model, rf_params,
            cv=skf, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        rf_grid.fit(X_scaled, self.y)
        self.models['rf'] = rf_grid.best_estimator_
        
        print(f"逻辑回归最佳参数: {lr_grid.best_params_}")
        print(f"逻辑回归最佳交叉验证AUC: {lr_grid.best_score_:.4f}")
        print(f"随机森林最佳参数: {rf_grid.best_params_}")
        print(f"随机森林最佳交叉验证AUC: {rf_grid.best_score_:.4f}")
        
        return self.models
    
    def comprehensive_cross_validation(self):
        """全面的交叉验证评估"""
        print("\n进行全面交叉验证评估...")
        
        X_scaled = self.scaler.transform(self.X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"评估{model_name}模型...")
            
            auc_scores = []
            f1_scores = []
            precision_scores = []
            recall_scores = []
            accuracy_scores = []
            
            for train_idx, val_idx in skf.split(X_scaled, self.y):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred_proba = model.decision_function(X_val)
                    y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
                
                y_pred = (y_pred_proba >= 0.5).astype(int)
                
                # 计算指标
                auc_scores.append(roc_auc_score(y_val, y_pred_proba))
                f1_scores.append(f1_score(y_val, y_pred))
                precision_scores.append(precision_score(y_val, y_pred))
                recall_scores.append(recall_score(y_val, y_pred))
                accuracy_scores.append(accuracy_score(y_val, y_pred))
            
            results[model_name] = {
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
        cv_df = pd.DataFrame(results).T
        cv_df.to_csv(f"{self.output_path}/robust_cross_validation_results.csv")
        
        print("鲁棒模型交叉验证结果:")
        print(cv_df)
        
        self.cv_results = results
        return results
    
    def final_evaluation_with_holdout(self):
        """使用留出法进行最终评估"""
        print("\n使用留出法进行最终评估...")
        
        # 留出20%作为测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        final_results = {}
        
        for model_name, model in self.models.items():
            print(f"最终评估{model_name}模型...")
            
            # 训练模型
            model.fit(X_train_scaled, y_train)
            
            # 预测
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test_scaled)
                y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # 计算指标
            auc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            final_results[model_name] = {
                'AUC': auc,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Accuracy': accuracy
            }
            
            print(f"{model_name}模型测试集结果:")
            print(f"  AUC: {auc:.4f}")
            print(f"  F1分数: {f1:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  精确率: {precision:.4f}")
            print(f"  准确率: {accuracy:.4f}")
        
        # 保存最终结果
        final_df = pd.DataFrame(final_results).T
        final_df.to_csv(f"{self.output_path}/robust_final_evaluation.csv")
        
        return final_results
    
    def analyze_overfitting_risk(self):
        """分析过拟合风险"""
        print("\n分析过拟合风险...")
        
        # 计算训练集性能
        X_scaled = self.scaler.fit_transform(self.X)
        
        training_results = {}
        for model_name, model in self.models.items():
            model.fit(X_scaled, self.y)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_scaled)
                y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            training_results[model_name] = {
                'AUC': roc_auc_score(self.y, y_pred_proba),
                'F1_Score': f1_score(self.y, y_pred),
                'Precision': precision_score(self.y, y_pred),
                'Recall': recall_score(self.y, y_pred),
                'Accuracy': accuracy_score(self.y, y_pred)
            }
        
        # 对比训练集和交叉验证结果
        print("\n过拟合风险分析:")
        print("-" * 60)
        
        for model_name in self.models.keys():
            train_auc = training_results[model_name]['AUC']
            cv_auc = self.cv_results[model_name]['AUC_mean']
            auc_gap = train_auc - cv_auc
            
            train_f1 = training_results[model_name]['F1_Score']
            cv_f1 = self.cv_results[model_name]['F1_mean']
            f1_gap = train_f1 - cv_f1
            
            print(f"{model_name}模型:")
            print(f"  训练集AUC: {train_auc:.4f}")
            print(f"  交叉验证AUC: {cv_auc:.4f}")
            print(f"  AUC差异: {auc_gap:.4f} ({'⚠️ 过拟合风险' if auc_gap > 0.05 else '✅ 正常'})")
            print(f"  训练集F1: {train_f1:.4f}")
            print(f"  交叉验证F1: {cv_f1:.4f}")
            print(f"  F1差异: {f1_gap:.4f} ({'⚠️ 过拟合风险' if f1_gap > 0.1 else '✅ 正常'})")
            print()
        
        return training_results
    
    def run_robust_analysis(self):
        """运行鲁棒分析流程"""
        print("开始防过拟合的鲁棒分析...")
        
        # 1. 数据加载与预处理
        self.load_and_preprocess_data()
        
        # 2. 创建核心特征
        self.create_essential_features()
        
        # 3. 选择鲁棒特征
        self.select_robust_features()
        
        # 4. 训练鲁棒模型
        self.train_robust_models()
        
        # 5. 全面交叉验证评估
        self.comprehensive_cross_validation()
        
        # 6. 留出法最终评估
        final_results = self.final_evaluation_with_holdout()
        
        # 7. 过拟合风险分析
        training_results = self.analyze_overfitting_risk()
        
        print("\n防过拟合的鲁棒分析完成！")
        print(f"所有结果已保存到: {self.output_path}")
        
        return final_results, training_results


def main():
    """主函数"""
    # 设置路径
    data_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/girl_output.csv"
    output_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/4"
    
    # 创建鲁棒模型实例
    model = RobustAntiOverfittingModel(data_path, output_path)
    
    # 运行鲁棒分析
    model.run_robust_analysis()


if __name__ == "__main__":
    main()
