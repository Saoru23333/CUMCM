#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：女胎异常判定的改进版多因素加权Z值融合模型
改进策略：
1. 优化特征选择和特征工程策略
2. 改进权重融合算法
3. 实现更精细的动态阈值策略
4. 添加集成学习方法
5. 优化类别不平衡处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                           classification_report, confusion_matrix, f1_score,
                           precision_score, recall_score, accuracy_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedFeatureEngineeringModel:
    """改进版手动特征工程的女胎异常判定模型"""
    
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = RobustScaler()  # 使用RobustScaler，对异常值更鲁棒
        self.models = {}
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
    
    def advanced_gc_quality_correction(self):
        """改进的GC含量与测序质量校正"""
        print("\n创建改进的GC含量与测序质量校正因子...")
        
        # 1. 更精细的GC含量偏差校正
        if 'GC含量' in self.data.columns:
            gc_mean = self.data['GC含量'].mean()
            gc_std = self.data['GC含量'].std()
            
            for chrom in ['13', '18', '21']:
                gc_col = f'{chrom}号染色体的GC含量'
                if gc_col in self.data.columns:
                    # 标准化GC含量偏差
                    self.data[f'{chrom}_gc_std'] = (self.data[gc_col] - gc_mean) / gc_std
                    # 相对GC含量偏差
                    self.data[f'{chrom}_gc_relative'] = (self.data[gc_col] - gc_mean) / gc_mean
                    # GC含量稳定性指标
                    self.data[f'{chrom}_gc_stability'] = 1 / (1 + np.abs(self.data[f'{chrom}_gc_std']))
        
        # 2. 综合测序质量评分
        quality_components = []
        
        if '被过滤掉读段数的比例' in self.data.columns:
            # 过滤率质量评分（指数衰减）
            self.data['filter_quality'] = np.exp(-5 * self.data['被过滤掉读段数的比例'])
            quality_components.append('filter_quality')
        
        if '重复读段的比例' in self.data.columns:
            # 重复率质量评分
            self.data['duplicate_quality'] = np.exp(-10 * self.data['重复读段的比例'])
            quality_components.append('duplicate_quality')
        
        if '在参考基因组上比对的比例' in self.data.columns:
            # 比对率质量评分
            self.data['mapping_quality'] = self.data['在参考基因组上比对的比例'] ** 2
            quality_components.append('mapping_quality')
        
        if '唯一比对的读段数' in self.data.columns:
            # 读段数质量评分（对数变换）
            self.data['read_count_quality'] = np.log1p(self.data['唯一比对的读段数']) / np.log1p(self.data['唯一比对的读段数'].max())
            quality_components.append('read_count_quality')
        
        # 综合质量得分（加权几何平均）
        if quality_components:
            quality_df = self.data[quality_components]
            # 避免0值
            quality_df = quality_df + 1e-8
            self.data['comprehensive_quality'] = quality_df.prod(axis=1) ** (1/len(quality_components))
            # 标准化
            self.data['quality_correction_factor'] = (self.data['comprehensive_quality'] - self.data['comprehensive_quality'].mean()) / self.data['comprehensive_quality'].std()
        
        print("改进的GC含量与测序质量校正因子创建完成")
        
        return self.data
    
    def enhanced_x_chromosome_reference(self):
        """增强的X染色体背景参考"""
        print("\n创建增强的X染色体背景参考...")
        
        # 1. X染色体Z值的多维度分析
        if 'X染色体的Z值' in self.data.columns:
            x_z = self.data['X染色体的Z值']
            
            # 基础标准化
            self.data['x_z_standardized'] = (x_z - x_z.mean()) / x_z.std()
            
            # X染色体稳定性评分
            x_z_normal_range = [-2, 2]
            self.data['x_z_stability_score'] = np.where(
                (x_z >= x_z_normal_range[0]) & (x_z <= x_z_normal_range[1]), 
                1.0, 
                1.0 / (1.0 + np.abs(x_z - np.clip(x_z, x_z_normal_range[0], x_z_normal_range[1])))
            )
            
            # X染色体异常指示器
            self.data['x_z_abnormal'] = (np.abs(x_z) > 2).astype(int)
            
            # X染色体趋势指标（相对于正常范围的中心偏移）
            x_z_center = (x_z_normal_range[0] + x_z_normal_range[1]) / 2
            self.data['x_z_center_offset'] = x_z - x_z_center
        
        # 2. X染色体浓度的深度分析
        if 'X染色体浓度' in self.data.columns:
            x_conc = self.data['X染色体浓度']
            
            # 浓度标准化
            self.data['x_conc_standardized'] = (x_conc - x_conc.mean()) / x_conc.std()
            
            # 浓度稳定性评分
            conc_std = x_conc.std()
            self.data['x_conc_stability'] = 1 / (1 + np.abs(x_conc - x_conc.mean()) / conc_std)
            
            # 浓度异常指示器
            conc_threshold = 2 * conc_std
            self.data['x_conc_abnormal'] = (np.abs(x_conc - x_conc.mean()) > conc_threshold).astype(int)
        
        print("增强的X染色体背景参考创建完成")
        
        return self.data
    
    def intelligent_z_value_fusion(self):
        """智能Z值融合算法"""
        print("\n创建智能Z值融合算法...")
        
        target_chromosomes = ['13', '18', '21']
        
        for chrom in target_chromosomes:
            z_col = f'{chrom}号染色体的Z值'
            if z_col in self.data.columns:
                z_values = self.data[z_col]
                
                # 1. 基础Z值处理
                self.data[f'{chrom}_z_raw'] = z_values
                
                # 2. 多维度校正的Z值
                correction_components = []
                correction_weights = []
                
                # GC含量校正
                gc_correction_col = f'{chrom}_gc_std'
                if gc_correction_col in self.data.columns:
                    gc_corrected = z_values - 0.3 * self.data[gc_correction_col]
                    correction_components.append(gc_corrected)
                    correction_weights.append(0.3)
                
                # 质量校正
                if 'quality_correction_factor' in self.data.columns:
                    quality_corrected = z_values - 0.2 * self.data['quality_correction_factor']
                    correction_components.append(quality_corrected)
                    correction_weights.append(0.2)
                
                # X染色体背景校正
                if 'x_z_standardized' in self.data.columns:
                    x_corrected = z_values - 0.15 * self.data['x_z_standardized']
                    correction_components.append(x_corrected)
                    correction_weights.append(0.15)
                
                # 3. 智能融合策略
                if correction_components:
                    # 加权融合
                    weights = np.array(correction_weights)
                    weights = weights / weights.sum()  # 归一化权重
                    
                    self.data[f'{chrom}_z_intelligent_fused'] = np.average(
                        correction_components, axis=0, weights=weights
                    )
                else:
                    self.data[f'{chrom}_z_intelligent_fused'] = z_values
                
                # 4. 创建增强的风险评分
                # 基础风险评分
                self.data[f'{chrom}_risk_basic'] = np.abs(self.data[f'{chrom}_z_intelligent_fused'])
                
                # 加权风险评分（考虑Z值大小和稳定性）
                z_abs = np.abs(self.data[f'{chrom}_z_intelligent_fused'])
                # 非线性风险评分
                self.data[f'{chrom}_risk_enhanced'] = np.where(
                    z_abs <= 1, z_abs,
                    np.where(z_abs <= 2, 1 + (z_abs - 1) * 2, 3 + (z_abs - 2) * 3)
                )
                
                # 5. 异常指示器（多阈值）
                self.data[f'{chrom}_abnormal_mild'] = (np.abs(self.data[f'{chrom}_z_intelligent_fused']) > 1.5).astype(int)
                self.data[f'{chrom}_abnormal_moderate'] = (np.abs(self.data[f'{chrom}_z_intelligent_fused']) > 2.0).astype(int)
                self.data[f'{chrom}_abnormal_severe'] = (np.abs(self.data[f'{chrom}_z_intelligent_fused']) > 2.5).astype(int)
        
        print("智能Z值融合算法创建完成")
        
        return self.data
    
    def advanced_clinical_integration(self):
        """高级临床指标整合"""
        print("\n创建高级临床指标整合...")
        
        # 1. 年龄的精细处理
        if '年龄' in self.data.columns:
            age = self.data['年龄']
            
            # 年龄标准化
            self.data['age_standardized'] = (age - age.mean()) / age.std()
            
            # 年龄风险分层
            self.data['age_risk_low'] = (age < 30).astype(int)
            self.data['age_risk_medium'] = ((age >= 30) & (age < 35)).astype(int)
            self.data['age_risk_high'] = (age >= 35).astype(int)
            
            # 年龄风险评分（连续）
            self.data['age_risk_continuous'] = np.maximum(0, (age - 30) / 10)
        
        # 2. BMI的精细处理
        if '孕妇BMI' in self.data.columns:
            bmi = self.data['孕妇BMI']
            
            # BMI标准化
            self.data['bmi_standardized'] = (bmi - bmi.mean()) / bmi.std()
            
            # BMI风险分层
            self.data['bmi_risk_underweight'] = (bmi < 18.5).astype(int)
            self.data['bmi_risk_normal'] = ((bmi >= 18.5) & (bmi < 25)).astype(int)
            self.data['bmi_risk_overweight'] = ((bmi >= 25) & (bmi < 30)).astype(int)
            self.data['bmi_risk_obese'] = (bmi >= 30).astype(int)
            
            # BMI风险评分（连续）
            self.data['bmi_risk_continuous'] = np.maximum(0, (bmi - 25) / 10)
        
        # 3. 综合临床风险评分
        clinical_components = []
        clinical_weights = []
        
        if 'age_risk_continuous' in self.data.columns:
            clinical_components.append(self.data['age_risk_continuous'])
            clinical_weights.append(0.4)
        
        if 'bmi_risk_continuous' in self.data.columns:
            clinical_components.append(self.data['bmi_risk_continuous'])
            clinical_weights.append(0.3)
        
        # 添加交互项
        if 'age_standardized' in self.data.columns and 'bmi_standardized' in self.data.columns:
            self.data['age_bmi_interaction'] = self.data['age_standardized'] * self.data['bmi_standardized']
            clinical_components.append(self.data['age_bmi_interaction'])
            clinical_weights.append(0.3)
        
        if clinical_components:
            weights = np.array(clinical_weights)
            weights = weights / weights.sum()
            self.data['clinical_risk_enhanced'] = np.average(clinical_components, axis=0, weights=weights)
        else:
            self.data['clinical_risk_enhanced'] = 0
        
        print("高级临床指标整合完成")
        
        return self.data
    
    def create_comprehensive_features(self):
        """创建综合特征集"""
        print("\n创建综合特征集...")
        
        # 1. 核心融合特征
        core_features = []
        for chrom in ['13', '18', '21']:
            fused_z_col = f'{chrom}_z_intelligent_fused'
            if fused_z_col in self.data.columns:
                core_features.append(fused_z_col)
        
        # 2. 增强的风险评分
        risk_features = []
        for chrom in ['13', '18', '21']:
            risk_col = f'{chrom}_risk_enhanced'
            if risk_col in self.data.columns:
                risk_features.append(risk_col)
        
        # 3. 异常指示器
        indicator_features = []
        for chrom in ['13', '18', '21']:
            for severity in ['mild', 'moderate', 'severe']:
                indicator_col = f'{chrom}_abnormal_{severity}'
                if indicator_col in self.data.columns:
                    indicator_features.append(indicator_col)
        
        # 4. X染色体特征
        x_features = []
        x_feature_candidates = [
            'x_z_standardized', 'x_z_stability_score', 'x_z_abnormal',
            'x_conc_standardized', 'x_conc_stability', 'x_conc_abnormal'
        ]
        for feature in x_feature_candidates:
            if feature in self.data.columns:
                x_features.append(feature)
        
        # 5. 临床特征
        clinical_features = []
        clinical_candidates = [
            'age_standardized', 'bmi_standardized', 'clinical_risk_enhanced',
            'age_risk_low', 'age_risk_medium', 'age_risk_high',
            'bmi_risk_underweight', 'bmi_risk_normal', 'bmi_risk_overweight', 'bmi_risk_obese',
            'age_bmi_interaction'
        ]
        for feature in clinical_candidates:
            if feature in self.data.columns:
                clinical_features.append(feature)
        
        # 6. 质量特征
        quality_features = []
        quality_candidates = [
            'quality_correction_factor', 'comprehensive_quality',
            'filter_quality', 'duplicate_quality', 'mapping_quality', 'read_count_quality'
        ]
        for feature in quality_candidates:
            if feature in self.data.columns:
                quality_features.append(feature)
        
        # 7. GC含量特征
        gc_features = []
        for chrom in ['13', '18', '21']:
            for gc_type in ['gc_std', 'gc_relative', 'gc_stability']:
                gc_col = f'{chrom}_{gc_type}'
                if gc_col in self.data.columns:
                    gc_features.append(gc_col)
        
        # 合并所有特征
        all_features = (core_features + risk_features + indicator_features + 
                       x_features + clinical_features + quality_features + gc_features)
        
        # 检查特征是否存在
        available_features = [col for col in all_features if col in self.data.columns]
        print(f"综合特征集数量: {len(available_features)}")
        print(f"特征类别分布:")
        print(f"  核心融合特征: {len([f for f in available_features if f in core_features])}")
        print(f"  风险评分特征: {len([f for f in available_features if f in risk_features])}")
        print(f"  异常指示器: {len([f for f in available_features if f in indicator_features])}")
        print(f"  X染色体特征: {len([f for f in available_features if f in x_features])}")
        print(f"  临床特征: {len([f for f in available_features if f in clinical_features])}")
        print(f"  质量特征: {len([f for f in available_features if f in quality_features])}")
        print(f"  GC含量特征: {len([f for f in available_features if f in gc_features])}")
        
        # 准备特征矩阵和标签
        self.X = self.data[available_features].copy()
        self.y = self.data['label'].copy()
        self.feature_names = available_features
        
        # 处理缺失值
        self.X = self.X.fillna(self.X.median())
        
        print(f"最终特征矩阵形状: {self.X.shape}")
        print(f"标签分布: {self.y.value_counts()}")
        
        return self.X, self.y
    
    def intelligent_feature_selection(self):
        """智能特征选择"""
        print("\n进行智能特征选择...")
        
        # 1. 基于统计检验的特征选择
        selector_f = SelectKBest(score_func=f_classif, k=min(20, self.X.shape[1]))
        X_selected_f = selector_f.fit_transform(self.X, self.y)
        selected_features_f = [self.feature_names[i] for i in selector_f.get_support(indices=True)]
        
        print(f"统计检验选择的特征数量: {len(selected_features_f)}")
        
        # 2. 基于递归特征消除的特征选择
        lr_temp = LogisticRegression(random_state=42, max_iter=1000)
        selector_rfe = RFE(estimator=lr_temp, n_features_to_select=min(15, self.X.shape[1]))
        X_selected_rfe = selector_rfe.fit_transform(self.X, self.y)
        selected_features_rfe = [self.feature_names[i] for i in selector_rfe.get_support(indices=True)]
        
        print(f"递归特征消除选择的特征数量: {len(selected_features_rfe)}")
        
        # 3. 特征重要性分析
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(self.X, self.y)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 选择重要性前15的特征
        top_features = feature_importance.head(15)['feature'].tolist()
        
        print(f"随机森林重要性选择的特征数量: {len(top_features)}")
        
        # 4. 综合特征选择（取交集和并集）
        # 取三种方法的并集，但限制在20个特征以内
        all_selected = list(set(selected_features_f + selected_features_rfe + top_features))
        
        if len(all_selected) > 20:
            # 如果特征太多，按重要性排序选择前20个
            importance_dict = dict(zip(feature_importance['feature'], feature_importance['importance']))
            all_selected = sorted(all_selected, key=lambda x: importance_dict.get(x, 0), reverse=True)[:20]
        
        print(f"最终选择的特征数量: {len(all_selected)}")
        
        # 更新特征矩阵
        self.X = self.X[all_selected]
        self.feature_names = all_selected
        
        print(f"特征选择后的矩阵形状: {self.X.shape}")
        
        return self.X, self.y
    
    def train_ensemble_models(self):
        """训练集成模型"""
        print("\n训练集成模型...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(self.X)
        
        # 1. 逻辑回归模型
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        lr_params = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        # 2. 随机森林模型
        rf_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_estimators=100
        )
        
        rf_params = {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # 3. 集成模型
        ensemble_model = VotingClassifier(
            estimators=[
                ('lr', lr_model),
                ('rf', rf_model)
            ],
            voting='soft'  # 使用概率投票
        )
        
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
        
        print("集成模型训练...")
        ensemble_model.fit(X_scaled, self.y)
        self.models['ensemble'] = ensemble_model
        
        print(f"逻辑回归最佳参数: {lr_grid.best_params_}")
        print(f"逻辑回归最佳AUC: {lr_grid.best_score_:.4f}")
        print(f"随机森林最佳参数: {rf_grid.best_params_}")
        print(f"随机森林最佳AUC: {rf_grid.best_score_:.4f}")
        
        return self.models
    
    def advanced_cross_validation(self):
        """高级交叉验证评估"""
        print("\n进行高级交叉验证评估...")
        
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
                    y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))  # sigmoid变换
                
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
        cv_df.to_csv(f"{self.output_path}/improved_cross_validation_results.csv")
        
        print("高级交叉验证结果:")
        print(cv_df)
        
        self.cv_results = results
        return results
    
    def final_evaluation_and_threshold_optimization(self):
        """最终评估和阈值优化"""
        print("\n进行最终评估和阈值优化...")
        
        X_scaled = self.scaler.fit_transform(self.X)
        
        final_results = {}
        
        for model_name, model in self.models.items():
            print(f"最终评估{model_name}模型...")
            
            # 训练最终模型
            model.fit(X_scaled, self.y)
            
            # 预测
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X_scaled)[:, 1]
            else:
                pred_proba = model.decision_function(X_scaled)
                pred_proba = 1 / (1 + np.exp(-pred_proba))
            
            # 基础评估指标
            pred_05 = (pred_proba >= 0.5).astype(int)
            auc = roc_auc_score(self.y, pred_proba)
            precision = precision_score(self.y, pred_05)
            recall = recall_score(self.y, pred_05)
            f1 = f1_score(self.y, pred_05)
            accuracy = accuracy_score(self.y, pred_05)
            
            # 阈值优化
            thresholds = np.arange(0.05, 0.95, 0.01)
            threshold_results = []
            
            for threshold in thresholds:
                y_pred = (pred_proba >= threshold).astype(int)
                
                tn, fp, fn, tp = confusion_matrix(self.y, y_pred).ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1_t = 2 * precision_t * sensitivity / (precision_t + sensitivity) if (precision_t + sensitivity) > 0 else 0
                
                threshold_results.append({
                    'threshold': threshold,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision_t,
                    'f1_score': f1_t
                })
            
            threshold_df = pd.DataFrame(threshold_results)
            
            # 寻找最优阈值
            # 1. F1分数最高的阈值
            best_f1_idx = threshold_df['f1_score'].idxmax()
            optimal_f1_threshold = threshold_df.loc[best_f1_idx, 'threshold']
            
            # 2. 基于临床需求的阈值
            high_sensitivity_idx = threshold_df[threshold_df['sensitivity'] >= 0.95].index
            dl = threshold_df.loc[high_sensitivity_idx[0], 'threshold'] if len(high_sensitivity_idx) > 0 else 0.1
            
            high_precision_idx = threshold_df[threshold_df['precision'] >= 0.95].index
            dh = threshold_df.loc[high_precision_idx[-1], 'threshold'] if len(high_precision_idx) > 0 else 0.8
            
            # 3. Youden指数最优阈值
            youden_scores = threshold_df['sensitivity'] + threshold_df['specificity'] - 1
            best_youden_idx = youden_scores.idxmax()
            youden_threshold = threshold_df.loc[best_youden_idx, 'threshold']
            
            final_results[model_name] = {
                'AUC': auc,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Accuracy': accuracy,
                'optimal_f1_threshold': optimal_f1_threshold,
                'low_risk_threshold': dl,
                'high_risk_threshold': dh,
                'youden_threshold': youden_threshold,
                'uncertain_interval': dh - dl
            }
            
            print(f"{model_name}模型结果:")
            print(f"  AUC: {auc:.4f}")
            print(f"  F1分数: {f1:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  精确率: {precision:.4f}")
            print(f"  最优F1阈值: {optimal_f1_threshold:.3f}")
            print(f"  低风险阈值: {dl:.3f}")
            print(f"  高风险阈值: {dh:.3f}")
        
        # 保存最终结果
        final_df = pd.DataFrame(final_results).T
        final_df.to_csv(f"{self.output_path}/improved_final_evaluation.csv")
        
        return final_results
    
    def run_complete_improved_analysis(self):
        """运行完整的改进分析流程"""
        print("开始第四问改进版分析...")
        
        # 1. 数据加载与预处理
        self.load_and_preprocess_data()
        
        # 2. 改进的GC含量与测序质量校正
        self.advanced_gc_quality_correction()
        
        # 3. 增强的X染色体背景参考
        self.enhanced_x_chromosome_reference()
        
        # 4. 智能Z值融合算法
        self.intelligent_z_value_fusion()
        
        # 5. 高级临床指标整合
        self.advanced_clinical_integration()
        
        # 6. 创建综合特征集
        self.create_comprehensive_features()
        
        # 7. 智能特征选择
        self.intelligent_feature_selection()
        
        # 8. 训练集成模型
        self.train_ensemble_models()
        
        # 9. 高级交叉验证评估
        self.advanced_cross_validation()
        
        # 10. 最终评估和阈值优化
        final_results = self.final_evaluation_and_threshold_optimization()
        
        print("\n第四问改进版分析完成！")
        print(f"所有结果已保存到: {self.output_path}")
        
        return final_results


def main():
    """主函数"""
    # 设置路径
    data_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/girl_output.csv"
    output_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/4"
    
    # 创建改进版模型实例
    model = ImprovedFeatureEngineeringModel(data_path, output_path)
    
    # 运行完整分析
    model.run_complete_improved_analysis()


if __name__ == "__main__":
    main()
