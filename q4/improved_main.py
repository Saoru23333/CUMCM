#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：女胎异常判定的鲁棒诊断模型（改进版）
解决过拟合问题，优化模型选择和阈值策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                           classification_report, confusion_matrix, f1_score,
                           precision_score, recall_score, accuracy_score)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedFemaleFetusDiagnosisModel:
    """改进版女胎异常判定模型"""
    
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.lr_model = None
        self.cv_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_and_preprocess_data(self):
        """数据加载与预处理"""
        print("正在加载数据...")
        self.data = pd.read_csv(self.data_path)
        
        # 数据基本信息
        print(f"数据形状: {self.data.shape}")
        
        # 创建标签：染色体的非整倍体 -> 0(正常) 1(异常)
        self.data['label'] = self.data['染色体的非整倍体'].fillna('').astype(str)
        self.data['label'] = (self.data['label'] != '').astype(int)
        
        # 选择特征
        feature_columns = [
            '年龄', '孕妇BMI',  # 孕妇生理指标
            '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',  # Z值
            'GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',  # GC含量
            '被过滤掉读段数的比例', '重复读段的比例',  # 测序质量
            '在参考基因组上比对的比例', '唯一比对的读段数'  # 其他测序指标
        ]
        
        # 检查特征是否存在
        available_features = [col for col in feature_columns if col in self.data.columns]
        print(f"可用特征: {available_features}")
        
        # 准备特征矩阵和标签
        self.X = self.data[available_features].copy()
        self.y = self.data['label'].copy()
        self.feature_names = available_features
        
        # 处理缺失值
        self.X = self.X.fillna(self.X.median())
        
        print(f"特征矩阵形状: {self.X.shape}")
        print(f"标签分布: {self.y.value_counts()}")
        print(f"异常样本比例: {self.y.mean():.3f}")
        
        return self.X, self.y
    
    def train_models_with_improved_regularization(self):
        """改进的模型训练，增强正则化"""
        print("\n开始训练改进版模型...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(self.X)
        
        # 计算类别权重
        class_weights = len(self.y[self.y == 0]) / len(self.y[self.y == 1])
        print(f"类别权重 (scale_pos_weight): {class_weights:.2f}")
        
        # 改进的XGBoost参数 - 增强正则化
        print("训练改进版XGBoost模型...")
        self.xgb_model = xgb.XGBClassifier(
            scale_pos_weight=class_weights,
            random_state=42,
            eval_metric='logloss',
            # 增强正则化参数
            reg_alpha=1.0,  # L1正则化
            reg_lambda=1.0, # L2正则化
            max_depth=3,    # 降低树深度
            learning_rate=0.1,  # 降低学习率
            subsample=0.8,  # 子采样
            colsample_bytree=0.8,  # 特征采样
            min_child_weight=3,  # 增加最小子节点权重
            gamma=0.1  # 增加最小分割损失
        )
        
        # 逻辑回归模型
        print("训练逻辑回归模型...")
        self.lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            C=0.1,  # 增强正则化
            penalty='l2'
        )
        
        # 改进的超参数网格 - 更保守的参数
        xgb_params = {
            'max_depth': [2, 3, 4],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [50, 100, 150],
            'reg_alpha': [0.5, 1.0, 2.0],
            'reg_lambda': [0.5, 1.0, 2.0],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        lr_params = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        # 分层K折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # XGBoost网格搜索
        print("XGBoost超参数调优...")
        xgb_grid = GridSearchCV(
            self.xgb_model, xgb_params, 
            cv=skf, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        xgb_grid.fit(X_scaled, self.y)
        self.xgb_model = xgb_grid.best_estimator_
        
        # 逻辑回归网格搜索
        print("逻辑回归超参数调优...")
        lr_grid = GridSearchCV(
            self.lr_model, lr_params,
            cv=skf, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        lr_grid.fit(X_scaled, self.y)
        self.lr_model = lr_grid.best_estimator_
        
        print(f"XGBoost最佳参数: {xgb_grid.best_params_}")
        print(f"逻辑回归最佳参数: {lr_grid.best_params_}")
        
        # 保存最佳参数
        best_params = {
            'xgb_best_params': xgb_grid.best_params_,
            'lr_best_params': lr_grid.best_params_,
            'xgb_best_score': xgb_grid.best_score_,
            'lr_best_score': lr_grid.best_score_
        }
        
        pd.DataFrame([best_params]).to_csv(f"{self.output_path}/improved_best_parameters.csv", index=False)
        
        return self.xgb_model, self.lr_model
    
    def improved_cross_validation_evaluation(self):
        """改进的交叉验证评估"""
        print("\n进行改进的交叉验证评估...")
        
        X_scaled = self.scaler.transform(self.X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        models = {
            'XGBoost': self.xgb_model,
            'Logistic Regression': self.lr_model
        }
        
        cv_results = {}
        
        for name, model in models.items():
            print(f"评估 {name}...")
            
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
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = model.predict(X_val)
                
                # 计算指标
                auc_scores.append(roc_auc_score(y_val, y_pred_proba))
                f1_scores.append(f1_score(y_val, y_pred))
                precision_scores.append(precision_score(y_val, y_pred))
                recall_scores.append(recall_score(y_val, y_pred))
                accuracy_scores.append(accuracy_score(y_val, y_pred))
            
            cv_results[name] = {
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
        cv_df = pd.DataFrame(cv_results).T
        cv_df.to_csv(f"{self.output_path}/improved_cross_validation_results.csv")
        
        print("改进的交叉验证结果:")
        print(cv_df)
        
        self.cv_results = cv_results
        return cv_results
    
    def improved_model_selection(self):
        """改进的模型选择策略"""
        print("\n进行改进的模型选择...")
        
        # 基于交叉验证结果选择最佳模型
        xgb_auc = self.cv_results['XGBoost']['AUC_mean']
        lr_auc = self.cv_results['Logistic Regression']['AUC_mean']
        
        print(f"XGBoost交叉验证AUC: {xgb_auc:.4f}")
        print(f"逻辑回归交叉验证AUC: {lr_auc:.4f}")
        
        # 选择交叉验证AUC更高的模型
        if lr_auc >= xgb_auc:
            self.best_model = self.lr_model
            self.best_model_name = 'Logistic Regression'
            print("选择逻辑回归作为最佳模型（基于交叉验证结果）")
        else:
            self.best_model = self.xgb_model
            self.best_model_name = 'XGBoost'
            print("选择XGBoost作为最佳模型（基于交叉验证结果）")
        
        return self.best_model
    
    def improved_final_evaluation(self):
        """改进的最终评估"""
        print("\n进行改进的最终评估...")
        
        # 使用最佳模型进行最终评估
        X_scaled = self.scaler.fit_transform(self.X)
        
        # 训练最佳模型
        self.best_model.fit(X_scaled, self.y)
        
        # 预测
        best_pred_proba = self.best_model.predict_proba(X_scaled)[:, 1]
        best_pred = (best_pred_proba >= 0.5).astype(int)
        
        # 计算评估指标
        auc = roc_auc_score(self.y, best_pred_proba)
        precision = precision_score(self.y, best_pred)
        recall = recall_score(self.y, best_pred)
        f1 = f1_score(self.y, best_pred)
        accuracy = accuracy_score(self.y, best_pred)
        
        print("=" * 60)
        print(f"改进版最终模型评估结果 ({self.best_model_name})")
        print("=" * 60)
        print(f"AUC: {auc:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"准确率: {accuracy:.4f}")
        print("=" * 60)
        
        # 保存最终评估结果
        final_results = {
            'Model': self.best_model_name,
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Accuracy': accuracy
        }
        
        pd.DataFrame([final_results]).to_csv(f"{self.output_path}/improved_final_evaluation.csv", index=False)
        
        return best_pred_proba
    
    def improved_threshold_optimization(self, pred_proba):
        """改进的阈值优化"""
        print("\n进行改进的阈值优化...")
        
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
        results_df.to_csv(f"{self.output_path}/improved_threshold_analysis.csv", index=False)
        
        # 改进的阈值选择策略
        # 1. 寻找F1分数最高的阈值
        best_f1_idx = results_df['f1_score'].idxmax()
        optimal_threshold = results_df.loc[best_f1_idx, 'threshold']
        
        # 2. 基于临床需求调整阈值
        # 低风险阈值：90%灵敏度
        high_sensitivity_idx = results_df[results_df['sensitivity'] >= 0.90].index
        if len(high_sensitivity_idx) > 0:
            dl = results_df.loc[high_sensitivity_idx[0], 'threshold']
        else:
            dl = 0.1
        
        # 高风险阈值：80%精确率
        high_precision_idx = results_df[results_df['precision'] >= 0.80].index
        if len(high_precision_idx) > 0:
            dh = results_df.loc[high_precision_idx[-1], 'threshold']
        else:
            dh = 0.8
        
        print(f"最优F1阈值: {optimal_threshold:.3f}")
        print(f"低风险阈值 (DL): {dl:.3f}")
        print(f"高风险阈值 (DH): {dh:.3f}")
        print(f"不确定区间: {dh-dl:.3f} ({(dh-dl)*100:.1f}%)")
        
        # 绘制改进的阈值分析图
        self.plot_improved_threshold_analysis(results_df, dl, dh, optimal_threshold)
        
        # 保存改进的决策规则
        decision_rules = {
            'optimal_f1_threshold': optimal_threshold,
            'low_risk_threshold': dl,
            'high_risk_threshold': dh,
            'uncertain_interval': dh - dl,
            'decision_rules': {
                'high_risk': f'D >= {dh:.3f} -> 高风险异常',
                'low_risk': f'D < {dl:.3f} -> 低风险正常',
                'uncertain': f'{dl:.3f} <= D < {dh:.3f} -> 结果不确定，建议复检'
            }
        }
        
        pd.DataFrame([decision_rules]).to_csv(f"{self.output_path}/improved_decision_rules.csv", index=False)
        
        return dl, dh, optimal_threshold
    
    def plot_improved_threshold_analysis(self, results_df, dl, dh, optimal_threshold):
        """绘制改进的阈值分析图"""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(results_df['threshold'], results_df['sensitivity'], label='灵敏度', linewidth=2)
        plt.axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='90%目标')
        plt.axvline(x=dl, color='g', linestyle='--', alpha=0.5, label=f'DL={dl:.3f}')
        plt.xlabel('阈值')
        plt.ylabel('灵敏度')
        plt.title('灵敏度 vs 阈值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(results_df['threshold'], results_df['precision'], label='精确率', linewidth=2)
        plt.axhline(y=0.80, color='r', linestyle='--', alpha=0.5, label='80%目标')
        plt.axvline(x=dh, color='g', linestyle='--', alpha=0.5, label=f'DH={dh:.3f}')
        plt.xlabel('阈值')
        plt.ylabel('精确率')
        plt.title('精确率 vs 阈值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(results_df['threshold'], results_df['f1_score'], label='F1分数', linewidth=2)
        plt.axvline(x=optimal_threshold, color='orange', linestyle='--', alpha=0.7, label=f'最优={optimal_threshold:.3f}')
        plt.xlabel('阈值')
        plt.ylabel('F1分数')
        plt.title('F1分数 vs 阈值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.plot(results_df['sensitivity'], results_df['precision'], linewidth=2)
        plt.xlabel('灵敏度')
        plt.ylabel('精确率')
        plt.title('精确率 vs 灵敏度')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.plot(results_df['threshold'], results_df['specificity'], label='特异性', linewidth=2)
        plt.xlabel('阈值')
        plt.ylabel('特异性')
        plt.title('特异性 vs 阈值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        # 显示阈值区间
        plt.bar(['低风险', '不确定', '高风险'], 
                [dl, dh-dl, 1-dh], 
                color=['green', 'orange', 'red'], alpha=0.7)
        plt.ylabel('阈值区间')
        plt.title('决策阈值分布')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/improved_threshold_optimization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_improved_report(self):
        """生成改进版报告"""
        print("\n生成改进版报告...")
        
        report = f"""
# 第四问：女胎异常判定模型报告（改进版）

## 模型概述
- 最佳模型: {self.best_model_name}
- 数据样本数: {len(self.y)}
- 异常样本比例: {self.y.mean():.3f}
- 特征数量: {len(self.feature_names)}

## 改进措施
1. **增强正则化**：XGBoost增加了L1/L2正则化、子采样等参数
2. **改进模型选择**：基于交叉验证结果而非最终评估结果选择模型
3. **优化阈值策略**：缩小不确定区间，提高临床实用性
4. **全面评估指标**：增加了准确率等更多评估指标

## 交叉验证结果

### XGBoost
- AUC: {self.cv_results['XGBoost']['AUC_mean']:.4f} ± {self.cv_results['XGBoost']['AUC_std']:.4f}
- F1分数: {self.cv_results['XGBoost']['F1_mean']:.4f} ± {self.cv_results['XGBoost']['F1_std']:.4f}
- 精确率: {self.cv_results['XGBoost']['Precision_mean']:.4f} ± {self.cv_results['XGBoost']['Precision_std']:.4f}
- 召回率: {self.cv_results['XGBoost']['Recall_mean']:.4f} ± {self.cv_results['XGBoost']['Recall_std']:.4f}
- 准确率: {self.cv_results['XGBoost']['Accuracy_mean']:.4f} ± {self.cv_results['XGBoost']['Accuracy_std']:.4f}

### Logistic Regression
- AUC: {self.cv_results['Logistic Regression']['AUC_mean']:.4f} ± {self.cv_results['Logistic Regression']['AUC_std']:.4f}
- F1分数: {self.cv_results['Logistic Regression']['F1_mean']:.4f} ± {self.cv_results['Logistic Regression']['F1_std']:.4f}
- 精确率: {self.cv_results['Logistic Regression']['Precision_mean']:.4f} ± {self.cv_results['Logistic Regression']['Precision_std']:.4f}
- 召回率: {self.cv_results['Logistic Regression']['Recall_mean']:.4f} ± {self.cv_results['Logistic Regression']['Recall_std']:.4f}
- 准确率: {self.cv_results['Logistic Regression']['Accuracy_mean']:.4f} ± {self.cv_results['Logistic Regression']['Accuracy_std']:.4f}

## 特征重要性 (XGBoost)
"""
        
        if hasattr(self.xgb_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(f"{self.output_path}/improved_feature_importance.csv", index=False)
            
            report += "\n".join([f"- {row['feature']}: {row['importance']:.4f}" 
                               for _, row in feature_importance.head(10).iterrows()])
        
        # 保存报告
        with open(f"{self.output_path}/improved_model_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("改进版报告已生成")
    
    def run_improved_analysis(self):
        """运行改进版完整分析流程"""
        print("开始第四问改进版分析...")
        
        # 1. 数据加载与预处理
        self.load_and_preprocess_data()
        
        # 2. 改进的模型训练
        self.train_models_with_improved_regularization()
        
        # 3. 改进的交叉验证评估
        self.improved_cross_validation_evaluation()
        
        # 4. 改进的模型选择
        self.improved_model_selection()
        
        # 5. 改进的最终评估
        pred_proba = self.improved_final_evaluation()
        
        # 6. 改进的阈值优化
        self.improved_threshold_optimization(pred_proba)
        
        # 7. 生成改进版报告
        self.generate_improved_report()
        
        print("\n第四问改进版分析完成！")
        print(f"所有结果已保存到: {self.output_path}")


def main():
    """主函数"""
    # 设置路径
    data_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/girl_output.csv"
    output_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/4"
    
    # 创建改进版模型实例
    model = ImprovedFemaleFetusDiagnosisModel(data_path, output_path)
    
    # 运行改进版分析
    model.run_improved_analysis()


if __name__ == "__main__":
    main()
