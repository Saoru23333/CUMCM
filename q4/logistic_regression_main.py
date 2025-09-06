#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：女胎异常判定的鲁棒诊断模型（逻辑回归专用版）
专注于逻辑回归模型，避免过拟合问题
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

class LogisticRegressionDiagnosisModel:
    """逻辑回归专用女胎异常判定模型"""
    
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
    
    def exploratory_data_analysis(self):
        """探索性数据分析"""
        print("\n进行探索性数据分析...")
        
        # 基础统计
        stats = self.X.describe()
        stats.to_csv(f"{self.output_path}/lr_feature_statistics.csv")
        
        # 相关性分析
        corr_matrix = self.X.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/lr_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Z值分布对比
        z_features = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
        available_z_features = [f for f in z_features if f in self.X.columns]
        
        if available_z_features:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(available_z_features[:4]):
                if i < len(axes):
                    # 正常样本
                    normal_data = self.X[self.y == 0][feature]
                    # 异常样本
                    abnormal_data = self.X[self.y == 1][feature]
                    
                    axes[i].hist(normal_data, alpha=0.7, label='正常', bins=30, color='blue')
                    axes[i].hist(abnormal_data, alpha=0.7, label='异常', bins=30, color='red')
                    axes[i].set_title(f'{feature}分布对比')
                    axes[i].set_xlabel('Z值')
                    axes[i].set_ylabel('频数')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_path}/lr_z_value_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print("探索性数据分析完成")
    
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
        
        pd.DataFrame([best_params]).to_csv(f"{self.output_path}/lr_best_parameters.csv", index=False)
        
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
        cv_df = pd.DataFrame([cv_results], index=['Logistic Regression'])
        cv_df.to_csv(f"{self.output_path}/lr_cross_validation_results.csv")
        
        print("交叉验证结果:")
        print(cv_df)
        
        self.cv_results = cv_results
        return cv_results
    
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
        print("逻辑回归最终模型评估结果 (阈值=0.5)")
        print("=" * 60)
        print(f"AUC: {auc:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"准确率: {accuracy:.4f}")
        print("=" * 60)
        
        # 保存最终评估结果
        final_results = {
            'Model': 'Logistic Regression',
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Accuracy': accuracy
        }
        
        pd.DataFrame([final_results]).to_csv(f"{self.output_path}/lr_final_evaluation.csv", index=False)
        
        # 绘制ROC曲线
        self.plot_roc_curve(lr_pred_proba)
        
        # 绘制PR曲线
        self.plot_pr_curve(lr_pred_proba)
        
        return lr_pred_proba
    
    def plot_roc_curve(self, pred_proba):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))
        
        # 逻辑回归 ROC
        fpr, tpr, _ = roc_curve(self.y, pred_proba)
        auc = roc_auc_score(self.y, pred_proba)
        plt.plot(fpr, tpr, label=f'逻辑回归 (AUC = {auc:.3f})', linewidth=2, color='blue')
        
        # 对角线
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器', alpha=0.5)
        
        plt.xlabel('假正率 (1 - 特异性)')
        plt.ylabel('真正率 (灵敏度)')
        plt.title('逻辑回归 ROC曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/lr_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pr_curve(self, pred_proba):
        """绘制PR曲线"""
        plt.figure(figsize=(10, 8))
        
        # 逻辑回归 PR
        precision, recall, _ = precision_recall_curve(self.y, pred_proba)
        plt.plot(recall, precision, label='逻辑回归', linewidth=2, color='blue')
        
        # 基线
        baseline = self.y.mean()
        plt.axhline(y=baseline, color='k', linestyle='--', label=f'基线 ({baseline:.3f})', alpha=0.5)
        
        plt.xlabel('召回率 (灵敏度)')
        plt.ylabel('精确率')
        plt.title('逻辑回归 精确率-召回率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/lr_pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def optimize_thresholds(self, pred_proba):
        """优化阈值"""
        print("\n优化决策阈值...")
        
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
        results_df.to_csv(f"{self.output_path}/lr_threshold_analysis.csv", index=False)
        
        # 寻找最优阈值
        # 1. F1分数最高的阈值
        best_f1_idx = results_df['f1_score'].idxmax()
        optimal_f1_threshold = results_df.loc[best_f1_idx, 'threshold']
        
        # 2. 基于临床需求的阈值
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
        
        print(f"最优F1阈值: {optimal_f1_threshold:.3f}")
        print(f"低风险阈值 (DL): {dl:.3f}")
        print(f"高风险阈值 (DH): {dh:.3f}")
        print(f"不确定区间: {dh-dl:.3f} ({(dh-dl)*100:.1f}%)")
        
        # 绘制阈值分析图
        self.plot_threshold_analysis(results_df, dl, dh, optimal_f1_threshold)
        
        # 保存阈值决策规则
        decision_rules = {
            'optimal_f1_threshold': optimal_f1_threshold,
            'low_risk_threshold': dl,
            'high_risk_threshold': dh,
            'uncertain_interval': dh - dl,
            'decision_rules': {
                'high_risk': f'D >= {dh:.3f} -> 高风险异常',
                'low_risk': f'D < {dl:.3f} -> 低风险正常',
                'uncertain': f'{dl:.3f} <= D < {dh:.3f} -> 结果不确定，建议复检'
            }
        }
        
        pd.DataFrame([decision_rules]).to_csv(f"{self.output_path}/lr_decision_rules.csv", index=False)
        
        return dl, dh, optimal_f1_threshold
    
    def plot_threshold_analysis(self, results_df, dl, dh, optimal_f1_threshold):
        """绘制阈值分析图"""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(results_df['threshold'], results_df['sensitivity'], label='灵敏度', linewidth=2)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%目标')
        plt.axvline(x=dl, color='g', linestyle='--', alpha=0.5, label=f'DL={dl:.3f}')
        plt.xlabel('阈值')
        plt.ylabel('灵敏度')
        plt.title('灵敏度 vs 阈值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(results_df['threshold'], results_df['precision'], label='精确率', linewidth=2)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%目标')
        plt.axvline(x=dh, color='g', linestyle='--', alpha=0.5, label=f'DH={dh:.3f}')
        plt.xlabel('阈值')
        plt.ylabel('精确率')
        plt.title('精确率 vs 阈值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(results_df['threshold'], results_df['f1_score'], label='F1分数', linewidth=2)
        plt.axvline(x=optimal_f1_threshold, color='orange', linestyle='--', alpha=0.7, label=f'最优={optimal_f1_threshold:.3f}')
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
        plt.savefig(f"{self.output_path}/lr_threshold_optimization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
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
        feature_importance.to_csv(f"{self.output_path}/lr_feature_importance.csv", index=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(10)
        
        colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
        plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('回归系数')
        plt.title('逻辑回归特征重要性（回归系数）')
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.legend(['负系数（降低风险）', '正系数（增加风险）'], loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/lr_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("特征重要性分析完成")
        return feature_importance
    
    def generate_final_report(self):
        """生成最终报告"""
        print("\n生成最终报告...")
        
        report = f"""
# 第四问：女胎异常判定模型报告（逻辑回归专用版）

## 模型概述
- 模型类型: 逻辑回归
- 数据样本数: {len(self.y)}
- 异常样本比例: {self.y.mean():.3f}
- 特征数量: {len(self.feature_names)}

## 模型优势
1. **避免过拟合**：逻辑回归模型简单稳定，泛化能力强
2. **可解释性强**：回归系数具有明确的生物学意义
3. **计算效率高**：训练和预测速度快
4. **类别不平衡处理**：使用class_weight='balanced'自动处理

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
            feature_importance = pd.read_csv(f"{self.output_path}/lr_feature_importance.csv")
            report += "\n".join([f"- {row['feature']}: {row['coefficient']:.4f}" 
                               for _, row in feature_importance.head(10).iterrows()])
        except:
            report += "特征重要性分析未完成"
        
        # 保存报告
        with open(f"{self.output_path}/lr_model_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("最终报告已生成")
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("开始第四问逻辑回归专用分析...")
        
        # 1. 数据加载与预处理
        self.load_and_preprocess_data()
        
        # 2. 探索性数据分析
        self.exploratory_data_analysis()
        
        # 3. 训练逻辑回归模型
        self.train_logistic_regression_model()
        
        # 4. 交叉验证评估
        self.cross_validation_evaluation()
        
        # 5. 最终模型评估
        pred_proba = self.final_model_evaluation()
        
        # 6. 优化阈值
        self.optimize_thresholds(pred_proba)
        
        # 7. 分析特征重要性
        self.analyze_feature_importance()
        
        # 8. 生成最终报告
        self.generate_final_report()
        
        print("\n第四问逻辑回归专用分析完成！")
        print(f"所有结果已保存到: {self.output_path}")


def main():
    """主函数"""
    # 设置路径
    data_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/girl_output.csv"
    output_path = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/4"
    
    # 创建逻辑回归模型实例
    model = LogisticRegressionDiagnosisModel(data_path, output_path)
    
    # 运行完整分析
    model.run_complete_analysis()


if __name__ == "__main__":
    main()
