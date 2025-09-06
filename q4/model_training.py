#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：女胎异常判定 - 模型训练
根据Instruction.md的思路，构建逻辑回归模型并学习权重
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FemaleFetusModelTrainer:
    """女胎异常判定模型训练器"""
    
    def __init__(self, features, target):
        """
        初始化模型训练器
        
        Args:
            features: 处理后的特征数据 (DataFrame)
            target: 标签数据 (Series)
        """
        self.features = features.copy()
        self.target = target.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.model_weights = None
        self.training_results = {}
        
    def prepare_training_data(self, test_size=0.2, random_state=42):
        """
        准备训练数据
        根据Instruction.md，将数据划分为训练集和测试集
        """
        print("=== 准备训练数据 ===")
        
        # 检查数据质量
        print(f"特征矩阵形状：{self.features.shape}")
        print(f"标签分布：正常={sum(self.target==0)}，异常={sum(self.target==1)}")
        
        # 处理缺失值
        self.features = self.features.fillna(self.features.median())
        
        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.target  # 保持类别比例
        )
        
        # 保存特征名称
        self.feature_names = list(self.features.columns)
        
        print(f"训练集大小：{self.X_train.shape}")
        print(f"测试集大小：{self.X_test.shape}")
        print(f"训练集标签分布：正常={sum(self.y_train==0)}，异常={sum(self.y_train==1)}")
        print(f"测试集标签分布：正常={sum(self.y_test==0)}，异常={sum(self.y_test==1)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self, class_weight='balanced', random_state=42):
        """
        步骤3.3: 训练逻辑回归模型
        根据Instruction.md，使用逻辑回归学习最优权重
        """
        print("\n=== 步骤3.3: 训练逻辑回归模型 ===")
        
        # 创建逻辑回归模型
        self.model = LogisticRegression(
            class_weight=class_weight,  # 处理类别不平衡
            random_state=random_state,
            max_iter=1000,
            solver='liblinear'  # 适合小数据集
        )
        
        # 训练模型
        print("正在训练模型...")
        self.model.fit(self.X_train, self.y_train)
        
        # 获取模型权重
        self.model_weights = pd.DataFrame({
            'feature': self.feature_names,
            'weight': self.model.coef_[0],
            'abs_weight': np.abs(self.model.coef_[0])
        }).sort_values('abs_weight', ascending=False)
        
        print("模型训练完成")
        print(f"模型权重（前10个最重要的特征）：")
        print(self.model_weights.head(10))
        
        return self.model
    
    def cross_validate_model(self, cv_folds=5):
        """交叉验证评估模型"""
        print("\n=== 交叉验证评估 ===")
        
        # 使用分层K折交叉验证
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # 计算交叉验证分数
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, 
            cv=cv, scoring='roc_auc'
        )
        
        print(f"交叉验证AUC分数：{cv_scores}")
        print(f"平均AUC：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.training_results['cv_scores'] = cv_scores
        self.training_results['cv_mean'] = cv_scores.mean()
        self.training_results['cv_std'] = cv_scores.std()
        
        return cv_scores
    
    def evaluate_model_performance(self):
        """评估模型性能"""
        print("\n=== 模型性能评估 ===")
        
        # 预测
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # 计算AUC
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        # 分类报告
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"AUC分数：{auc_score:.4f}")
        print("\n分类报告：")
        print(classification_report(self.y_test, y_pred))
        
        print("\n混淆矩阵：")
        print(cm)
        
        # 保存结果
        self.training_results.update({
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })
        
        return auc_score, class_report, cm
    
    def visualize_model_performance(self):
        """可视化模型性能"""
        print("\n=== 可视化模型性能 ===")
        
        if 'y_pred_proba' not in self.training_results:
            print("请先运行模型评估")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能可视化', fontsize=16, fontweight='bold')
        
        # 1. ROC曲线
        fpr, tpr, _ = roc_curve(self.y_test, self.training_results['y_pred_proba'])
        auc_score = self.training_results['auc_score']
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC曲线 (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('假正率 (FPR)')
        axes[0, 0].set_ylabel('真正率 (TPR)')
        axes[0, 0].set_title('ROC曲线')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 混淆矩阵热力图
        cm = self.training_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('混淆矩阵')
        axes[0, 1].set_xlabel('预测标签')
        axes[0, 1].set_ylabel('真实标签')
        
        # 3. 特征重要性
        top_features = self.model_weights.head(10)
        axes[1, 0].barh(range(len(top_features)), top_features['weight'])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'])
        axes[1, 0].set_xlabel('权重')
        axes[1, 0].set_title('特征重要性（前10个）')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 预测概率分布
        normal_proba = self.training_results['y_pred_proba'][self.y_test == 0]
        abnormal_proba = self.training_results['y_pred_proba'][self.y_test == 1]
        
        axes[1, 1].hist(normal_proba, bins=20, alpha=0.7, label='正常', color='lightblue')
        axes[1, 1].hist(abnormal_proba, bins=20, alpha=0.7, label='异常', color='lightcoral')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='阈值=0.5')
        axes[1, 1].set_xlabel('预测概率')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('预测概率分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/model_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n=== 特征重要性分析 ===")
        
        # 创建特征重要性可视化
        plt.figure(figsize=(12, 8))
        
        # 选择前15个最重要的特征
        top_features = self.model_weights.head(15)
        
        # 创建水平条形图
        colors = ['red' if w < 0 else 'blue' for w in top_features['weight']]
        bars = plt.barh(range(len(top_features)), top_features['weight'], color=colors, alpha=0.7)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('权重')
        plt.title('特征重要性分析（前15个特征）', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.text(0.02, len(top_features)-1, '正权重（增加异常风险）', 
                color='blue', fontweight='bold')
        plt.text(-0.02, len(top_features)-1, '负权重（降低异常风险）', 
                color='red', fontweight='bold', ha='right')
        
        plt.tight_layout()
        plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存特征重要性数据
        self.model_weights.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/feature_importance.csv', 
                                 index=False)
        print("特征重要性数据已保存到 feature_importance.csv")
    
    def save_model(self):
        """保存训练好的模型"""
        if self.model is not None:
            # 保存模型
            joblib.dump(self.model, '/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/trained_model.pkl')
            
            # 保存特征名称
            pd.Series(self.feature_names).to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/feature_names.csv', 
                                                header=['feature_name'])
            
            # 保存训练结果
            results_df = pd.DataFrame({
                'metric': ['AUC', 'CV_Mean', 'CV_Std'],
                'value': [
                    self.training_results.get('auc_score', 0),
                    self.training_results.get('cv_mean', 0),
                    self.training_results.get('cv_std', 0)
                ]
            })
            results_df.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/training_results.csv', 
                             index=False)
            
            print("模型和训练结果已保存")
            print("模型文件：trained_model.pkl")
            print("特征名称：feature_names.csv")
            print("训练结果：training_results.csv")
    
    def run_training(self):
        """运行完整的模型训练流程"""
        print("=== 女胎异常判定模型训练开始 ===")
        
        # 1. 准备训练数据
        self.prepare_training_data()
        
        # 2. 训练逻辑回归模型
        self.train_logistic_regression()
        
        # 3. 交叉验证
        self.cross_validate_model()
        
        # 4. 评估模型性能
        self.evaluate_model_performance()
        
        # 5. 可视化模型性能
        self.visualize_model_performance()
        
        # 6. 分析特征重要性
        self.analyze_feature_importance()
        
        # 7. 保存模型
        self.save_model()
        
        print("\n=== 模型训练完成 ===")
        
        return self.model, self.training_results

def main():
    """主函数"""
    print("请先运行 feature_engineering.py 获得处理后的特征数据")
    return None, None

if __name__ == "__main__":
    main()
