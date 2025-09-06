#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：女胎异常判定 - 模型评估
根据Instruction.md的思路，全面评估模型性能并生成报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, log_loss
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FemaleFetusModelEvaluator:
    """女胎异常判定模型评估器"""
    
    def __init__(self, model, X_test, y_test, y_pred_proba, thresholds=None):
        """
        初始化模型评估器
        
        Args:
            model: 训练好的模型
            X_test: 测试集特征
            y_test: 测试集标签
            y_pred_proba: 预测概率
            thresholds: 优化后的阈值
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred_proba = y_pred_proba
        self.thresholds = thresholds
        self.evaluation_results = {}
        
    def comprehensive_performance_evaluation(self):
        """
        全面性能评估
        根据Instruction.md，评估灵敏度、特异性、精确率、F1-Score、ROC-AUC等指标
        """
        print("=== 全面性能评估 ===")
        
        # 1. 传统二分类评估（阈值=0.5）
        y_pred_binary = (self.y_pred_proba >= 0.5).astype(int)
        
        # 计算混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        # 计算基本指标
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 灵敏度/召回率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # 精确率
        f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # 计算其他指标
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        mcc = matthews_corrcoef(self.y_test, y_pred_binary)
        kappa = cohen_kappa_score(self.y_test, y_pred_binary)
        logloss = log_loss(self.y_test, self.y_pred_proba)
        
        # ROC-AUC
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        
        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        
        print("传统二分类评估结果（阈值=0.5）：")
        print(f"  准确率 (Accuracy): {accuracy:.4f}")
        print(f"  灵敏度 (Sensitivity/Recall): {sensitivity:.4f}")
        print(f"  特异性 (Specificity): {specificity:.4f}")
        print(f"  精确率 (Precision): {precision:.4f}")
        print(f"  F1分数 (F1-Score): {f1_score:.4f}")
        print(f"  ROC-AUC: {auc_score:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  Matthews相关系数: {mcc:.4f}")
        print(f"  Cohen's Kappa: {kappa:.4f}")
        print(f"  对数损失: {logloss:.4f}")
        
        # 2. 三区决策评估（如果提供了阈值）
        if self.thresholds is not None:
            print("\n三区决策评估结果：")
            self.evaluate_three_zone_performance()
        
        # 保存评估结果
        self.evaluation_results = {
            'binary_classification': {
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1_score,
                'auc_score': auc_score,
                'pr_auc': pr_auc,
                'mcc': mcc,
                'kappa': kappa,
                'logloss': logloss,
                'confusion_matrix': cm
            }
        }
        
        return self.evaluation_results
    
    def evaluate_three_zone_performance(self):
        """评估三区决策性能"""
        if self.thresholds is None:
            print("未提供阈值，跳过三区决策评估")
            return
        
        # 应用三区决策规则
        decisions = []
        for score in self.y_pred_proba:
            if score >= self.thresholds['D_H']:
                decisions.append('高风险异常')
            elif score < self.thresholds['D_L']:
                decisions.append('低风险正常')
            else:
                decisions.append('结果不确定')
        
        decisions = np.array(decisions)
        
        # 计算三区决策性能
        high_risk_mask = decisions == '高风险异常'
        low_risk_mask = decisions == '低风险正常'
        uncertain_mask = decisions == '结果不确定'
        
        # 高风险决策性能
        if sum(high_risk_mask) > 0:
            high_risk_precision = sum(self.y_test[high_risk_mask] == 1) / sum(high_risk_mask)
            high_risk_recall = sum((self.y_test == 1) & high_risk_mask) / sum(self.y_test == 1)
            print(f"  高风险决策精确率: {high_risk_precision:.4f}")
            print(f"  高风险决策召回率: {high_risk_recall:.4f}")
        
        # 低风险决策性能
        if sum(low_risk_mask) > 0:
            low_risk_precision = sum(self.y_test[low_risk_mask] == 0) / sum(low_risk_mask)
            low_risk_recall = sum((self.y_test == 0) & low_risk_mask) / sum(self.y_test == 0)
            print(f"  低风险决策精确率: {low_risk_precision:.4f}")
            print(f"  低风险决策召回率: {low_risk_recall:.4f}")
        
        # 不确定区域分析
        if sum(uncertain_mask) > 0:
            uncertain_abnormal_rate = sum(self.y_test[uncertain_mask] == 1) / sum(uncertain_mask)
            uncertain_coverage = sum(uncertain_mask) / len(uncertain_mask)
            print(f"  不确定区域异常率: {uncertain_abnormal_rate:.4f}")
            print(f"  不确定区域覆盖率: {uncertain_coverage:.4f}")
        
        # 保存三区决策结果
        self.evaluation_results['three_zone'] = {
            'decisions': decisions,
            'high_risk_precision': high_risk_precision if sum(high_risk_mask) > 0 else 0,
            'high_risk_recall': high_risk_recall if sum(high_risk_mask) > 0 else 0,
            'low_risk_precision': low_risk_precision if sum(low_risk_mask) > 0 else 0,
            'low_risk_recall': low_risk_recall if sum(low_risk_mask) > 0 else 0,
            'uncertain_abnormal_rate': uncertain_abnormal_rate if sum(uncertain_mask) > 0 else 0,
            'uncertain_coverage': uncertain_coverage if sum(uncertain_mask) > 0 else 0
        }
    
    def clinical_significance_analysis(self):
        """临床意义分析"""
        print("\n=== 临床意义分析 ===")
        
        # 分析不同阈值下的临床性能
        thresholds_to_analyze = [0.3, 0.4, 0.5, 0.6, 0.7]
        clinical_metrics = []
        
        for threshold in thresholds_to_analyze:
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            
            # 计算混淆矩阵
            cm = confusion_matrix(self.y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                # 计算临床相关指标
                false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                clinical_metrics.append({
                    'threshold': threshold,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'false_positive_rate': false_positive_rate,
                    'false_negative_rate': false_negative_rate
                })
        
        clinical_df = pd.DataFrame(clinical_metrics)
        
        print("不同阈值下的临床性能：")
        print(clinical_df.round(4))
        
        # 可视化临床性能
        plt.figure(figsize=(15, 10))
        
        # 子图1：灵敏度vs特异性
        plt.subplot(2, 3, 1)
        plt.plot(clinical_df['threshold'], clinical_df['sensitivity'], 'b-o', label='灵敏度')
        plt.plot(clinical_df['threshold'], clinical_df['specificity'], 'r-o', label='特异性')
        plt.xlabel('阈值')
        plt.ylabel('性能指标')
        plt.title('灵敏度与特异性')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：精确率
        plt.subplot(2, 3, 2)
        plt.plot(clinical_df['threshold'], clinical_df['precision'], 'g-o', label='精确率')
        plt.xlabel('阈值')
        plt.ylabel('精确率')
        plt.title('精确率变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图3：假阳性率vs假阴性率
        plt.subplot(2, 3, 3)
        plt.plot(clinical_df['threshold'], clinical_df['false_positive_rate'], 'orange', marker='o', label='假阳性率')
        plt.plot(clinical_df['threshold'], clinical_df['false_negative_rate'], 'purple', marker='o', label='假阴性率')
        plt.xlabel('阈值')
        plt.ylabel('错误率')
        plt.title('假阳性率与假阴性率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图4：ROC曲线
        plt.subplot(2, 3, 4)
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        plt.plot(fpr, tpr, 'b-', label=f'ROC曲线 (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('ROC曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图5：精确率-召回率曲线
        plt.subplot(2, 3, 5)
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        plt.plot(recall_curve, precision_curve, 'g-', label=f'PR曲线 (AUC = {pr_auc:.3f})')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图6：预测概率分布
        plt.subplot(2, 3, 6)
        normal_proba = self.y_pred_proba[self.y_test == 0]
        abnormal_proba = self.y_pred_proba[self.y_test == 1]
        plt.hist(normal_proba, bins=20, alpha=0.7, label='正常', color='lightblue', density=True)
        plt.hist(abnormal_proba, bins=20, alpha=0.7, label='异常', color='lightcoral', density=True)
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='阈值=0.5')
        plt.xlabel('预测概率')
        plt.ylabel('密度')
        plt.title('预测概率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/clinical_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存临床分析结果
        clinical_df.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/clinical_analysis.csv', 
                          index=False)
        
        self.evaluation_results['clinical_analysis'] = clinical_df
        
        return clinical_df
    
    def generate_comprehensive_report(self):
        """生成综合评估报告"""
        print("\n=== 生成综合评估报告 ===")
        
        report = []
        report.append("=" * 80)
        report.append("女胎异常判定模型 - 综合评估报告")
        report.append("=" * 80)
        report.append("")
        
        # 1. 模型基本信息
        report.append("1. 模型基本信息")
        report.append("-" * 40)
        report.append(f"模型类型: 逻辑回归 (Logistic Regression)")
        report.append(f"测试集样本数: {len(self.y_test)}")
        report.append(f"特征数量: {self.X_test.shape[1]}")
        report.append(f"正常样本数: {sum(self.y_test == 0)} ({sum(self.y_test == 0)/len(self.y_test)*100:.1f}%)")
        report.append(f"异常样本数: {sum(self.y_test == 1)} ({sum(self.y_test == 1)/len(self.y_test)*100:.1f}%)")
        report.append("")
        
        # 2. 传统二分类性能
        if 'binary_classification' in self.evaluation_results:
            bc_results = self.evaluation_results['binary_classification']
            report.append("2. 传统二分类性能评估 (阈值=0.5)")
            report.append("-" * 40)
            report.append(f"准确率 (Accuracy): {bc_results['accuracy']:.4f}")
            report.append(f"灵敏度 (Sensitivity/Recall): {bc_results['sensitivity']:.4f}")
            report.append(f"特异性 (Specificity): {bc_results['specificity']:.4f}")
            report.append(f"精确率 (Precision): {bc_results['precision']:.4f}")
            report.append(f"F1分数 (F1-Score): {bc_results['f1_score']:.4f}")
            report.append(f"ROC-AUC: {bc_results['auc_score']:.4f}")
            report.append(f"PR-AUC: {bc_results['pr_auc']:.4f}")
            report.append(f"Matthews相关系数: {bc_results['mcc']:.4f}")
            report.append(f"Cohen's Kappa: {bc_results['kappa']:.4f}")
            report.append(f"对数损失: {bc_results['logloss']:.4f}")
            report.append("")
        
        # 3. 三区决策性能
        if 'three_zone' in self.evaluation_results:
            tz_results = self.evaluation_results['three_zone']
            report.append("3. 三区动态阈值决策性能")
            report.append("-" * 40)
            if self.thresholds:
                report.append(f"低风险阈值 (D_L): {self.thresholds['D_L']:.4f}")
                report.append(f"高风险阈值 (D_H): {self.thresholds['D_H']:.4f}")
            report.append(f"高风险决策精确率: {tz_results['high_risk_precision']:.4f}")
            report.append(f"高风险决策召回率: {tz_results['high_risk_recall']:.4f}")
            report.append(f"低风险决策精确率: {tz_results['low_risk_precision']:.4f}")
            report.append(f"低风险决策召回率: {tz_results['low_risk_recall']:.4f}")
            report.append(f"不确定区域异常率: {tz_results['uncertain_abnormal_rate']:.4f}")
            report.append(f"不确定区域覆盖率: {tz_results['uncertain_coverage']:.4f}")
            report.append("")
        
        # 4. 临床意义分析
        if 'clinical_analysis' in self.evaluation_results:
            report.append("4. 临床意义分析")
            report.append("-" * 40)
            report.append("不同阈值下的关键性能指标:")
            clinical_df = self.evaluation_results['clinical_analysis']
            for _, row in clinical_df.iterrows():
                report.append(f"  阈值={row['threshold']:.1f}: 灵敏度={row['sensitivity']:.3f}, "
                            f"特异性={row['specificity']:.3f}, 精确率={row['precision']:.3f}")
            report.append("")
        
        # 5. 模型优势与局限性
        report.append("5. 模型优势与局限性")
        report.append("-" * 40)
        report.append("优势:")
        report.append("  - 采用多因素融合方法，综合考虑Z值、GC含量、测序质量等因素")
        report.append("  - 实现了质量校正，有效降低了测序噪音的影响")
        report.append("  - 采用三区动态阈值，提高了决策的稳健性")
        report.append("  - 模型具有良好的可解释性，权重系数具有明确的临床意义")
        report.append("")
        report.append("局限性:")
        report.append("  - 模型性能依赖于训练数据的质量和代表性")
        report.append("  - 对于极少数样本的预测可能存在不确定性")
        report.append("  - 需要定期重新训练以适应新的数据分布")
        report.append("")
        
        # 6. 建议与展望
        report.append("6. 建议与展望")
        report.append("-" * 40)
        report.append("建议:")
        report.append("  - 建议在临床应用中结合其他检测方法进行综合判断")
        report.append("  - 对于不确定区域的样本，建议进行复检或结合其他临床信息")
        report.append("  - 定期评估模型性能，必要时进行模型更新")
        report.append("")
        report.append("展望:")
        report.append("  - 可以进一步集成更多维度的数据，如基因型信息等")
        report.append("  - 考虑引入深度学习方法来提升模型性能")
        report.append("  - 建立更大规模的多中心验证数据集")
        report.append("")
        
        report.append("=" * 80)
        report.append("报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        report.append("=" * 80)
        
        # 保存报告
        report_text = "\n".join(report)
        with open('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/comprehensive_evaluation_report.txt', 
                 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("综合评估报告已生成并保存")
        print("报告文件：comprehensive_evaluation_report.txt")
        
        # 打印报告摘要
        print("\n" + "="*50)
        print("评估报告摘要")
        print("="*50)
        if 'binary_classification' in self.evaluation_results:
            bc = self.evaluation_results['binary_classification']
            print(f"ROC-AUC: {bc['auc_score']:.4f}")
            print(f"灵敏度: {bc['sensitivity']:.4f}")
            print(f"特异性: {bc['specificity']:.4f}")
            print(f"F1分数: {bc['f1_score']:.4f}")
        
        return report_text
    
    def save_evaluation_results(self):
        """保存评估结果"""
        # 保存主要评估指标
        if 'binary_classification' in self.evaluation_results:
            bc_results = self.evaluation_results['binary_classification']
            metrics_df = pd.DataFrame({
                'metric': ['accuracy', 'sensitivity', 'specificity', 'precision', 
                          'f1_score', 'auc_score', 'pr_auc', 'mcc', 'kappa', 'logloss'],
                'value': [bc_results['accuracy'], bc_results['sensitivity'], 
                         bc_results['specificity'], bc_results['precision'],
                         bc_results['f1_score'], bc_results['auc_score'],
                         bc_results['pr_auc'], bc_results['mcc'],
                         bc_results['kappa'], bc_results['logloss']]
            })
            metrics_df.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/evaluation_metrics.csv', 
                             index=False)
        
        # 保存混淆矩阵
        if 'binary_classification' in self.evaluation_results:
            cm = self.evaluation_results['binary_classification']['confusion_matrix']
            cm_df = pd.DataFrame(cm, 
                               index=['实际正常', '实际异常'],
                               columns=['预测正常', '预测异常'])
            cm_df.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/confusion_matrix.csv')
        
        print("评估结果已保存")
        print("评估指标：evaluation_metrics.csv")
        print("混淆矩阵：confusion_matrix.csv")
    
    def run_evaluation(self):
        """运行完整的模型评估流程"""
        print("=== 女胎异常判定模型评估开始 ===")
        
        # 1. 全面性能评估
        self.comprehensive_performance_evaluation()
        
        # 2. 临床意义分析
        self.clinical_significance_analysis()
        
        # 3. 生成综合报告
        self.generate_comprehensive_report()
        
        # 4. 保存评估结果
        self.save_evaluation_results()
        
        print("\n=== 模型评估完成 ===")
        
        return self.evaluation_results

def main():
    """主函数"""
    print("请先运行 model_training.py 和 threshold_optimization.py 获得模型和阈值")
    return None

if __name__ == "__main__":
    main()
