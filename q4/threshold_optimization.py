#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：女胎异常判定 - 动态阈值优化
根据Instruction.md的思路，实现三区动态阈值设定和决策规则
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DynamicThresholdOptimizer:
    """动态阈值优化器"""
    
    def __init__(self, model, X_test, y_test, y_pred_proba):
        """
        初始化阈值优化器
        
        Args:
            model: 训练好的模型
            X_test: 测试集特征
            y_test: 测试集标签
            y_pred_proba: 预测概率
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred_proba = y_pred_proba
        self.thresholds = {}
        self.decision_rules = {}
        self.optimization_results = {}
        
    def analyze_score_distributions(self):
        """
        分析健康群体和异常群体中判别函数得分的分布
        根据Instruction.md，基于得分分布设定阈值
        """
        print("=== 分析得分分布 ===")
        
        # 获取健康群体和异常群体的得分
        healthy_scores = self.y_pred_proba[self.y_test == 0]
        abnormal_scores = self.y_pred_proba[self.y_test == 1]
        
        print(f"健康群体得分统计：")
        print(f"  均值：{healthy_scores.mean():.4f}")
        print(f"  标准差：{healthy_scores.std():.4f}")
        print(f"  范围：{healthy_scores.min():.4f} - {healthy_scores.max():.4f}")
        
        print(f"异常群体得分统计：")
        print(f"  均值：{abnormal_scores.mean():.4f}")
        print(f"  标准差：{abnormal_scores.std():.4f}")
        print(f"  范围：{abnormal_scores.min():.4f} - {abnormal_scores.max():.4f}")
        
        # 可视化得分分布
        plt.figure(figsize=(12, 8))
        
        # 子图1：得分分布直方图
        plt.subplot(2, 2, 1)
        plt.hist(healthy_scores, bins=30, alpha=0.7, label='健康群体', color='lightblue', density=True)
        plt.hist(abnormal_scores, bins=30, alpha=0.7, label='异常群体', color='lightcoral', density=True)
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='传统阈值=0.5')
        plt.xlabel('判别函数得分')
        plt.ylabel('密度')
        plt.title('得分分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：累积分布函数
        plt.subplot(2, 2, 2)
        healthy_sorted = np.sort(healthy_scores)
        abnormal_sorted = np.sort(abnormal_scores)
        
        plt.plot(healthy_sorted, np.linspace(0, 1, len(healthy_sorted)), 
                label='健康群体CDF', color='blue')
        plt.plot(abnormal_sorted, np.linspace(0, 1, len(abnormal_sorted)), 
                label='异常群体CDF', color='red')
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('判别函数得分')
        plt.ylabel('累积概率')
        plt.title('累积分布函数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图3：箱线图
        plt.subplot(2, 2, 3)
        data_to_plot = [healthy_scores, abnormal_scores]
        labels = ['健康群体', '异常群体']
        box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        plt.ylabel('判别函数得分')
        plt.title('得分分布箱线图')
        plt.grid(True, alpha=0.3)
        
        # 子图4：Q-Q图
        plt.subplot(2, 2, 4)
        stats.probplot(healthy_scores, dist="norm", plot=plt)
        plt.title('健康群体Q-Q图')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/score_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return healthy_scores, abnormal_scores
    
    def optimize_three_zone_thresholds(self, healthy_scores, abnormal_scores):
        """
        优化三区阈值设定
        根据Instruction.md，设定高风险阈值D_H和低风险阈值D_L
        """
        print("\n=== 优化三区阈值设定 ===")
        
        # 方法1：基于分位数的方法
        # 低风险阈值：健康群体95%分位数
        D_L_quantile = np.percentile(healthy_scores, 95)
        
        # 高风险阈值：异常群体5%分位数
        D_H_quantile = np.percentile(abnormal_scores, 5)
        
        print(f"基于分位数的阈值：")
        print(f"  低风险阈值 D_L = {D_L_quantile:.4f}")
        print(f"  高风险阈值 D_H = {D_H_quantile:.4f}")
        
        # 方法2：基于统计距离的方法
        # 使用均值+标准差的方法
        healthy_mean = healthy_scores.mean()
        healthy_std = healthy_scores.std()
        abnormal_mean = abnormal_scores.mean()
        abnormal_std = abnormal_scores.std()
        
        # 低风险阈值：健康群体均值 + 2倍标准差
        D_L_stat = healthy_mean + 2 * healthy_std
        
        # 高风险阈值：异常群体均值 - 2倍标准差
        D_H_stat = abnormal_mean - 2 * abnormal_std
        
        print(f"基于统计距离的阈值：")
        print(f"  低风险阈值 D_L = {D_L_stat:.4f}")
        print(f"  高风险阈值 D_H = {D_H_stat:.4f}")
        
        # 方法3：基于ROC曲线的方法
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        
        # 找到最优阈值（约登指数最大）
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        
        # 基于最优阈值设定三区阈值
        D_L_roc = optimal_threshold - 0.1  # 低风险阈值
        D_H_roc = optimal_threshold + 0.1  # 高风险阈值
        
        print(f"基于ROC曲线的阈值：")
        print(f"  最优阈值 = {optimal_threshold:.4f}")
        print(f"  低风险阈值 D_L = {D_L_roc:.4f}")
        print(f"  高风险阈值 D_H = {D_H_roc:.4f}")
        
        # 选择最佳阈值组合
        # 这里选择基于分位数的方法，因为它更直观
        self.thresholds = {
            'D_L': D_L_quantile,
            'D_H': D_H_quantile,
            'optimal': optimal_threshold
        }
        
        print(f"\n最终选择的阈值：")
        print(f"  低风险阈值 D_L = {self.thresholds['D_L']:.4f}")
        print(f"  高风险阈值 D_H = {self.thresholds['D_H']:.4f}")
        
        return self.thresholds
    
    def implement_decision_rules(self):
        """
        实现三区决策规则
        根据Instruction.md的决策规则：
        - D >= D_H: 高风险异常
        - D < D_L: 低风险正常
        - D_L <= D < D_H: 结果不确定
        """
        print("\n=== 实现三区决策规则 ===")
        
        # 应用决策规则
        decisions = []
        for score in self.y_pred_proba:
            if score >= self.thresholds['D_H']:
                decisions.append('高风险异常')
            elif score < self.thresholds['D_L']:
                decisions.append('低风险正常')
            else:
                decisions.append('结果不确定')
        
        decisions = np.array(decisions)
        
        # 统计决策结果
        decision_counts = pd.Series(decisions).value_counts()
        print("决策结果统计：")
        for decision, count in decision_counts.items():
            percentage = count / len(decisions) * 100
            print(f"  {decision}: {count} ({percentage:.1f}%)")
        
        # 分析决策准确性
        print("\n决策准确性分析：")
        
        # 高风险异常决策的准确性
        high_risk_mask = decisions == '高风险异常'
        if sum(high_risk_mask) > 0:
            high_risk_accuracy = sum(self.y_test[high_risk_mask] == 1) / sum(high_risk_mask)
            print(f"  高风险异常决策准确率：{high_risk_accuracy:.3f}")
        
        # 低风险正常决策的准确性
        low_risk_mask = decisions == '低风险正常'
        if sum(low_risk_mask) > 0:
            low_risk_accuracy = sum(self.y_test[low_risk_mask] == 0) / sum(low_risk_mask)
            print(f"  低风险正常决策准确率：{low_risk_accuracy:.3f}")
        
        # 不确定结果分析
        uncertain_mask = decisions == '结果不确定'
        if sum(uncertain_mask) > 0:
            uncertain_abnormal_rate = sum(self.y_test[uncertain_mask] == 1) / sum(uncertain_mask)
            print(f"  不确定结果中异常比例：{uncertain_abnormal_rate:.3f}")
        
        self.decision_rules = {
            'decisions': decisions,
            'decision_counts': decision_counts,
            'high_risk_accuracy': high_risk_accuracy if sum(high_risk_mask) > 0 else 0,
            'low_risk_accuracy': low_risk_accuracy if sum(low_risk_mask) > 0 else 0,
            'uncertain_abnormal_rate': uncertain_abnormal_rate if sum(uncertain_mask) > 0 else 0
        }
        
        return self.decision_rules
    
    def visualize_three_zone_decision(self):
        """可视化三区决策结果"""
        print("\n=== 可视化三区决策结果 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('三区动态阈值决策可视化', fontsize=16, fontweight='bold')
        
        # 1. 得分分布与阈值
        axes[0, 0].hist(self.y_pred_proba[self.y_test == 0], bins=30, alpha=0.7, 
                       label='健康群体', color='lightblue', density=True)
        axes[0, 0].hist(self.y_pred_proba[self.y_test == 1], bins=30, alpha=0.7, 
                       label='异常群体', color='lightcoral', density=True)
        
        # 添加阈值线
        axes[0, 0].axvline(x=self.thresholds['D_L'], color='green', linestyle='--', 
                          linewidth=2, label=f'低风险阈值 D_L={self.thresholds["D_L"]:.3f}')
        axes[0, 0].axvline(x=self.thresholds['D_H'], color='red', linestyle='--', 
                          linewidth=2, label=f'高风险阈值 D_H={self.thresholds["D_H"]:.3f}')
        
        axes[0, 0].set_xlabel('判别函数得分')
        axes[0, 0].set_ylabel('密度')
        axes[0, 0].set_title('得分分布与阈值设定')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 决策区域划分
        x_range = np.linspace(0, 1, 1000)
        y_range = np.ones_like(x_range)
        
        # 低风险区域
        low_risk_mask = x_range < self.thresholds['D_L']
        axes[0, 1].fill_between(x_range[low_risk_mask], 0, y_range[low_risk_mask], 
                               alpha=0.3, color='green', label='低风险正常区域')
        
        # 不确定区域
        uncertain_mask = (x_range >= self.thresholds['D_L']) & (x_range < self.thresholds['D_H'])
        axes[0, 1].fill_between(x_range[uncertain_mask], 0, y_range[uncertain_mask], 
                               alpha=0.3, color='yellow', label='不确定区域')
        
        # 高风险区域
        high_risk_mask = x_range >= self.thresholds['D_H']
        axes[0, 1].fill_between(x_range[high_risk_mask], 0, y_range[high_risk_mask], 
                               alpha=0.3, color='red', label='高风险异常区域')
        
        axes[0, 1].set_xlabel('判别函数得分')
        axes[0, 1].set_ylabel('')
        axes[0, 1].set_title('三区决策区域划分')
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1.2)
        
        # 3. 决策结果统计
        decision_counts = self.decision_rules['decision_counts']
        colors = ['green', 'yellow', 'red']
        wedges, texts, autotexts = axes[1, 0].pie(decision_counts.values, 
                                                 labels=decision_counts.index,
                                                 autopct='%1.1f%%',
                                                 colors=colors)
        axes[1, 0].set_title('决策结果分布')
        
        # 4. 决策准确性对比
        accuracy_data = [
            self.decision_rules['low_risk_accuracy'],
            self.decision_rules['uncertain_abnormal_rate'],
            self.decision_rules['high_risk_accuracy']
        ]
        accuracy_labels = ['低风险准确率', '不确定区异常率', '高风险准确率']
        
        bars = axes[1, 1].bar(accuracy_labels, accuracy_data, 
                             color=['green', 'orange', 'red'], alpha=0.7)
        axes[1, 1].set_ylabel('准确率/异常率')
        axes[1, 1].set_title('决策准确性分析')
        axes[1, 1].set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, accuracy_data):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/three_zone_decision.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_threshold_performance(self):
        """评估阈值性能"""
        print("\n=== 评估阈值性能 ===")
        
        # 计算不同阈值下的性能指标
        thresholds_to_test = np.linspace(0.1, 0.9, 17)
        performance_metrics = []
        
        for threshold in thresholds_to_test:
            # 二分类预测
            y_pred_binary = (self.y_pred_proba >= threshold).astype(int)
            
            # 计算性能指标
            tp = sum((y_pred_binary == 1) & (self.y_test == 1))
            fp = sum((y_pred_binary == 1) & (self.y_test == 0))
            fn = sum((y_pred_binary == 0) & (self.y_test == 1))
            tn = sum((y_pred_binary == 0) & (self.y_test == 0))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            performance_metrics.append({
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1_score
            })
        
        performance_df = pd.DataFrame(performance_metrics)
        
        # 可视化性能曲线
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(performance_df['threshold'], performance_df['sensitivity'], 'b-', label='灵敏度')
        plt.plot(performance_df['threshold'], performance_df['specificity'], 'r-', label='特异性')
        plt.axvline(x=self.thresholds['D_L'], color='green', linestyle='--', alpha=0.7, label='D_L')
        plt.axvline(x=self.thresholds['D_H'], color='red', linestyle='--', alpha=0.7, label='D_H')
        plt.xlabel('阈值')
        plt.ylabel('性能指标')
        plt.title('灵敏度与特异性曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(performance_df['threshold'], performance_df['precision'], 'g-', label='精确率')
        plt.plot(performance_df['threshold'], performance_df['f1_score'], 'm-', label='F1分数')
        plt.axvline(x=self.thresholds['D_L'], color='green', linestyle='--', alpha=0.7, label='D_L')
        plt.axvline(x=self.thresholds['D_H'], color='red', linestyle='--', alpha=0.7, label='D_H')
        plt.xlabel('阈值')
        plt.ylabel('性能指标')
        plt.title('精确率与F1分数曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        # ROC曲线
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        plt.plot(fpr, tpr, 'b-', label='ROC曲线')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('ROC曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # 精确率-召回率曲线
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        plt.plot(recall_curve, precision_curve, 'g-', label='PR曲线')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/threshold_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存性能数据
        performance_df.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/threshold_performance.csv', 
                             index=False)
        
        self.optimization_results = {
            'performance_df': performance_df,
            'thresholds': self.thresholds,
            'decision_rules': self.decision_rules
        }
        
        return self.optimization_results
    
    def save_optimization_results(self):
        """保存优化结果"""
        # 保存阈值
        thresholds_df = pd.DataFrame({
            'threshold_type': ['D_L', 'D_H', 'optimal'],
            'threshold_value': [self.thresholds['D_L'], self.thresholds['D_H'], self.thresholds['optimal']]
        })
        thresholds_df.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/optimized_thresholds.csv', 
                            index=False)
        
        # 保存决策规则结果
        decision_results_df = pd.DataFrame({
            'metric': ['high_risk_accuracy', 'low_risk_accuracy', 'uncertain_abnormal_rate'],
            'value': [
                self.decision_rules['high_risk_accuracy'],
                self.decision_rules['low_risk_accuracy'],
                self.decision_rules['uncertain_abnormal_rate']
            ]
        })
        decision_results_df.to_csv('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/q4/decision_rules_results.csv', 
                                  index=False)
        
        print("阈值优化结果已保存")
        print("阈值文件：optimized_thresholds.csv")
        print("决策规则结果：decision_rules_results.csv")
    
    def run_optimization(self):
        """运行完整的阈值优化流程"""
        print("=== 动态阈值优化开始 ===")
        
        # 1. 分析得分分布
        healthy_scores, abnormal_scores = self.analyze_score_distributions()
        
        # 2. 优化三区阈值
        self.optimize_three_zone_thresholds(healthy_scores, abnormal_scores)
        
        # 3. 实现决策规则
        self.implement_decision_rules()
        
        # 4. 可视化决策结果
        self.visualize_three_zone_decision()
        
        # 5. 评估阈值性能
        self.evaluate_threshold_performance()
        
        # 6. 保存优化结果
        self.save_optimization_results()
        
        print("\n=== 动态阈值优化完成 ===")
        
        return self.optimization_results

def main():
    """主函数"""
    print("请先运行 model_training.py 获得训练好的模型和预测结果")
    return None

if __name__ == "__main__":
    main()
