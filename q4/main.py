#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问：女胎异常判定 - 主程序
根据Instruction.md的思路，整合所有模块，实现完整的女胎异常判定系统
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from data_exploration import FemaleFetusDataExplorer
from feature_engineering import FemaleFetusFeatureEngineer
from model_training import FemaleFetusModelTrainer
from threshold_optimization import DynamicThresholdOptimizer
from model_evaluation import FemaleFetusModelEvaluator

def main():
    """主函数 - 运行完整的女胎异常判定流程"""
    print("=" * 80)
    print("第四问：女胎异常判定系统")
    print("根据Instruction.md的思路构建多因素融合模型")
    print("=" * 80)
    
    try:
        # 步骤1：数据探索性分析
        print("\n【步骤1】数据探索性分析")
        print("-" * 50)
        explorer = FemaleFetusDataExplorer('/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/girl_output.csv')
        features, target = explorer.run_exploration()
        
        if features is None or target is None:
            print("数据探索失败，程序终止")
            return
        
        # 步骤2：特征工程
        print("\n【步骤2】特征工程")
        print("-" * 50)
        engineer = FemaleFetusFeatureEngineer(features, target)
        processed_features, processed_target = engineer.run_feature_engineering()
        
        if processed_features is None:
            print("特征工程失败，程序终止")
            return
        
        # 步骤3：模型训练
        print("\n【步骤3】模型训练")
        print("-" * 50)
        trainer = FemaleFetusModelTrainer(processed_features, processed_target)
        model, training_results = trainer.run_training()
        
        if model is None:
            print("模型训练失败，程序终止")
            return
        
        # 步骤4：动态阈值优化
        print("\n【步骤4】动态阈值优化")
        print("-" * 50)
        optimizer = DynamicThresholdOptimizer(
            model, trainer.X_test, trainer.y_test, 
            training_results['y_pred_proba']
        )
        optimization_results = optimizer.run_optimization()
        
        if optimization_results is None:
            print("阈值优化失败，程序终止")
            return
        
        # 步骤5：模型评估
        print("\n【步骤5】模型评估")
        print("-" * 50)
        evaluator = FemaleFetusModelEvaluator(
            model, trainer.X_test, trainer.y_test,
            training_results['y_pred_proba'],
            optimization_results['thresholds']
        )
        evaluation_results = evaluator.run_evaluation()
        
        # 生成最终总结
        print("\n" + "=" * 80)
        print("女胎异常判定系统构建完成")
        print("=" * 80)
        
        # 输出关键结果
        if 'binary_classification' in evaluation_results:
            bc = evaluation_results['binary_classification']
            print(f"\n关键性能指标：")
            print(f"  ROC-AUC: {bc['auc_score']:.4f}")
            print(f"  灵敏度: {bc['sensitivity']:.4f}")
            print(f"  特异性: {bc['specificity']:.4f}")
            print(f"  F1分数: {bc['f1_score']:.4f}")
        
        if optimization_results and 'thresholds' in optimization_results:
            thresholds = optimization_results['thresholds']
            print(f"\n优化后的阈值：")
            print(f"  低风险阈值 D_L: {thresholds['D_L']:.4f}")
            print(f"  高风险阈值 D_H: {thresholds['D_H']:.4f}")
        
        print(f"\n生成的文件：")
        print(f"  - 数据分布分析图: data_distribution.png")
        print(f"  - Z值随孕周变化图: z_values_by_gestational_week.png")
        print(f"  - 特征相关性图: feature_correlations.png")
        print(f"  - 特征变换过程图: feature_transformation.png")
        print(f"  - 综合异常信号图: anomaly_signals.png")
        print(f"  - 模型性能图: model_performance.png")
        print(f"  - 特征重要性图: feature_importance.png")
        print(f"  - 三区决策图: three_zone_decision.png")
        print(f"  - 阈值性能图: threshold_performance.png")
        print(f"  - 临床分析图: clinical_analysis.png")
        print(f"  - 综合评估报告: comprehensive_evaluation_report.txt")
        
        print(f"\n数据文件：")
        print(f"  - 处理后的特征: processed_features.csv")
        print(f"  - 处理后的标签: processed_targets.csv")
        print(f"  - 健康群体统计: healthy_stats.csv")
        print(f"  - 特征重要性: feature_importance.csv")
        print(f"  - 训练结果: training_results.csv")
        print(f"  - 优化阈值: optimized_thresholds.csv")
        print(f"  - 决策规则结果: decision_rules_results.csv")
        print(f"  - 评估指标: evaluation_metrics.csv")
        print(f"  - 混淆矩阵: confusion_matrix.csv")
        print(f"  - 临床分析: clinical_analysis.csv")
        print(f"  - 训练好的模型: trained_model.pkl")
        
        print(f"\n系统构建成功！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n程序执行过程中出现错误：{str(e)}")
        print("请检查数据文件路径和格式是否正确")
        import traceback
        traceback.print_exc()

def quick_demo():
    """快速演示模式 - 使用示例数据"""
    print("=" * 80)
    print("女胎异常判定系统 - 快速演示模式")
    print("=" * 80)
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 200
    
    # 生成示例特征
    features_data = {
        '13号染色体的Z值': np.random.normal(0, 1, n_samples),
        '18号染色体的Z值': np.random.normal(0, 1, n_samples),
        '21号染色体的Z值': np.random.normal(0, 1, n_samples),
        'X染色体的Z值': np.random.normal(0, 1, n_samples),
        '13号染色体的GC含量': np.random.normal(0.4, 0.05, n_samples),
        '18号染色体的GC含量': np.random.normal(0.4, 0.05, n_samples),
        '21号染色体的GC含量': np.random.normal(0.4, 0.05, n_samples),
        '孕妇BMI': np.random.normal(25, 5, n_samples),
        '年龄': np.random.normal(30, 5, n_samples),
        '被过滤掉读段数的比例': np.random.normal(0.02, 0.01, n_samples)
    }
    
    features = pd.DataFrame(features_data)
    
    # 生成示例标签（基于Z值的简单规则）
    target = ((features['13号染色体的Z值'].abs() > 2) | 
              (features['18号染色体的Z值'].abs() > 2) | 
              (features['21号染色体的Z值'].abs() > 2)).astype(int)
    
    print(f"示例数据：{n_samples}个样本，{features.shape[1]}个特征")
    print(f"标签分布：正常={sum(target==0)}，异常={sum(target==1)}")
    
    try:
        # 运行特征工程
        print("\n运行特征工程...")
        engineer = FemaleFetusFeatureEngineer(features, target)
        processed_features, processed_target = engineer.run_feature_engineering()
        
        # 运行模型训练
        print("\n运行模型训练...")
        trainer = FemaleFetusModelTrainer(processed_features, processed_target)
        model, training_results = trainer.run_training()
        
        # 运行阈值优化
        print("\n运行阈值优化...")
        optimizer = DynamicThresholdOptimizer(
            model, trainer.X_test, trainer.y_test, 
            training_results['y_pred_proba']
        )
        optimization_results = optimizer.run_optimization()
        
        # 运行模型评估
        print("\n运行模型评估...")
        evaluator = FemaleFetusModelEvaluator(
            model, trainer.X_test, trainer.y_test,
            training_results['y_pred_proba'],
            optimization_results['thresholds']
        )
        evaluation_results = evaluator.run_evaluation()
        
        print("\n快速演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 检查是否存在数据文件
    data_file = '/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/girl_output.csv'
    
    if os.path.exists(data_file):
        print("检测到数据文件，运行完整流程...")
        main()
    else:
        print("未检测到数据文件，运行快速演示模式...")
        quick_demo()
