# 第四问：女胎异常判定系统

## 概述

本系统根据Instruction.md的思路，构建了一个专门针对女胎的异常判定模型。由于女胎和母体均不携带Y染色体，无法通过Y染色体浓度来评估检测的准确性，因此需要建立一个多因素融合模型，综合利用NIPT检测产生的多维度数据。

## 系统架构

### 核心模块

1. **数据探索性分析** (`data_exploration.py`)
   - 筛选女胎样本
   - 识别特征和标签
   - 分析数据分布和相关性

2. **特征工程** (`feature_engineering.py`)
   - 特征标准化（Z-score Normalization）
   - 构建质量校正的Z值指标
   - 生成综合异常信号

3. **模型训练** (`model_training.py`)
   - 逻辑回归模型训练
   - 权重学习
   - 交叉验证

4. **动态阈值优化** (`threshold_optimization.py`)
   - 三区阈值设定
   - 决策规则实现
   - 性能评估

5. **模型评估** (`model_evaluation.py`)
   - 全面性能评估
   - 临床意义分析
   - 综合报告生成

## 使用方法

### 环境要求

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 运行方式

#### 方式1：运行完整流程
```bash
python main.py
```

#### 方式2：分步运行
```bash
# 1. 数据探索
python data_exploration.py

# 2. 特征工程
python feature_engineering.py

# 3. 模型训练
python model_training.py

# 4. 阈值优化
python threshold_optimization.py

# 5. 模型评估
python model_evaluation.py
```

### 数据格式要求

输入数据应为CSV格式，包含以下关键列：

- **Z值列**：13号、18号、21号、X染色体的Z值
- **GC含量列**：对应染色体的GC含量
- **孕妇生理指标**：BMI、年龄等
- **测序质量指标**：被过滤读段比例等
- **标签列**：判定结果（空白为正常，有内容为异常）

## 核心算法

### 1. 特征标准化

使用健康孕妇群体的均值和标准差进行Z-score标准化：

```
x' = (x - μ_healthy) / σ_healthy
```

### 2. 质量校正的Z值指标

构建校正因子，考虑GC含量和测序质量：

```
δ_j = f(Z'_j, Z_X, BMI, Age, ...)
```

其中：
- `Z'_j`：染色体j经质量校正后的Z值
- `Z_X`：X染色体Z值作为参考
- `BMI`、`Age`：孕妇生理指标

### 3. 逻辑回归模型

最终判别函数：

```
D_j = w_1 * δ_j + w_2 * BMI_norm + w_3 * Age_norm + ...
```

权重通过逻辑回归学习得到。

### 4. 三区动态阈值

- **高风险阈值 D_H**：异常群体5%分位数
- **低风险阈值 D_L**：健康群体95%分位数
- **决策规则**：
  - D ≥ D_H：高风险异常
  - D < D_L：低风险正常
  - D_L ≤ D < D_H：结果不确定

## 输出文件

### 可视化文件
- `data_distribution.png`：数据分布分析
- `z_values_by_gestational_week.png`：Z值随孕周变化
- `feature_correlations.png`：特征相关性热力图
- `feature_transformation.png`：特征变换过程
- `anomaly_signals.png`：综合异常信号分布
- `model_performance.png`：模型性能可视化
- `feature_importance.png`：特征重要性分析
- `three_zone_decision.png`：三区决策可视化
- `threshold_performance.png`：阈值性能分析
- `clinical_analysis.png`：临床意义分析

### 数据文件
- `processed_features.csv`：处理后的特征数据
- `processed_targets.csv`：处理后的标签数据
- `healthy_stats.csv`：健康群体统计量
- `feature_importance.csv`：特征重要性数据
- `training_results.csv`：训练结果
- `optimized_thresholds.csv`：优化后的阈值
- `decision_rules_results.csv`：决策规则结果
- `evaluation_metrics.csv`：评估指标
- `confusion_matrix.csv`：混淆矩阵
- `clinical_analysis.csv`：临床分析结果
- `trained_model.pkl`：训练好的模型

### 报告文件
- `comprehensive_evaluation_report.txt`：综合评估报告

## 性能指标

系统评估以下关键指标：

- **灵敏度 (Sensitivity)**：识别异常样本的能力
- **特异性 (Specificity)**：识别正常样本的能力
- **精确率 (Precision)**：预测为异常中真正异常的比例
- **F1-Score**：精确率和召回率的调和平均
- **ROC-AUC**：ROC曲线下面积
- **PR-AUC**：精确率-召回率曲线下面积

## 临床意义

### 优势
- 多因素融合，综合考虑多种检测指标
- 质量校正，有效降低测序噪音
- 三区动态阈值，提高决策稳健性
- 良好的可解释性

### 局限性
- 依赖训练数据的质量和代表性
- 对极少数样本预测可能存在不确定性
- 需要定期重新训练

### 建议
- 结合其他检测方法进行综合判断
- 对不确定区域样本进行复检
- 定期评估和更新模型

## 技术特点

1. **模块化设计**：每个功能独立成模块，便于维护和扩展
2. **完整的评估体系**：从数据探索到模型评估的完整流程
3. **可视化丰富**：提供多种图表帮助理解数据和模型
4. **临床导向**：重点关注临床应用的实用性和可靠性
5. **可重现性**：设置随机种子，确保结果可重现

## 注意事项

1. 确保输入数据格式正确，包含必要的特征列
2. 数据质量对模型性能有重要影响
3. 建议在临床应用中结合其他检测方法
4. 定期评估模型性能，必要时进行更新

## 联系方式

如有问题或建议，请联系开发团队。
