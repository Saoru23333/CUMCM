# 第一问：混合效应模型分析

本目录包含用于分析Y染色体浓度与BMI、孕周关系的混合效应模型代码。根据Instruction.md中的指导，我们使用带有随机截距的混合效应模型来处理重复测量数据的非独立性问题。

## 文件结构说明

### 核心模型文件

#### 1. `mixed_effects.py` - 主模型文件
**功能**：拟合混合效应模型并生成基础输出文件
**模型形式**：Y_ij = (β₀ + u_i) + β₁·Week_ij + β₂·BMI_ij + ε_ij

**使用方法**：
```bash
python mixed_effects.py
```

**输出文件**：
- `mixed_effects_predictions.csv` - 预测结果和残差
- `mixed_effects_model_info.md` - 模型详细信息
- `mixed_effects_coefs.csv` - 模型系数
- `fixed_effects_functions.csv` - 固定效应函数值

#### 2. `mixed_effects_analysis.py` - 模型分析文件
**功能**：进行相关性分析和混合效应模型拟合，生成评估报告

**使用方法**：
```bash
python mixed_effects_analysis.py
```

**输出文件**：
- `simple_correlations.csv` - 简单相关性分析结果
- `mixed_effects_eval.txt` - 模型评估报告
- `mixed_effects_coefs.csv` - 模型系数

#### 3. `mixed_effects_eval.py` - 模型评估文件
**功能**：详细的模型评估，包括与简单线性模型的比较

**使用方法**：
```bash
python mixed_effects_eval.py
```

**输出文件**：
- `mixed_effects_eval.txt` - 详细评估报告
- `mixed_effects_coefs.csv` - 模型系数

#### 4. `mixed_effects_visualize.py` - 可视化文件
**功能**：生成各种可视化图表

**使用方法**：
```bash
python mixed_effects_visualize.py
```

**输出文件**：
- `mixed_effects_surface.png` - 3D固定效应曲面图
- `individual_trajectories.png` - 个体轨迹图
- `residual_analysis.png` - 残差分析图
- `random_intercepts_distribution.png` - 随机截距分布图

### 辅助分析文件

#### 5. `distance_correlation_analysis.py` - 距离相关性分析
**功能**：计算所有数值变量与Y染色体浓度的距离相关系数

**使用方法**：
```bash
python distance_correlation_analysis.py
```

**输出文件**：
- `distance_correlation_results.csv` - 距离相关性分析结果

#### 6. `Instruction.md` - 理论指导文档
**功能**：详细说明为什么选择混合效应模型以及模型的理论基础

## 运行顺序建议

### 完整分析流程
1. **数据探索**：
   ```bash
   python distance_correlation_analysis.py
   ```

2. **模型拟合**：
   ```bash
   python mixed_effects.py
   ```

3. **模型分析**：
   ```bash
   python mixed_effects_analysis.py
   ```

4. **详细评估**：
   ```bash
   python mixed_effects_eval.py
   ```

5. **结果可视化**：
   ```bash
   python mixed_effects_visualize.py
   ```

### 快速分析流程
如果只需要基本结果，可以只运行：
```bash
python mixed_effects.py
python mixed_effects_analysis.py
```

## 输出文件说明

### 主要输出文件

1. **`mixed_effects_predictions.csv`**
   - 包含每个观测的预测值、实际值和残差
   - 用于模型诊断和结果验证

2. **`mixed_effects_model_info.md`**
   - 模型参数、固定效应、随机效应
   - 组内相关系数(ICC)及其解释

3. **`mixed_effects_eval.txt`**
   - 显著性检验结果
   - 模型性能指标
   - 与简单线性模型的比较

4. **`mixed_effects_coefs.csv`**
   - 固定效应系数、标准误、t值、p值
   - 用于回答第一问的显著性检验要求

### 可视化文件

1. **`mixed_effects_surface.png`** - 3D曲面图显示固定效应
2. **`individual_trajectories.png`** - 个体轨迹图显示随机截距影响
3. **`residual_analysis.png`** - 残差分析图用于模型诊断
4. **`random_intercepts_distribution.png`** - 随机截距分布图

## 模型解释

### 混合效应模型优势
1. **处理非独立数据**：正确处理同一孕妇多次检测的相关性
2. **量化个体差异**：通过随机截距捕捉孕妇间的个体差异
3. **更准确的推断**：提供更稳健的p值和置信区间

### 关键指标
- **ICC (组内相关系数)**：衡量总变异中由孕妇间差异导致的比例
- **固定效应**：BMI和孕周对Y染色体浓度的平均影响
- **随机截距**：每个孕妇的个体化基线差异

## 注意事项

1. **数据路径**：确保`data/output.csv`文件存在且路径正确
2. **输出目录**：所有结果保存在`./1/`目录下
3. **依赖包**：需要安装`statsmodels`、`pandas`、`numpy`、`matplotlib`等包
4. **内存使用**：对于大数据集，混合效应模型可能需要较多内存

## 故障排除

### 常见问题
1. **数据文件未找到**：检查`data/output.csv`是否存在
2. **模型拟合失败**：检查数据质量和样本量
3. **可视化错误**：确保安装了matplotlib和相关字体

### 联系信息
如有问题，请参考`Instruction.md`中的详细理论说明。
