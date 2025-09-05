import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SuccessProbabilityCalculator:
    """
    成功概率计算器
    计算 p_s(t, X_i) = P(C_i(t) >= C_min)
    基于q1的混合效应模型：C_i(t) = (β₀ + u_i) + β₁·Week + β₂·BMI + ε
    """
    
    def __init__(self, mixed_effects_coefs_path, c_min=0.04):
        """
        初始化成功概率计算器
        
        参数:
        mixed_effects_coefs_path: 混合效应模型系数文件路径
        c_min: 检测成功的最低Y染色体浓度阈值，默认0.04 (4%)
        """
        self.c_min = c_min
        
        # 加载混合效应模型系数
        self.coefs = pd.read_csv(mixed_effects_coefs_path)
        
        # 提取模型参数
        self.beta_0 = self.coefs[self.coefs['变量'] == 'Intercept']['系数'].iloc[0]  # 固定截距
        self.beta_1 = self.coefs[self.coefs['变量'] == 'week']['系数'].iloc[0]      # 孕周系数
        self.beta_2 = self.coefs[self.coefs['变量'] == 'bmi']['系数'].iloc[0]       # BMI系数
        
        # 从模型评估文件中获取残差标准差
        self._load_model_parameters()
        
        print(f"混合效应模型参数加载完成:")
        print(f"  固定截距 (β₀): {self.beta_0:.6f}")
        print(f"  孕周系数 (β₁): {self.beta_1:.6f}")
        print(f"  BMI系数 (β₂): {self.beta_2:.6f}")
        print(f"  残差标准差 (σ_ε): {self.residual_std:.6f}")
        print(f"  随机截距标准差 (σ_u): {self.random_intercept_std:.6f}")
        print(f"  检测成功阈值 (C_min): {self.c_min}")
    
    def _load_model_parameters(self):
        """从模型评估文件中加载残差标准差等参数"""
        eval_file = './1/mixed_effects_eval.txt'
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 提取残差标准差
            for line in content.split('\n'):
                if '残差标准差:' in line:
                    self.residual_std = float(line.split(':')[1].strip())
                elif '随机截距标准差:' in line:
                    self.random_intercept_std = float(line.split(':')[1].strip())
                    
        except FileNotFoundError:
            print(f"警告: 未找到模型评估文件 {eval_file}")
            print("使用默认参数值")
            self.residual_std = 0.017526  # 从之前的结果中获取
            self.random_intercept_std = 0.027380
    
    def predict_y_concentration(self, bmi, gestational_week, include_random_effect=True):
        """
        基于混合效应模型预测Y染色体浓度
        C_i(t) = (β₀ + u_i) + β₁·Week + β₂·BMI + ε
        
        参数:
        bmi: BMI值
        gestational_week: 孕周
        include_random_effect: 是否包含随机截距效应
        
        返回:
        预测的Y染色体浓度
        """
        # 固定效应部分
        fixed_effect = self.beta_0 + self.beta_1 * gestational_week + self.beta_2 * bmi
        
        if include_random_effect:
            # 包含随机截距效应（个体差异）
            # 使用随机截距的标准差来模拟个体差异
            random_intercept = np.random.normal(0, self.random_intercept_std)
            return fixed_effect + random_intercept
        else:
            # 只返回固定效应（平均预测）
            return fixed_effect
    
    def predict_y_concentration_mean(self, bmi, gestational_week):
        """
        预测Y染色体浓度的平均值（不包含随机效应）
        用于确定性预测
        
        参数:
        bmi: BMI值
        gestational_week: 孕周
        
        返回:
        预测的Y染色体浓度平均值
        """
        return self.beta_0 + self.beta_1 * gestational_week + self.beta_2 * bmi
    
    def calculate_success_probability(self, bmi, gestational_week, method='normal'):
        """
        计算成功概率 p_s(t, X_i) = P(C_i(t) >= C_min)
        基于混合效应模型的正态分布假设
        
        参数:
        bmi: BMI值
        gestational_week: 孕周
        method: 概率计算方法 (只支持 'normal')
        
        返回:
        成功概率值
        """
        if method == 'normal':
            # 基于混合效应模型预测 + 正态分布假设（时间依赖方差 + BMI依赖）
            predicted_concentration = self.predict_y_concentration_mean(bmi, gestational_week)
            return self._calculate_normal_probability(predicted_concentration, gestational_week, bmi)
        else:
            raise ValueError("method参数只支持 'normal'")
    
    def _calculate_normal_probability(self, predicted_concentration, gestational_week, bmi):
        """
        基于混合效应模型计算成功概率，采用孕周依赖的总标准差
        直觉：越早孕周测量不确定性越大 → 成功率应更低
        """
        # 基础方差（随机截距 + 残差）
        base_variance = self.random_intercept_std**2 + self.residual_std**2
        base_std = np.sqrt(base_variance)

        # 孕周依赖方差放大系数：早期更大，随孕周递减至1
        # 参数可调：alpha 控制早期放大量级，scale 控制衰减速度（天）
        # 放大项加入BMI依赖：高BMI早期不确定性更大 → 成功率更低
        # 温和方差放大（前移设置）：适度降低早期惩罚、稍增强BMI梯度、放缓衰减
        alpha_base = 0.85
        alpha_bmi = 0.025 * (bmi - 30.0)  # 略增BMI系数以放大组差
        alpha = max(0.6, min(1.6, alpha_base + alpha_bmi))
        scale = 30.0
        multiplier = 1.0 + alpha * np.exp(-(gestational_week - 118.0) / scale)
        multiplier = float(max(1.0, multiplier))  # 不小于1

        total_std = base_std * multiplier

        # 计算成功概率 P(C >= C_min)
        z_score = (self.c_min - predicted_concentration) / total_std
        success_prob = 1 - stats.norm.cdf(z_score)

        return max(0, min(1, success_prob))
    
    
    def calculate_success_probability_grid(self, bmi_range, gestational_range, method='normal'):
        """
        计算网格上的成功概率
        
        参数:
        bmi_range: BMI范围 (min, max, n_points)
        gestational_range: 孕周范围 (min, max, n_points)
        method: 概率计算方法
        
        返回:
        成功概率网格
        """
        bmi_min, bmi_max, bmi_n = bmi_range
        gestational_min, gestational_max, gestational_n = gestational_range
        
        bmi_grid = np.linspace(bmi_min, bmi_max, bmi_n)
        gestational_grid = np.linspace(gestational_min, gestational_max, gestational_n)
        
        success_prob_grid = np.zeros((len(gestational_grid), len(bmi_grid)))
        
        for i, gestational_week in enumerate(gestational_grid):
            for j, bmi in enumerate(bmi_grid):
                success_prob_grid[i, j] = self.calculate_success_probability(
                    bmi, gestational_week, method=method
                )
        
        return bmi_grid, gestational_grid, success_prob_grid
    
    def visualize_success_probability(self, bmi_range=(20, 40, 20), gestational_range=(80, 200, 20), 
                                    method='normal', save_path=None):
        """
        可视化成功概率
        
        参数:
        bmi_range: BMI范围 (min, max, n_points)
        gestational_range: 孕周范围 (min, max, n_points)
        method: 概率计算方法
        save_path: 保存路径
        """
        bmi_grid, gestational_grid, success_prob_grid = self.calculate_success_probability_grid(
            bmi_range, gestational_range, method
        )
        
        # 创建热力图
        plt.figure(figsize=(12, 8))
        
        # 绘制热力图
        im = plt.imshow(success_prob_grid, cmap='RdYlBu_r', aspect='auto', 
                       extent=[bmi_grid.min(), bmi_grid.max(), 
                              gestational_grid.min(), gestational_grid.max()],
                       origin='lower')
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('成功概率', fontsize=12)
        
        # 设置标签和标题
        plt.xlabel('BMI', fontsize=12)
        plt.ylabel('孕周', fontsize=12)
        plt.title(f'Y染色体检测成功概率热力图\n(阈值 C_min = {self.c_min})', fontsize=14)
        
        # 添加等高线
        contour = plt.contour(bmi_grid, gestational_grid, success_prob_grid, 
                            levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='black', alpha=0.5)
        plt.clabel(contour, inline=True, fontsize=10)
        
        # 添加一些示例数据点用于参考
        # 生成一些示例BMI和孕周组合
        example_bmis = np.linspace(bmi_grid.min(), bmi_grid.max(), 10)
        example_weeks = np.linspace(gestational_grid.min(), gestational_grid.max(), 10)
        example_concentrations = []
        
        for bmi in example_bmis:
            for week in example_weeks:
                conc = self.predict_y_concentration_mean(bmi, week)
                example_concentrations.append(conc)
        
        # 创建网格用于散点图
        bmi_mesh, week_mesh = np.meshgrid(example_bmis, example_weeks)
        conc_mesh = np.array(example_concentrations).reshape(week_mesh.shape)
        
        scatter = plt.scatter(bmi_mesh.flatten(), week_mesh.flatten(), 
                            c=conc_mesh.flatten(), 
                            cmap='viridis', alpha=0.6, s=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"成功概率热力图已保存到: {save_path}")
        
        plt.show()
        
        return bmi_grid, gestational_grid, success_prob_grid
    
    def analyze_success_probability_by_factors(self, method='normal', bmi_range=(20, 40), gestational_range=(80, 200), n_samples=1000):
        """
        分析不同因素对成功概率的影响
        基于混合效应模型生成样本进行分析
        
        参数:
        method: 概率计算方法
        bmi_range: BMI范围 (min, max)
        gestational_range: 孕周范围 (min, max)
        n_samples: 生成样本数量
        
        返回:
        分析结果字典
        """
        # 生成分析样本
        np.random.seed(42)  # 确保结果可重复
        bmi_samples = np.random.uniform(bmi_range[0], bmi_range[1], n_samples)
        gestational_samples = np.random.uniform(gestational_range[0], gestational_range[1], n_samples)
        
        # 计算每个样本点的成功概率
        success_probs = []
        predicted_concentrations = []
        
        for bmi, week in zip(bmi_samples, gestational_samples):
            prob = self.calculate_success_probability(bmi, week, method=method)
            conc = self.predict_y_concentration_mean(bmi, week)
            success_probs.append(prob)
            predicted_concentrations.append(conc)
        
        # 创建分析数据框
        analysis_data = pd.DataFrame({
            'BMI': bmi_samples,
            '孕周': gestational_samples,
            '预测Y染色体浓度': predicted_concentrations,
            '成功概率': success_probs
        })
        
        # 按BMI分组分析
        bmi_bins = pd.cut(analysis_data['BMI'], bins=5, labels=['很低', '低', '中等', '高', '很高'])
        bmi_analysis = analysis_data.groupby(bmi_bins)['成功概率'].agg(['mean', 'std', 'count'])
        
        # 按孕周分组分析
        gestational_bins = pd.cut(analysis_data['孕周'], bins=5, labels=['早期', '早中期', '中期', '中晚期', '晚期'])
        gestational_analysis = analysis_data.groupby(gestational_bins)['成功概率'].agg(['mean', 'std', 'count'])
        
        # 相关性分析
        bmi_correlation = analysis_data['BMI'].corr(analysis_data['成功概率'])
        gestational_correlation = analysis_data['孕周'].corr(analysis_data['成功概率'])
        
        analysis_results = {
            'analysis_data': analysis_data,
            'bmi_analysis': bmi_analysis,
            'gestational_analysis': gestational_analysis,
            'bmi_correlation': bmi_correlation,
            'gestational_correlation': gestational_correlation,
            'overall_success_rate': np.mean(success_probs),
            'success_rate_std': np.std(success_probs)
        }
        
        return analysis_results
    
    def generate_success_probability_report(self, method='normal', save_path=None, n_samples=1000):
        """
        生成成功概率分析报告
        基于混合效应模型
        
        参数:
        method: 概率计算方法
        save_path: 报告保存路径
        n_samples: 生成样本数量
        """
        print("=" * 60)
        print("Y染色体检测成功概率分析报告")
        print("=" * 60)
        print(f"检测成功阈值 (C_min): {self.c_min}")
        print(f"概率计算方法: {method}")
        print(f"基于混合效应模型: C_i(t) = (β₀ + u_i) + β₁·Week + β₂·BMI + ε")
        print(f"模型参数:")
        print(f"  β₀ (固定截距): {self.beta_0:.6f}")
        print(f"  β₁ (孕周系数): {self.beta_1:.6f}")
        print(f"  β₂ (BMI系数): {self.beta_2:.6f}")
        print(f"  σ_u (随机截距标准差): {self.random_intercept_std:.6f}")
        print(f"  σ_ε (残差标准差): {self.residual_std:.6f}")
        print()
        
        # 使用分析函数获取结果
        analysis_results = self.analyze_success_probability_by_factors(method=method, n_samples=n_samples)
        analysis_data = analysis_results['analysis_data']
        
        # 基本统计
        print("基本统计信息:")
        print(f"  平均成功概率: {analysis_results['overall_success_rate']:.4f}")
        print(f"  成功概率标准差: {analysis_results['success_rate_std']:.4f}")
        print(f"  最小成功概率: {analysis_data['成功概率'].min():.4f}")
        print(f"  最大成功概率: {analysis_data['成功概率'].max():.4f}")
        print()
        
        # 按BMI分析
        print("按BMI分组的成功概率分析:")
        print(analysis_results['bmi_analysis'])
        print()
        
        # 按孕周分析
        print("按孕周分组的成功概率分析:")
        print(analysis_results['gestational_analysis'])
        print()
        
        # 相关性分析
        print("相关性分析:")
        print(f"  BMI与成功概率的相关系数: {analysis_results['bmi_correlation']:.4f}")
        print(f"  孕周与成功概率的相关系数: {analysis_results['gestational_correlation']:.4f}")
        print()
        
        # 保存结果
        if save_path:
            # 保存详细结果
            detailed_results = analysis_data[['BMI', '孕周', '预测Y染色体浓度', '成功概率']].copy()
            detailed_results.to_csv(save_path.replace('.txt', '_detailed.csv'), 
                                  index=False, encoding='utf-8-sig')
            
            # 保存分析报告
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("Y染色体检测成功概率分析报告\n")
                f.write("=" * 60 + "\n")
                f.write(f"检测成功阈值 (C_min): {self.c_min}\n")
                f.write(f"概率计算方法: {method}\n")
                f.write(f"基于混合效应模型: C_i(t) = (β₀ + u_i) + β₁·Week + β₂·BMI + ε\n")
                f.write(f"模型参数:\n")
                f.write(f"  β₀ (固定截距): {self.beta_0:.6f}\n")
                f.write(f"  β₁ (孕周系数): {self.beta_1:.6f}\n")
                f.write(f"  β₂ (BMI系数): {self.beta_2:.6f}\n")
                f.write(f"  σ_u (随机截距标准差): {self.random_intercept_std:.6f}\n")
                f.write(f"  σ_ε (残差标准差): {self.residual_std:.6f}\n\n")
                
                f.write("基本统计信息:\n")
                f.write(f"  平均成功概率: {analysis_results['overall_success_rate']:.4f}\n")
                f.write(f"  成功概率标准差: {analysis_results['success_rate_std']:.4f}\n")
                f.write(f"  最小成功概率: {analysis_data['成功概率'].min():.4f}\n")
                f.write(f"  最大成功概率: {analysis_data['成功概率'].max():.4f}\n\n")
                
                f.write("按BMI分组的成功概率分析:\n")
                f.write(analysis_results['bmi_analysis'].to_string())
                f.write("\n\n")
                
                f.write("按孕周分组的成功概率分析:\n")
                f.write(analysis_results['gestational_analysis'].to_string())
                f.write("\n\n")
                
                f.write("相关性分析:\n")
                f.write(f"  BMI与成功概率的相关系数: {analysis_results['bmi_correlation']:.4f}\n")
                f.write(f"  孕周与成功概率的相关系数: {analysis_results['gestational_correlation']:.4f}\n")
            
            print(f"分析报告已保存到: {save_path}")
            print(f"详细结果已保存到: {save_path.replace('.txt', '_detailed.csv')}")


def main():
    """主函数 - 演示成功概率计算器的使用"""
    # 初始化成功概率计算器
    calculator = SuccessProbabilityCalculator('./1/mixed_effects_coefs.csv', c_min=0.04)
    
    print("成功概率计算器初始化完成！")
    print(f"检测成功阈值: {calculator.c_min}")
    print()
    
    # 示例：计算特定BMI和孕周的成功概率
    example_bmi = 30.0
    example_gestational_week = 120
    
    print(f"示例计算:")
    print(f"BMI: {example_bmi}, 孕周: {example_gestational_week}")
    
    # 预测Y染色体浓度（平均值）
    predicted_concentration = calculator.predict_y_concentration_mean(example_bmi, example_gestational_week)
    print(f"预测Y染色体浓度（平均值）: {predicted_concentration:.6f}")
    
    # 计算成功概率
    success_prob = calculator.calculate_success_probability(example_bmi, example_gestational_week, method='normal')
    print(f"正态分布方法成功概率: {success_prob:.4f}")
    
    print()
    
    # 生成分析报告
    calculator.generate_success_probability_report(method='normal', save_path='./2/success_probability_report.txt')
    
    # 可视化成功概率
    calculator.visualize_success_probability(method='normal', save_path='./2/success_probability_heatmap.png')
    
    print("\n成功概率分析完成！")


if __name__ == "__main__":
    main()
