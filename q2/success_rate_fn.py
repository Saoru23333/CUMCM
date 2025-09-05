import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import UnivariateSpline
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SuccessProbabilityCalculator:
    """
    成功概率计算器
    计算 p_s(t, X_i) = P(C_i(t) >= C_min)
    使用原始数据 + 插值方法
    """
    
    def __init__(self, original_data_path, c_min=0.08):
        """
        初始化成功概率计算器
        
        参数:
        original_data_path: 原始数据文件路径
        c_min: 检测成功的最低Y染色体浓度阈值，默认0.08
        """
        self.c_min = c_min
        self.original_data_path = original_data_path
        self.original_data = pd.read_csv(original_data_path)
        
        # 重命名列以便使用
        self.original_data = self.original_data.rename(columns={
            '孕妇BMI': 'BMI',
            '检测孕周': '孕周',
            'Y染色体浓度': 'Y染色体浓度'
        })
        
        print(f"加载原始数据: {len(self.original_data)} 条记录")
        print(f"BMI范围: {self.original_data['BMI'].min():.2f} - {self.original_data['BMI'].max():.2f}")
        print(f"孕周范围: {self.original_data['孕周'].min()} - {self.original_data['孕周'].max()}")
        print(f"Y染色体浓度范围: {self.original_data['Y染色体浓度'].min():.4f} - {self.original_data['Y染色体浓度'].max():.4f}")
        
        # 预计算残差标准差，避免重复计算
        self._precompute_residual_std()
    
    def _precompute_residual_std(self):
        """预计算残差标准差"""
        print("正在预计算残差标准差...")
        
        # 使用简化的方法：直接使用原始数据的标准差作为残差估计
        # 这样可以避免复杂的插值计算
        actual_concentrations = self.original_data['Y染色体浓度'].values
        
        # 计算Y染色体浓度的标准差作为残差估计
        self.residual_std = np.std(actual_concentrations)
        
        print(f"残差标准差: {self.residual_std:.6f}")
    
    def predict_y_concentration(self, bmi, gestational_week):
        """
        基于原始数据预测Y染色体浓度
        使用加权距离插值方法
        
        参数:
        bmi: BMI值
        gestational_week: 孕周
        
        返回:
        预测的Y染色体浓度
        """
        return self._interpolate_from_original_data(bmi, gestational_week)
    
    def _interpolate_from_original_data(self, bmi, gestational_week):
        """从原始数据中插值预测Y染色体浓度"""
        # 计算到所有数据点的距离
        bmi_diff = np.abs(self.original_data['BMI'] - bmi)
        gestational_diff = np.abs(self.original_data['孕周'] - gestational_week)
        
        # 计算综合距离，BMI和孕周使用不同的权重
        # BMI权重较大，因为BMI对Y染色体浓度的影响更显著
        total_diff = np.sqrt(bmi_diff**2 + (gestational_diff/5)**2)  # 孕周权重较小
        
        # 找到最近的几个点进行加权平均
        closest_indices = np.argsort(total_diff)[:10]  # 使用更多点提高稳定性
        weights = 1 / (total_diff[closest_indices] + 1e-6)
        weights = weights / np.sum(weights)
        
        predicted_concentration = np.sum(weights * self.original_data.iloc[closest_indices]['Y染色体浓度'])
        return predicted_concentration
    
    def calculate_success_probability(self, bmi, gestational_week, method='normal'):
        """
        计算成功概率 p_s(t, X_i) = P(C_i(t) >= C_min)
        只使用正态分布假设方法
        
        参数:
        bmi: BMI值
        gestational_week: 孕周
        method: 概率计算方法 (只支持 'normal')
        
        返回:
        成功概率值
        """
        if method == 'normal':
            # 基于GAM模型预测 + 正态分布假设
            predicted_concentration = self.predict_y_concentration(bmi, gestational_week)
            return self._calculate_normal_probability(predicted_concentration)
        else:
            raise ValueError("method参数只支持 'normal'")
    
    def _calculate_normal_probability(self, predicted_concentration):
        """基于原始数据计算成功概率"""
        # 使用预计算的残差标准差
        residual_std = self.residual_std
        
        # 计算成功概率 P(C >= C_min)
        z_score = (self.c_min - predicted_concentration) / residual_std
        success_prob = 1 - stats.norm.cdf(z_score)
        
        return max(0, min(1, success_prob))  # 确保概率在[0,1]范围内
    
    
    def _calculate_empirical_probability(self, predicted_concentration):
        """基于经验分布计算成功概率"""
        # 使用残差分布
        residuals = self.gam_results['残差']
        
        # 计算所有可能的浓度值
        all_concentrations = predicted_concentration + residuals
        
        # 计算成功概率
        success_count = np.sum(all_concentrations >= self.c_min)
        success_prob = success_count / len(all_concentrations)
        
        return success_prob
    
    def _calculate_bootstrap_probability(self, bmi, gestational_week, n_bootstrap=1000):
        """基于Bootstrap方法计算成功概率"""
        residuals = self.gam_results['残差']
        bootstrap_success_count = 0
        
        for _ in range(n_bootstrap):
            # 随机采样残差
            bootstrap_residual = np.random.choice(residuals)
            bootstrap_concentration = self.predict_y_concentration(bmi, gestational_week) + bootstrap_residual
            
            if bootstrap_concentration >= self.c_min:
                bootstrap_success_count += 1
        
        return bootstrap_success_count / n_bootstrap
    
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
        
        # 添加数据点
        scatter = plt.scatter(self.gam_results['BMI'], self.gam_results['孕周'], 
                            c=self.gam_results['预测Y染色体浓度'], 
                            cmap='viridis', alpha=0.6, s=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"成功概率热力图已保存到: {save_path}")
        
        plt.show()
        
        return bmi_grid, gestational_grid, success_prob_grid
    
    def analyze_success_probability_by_factors(self, method='normal'):
        """
        分析不同因素对成功概率的影响
        
        参数:
        method: 概率计算方法
        
        返回:
        分析结果字典
        """
        # 计算每个数据点的成功概率
        success_probs = []
        for _, row in self.gam_results.iterrows():
            prob = self.calculate_success_probability(row['BMI'], row['孕周'], method=method)
            success_probs.append(prob)
        
        self.gam_results['成功概率'] = success_probs
        
        # 按BMI分组分析
        bmi_bins = pd.cut(self.gam_results['BMI'], bins=5, labels=['很低', '低', '中等', '高', '很高'])
        bmi_analysis = self.gam_results.groupby(bmi_bins)['成功概率'].agg(['mean', 'std', 'count'])
        
        # 按孕周分组分析
        gestational_bins = pd.cut(self.gam_results['孕周'], bins=5, labels=['早期', '早中期', '中期', '中晚期', '晚期'])
        gestational_analysis = self.gam_results.groupby(gestational_bins)['成功概率'].agg(['mean', 'std', 'count'])
        
        # 相关性分析
        bmi_correlation = self.gam_results['BMI'].corr(self.gam_results['成功概率'])
        gestational_correlation = self.gam_results['孕周'].corr(self.gam_results['成功概率'])
        
        analysis_results = {
            'bmi_analysis': bmi_analysis,
            'gestational_analysis': gestational_analysis,
            'bmi_correlation': bmi_correlation,
            'gestational_correlation': gestational_correlation,
            'overall_success_rate': np.mean(success_probs),
            'success_rate_std': np.std(success_probs)
        }
        
        return analysis_results
    
    def generate_success_probability_report(self, method='normal', save_path=None):
        """
        生成成功概率分析报告
        
        参数:
        method: 概率计算方法
        save_path: 报告保存路径
        """
        print("=" * 60)
        print("Y染色体检测成功概率分析报告")
        print("=" * 60)
        print(f"检测成功阈值 (C_min): {self.c_min}")
        print(f"概率计算方法: {method}")
        print(f"数据点数量: {len(self.gam_results)}")
        print()
        
        # 计算成功概率
        success_probs = []
        for _, row in self.gam_results.iterrows():
            prob = self.calculate_success_probability(row['BMI'], row['孕周'], method=method)
            success_probs.append(prob)
        
        self.gam_results['成功概率'] = success_probs
        
        # 基本统计
        print("基本统计信息:")
        print(f"  平均成功概率: {np.mean(success_probs):.4f}")
        print(f"  成功概率标准差: {np.std(success_probs):.4f}")
        print(f"  最小成功概率: {np.min(success_probs):.4f}")
        print(f"  最大成功概率: {np.max(success_probs):.4f}")
        print()
        
        # 按BMI分析
        bmi_bins = pd.cut(self.gam_results['BMI'], bins=5, labels=['很低', '低', '中等', '高', '很高'])
        bmi_analysis = self.gam_results.groupby(bmi_bins)['成功概率'].agg(['mean', 'std', 'count'])
        
        print("按BMI分组的成功概率分析:")
        print(bmi_analysis)
        print()
        
        # 按孕周分析
        gestational_bins = pd.cut(self.gam_results['孕周'], bins=5, labels=['早期', '早中期', '中期', '中晚期', '晚期'])
        gestational_analysis = self.gam_results.groupby(gestational_bins)['成功概率'].agg(['mean', 'std', 'count'])
        
        print("按孕周分组的成功概率分析:")
        print(gestational_analysis)
        print()
        
        # 相关性分析
        bmi_correlation = self.gam_results['BMI'].corr(self.gam_results['成功概率'])
        gestational_correlation = self.gam_results['孕周'].corr(self.gam_results['成功概率'])
        
        print("相关性分析:")
        print(f"  BMI与成功概率的相关系数: {bmi_correlation:.4f}")
        print(f"  孕周与成功概率的相关系数: {gestational_correlation:.4f}")
        print()
        
        # 保存结果
        if save_path:
            # 保存详细结果
            detailed_results = self.gam_results[['BMI', '孕周', '预测Y染色体浓度', '成功概率']].copy()
            detailed_results.to_csv(save_path.replace('.txt', '_detailed.csv'), 
                                  index=False, encoding='utf-8-sig')
            
            # 保存分析报告
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("Y染色体检测成功概率分析报告\n")
                f.write("=" * 60 + "\n")
                f.write(f"检测成功阈值 (C_min): {self.c_min}\n")
                f.write(f"概率计算方法: {method}\n")
                f.write(f"数据点数量: {len(self.gam_results)}\n\n")
                
                f.write("基本统计信息:\n")
                f.write(f"  平均成功概率: {np.mean(success_probs):.4f}\n")
                f.write(f"  成功概率标准差: {np.std(success_probs):.4f}\n")
                f.write(f"  最小成功概率: {np.min(success_probs):.4f}\n")
                f.write(f"  最大成功概率: {np.max(success_probs):.4f}\n\n")
                
                f.write("按BMI分组的成功概率分析:\n")
                f.write(bmi_analysis.to_string())
                f.write("\n\n")
                
                f.write("按孕周分组的成功概率分析:\n")
                f.write(gestational_analysis.to_string())
                f.write("\n\n")
                
                f.write("相关性分析:\n")
                f.write(f"  BMI与成功概率的相关系数: {bmi_correlation:.4f}\n")
                f.write(f"  孕周与成功概率的相关系数: {gestational_correlation:.4f}\n")
            
            print(f"分析报告已保存到: {save_path}")
            print(f"详细结果已保存到: {save_path.replace('.txt', '_detailed.csv')}")


def main():
    """主函数 - 演示成功概率计算器的使用"""
    # 初始化成功概率计算器
    calculator = SuccessProbabilityCalculator('./data/output.csv', c_min=0.08)
    
    print("成功概率计算器初始化完成！")
    print(f"检测成功阈值: {calculator.c_min}")
    print()
    
    # 示例：计算特定BMI和孕周的成功概率
    example_bmi = 30.0
    example_gestational_week = 120
    
    print(f"示例计算:")
    print(f"BMI: {example_bmi}, 孕周: {example_gestational_week}")
    
    # 预测Y染色体浓度
    predicted_concentration = calculator.predict_y_concentration(example_bmi, example_gestational_week)
    print(f"预测Y染色体浓度: {predicted_concentration:.6f}")
    
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
