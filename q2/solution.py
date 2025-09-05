import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from success_rate_fn import SuccessProbabilityCalculator

class RiskOptimizer:
    """
    目标函数定义和优化
    最小化所有孕妇的总期望潜在风险
    """
    
    def __init__(self, success_calc, delta_t_retest=14):
        self.success_calc = success_calc
        self.delta_t_retest = delta_t_retest
    
    def R(self, t):
        """
        风险函数 R(t) - 使用sigmoid函数
        考虑早期检测的准确性问题，后期检测的紧迫性
        使用sigmoid函数实现平滑的风险变化
        """
        # Sigmoid函数参数
        # 早期风险较高，中期风险最低，后期风险快速上升
        # 使用两个sigmoid函数的组合来模拟这种变化
        
        # 第一个sigmoid：从早期高风险到中期低风险
        # 在t=120左右达到最低点
        early_to_mid = 0.15 - 0.04 / (1 + np.exp(-0.1 * (t - 120)))
        
        # 第二个sigmoid：从中期低风险到后期高风险
        # 在t=140之后开始快速上升
        mid_to_late = 0.02 / (1 + np.exp(-0.15 * (t - 140)))
        
        # 组合两个sigmoid函数
        risk = early_to_mid + mid_to_late
        
        # 确保风险值在合理范围内
        return max(0.05, min(0.25, risk))
    
    def E_Ri(self, bmi, t_g):
        """
        期望风险 E[R_i(t_g)]
        = p_s(t_g, X_i) * R(t_g) + (1 - p_s(t_g, X_i)) * R(t_g + Δt_retest)
        """
        p_s = self.success_calc.calculate_success_probability(bmi, t_g, method='normal')
        return p_s * self.R(t_g) + (1 - p_s) * self.R(t_g + self.delta_t_retest)
    
    def Z_g(self, t_g, group_bmis):
        """
        组目标函数 Z_g(t_g) = (1/|I_g|) * Σ[E[R_i(t_g)]]
        """
        return np.mean([self.E_Ri(bmi, t_g) for bmi in group_bmis])
    
    def optimize_group(self, group_bmis, t_range=(70, 175)):
        """优化组g的最佳检测时点 t_g*"""
        result = minimize_scalar(
            lambda t: self.Z_g(t, group_bmis),
            bounds=t_range,
            method='bounded'
        )
        return result.x, result.fun
    
    def optimize_all_groups(self, clustering_results):
        """为所有组求解最佳检测时点"""
        results = {}
        for group_id in sorted(clustering_results['聚类标签'].unique()):
            group_bmis = clustering_results[clustering_results['聚类标签'] == group_id]['BMI'].values
            optimal_t, min_risk = self.optimize_group(group_bmis)
            results[group_id] = {'optimal_t': optimal_t, 'min_risk': min_risk, 'group_size': len(group_bmis)}
            print(f"组 {group_id}: t* = {optimal_t:.1f}天, 最小风险 = {min_risk:.4f}")
        return results


def main():
    """主函数"""
    # 初始化
    success_calc = SuccessProbabilityCalculator('./data/output.csv', c_min=0.08)
    optimizer = RiskOptimizer(success_calc)
    
    # 加载聚类结果
    clustering_results = pd.read_csv('./2/bmi_clustering_results.csv')
    
    # 优化
    results = optimizer.optimize_all_groups(clustering_results)
    
    # 保存结果
    results_df = pd.DataFrame([
        {'组ID': gid, '最佳检测时点': res['optimal_t'], '最小风险': res['min_risk'], '组大小': res['group_size']}
        for gid, res in results.items()
    ])
    results_df.to_csv('./2/optimization_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: ./2/optimization_results.csv")


if __name__ == "__main__":
    main()
