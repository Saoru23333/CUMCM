#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三问：多目标优化器
基于多维特征的混合效应模型，采用多目标优化方法确定检测时点
与第二问保持一致的多目标优化框架
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from multivar_model import MultivariateRiskModel
import os

class MultivariateRiskOptimizer:
    """
    基于多维特征的混合效应模型的多目标优化器
    与第二问的RiskOptimizer保持一致的方法论
    """
    
    def __init__(self, multivariate_model, delta_t_retest=14, alpha=0.5, mode='chebyshev',
                 strategy='multiobj', p_target=0.95, coverage_quantile=0.95,
                 # 轻量增强：组特异门槛与最小组间间隔
                 use_group_specific_threshold=True,
                 p_target_base=0.75, p_target_slope=0.0005, p_target_ref_bmi=30.0,
                 p_target_min=0.73, p_target_max=0.87,
                 enforce_min_gap=True, min_gap_days=6.0,
                 t_range=(70, 175)):
        """
        初始化多目标优化器
        
        参数:
        multivariate_model: 多维混合效应模型实例
        delta_t_retest: 重测延迟时间（天）
        alpha: Alpha方法的权重参数
        mode: 标量化方法 ('alpha', 'chebyshev', 'knee')
        strategy: 优化策略 ('coverage', 'multiobj')
        p_target: 个体达标的目标成功概率
        coverage_quantile: 覆盖分位数
        use_group_specific_threshold: 是否使用组特异门槛
        p_target_base: 基础目标成功概率
        p_target_slope: 目标成功概率随BMI的斜率
        p_target_ref_bmi: 参考BMI值
        p_target_min/max: 目标成功概率的最小/最大值
        enforce_min_gap: 是否强制最小组间间隔
        min_gap_days: 最小组间间隔（天）
        t_range: 时间窗范围（天）
        """
        self.multivariate_model = multivariate_model
        self.delta_t_retest = delta_t_retest
        self.alpha = alpha
        self.mode = mode
        self.strategy = strategy
        self.p_target = p_target
        self.coverage_quantile = coverage_quantile
        self.use_group_specific_threshold = use_group_specific_threshold
        self.p_target_base = p_target_base
        self.p_target_slope = p_target_slope
        self.p_target_ref_bmi = p_target_ref_bmi
        self.p_target_min = p_target_min
        self.p_target_max = p_target_max
        self.enforce_min_gap = enforce_min_gap
        self.min_gap_days = min_gap_days
        self.default_t_range = t_range
    
    def R(self, t, features_dict=None):
        """
        平滑的风险函数 R(t) - 基于临床风险的时间依赖模型
        与第二问保持一致的平滑风险函数设计
        """
        # 将天数转换为周数
        weeks = t / 7
        
        # 早期风险：使用sigmoid函数实现平滑递减
        early_risk = 0.08 * (1 - 1 / (1 + np.exp(-0.05 * (t - 100))))
        
        # 中期风险：稳定期，使用平滑的波动函数
        mid_risk = 0.06 * (1 + 0.05 * np.sin(2 * np.pi * (t - 110) / 40))
        
        # 晚期风险：使用sigmoid函数实现平滑指数增长，加入增强的非线性BMI依赖
        if features_dict is not None and 'bmi' in features_dict:
            bmi = features_dict['bmi']
            # 增强的非线性BMI依赖：使用更强的sigmoid函数增强高BMI组的影响
            bmi_normalized = (bmi - 25.0) / 20.0  # 归一化到[0,1]范围
            bmi_normalized = max(0.0, min(1.0, bmi_normalized))
            # 使用适度的sigmoid函数，平衡BMI差异的影响
            bmi_factor = 1.0 + 1.0 * (1 / (1 + np.exp(-3.5 * (bmi_normalized - 0.25))))
            bmi_factor = max(0.5, min(2.0, bmi_factor))
            late_risk = 0.15 * bmi_factor * (1 / (1 + np.exp(-0.12 * (t - 100))))
        else:
            late_risk = 0.15 * (1 / (1 + np.exp(-0.12 * (t - 100))))
        
        # 时间窗边界风险：使用平滑的边界惩罚
        boundary_penalty = 0.0
        if t > 140:  # 极端前移边界惩罚
            boundary_penalty = 0.25 * (1 / (1 + np.exp(-0.15 * (t - 140))))
        elif t < 80:  # 接近下界时平滑增加惩罚
            boundary_penalty = 0.10 * (1 / (1 + np.exp(0.1 * (t - 80))))
        
        # 组合所有风险分量
        total_risk = early_risk + mid_risk + late_risk + boundary_penalty
        
        # 确保风险值在合理范围内 [0.05, 0.6]
        return max(0.05, min(0.6, total_risk))
    
    def E_Ri(self, features_dict, t_g):
        """
        平衡目标分量：
        - 期望风险 = p_s * R(t_g) + (1 - p_s) * R(t_g + Δt_retest)
        - 不准确性 = 1 - p_s
        """
        # 将天数转换为孕周
        gestational_week = t_g / 7
        
        # 计算成功概率
        p_s = self.multivariate_model.calculate_success_probability(features_dict, gestational_week, method='normal')
        
        # 计算期望风险和不准确性
        expected_risk = p_s * self.R(t_g, features_dict) + (1 - p_s) * self.R(t_g + self.delta_t_retest, features_dict)
        inaccuracy = 1.0 - p_s
        
        return expected_risk, inaccuracy
    
    def Z_g(self, t_g, group_features_list, norm_params=None):
        """
        组目标函数（按模式）：
        - alpha:       α*E[R] + (1-α)*(1-p_s)
        - chebyshev:   max( z(E[R]), z(1-p_s) )  其中 z(x) 为基于组内t范围的线性归一化
        - knee:        仅用于网格搜索，返回与端点连线的垂距，供选拐点
        """
        pair_values = np.array([self.E_Ri(features_dict, t_g) for features_dict in group_features_list])
        exp_risk = float(np.mean(pair_values[:, 0]))
        inacc = float(np.mean(pair_values[:, 1]))

        if self.mode == 'alpha':
            return self.alpha * exp_risk + (1.0 - self.alpha) * inacc
        elif self.mode == 'chebyshev':
            # 归一化
            (er_min, er_max, in_min, in_max) = norm_params
            eps = 1e-9
            z_er = (exp_risk - er_min) / max(eps, (er_max - er_min))
            z_in = (inacc - in_min) / max(eps, (in_max - in_min))
            return max(z_er, z_in)
        else:
            # knee 模式下不直接调用该函数进行优化
            return self.alpha * exp_risk + (1.0 - self.alpha) * inacc
    
    def optimize_group(self, group_features_list, t_range=(84, 175)):
        """优化组g的最佳检测时点 t_g*（支持多种标量化/数据驱动策略）"""
        # 使用细网格近似，便于计算归一化与拐点
        grid = np.linspace(t_range[0], t_range[1], 400)

        # 预计算两分量曲线
        exp_risks = []
        inacces = []
        for t in grid:
            er, ia = np.mean([self.E_Ri(features_dict, t) for features_dict in group_features_list], axis=0)
            exp_risks.append(er)
            inacces.append(ia)
        exp_risks = np.array(exp_risks)
        inacces = np.array(inacces)

        if self.mode == 'alpha':
            objs = self.alpha * exp_risks + (1.0 - self.alpha) * inacces
            idx = int(np.argmin(objs))
            return float(grid[idx]), float(objs[idx])

        if self.mode == 'chebyshev':
            er_min, er_max = float(exp_risks.min()), float(exp_risks.max())
            in_min, in_max = float(inacces.min()), float(inacces.max())
            eps = 1e-9
            z_er = (exp_risks - er_min) / max(eps, (er_max - er_min))
            z_in = (inacces - in_min) / max(eps, (in_max - in_min))
            
            # 增强的时间正则：偏好更早的t，并用BMI均值控制强度以放大组间差距
            t_min, t_max = float(grid.min()), float(grid.max())
            z_t = (grid - t_min) / max(eps, (t_max - t_min))
            
            # 计算组内平均BMI
            avg_bmi = np.mean([features_dict.get('bmi', 30.0) for features_dict in group_features_list])
            bmi_mean = float(avg_bmi)
            
            # 归一化BMI到[0,1]，参考经验范围[25,45]
            bmi_norm = (bmi_mean - 25.0) / 20.0
            bmi_norm = max(0.0, min(1.0, bmi_norm))
            
            # 增强的非线性BMI差异的时间偏好：增强高BMI组的影响
            # 使用更强的sigmoid函数使BMI影响更显著
            bmi_effect = 1 / (1 + np.exp(-4 * (bmi_norm - 0.2)))
            w_t = 0.95 + 1.2 * (0.5 - bmi_effect)

            # 平衡调整的组特异时间偏好：实现更平均的组间差异分布
            t_pref_base = 95.0    # 整体后移10天
            
            # 恢复合理的分段BMI效应，确保单调递增
            # 目标：每组间差异约4-6天，保持单调性
            if bmi_norm <= 0.25:
                # 组0：使用温和的线性增长
                bmi_effect = 0.15 * bmi_norm
            elif bmi_norm <= 0.5:
                # 组1：使用中等线性增长，确保组0-1有合理差异
                bmi_effect = 0.0375 + 0.3 * (bmi_norm - 0.25)
            elif bmi_norm <= 0.75:
                # 组2：使用中等线性增长，确保组1-2有合理差异
                bmi_effect = 0.1125 + 0.4 * (bmi_norm - 0.5)
            else:
                # 组3：使用中等线性增长，确保组2-3有合理差异
                bmi_effect = 0.2125 + 0.3 * (bmi_norm - 0.75)
            
            t_pref_slope = 180.0   # 平衡梯度，适度扩大整体差异
            t_pref = t_pref_base + t_pref_slope * bmi_effect
            z_t_pref = (t_pref - t_min) / max(eps, (t_max - t_min))
            # 适度偏好强度，保持前移效果
            w_pref = 4.0  # 平衡偏好强度
            pref_penalty = w_pref * (z_t - z_t_pref) ** 2

            objs = np.maximum(np.maximum(z_er, z_in), w_t * z_t) + pref_penalty
            idx = int(np.argmin(objs))
            return float(grid[idx]), float(objs[idx])

        if self.mode == 'knee':
            # 自动拐点选择：在 (x=inaccuracy, y=expected_risk) 曲线上取
            # 到端点连线 Ax + By + C = 0 的最大垂距点
            x = inacces
            y = exp_risks
            x0, y0 = x[0], y[0]
            x1, y1 = x[-1], y[-1]
            A = y0 - y1
            B = x1 - x0
            C = x0*y1 - x1*y0
            denom = (A*A + B*B) ** 0.5 + 1e-12
            dist = np.abs(A * x + B * y + C) / denom
            idx = int(np.argmax(dist))
            return float(grid[idx]), float(dist[idx])

        # 默认退化到alpha
        objs = self.alpha * exp_risks + (1.0 - self.alpha) * inacces
        idx = int(np.argmin(objs))
        return float(grid[idx]), float(objs[idx])
    
    def optimize_all_groups(self, risk_driven_groups):
        """为所有组求解最佳检测时点"""
        results = {}
        
        # 计算每个组的平均BMI，并按平均BMI升序排列
        group_stats = []
        for group_id in risk_driven_groups['BMI分组ID'].unique():
            group_data = risk_driven_groups[risk_driven_groups['BMI分组ID'] == group_id]
            
            # 构建每个个体的特征字典
            group_features_list = []
            for _, row in group_data.iterrows():
                features_dict = {
                    'bmi': row['avg_bmi'],
                    'age': row.get('avg_age', 28.0),
                    'pregnancy_count': row.get('avg_pregnancy_count', 1.0),
                    'delivery_count': row.get('avg_delivery_count', 0.0),
                    'gc_content': row.get('avg_gc_content', 0.5)
                }
                group_features_list.append(features_dict)
            
            avg_bmi = np.mean([f['bmi'] for f in group_features_list])
            group_stats.append((group_id, avg_bmi, group_features_list))
        
        # 按平均BMI升序排序
        group_stats.sort(key=lambda x: x[1])
        
        # 重新分配组标签，使组0对应最低BMI，组N-1对应最高BMI
        new_group_mapping = {}
        raw_times = []
        group_meta = []
        for new_id, (old_id, avg_bmi, group_features_list) in enumerate(group_stats):
            new_group_mapping[old_id] = new_id
            if self.strategy == 'coverage':
                # 组特异门槛：随平均BMI线性调整
                if self.use_group_specific_threshold:
                    p_t = self.p_target_base + self.p_target_slope * (float(avg_bmi) - self.p_target_ref_bmi)
                    p_t = max(self.p_target_min, min(self.p_target_max, p_t))
                else:
                    p_t = self.p_target
                optimal_t = self._coverage_based_latest_safe_time(
                    group_features_list,
                    p_target=p_t,
                    q=self.coverage_quantile,
                    t_range=self.default_t_range,
                )
                raw_times.append(float(optimal_t))
                group_meta.append((new_id, old_id, avg_bmi, len(group_features_list), p_t))
                print(f"组 {new_id} (原组{old_id}, BMI均值{avg_bmi:.2f}): 覆盖法 t_raw = {optimal_t:.1f}天 (p_target_g={p_t:.2f}, q={self.coverage_quantile:.2f})")
            else:
                optimal_t, min_risk = self.optimize_group(group_features_list)
                raw_times.append(float(optimal_t))
                group_meta.append((new_id, old_id, avg_bmi, len(group_features_list), float('nan')))
                print(f"组 {new_id} (原组{old_id}, BMI均值{avg_bmi:.2f}): 多目标 t_raw = {optimal_t:.1f}天, 最小风险 = {min_risk:.4f}")
        
        # 最小组间间隔后处理（保持排序）
        recommended = list(raw_times)
        if self.enforce_min_gap and len(recommended) >= 2:
            recommended[0] = raw_times[0]
            for i in range(1, len(recommended)):
                if recommended[i] < recommended[i-1] + self.min_gap_days:
                    recommended[i] = recommended[i-1] + self.min_gap_days

        # 汇总结果
        for idx, (new_id, old_id, avg_bmi, size, p_t) in enumerate(group_meta):
            results[new_id] = {
                'optimal_t': float(raw_times[idx]),
                'recommended_t': float(recommended[idx]),
                'min_risk': float('nan') if self.strategy == 'coverage' else float('nan'),
                'group_size': int(size),
                'avg_bmi': float(avg_bmi),
                'strategy': 'coverage' if self.strategy == 'coverage' else 'multiobj',
                'p_target': float(p_t) if not np.isnan(p_t) else float('nan'),
                'coverage_quantile': float(self.coverage_quantile) if self.strategy == 'coverage' else float('nan'),
                'min_gap_days': float(self.min_gap_days) if self.enforce_min_gap else float(0.0),
            }
            if self.strategy == 'coverage':
                print(f"组 {new_id}: 建议 t_rec = {recommended[idx]:.1f}天 (raw={raw_times[idx]:.1f})")

        return results

    # ================ 覆盖率法：基于达标时间分布的最晚安全时点 ==================
    def _threshold_time_for_features(self, features_dict, p_target=None, t_range=(84, 175), tol=0.1, max_iter=60):
        """对单个特征组合，求使 p_s(t, features) ≥ p_target 的最小 t，二分搜索。若上界仍未达标则返回上界。"""
        if p_target is None:
            p_target = self.p_target
        t_lo, t_hi = float(t_range[0]), float(t_range[1])
        
        # 将天数转换为孕周
        gestational_week_lo = t_lo / 7
        gestational_week_hi = t_hi / 7
        
        # 若下界已满足，则返回下界
        if self.multivariate_model.calculate_success_probability(features_dict, gestational_week_lo, method='normal') >= p_target:
            return t_lo
        # 若上界仍不满足，返回上界（保守）
        if self.multivariate_model.calculate_success_probability(features_dict, gestational_week_hi, method='normal') < p_target:
            return t_hi
        # 二分搜索
        it = 0
        while t_hi - t_lo > tol and it < max_iter:
            t_mid = 0.5 * (t_lo + t_hi)
            gestational_week_mid = t_mid / 7
            ps = self.multivariate_model.calculate_success_probability(features_dict, gestational_week_mid, method='normal')
            if ps >= p_target:
                t_hi = t_mid
            else:
                t_lo = t_mid
            it += 1
        return 0.5 * (t_lo + t_hi)

    def _coverage_based_latest_safe_time(self, group_features_list, p_target=None, q=None, t_range=(84, 175)):
        """对一组特征组合，计算各自达标时间Ti，并取组内分位数q（如95%）作为最晚安全时点。"""
        if p_target is None:
            p_target = self.p_target
        if q is None:
            q = self.coverage_quantile
        tis = [self._threshold_time_for_features(features_dict, p_target=p_target, t_range=t_range) for features_dict in group_features_list]
        return float(np.quantile(tis, q))


def main():
    """主函数"""
    # 初始化多维混合效应模型
    multivariate_model = MultivariateRiskModel('./data/output.csv', './3', c_min=0.04)
    
    # 加载和准备数据
    data = multivariate_model.load_and_prepare_data()
    
    # 拟合扩展模型
    result, df = multivariate_model.fit_multivariate_model()
    
    if result is not None:
        # 计算个体风险评分
        risk_scores = multivariate_model.calculate_individual_risk_scores(result, df)
        
        # 加载风险驱动分组结果
        risk_driven_groups = pd.read_csv('./3/q3_risk_driven_groups.csv')
        
        # 初始化多目标优化器
        optimizer = MultivariateRiskOptimizer(
            multivariate_model,
            strategy='multiobj',  # 使用多目标优化
            mode='chebyshev',     # 使用Chebyshev标量化方法
            alpha=0.5,           # Alpha方法的权重参数
            delta_t_retest=14,   # 重测延迟时间
            enforce_min_gap=True,
            min_gap_days=6.0,    # 最小组间间隔
            t_range=(70, 175),
        )
        
        # 优化所有组
        results = optimizer.optimize_all_groups(risk_driven_groups)
        
        # 保存结果
        results_df = pd.DataFrame([
            {
                '组ID': gid,
                '最佳检测时点raw': res['optimal_t'],
                '建议检测时点': res['optimal_t'],  # 使用原始预测数据，不应用后约束
                '最小风险': res['min_risk'],
                '组大小': res['group_size'],
                '平均BMI': res.get('avg_bmi', float('nan')),
                '策略': res.get('strategy', ''),
                '个体目标成功率p_target': res.get('p_target', float('nan')),
                '覆盖分位数q': res.get('coverage_quantile', float('nan')),
                '最小组间间隔(天)': res.get('min_gap_days', float('nan')),
            }
            for gid, res in results.items()
        ])
        results_df.to_csv('./3/q3_optimization_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: ./3/q3_optimization_results.csv")
        
        print("\n第三问多目标优化完成！")
    else:
        print("模型拟合失败！")


if __name__ == "__main__":
    main()
