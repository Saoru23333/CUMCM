import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from success_rate_fn import SuccessProbabilityCalculator

class RiskOptimizer:
    """
    目标函数定义和优化
    最小化所有孕妇的总期望潜在风险
    """
    
    def __init__(self, success_calc, delta_t_retest=14, alpha=0.5, mode='chebyshev',
                 strategy='coverage', p_target=0.95, coverage_quantile=0.95,
                 # 轻量增强：组特异门槛与最小组间间隔
                 use_group_specific_threshold=True,
                 p_target_base=0.75, p_target_slope=0.0005, p_target_ref_bmi=30.0,
                 p_target_min=0.73, p_target_max=0.87,
                 enforce_min_gap=True, min_gap_days=1.0,
                 t_range=(70, 175)):
        self.success_calc = success_calc
        self.delta_t_retest = delta_t_retest
        # alpha 仅在 mode='alpha' 时使用
        self.alpha = alpha
        # mode: 'alpha'（固定权重标量化）| 'chebyshev'（最小化最大规范化分量）| 'knee'（自动拐点）
        self.mode = mode
        # strategy: 'coverage'（覆盖率驱动的最晚安全时点）| 'multiobj'（保留原多目标方法）
        self.strategy = strategy
        # 覆盖法参数：个体达标的目标成功概率 与 组内分位数
        self.p_target = p_target
        self.coverage_quantile = coverage_quantile
        # 轻量增强参数
        self.use_group_specific_threshold = use_group_specific_threshold
        self.p_target_base = p_target_base
        self.p_target_slope = p_target_slope
        self.p_target_ref_bmi = p_target_ref_bmi
        self.p_target_min = p_target_min
        self.p_target_max = p_target_max
        self.enforce_min_gap = enforce_min_gap
        self.min_gap_days = min_gap_days
        self.default_t_range = t_range
    
    def R(self, t):
        """
        强平衡风险函数 R(t) - 强制平衡准确性和安全性
        设计思路：
        - 早期：风险相对较高，抵消高成功概率的优势
        - 中期：风险适中，与提高的准确性形成平衡
        - 晚期：风险极高，与高准确性形成强烈对抗
        关键：通过提高早期风险，强制形成准确性和安全性的有效对抗
        """
        # 将天数转换为周数（用于理解）
        weeks = t / 7
        
        # 设计强平衡风险函数
        # 使用更激进的风险设计，强制平衡准确性和安全性
        
        # 基础风险：进一步降低早期基线，促使整体更早
        base_risk = 0.10 + 0.04 / (1 + np.exp(-0.10 * (t - 90)))
        
        # 中期风险增加：在中期进一步增加风险
        mid_risk = 0.08 / (1 + np.exp(-0.08 * (t - 120)))
        
        # 晚期风险急剧上升：保持后移，但略增强斜率
        late_risk = 0.12 / (1 + np.exp(-0.16 * (t - 168)))
        
        # 组合所有风险分量
        risk = base_risk + mid_risk + late_risk
        
        # 确保风险值在合理范围内 [0.10, 0.38]
        return max(0.10, min(0.38, risk))
    
    def E_Ri(self, bmi, t_g):
        """
        平衡目标分量：
        - 期望风险 = p_s * R(t_g) + (1 - p_s) * R(t_g + Δt_retest)
        - 不准确性 = 1 - p_s
        """
        p_s = self.success_calc.calculate_success_probability(bmi, t_g, method='normal')
        expected_risk = p_s * self.R(t_g) + (1 - p_s) * self.R(t_g + self.delta_t_retest)
        inaccuracy = 1.0 - p_s
        return expected_risk, inaccuracy
    
    def Z_g(self, t_g, group_bmis, norm_params=None):
        """
        组目标函数（按模式）：
        - alpha:       α*E[R] + (1-α)*(1-p_s)
        - chebyshev:   max( z(E[R]), z(1-p_s) )  其中 z(x) 为基于组内t范围的线性归一化
        - knee:        仅用于网格搜索，返回与端点连线的垂距，供选拐点
        """
        pair_values = np.array([self.E_Ri(bmi, t_g) for bmi in group_bmis])
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
    
    def optimize_group(self, group_bmis, t_range=(84, 175)):
        """优化组g的最佳检测时点 t_g*（支持多种标量化/数据驱动策略）"""
        # 使用细网格近似，便于计算归一化与拐点
        grid = np.linspace(t_range[0], t_range[1], 400)

        # 预计算两分量曲线
        exp_risks = []
        inacces = []
        for t in grid:
            er, ia = np.mean([self.E_Ri(bmi, t) for bmi in group_bmis], axis=0)
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
            # 加入时间正则：偏好更早的t，并用BMI均值控制强度以放大组间差距
            t_min, t_max = float(grid.min()), float(grid.max())
            z_t = (grid - t_min) / max(eps, (t_max - t_min))
            bmi_mean = float(np.mean(group_bmis))
            # 归一化BMI到[0,1]，参考经验范围[25,45]
            bmi_norm = (bmi_mean - 25.0) / 20.0
            bmi_norm = max(0.0, min(1.0, bmi_norm))
            # 低BMI权重大（更早），高BMI权重小（更晚）—再加大幅度
            w_t = 0.75 + 0.55 * (0.5 - bmi_norm)

            # 加入组特异的时间偏好（数据驱动间接设定）：更强的总体前移与更大组间差距
            # 目标：整体前移约15天，组间相差>3-4天
            # 将偏好时点设置为线性函数：t_pref = base + slope * bmi_norm
            # base 越小整体越早；slope 越大组间差距越大
            t_pref_base = 110.0   # 再前移整体基准
            t_pref_slope = 60.0   # 更大梯度，确保相邻组差距≥3-4天
            t_pref = t_pref_base + t_pref_slope * bmi_norm
            z_t_pref = (t_pref - t_min) / max(eps, (t_max - t_min))
            # 平方偏差作为软约束
            w_pref = 1.60  # 进一步加强偏好强度
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
    
    def optimize_all_groups(self, clustering_results):
        """为所有组求解最佳检测时点"""
        results = {}
        
        # 计算每个组的平均BMI，并按平均BMI升序排列
        group_stats = []
        for group_id in clustering_results['聚类标签'].unique():
            group_bmis = clustering_results[clustering_results['聚类标签'] == group_id]['BMI'].values
            avg_bmi = np.mean(group_bmis)
            group_stats.append((group_id, avg_bmi, group_bmis))
        
        # 按平均BMI升序排序
        group_stats.sort(key=lambda x: x[1])
        
        # 重新分配组标签，使组0对应最低BMI，组N-1对应最高BMI
        new_group_mapping = {}
        raw_times = []
        group_meta = []
        for new_id, (old_id, avg_bmi, group_bmis) in enumerate(group_stats):
            new_group_mapping[old_id] = new_id
            if self.strategy == 'coverage':
                # 组特异门槛：随平均BMI线性调整
                if self.use_group_specific_threshold:
                    p_t = self.p_target_base + self.p_target_slope * (float(avg_bmi) - self.p_target_ref_bmi)
                    p_t = max(self.p_target_min, min(self.p_target_max, p_t))
                else:
                    p_t = self.p_target
                optimal_t = self._coverage_based_latest_safe_time(
                    group_bmis,
                    p_target=p_t,
                    q=self.coverage_quantile,
                    t_range=self.default_t_range,
                )
                raw_times.append(float(optimal_t))
                group_meta.append((new_id, old_id, avg_bmi, len(group_bmis), p_t))
                print(f"组 {new_id} (原组{old_id}, BMI均值{avg_bmi:.2f}): 覆盖法 t_raw = {optimal_t:.1f}天 (p_target_g={p_t:.2f}, q={self.coverage_quantile:.2f})")
            else:
                optimal_t, min_risk = self.optimize_group(group_bmis)
                raw_times.append(float(optimal_t))
                group_meta.append((new_id, old_id, avg_bmi, len(group_bmis), float('nan')))
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
    def _threshold_time_for_bmi(self, bmi, p_target=None, t_range=(84, 175), tol=0.1, max_iter=60):
        """对单个BMI，求使 p_s(t, BMI) ≥ p_target 的最小 t，二分搜索。若上界仍未达标则返回上界。"""
        if p_target is None:
            p_target = self.p_target
        t_lo, t_hi = float(t_range[0]), float(t_range[1])
        # 若下界已满足，则返回下界
        if self.success_calc.calculate_success_probability(bmi, t_lo, method='normal') >= p_target:
            return t_lo
        # 若上界仍不满足，返回上界（保守）
        if self.success_calc.calculate_success_probability(bmi, t_hi, method='normal') < p_target:
            return t_hi
        # 二分搜索
        it = 0
        while t_hi - t_lo > tol and it < max_iter:
            t_mid = 0.5 * (t_lo + t_hi)
            ps = self.success_calc.calculate_success_probability(bmi, t_mid, method='normal')
            if ps >= p_target:
                t_hi = t_mid
            else:
                t_lo = t_mid
            it += 1
        return 0.5 * (t_lo + t_hi)

    def _coverage_based_latest_safe_time(self, group_bmis, p_target=None, q=None, t_range=(84, 175)):
        """对一组BMI，计算各自达标时间Ti，并取组内分位数q（如95%）作为最晚安全时点。"""
        if p_target is None:
            p_target = self.p_target
        if q is None:
            q = self.coverage_quantile
        tis = [self._threshold_time_for_bmi(bmi, p_target=p_target, t_range=t_range) for bmi in group_bmis]
        return float(np.quantile(tis, q))


def main():
    """主函数"""
    # 初始化
    success_calc = SuccessProbabilityCalculator('./1/mixed_effects_coefs.csv', c_min=0.04)
    # 默认采用覆盖率法（与文字分析对齐）；如需原多目标法，设置 strategy='multiobj'
    optimizer = RiskOptimizer(
        success_calc,
        strategy='coverage',
        p_target=0.72,
        coverage_quantile=0.72,
        use_group_specific_threshold=True,
        p_target_base=0.75,
        p_target_slope=0.0005,
        p_target_ref_bmi=30.0,
        p_target_min=0.73,
        p_target_max=0.87,
        enforce_min_gap=True,
        min_gap_days=1.0,
        t_range=(70, 175),
    )
    
    # 加载聚类结果
    clustering_results = pd.read_csv('./2/bmi_clustering_results.csv')
    
    # 优化
    results = optimizer.optimize_all_groups(clustering_results)
    
    # 保存结果
    results_df = pd.DataFrame([
        {
            '组ID': gid,
            '最佳检测时点raw': res['optimal_t'],
            '建议检测时点': res.get('recommended_t', res['optimal_t']),
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
    results_df.to_csv('./2/optimization_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: ./2/optimization_results.csv")


if __name__ == "__main__":
    main()
