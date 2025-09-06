import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, List

from success_rate_fn import SuccessProbabilityCalculator
from solution import RiskOptimizer
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def run_one_simulation(
    base_clustering_df: pd.DataFrame,
    c_min: float,
    bmi_noise_std: float,
    random_state: int,
    optimizer_kwargs: Dict
) -> Dict[int, float]:
    """
    单次模拟：
    - 对BMI加入测量误差 N(0, bmi_noise_std)
    - 使用扰动后的BMI与给定 c_min 重跑覆盖率法优化
    - 返回每个组的推荐检测时点（保持按平均BMI排序后的新组编号）
    """
    rng = np.random.default_rng(random_state)

    # 拷贝并扰动 BMI
    df = base_clustering_df.copy()
    if 'BMI' not in df.columns:
        raise ValueError("聚类结果需包含 'BMI' 列")
    df['BMI'] = df['BMI'] + rng.normal(0.0, bmi_noise_std, size=len(df))

    # 读取混合效应模型并设置阈值
    success_calc = SuccessProbabilityCalculator('./1/mixed_effects_coefs.csv', c_min=c_min)

    # 使用覆盖率法优化（与现有主流程一致）
    optimizer = RiskOptimizer(success_calc, **optimizer_kwargs)
    results = optimizer.optimize_all_groups(df)
    return {gid: res.get('recommended_t', res['optimal_t']) for gid, res in results.items()}


def sensitivity_analysis(
    clustering_results_path: str = './2/bmi_clustering_results.csv',
    out_csv_path: str = './2/optimization_sensitivity.csv',
    n_runs: int = 200,
    bmi_noise_std_list: List[float] = [0.0, 0.2, 0.5, 1.0],
    c_min_list: List[float] = [0.035, 0.04, 0.045],
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    对检测误差进行敏感性分析：
    - BMI 测量误差：对输入 BMI 加性高斯噪声
    - 阈值 c_min 不确定性：在一组候选 c_min 上评估
    输出每个 (c_min, bmi_noise_std, 组) 的推荐时点均值与标准差。
    """
    if not os.path.exists(clustering_results_path):
        raise FileNotFoundError(f"未找到聚类结果文件: {clustering_results_path}")

    base_df = pd.read_csv(clustering_results_path)

    # 与 solution.py 主流程保持一致的覆盖率法参数
    optimizer_kwargs = dict(
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

    records = []
    rng = np.random.default_rng(random_seed)

    for c_min in c_min_list:
        for bmi_std in bmi_noise_std_list:
            group_to_ts: Dict[int, List[float]] = {}

            for run_idx in range(n_runs):
                rs = int(rng.integers(0, 1_000_000_000))
                rec = run_one_simulation(
                    base_clustering_df=base_df,
                    c_min=c_min,
                    bmi_noise_std=bmi_std,
                    random_state=rs,
                    optimizer_kwargs=optimizer_kwargs,
                )
                for gid, t_rec in rec.items():
                    group_to_ts.setdefault(gid, []).append(float(t_rec))

            # 汇总: 每个组的均值/标准差
            for gid, ts in sorted(group_to_ts.items()):
                ts_arr = np.asarray(ts, dtype=float)
                records.append({
                    'c_min': c_min,
                    'bmi_noise_std': bmi_std,
                    'group_id': gid,
                    'recommended_t_mean': float(np.mean(ts_arr)),
                    'recommended_t_std': float(np.std(ts_arr, ddof=1)) if len(ts_arr) > 1 else 0.0,
                    'recommended_t_min': float(np.min(ts_arr)),
                    'recommended_t_max': float(np.max(ts_arr)),
                    'n_runs': int(len(ts_arr)),
                })

    result_df = pd.DataFrame.from_records(records)
    result_df.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
    print(f"敏感性结果已保存到: {out_csv_path}")
    return result_df


def main():
    # 运行敏感性分析并绘制简洁的误差-波动关系图
    df = sensitivity_analysis(
        clustering_results_path='./2/bmi_clustering_results.csv',
        out_csv_path='./2/optimization_sensitivity.csv',
        n_runs=150,
        bmi_noise_std_list=[0.0, 0.2, 0.5, 1.0],
        c_min_list=[0.035, 0.04, 0.045],
        random_seed=2025,
    )

    # 聚合：按 c_min 与 bmi_noise_std 取所有组的推荐时点STD的中位数，作为稳健性指标
    agg = (
        df.groupby(['c_min', 'bmi_noise_std'])['recommended_t_std']
          .median()
          .reset_index()
    )

    # 创建三个子图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: 原始误差-波动关系图
    for c in sorted(agg['c_min'].unique()):
        sub = agg[agg['c_min'] == c]
        axes[0].plot(sub['bmi_noise_std'], sub['recommended_t_std'], marker='o', label=f'c_min={c:.3f}')
    axes[0].set_xlabel('BMI测量误差标准差')
    axes[0].set_ylabel('建议检测时点标准差（组内中位数）')
    axes[0].set_title('检测误差对推荐时点波动的影响')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    # 图2: c_min vs recommended_t_mean (按BMI分组)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 不同颜色代表不同组
    for group_id in sorted(df['group_id'].unique()):
        group_data = df[df['group_id'] == group_id]
        # 对每个bmi_noise_std取均值
        group_agg = group_data.groupby('c_min')['recommended_t_mean'].mean().reset_index()
        axes[1].plot(group_agg['c_min'], group_agg['recommended_t_mean'], 
                    marker='o', color=colors[group_id], label=f'组{group_id}')
    axes[1].set_xlabel('检测阈值 c_min')
    axes[1].set_ylabel('建议检测时点均值')
    axes[1].set_title('不同阈值下各组的推荐检测时点')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    # 图3: bmi_noise_std vs recommended_t_mean (带误差棒)
    for group_id in sorted(df['group_id'].unique()):
        group_data = df[df['group_id'] == group_id]
        # 对每个bmi_noise_std取均值和标准差
        group_agg = group_data.groupby('bmi_noise_std').agg({
            'recommended_t_mean': 'mean',
            'recommended_t_std': 'mean'
        }).reset_index()
        axes[2].errorbar(group_agg['bmi_noise_std'], group_agg['recommended_t_mean'], 
                        yerr=group_agg['recommended_t_std'], 
                        marker='o', color=colors[group_id], 
                        label=f'组{group_id}', capsize=5)
    axes[2].set_xlabel('BMI测量误差标准差')
    axes[2].set_ylabel('建议检测时点均值')
    axes[2].set_title('BMI测量误差对推荐时点的影响（带误差棒）')
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    out_png = './2/optimization_sensitivity.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"敏感性分析图已保存到: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()


