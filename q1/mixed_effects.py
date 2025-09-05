import pandas as pd
import numpy as np
import os
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

def fit_mixed_effects_model(data, save_dir):
    """
    拟合带有随机截距的混合效应模型
    模型形式：Y_ij = (β₀ + u_i) + β₁·Week_ij + β₂·BMI_ij + ε_ij
    
    其中：
    - Y_ij: 第i位孕妇第j次检测的Y染色体浓度
    - β₀: 固定截距（全体平均基线）
    - u_i: 第i位孕妇的随机截距（个体差异）
    - β₁: 孕周的固定效应系数
    - β₂: BMI的固定效应系数
    - ε_ij: 残差项
    """
    print("正在拟合混合效应模型...")
    
    # 准备数据
    df = data[['孕妇代码', '检测孕周', '孕妇BMI', 'Y染色体浓度']].copy()
    df = df.dropna()
    
    # 重命名列以便在公式中使用
    df = df.rename(columns={
        '孕妇代码': 'subject_id',
        '检测孕周': 'week',
        '孕妇BMI': 'bmi',
        'Y染色体浓度': 'y_concentration'
    })
    
    # 拟合混合效应模型
    # 使用孕妇代码作为分组变量，随机截距模型
    formula = 'y_concentration ~ week + bmi'
    groups = df['subject_id']
    
    try:
        # 拟合模型
        model = mixedlm(formula, df, groups=groups, re_formula='1')
        result = model.fit()
        
        print("混合效应模型拟合成功！")
        
        # 保存模型结果
        save_mixed_effects_results(result, df, save_dir)
        
        return result, df
        
    except Exception as e:
        print(f"模型拟合失败: {e}")
        return None, None

def save_mixed_effects_results(result, df, save_dir):
    """
    保存混合效应模型结果
    """
    # 1. 保存预测结果
    predictions = result.fittedvalues
    residuals = result.resid
    
    results_df = pd.DataFrame({
        'subject_id': df['subject_id'],
        'week': df['week'],
        'bmi': df['bmi'],
        'actual_y_concentration': df['y_concentration'],
        'predicted_y_concentration': predictions,
        'residuals': residuals
    })
    
    # 重命名列以保持与原来输出格式一致
    results_df = results_df.rename(columns={
        'subject_id': '孕妇代码',
        'week': '检测孕周',
        'bmi': '孕妇BMI',
        'actual_y_concentration': '实际Y染色体浓度',
        'predicted_y_concentration': '预测Y染色体浓度',
        'residuals': '残差'
    })
    
    results_df.to_csv(os.path.join(save_dir, 'mixed_effects_predictions.csv'), index=False, encoding='utf-8-sig')
    
    # 2. 保存模型信息
    with open(os.path.join(save_dir, 'mixed_effects_model_info.md'), 'w', encoding='utf-8') as f:
        f.write("# 混合效应模型结果\n")
        f.write("=" * 50 + "\n\n")
        f.write("## 模型形式\n")
        f.write("Y_ij = (β₀ + u_i) + β₁·Week_ij + β₂·BMI_ij + ε_ij\n\n")
        f.write("其中：\n")
        f.write("- Y_ij: 第i位孕妇第j次检测的Y染色体浓度\n")
        f.write("- β₀: 固定截距（全体平均基线）\n")
        f.write("- u_i: 第i位孕妇的随机截距（个体差异）\n")
        f.write("- β₁: 孕周的固定效应系数\n")
        f.write("- β₂: BMI的固定效应系数\n")
        f.write("- ε_ij: 残差项\n\n")
        
        f.write("## 模型参数\n")
        f.write(f"样本数: {result.nobs}\n")
        f.write(f"组数（孕妇数）: {len(result.random_effects)}\n")
        f.write(f"平均每组样本数: {result.nobs / len(result.random_effects):.2f}\n")
        f.write(f"对数似然: {result.llf:.4f}\n")
        f.write(f"AIC: {result.aic:.4f}\n")
        f.write(f"BIC: {result.bic:.4f}\n\n")
        
        f.write("## 固定效应\n")
        f.write("| 参数 | 估计值 | 标准误 | t值 | p值 |\n")
        f.write("|------|--------|--------|-----|-----|\n")
        
        for param in result.params.index:
            if param != 'Group Var':
                coef = result.params[param]
                se = result.bse[param]
                t_val = result.tvalues[param]
                p_val = result.pvalues[param]
                f.write(f"| {param} | {coef:.6f} | {se:.6f} | {t_val:.4f} | {p_val:.6e} |\n")
        
        f.write("\n## 随机效应\n")
        f.write(f"组间方差 (σ²_u): {result.cov_re.iloc[0,0]:.6f}\n")
        f.write(f"组内方差 (σ²_ε): {result.scale:.6f}\n")
        
        # 计算组内相关系数 (ICC)
        sigma_u_sq = result.cov_re.iloc[0,0]
        sigma_e_sq = result.scale
        icc = sigma_u_sq / (sigma_u_sq + sigma_e_sq)
        f.write(f"组内相关系数 (ICC): {icc:.6f}\n")
        f.write(f"ICC解释: 总变异中{icc*100:.1f}%由孕妇间个体差异导致\n\n")
        
        f.write("## 模型诊断\n")
        f.write(f"残差标准差: {np.sqrt(result.scale):.6f}\n")
        f.write(f"随机截距标准差: {np.sqrt(sigma_u_sq):.6f}\n")
    
    # 3. 保存线性模型系数（保持与原来输出格式一致）
    coef_data = []
    for param in result.params.index:
        if param != 'Group Var':
            coef_data.append({
                '变量': param,
                '系数': result.params[param],
                '标准误': result.bse[param],
                't值': result.tvalues[param],
                'p值': result.pvalues[param]
            })
    
    coef_df = pd.DataFrame(coef_data)
    coef_df.to_csv(os.path.join(save_dir, 'mixed_effects_coefs.csv'), index=False, encoding='utf-8-sig')
    
    # 4. 保存样条函数数据（为了保持输出格式一致，这里保存固定效应的线性函数）
    # 创建BMI和孕周的范围
    bmi_range = np.linspace(df['bmi'].min(), df['bmi'].max(), 100)
    week_range = np.linspace(df['week'].min(), df['week'].max(), 100)
    
    # 获取固定效应系数
    intercept = result.params['Intercept']
    week_coef = result.params['week']
    bmi_coef = result.params['bmi']
    
    # 计算线性贡献（固定效应部分）
    bmi_contribution = bmi_coef * bmi_range
    week_contribution = week_coef * week_range
    
    spline_data = pd.DataFrame({
        'BMI': bmi_range,
        'BMI样条值': bmi_contribution,  # 这里实际是线性贡献
        '孕周': week_range,
        '孕周样条值': week_contribution  # 这里实际是线性贡献
    })
    spline_data.to_csv(os.path.join(save_dir, 'fixed_effects_functions.csv'), index=False, encoding='utf-8-sig')

def calculate_model_summary(result):
    """
    计算模型摘要统计
    """
    summary = {
        'n_obs': result.nobs,
        'n_groups': len(result.random_effects),
        'avg_obs_per_group': result.nobs / len(result.random_effects),
        'llf': result.llf,
        'aic': result.aic,
        'bic': result.bic,
        'sigma_u_sq': result.cov_re.iloc[0,0],
        'sigma_e_sq': result.scale,
        'icc': result.cov_re.iloc[0,0] / (result.cov_re.iloc[0,0] + result.scale)
    }
    return summary

if __name__ == "__main__":
    # 1. 加载数据
    print("正在加载数据...")
    data = pd.read_csv('./data/output.csv')
    
    # 选择需要的列
    selected_data = data[['孕妇代码', '检测孕周', '孕妇BMI', 'Y染色体浓度']].copy()
    selected_data = selected_data.dropna()
    
    print(f"数据加载完成，共{len(selected_data)}条记录")
    print(f"涉及{selected_data['孕妇代码'].nunique()}位孕妇")
    print(f"平均每位孕妇检测{len(selected_data)/selected_data['孕妇代码'].nunique():.1f}次")
    
    # 2. 拟合混合效应模型
    result, df = fit_mixed_effects_model(selected_data, "./1")
    
    if result is not None:
        print("混合效应模型拟合完成！")
        print("模型形式：Y_ij = (β₀ + u_i) + β₁·Week_ij + β₂·BMI_ij + ε_ij")
        print("所有结果已保存到 ./1 目录")
        
        # 打印模型摘要
        summary = calculate_model_summary(result)
        print(f"\n模型摘要:")
        print(f"样本数: {summary['n_obs']}")
        print(f"孕妇数: {summary['n_groups']}")
        print(f"平均每位孕妇检测次数: {summary['avg_obs_per_group']:.1f}")
        print(f"组内相关系数 (ICC): {summary['icc']:.4f}")
        print(f"ICC解释: 总变异中{summary['icc']*100:.1f}%由孕妇间个体差异导致")
    else:
        print("模型拟合失败！")
