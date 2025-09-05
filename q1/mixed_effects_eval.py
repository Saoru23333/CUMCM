import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

def fit_mixed_effects_model(df):
    """
    拟合混合效应模型并返回结果
    """
    # 重命名列以便在公式中使用
    df_model = df.rename(columns={
        '孕妇代码': 'subject_id',
        '检测孕周': 'week',
        '孕妇BMI': 'bmi',
        'Y染色体浓度': 'y_concentration'
    })
    
    # 拟合混合效应模型
    formula = 'y_concentration ~ week + bmi'
    groups = df_model['subject_id']
    
    try:
        model = mixedlm(formula, df_model, groups=groups, re_formula='1')
        result = model.fit()
        return result
    except Exception as e:
        print(f"模型拟合失败: {e}")
        return None

def fit_linear_model(df):
    """
    拟合简单线性模型作为对比
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    X = df[['检测孕周', '孕妇BMI']].values
    y = df['Y染色体浓度'].values
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    return model, y_pred, r2

def calculate_icc(result):
    """计算组内相关系数"""
    sigma_u_sq = result.cov_re.iloc[0,0]  # 组间方差
    sigma_e_sq = result.scale  # 组内方差
    icc = sigma_u_sq / (sigma_u_sq + sigma_e_sq)
    return icc, sigma_u_sq, sigma_e_sq

def calculate_r_squared(result, df):
    """计算R²"""
    # 计算总平方和
    y_mean = df['Y染色体浓度'].mean()
    sst = ((df['Y染色体浓度'] - y_mean) ** 2).sum()
    
    # 计算残差平方和
    sse = (result.resid ** 2).sum()
    
    # 计算R²
    r_squared = 1 - sse / sst
    return r_squared, sst, sse

def likelihood_ratio_test(result_mixed, result_linear=None):
    """
    似然比检验：比较混合效应模型与线性模型
    """
    if result_linear is None:
        return None, None, None
    
    # 计算似然比统计量
    lr_stat = 2 * (result_mixed.llf - result_linear.llf)
    
    # 自由度差（混合效应模型多了一个随机截距方差参数）
    df_diff = 1
    
    # 计算p值
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    
    return lr_stat, df_diff, p_value

def main():
    out_dir = './1'
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载数据
    data = pd.read_csv('data/output.csv')
    df = data[['孕妇代码', '孕妇BMI', '检测孕周', 'Y染色体浓度']].dropna().copy()
    
    print(f"数据加载完成，共{len(df)}条记录")
    print(f"涉及{df['孕妇代码'].nunique()}位孕妇")
    
    # 1. 拟合混合效应模型
    print("正在拟合混合效应模型...")
    result_mixed = fit_mixed_effects_model(df)
    
    if result_mixed is None:
        print("混合效应模型拟合失败！")
        return
    
    # 2. 拟合简单线性模型作为对比
    print("正在拟合简单线性模型...")
    linear_model, y_pred_linear, r2_linear = fit_linear_model(df)
    
    # 3. 计算模型统计量
    print("正在计算模型统计量...")
    
    # 混合效应模型统计量
    icc, sigma_u_sq, sigma_e_sq = calculate_icc(result_mixed)
    r_squared_mixed, sst, sse_mixed = calculate_r_squared(result_mixed, df)
    
    # 线性模型统计量
    sse_linear = ((df['Y染色体浓度'] - y_pred_linear) ** 2).sum()
    r_squared_linear = 1 - sse_linear / sst
    
    # 4. 输出结果到文件
    with open(os.path.join(out_dir, 'mixed_effects_eval.txt'), 'w', encoding='utf-8') as f:
        f.write('混合效应模型显著性检验与性能度量\n')
        f.write('=' * 50 + '\n')
        
        f.write('一、混合效应模型结果\n')
        f.write(f'样本数: {result_mixed.nobs}\n')
        f.write(f'组数（孕妇数）: {len(result_mixed.random_effects)}\n')
        f.write(f'平均每组样本数: {result_mixed.nobs / len(result_mixed.random_effects):.2f}\n')
        f.write(f'对数似然: {result_mixed.llf:.4f}\n')
        f.write(f'AIC: {result_mixed.aic:.4f}\n')
        f.write(f'BIC: {result_mixed.bic:.4f}\n')
        f.write(f'R²: {r_squared_mixed:.6f}\n\n')
        
        f.write('二、固定效应显著性检验\n')
        f.write('| 变量 | 系数 | 标准误 | t值 | p值 | 显著性 |\n')
        f.write('|------|------|--------|-----|-----|--------|\n')
        
        for param in result_mixed.params.index:
            if param != 'Group Var':
                coef = result_mixed.params[param]
                se = result_mixed.bse[param]
                t_val = result_mixed.tvalues[param]
                p_val = result_mixed.pvalues[param]
                
                # 显著性标记
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                
                f.write(f'| {param} | {coef:.6f} | {se:.6f} | {t_val:.4f} | {p_val:.6e} | {sig} |\n')
        
        f.write('\n三、随机效应分析\n')
        f.write(f'组间方差 (σ²_u): {sigma_u_sq:.6f}\n')
        f.write(f'组内方差 (σ²_ε): {sigma_e_sq:.6f}\n')
        f.write(f'组内相关系数 (ICC): {icc:.6f}\n')
        f.write(f'ICC解释: 总变异中{icc*100:.1f}%由孕妇间个体差异导致\n\n')
        
        f.write('四、模型比较\n')
        f.write(f'混合效应模型 R²: {r_squared_mixed:.6f}\n')
        f.write(f'简单线性模型 R²: {r_squared_linear:.6f}\n')
        f.write(f'R²提升: {r_squared_mixed - r_squared_linear:.6f}\n\n')
        
        f.write('五、模型诊断\n')
        f.write(f'残差标准差: {np.sqrt(sigma_e_sq):.6f}\n')
        f.write(f'随机截距标准差: {np.sqrt(sigma_u_sq):.6f}\n')
        
        # 显著性解释
        f.write('\n六、显著性解释\n')
        f.write('注：*** p<0.001, ** p<0.01, * p<0.05, ns 不显著\n\n')
        
        for param in result_mixed.params.index:
            if param != 'Group Var':
                p_val = result_mixed.pvalues[param]
                coef = result_mixed.params[param]
                
                if param == 'week':
                    if p_val < 0.05:
                        f.write(f'孕周对Y染色体浓度有显著影响 (p={p_val:.6e})，每增加1周，浓度平均变化{coef:.6f}\n')
                    else:
                        f.write(f'孕周对Y染色体浓度无显著影响 (p={p_val:.6e})\n')
                elif param == 'bmi':
                    if p_val < 0.05:
                        f.write(f'BMI对Y染色体浓度有显著影响 (p={p_val:.6e})，每增加1单位，浓度平均变化{coef:.6f}\n')
                    else:
                        f.write(f'BMI对Y染色体浓度无显著影响 (p={p_val:.6e})\n')
        
        f.write(f'\n七、模型选择建议\n')
        if icc > 0.1:
            f.write(f'ICC={icc:.3f} > 0.1，表明孕妇间存在显著个体差异，使用混合效应模型是必要的。\n')
        else:
            f.write(f'ICC={icc:.3f} ≤ 0.1，孕妇间个体差异较小，简单线性模型可能足够。\n')
    
    # 5. 保存线性模型系数（保持与原来输出格式一致）
    coef_data = []
    for param in result_mixed.params.index:
        if param != 'Group Var':
            coef_data.append({
                '变量': param,
                '系数': result_mixed.params[param],
                '标准误': result_mixed.bse[param],
                't值': result_mixed.tvalues[param],
                'p值': result_mixed.pvalues[param],
                'R2(混合效应模型)': [r_squared_mixed if param == 'Intercept' else np.nan][0]
            })
    
    coef_df = pd.DataFrame(coef_data)
    coef_df.to_csv(os.path.join(out_dir, 'linear_model_coefs.csv'), index=False, encoding='utf-8-sig')
    
    print('混合效应模型评估完成，结果已写入 ./1/gam_eval.txt 与 ./1/linear_model_coefs.csv')
    print(f'混合效应模型 R²: {r_squared_mixed:.6f}')
    print(f'简单线性模型 R²: {r_squared_linear:.6f}')
    print(f'组内相关系数 (ICC): {icc:.6f}')
    print(f'ICC解释: 总变异中{icc*100:.1f}%由孕妇间个体差异导致')

if __name__ == '__main__':
    main()
