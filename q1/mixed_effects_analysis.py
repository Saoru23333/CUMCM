import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm
import warnings
warnings.filterwarnings('ignore')

def corr_pair(y: np.ndarray, x: np.ndarray):
    """计算Pearson和Spearman相关系数"""
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return pr, pp, sr, sp

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

def main():
    out_dir = './1'
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载数据
    data = pd.read_csv('data/output.csv')
    cols = ['孕妇代码', 'Y染色体浓度', '孕妇BMI', '检测孕周']
    df = data[cols].dropna().copy()
    
    print(f"数据加载完成，共{len(df)}条记录")
    print(f"涉及{df['孕妇代码'].nunique()}位孕妇")
    
    # 1. 计算简单相关性分析
    print("正在计算相关性分析...")
    y = df['Y染色体浓度'].to_numpy(float)
    results = []
    
    for col in ['孕妇BMI', '检测孕周']:
        x = df[col].to_numpy(float)
        pr, pp, sr, sp = corr_pair(y, x)
        results.append({
            '指标': col,
            'Pearson_r': pr,
            'Pearson_p': pp,
            'Spearman_rho': sr,
            'Spearman_p': sp,
            'N': len(x)
        })
    
    # 保存相关性结果
    out_df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, 'simple_correlations.csv')
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'已保存相关性结果至 {out_path}')
    
    # 2. 拟合混合效应模型
    print("正在拟合混合效应模型...")
    result = fit_mixed_effects_model(df)
    
    if result is None:
        print("模型拟合失败！")
        return
    
    # 3. 计算模型统计量
    print("正在计算模型统计量...")
    
    # 计算ICC
    icc, sigma_u_sq, sigma_e_sq = calculate_icc(result)
    
    # 计算R²
    r_squared, sst, sse = calculate_r_squared(result, df)
    
    # 4. 输出结果到文件
    with open(os.path.join(out_dir, 'mixed_effects_eval.txt'), 'w', encoding='utf-8') as f:
        f.write('混合效应模型显著性检验与性能度量\n')
        f.write('=' * 50 + '\n')
        f.write('一、混合效应模型结果\n')
        f.write(f'样本数: {result.nobs}\n')
        f.write(f'组数（孕妇数）: {len(result.random_effects)}\n')
        f.write(f'平均每组样本数: {result.nobs / len(result.random_effects):.2f}\n')
        f.write(f'对数似然: {result.llf:.4f}\n')
        f.write(f'AIC: {result.aic:.4f}\n')
        f.write(f'BIC: {result.bic:.4f}\n')
        f.write(f'R²: {r_squared:.6f}\n\n')
        
        f.write('二、固定效应显著性检验\n')
        f.write('| 变量 | 系数 | 标准误 | t值 | p值 |\n')
        f.write('|------|------|--------|-----|-----|\n')
        
        for param in result.params.index:
            if param != 'Group Var':
                coef = result.params[param]
                se = result.bse[param]
                t_val = result.tvalues[param]
                p_val = result.pvalues[param]
                f.write(f'| {param} | {coef:.6f} | {se:.6f} | {t_val:.4f} | {p_val:.6e} |\n')
        
        f.write('\n三、随机效应分析\n')
        f.write(f'组间方差 (σ²_u): {sigma_u_sq:.6f}\n')
        f.write(f'组内方差 (σ²_ε): {sigma_e_sq:.6f}\n')
        f.write(f'组内相关系数 (ICC): {icc:.6f}\n')
        f.write(f'ICC解释: 总变异中{icc*100:.1f}%由孕妇间个体差异导致\n\n')
        
        f.write('四、模型诊断\n')
        f.write(f'残差标准差: {np.sqrt(sigma_e_sq):.6f}\n')
        f.write(f'随机截距标准差: {np.sqrt(sigma_u_sq):.6f}\n')
        
        # 显著性解释
        f.write('\n五、显著性解释\n')
        for param in result.params.index:
            if param != 'Group Var':
                p_val = result.pvalues[param]
                if p_val < 0.001:
                    significance = "***"
                elif p_val < 0.01:
                    significance = "**"
                elif p_val < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
                
                f.write(f'{param}: p={p_val:.6e} ({significance})\n')
        
        f.write('\n注：*** p<0.001, ** p<0.01, * p<0.05, ns 不显著\n')
    
    # 5. 保存线性模型系数（保持与原来输出格式一致）
    coef_data = []
    for param in result.params.index:
        if param != 'Group Var':
            coef_data.append({
                '变量': param,
                '系数': result.params[param],
                '标准误': result.bse[param],
                't值': result.tvalues[param],
                'p值': result.pvalues[param],
                'R2(混合效应模型)': [r_squared if param == 'Intercept' else np.nan][0]
            })
    
    coef_df = pd.DataFrame(coef_data)
    coef_df.to_csv(os.path.join(out_dir, 'mixed_effects_coefs.csv'), index=False, encoding='utf-8-sig')
    
    print('混合效应模型分析完成，结果已写入 ./1/mixed_effects_eval.txt 与 ./1/mixed_effects_coefs.csv')
    print(f'模型R²: {r_squared:.6f}')
    print(f'组内相关系数 (ICC): {icc:.6f}')
    print(f'ICC解释: 总变异中{icc*100:.1f}%由孕妇间个体差异导致')

if __name__ == '__main__':
    main()
