import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import os

def fit_gam_splines(bmi, gestational_week, y_concentration, save_dir):
    """
    拟合真正的GAM模型，使用样条函数
    GAM: Y = f1(BMI) + f2(孕周) + ε
    其中f1和f2是平滑的非参数函数
    """
    # 对BMI进行样条拟合
    # 首先对BMI数据进行排序
    bmi_sorted_indices = np.argsort(bmi)
    bmi_sorted = bmi[bmi_sorted_indices]
    y_bmi_sorted = y_concentration[bmi_sorted_indices]
    
    # 使用UnivariateSpline进行平滑样条拟合
    bmi_spline = UnivariateSpline(bmi_sorted, y_bmi_sorted, s=len(bmi)*0.1, k=3)
    
    # 对孕周进行样条拟合
    # 首先对孕周数据进行排序
    gestational_sorted_indices = np.argsort(gestational_week)
    gestational_sorted = gestational_week[gestational_sorted_indices]
    y_gestational_sorted = y_concentration[gestational_sorted_indices]
    
    # 使用UnivariateSpline进行平滑样条拟合
    gestational_spline = UnivariateSpline(gestational_sorted, y_gestational_sorted, s=len(gestational_week)*0.1, k=3)
    
    # 计算每个变量的贡献（需要按原始顺序）
    bmi_contribution = bmi_spline(bmi)
    gestational_contribution = gestational_spline(gestational_week)
    
    # 计算残差（交互项）
    residuals = y_concentration - bmi_contribution - gestational_contribution
    
    # 保存模型结果
    save_gam_results(bmi, gestational_week, y_concentration, bmi_contribution, 
                     gestational_contribution, residuals, bmi_spline, gestational_spline, save_dir)
    
    return bmi_spline, gestational_spline

def save_gam_results(bmi, gestational_week, y_concentration, bmi_contribution, 
                     gestational_contribution, residuals, bmi_spline, gestational_spline, save_dir):
    """
    保存GAM模型结果
    """
    # 保存预测结果
    results_df = pd.DataFrame({
        'BMI': bmi,
        '孕周': gestational_week,
        '实际Y染色体浓度': y_concentration,
        'BMI贡献': bmi_contribution,
        '孕周贡献': gestational_contribution,
        '残差': residuals,
        '预测Y染色体浓度': bmi_contribution + gestational_contribution
    })
    results_df.to_csv(os.path.join(save_dir, 'gam_predictions.csv'), index=False, encoding='utf-8-sig')
    
    # 保存模型信息
    with open(os.path.join(save_dir, 'gam_model_info.txt'), 'w', encoding='utf-8') as f:
        f.write("广义可加模型(GAM)结果\n")
        f.write("=" * 50 + "\n\n")
        f.write("模型形式：Y染色体浓度 = f1(BMI) + f2(孕周) + ε\n\n")
        f.write("其中：\n")
        f.write("f1(BMI): BMI的平滑函数贡献\n")
        f.write("f2(孕周): 孕周的平滑函数贡献\n")
        f.write("ε: 残差项（包含交互效应和随机误差）\n\n")
        f.write("样条函数参数：\n")
        f.write(f"BMI样条平滑参数: {bmi_spline.get_residual()}\n")
        f.write(f"孕周样条平滑参数: {gestational_spline.get_residual()}\n")
        f.write(f"数据点数: {len(bmi)}\n")
    
    # 保存样条函数数据点（用于后续可视化或预测）
    bmi_range = np.linspace(bmi.min(), bmi.max(), 100)
    gestational_range = np.linspace(gestational_week.min(), gestational_week.max(), 100)
    
    bmi_spline_values = bmi_spline(bmi_range)
    gestational_spline_values = gestational_spline(gestational_range)
    
    spline_data = pd.DataFrame({
        'BMI': bmi_range,
        'BMI样条值': bmi_spline_values,
        '孕周': gestational_range,
        '孕周样条值': gestational_spline_values
    })
    spline_data.to_csv(os.path.join(save_dir, 'spline_functions.csv'), index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    # 1. 加载数据
    print("正在加载数据...")
    data = pd.read_csv('./data/output.csv')
    
    # 选择需要的列
    selected_data = data[['孕妇BMI', '检测孕周', 'Y染色体浓度']].copy()
    selected_data = selected_data.dropna()
    
    print(f"数据加载完成，共{len(selected_data)}条记录")
    
    # 2. 拟合GAM模型
    print("正在拟合GAM模型...")
    bmi_spline, gestational_spline = fit_gam_splines(
        selected_data['孕妇BMI'].values,
        selected_data['检测孕周'].values,
        selected_data['Y染色体浓度'].values,
        "./1"
    )
    
    print("GAM模型拟合完成！")
    print("模型形式：Y染色体浓度 = f1(BMI) + f2(孕周) + ε")
    print("所有结果已保存到 ./1 目录")
