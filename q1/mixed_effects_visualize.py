import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from statsmodels.formula.api import mixedlm
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 中文字体与负号设置（macOS优先尝试苹方/黑体，回退到常见字体）
mpl.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Songti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False

# 定义自定义色系
CUSTOM_COLORS = ['#845EC2', '#D65DB1', '#FF6F91', '#FF9671', '#FFC75F', '#F9F871']
CUSTOM_CMAP = LinearSegmentedColormap.from_list('custom', CUSTOM_COLORS, N=256)

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

def create_mixed_effects_surface(result, bmi_range, week_range):
    """
    创建混合效应模型的预测曲面
    注意：这里只显示固定效应部分，不包含随机截距
    """
    # 获取固定效应系数
    intercept = result.params['Intercept']
    week_coef = result.params['week']
    bmi_coef = result.params['bmi']
    
    # 创建网格
    BB, GG = np.meshgrid(bmi_range, week_range)
    
    # 计算固定效应预测值（不包含随机截距）
    Z = intercept + week_coef * GG + bmi_coef * BB
    
    return BB, GG, Z

def plot_individual_trajectories(df, result, save_dir):
    """
    绘制个体轨迹图，显示随机截距的影响
    """
    # 获取随机截距
    random_effects = result.random_effects
    
    # 选择前10个个体进行可视化
    subjects = list(random_effects.keys())[:10]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, subject in enumerate(subjects):
        if i >= 10:
            break
            
        # 获取该个体的数据
        subject_data = df[df['孕妇代码'] == subject].copy()
        subject_data = subject_data.sort_values('检测孕周')
        
        if len(subject_data) < 2:
            continue
        
        # 获取该个体的随机截距
        random_intercept = random_effects[subject][0]
        
        # 计算固定效应预测
        fixed_pred = (result.params['Intercept'] + 
                     result.params['week'] * subject_data['检测孕周'] + 
                     result.params['bmi'] * subject_data['孕妇BMI'])
        
        # 计算总预测（固定效应 + 随机截距）
        total_pred = fixed_pred + random_intercept
        
        # 绘制
        ax = axes[i]
        ax.scatter(subject_data['检测孕周'], subject_data['Y染色体浓度'], 
                  color='red', alpha=0.7, s=50, label='实际值')
        ax.plot(subject_data['检测孕周'], fixed_pred, 
               color='blue', linestyle='--', alpha=0.7, label='固定效应')
        ax.plot(subject_data['检测孕周'], total_pred, 
               color='green', linestyle='-', linewidth=2, label='总预测')
        
        ax.set_xlabel('检测孕周')
        ax.set_ylabel('Y染色体浓度')
        ax.set_title(f'个体 {subject}\n随机截距: {random_intercept:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(subjects), 10):
        axes[i].set_visible(False)
    
    plt.suptitle('个体轨迹图：随机截距的影响', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'individual_trajectories.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print('已保存个体轨迹图至 ./1/individual_trajectories.png')

def plot_residual_analysis(result, df, save_dir):
    """
    绘制残差分析图
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 残差 vs 拟合值
    fitted_values = result.fittedvalues
    residuals = result.resid
    
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('拟合值')
    axes[0, 0].set_ylabel('残差')
    axes[0, 0].set_title('残差 vs 拟合值')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q图
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('残差Q-Q图')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 残差 vs BMI
    axes[1, 0].scatter(df['孕妇BMI'], residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('BMI')
    axes[1, 0].set_ylabel('残差')
    axes[1, 0].set_title('残差 vs BMI')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 残差 vs 孕周
    axes[1, 1].scatter(df['检测孕周'], residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('检测孕周')
    axes[1, 1].set_ylabel('残差')
    axes[1, 1].set_title('残差 vs 孕周')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('混合效应模型残差分析', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print('已保存残差分析图至 ./1/residual_analysis.png')

def main():
    out_dir = './1'
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载数据
    df = pd.read_csv('data/output.csv')[['孕妇代码', '孕妇BMI', '检测孕周', 'Y染色体浓度']].dropna()
    
    print(f"数据加载完成，共{len(df)}条记录")
    print(f"涉及{df['孕妇代码'].nunique()}位孕妇")
    
    # 拟合混合效应模型
    result = fit_mixed_effects_model(df)
    
    if result is None:
        print("模型拟合失败！")
        return
    
    # 创建预测曲面（固定效应部分）
    bmi_range = np.linspace(df['孕妇BMI'].min(), df['孕妇BMI'].max(), 60)
    week_range = np.linspace(df['检测孕周'].min(), df['检测孕周'].max(), 60)
    BB, GG, Z = create_mixed_effects_surface(result, bmi_range, week_range)
    
    # 1. 绘制3D曲面图（固定效应）
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(BB, GG, Z, cmap=CUSTOM_CMAP, edgecolor='none', alpha=0.85)
    ax.set_xlabel('BMI')
    ax.set_ylabel('孕周')
    ax.set_zlabel('Y染色体浓度')
    ax.set_title('混合效应模型固定效应曲面：Y = β₀ + β₁·Week + β₂·BMI')
    fig.colorbar(surf, shrink=0.6, aspect=12, label='预测Y浓度（固定效应）')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mixed_effects_surface.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('已保存3D曲面图至 ./1/mixed_effects_surface.png')
    
    # 2. 绘制个体轨迹图
    plot_individual_trajectories(df, result, out_dir)
    
    # 3. 绘制残差分析图
    plot_residual_analysis(result, df, out_dir)
    
    # 4. 绘制随机截距分布图
    random_effects = result.random_effects
    random_intercepts = [re[0] for re in random_effects.values()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(random_intercepts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='总体均值')
    plt.xlabel('随机截距')
    plt.ylabel('频数')
    plt.title('随机截距分布图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'random_intercepts_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print('已保存随机截距分布图至 ./1/random_intercepts_distribution.png')
    
    print('\n所有可视化图表已生成完成！')

if __name__ == '__main__':
    main()
