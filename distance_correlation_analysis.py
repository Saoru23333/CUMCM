import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import os

def distance_correlation(x, y):
    """
    计算两个变量之间的距离相关系数
    
    距离相关系数的核心思想：
    比较两组变量内部所有样本点两两之间的距离。
    如果两个变量相关，那么在一个变量中距离近的点对，在另一个变量中距离也应该近。
    它量化了这种"距离相似性"。
    
    参数:
    x, y: 一维数组
    
    返回:
    dcor: 距离相关系数 (0-1之间，0表示无相关性，1表示完全相关)
    """
    n = len(x)
    
    # 计算距离矩阵
    # 对于一维数据，距离就是绝对差值
    x_dist = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
    y_dist = np.abs(y[:, np.newaxis] - y[np.newaxis, :])
    
    # 中心化距离矩阵
    # 减去行均值、列均值，再加上总体均值
    x_row_mean = np.mean(x_dist, axis=1, keepdims=True)
    x_col_mean = np.mean(x_dist, axis=0, keepdims=True)
    x_grand_mean = np.mean(x_dist)
    x_centered = x_dist - x_row_mean - x_col_mean + x_grand_mean
    
    y_row_mean = np.mean(y_dist, axis=1, keepdims=True)
    y_col_mean = np.mean(y_dist, axis=0, keepdims=True)
    y_grand_mean = np.mean(y_dist)
    y_centered = y_dist - y_row_mean - y_col_mean + y_grand_mean
    
    # 计算距离协方差
    dcov_xy = np.sqrt(np.mean(x_centered * y_centered))
    
    # 计算距离方差
    dcov_xx = np.sqrt(np.mean(x_centered * x_centered))
    dcov_yy = np.sqrt(np.mean(y_centered * y_centered))
    
    # 计算距离相关系数
    if dcov_xx == 0 or dcov_yy == 0:
        dcor = 0
    else:
        dcor = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
    
    return dcor

def identify_numeric_columns(df):
    """
    识别数据框中的数值型列
    """
    numeric_cols = []
    for col in df.columns:
        # 跳过非数值列
        if col in ['序号', '孕妇代码', '末次月经', 'IVF妊娠', '检测日期', '染色体的非整倍体', '胎儿是否健康']:
            continue
        
        # 尝试转换为数值型
        try:
            pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    return numeric_cols

def main():
    """
    主函数：分析所有数值变量与Y染色体浓度的距离相关性
    """
    # 创建输出目录
    out_dir = './1'
    os.makedirs(out_dir, exist_ok=True)
    
    # 读取数据
    print("正在读取数据...")
    data = pd.read_csv('data/output.csv')
    print(f"数据形状: {data.shape}")
    
    # 识别数值型列
    numeric_cols = identify_numeric_columns(data)
    print(f"识别到 {len(numeric_cols)} 个数值型变量")
    print("数值型变量:", numeric_cols)
    
    # 检查Y染色体浓度列是否存在
    if 'Y染色体浓度' not in numeric_cols:
        print("错误：未找到'Y染色体浓度'列")
        return
    
    # 移除Y染色体浓度本身，避免自相关
    predictor_cols = [col for col in numeric_cols if col != 'Y染色体浓度']
    
    # 获取Y染色体浓度数据
    y_data = data['Y染色体浓度'].dropna()
    
    # 存储结果
    results = []
    
    print("\n开始计算距离相关系数...")
    for col in predictor_cols:
        print(f"正在分析: {col}")
        
        # 获取该列数据，并确保与Y染色体浓度有相同的索引
        col_data = data[col].dropna()
        
        # 找到两个变量都有值的共同索引
        common_idx = y_data.index.intersection(col_data.index)
        
        if len(common_idx) < 10:  # 如果共同样本太少，跳过
            print(f"  跳过 {col}: 共同样本数太少 ({len(common_idx)})")
            continue
        
        # 获取共同样本的数据
        y_common = y_data.loc[common_idx].values
        x_common = col_data.loc[common_idx].values
        
        # 计算距离相关系数
        try:
            dcor = distance_correlation(x_common, y_common)
            
            # 同时计算Pearson相关系数作为对比
            pearson_r, pearson_p = pearsonr(x_common, y_common)
            
            results.append({
                '变量': col,
                '距离相关系数': dcor,
                'Pearson相关系数': pearson_r,
                'Pearson_p值': pearson_p,
                '样本数': len(common_idx)
            })
            
            print(f"  距离相关系数: {dcor:.4f}, Pearson相关系数: {pearson_r:.4f}")
            
        except Exception as e:
            print(f"  计算 {col} 时出错: {e}")
            continue
    
    # 转换为DataFrame并按距离相关系数降序排列
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('距离相关系数', ascending=False)
    
    # 保存结果
    output_path = os.path.join(out_dir, 'distance_correlation_results.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n分析完成！结果已保存至: {output_path}")
    print(f"\n距离相关系数排名前10的变量:")
    print("=" * 80)
    print(f"{'排名':<4} {'变量':<20} {'距离相关系数':<12} {'Pearson相关系数':<15} {'样本数':<8}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"{i:<4} {row['变量']:<20} {row['距离相关系数']:<12.4f} {row['Pearson相关系数']:<15.4f} {row['样本数']:<8}")
    
    print(f"\n总共分析了 {len(results_df)} 个变量")
    print(f"距离相关系数范围: {results_df['距离相关系数'].min():.4f} - {results_df['距离相关系数'].max():.4f}")

if __name__ == '__main__':
    main()
