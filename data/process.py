import pandas as pd
import numpy as np

def preprocess_data(input_file, output_file):
    """
    对output.csv数据进行进一步预处理：
    1. 只保留GC含量在0.4~0.6之间的数据
    2. 处理同一孕妇连续相同检测抽血次数的重复数据，只保留最后一行
    """
    
    # 读取数据
    print("正在读取数据...")
    df = pd.read_csv(input_file)
    print(f"原始数据行数: {len(df)}")
    
    # 1. 过滤GC含量在0.4~0.6之间的数据
    print("正在过滤GC含量...")
    gc_mask = (df['GC含量'] >= 0.4) & (df['GC含量'] <= 0.6)
    df_filtered = df[gc_mask].copy()
    print(f"GC含量过滤后行数: {len(df_filtered)}")
    
    # 2. 处理同一孕妇连续相同检测抽血次数的重复数据
    print("正在处理重复数据...")
    
    # 按孕妇代码和检测抽血次数排序
    df_filtered = df_filtered.sort_values(['孕妇代码', '检测抽血次数', '序号'])
    
    # 找出需要删除的重复行
    rows_to_remove = []
    
    for i in range(len(df_filtered) - 1):
        current_row = df_filtered.iloc[i]
        next_row = df_filtered.iloc[i + 1]
        
        # 检查是否是同一孕妇且检测抽血次数相同
        if (current_row['孕妇代码'] == next_row['孕妇代码'] and 
            current_row['检测抽血次数'] == next_row['检测抽血次数']):
            # 标记当前行为需要删除（保留最后一行）
            rows_to_remove.append(i)
    
    # 删除重复行
    df_cleaned = df_filtered.drop(df_filtered.index[rows_to_remove])
    print(f"去除重复数据后行数: {len(df_cleaned)}")
    print(f"删除了 {len(rows_to_remove)} 行重复数据")
    
    # 重新排序
    df_cleaned = df_cleaned.sort_values('序号').reset_index(drop=True)
    
    # 保存处理后的数据
    print(f"正在保存数据到 {output_file}...")
    df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 输出统计信息
    print("\n=== 预处理完成 ===")
    print(f"原始数据行数: {len(df)}")
    print(f"GC含量过滤后行数: {len(df_filtered)}")
    print(f"最终数据行数: {len(df_cleaned)}")
    print(f"GC含量范围: {df_cleaned['GC含量'].min():.4f} - {df_cleaned['GC含量'].max():.4f}")
    
    # 显示重复数据处理的详细信息
    if len(rows_to_remove) > 0:
        print(f"\n重复数据处理详情:")
        print(f"删除了 {len(rows_to_remove)} 行重复数据")
        
        # 统计每个孕妇的重复情况
        duplicate_patients = {}
        for idx in rows_to_remove:
            patient_code = df_filtered.iloc[idx]['孕妇代码']
            test_count = df_filtered.iloc[idx]['检测抽血次数']
            key = f"{patient_code}-第{test_count}次"
            duplicate_patients[key] = duplicate_patients.get(key, 0) + 1
        
        print("重复数据分布:")
        for key, count in duplicate_patients.items():
            print(f"  {key}: {count}行重复数据")
    
    return df_cleaned

if __name__ == "__main__":
    # 执行预处理
    input_file = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/output.csv"
    output_file = "/Users/torealu/Desktop/2025秋/数学建模/src/CUMCM/data/output_processed.csv"
    
    processed_df = preprocess_data(input_file, output_file)
    
    print(f"\n预处理完成！处理后的数据已保存到: {output_file}")
