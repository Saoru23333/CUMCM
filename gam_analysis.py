import os
import numpy as np
import pandas as pd
from scipy import stats


def corr_pair(y: np.ndarray, x: np.ndarray):
	pr, pp = stats.pearsonr(x, y)
	sr, sp = stats.spearmanr(x, y)
	return pr, pp, sr, sp


def main():
	out_dir = './1'
	os.makedirs(out_dir, exist_ok=True)
	data = pd.read_csv('data/output.csv')
	cols = ['Y染色体浓度', '孕妇BMI', '检测孕周']
	df = data[cols].dropna().copy()

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

	out_df = pd.DataFrame(results)
	out_path = os.path.join(out_dir, 'simple_correlations.csv')
	out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
	print(f'已保存相关性结果至 {out_path}')


if __name__ == '__main__':
	main()
