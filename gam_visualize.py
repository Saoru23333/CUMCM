import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import UnivariateSpline

# 中文字体与负号设置（macOS优先尝试苹方/黑体，回退到常见字体）
mpl.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Songti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


def build_splines(bmi: np.ndarray, gw: np.ndarray, y: np.ndarray):
	bmi_idx = np.argsort(bmi)
	gw_idx = np.argsort(gw)
	bmi_s = UnivariateSpline(bmi[bmi_idx], y[bmi_idx], s=len(bmi)*0.1, k=3)
	gw_s = UnivariateSpline(gw[gw_idx], y[gw_idx], s=len(gw)*0.1, k=3)
	return bmi_s, gw_s


def main():
	out_dir = './1'
	os.makedirs(out_dir, exist_ok=True)
	df = pd.read_csv('data/output.csv')[['孕妇BMI', '检测孕周', 'Y染色体浓度']].dropna()
	bmi = df['孕妇BMI'].to_numpy(float)
	gw = df['检测孕周'].to_numpy(float)
	y = df['Y染色体浓度'].to_numpy(float)

	bmi_s, gw_s = build_splines(bmi, gw, y)

	bmi_grid = np.linspace(bmi.min(), bmi.max(), 60)
	gw_grid = np.linspace(gw.min(), gw.max(), 60)
	BB, GG = np.meshgrid(bmi_grid, gw_grid)
	Z = bmi_s(BB) + gw_s(GG)

	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(BB, GG, Z, cmap='viridis', edgecolor='none', alpha=0.85)
	ax.set_xlabel('BMI')
	ax.set_ylabel('孕周')
	ax.set_zlabel('Y染色体浓度')
	ax.set_title('GAM拟合曲面：Y = f(BMI) + f(孕周)')
	fig.colorbar(surf, shrink=0.6, aspect=12, label='预测Y浓度')
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, 'gam_surface.png'), dpi=300, bbox_inches='tight')
	plt.close(fig)
	print('已保存3D曲面图至 ./1/gam_surface.png')


if __name__ == '__main__':
	main()
