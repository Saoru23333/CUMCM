import os
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy import stats


def make_bspline_basis(x: np.ndarray, degree: int = 3, num_internal_knots: int = 4):
	"""构造B样条基底矩阵 X -> [B1(x), ..., Bm(x)]."""
	x = np.asarray(x, dtype=float)
	# 内部结点按分位数均匀放置
	quantiles = np.linspace(0, 1, num_internal_knots + 2)[1:-1]
	internal_knots = np.quantile(x, quantiles) if num_internal_knots > 0 else np.array([])
	xmin, xmax = float(np.min(x)), float(np.max(x))
	# 结点向量：边界重复 degree+1 次
	t = np.r_[np.repeat(xmin, degree + 1), internal_knots, np.repeat(xmax, degree + 1)]
	n_basis = len(t) - degree - 1
	# 构造基底矩阵
	B = np.empty((x.shape[0], n_basis), dtype=float)
	for j in range(n_basis):
		c = np.zeros(n_basis)
		c[j] = 1.0
		spline = BSpline(t, c, degree, extrapolate=True)
		B[:, j] = spline(x)
	return B, t, degree


def ols_fit(X: np.ndarray, y: np.ndarray):
	"""最小二乘估计，返回beta, yhat, 残差, SSE, MSE, 协方差矩阵。"""
	XtX = X.T @ X
	Xty = X.T @ y
	beta = np.linalg.solve(XtX, Xty)
	yhat = X @ beta
	resid = y - yhat
	SSE = float(resid.T @ resid)
	n, p = X.shape
	dof = n - p
	MSE = SSE / dof
	cov_beta = MSE * np.linalg.inv(XtX)
	return beta, yhat, resid, SSE, MSE, cov_beta, dof


def partial_f_test(y: np.ndarray, X_full: np.ndarray, X_reduced: np.ndarray, df_full: int):
	"""针对一组变量的部分F检验（全模型 vs 约简模型）。"""
	_, _, _, SSE_full, _, _, _ = ols_fit(X_full, y)
	_, _, _, SSE_red, _, _, _ = ols_fit(X_reduced, y)
	df_red = X_reduced.shape[0] - X_reduced.shape[1]
	# 自由度差
	df_num = df_red - df_full
	F = ((SSE_red - SSE_full) / df_num) / (SSE_full / df_full)
	p = stats.f.sf(F, df_num, df_full)
	return F, p, df_num, df_full


def main():
	out_dir = './1'
	os.makedirs(out_dir, exist_ok=True)
	data = pd.read_csv('data/output.csv')
	df = data[['孕妇BMI', '检测孕周', 'Y染色体浓度']].dropna().copy()
	bmi = df['孕妇BMI'].to_numpy(float)
	gw = df['检测孕周'].to_numpy(float)
	y = df['Y染色体浓度'].to_numpy(float)
	N = y.shape[0]

	# 1) 样条GAM整体F检验（检验所有样条项是否整体显著）
	B_bmi, knots_bmi, k_bmi = make_bspline_basis(bmi, degree=3, num_internal_knots=4)
	B_gw, knots_gw, k_gw = make_bspline_basis(gw, degree=3, num_internal_knots=4)
	Intercept = np.ones((N, 1))
	X_full = np.concatenate([Intercept, B_bmi, B_gw], axis=1)
	X_intercept = Intercept  # 仅截距模型

	# 全模型与仅截距模型比较 -> 整体F
	beta_full, yhat_full, resid_full, SSE_full, MSE_full, cov_full, df_full = ols_fit(X_full, y)
	beta_int, yhat_int, resid_int, SSE_int, MSE_int, cov_int, df_int = ols_fit(X_intercept, y)
	p_all = X_full.shape[1] - X_intercept.shape[1]
	F_overall = ((SSE_int - SSE_full) / p_all) / (SSE_full / df_full)
	p_overall = stats.f.sf(F_overall, p_all, df_full)
	# R^2（样条GAM）
	SST = float(((y - y.mean()) ** 2).sum())
	R2_gam = 1.0 - SSE_full / SST

	# 针对每个变量的部分F检验（组检验）
	X_wo_bmi = np.concatenate([Intercept, B_gw], axis=1)
	F_bmi, p_bmi, dfnum_bmi, dfden_bmi = partial_f_test(y, X_full, X_wo_bmi, df_full)
	X_wo_gw = np.concatenate([Intercept, B_bmi], axis=1)
	F_gw, p_gw, dfnum_gw, dfden_gw = partial_f_test(y, X_full, X_wo_gw, df_full)

	# 2) 线性模型的单变量t检验（简化模型：y = a + b*bmi + c*孕周）
	X_lin = np.column_stack([np.ones(N), bmi, gw])
	beta_lin, yhat_lin, resid_lin, SSE_lin, MSE_lin, cov_lin, df_lin = ols_fit(X_lin, y)
	se_lin = np.sqrt(np.diag(cov_lin))
	t_bmi = beta_lin[1] / se_lin[1]
	p_t_bmi = 2 * stats.t.sf(np.abs(t_bmi), df_lin)
	t_gw = beta_lin[2] / se_lin[2]
	p_t_gw = 2 * stats.t.sf(np.abs(t_gw), df_lin)
	# R^2（线性模型）
	R2_lin = 1.0 - SSE_lin / SST

	# 输出结果到文件
	with open(os.path.join(out_dir, 'gam_eval.txt'), 'w', encoding='utf-8') as f:
		f.write('显著性检验与性能度量\n')
		f.write('=' * 50 + '\n')
		f.write('一、样条GAM整体与分组F检验\n')
		f.write(f'整体F: {F_overall:.6f}, 自由度=({p_all}, {df_full}), p值={p_overall:.6e}\n')
		f.write(f'BMI组部分F: {F_bmi:.6f}, 自由度=({dfnum_bmi}, {dfden_bmi}), p值={p_bmi:.6e}\n')
		f.write(f'孕周组部分F: {F_gw:.6f}, 自由度=({dfnum_gw}, {dfden_gw}), p值={p_gw:.6e}\n')
		f.write(f'样条GAM的R^2: {R2_gam:.6f}\n')
		f.write('\n二、简化线性模型\n')
		f.write(f'BMI: t={t_bmi:.6f}, 自由度={df_lin}, p值={p_t_bmi:.6e}\n')
		f.write(f'孕周: t={t_gw:.6f}, 自由度={df_lin}, p值={p_t_gw:.6e}\n')
		f.write(f'线性模型的R^2: {R2_lin:.6f}\n')

	# 同时保存线性模型系数
	coef_df = pd.DataFrame({
		'变量': ['截距', 'BMI', '孕周'],
		'系数': beta_lin,
		'标准误': np.sqrt(np.diag(cov_lin)),
		't值': [np.nan, t_bmi, t_gw],
		'p值': [np.nan, p_t_bmi, p_t_gw],
		'R2(线性模型)': [R2_lin, np.nan, np.nan],
	})
	coef_df.to_csv(os.path.join(out_dir, 'linear_model_coefs.csv'), index=False, encoding='utf-8-sig')

	print('显著性检验完成，结果已写入 ./1/gam_eval.txt 与 ./1/linear_model_coefs.csv')
	print(f'样条GAM R^2: {R2_gam:.6f} | 线性模型 R^2: {R2_lin:.6f}')


if __name__ == '__main__':
	main()
