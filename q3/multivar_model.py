import pandas as pd
import numpy as np
import os
from statsmodels.formula.api import mixedlm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class MultivariateRiskModel:
    """
    扩展的混合效应模型，纳入更多变量并计算个体风险评分
    """
    
    def __init__(self, data_path, save_dir, c_min=0.04):
        self.data_path = data_path
        self.save_dir = save_dir
        self.c_min = c_min  # 检测成功的最低Y染色体浓度阈值
        self.model = None
        self.data = None
        self.risk_scores = None
        self.model_result = None
        
    def load_and_prepare_data(self):
        """加载数据并进行预处理"""
        print("正在加载数据...")
        self.data = pd.read_csv(self.data_path)
        
        # 选择男胎孕妇数据
        male_fetus_data = self.data[self.data['胎儿是否健康'] == '是'].copy()
        
        # 选择需要的变量（移除孕妇代码作为自变量，保留Y染色体浓度作为因变量，添加GC含量）
        required_cols = ['孕妇代码', '检测孕周', '孕妇BMI', 'Y染色体浓度', '年龄', '身高', '体重', 
                        '怀孕次数', '生产次数', 'GC含量']
        
        # 检查列是否存在
        available_cols = [col for col in required_cols if col in male_fetus_data.columns]
        self.data = male_fetus_data[available_cols].copy()
        
        # 处理缺失值
        self.data = self.data.dropna()
        
        # 处理怀孕次数和生产次数的特殊值
        if '怀孕次数' in self.data.columns:
            self.data['怀孕次数'] = self.data['怀孕次数'].replace('≥3', 3)
            self.data['怀孕次数'] = pd.to_numeric(self.data['怀孕次数'], errors='coerce')
        
        if '生产次数' in self.data.columns:
            self.data['生产次数'] = pd.to_numeric(self.data['生产次数'], errors='coerce')
        
        # 再次删除缺失值
        self.data = self.data.dropna()
        
        print(f"数据加载完成，共{len(self.data)}条记录")
        print(f"涉及{self.data['孕妇代码'].nunique()}位孕妇")
        print(f"可用变量: {list(self.data.columns)}")
        
        return self.data
    
    def check_multicollinearity(self, df):
        """检查多重共线性"""
        print("检查多重共线性...")
        
        # 准备数值变量（添加GC含量，使用重命名后的列名）
        numeric_vars = ['week', 'bmi', 'age', 'height', 'weight', 'pregnancy_count', 'delivery_count', 'gc_content']
        available_vars = [var for var in numeric_vars if var in df.columns]
        
        if len(available_vars) < 2:
            print("可用变量不足，跳过多重共线性检查")
            return available_vars
        
        # 计算VIF
        X = df[available_vars].copy()
        X = add_constant(X)
        
        vif_data = pd.DataFrame()
        vif_data["变量"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        
        print("方差膨胀因子 (VIF):")
        print(vif_data)
        
        # 移除高VIF变量（VIF > 10）
        high_vif_vars = vif_data[vif_data['VIF'] > 10]['变量'].tolist()
        if 'const' in high_vif_vars:
            high_vif_vars.remove('const')
        
        if high_vif_vars:
            print(f"移除高VIF变量: {high_vif_vars}")
            available_vars = [var for var in available_vars if var not in high_vif_vars]
        
        return available_vars
    
    def fit_multivariate_model(self):
        """拟合扩展的混合效应模型"""
        print("正在拟合扩展的混合效应模型...")
        
        # 准备数据
        df = self.data.copy()
        
        # 重命名列以便在公式中使用（保留孕妇代码用于分组，保留Y染色体浓度作为因变量，添加GC含量）
        df = df.rename(columns={
            '孕妇代码': 'subject_id',
            '检测孕周': 'week',
            '孕妇BMI': 'bmi',
            'Y染色体浓度': 'y_concentration',
            '年龄': 'age',
            '身高': 'height',
            '体重': 'weight',
            '怀孕次数': 'pregnancy_count',
            '生产次数': 'delivery_count',
            'GC含量': 'gc_content'
        })
        
        # 检查多重共线性
        available_vars = self.check_multicollinearity(df)
        
        # 构建模型公式
        # 基础变量（不包括孕妇代码，因为它用于分组）
        base_vars = ['week', 'bmi']
        
        # 添加其他变量（避免多重共线性，不包括孕妇代码作为自变量）
        additional_vars = []
        for var in ['age', 'pregnancy_count', 'delivery_count', 'gc_content']:
            if var in available_vars:
                additional_vars.append(var)
        
        # 处理身高体重与BMI的共线性
        if 'height' in available_vars and 'weight' in available_vars:
            # 创建身高中心化变量
            df['height_centered'] = df['height'] - df['height'].mean()
            additional_vars.append('height_centered')
        
        # 构建最终公式 - Y染色体浓度作为因变量，孕妇代码用于分组但不作为自变量
        all_vars = base_vars + additional_vars
        formula = 'y_concentration ~ ' + ' + '.join(all_vars)
        
        print(f"模型公式: {formula}")
        
        try:
            # 拟合混合效应模型
            groups = df['subject_id']
            self.model = mixedlm(formula, df, groups=groups, re_formula='1')
            result = self.model.fit()
            
            print("扩展混合效应模型拟合成功！")
            
            # 保存模型结果
            self.model_result = result
            self.save_model_results(result, df)
            
            return result, df
            
        except Exception as e:
            print(f"模型拟合失败: {e}")
            return None, None
    
    def predict_y_concentration_mean(self, features_dict, gestational_week):
        """
        基于多维特征预测Y染色体浓度平均值
        
        参数:
        features_dict: 特征字典，包含BMI、年龄等
        gestational_week: 孕周
        
        返回:
        预测的Y染色体浓度平均值
        """
        if self.model_result is None:
            raise ValueError("模型尚未拟合，请先调用 fit_multivariate_model()")
        
        # 获取模型参数
        params = self.model_result.params
        
        # 计算预测值
        predicted_y = params.get('Intercept', 0)
        predicted_y += params.get('week', 0) * gestational_week
        
        # 添加其他特征的贡献
        for feature, value in features_dict.items():
            if feature in params.index:
                predicted_y += params[feature] * value
        
        return predicted_y
    
    def calculate_success_probability(self, features_dict, gestational_week, method='normal'):
        """
        计算成功概率 p_s(t, X_i) = P(C_i(t) >= C_min)
        基于扩展的混合效应模型
        
        参数:
        features_dict: 特征字典，包含BMI、年龄等
        gestational_week: 孕周
        method: 概率计算方法 (只支持 'normal')
        
        返回:
        成功概率值
        """
        if method != 'normal':
            raise ValueError("method参数只支持 'normal'")
        
        # 预测Y染色体浓度
        predicted_concentration = self.predict_y_concentration_mean(features_dict, gestational_week)
        
        # 计算成功概率
        return self._calculate_normal_probability(predicted_concentration, features_dict, gestational_week)
    
    def _calculate_normal_probability(self, predicted_concentration, features_dict, gestational_week):
        """
        基于混合效应模型计算成功概率，采用孕周依赖的总标准差
        """
        if self.model_result is None:
            raise ValueError("模型尚未拟合")
        
        # 获取模型参数
        sigma_u_sq = self.model_result.cov_re.iloc[0,0]  # 随机截距方差
        sigma_e_sq = self.model_result.scale  # 残差方差
        
        # 基础方差（随机截距 + 残差）
        base_variance = sigma_u_sq + sigma_e_sq
        base_std = np.sqrt(base_variance)
        
        # 孕周依赖方差放大系数：早期更大，随孕周递减至1
        # 加入BMI依赖：高BMI早期不确定性更大
        bmi = features_dict.get('bmi', 30.0)  # 默认BMI为30
        alpha_base = 0.85
        alpha_bmi = 0.025 * (bmi - 30.0)
        alpha = max(0.6, min(1.6, alpha_base + alpha_bmi))
        scale = 30.0
        multiplier = 1.0 + alpha * np.exp(-(gestational_week - 118.0) / scale)
        multiplier = float(max(1.0, multiplier))  # 不小于1
        
        total_std = base_std * multiplier
        
        # 计算成功概率 P(C >= C_min)
        z_score = (self.c_min - predicted_concentration) / total_std
        success_prob = 1 - stats.norm.cdf(z_score)
        
        return max(0, min(1, success_prob))
    
    def calculate_achievement_time(self, features_dict, p_target=0.75, week_range=(80, 180)):
        """
        计算个体达标时间 T(x) = inf{t: p_s(t|x) >= p_target}
        
        参数:
        features_dict: 特征字典
        p_target: 目标成功概率
        week_range: 搜索的孕周范围
        
        返回:
        达标时间（孕周）
        """
        week_min, week_max = week_range
        
        # 在孕周范围内搜索
        for week in range(week_min, week_max + 1):
            success_prob = self.calculate_success_probability(features_dict, week)
            if success_prob >= p_target:
                return week
        
        # 如果未找到，返回最大孕周
        return week_max

    def calculate_individual_risk_scores(self, result, df, p_target=0.75):
        """计算每个孕妇的个体风险评分（基于成功概率和达标时间）"""
        print("正在计算个体风险评分...")
        
        # 计算每个个体的风险评分（达标时间）
        risk_scores = []
        
        for subject_id in df['subject_id'].unique():
            subject_data = df[df['subject_id'] == subject_id]
            
            # 使用该个体的平均特征（包括GC含量）
            feature_cols = ['bmi', 'age', 'pregnancy_count', 'delivery_count', 'gc_content']
            available_feature_cols = [col for col in feature_cols if col in subject_data.columns]
            avg_features = subject_data[available_feature_cols].mean()
            
            # 构建特征字典（不包括孕周，因为孕周是时间变量）
            features_dict = {}
            for col in available_feature_cols:
                if col != 'week':  # 孕周不作为特征，而是时间变量
                    features_dict[col] = avg_features[col]
            
            # 计算达标时间（个体风险评分）
            achievement_time = self.calculate_achievement_time(features_dict, p_target)
            
            # 计算当前平均孕周的成功概率
            avg_week = subject_data['week'].mean()
            current_success_prob = self.calculate_success_probability(features_dict, avg_week)
            
            # 计算预测的Y染色体浓度（用于参考）
            predicted_y = self.predict_y_concentration_mean(features_dict, avg_week)
            
            risk_scores.append({
                'subject_id': subject_id,
                'risk_score': achievement_time,  # 达标时间作为风险评分
                'achievement_time': achievement_time,
                'current_success_prob': current_success_prob,
                'predicted_y': predicted_y,
                'avg_bmi': avg_features['bmi'],
                'avg_age': avg_features['age'],
                'avg_week': avg_week
            })
        
        self.risk_scores = pd.DataFrame(risk_scores)
        
        # 保存风险评分
        self.risk_scores.to_csv(os.path.join(self.save_dir, 'q3_individual_risk_scores.csv'), 
                               index=False, encoding='utf-8-sig')
        
        print(f"个体风险评分计算完成，共{len(self.risk_scores)}位孕妇")
        print(f"达标时间范围: {self.risk_scores['achievement_time'].min():.1f} - {self.risk_scores['achievement_time'].max():.1f} 孕周")
        print(f"当前成功概率范围: {self.risk_scores['current_success_prob'].min():.4f} - {self.risk_scores['current_success_prob'].max():.4f}")
        
        return self.risk_scores
    
    def save_model_results(self, result, df):
        """保存模型结果"""
        # 1. 保存模型系数
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
        coef_df.to_csv(os.path.join(self.save_dir, 'q3_multivar_model_coefs.csv'), 
                      index=False, encoding='utf-8-sig')
        
        # 2. 保存模型信息
        with open(os.path.join(self.save_dir, 'q3_multivar_model_info.md'), 'w', encoding='utf-8') as f:
            f.write("# 扩展混合效应模型结果\n")
            f.write("=" * 50 + "\n\n")
            f.write("## 模型形式\n")
            f.write("Y_ij = (β₀ + u_i) + β₁·Week_ij + β₂·BMI_ij + β₃·Age_ij + β₄·GC_ij + ... + ε_ij\n\n")
            f.write("其中：\n")
            f.write("- Y_ij: 第i位孕妇第j次检测的Y染色体浓度\n")
            f.write("- β₀: 固定截距（全体平均基线）\n")
            f.write("- u_i: 第i位孕妇的随机截距（个体差异）\n")
            f.write("- β₁, β₂, β₃, β₄, ...: 各变量的固定效应系数（包括GC含量）\n")
            f.write("- ε_ij: 残差项\n")
            f.write("- 注意：孕妇代码仅用于分组，不作为自变量\n\n")
            
            f.write("## 模型参数\n")
            f.write(f"样本数: {result.nobs}\n")
            f.write(f"组数（孕妇数）: {len(result.random_effects)}\n")
            f.write(f"平均每组样本数: {result.nobs / len(result.random_effects):.2f}\n")
            f.write(f"对数似然: {result.llf:.4f}\n")
            f.write(f"AIC: {result.aic:.4f}\n")
            f.write(f"BIC: {result.bic:.4f}\n\n")
            
            f.write("## 固定效应\n")
            f.write("| 变量 | 系数 | 标准误 | t值 | p值 |\n")
            f.write("|------|------|--------|-----|-----|\n")
            
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
        
        print("模型结果已保存")

def main():
    """主函数"""
    # 初始化
    model = MultivariateRiskModel('./data/output.csv', './3', c_min=0.04)
    
    # 加载和准备数据
    data = model.load_and_prepare_data()
    
    # 拟合扩展模型
    result, df = model.fit_multivariate_model()
    
    if result is not None:
        # 计算个体风险评分
        risk_scores = model.calculate_individual_risk_scores(result, df)
        
        # 演示成功概率函数的使用
        print("\n=== 成功概率函数演示 ===")
        example_features = {
            'bmi': 30.0,
            'age': 28.0,
            'pregnancy_count': 1.0,
            'delivery_count': 0.0,
            'gc_content': 0.5
        }
        
        # 计算不同孕周的成功概率
        for week in [100, 120, 140, 160]:
            success_prob = model.calculate_success_probability(example_features, week)
            print(f"孕周 {week}: 成功概率 = {success_prob:.4f}")
        
        # 计算达标时间
        achievement_time = model.calculate_achievement_time(example_features, p_target=0.75)
        print(f"达标时间（75%成功概率）: {achievement_time} 孕周")
        
        print("\n扩展混合效应模型和个体风险评分计算完成！")
        print("所有结果已保存到 ./3 目录")
    else:
        print("模型拟合失败！")

if __name__ == "__main__":
    main()
