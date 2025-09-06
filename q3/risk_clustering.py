import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class RiskStratification:
    """
    基于风险评分的分层和BMI分组映射
    """
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.risk_scores = None
        self.clustering_results = None
        self.bmi_groups = None
        self.decision_tree = None
        
    def load_risk_scores(self, risk_scores_path):
        """加载个体风险评分"""
        print("正在加载个体风险评分...")
        self.risk_scores = pd.read_csv(risk_scores_path)
        print(f"加载完成，共{len(self.risk_scores)}位孕妇")
        return self.risk_scores
    
    def determine_optimal_k(self, max_k=8):
        """使用肘部法则确定最优聚类数K"""
        print("正在确定最优聚类数...")
        
        risk_data = self.risk_scores[['risk_score']].values
        
        # 计算不同K值的惯性（inertia）
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(risk_data)
            inertias.append(kmeans.inertia_)
        
        # 计算肘部点（二阶导数最大点）
        if len(inertias) >= 3:
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                second_derivatives.append(second_deriv)
            
            optimal_k = k_range[np.argmax(second_derivatives) + 1]
        else:
            optimal_k = 3  # 默认值
        
        # 可视化肘部法则
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'最优K={optimal_k}')
        plt.xlabel('聚类数 K')
        plt.ylabel('惯性 (Inertia)')
        plt.title('肘部法则确定最优聚类数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'q3_elbow_method.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"最优聚类数: {optimal_k}")
        return optimal_k
    
    def perform_risk_clustering(self, k=None):
        """对风险评分进行K-means聚类"""
        print("正在进行风险分层聚类...")
        
        if k is None:
            k = self.determine_optimal_k()
        
        # 准备数据
        risk_data = self.risk_scores[['risk_score']].values
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(risk_data)
        
        # 创建聚类结果
        self.clustering_results = self.risk_scores.copy()
        self.clustering_results['风险分层ID'] = cluster_labels
        
        # 按风险评分排序分层
        risk_stratum_order = self.clustering_results.groupby('风险分层ID')['risk_score'].mean().sort_values().index
        stratum_mapping = {old_id: new_id for new_id, old_id in enumerate(risk_stratum_order)}
        self.clustering_results['风险分层ID'] = self.clustering_results['风险分层ID'].map(stratum_mapping)
        
        # 保存聚类结果
        self.clustering_results.to_csv(os.path.join(self.save_dir, 'q3_risk_stratification.csv'), 
                                     index=False, encoding='utf-8-sig')
        
        print(f"风险分层完成，共{k}个风险分层")
        
        return self.clustering_results
    
    def map_risk_to_bmi_groups(self):
        """使用数据驱动的BMI分组规则"""
        print("正在使用数据驱动的BMI分组规则...")
        
        # 分析BMI分布，找到更自然的切割点
        bmi_data = self.clustering_results['avg_bmi']
        
        # 使用分位数方法，但调整为更"真实"的数值
        # 避免整数边界，使用更自然的切割点
        q25 = np.percentile(bmi_data, 25)
        q50 = np.percentile(bmi_data, 50) 
        q75 = np.percentile(bmi_data, 75)
        
        # 将切割点调整为更自然的数值（保留一位小数，避免整数）
        natural_cutpoints = [
            round(q25, 1),  # 25分位数
            round(q50, 1),  # 50分位数  
            round(q75, 1)   # 75分位数
        ]
        
        # 确保切割点之间有合理的间隔（至少1.5）
        final_cutpoints = []
        for i, cutpoint in enumerate(natural_cutpoints):
            if i == 0 or cutpoint - final_cutpoints[-1] >= 1.5:
                final_cutpoints.append(cutpoint)
        
        print(f"数据驱动的切割点: {final_cutpoints}")
        print(f"原始分位数: 25%={q25:.2f}, 50%={q50:.2f}, 75%={q75:.2f}")
        
        # 创建BMI分组
        self.bmi_groups = self.create_bmi_groups(final_cutpoints)
        
        # 分析每个BMI组的风险分层分布
        self._analyze_natural_groups()
        
        # 更新风险分层结果，添加BMI分组信息
        self.clustering_results = self.bmi_groups.copy()
        self.clustering_results.to_csv(os.path.join(self.save_dir, 'q3_risk_stratification.csv'), 
                                     index=False, encoding='utf-8-sig')
        
        print(f"数据驱动BMI分组完成，切割点: {final_cutpoints}")
        
        return self.bmi_groups, final_cutpoints
    
    def _analyze_clinical_groups(self):
        """分析临床分组的风险分布"""
        print("正在分析临床BMI分组的风险分布...")
        
        # 分析每个BMI组的风险分层分布
        group_analysis = []
        
        for group_id in sorted(self.bmi_groups['BMI分组ID'].unique()):
            group_data = self.bmi_groups[self.bmi_groups['BMI分组ID'] == group_id]
            
            # 计算该组的统计信息
            group_stats = {
                'BMI分组ID': group_id,
                'BMI分组': f'组{group_id}',
                '样本数': len(group_data),
                'BMI范围': f"{group_data['avg_bmi'].min():.2f} - {group_data['avg_bmi'].max():.2f}",
                'BMI均值': group_data['avg_bmi'].mean(),
                'BMI标准差': group_data['avg_bmi'].std(),
                '平均风险评分': group_data['risk_score'].mean(),
                '风险评分标准差': group_data['risk_score'].std(),
                '主要风险分层': group_data['风险分层ID'].mode().iloc[0] if len(group_data['风险分层ID'].mode()) > 0 else -1
            }
            
            # 风险分层分布
            risk_distribution = group_data['风险分层ID'].value_counts().to_dict()
            group_stats['风险分层分布'] = risk_distribution
            
            group_analysis.append(group_stats)
        
        # 保存分析结果
        group_analysis_df = pd.DataFrame(group_analysis)
        group_analysis_df.to_csv(os.path.join(self.save_dir, 'q3_clinical_groups_analysis.csv'), 
                                index=False, encoding='utf-8-sig')
        
        # 保存详细分析报告
        with open(os.path.join(self.save_dir, 'q3_clinical_groups_report.txt'), 'w', encoding='utf-8') as f:
            f.write("临床BMI分组风险分析报告:\n")
            f.write("=" * 60 + "\n\n")
            
            for group_id in sorted(self.bmi_groups['BMI分组ID'].unique()):
                group_data = self.bmi_groups[self.bmi_groups['BMI分组ID'] == group_id]
                
                f.write(f"BMI组{group_id}:\n")
                f.write(f"  样本数: {len(group_data)}\n")
                f.write(f"  BMI范围: {group_data['avg_bmi'].min():.2f} - {group_data['avg_bmi'].max():.2f}\n")
                f.write(f"  BMI均值: {group_data['avg_bmi'].mean():.2f} ± {group_data['avg_bmi'].std():.2f}\n")
                f.write(f"  平均风险评分: {group_data['risk_score'].mean():.4f} ± {group_data['risk_score'].std():.4f}\n")
                f.write(f"  风险分层分布: {group_data['风险分层ID'].value_counts().to_dict()}\n")
                f.write(f"  主要风险分层: {group_data['风险分层ID'].mode().iloc[0] if len(group_data['风险分层ID'].mode()) > 0 else '无'}\n\n")
        
        print("临床分组分析完成")
        return group_analysis_df
    
    def _analyze_natural_groups(self):
        """分析数据驱动分组的风险分布"""
        print("正在分析数据驱动BMI分组的风险分布...")
        
        # 分析每个BMI组的风险分层分布
        group_analysis = []
        
        for group_id in sorted(self.bmi_groups['BMI分组ID'].unique()):
            group_data = self.bmi_groups[self.bmi_groups['BMI分组ID'] == group_id]
            
            # 计算该组的统计信息
            group_stats = {
                'BMI分组ID': group_id,
                'BMI分组': f'组{group_id}',
                '样本数': len(group_data),
                'BMI范围': f"{group_data['avg_bmi'].min():.2f} - {group_data['avg_bmi'].max():.2f}",
                'BMI均值': group_data['avg_bmi'].mean(),
                'BMI标准差': group_data['avg_bmi'].std(),
                '平均风险评分': group_data['risk_score'].mean(),
                '风险评分标准差': group_data['risk_score'].std(),
                '主要风险分层': group_data['风险分层ID'].mode().iloc[0] if len(group_data['风险分层ID'].mode()) > 0 else -1
            }
            
            # 风险分层分布
            risk_distribution = group_data['风险分层ID'].value_counts().to_dict()
            group_stats['风险分层分布'] = risk_distribution
            
            group_analysis.append(group_stats)
        
        # 只保存简要分析结果
        group_analysis_df = pd.DataFrame(group_analysis)
        group_analysis_df.to_csv(os.path.join(self.save_dir, 'q3_bmi_groups_summary.csv'), 
                                index=False, encoding='utf-8-sig')
        
        print("数据驱动分组分析完成")
        return group_analysis_df
    
    def extract_bmi_cutpoints(self):
        """从决策树中提取BMI切割点"""
        tree = self.decision_tree.tree_
        cutpoints = []
        
        def extract_cutpoints_recursive(node, depth=0):
            if tree.children_left[node] != tree.children_right[node]:  # 不是叶子节点
                feature = tree.feature[node]
                threshold = tree.threshold[node]
                if feature == 0:  # BMI特征
                    cutpoints.append(threshold)
                extract_cutpoints_recursive(tree.children_left[node], depth + 1)
                extract_cutpoints_recursive(tree.children_right[node], depth + 1)
        
        extract_cutpoints_recursive(0)
        return sorted(cutpoints)
    
    def create_bmi_groups(self, cutpoints):
        """根据切割点创建BMI分组"""
        bmi_groups = self.clustering_results.copy()
        
        # 创建BMI分组标签
        bmi_groups['BMI分组'] = pd.cut(
            bmi_groups['avg_bmi'], 
            bins=[-np.inf] + cutpoints + [np.inf],
            labels=[f'组{i}' for i in range(len(cutpoints) + 1)]
        )
        
        # 重新编号为0, 1, 2, ...
        group_mapping = {f'组{i}': i for i in range(len(cutpoints) + 1)}
        bmi_groups['BMI分组ID'] = bmi_groups['BMI分组'].map(group_mapping)
        
        return bmi_groups
    
    def analyze_group_differences(self):
        """分析各组的差异"""
        print("正在分析各组差异...")
        
        # 按BMI分组分析
        group_analysis = self.bmi_groups.groupby('BMI分组ID').agg({
            'risk_score': ['count', 'mean', 'std'],
            'avg_bmi': ['mean', 'std', 'min', 'max'],
            'avg_age': ['mean', 'std'],
            '风险分层ID': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1
        }).round(4)
        
        group_analysis.columns = ['样本数', '风险评分均值', '风险评分标准差',
                                'BMI均值', 'BMI标准差', 'BMI最小值', 'BMI最大值',
                                '年龄均值', '年龄标准差', '主要风险分层']
        
        # 创建可视化
        self.create_visualization()
        
        print("组差异分析完成")
        return group_analysis
    
    def create_visualization(self):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 风险评分分布
        axes[0, 0].hist(self.clustering_results['risk_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('风险评分')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('风险评分分布')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 风险分层与BMI的关系
        for stratum_id in sorted(self.clustering_results['风险分层ID'].unique()):
            stratum_data = self.clustering_results[self.clustering_results['风险分层ID'] == stratum_id]
            axes[0, 1].scatter(stratum_data['avg_bmi'], stratum_data['risk_score'], 
                             label=f'风险分层{stratum_id}', alpha=0.7)
        axes[0, 1].set_xlabel('BMI')
        axes[0, 1].set_ylabel('风险评分')
        axes[0, 1].set_title('风险分层与BMI关系')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. BMI分组分布
        bmi_group_counts = self.bmi_groups['BMI分组ID'].value_counts().sort_index()
        axes[1, 0].bar(bmi_group_counts.index, bmi_group_counts.values, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('BMI分组ID')
        axes[1, 0].set_ylabel('样本数')
        axes[1, 0].set_title('BMI分组分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 风险分层与BMI分组的对应关系
        cross_tab = pd.crosstab(self.bmi_groups['风险分层ID'], self.bmi_groups['BMI分组ID'])
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('BMI分组ID')
        axes[1, 1].set_ylabel('风险分层ID')
        axes[1, 1].set_title('风险分层与BMI分组对应关系')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'q3_risk_stratification_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("可视化图表已保存")

def main():
    """主函数"""
    # 初始化
    stratifier = RiskStratification('./3')
    
    # 加载风险评分
    risk_scores = stratifier.load_risk_scores('./3/q3_individual_risk_scores.csv')
    
    # 进行风险分层聚类
    clustering_results = stratifier.perform_risk_clustering()
    
    # 映射到BMI分组
    bmi_groups, cutpoints = stratifier.map_risk_to_bmi_groups()
    
    # 分析组差异
    group_analysis = stratifier.analyze_group_differences()
    
    print("风险分层和BMI分组完成！")
    print("所有结果已保存到 ./3 目录")

if __name__ == "__main__":
    main()
