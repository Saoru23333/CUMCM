#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于BMI的K-means聚类分析
使用Elbow方法确定最优聚类数，然后对数据进行分组
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BMIKMeansClusterer:
    """
    基于BMI的K-means聚类分析器
    """
    
    def __init__(self, data_path):
        """
        初始化聚类分析器
        
        参数:
        data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.bmi_data = None
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.optimal_k = None
        self.elbow_scores = []
        self.silhouette_scores = []
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        # 尝试加载GAM预测结果
        if os.path.exists('./1/gam_predictions.csv'):
            self.data = pd.read_csv('./1/gam_predictions.csv')
            print(f"从GAM预测结果加载了 {len(self.data)} 条记录")
        else:
            # 如果没有GAM结果，加载原始数据
            self.data = pd.read_csv(self.data_path)
            print(f"从原始数据加载了 {len(self.data)} 条记录")
        
        # 提取BMI数据
        if 'BMI' in self.data.columns:
            self.bmi_data = self.data[['BMI']].copy()
        elif '孕妇BMI' in self.data.columns:
            self.bmi_data = self.data[['孕妇BMI']].copy()
            self.bmi_data.columns = ['BMI']
        else:
            raise ValueError("数据中未找到BMI列")
        
        # 移除缺失值
        self.bmi_data = self.bmi_data.dropna()
        print(f"有效BMI数据: {len(self.bmi_data)} 条")
        
        return self.bmi_data
    
    def find_optimal_k(self, max_k=10):
        """
        使用Elbow方法找到最优聚类数
        
        参数:
        max_k: 最大聚类数
        
        返回:
        最优聚类数
        """
        print("正在使用Elbow方法确定最优聚类数...")
        
        # 标准化数据
        bmi_scaled = self.scaler.fit_transform(self.bmi_data)
        
        # 计算不同k值的惯性（inertia）
        k_range = range(2, max_k + 1)
        self.elbow_scores = []
        self.silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(bmi_scaled)
            
            # 计算惯性（组内平方和）
            inertia = kmeans.inertia_
            self.elbow_scores.append(inertia)
            
            # 计算轮廓系数
            silhouette_avg = silhouette_score(bmi_scaled, kmeans.labels_)
            self.silhouette_scores.append(silhouette_avg)
            
            print(f"k={k}: 惯性={inertia:.2f}, 轮廓系数={silhouette_avg:.3f}")
        
        # 使用肘部法则确定最优k
        # 计算二阶导数来找到肘部点
        if len(self.elbow_scores) >= 3:
            # 计算一阶导数
            first_derivative = np.diff(self.elbow_scores)
            # 计算二阶导数
            second_derivative = np.diff(first_derivative)
            
            # 找到二阶导数最大的点（肘部点）
            elbow_idx = np.argmax(second_derivative) + 2  # +2因为二阶导数比原数组少2个元素
            self.optimal_k = k_range[elbow_idx]
        else:
            # 如果数据点太少，选择轮廓系数最大的k
            self.optimal_k = k_range[np.argmax(self.silhouette_scores)]
        
        print(f"最优聚类数: {self.optimal_k}")
        return self.optimal_k
    
    def perform_clustering(self, k=None):
        """
        执行K-means聚类
        
        参数:
        k: 聚类数，如果为None则使用最优k
        
        返回:
        聚类结果
        """
        if k is None:
            k = self.optimal_k
        
        print(f"正在执行K-means聚类 (k={k})...")
        
        # 标准化数据
        bmi_scaled = self.scaler.fit_transform(self.bmi_data)
        
        # 执行K-means聚类
        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(bmi_scaled)
        
        # 将聚类结果添加到数据中
        self.bmi_data['聚类标签'] = cluster_labels
        self.bmi_data['聚类中心距离'] = self.kmeans_model.transform(bmi_scaled).min(axis=1)
        
        # 计算每个聚类的统计信息
        cluster_stats = self.bmi_data.groupby('聚类标签')['BMI'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        cluster_stats.columns = ['样本数', '平均BMI', 'BMI标准差', '最小BMI', '最大BMI']
        
        print("聚类结果统计:")
        print(cluster_stats)
        
        return cluster_labels, cluster_stats
    
    def visualize_elbow_method(self, save_path=None):
        """
        可视化Elbow方法结果
        
        参数:
        save_path: 保存路径
        """
        if not self.elbow_scores:
            print("请先运行find_optimal_k方法")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        k_range = range(2, len(self.elbow_scores) + 2)
        
        # 绘制肘部图
        ax1.plot(k_range, self.elbow_scores, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=self.optimal_k, color='red', linestyle='--', alpha=0.7, 
                   label=f'最优k={self.optimal_k}')
        ax1.set_xlabel('聚类数 (k)', fontsize=12)
        ax1.set_ylabel('惯性 (Inertia)', fontsize=12)
        ax1.set_title('Elbow方法 - 惯性随聚类数变化', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制轮廓系数图
        ax2.plot(k_range, self.silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=self.optimal_k, color='red', linestyle='--', alpha=0.7,
                   label=f'最优k={self.optimal_k}')
        ax2.set_xlabel('聚类数 (k)', fontsize=12)
        ax2.set_ylabel('轮廓系数 (Silhouette Score)', fontsize=12)
        ax2.set_title('轮廓系数随聚类数变化', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Elbow方法图已保存到: {save_path}")
        
        plt.show()
    
    def visualize_clustering_results(self, save_path=None):
        """
        可视化聚类结果
        
        参数:
        save_path: 保存路径
        """
        if self.kmeans_model is None:
            print("请先运行perform_clustering方法")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制BMI分布直方图（按聚类着色）
        colors = plt.cm.Set3(np.linspace(0, 1, self.optimal_k))
        
        for i in range(self.optimal_k):
            cluster_data = self.bmi_data[self.bmi_data['聚类标签'] == i]['BMI']
            ax1.hist(cluster_data, bins=20, alpha=0.7, color=colors[i], 
                    label=f'聚类 {i} (n={len(cluster_data)})')
        
        ax1.set_xlabel('BMI', fontsize=12)
        ax1.set_ylabel('频数', fontsize=12)
        ax1.set_title('BMI分布 - 按聚类分组', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制聚类中心
        cluster_centers = self.kmeans_model.cluster_centers_
        cluster_centers_original = self.scaler.inverse_transform(cluster_centers)
        
        # 绘制BMI散点图
        scatter = ax2.scatter(range(len(self.bmi_data)), self.bmi_data['BMI'], 
                            c=self.bmi_data['聚类标签'], cmap='Set3', alpha=0.6)
        
        # 添加聚类中心线
        for i, center in enumerate(cluster_centers_original):
            ax2.axhline(y=center[0], color=colors[i], linestyle='--', alpha=0.8,
                       label=f'聚类 {i} 中心: {center[0]:.2f}')
        
        ax2.set_xlabel('样本索引', fontsize=12)
        ax2.set_ylabel('BMI', fontsize=12)
        ax2.set_title('BMI聚类结果', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"聚类结果图已保存到: {save_path}")
        
        plt.show()
    
    def analyze_clusters(self):
        """
        分析聚类结果
        
        返回:
        聚类分析结果
        """
        if self.kmeans_model is None:
            print("请先运行perform_clustering方法")
            return None
        
        print("=" * 60)
        print("BMI聚类分析结果")
        print("=" * 60)
        
        # 基本统计
        cluster_stats = self.bmi_data.groupby('聚类标签')['BMI'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        cluster_stats.columns = ['样本数', '平均BMI', 'BMI标准差', '最小BMI', '最大BMI']
        
        print("各聚类统计信息:")
        print(cluster_stats)
        print()
        
        # 聚类中心
        cluster_centers = self.kmeans_model.cluster_centers_
        cluster_centers_original = self.scaler.inverse_transform(cluster_centers)
        
        print("聚类中心 (原始尺度):")
        for i, center in enumerate(cluster_centers_original):
            print(f"聚类 {i}: BMI = {center[0]:.3f}")
        print()
        
        # 聚类质量评估
        bmi_scaled = self.scaler.transform(self.bmi_data[['BMI']])
        silhouette_avg = silhouette_score(bmi_scaled, self.bmi_data['聚类标签'])
        print(f"轮廓系数: {silhouette_avg:.3f}")
        print(f"惯性 (组内平方和): {self.kmeans_model.inertia_:.2f}")
        print()
        
        # 聚类标签和BMI范围
        print("聚类标签和BMI范围:")
        for i in range(self.optimal_k):
            cluster_data = self.bmi_data[self.bmi_data['聚类标签'] == i]
            bmi_min = cluster_data['BMI'].min()
            bmi_max = cluster_data['BMI'].max()
            bmi_mean = cluster_data['BMI'].mean()
            print(f"聚类 {i}: BMI范围 [{bmi_min:.2f}, {bmi_max:.2f}], 平均 {bmi_mean:.2f}")
        
        return {
            'cluster_stats': cluster_stats,
            'cluster_centers': cluster_centers_original,
            'silhouette_score': silhouette_avg,
            'inertia': self.kmeans_model.inertia_
        }
    
    def save_results(self, save_dir='./2'):
        """
        保存聚类结果
        
        参数:
        save_dir: 保存目录
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存聚类结果数据
        results_path = os.path.join(save_dir, 'bmi_clustering_results.csv')
        self.bmi_data.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"聚类结果已保存到: {results_path}")
        
        # 保存聚类统计信息
        if self.kmeans_model is not None:
            cluster_stats = self.bmi_data.groupby('聚类标签')['BMI'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(3)
            cluster_stats.columns = ['样本数', '平均BMI', 'BMI标准差', '最小BMI', '最大BMI']
            
            stats_path = os.path.join(save_dir, 'bmi_cluster_statistics.csv')
            cluster_stats.to_csv(stats_path, encoding='utf-8-sig')
            print(f"聚类统计信息已保存到: {stats_path}")
        
        # 保存分析报告
        report_path = os.path.join(save_dir, 'bmi_clustering_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BMI K-means聚类分析报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"数据来源: {self.data_path}\n")
            f.write(f"总样本数: {len(self.bmi_data)}\n")
            f.write(f"最优聚类数: {self.optimal_k}\n\n")
            
            if self.kmeans_model is not None:
                f.write("聚类统计信息:\n")
                f.write(cluster_stats.to_string())
                f.write("\n\n")
                
                f.write("聚类中心 (原始尺度):\n")
                cluster_centers = self.kmeans_model.cluster_centers_
                cluster_centers_original = self.scaler.inverse_transform(cluster_centers)
                for i, center in enumerate(cluster_centers_original):
                    f.write(f"聚类 {i}: BMI = {center[0]:.3f}\n")
                f.write("\n")
                
                f.write("聚类质量评估:\n")
                bmi_scaled = self.scaler.transform(self.bmi_data[['BMI']])
                silhouette_avg = silhouette_score(bmi_scaled, self.bmi_data['聚类标签'])
                f.write(f"轮廓系数: {silhouette_avg:.3f}\n")
                f.write(f"惯性 (组内平方和): {self.kmeans_model.inertia_:.2f}\n")
        
        print(f"分析报告已保存到: {report_path}")


def main():
    """主函数 - 演示BMI K-means聚类分析"""
    print("BMI K-means聚类分析")
    print("=" * 60)
    
    # 初始化聚类分析器
    clusterer = BMIKMeansClusterer('./data/output.csv')
    
    # 加载数据
    clusterer.load_data()
    
    # 找到最优聚类数
    optimal_k = clusterer.find_optimal_k(max_k=8)
    
    # 可视化Elbow方法结果
    clusterer.visualize_elbow_method(save_path='./2/bmi_elbow_method.png')
    
    # 执行聚类
    cluster_labels, cluster_stats = clusterer.perform_clustering()
    
    # 可视化聚类结果
    clusterer.visualize_clustering_results(save_path='./2/bmi_clustering_results.png')
    
    # 分析聚类结果
    analysis_results = clusterer.analyze_clusters()
    
    # 保存结果
    clusterer.save_results(save_dir='./2')
    
    print("\nBMI K-means聚类分析完成！")
    print("所有结果已保存到 ./2/ 目录")


if __name__ == "__main__":
    main()
