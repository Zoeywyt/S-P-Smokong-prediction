import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import dump  # 导入joblib库用于保存模型

def auto_select_n_factors(X, var_threshold=0.8, max_factors=10):
    pca = PCA(n_components=min(max_factors, X.shape[1]), random_state=0)
    pca.fit(X)
    explained = np.cumsum(pca.explained_variance_ratio_)
    n_factors = np.argmax(explained >= var_threshold) + 1
    return n_factors, pca

def auto_select_n_clusters(X, min_k=2, max_k=7):
    best_k = min_k
    best_score = -1
    for k in range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def analyze_subjective_group(df, subjective_features, ordered_vars, unordered_vars,
                           do_pc1=True, do_cluster=True, do_factors=True, do_rf=True):
    """
    统一主观特征分析流程：分职业+客观分型（obj_cluster），支持主观PC1分组、聚类、主观因子分析、RF特征重要性、合成可视化。
    """
    for occp_val, occp_group in df.groupby('occp'):
        for obj_cluster, sub_group in occp_group.groupby('obj_cluster'):
            data = sub_group.copy().reset_index(drop=True)
            # 编码无序变量
            subj_df = data[subjective_features].copy()
            for col in unordered_vars:
                if col in subj_df.columns:
                    subj_df[col] = LabelEncoder().fit_transform(subj_df[col].astype(str))
            subj_df = subj_df.fillna(subj_df.median())
            scaler = StandardScaler()
            X = scaler.fit_transform(subj_df)
            # 1. 主观PC1分组
            if do_pc1:
                pca_pc1 = PCA(n_components=1)
                pc1 = pca_pc1.fit_transform(X)[:, 0]
                data['SubjPC1'] = pc1
                data['SubjPC1_group'] = pd.qcut(data['SubjPC1'], 4, labels=[f'G{i+1}' for i in range(4)])
                # 可视化主观PC1分布、分组箱线图、载荷
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                sns.histplot(data['SubjPC1'], bins=20, kde=True, ax=axes[0])
                axes[0].set_title(f'Occupation {occp_val} ObjCluster {obj_cluster} - SubjPC1 Distribution')
                sns.boxplot(x='SubjPC1_group', y='SubjPC1', data=data, order=[f'G{i+1}' for i in range(4)], ax=axes[1])
                axes[1].set_title('SubjPC1 Group Boxplot')
                pc1_loadings = pd.Series(pca_pc1.components_[0], index=subj_df.columns)
                pc1_loadings.sort_values(key=np.abs, ascending=False).plot(kind='bar', color='salmon', ax=axes[2])
                axes[2].set_title('SubjPC1 Feature Loadings')
                plt.tight_layout()
                plt.savefig(f'职业{occp_val}_客观分型{obj_cluster}_主观PC1分布与载荷.png')
                plt.close()
            # 2. 主观特征聚类
            if do_cluster:
                n_clusters = auto_select_n_clusters(X, min_k=2, max_k=7)
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                data['subj_cluster'] = kmeans.fit_predict(X)
                # 保存聚类模型
                model_name = f'Occp{occp_val}_subj.joblib'
                dump(kmeans, model_name)

                # 聚类与戒烟状态分布
                if 'sign' in data.columns:
                    ctab = pd.crosstab(data['subj_cluster'], data['sign'], normalize='index')
                    ctab.plot(kind='bar', stacked=True, colormap='Set2')
                    plt.title(f'Occupation {occp_val} ObjCluster {obj_cluster} Subjective Cluster by Smoking Status')
                    plt.tight_layout()
                    plt.savefig(f'职业{occp_val}_客观分型{obj_cluster}_主观聚类类型与戒烟状态分布.png')
                    plt.close()
                # 有序/无序特征聚类均值/比例
                subj_ordered_vars = [v for v in ordered_vars if v in data.columns]
                if subj_ordered_vars:
                    means = data.groupby('subj_cluster')[subj_ordered_vars].mean()
                    sns.heatmap(means, annot=True, cmap='YlOrRd', fmt='.2f')
                    plt.title(f'Occupation {occp_val} ObjCluster {obj_cluster} Subjective Cluster Ordered Feature Means')
                    plt.tight_layout()
                    plt.savefig(f'职业{occp_val}_客观分型{obj_cluster}_主观聚类有序特征均值热力图.png')
                    plt.close()
                subj_unordered_vars = [v for v in unordered_vars if v in data.columns]
                if subj_unordered_vars:
                    n_vars = len(subj_unordered_vars)
                    fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 5), sharey=True)
                    if n_vars == 1:
                        axes = [axes]
                    for i, var in enumerate(subj_unordered_vars):
                        ctab = pd.crosstab(data['subj_cluster'], data[var], normalize='index')
                        ctab.plot(kind='bar', stacked=True, colormap='tab20', ax=axes[i], legend=False)
                        axes[i].set_title(f'{var} Proportion')
                        axes[i].set_xlabel('Subjective Cluster')
                        axes[i].set_ylabel('Proportion' if i == 0 else '')
                        axes[i].set_xticklabels(ctab.index, rotation=0)
                        if i == 0:
                            axes[i].legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left')
                        else:
                            if axes[i].legend_ is not None:
                                axes[i].legend_.remove()
                    plt.tight_layout()
                    plt.savefig(f'职业{occp_val}_客观分型{obj_cluster}_主观聚类无序特征类别比例.png')
                    plt.close()
            # 3. 主观因子分析+聚类+分箱+可视化
            if do_factors:
                n_factors, pca = auto_select_n_factors(X, var_threshold=0.8, max_factors=10)
                X_pca = pca.transform(X)[:, :n_factors]
                factor_names = [f'SubjF{i+1}' for i in range(n_factors)]
                for i, name in enumerate(factor_names):
                    data[name] = X_pca[:, i]
                n_clusters = auto_select_n_clusters(X_pca, min_k=2, max_k=7)
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                data['subj_cluster'] = kmeans.fit_predict(X_pca)
                # 主观因子分箱
                for f in factor_names:
                    data[f'{f}_bin'] = pd.qcut(data[f], 4, labels=[f'Q{i+1}' for i in range(4)])
                # 主观因子载荷热力图
                loadings = pd.DataFrame(pca.components_[:n_factors].T, index=subj_df.columns, columns=factor_names)
                sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt='.2f')
                plt.title(f'Occupation {occp_val} ObjCluster {obj_cluster} Subjective Factor Loadings')
                plt.tight_layout()
                plt.savefig(f'职业{occp_val}_客观分型{obj_cluster}_主观特征因子载荷热力图.png')
                plt.close()
                # 主观因子分箱与戒烟成功率
                if 'sign' in data.columns:
                    for f in factor_names:
                        sns.boxplot(x=f'{f}_bin', y='sign', data=data)
                        plt.title(f'Occupation {occp_val} ObjCluster {obj_cluster} {f} Bin vs. Smoking Cessation')
                        plt.tight_layout()
                        plt.savefig(f'职业{occp_val}_客观分型{obj_cluster}_{f}_分箱与戒烟成功率.png')
                        plt.close()
            # 4. RF特征重要性
            if do_rf and 'sign' in data.columns:
                from sklearn.ensemble import RandomForestClassifier
                feature_cols = subjective_features.copy()
                if do_factors:
                    feature_cols += factor_names
                if do_cluster or do_factors:
                    feature_cols += ['subj_cluster']
                X_feat = data[feature_cols].copy()
                y_feat = data['sign']
                for col in X_feat.select_dtypes(include='object').columns:
                    X_feat[col] = LabelEncoder().fit_transform(X_feat[col].astype(str))
                X_feat = X_feat.fillna(X_feat.median())
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_feat, y_feat)
                importances = pd.Series(rf.feature_importances_, index=X_feat.columns).sort_values(ascending=False)
                importances.plot(kind='bar', figsize=(10,4))
                plt.title(f'RF Feature Importance (occp={occp_val}, obj_cluster={obj_cluster})')
                plt.tight_layout()
                plt.savefig(f'职业{occp_val}_客观分型{obj_cluster}_主观特征重要性RF.png')
                plt.close()

# 主流程调用
input_file = r'D:\model\PII\客观聚类结果.xlsx'
df = pd.read_excel(input_file)
subjective_features = ['BD1_11', 'HE_BMI', 'BD2_1', 'BP1', 'BO1', 'BO2_1', 'BO1_1', 'D_1_1', 'pa_aerobic', 'BE8_1']
ordered_vars = ['HE_BMI', 'BD2_1', 'BP1', 'BO1', 'D_1_1', 'BE8_1','BD1_11']
unordered_vars = ['BO2_1', 'BO1_1', 'pa_aerobic']
if 'occp' not in df.columns:
    raise ValueError('数据中缺少职业(occp)列')
if 'obj_cluster' not in df.columns:
    raise ValueError('数据中缺少客观分型(obj_cluster)列')
if 'sign' not in df.columns:
    raise ValueError('数据中缺少戒烟状态(sign)列')

analyze_subjective_group(df, subjective_features, ordered_vars, unordered_vars,
                        do_pc1=True, do_cluster=True, do_factors=True, do_rf=True)
print('主观因子分析+聚类、客观分型分层统计已完成。')


# 保存主观特征分析结果到表格
output_file = '主观特征分析结果.xlsx'
all_results = []

for occp_val, occp_group in df.groupby('occp'):
    for obj_cluster, sub_group in occp_group.groupby('obj_cluster'):
        data = sub_group.copy().reset_index(drop=True)
        # 编码无序变量
        subj_df = data[subjective_features].copy()
        for col in unordered_vars:
            if col in subj_df.columns:
                subj_df[col] = LabelEncoder().fit_transform(subj_df[col].astype(str))
        subj_df = subj_df.fillna(subj_df.median())
        scaler = StandardScaler()
        X = scaler.fit_transform(subj_df)
        # 主观PC1分组
        pca_pc1 = PCA(n_components=1)
        pc1 = pca_pc1.fit_transform(X)[:, 0]
        data['SubjPC1'] = pc1
        data['SubjPC1_group'] = pd.qcut(data['SubjPC1'], 4, labels=[f'G{i+1}' for i in range(4)])
        # 主观特征聚类
        n_clusters = auto_select_n_clusters(X, min_k=2, max_k=7)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        data['subj_cluster'] = kmeans.fit_predict(X)
        # 主观因子分析
        n_factors, pca = auto_select_n_factors(X, var_threshold=0.8, max_factors=10)
        X_pca = pca.transform(X)[:, :n_factors]
        factor_names = [f'SubjF{i+1}' for i in range(n_factors)]
        for i, name in enumerate(factor_names):
            data[name] = X_pca[:, i]
        # 保存结果
        data['occp'] = occp_val
        data['obj_cluster'] = obj_cluster
        all_results.append(data)

# 合并所有结果并保存
final_results = pd.concat(all_results, ignore_index=True)
final_results.to_excel(output_file, index=False)
print(f'主观特征分析结果已保存到 {output_file}')