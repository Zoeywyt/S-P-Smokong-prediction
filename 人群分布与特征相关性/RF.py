import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pandas.api.types import CategoricalDtype
from sklearn.ensemble import RandomForestClassifier


def auto_select_n_factors(X, var_threshold=0.8, max_factors=10):
    pca = PCA(n_components=min(max_factors, X.shape[1]), random_state=0)
    pca.fit(X)
    explained = np.cumsum(pca.explained_variance_ratio_)
    n_factors = np.argmax(explained >= var_threshold) + 1
    return n_factors, pca


def auto_select_n_clusters(X, min_k=2, max_k=7):
    best_k = min_k
    best_score = -1
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def analyze_subjective_factors_auto(data, subjective_features, unordered_vars, var_threshold=0.8, max_factors=10,
                                    min_k=2, max_k=7):
    subj_df = data[subjective_features].copy()
    for col in subjective_features:
        if col in unordered_vars:
            subj_df[col] = LabelEncoder().fit_transform(subj_df[col].astype(str))
    subj_df = subj_df.fillna(subj_df.median())
    scaler = StandardScaler()
    X = scaler.fit_transform(subj_df)
    n_factors, pca = auto_select_n_factors(X, var_threshold, max_factors)
    X_pca = pca.transform(X)[:, :n_factors]
    factor_names = [f'SubjF{i + 1}' for i in range(n_factors)]
    for i, name in enumerate(factor_names):
        data[name] = X_pca[:, i]
    n_clusters = auto_select_n_clusters(X_pca, min_k, max_k)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['subj_cluster'] = kmeans.fit_predict(X_pca)
    return data, pca, factor_names, n_factors, n_clusters


# 主流程
def process_occp(occp_val, data, subjective_features, objective_features, unordered_vars):
    pc1_file = f'occp_{occp_val}_PC1group_subjective_compare.xlsx'
    if not os.path.exists(pc1_file):
        print(f'职业{occp_val} 无PC1得分文件，跳过')
        return None
    pc1_df = pd.read_excel(pc1_file)
    if 'id' not in pc1_df.columns or 'PC1' not in pc1_df.columns:
        print(f'职业{occp_val} PC1文件缺少必要列，跳过')
        return None
    pc1_df['PC1_group'] = pd.qcut(pc1_df['PC1'], 4, labels=[f'G{i + 1}' for i in range(4)])
    merged = pd.merge(data, pc1_df[['id', 'PC1_group']], on='id', how='left')
    merged, pca_subj, factor_names, n_factors_auto, n_clusters_auto = analyze_subjective_factors_auto(
        merged, subjective_features, unordered_vars, var_threshold=0.8, max_factors=10, min_k=2, max_k=7)
    print(f'职业{occp_val} 自动选定主观因子数: {n_factors_auto}, 聚类数: {n_clusters_auto}')

    results = []
    for cluster in range(n_clusters_auto):
        cluster_data = merged[merged['subj_cluster'] == cluster]
        if len(cluster_data) == 0:
            continue

        # RF特征重要性分析（主观特征 + 客观特征 + 主观因子）
        feature_cols = subjective_features + objective_features + [f'SubjF{i + 1}' for i in range(n_factors_auto)]
        X_feat = cluster_data[feature_cols].copy()
        y_feat = cluster_data['sign']
        for col in X_feat.select_dtypes(include='object').columns:
            X_feat[col] = LabelEncoder().fit_transform(X_feat[col].astype(str))
        X_feat = X_feat.fillna(X_feat.median())
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_feat, y_feat)
        importances = pd.Series(rf.feature_importances_, index=X_feat.columns).sort_index()

        result = {
            'occp': occp_val,
            'subj_cluster': cluster
        }
        result.update(importances)
        results.append(result)

    return pd.DataFrame(results)


if __name__ == '__main__':
    raw_path = r'D:\model\PII\BorderlineSMOTE_K5_smote_all_decoded.xlsx'
    subjective_features = ['BD1_11', 'HE_BMI', 'BD2_1', 'BP1', 'BO1', 'BO2_1', 'BO1_1', 'D_1_1', 'pa_aerobic', 'BE8_1']
    objective_features = ['sex', 'age', 'incm5',  'educ', 'town_t', 'marri_1', 'BS2_1', 'BS6_3', 'BD2', 'BS13']
    unordered_vars = ['sex', 'occp', 'marri_1', 'BS13', 'town_t', 'pa_aerobic', 'BO1_1', 'BO2_1']
    ordered_vars = ['incm5', 'age', 'educ', 'BS2_1', 'BD2', 'D_1_1', 'BO1', 'HE_BMI', 'BP1', 'BE8_1',
                    'BD2_1', 'BD1_11']

    data = pd.read_excel(raw_path)
    if 'id' not in data.columns:
        data = data.reset_index(drop=True)
        data['id'] = data.index
        data.to_excel(raw_path, index=False)

    all_results = []
    for occp_val, group in data.groupby('occp'):
        print(f'正在处理职业{occp_val}...')
        result = process_occp(occp_val, group, subjective_features, objective_features, unordered_vars)
        if result is not None:
            all_results.append(result)

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_excel('all_occp_feature_importance.xlsx', index=False)
    else:
        print('未得到任何有效结果，请检查数据和文件路径。')
    