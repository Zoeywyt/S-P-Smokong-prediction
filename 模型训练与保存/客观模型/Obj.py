import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib  # 导入joblib库

# 配置
input_file = r'D:\model\PII\BorderlineSMOTE_K5_smote_all_decoded.xlsx'  # 输入数据文件
output_file = '客观聚类结果.xlsx'  # 输出文件
factor_cols = ['age', 'BS6_3', 'incm5', 'educ', 'BD2', 'BS2_1', 'sex', 'marri_1', 'BS13', 'town_t']
ordered_vars = ['age', 'BS6_3', 'incm5', 'educ', 'BD2', 'BS2_1']
unordered_vars = ['sex', 'marri_1', 'BS13', 'town_t']

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

# 读取数据
df = pd.read_excel(input_file)
if 'occp' not in df.columns:
    raise ValueError('数据中缺少职业(occp)列')

# 结果收集
df_result = []
for occp_val, group in df.groupby('occp'):
    group = group.copy().reset_index(drop=True)
    # 编码无序变量
    factor_df = group[factor_cols].copy()
    for col in unordered_vars:
        if col in factor_df.columns:
            factor_df[col] = LabelEncoder().fit_transform(factor_df[col].astype(str))
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(factor_df)
    # 客观特征聚类（分型）
    n_clusters = auto_select_n_clusters(X, min_k=2, max_k=7)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    group['obj_cluster'] = kmeans.fit_predict(X)
    
    # 保存模型
    model_filename = f'Occp{occp_val}_obj.joblib'
    joblib.dump(kmeans, model_filename)
    print(f'职业 {occp_val} 的KMeans模型已保存为 {model_filename}')

    # 因子分析
    pca = PCA(n_components=min(len(factor_cols), 10))  # 限制最多提取10个因子
    X_pca = pca.fit_transform(X)
    factor_names = [f'Factor{i+1}' for i in range(X_pca.shape[1])]
    for i, name in enumerate(factor_names):
        group[name] = X_pca[:, i]
    # 保存因子载荷
    loadings = pd.DataFrame(pca.components_.T, index=factor_cols, columns=factor_names)
    loadings.to_excel(f'职业{occp_val}_因子载荷.xlsx')
    
    # 将处理后的分组添加到结果列表中
    df_result.append(group)

# 合并结果并保存
df_out = pd.concat(df_result, ignore_index=True)
df_out.to_excel(output_file, index=False)