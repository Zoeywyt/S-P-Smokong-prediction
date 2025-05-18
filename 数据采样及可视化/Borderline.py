import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据读取
data = pd.read_excel(r'D:\model\PII\模型评估\戒烟预测与干预分析\数据采样及可视化\采样前数据.xlsx')
X = data.drop('sign', axis=1)
y = data['sign']

# 计算少数类
class_distribution = y.value_counts()
minority_class = class_distribution.idxmin()
majority_class = class_distribution.idxmax()
minority_count = class_distribution[minority_class]
majority_count = class_distribution[majority_class]
imbalance_ratio = minority_count / majority_count

print(f"类别分布：\n{class_distribution}")
print(f"少数类：{minority_class}，样本数量：{minority_count}")
print(f"多数类：{majority_class}，样本数量：{majority_count}")
print(f"不平衡比例：{imbalance_ratio}")

if imbalance_ratio < 0.3:
    print("数据集存在不平衡问题。")
else:
    print("数据集不存在明显的不平衡问题。")

# 手动指定特征类型（有序、无序、连续变量分开处理）
unordered_vars = ['sex', 'occp','marri_1', 'BS13', 'town_t', 'pa_aerobic', 'BO1_1', 'BO2_1']
ordered_vars = ['incm5', 'ho_incm5', 'age', 'educ', 'BS2_1', 'BD2', 'D_1_1', 'BO1', 'HE_BMI', 'BP1', 'BE8_1', 'BD2_1', 'BD1_11']
continuous_cols = ['age', 'ainc', 'BD2', 'HE_BMI', 'BS2_1', 'BS6_3', 'BE8_1']  # 手动指定连续型特征列名
categorical_cols = ['sex', 'incm5', 'ho_incm5', 'educ', 'occp', 'town_t','marri_1', 'D_1_1', 'BO1', 'BO1_1', 'BO2_1', 'BD1_11', 'BD2_1', 'BP1', 'BS13', 'pa_aerobic']  # 手动指定分类特征列名

# 检查变量是否在数据中
unordered_vars = [col for col in unordered_vars if col in X.columns]
ordered_vars = [col for col in ordered_vars if col in X.columns]
continuous_cols = [col for col in continuous_cols if col in X.columns]

# 定义不同的采样方法和K值
sampling_methods = {
    'SMOTE': [3, 5, 7],
    'BorderlineSMOTE': [3, 5, 7]
}

# 定义不同的模型
models = {
    'NB': GaussianNB(),
    'LR': LogisticRegression(max_iter=1000),
    'MLPNN': MLPClassifier(random_state=42),
    'RF': RandomForestClassifier(random_state=42),
    'GBT': GradientBoostingClassifier(random_state=42)
}

best_method = None
best_k = None
best_f1 = -1
best_decoded_data = []
all_metrics = {}

for method_name, ks in sampling_methods.items():
    for k in ks:
        decoded_data = []
        all_y_true = []
        all_y_pred = []
        all_group_metrics = []
        model_metrics = {model_name: {'accuracy': [], 'precision': [],'recall': [], 'f1-score': [], 'auc': []} for model_name in models.keys()}

        for occp_val, group in data.groupby('occp'):
            Xg = group.drop('sign', axis=1)
            yg = group['sign']
            # 删除含NaN的行
            valid_idx = Xg.dropna().index
            Xg = Xg.loc[valid_idx]
            yg = yg.loc[valid_idx]
            if yg.nunique() < 2:
                print(f'职业{occp_val}只有一个类别，跳过')
                continue
            # 连续变量和分类变量分开
            Xg_num = Xg[[col for col in Xg.columns if col not in categorical_cols]].copy()
            Xg_cat = Xg[categorical_cols].astype(str).copy()
            onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            Xg_cat_oh = onehot.fit_transform(Xg_cat)
            # 拼接连续+独热
            Xg_all = np.hstack([Xg_num.values, Xg_cat_oh])
            # 根据方法名选择采样方法
            if method_name == 'SMOTE':
                smote = SMOTE(random_state=42, k_neighbors=k)
            else:
                smote = BorderlineSMOTE(random_state=42, k_neighbors=k)
            Xg_res, yg_res = smote.fit_resample(Xg_all, yg)
            # 采样后划分训练/测试集
            X_train, X_test, y_train, y_test = train_test_split(Xg_res, yg_res, test_size=0.2, random_state=42, stratify=yg_res)

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                # 计算指标
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                precision = report['0']['precision']
                recall = report['0']['recall']
                f1 = report['0']['f1-score']
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan

                model_metrics[model_name]['accuracy'].append(accuracy)
                model_metrics[model_name]['precision'].append(precision)
                model_metrics[model_name]['recall'].append(recall)
                model_metrics[model_name]['f1-score'].append(f1)
                model_metrics[model_name]['auc'].append(auc)

                # 保存结果
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)

            # 记录分组指标
            all_group_metrics.append({
                'occp': occp_val,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': report['0']['precision'],
               'recall': report['0']['recall'],
                'f1-score': report['0']['f1-score'],
               'support': report['0']['support']
            })
            print(f'职业{occp_val}，{method_name}，K={k}评估完成，样本数: {len(y_test)}')
            # 还原OneHot编码为原始分类变量
            Xg_res_num = pd.DataFrame(Xg_res[:, :len(Xg_num.columns)], columns=Xg_num.columns)
            Xg_res_cat = pd.DataFrame(onehot.inverse_transform(Xg_res[:, len(Xg_num.columns):]), columns=onehot.feature_names_in_)
            # 除HE_BMI和ainc外，所有连续变量都转为整数
            for col in Xg_res_num.columns:
                if col not in ['HE_BMI', 'ainc']:
                    Xg_res_num[col] = np.round(Xg_res_num[col]).astype(int)
            Xg_res_df = pd.concat([Xg_res_num, Xg_res_cat], axis=1)[Xg.columns]
            Xg_res_df['sign'] = yg_res
            decoded_data.append(Xg_res_df)

        # 合并所有分组的预测结果，计算加权整体F1分数
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        current_f1 = f1_score(all_y_true, all_y_pred, average='binary', pos_label=0)
        print(f'{method_name}，K={k}的加权整体F1分数: {current_f1:.4f}')

        # 保存最优结果
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_method = method_name
            best_k = k
            best_decoded_data = decoded_data

        # 记录每个模型在该采样方法和K值下的平均指标
        for model_name in models.keys():
            avg_accuracy = np.mean(model_metrics[model_name]['accuracy'])
            avg_precision = np.mean(model_metrics[model_name]['precision'])
            avg_recall = np.mean(model_metrics[model_name]['recall'])
            avg_f1 = np.mean(model_metrics[model_name]['f1-score'])
            avg_auc = np.nanmean(model_metrics[model_name]['auc'])

            if (method_name, k) not in all_metrics:
                all_metrics[(method_name, k)] = {}
            all_metrics[(method_name, k)][model_name] = {
                'accuracy': avg_accuracy,
                'precision': avg_precision,
               'recall': avg_recall,
                'f1-score': avg_f1,
                'auc': avg_auc
            }

# 打印每个模型在最优采样方法和K值下的指标
# if best_method and best_k:
#     print(f"最优采样方法 {best_method}，K={best_k} 下各模型指标：")
#     for model_name in models.keys():
#         print(f"{model_name}: ")
#         print(f"  准确率: {all_metrics[(best_method, best_k)][model_name]['accuracy']:.4f}")
#         print(f"  精确率: {all_metrics[(best_method, best_k)][model_name]['precision']:.4f}")
#         print(f"  召回率: {all_metrics[(best_method, best_k)][model_name]['recall']:.4f}")
#         print(f"  F1分数: {all_metrics[(best_method, best_k)][model_name]['f1-score']:.4f}")
#         print(f"  AUC: {all_metrics[(best_method, best_k)][model_name]['auc']:.4f}")

# ========== 可视化加权F1分数和最优模型指标 =========
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Draw weighted F1 score line plot for K=3,5,7 for each sampling method (use RF model)
f1_plot_data = []
for (method_name, k), model_dict in all_metrics.items():
    # Use RF model's f1-score
    f1_plot_data.append({'Sampling Method': method_name, 'K': k, 'F1 Score': model_dict['RF']['f1-score']})
f1_df = pd.DataFrame(f1_plot_data)
print(f1_df)
# plt.figure(figsize=(8,6))
# sns.lineplot(data=f1_df, x='K', y='F1 Score', hue='Sampling Method', marker='o')
# plt.title('Weighted F1 Score by K for Each Sampling Method (RF)')
# plt.ylabel('Weighted F1 Score')
# plt.xlabel('K')
# plt.ylim(0, 1)
# plt.tight_layout()
# plt.savefig('sampling_f1_by_k_RF.png')
# plt.close()

# 2. Bar plot for main metrics of each model under BorderlineSMOTE K=5 (in English)
best_metrics = all_metrics.get(('BorderlineSMOTE', 5), {})
if best_metrics:
    best_plot_data = []
    for model_name, vals in best_metrics.items():
        best_plot_data.append({'Model': model_name, 'Accuracy': vals['accuracy'], 'Precision': vals['precision'], 'Recall': vals['recall'], 'F1 Score': vals['f1-score'], 'AUC': vals['auc']})
    best_df = pd.DataFrame(best_plot_data)
    best_df_melt = best_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    # plt.figure(figsize=(10,6))
    # sns.barplot(data=best_df_melt, x='Model', y='Score', hue='Metric')
    # plt.title('BorderlineSMOTE K=5 Model Metrics')
    # plt.ylim(0, 1)
    # plt.tight_layout()
    # plt.savefig('BorderlineSMOTE_K5_model_metrics_english.png')
    # plt.close()
    # print('Saved BorderlineSMOTE K=5 model metrics comparison chart (English).')
else:
    print('No BorderlineSMOTE K=5 model metrics found, cannot plot.')

# 保存最优采样方法处理后的数据
if best_decoded_data:
    best_smote_all_decoded_df = pd.concat(best_decoded_data, ignore_index=True)
    best_smote_all_decoded_df.to_excel(f'{best_method}_K{best_k}_smote_all_decoded.xlsx', index=False)
    print(f'最优采样方法{best_method}，K={best_k}处理后的数据已保存为 {best_method}_K{best_k}_smote_all_decoded.xlsx')

print(f'最优采样方法为{best_method}，K={best_k}，加权整体F1分数为{best_f1:.4f}')
