import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# 输入数据
input_data = {
    'age': [37],
    'BS6_3': [5],
    'incm5': [3],
    'educ': [4],
    'BD2': [20],
    'BS2_1': [20],
    'sex': [1],
    'marri_1': [1],
    'BS13': [2],
    'town_t': [1],
    'BD1_11': [2],
    'HE_BMI': [22],
    'BD2_1': [3],
    'BP1': [3],
    'BO1': [4],
    'BO2_1': [4],
    'BO1_1': [3],
    'D_1_1': [3],
    'pa_aerobic': [0],
    'BE8_1': [10],
    'occp': [3]
}
unordered_vars = ['BO2_1', 'BO1_1', 'pa_aerobic', 'sex', 'marri_1', 'BS13', 'town_t']
ordered_vars = ['HE_BMI', 'BD2_1', 'BP1', 'BO1', 'D_1_1', 'BE8_1', 'BD1_11', 'age', 'BS6_3', 'incm5', 'educ', 'BD2', 'BS2_1']
# 将输入数据转换为 DataFrame
df_input = pd.DataFrame(input_data)

# 职业序号
X1 = df_input['occp'].values[0]  # 直接使用整数类型的值

# 客观特征相关配置
obj_factor_cols = ['age', 'BS6_3', 'incm5', 'educ', 'BD2', 'BS2_1', 'sex', 'marri_1', 'BS13', 'town_t']
obj_unordered_vars = ['sex', 'marri_1', 'BS13', 'town_t']

# 主观特征相关配置
subj_subjective_features = ['BD1_11', 'HE_BMI', 'BD2_1', 'BP1', 'BO1', 'BO2_1', 'BO1_1', 'D_1_1', 'pa_aerobic', 'BE8_1']
subj_unordered_vars = ['BO2_1', 'BO1_1', 'pa_aerobic']

# 加载客观特征聚类模型
obj_model_filename = f'Occp{X1}_obj.joblib'
obj_kmeans = joblib.load(obj_model_filename)

# 读取客观特征训练数据
data_folder = '数据'
obj_train_files = [f for f in os.listdir(data_folder) if f.startswith(f'Occp{X1}_') and f.endswith('.xlsx')]
obj_train_dfs = [pd.read_excel(os.path.join(data_folder, f)) for f in obj_train_files]
obj_train_df = pd.concat(obj_train_dfs, ignore_index=True)

# 预处理客观特征训练数据
obj_train_factor_df = obj_train_df[obj_factor_cols].copy()
obj_label_encoders = {}
for col in obj_unordered_vars:
    if col in obj_train_factor_df.columns:
        le = LabelEncoder()
        obj_train_factor_df[col] = le.fit_transform(obj_train_factor_df[col].astype(str))
        obj_label_encoders[col] = le

# 用训练数据的标准化参数对输入数据进行标准化
scaler = StandardScaler()
scaler.fit(obj_train_factor_df)

# 预处理输入的客观特征数据
obj_factor_df = df_input[obj_factor_cols].copy()
for col in obj_unordered_vars:
    if col in obj_factor_df.columns:
        le = obj_label_encoders[col]
        obj_factor_df[col] = le.transform(obj_factor_df[col].astype(str))
obj_X = scaler.transform(obj_factor_df)

# 预测客观特征聚类标签
X2 = obj_kmeans.predict(obj_X)[0]

# 保存客观特征因子值
obj_factors = pd.DataFrame(obj_X, columns=[f'Factor{i + 1}' for i in range(obj_X.shape[1])])

# 加载主观特征聚类模型
subj_model_filename = f'Occp{X1}_subj.joblib'
subj_kmeans = joblib.load(subj_model_filename)

# 读取主观特征训练数据
subj_train_file = f'Occp{X1}_{X2}.xlsx'
subj_train_file_path = os.path.join(data_folder, subj_train_file)
if os.path.exists(subj_train_file_path):
    subj_train_df = pd.read_excel(subj_train_file_path)
else:
    raise FileNotFoundError(f"未找到主观特征训练数据文件: {subj_train_file_path}")

# 预处理主观特征训练数据
subj_train_df = subj_train_df[subj_subjective_features].copy()
subj_label_encoders = {}
for col in subj_unordered_vars:
    if col in subj_train_df.columns:
        le = LabelEncoder()
        subj_train_df[col] = le.fit_transform(subj_train_df[col].astype(str))
        subj_label_encoders[col] = le
subj_train_df = subj_train_df.fillna(subj_train_df.median())

# 用训练数据的标准化参数对输入数据进行标准化
scaler = StandardScaler()
scaler.fit(subj_train_df)

# 预处理输入的主观特征数据
subj_df = df_input[subj_subjective_features].copy()
for col in subj_unordered_vars:
    if col in subj_df.columns:
        le = subj_label_encoders[col]
        subj_df[col] = le.transform(subj_df[col].astype(str))
subj_df = subj_df.fillna(subj_df.median())
subj_X = scaler.transform(subj_df)

# 预测主观特征聚类标签
X3 = subj_kmeans.predict(subj_X)[0]

# 保存主观特征因子值
subj_factors = pd.DataFrame(subj_X, columns=[f'SubjF{i + 1}' for i in range(subj_X.shape[1])])

# 生成新标签
new_label = f'Occp{X1}_Obj{X2}_Subj{X3}'
print(f'新标签: {new_label}')

# 输出因子值（可选，根据需求决定是否打印或保存这些因子值）
print("客观特征因子值:")
print(obj_factors)
print("\n主观特征因子值:")
print(subj_factors)

# 查找以 Occp{X1}_{X2}_ 开头的 pkl 模型文件
model_files = [f for f in os.listdir('.') if f.startswith(f'Occp{X1}_{X2}_') and f.endswith('.pkl')]
if model_files:
    model_filename = model_files[0]
    scenario = model_filename.split('_')[2]  # 从模型文件名中提取 scenario
    model = joblib.load(model_filename)
    feature_names = joblib.load(model_filename.replace('.pkl', '_feature_names.pkl'))  # 加载特征名称
else:
    print(f"未找到以 Occp{X1}_{X2}_ 开头的 pkl 模型文件，使用 random_forest_model.pkl 进行预测。")
    model = joblib.load('random_forest_model.pkl')
    scenario = None
    feature_names = None

# 准备特征进行预测
factor_features = [f'Factor{i + 1}' for i in range(obj_X.shape[1])]
factor_analysis_features = [f'SubjF{i + 1}' for i in range(subj_X.shape[1])]
subjective_features = ['BD1_11', 'HE_BMI', 'BD2_1', 'BP1', 'BO1', 'BO2_1', 'BO1_1', 'D_1_1', 'pa_aerobic', 'BE8_1']
objective_features = ['age', 'BS6_3', 'incm5', 'educ', 'BD2', 'BS2_1', 'sex', 'marri_1', 'BS13', 'town_t']
categorical_cols = ['BO2_1', 'BO1_1', 'pa_aerobic', 'sex', 'marri_1', 'BS13', 'town_t']

df = pd.get_dummies(df_input, columns=categorical_cols)

existing_subjective_features = [col for col in subjective_features if col in df.columns]
for col in subjective_features:
    if col not in df.columns:
        new_cols = [c for c in df.columns if c.startswith(col + '_')]
        existing_subjective_features.extend(new_cols)

existing_objective_features = [col for col in objective_features if col in df.columns]
for col in objective_features:
    if col not in df.columns:
        new_cols = [c for c in df.columns if c.startswith(col + '_')]
        existing_objective_features.extend(new_cols)

# 添加因子列到 df
for col in factor_features:
    df[col] = obj_factors[col]
for col in factor_analysis_features:
    df[col] = subj_factors[col]

if scenario:
    if scenario == 'Origin+Origin':
        X = pd.concat([df[existing_subjective_features], df[existing_objective_features]], axis=1)
    elif scenario == 'Origin+Sub':
        X = pd.concat([df[existing_objective_features], df.filter(regex='subj_cluster_'), df[factor_analysis_features]],
                      axis=1)
    elif scenario == 'Origin+Origin+Sub':
        X = pd.concat([df[existing_subjective_features], df[existing_objective_features], df.filter(regex='subj_cluster_'),
                       df[factor_analysis_features]], axis=1)
    elif scenario == 'Origin+Origin+Obj':
        X = pd.concat([df[existing_subjective_features], df[existing_objective_features], df.filter(regex='obj_cluster_'),
                       df[factor_features]], axis=1)
    elif scenario == 'Origin+Origin+Sub+Obj':
        X = pd.concat([df[existing_subjective_features], df[existing_objective_features], df.filter(regex='subj_cluster_'),
                       df[factor_analysis_features], df.filter(regex='obj_cluster_'), df[factor_features]], axis=1)
    X = X.fillna(X.median())
else:
    # 如果未找到匹配的 scenario，使用默认特征组合
    X = pd.concat([df[existing_subjective_features], df[existing_objective_features]], axis=1)
    X = X.fillna(X.median())

# 确保预测时使用的特征名称与训练时一致
if feature_names is not None:
    missing_features = [col for col in feature_names if col not in X.columns]
    extra_features = [col for col in X.columns if col not in feature_names]
    for col in missing_features:
        X[col] = 0
    X = X[feature_names]

# 进行预测
prediction = model.predict(X)
print(f'预测结果: {prediction}')