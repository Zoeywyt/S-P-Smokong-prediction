import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 数据读取
input_file = r'D:\model\PII\BorderlineSMOTE_K5_smote_all_decoded.xlsx'
df = pd.read_excel(input_file)

# 特征设置
continuous_cols = ['HE_BMI', 'BD2_1', 'BP1', 'BO1', 'D_1_1', 'BE8_1', 'BD1_11', 'age', 'BS6_3', 'incm5', 'educ', 'BD2',
                   'BS2_1', 'SubjPC1_group', 'subj_cluster', 'obj_cluster']
categorical_cols = ['BO2_1', 'BO1_1', 'pa_aerobic', 'sex', 'marri_1', 'BS13', 'town_t']

# 特征配置
subjective_features = ['BD1_11', 'HE_BMI', 'BD2_1', 'BP1', 'BO1', 'BO2_1', 'BO1_1', 'D_1_1', 'pa_aerobic', 'BE8_1']
objective_features = ['age', 'BS6_3', 'incm5', 'educ', 'BD2', 'BS2_1', 'sex', 'marri_1', 'BS13', 'town_t']

# 提取分组前所需特征
all_features = subjective_features + objective_features
X_all = df[all_features].fillna(df[all_features].median())
y = df['sign']

# 处理分类特征
df = pd.get_dummies(df, columns=categorical_cols)

# 动态构建新的特征列表
all_features = df.columns.difference(['sign']).tolist()
subjective_features = [col for col in all_features if col in subjective_features]
objective_features = [col for col in all_features if col in objective_features]


def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0)
    }
    return metrics


# 分组前的评估
# 2. 原始主观客观因素
all_metrics = evaluate_model(X_all, y)

# 使用全部数据进行训练
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_all, y)

# 保存模型
joblib.dump(rf, 'random_forest_model.pkl')

# 分组前结果
pre_group_results = {
    'All Features': all_metrics
}

pre_group_df = pd.DataFrame(pre_group_results).T
pre_group_df.to_excel('分组前模型评估结果.xlsx')

print('模型性能评估与主观特征分析完成，结果已保存。模型已保存为 random_forest_model.pkl')