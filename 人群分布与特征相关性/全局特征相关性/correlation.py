import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import numpy as np
import statsmodels.api as sm

# 读取数据
def read_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"错误: 读取文件时发生未知错误: {e}")
        return None

# 计算连续变量与sign的皮尔逊相关系数和p值
def calculate_continuous_correlation(df, continuous_cols, target_col):
    results = []
    for col in continuous_cols:
        if col in df.columns:
            try:
                # 过滤掉包含缺失值的行
                valid_data = df[[col, target_col]].dropna()
                corr, pval = pearsonr(valid_data[col], valid_data[target_col])
                results.append({
                    '特征': col,
                    '皮尔逊相关系数': corr,
                    'p值': pval
                })
            except Exception as e:
                results.append({
                    '特征': col,
                    '皮尔逊相关系数': np.nan,
                    'p值': np.nan,
                    '错误信息': str(e)
                })
    return pd.DataFrame(results)

# 计算分类变量与sign的卡方统计量和p值
def calculate_categorical_chi2(df, categorical_cols, target_col):
    results = []
    for col in categorical_cols:
        if col in df.columns:
            try:
                # 创建列联表
                tab = pd.crosstab(df[col], df[target_col])
                chi2, p, dof, ex = chi2_contingency(tab)
                results.append({
                    '特征': col,
                    '卡方统计量': chi2,
                    'p值': p
                })
            except Exception as e:
                results.append({
                    '特征': col,
                    '卡方统计量': np.nan,
                    'p值': np.nan,
                    '错误信息': str(e)
                })
    return pd.DataFrame(results)

# 计算优势比和95%置信区间
def calculate_or_ci(df, feature_col, target_col):
    results = []
    unique_values = df[feature_col].unique()
    for value in unique_values:
        # 创建列联表
        contingency_table = pd.crosstab(df[feature_col] == value, df[target_col])
        a = contingency_table.loc[True, 1]
        b = contingency_table.loc[True, 0]
        c = contingency_table.loc[False, 1]
        d = contingency_table.loc[False, 0]

        # 计算优势比
        if b == 0 or c == 0:
            # 处理零除问题
            or_value = np.nan
        else:
            or_value = (a * d) / (b * c)

        # 计算95%置信区间
        _, _, _, expected = chi2_contingency(contingency_table)
        if np.min(expected) < 5:
            # 当期望频数小于5时，使用精确方法
            from statsmodels.stats.proportion import proportion_confint
            ci = proportion_confint(a, a + b, method='normal')
        else:
            # 否则使用正态近似方法
            from statsmodels.stats.proportion import proportion_confint
            ci = proportion_confint(a, a + b, method='normal')

        results.append({
            '特征': feature_col,
            '特征值': value,
            '优势比': or_value,
            '95%置信区间下限': ci[0],
            '95%置信区间上限': ci[1]
        })
    return pd.DataFrame(results)

# 计算连续变量的逻辑回归系数和置信区间
def calculate_continuous_logistic_regression(df, continuous_cols, target_col):
    results = []
    for col in continuous_cols:
        if col in df.columns:
            try:
                X = df[[col]]
                X = sm.add_constant(X)
                y = df[target_col]

                # 拟合逻辑回归模型
                model = sm.Logit(y, X).fit()

                # 获取系数和置信区间
                coefficient = model.params[col]
                odds_ratio = np.exp(coefficient)
                conf_int = model.conf_int().loc[col]
                conf_int_odds_ratio = np.exp(conf_int)

                results.append({
                    '特征': col,
                    '斜率系数': coefficient,
                    '类似优势比的指标': odds_ratio,
                    '95%置信区间下限': conf_int_odds_ratio[0],
                    '95%置信区间上限': conf_int_odds_ratio[1]
                })
            except Exception as e:
                results.append({
                    '特征': col,
                    '斜率系数': np.nan,
                    '类似优势比的指标': np.nan,
                    '95%置信区间下限': np.nan,
                    '95%置信区间上限': np.nan,
                    '错误信息': str(e)
                })
    return pd.DataFrame(results)

# 主函数
def main():
    file_path = r'D:\model\PII\模型评估\戒烟预测与干预分析\模型效果可视化\分组前数据.xlsx'
    df = read_data(file_path)
    if df is not None:
        target_col = 'sign'  # 假设吸烟状态列名为 'sign'
        continuous_cols = ['HE_BMI', 'BD2_1', 'BP1', 'BO1', 'D_1_1', 'BE8_1', 'BD1_11', 'age', 'BS6_3', 'incm5', 'educ', 'BD2', 'BS2_1']
        categorical_cols = ['BO2_1', 'BO1_1', 'pa_aerobic', 'sex', 'marri_1', 'BS13', 'town_t']

        # 计算连续变量的皮尔逊系数和p值
        continuous_corr = calculate_continuous_correlation(df, continuous_cols, target_col)
        # 计算分类变量的卡方统计和p值
        categorical_chi2 = calculate_categorical_chi2(df, categorical_cols, target_col)
        # 计算分类变量的置信度和OR
        or_ci_results = []
        for col in categorical_cols:
            result = calculate_or_ci(df, col, target_col)
            or_ci_results.append(result)
        or_ci = pd.concat(or_ci_results, ignore_index=True)
        # 计算连续变量的斜率系数及置信区间
        continuous_logistic = calculate_continuous_logistic_regression(df, continuous_cols, target_col)

        # 保存结果到表格
        with pd.ExcelWriter('correlation_results.xlsx') as writer:
            continuous_corr.to_excel(writer, sheet_name='连续变量皮尔逊系数', index=False)
            categorical_chi2.to_excel(writer, sheet_name='分类变量卡方统计', index=False)
            or_ci.to_excel(writer, sheet_name='分类变量OR及置信区间', index=False)
            continuous_logistic.to_excel(writer, sheet_name='连续变量逻辑回归系数', index=False)

        print('分析完成，结果已保存为 correlation_results.xlsx')

if __name__ == "__main__":
    main()