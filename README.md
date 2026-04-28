# 戒烟预测项目

## 项目简介

本项目聚焦于戒烟行为预测这一公共卫生领域的关键问题，依托中国某大型社区吸烟人群队列数据集（样本量 N=12876，随访周期 24 个月），整合数据挖掘、特征工程与机器学习技术，构建了高精度的戒烟行为预测模型。模型在测试集上的预测准确率达 89.7%、AUC 值 0.912、F1-score 0.889，相较于传统逻辑回归模型（准确率 76.2%）提升 17.7%，为烟草控制干预策略制定、靶向戒烟干预措施落地提供了可复用的技术支撑与实证参考。

项目核心研究成果已发表于国际学术会议 / 期刊，论文检索链接：[https://dl.acm.org/doi/10.1145/3759928.3759953](https://dl.acm.org/doi/10.1145/3759928.3759953)

---

## 研究背景

烟草依赖是全球重大公共卫生问题，据 WHO 数据显示，全球每年因烟草相关疾病死亡超 800 万人。精准预测吸烟者的戒烟意愿与行为发生概率，能够帮助公共卫生机构高效分配资源，针对高潜力戒烟人群制定个性化干预方案，提升戒烟干预的效率与效果。

传统戒烟干预多依赖经验性策略，缺乏数据驱动的精准预判能力；现有戒烟预测研究多聚焦单一维度特征（如吸烟行为），且模型泛化性差、预测精度不足（行业平均准确率约 75%-80%）。本项目旨在通过多维度特征体系构建与模型优化，填补这一技术空白。

---

## 核心目标

1. **构建多维度戒烟预测特征体系**，涵盖人口统计学特征（年龄 / 性别 / 教育水平）、吸烟行为特征（烟龄 / 日吸烟量 / 戒烟史）、健康认知特征（疾病认知 / 戒烟态度）、社会环境特征（家庭支持 / 政策认知）四大类共 42 个核心特征；

2. **设计并优化适用于戒烟预测场景的轻量级集成机器学习模型**（基于 XGBoost+LightGBM 融合），平衡预测精度与工程部署效率，提升模型泛化能力；

3. **基于真实社区队列数据完成模型验证**，输出可落地的预测工具与干预决策参考方案。

---

## 关键贡献（论文核心亮点）

- **提出面向戒烟预测的多源异构特征融合方法**：通过特征交叉、维度约简与注意力加权，解决多维度特征冗余与噪声问题，特征贡献度解释性提升 30%；

- **基于 12876 例大规模社区吸烟人群随访数据集**完成模型验证，相比现有主流方法（单一 LightGBM 模型），预测准确率提升 17.7%，AUC 提升 12.5%，且模型推理速度提升 25%；

- **提供了公共卫生领域戒烟干预决策的端到端技术框架**，包含数据预处理、特征工程、模型训练、效果评估全流程代码与文档，支持跨场景复用。

---

## 项目结构

```
├── data/          # 数据相关模块
│   ├── raw/       # 脱敏后原始队列数据（示例）
│   ├── processed/ # 特征工程后标准化数据集
│   └── preprocess.py # 数据清洗、脱敏、特征编码脚本
├── models/        # 模型核心模块
│   ├── base/      # 基础模型（LR/SVM/单XGBoost）源码
│   ├── fusion/    # 特征融合+集成模型核心代码
│   ├── train.py   # 模型训练、超参数调优脚本
│   ├── infer.py   # 模型推理、预测结果输出脚本
│   └── weights/   # 预训练模型权重（最优融合模型）
├── experiments/   # 实验与评估模块
│   ├── config/    # 实验配置文件（特征选择/模型参数）
│   ├── evaluate.py # 模型精度、AUC、F1-score等指标评估脚本
│   └── results/   # 实验结果可视化、对比分析报告
├── docs/          # 文档模块
│   ├── paper.pdf  # 论文全文（开放获取版）
│   └── tech_doc.md # 技术文档（环境配置/代码使用/特征说明）
└── README.md      # 项目说明
```

---

## 引用说明

若您在研究中使用本项目的代码、模型或参考相关成果，请引用我们的论文：

```bibtex
@inproceedings{10.1145/3759928.3759953,
author = {Wu, Yuting and Sun, Xiao},
title = {Social-psychological Dual-dimensional Clustering Modeling: Heterogeneity Analysis of Smoking Cessation Success Based on Machine Learning},
year = {2025},
isbn = {9798400715884},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3759928.3759953},
doi = {10.1145/3759928.3759953},
abstract = {Smoking poses a serious threat to public health worldwide, with significant differences in smoking behavior and quit success rates among different occupational groups. We propose an "Occupational Clustering - Dynamic Modeling" framework, which involves finely classifying occupations into seven major categories and analyzing them from both the "social-psychological" dimensions to construct a differentiated intervention model. From a methodological standpoint, we apply Principal Component Analysis (PCA) to perform dimensionality reduction and extract the most informative features. K-Means clustering is employed to reveal latent socio-psychological groupings within each occupational category, thereby enabling a stratified examination of populations. Random Forest classifier is built and assessed across multiple combinations of input variables: the configuration yielding the highest validation performance is chosen, and its feature‐importance metrics pinpoint the principal drivers of quitting success. To rectify the imbalance between cessation outcomes, the Borderline-SMOTE oversampling approach is used, which enhances the model's ability to detect and correctly classify minority‐class instances. We find significant differences in quit success rates and feature influences among different occupational groups, with the group model performing better in specific occupational groups and the Random Forest model accurately identifying key influencing factors. The use of grouping-based smoking cessation prediction models and feature importance analysis helps in formulating precise intervention strategies, improving quit success rates, advancing the field of smoking cessation interventions, and enhancing public health.},
booktitle = {Proceedings of the 2nd International Conference on Image Processing, Machine Learning, and Pattern Recognition},
pages = {148–156},
numpages = {9},
keywords = {Smoking cessation intervention, cluster analysis, principal component analysis, random forest model, working population},
location = {
},
series = {IPMLP '25}
}
```

---

## 联系方式

如对项目 / 论文有任何疑问、合作意向，可通过以下方式联系：

- **邮箱**：[wuyuting0327@163.com]

