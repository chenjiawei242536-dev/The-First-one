# SVM 支持向量机 —— Spambase 垃圾邮件分类实验报告

> 课程：数据分析 · 分类算法（SVM 支持向量机）
> 对应课件「案例① 垃圾邮件分类」：Spambase 数据集 · 4601 封邮件 · 57 个特征
> 实验人：陈嘉伟（软工12401）

---

## 1. 实验目的

1. 理解 SVM（最大间隔分类器）的基本思想：硬间隔 / 软间隔、支持向量、核函数；
2. 掌握用 SVM 完成一个真实二分类任务的完整工程流程：数据加载 → 特征标准化 → 训练/测试集划分 → 模型训练 → 指标评估；
3. 对比**线性核 SVM** 与 **RBF 高斯核 SVM** 在垃圾邮件数据集上的表现，并结合课件知识点分析差异；
4. 理解超参数 `C`（软间隔惩罚系数）、`gamma`（RBF 核宽度）对模型的影响。

## 2. 数据集介绍

**数据集：Spambase（UCI 机器学习仓库），课件案例①所用数据。**

- 样本总数：**4601** 封邮件；
- 特征数：**57** 维（词频特征 + 大写字母平均/最长/总长度统计 + 特殊符号频率）；
- 标签：`1 = 垃圾邮件（spam）`，`0 = 正常邮件（non-spam）`；
- 类别分布：垃圾邮件 1813 封，正常邮件 2788 封（轻度不平衡）。

**下载地址（Kaggle / UCI 二选一）：**
- UCI 原版：<https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data>
- Kaggle：<https://www.kaggle.com/datasets/veleon/ham-and-spam-dataset>

> 说明：原始数据为无表头的 `.data` 文件，共 58 列，前 57 列为特征，最后一列为标签。

## 3. 实验环境与流程

- Python 3.12，`scikit-learn 1.4.0`、`pandas 2.2.3`、`numpy 1.26.4`
- 流程：
  1. 读取 `spambase.data`，分离特征 `X` 与标签 `y`；
  2. **特征标准化**（`StandardScaler`，均值 0、方差 1）——SVM 基于样本间距离计算间隔，对特征尺度非常敏感，必须做；
  3. 按 7:3 分层划分训练集（3220）/ 测试集（1381），`stratify=y` 保持类别比例；
  4. 训练两个模型：线性核 `LinearSVC(C=1)`、RBF 核 `SVC(kernel='rbf', C=10, gamma=0.01)`（参数取自课件案例①）；
  5. 在测试集上输出 Accuracy / Precision / Recall / F1 与混淆矩阵。

## 4. Python 完整代码

源代码见仓库文件 [`svm_spam.py`](./svm_spam.py)，核心代码如下：

```python
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# 1. 读取数据：前57列特征，最后一列标签
df = pd.read_csv("data/spambase.data", header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 2. 特征标准化（SVM 必须标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 7:3 分层划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 4. 模型一：线性核 SVM（C=1）
model_linear = LinearSVC(C=1, dual=False, max_iter=20000, random_state=42)
model_linear.fit(X_train, y_train)

# 5. 模型二：RBF 高斯核 SVM（C=10, gamma=0.01）
model_rbf = SVC(kernel="rbf", C=10, gamma=0.01, random_state=42)
model_rbf.fit(X_train, y_train)

# 6. 评估：Accuracy / Precision / Recall / F1 / 混淆矩阵
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"==== {name} ====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1       :", f1_score(y_test, y_pred))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))

evaluate(model_linear, X_test, y_test, "线性核 SVM (C=1)")
evaluate(model_rbf, X_test, y_test, "RBF SVM (C=10, gamma=0.01)")
print("RBF 支持向量总数:", int(model_rbf.n_support_.sum()))
```

## 5. 实际运行输出（本机真实结果）

```
样本总数: 4601  特征数: 57
垃圾邮件(1)数量: 1813  正常邮件(0)数量: 2788
训练集: 3220  测试集: 1381

[线性 SVM 训练耗时 0.070s]
==== 线性核 SVM (C=1) ====
准确率 Accuracy : 0.9283
精确率 Precision: 0.9238
召回率 Recall   : 0.8915
F1-score        : 0.9074
混淆矩阵 [TN FP / FN TP]:
[[797  40]
 [ 59 485]]

[RBF SVM 训练耗时 0.095s]
==== RBF 高斯核 SVM (C=10, gamma=0.01) ====
准确率 Accuracy : 0.9269
精确率 Precision: 0.9284
召回率 Recall   : 0.8824
F1-score        : 0.9048
混淆矩阵 [TN FP / FN TP]:
[[800  37]
 [ 64 480]]

RBF-SVM 各类支持向量数量: [368 349]
RBF-SVM 支持向量总数: 717 / 训练集 3220
```

### 结果汇总表

| 模型 | 准确率 | 精确率 | 召回率 | F1 | 训练耗时 |
| --- | --- | --- | --- | --- | --- |
| 线性核 SVM (C=1) | **0.9283** | 0.9238 | **0.8915** | **0.9074** | 0.070 s |
| RBF SVM (C=10, γ=0.01) | 0.9269 | **0.9284** | 0.8824 | 0.9048 | 0.095 s |

> 与课件参考值（线性 92.3%、RBF 93.5%）处于同一量级，数值差异来自随机划分种子与具体参数，属正常现象。

## 6. 结果分析（结合课件知识点）

### 6.1 为什么必须先标准化
SVM 的间隔由样本间**欧氏距离**决定。本数据中词频（0~1）与大写字母长度统计（可达数百）量纲差异很大，若不标准化，大尺度特征会主导距离计算，使间隔被扭曲。`StandardScaler` 把所有特征缩放到均值 0、方差 1，保证各特征在距离中权重均衡，这是 SVM 稳定收敛的前提。

### 6.2 线性核已足够：数据集近似线性可分
本次实验中，线性 SVM 准确率 92.83%、F1 0.9074，与 RBF 的 92.69% / 0.9048 基本持平，RBF 并未带来明显提升。这与课件对 Spambase 的判断一致——**"线性可分性较好"**：57 维词频特征在最大间隔超平面附近已能较好分开，非线性核的额外复杂度收益有限。线性核还具有训练快（0.070s）、模型简单、可解释性强的优点，是高维文本场景的首选 baseline。

### 6.3 精确率与召回率的权衡
- RBF 的**精确率略高（0.9284 vs 0.9238）**，即被判为垃圾邮件的邮件里真垃圾更多，误杀正常邮件更少（FP=37 < 40）；
- 但**召回率略低（0.8824 vs 0.8915）**，漏掉的垃圾邮件更多（FN=64 > 59）。
- 业务上若更怕"误杀正常邮件"，可优先看精确率；若更怕"放过垃圾邮件"，则应提高召回率。实际中用 F1 综合权衡，二者非常接近。

### 6.4 SVM 的稀疏性（支持向量）
RBF 模型共 3220 个训练样本，只有 **717 个支持向量**（正常类 368、垃圾类 349）参与决定决策边界，其余样本的拉格朗日乘子 αᵢ=0（对应课件 KKT 互补松弛条件）。这说明决策面只由"最靠近边界的危险样本"决定，这正是 SVM 抗过拟合、泛化能力强的理论来源。

### 6.5 超参数 C 与 gamma 的作用（拓展理解）
- **C（软间隔惩罚系数）**：C 越大，对错分样本惩罚越重，间隔越窄，容易过拟合；C 越小，允许更多样本进入间隔/被错分，间隔更宽，追求泛化但可能欠拟合。本实验线性取 C=1、RBF 取 C=10。
- **gamma（RBF 核宽度）**：gamma 越大，单个高斯核影响范围越小，决策边界越曲折，易过拟合；gamma 越小，边界越平滑，易欠拟合。
- 工程上应使用 `GridSearchCV` 对数网格搜索 `C ∈ [1e-3, 1e3]`、`gamma ∈ [1e-4, 1e1]` 并做交叉验证，而非手工定值。

## 7. 实验总结

1. SVM 在 Spambase 垃圾邮件任务上达到约 **93% 准确率**，与课件案例吻合，证明 SVM 在"高维、样本量中等、近似线性可分"的文本场景表现优秀；
2. **标准化是 SVM 的必备步骤**，不可省略；
3. 高维文本数据优先用线性核做 baseline；只有当数据存在明显非线性结构时，RBF 核才体现优势；
4. SVM 决策面由少数支持向量决定，具有稀疏性和较好的泛化保证；
5. 改进方向：用 `GridSearchCV` 对 C、gamma 网格调参；本数据类别轻度不平衡，可引入 `class_weight`；并可与逻辑回归、朴素贝叶斯对比。

## 8. 复现说明

```bash
# 安装依赖
pip install -r requirements.txt
# 下载数据集到 data/spambase.data（见第2节下载地址）
# 运行
python svm_spam.py
```
