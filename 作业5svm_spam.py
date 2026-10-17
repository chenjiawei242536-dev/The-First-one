# -*- coding: utf-8 -*-
"""
SVM 支持向量机 —— Spambase 垃圾邮件二分类实验
对应课件「案例① 垃圾邮件分类」：Spambase · 4601 封邮件 · 57 个特征
对比 线性核 SVM(C=1) 与 RBF 高斯核 SVM(C=10, gamma=0.01)

运行方式：
    python svm_spam.py
依赖：pandas numpy scikit-learn
数据集：data/spambase.data（UCI Spambase，最后一列为标签 1=垃圾邮件 0=正常邮件）
"""
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ========== 1. 读取数据 ==========
DATA_PATH = "data/spambase.data"
# 前 57 列为特征，最后一列为标签（1=spam，0=normal）
df = pd.read_csv(DATA_PATH, header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"样本总数: {len(X)}  特征数: {X.shape[1]}")
print(f"垃圾邮件(1)数量: {(y == 1).sum()}  正常邮件(0)数量: {(y == 0).sum()}")

# ========== 2. 特征标准化（SVM 对特征尺度敏感，必须做） ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 3. 划分训练集 / 测试集（7:3，分层抽样保持类别比例） ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集: {len(X_train)}  测试集: {len(X_test)}\n")


# ========== 4. 评估函数 ==========
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"==== {name} ====")
    print(f"准确率 Accuracy : {acc:.4f}")
    print(f"精确率 Precision: {prec:.4f}")
    print(f"召回率 Recall   : {rec:.4f}")
    print(f"F1-score        : {f1:.4f}")
    print(f"混淆矩阵 [TN FP / FN TP]:\n{cm}\n")
    return acc, prec, rec, f1


# ========== 5. 模型一：线性核 SVM（C=1） ==========
t0 = time.time()
model_linear = LinearSVC(C=1, dual=False, max_iter=20000, random_state=42)
model_linear.fit(X_train, y_train)
t_linear = time.time() - t0
print(f"[线性 SVM 训练耗时 {t_linear:.3f}s]")
evaluate(model_linear, X_test, y_test, "线性核 SVM (C=1)")

# ========== 6. 模型二：RBF 高斯核 SVM（C=10, gamma=0.01） ==========
t0 = time.time()
model_rbf = SVC(kernel="rbf", C=10, gamma=0.01, random_state=42)
model_rbf.fit(X_train, y_train)
t_rbf = time.time() - t0
print(f"[RBF SVM 训练耗时 {t_rbf:.3f}s]")
evaluate(model_rbf, X_test, y_test, "RBF 高斯核 SVM (C=10, gamma=0.01)")

# RBF 模型的支持向量数量（体现 SVM 的稀疏性）
print(f"RBF-SVM 各类支持向量数量: {model_rbf.n_support_}")
print(f"RBF-SVM 支持向量总数: {int(model_rbf.n_support_.sum())} / 训练集 {len(X_train)}")
#（注：内容由AI生成）
