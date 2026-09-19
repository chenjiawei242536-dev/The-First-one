import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ========== 1. 加载数据（稳定可用的在线数据源） ==========
url = "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv"
df = pd.read_csv(url)

print(df.head())
print(df.dtypes)

# ========== 2. 构造特征矩阵 X 与响应变量 y ==========
y = df["Sales"]
X = df[["Price", "Income", "Advertising", "ShelveLoc"]].copy()

# ShelveLoc 为定性变量（Bad / Good / Medium），转为类别型
X["ShelveLoc"] = X["ShelveLoc"].astype("category")

# 对定性变量做哑变量编码，drop_first=True 自动丢弃字典序第一类作为基准组
X = pd.get_dummies(X, columns=["ShelveLoc"], drop_first=True)

# 添加截距项
X = sm.add_constant(X)

# ========== 3. 拟合多元线性回归 ==========
model = sm.OLS(y, X).fit()
print(model.summary())

# ========== 4. ShelveLoc 基准组判断 ==========
print("\n=== ShelveLoc 基准组 ===")
print("基准组为：ShelveLoc[Bad]（被 get_dummies 的 drop_first=True 丢弃）")
print("进入模型的哑变量为：ShelveLoc[Good]、ShelveLoc[Medium]")

# ========== 5. ShelveLoc[Good] 系数的商业含义 ==========
coef_good = model.params["ShelveLoc_Good"]
print(f"\n=== ShelveLoc[Good] 系数 ===")
print(f"系数值：{coef_good:.4f}")
print(
    "在控制 Price、Income、Advertising 不变的条件下，"
    "货架位置为 Good 的门店，其汽车座椅销售量比货架位置为 Bad 的基准组"
    f"平均高出 {coef_good:.4f}（单位：千辆）。"
)

# ========== 6. 计算各变量的 VIF ==========
X_vif = X.astype(float)

vif_df = pd.DataFrame()
vif_df["variable"] = X_vif.columns
vif_df["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]
print("\n=== 各变量 VIF ===")
print(vif_df)

print("\n多重共线性判断：")
print("一般经验法则：VIF > 10 表示存在严重多重共线性；VIF > 5 需关注。")
