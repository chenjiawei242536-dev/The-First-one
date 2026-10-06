import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("netflix_titles.csv")

def parse_duration(d):
    if pd.isna(d):
        return np.nan
    num = ''.join([c for c in d if c.isdigit()])
    return int(num) if num else np.nan
df["duration_num"] = df["duration"].apply(parse_duration)
df = df.dropna(subset=["duration_num","release_year"])
y = df["duration_num"]

features_raw = df[["type","release_year","rating","country","listed_in"]]
X_raw = pd.get_dummies(features_raw, drop_first=True)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
elastic_cv = ElasticNetCV(
    l1_ratio=[0.1,0.3,0.5,0.7,0.9],
    alphas=np.logspace(-3,2,50),
    cv=kf, max_iter=20000
)
elastic_cv.fit(X_train, y_train)

print("最优alpha =", elastic_cv.alpha_)
print("最优l1_ratio =", elastic_cv.l1_ratio_)

best_elastic = ElasticNet(alpha=elastic_cv.alpha_, l1_ratio=elastic_cv.l1_ratio_, max_iter=20000)
best_elastic.fit(X_train,y_train)

y_pred = best_elastic.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"测试集MSE = {mse:.2f}")
print(f"测试集R2 = {r2:.3f}")

coef_df = pd.DataFrame({
    "feature": X_raw.columns,
    "coef": best_elastic.coef_
})
coef_df = coef_df.sort_values("coef", ascending=False)
print("\n特征系数：")
print(coef_df)

plt.figure(figsize=(10,5))
lambdas = np.logspace(-3,2,40)
coef_path = []
for lam in lambdas:
    mdl = ElasticNet(alpha=lam, l1_ratio=elastic_cv.l1_ratio_, max_iter=20000)
    mdl.fit(X_train,y_train)
    coef_path.append(mdl.coef_)
plt.plot(lambdas, coef_path)
plt.xscale("log")
plt.title("ElasticNet 系数路径 Netflix数据集")
plt.xlabel("alpha")
plt.ylabel("系数")
plt.grid(alpha=0.3)
plt.show()
