import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet

url = "https://www.statlearning.com/s/Hitters.csv"
df = pd.read_csv(url).dropna()
X_raw = df.drop(["Salary"], axis=1)
y = df["Salary"]
X_raw = pd.get_dummies(X_raw, drop_first=True)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

lambdas = np.logspace(2, -2, 100)
ridge_coefs, lasso_coefs, elastic_coefs = [], [], []

for lam in lambdas:
    ridge = Ridge(alpha=lam)
    ridge.fit(X, y)
    ridge_coefs.append(ridge.coef_)

    lasso = Lasso(alpha=lam, max_iter=10000)
    lasso.fit(X, y)
    lasso_coefs.append(lasso.coef_)

    elastic = ElasticNet(alpha=lam, l1_ratio=0.5, max_iter=10000)
    elastic.fit(X, y)
    elastic_coefs.append(elastic.coef_)

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.plot(lambdas, ridge_coefs)
plt.xscale("log")
plt.title("Ridge 系数路径")
plt.xlabel("λ")
plt.ylabel("系数")

plt.subplot(1,3,2)
plt.plot(lambdas, lasso_coefs)
plt.xscale("log")
plt.title("Lasso 系数路径")
plt.xlabel("λ")

plt.subplot(1,3,3)
plt.plot(lambdas, elastic_coefs)
plt.xscale("log")
plt.title("ElasticNet(l1_ratio=0.5) 系数路径")
plt.xlabel("λ")
plt.tight_layout()
plt.show()

kf = KFold(n_splits=10, shuffle=True, random_state=42)
def find_best_lambda(model_cls,**kwargs):
    scores=[]
    for lam in lambdas:
        mdl=model_cls(alpha=lam,**kwargs)
        cv=cross_val_score(mdl,X,y,cv=kf,scoring="neg_mean_squared_error")
        scores.append(-np.mean(cv))
    idx=np.argmin(scores)
    return lambdas[idx],scores[idx],scores

best_ridge_lam,ridge_mse,_=find_best_lambda(Ridge)
best_lasso_lam,lasso_mse,_=find_best_lambda(Lasso,max_iter=10000)
best_elastic_lam,elastic_mse,_=find_best_lambda(ElasticNet,l1_ratio=0.5,max_iter=10000)

print(f"Ridge最优λ={best_ridge_lam:.3f}, CV‑MSE={ridge_mse:.2f}")
print(f"Lasso最优λ={best_lasso_lam:.3f}, CV‑MSE={lasso_mse:.2f}")
print(f"ElasticNet最优λ={best_elastic_lam:.3f}, CV‑MSE={elastic_mse:.2f}")

ridge_best=Ridge(alpha=best_ridge_lam).fit(X,y)
lasso_best=Lasso(alpha=best_lasso_lam,max_iter=10000).fit(X,y)
elastic_best=ElasticNet(alpha=best_elastic_lam,l1_ratio=0.5,max_iter=10000).fit(X,y)

res_df=pd.DataFrame({
    "feature":X_raw.columns,
    "Ridge":ridge_best.coef_,
    "Lasso":lasso_best.coef_,
    "ElasticNet":elastic_best.coef_
})
print("\n系数对比：")
print(res_df)
