# SVM 垃圾邮件分类实验（Spambase）

数据分析课程作业：使用 SVM 对课件案例①的 Spambase 数据集做二分类。

## 文件结构
```
svm-spam-experiment/
├── svm_spam.py        # Python 源代码（可直接运行）
├── report.md          # 实验报告（代码 + 真实运行结果 + 分析）
├── requirements.txt   # 依赖
├── data/
│   └── spambase.data  # 数据集（本地保留，git 忽略，见下方下载地址）
└── README.md
```

## 运行
```bash
pip install -r requirements.txt
# 下载 UCI Spambase 到 data/spambase.data
# https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
python svm_spam.py
```

## 结果速览
| 模型 | 准确率 | F1 |
| --- | --- | --- |
| 线性核 SVM (C=1) | 0.9283 | 0.9074 |
| RBF SVM (C=10, γ=0.01) | 0.9269 | 0.9048 |

详见 `report.md`。
