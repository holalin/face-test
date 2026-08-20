import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib  # 用于保存模型

# 1. 加载你跑通的特征
X_train = np.load("features_train.npy")
y_train = np.load("labels_train.npy")

# 2. 训练分类器 (推荐逻辑回归，简单且输出概率)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 3. 保存这个“大脑”
joblib.dump(clf, "emotion_classifier.pkl")
print("分类器已练成！")