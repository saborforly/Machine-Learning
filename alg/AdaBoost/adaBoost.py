import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

# ����2ά��̬�ֲ������ɵ����ݰ���λ����Ϊ���࣬500������,2������������Э����ϵ��Ϊ2
X1, y1 = make_gaussian_quantiles(cov=2.0,n_samples=500, n_features=2,n_classes=2, random_state=1)
# ����2ά��̬�ֲ������ɵ����ݰ���λ����Ϊ���࣬400������,2������������ֵ��Ϊ3��Э����ϵ��Ϊ2
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,n_samples=400, n_features=2, n_classes=2, random_state=1)
#���������ݺϳ�һ������
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))
#���ӻ�����
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
#n_estimators��������������algorithm�����㷨SAMME.R SAMME,
#�ع�AdaBoostRegressor(DecisionTreeRegressor(max_depth=2),n_estimators=200,learning_rate=0.8,loss='linear')
#loss : {��linear��, ��square��, ��exponential��}, optional (default=��linear��)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=200, learning_rate=0.8)
bdt.fit(X, y)

print ("Score:", bdt.score(X,y))
bdt.predict(X_test)