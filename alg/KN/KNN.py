#coding:utf-8
#导入NearestNeighbor包 和 numpy
from sklearn.neighbors import NearestNeighbors
import numpy as np

#定义一个数组
X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
"""
NearestNeighbors用到的参数解释
n_neighbors=5,默认值为5，表示查询k个最近邻的数目
algorithm='auto',指定用于计算最近邻的算法，auto表示试图采用最适合的算法计算最近邻
fit(X)表示用X来训练算法
"""
nbrs = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(X)
#返回距离每个点k个最近的点和距离指数，indices可以理解为表示点的下标，distances为距离
distances, indices = nbrs.kneighbors(X)
print (indices)
print (distances)

#KNN分类
from sklearn.datasets import load_iris
from sklearn import neighbors
import sklearn

#查看iris数据集
iris = load_iris()
print (iris)

knn = neighbors.KNeighborsClassifier()
#训练数据集
knn.fit(iris.data, iris.target)
#预测
predict = knn.predict([[0.1,0.2,0.3,0.4]])
print (predict)
print (iris.target_names[predict]</span>)
