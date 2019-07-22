from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
#支持向量回归
clf = svm.SVR()
#引入核函数svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#分类clf = svm.SVC()
clf.fit(X, y)
clf.predict([[1, 1]])
print(clf.predict([[1, 1]]))
