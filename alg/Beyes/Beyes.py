#��˹���ر�Ҷ˹
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(X, Y)
#����ʽ�ֲ�clf = MultinomialNB().fit(X, y)
#��Ŭ���ֲ�clf = BernoulliNB()  clf.fit(X, Y)  BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print (clf.predict([[-0.8,-1]]))

'''
partial_fit˵����������ѵ��һ������
���ַ�������Ϊ���������ڲ�ͬ�����ݼ����Ӷ�ʵ�ֺ��ĺ�����ѧϰ�������ر����õģ������ݼ��ܴ��ʱ�򣬲��ʺ����ڴ�������
�÷�������һ�������ܺ���ֵ�ȶ��ԵĿ������������������ھ����ܴ�����ݿ飨ֻҪ�����ڴ��Ԥ�㿪����
'''
clf_pf = GaussianNB().partial_fit(X, Y, np.unique(Y))
print (clf_pf.predict([[-0.8,-1]]))



