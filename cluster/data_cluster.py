#coding=utf-8
#聚类，对样本聚类，找到感兴趣的样本数据，如感兴趣的客户等
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import random
inputfile="data/happiness_train_new.csv"

class cluster():
    def __init__(self,inputfile) :
        print("__init__")
        data = pd.read_csv(inputfile)
        data = data.as_matrix()
        random.shuffle(data)
        
        self.data=data[:,3:]
        p=0.8
        train = data[:int(len(data)*p),:] #前80%为训练集
        test = data[int(len(data)*p):,:] #后20%为测试集
        self.xtrain=train[:,4:]                    #其他数据作为训练的输入
        self.ytrain=train[:,3]                     #happiness作为预测的值
        
        self.xtest=test[:,4:]
        self.ytest=test[:,3]
        print(self.ytest)
    
    def k_means(self):
        from sklearn.cluster import KMeans
        k=4
        iteration=300
        model = KMeans(n_clusters=k,n_jobs=4,max_iter=iteration)#分为k类，并发数为4，iteration为聚类循环次数
        model.fit(self.data)
        
        r1 = pd.Series(model.labels_).value_counts()  #统计各个类别的数目
        r2 = pd.DataFrame(model.cluster_centers_)     #找出聚类的中心
        r = pd.concat([r2,r1],axis=1)
        r.columns=list(self.data.columns)+[u'类别数目']    #重命名表头
        print(r)#输出聚类的结果
        
        r=pd.concat([data,pd.Series(model.labels_,index=self.data.index)],axis=1)#详细输出每个样本对应的类别
        r.columns=list(self.data.columns)+[u'类别数目']    #重命名表头
        r.to_csv(outputfile)  #输出文件