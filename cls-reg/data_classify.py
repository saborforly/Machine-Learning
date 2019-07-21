#coding=utf-8

#针对幸福度指数拟使用分类算法进行预测
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import random
inputfile="data/happiness_train_new.csv"
class cla():
    def __init__(self,inputfile) :
        print("__init__")
        data = pd.read_csv(inputfile)
        self.data = data.as_matrix()
        random.shuffle(data)
        p=0.8
        train = data[:int(len(data)*p),:] #前80%为训练集
        test = data[int(len(data)*p):,:] #后20%为测试集
        self.xtrain=train[:,4:]                    #其他数据作为训练的输入
        self.ytrain=train[:,3]                     #happiness作为预测的值
        
        self.xtest=test[:,4:]
        self.ytest=test[:,3]
        print(self.ytest)
        #from sklearn.svm import SVC    #支持向量机 二分类 C表示分类
        #self.Logistic_R()
        #self.decision_tree() #决策树
        #self.knn()           #K紧邻
        #self.byes()          #贝叶斯
        #self.rand_forest()   #随机森林
        #self.adboost()       #adboost算法
        self.grad_tree_boost()
        
    def Logistic_R(self):
        from sklearn.linear_model import LogisticRegression as LR     
        from sklearn.linear_model import RandomizedLogisticRegression as RLR
        rlr=RLR()
        rlr.fit(self.xtrain, self.ytrain)
        rlr.get_support()
        print(u'有效特征为： ',','.join(self.data.columns[rlr.get_support()]))
        x=self.xtrain[data.columns[rlr.get_support()]].as_matrix()  #筛选好的特征
        lr=LR()
        lr.fit(x,self.xtrain)
        pre_ytest=lr.predict(self.xtest)
        print("accuracy_score= ", lr.score(pre_ytest,self.ytest))
        
        
    
    def decision_tree(self): #0.73    #决策树
        dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
        dtc.fit(self.xtrain, self.ytrain)
        from sklearn.metrics import accuracy_score
        pre_ytest=dtc.predict(self.xtest)
        print("accuracy_score= ", accuracy_score(pre_ytest,self.ytest))
        
        '''
        #保存模型
        treefile = 'data/tree.pkl'
        from sklearn.externals import joblib
        joblib.dump(dtc, treefile)
        #tree.predict_proba(dtc[:,:3])[:,1]
        '''
    def knn(self):    #0.72      K近邻
        import sklearn.neighbors as sn
        model=sn.KNeighborsClassifier(n_neighbors=30,weights='distance')
        model.fit(self.xtrain, self.ytrain)
        from sklearn.metrics import accuracy_score
        pre_ytest=model.predict(self.xtest)
        print("accuracy_score= ", accuracy_score(pre_ytest,self.ytest))
        
    def byes(self):   #0.58     贝叶斯
        import sklearn.naive_bayes as nb
        model=nb.GaussianNB()
        model.fit(self.xtrain,self.ytrain)
        from sklearn.metrics import accuracy_score
        pre_ytest=model.predict(self.xtest)
        print("accuracy_score= ", accuracy_score(pre_ytest,self.ytest))
    def rand_forest(self): #0.78   随机森林
        from sklearn.ensemble import RandomForestClassifier
        model=RandomForestClassifier()
        model.fit(self.xtrain,self.ytrain)
        from sklearn.metrics import accuracy_score
        pre_ytest=model.predict(self.xtest)
        print("accuracy_score= ", accuracy_score(pre_ytest,self.ytest))
    def adboost(self):  #0.62
        from sklearn.ensemble import AdaBoostClassifier
        model=AdaBoostClassifier(n_estimators=100,learning_rate=0.1)
        model.fit(self.xtrain,self.ytrain)
        from sklearn.metrics import accuracy_score
        pre_ytest=model.predict(self.xtest)
        print("accuracy_score= ", accuracy_score(pre_ytest,self.ytest))        
    def grad_tree_boost(self):  #0.67  递归树
        from sklearn.ensemble import GradientBoostingClassifier
        model=GradientBoostingClassifier(learning_rate=0.01, n_estimators=500,max_depth=4)
        model.fit(self.xtrain,self.ytrain)
        from sklearn.metrics import accuracy_score
        pre_ytest=model.predict(self.xtest)
        print("accuracy_score= ", accuracy_score(pre_ytest,self.ytest))
        print("各特征的重要程度： ",model.feature_importances_)
        
if __name__=="__main__":
    cla(inputfile)

