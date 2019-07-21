#coding=utf-8
import pandas
import numpy
import datetime,time
from scipy.interpolate import lagrange 
inputfile="data/happiness_train_abbr.csv"
outputfile="data/happiness_train_new.csv"

class data_clear:
    def __init__(self,inputfile):
        self.inputfile=inputfile
        self.data=pandas.read_csv(inputfile)
        self.detail=None
        self.excepte_value()
        self.miss_value()
    def miss_value(self):
    #缺失值处理
        #拉格朗日法插值,适用少量数据集
        #对预测数据进行处理，本例是happiness，为空则删除
        #self.data=self.data['happiness'].dropna(axis=0)
        for j in range(len(self.data)):
            #print(i,self.data[i][j])
            if (self.data['happiness'].isnull())[j]:
                print("空值 ",j)
                self.data=self.data.drop(index=j)
        #对其他数据使用插值法处理，插入均值
        detail=self.data.describe()
        print(detail)
        for i in self.data.columns:
            #1,data[i] is data['id']
            for j in self.data.index:
                if (self.data[i].isnull())[j]:
                    print('i= ',i)
                    self.data[i][j]=int(detail[i]['mean'])
        
        
        self.data.to_csv(outputfile,mode='w')
        
    def excepte_value(self):
    #异常值处理，(mean+-3sigma),小于0，将异常值全部变为空值,剃除处理日期变量
        self.data=self.data.drop(columns=['survey_time'])
        #print(self.data.dtypes)
        self.detail=self.data.describe()
        for i in self.data.columns:
            self.data[i][(self.data[i]<= 0)]=None
            self.data[i][(self.data[i]<self.detail[i]['mean']-3*self.detail[i]['std'])]=self.detail[i]['min']
            self.data[i][(self.data[i]>self.detail[i]['mean']+3*self.detail[i]['std'])]=self.detail[i]['max']

class data_reduction:
    #属性归约，删除合并属性
    def __init__(self,inputfile):
        self.inputfile=inputfile
        self.data=pandas.read_csv(inputfile)
        self.para_red()
    def para_red(self):
        #直接删除属性,缺失值过多
        self.data=self.data.drop(columns=['work_status','work_yr','work_type','work_manage'])
        self.data.to_csv(outputfile)
        
class data_transfer:
    #数据转化
    def __init__(self,inputfile):
        self.inputfile=inputfile
        self.data=pandas.read_csv(inputfile)
        self.value_reg()
    #将收入数值离散化
    def value_reg(self):
        columns=['income','family_income']
        k=10
        
        for i in columns:
            print(self.data[i].describe())
            self.data[i]=pandas.cut(self.data[i], k,labels=range(k))
            print(self.data[i])
        self.data.to_csv(outputfile)
    
    

if __name__ == '__main__':
    #data_reduction(inputfile)
    #r_data=data_clear(outputfile)
    data_transfer(outputfile)
    #a=r_data.data.describe()
    #print(a)
    #r_data=data_transfer(inputfile)
    #print(r_data.data['income'].describe())
    #data_clear(inputfile).miss_data()
    
    
    