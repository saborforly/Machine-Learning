#coding=utf-8
import pandas
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange 
inputfile="data/happiness_train_new.csv"
class data_analysis:
    #数据分析
    def __init__(self,inputfile):
        self.inputfile=inputfile
        self.data=pandas.read_csv(inputfile)
        self.eval_othes()
    def eval_othes(self):
        y='happiness'
        #获取相关属性与 预测属性
        colum=list(self.data.columns)
        rm=['Unnamed: 0', 'Unnamed: 0.1', 'id', 'happiness']
        #print(colum)
        for i in rm:
            colum.remove(i)
        print(colum)
        for i in colum:
            plt.scatter(self.data[i],self.data[y])
            plt.xlabel=i
            plt.ylabel=y
            print(self.data[i].describe())
            plt.show()
        
if __name__=='__main__':
    data_analysis(inputfile)