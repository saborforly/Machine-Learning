#coding=utf-8

#时序模型，根据时间序列预测
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA


class time_predict():
    
    def __init__(self,file):
        self.discfile = flie
        self.data = pd.read_excel(discfile, index_col = u'日期')
        self.D_data=None
        self.p=None
        self.q=None
    
    def stat_test(self,data): #时序图在均值波动；自相关图有很强的短期相关性；单位根检验pvalue值小于0.05；
        #时序图
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif']=['SimHei'] #正常显示汉字
        plt.rcParams['axes.unicode_minus']=False #正常显示正负号
        data.plot()
        plt.show()
        #自相关图
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(data).show()        
        #单位根统计量
        from statsmodels.tsa.stattools import adfuller as ADF
        from statsmodels.tsa.stattools import adfuller as ADF
        print(u'原始序列的ADF检验结果为：', ADF(data[u'销量']))
        #返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore
    
    def diff_data():  #对数据差分
        self.D_data = data.diff().dropna()
        self.D_data.columns = [u'销量差分']
        
    def Order(self):#定阶
        self.data[u'销量'] = self.data[u'销量'].astype(float)
        pmax = int(len(self.D_data)/10) #一般阶数不超过length/10
        qmax = int(len(self.D_data)/10) #一般阶数不超过length/10
        bic_matrix = [] #bic矩阵
        for p in range(pmax+1):
            tmp = []
            for q in range(qmax+1):
                try: #存在部分报错，所以用try来跳过报错。
                    tmp.append(ARIMA(self.data, (p,1,q)).fit().bic)
                except:
                    tmp.append(None)
            bic_matrix.append(tmp)
        
        bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
        
        self.p,self.q = bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
        print(u'BIC最小的p值和q值为：%s、%s' %(self.p,self.q))
        
        
    def predict(self,n):
        model = ARIMA(self.data, (self.p,1,self.q)).fit() #建立ARIMA(0, 1, 1)模型 中间的1表示1阶差分
        model.summary2() #给出一份模型报告
        model.forecast(n) #作为期5天的预测，返回预测结果、标准误差、置信区间。        
    
if __name__ == "__main__":
    file='../data/arima_data.xls'
    pre=time_predict(file)
    pre.stat_test(pre.data) #判断是否序列平稳性
    #不平稳则,对数据差分
    pre.diff_data(pre.data)
    pre.stat_test(pre.D_data)#判断是否序列平稳性,不平稳则重复
    #定阶
    pre.Order()
    pre.predict(5) #作为期5天的预测
    
    
    
    