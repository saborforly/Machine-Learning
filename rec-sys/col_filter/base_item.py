#coding=utf-8

#idea from https://github.com/Lockvictor/MovieLens-RecSys
class base_item():
    self.sim=None
    def Jacard(a,b):#适用于0,1矩阵， 同时喜欢物品a,b的人数与喜欢物品a，b的总数人数
        return 1.0*(a*b).sum()/(a+b-a*b).sum
    
    def similarity(self,x): #计算相似度
        y = np.ones((len(x),len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                y[i,j]=self.Jacard(x[i],x[j])
        return y
    
    def recommend(self,a): #推荐函数
        return np.dot(self.sim,a)*(1-a)