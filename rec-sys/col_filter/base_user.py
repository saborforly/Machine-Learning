#coding=utf-8

#idea from https://github.com/Lockvictor/MovieLens-RecSys
class base_user():
    
    def Picaxun(self,u,v,i): #基于用户u和v都拥有的物品数组i，计算用户u和v的相似度
        u_sum=0
        v_sum=0
        for k in i:
            u_sum=u_sum+u[k]
            v_sum=v_sum+v[k]
        u_mean=u_sum/len(i)
        v_mean=v_sum/len(i)
        
        fz=0
        fm_1=0
        fm_2=0
        for k in i:
            fz=fz+(u[k]-u_mean)*(v[k]-v_mean)
            fm_1=fm_1+(u[k]-u_mean)**2
            fm_2=fm_2+(v[k]-v_mean)**2
        s=fz/(fm_1**0.5 * fm_2**0.5)
        
    def similarity(self,x): #计算相似度
        y = np.ones((len(x),len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                i=np.nonzero(np.dot(x[i],x[j]))
                y[i,j]=self.Picaxun(x[i],x[j],i)
        return y
    
    def recomend():     #用户u对物品i的预测分
        a=np.array(x)
        y = np.ones((len(x),len(x)))
        for u in range(len(x)):
            kmean=x[u].sum()/len(x[u])
            for i in range(len(x[0])):
                array=np.nonzero(x[:,i])
                fz=0
                fm=0
                for k in array:
                    fz=fz+y[u,k]*(x[k,i]-x[k].sum()/len(x[k]))
                    fm=fm+abs(y[u,k])
                y[u,i]=kmean+fz/fm
        return y
                
            
            
        
        