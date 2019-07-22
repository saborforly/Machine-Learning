# coding: utf-8
import csv
import numpy as np
def create_csv():
    path = "aa.csv"
    with open(path,'w',newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ["input","lable"]
        csv_write.writerow(csv_head)
        data_row =[[3,4,5][4,5,6]],[7,8,9]
        csv_write.writerow(data_row)
#create_csv()
def read_csv():
    with open("aa.csv","r") as csvfile:
        reader = csv.reader(csvfile)
        #这里不需要readlines
        for line in reader:
            read_array=np.array(line[0])
            print(read_array)
            print(len(read_array))
read_csv()

a=[[1,2,3],[2,3,4]]
#np.array(a)
        
def print_iris():        
    from sklearn.datasets import load_iris
    iris = load_iris()
    data=iris.data
    target = iris.target
    #X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.2)
    print(data)
    
#print_iris()