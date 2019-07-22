import numpy as np  
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

# 1. 构造数据

sample_number = 200

# 第一个高斯分布参数
mean1 = [0, 4] # 两个维度上的均值
cov1 = [[5, 3], [3, 10]] # 两个维度的协方差矩阵，必须满足对称半正定

# 第二个高斯分布参数
mean2 = [7, 5]
cov2 = [[7, 2], [2, 15]]

# 从两个二元高斯分布中随机采样数据点
class1_x1, class1_x2 = np.random.multivariate_normal(mean1, cov1, sample_number).T # .T表示转置
class2_x1, class2_x2 = np.random.multivariate_normal(mean2, cov2, sample_number).T

# 两个高斯分布对应两个类标号
data = [[class1_x1[i],class1_x2[i],0] for i in range(sample_number)]+[[class2_x1[i],class2_x2[i],1] for i in range(sample_number)]

# 填充到pandas中
data = pd.DataFrame(data,columns=['score1','score2','result'])

score_data = data[['score1','score2']]
result_data = data['result']

# 2. 训练模型

average_precision = 0 # 平均准确度
iters = 10 # 交叉验证次数

for i in xrange(iters):
    # 数据划分，80%用于训练，20%用于预测
    x_train, x_test, y_train, y_test = train_test_split(score_data, result_data, test_size = 0.2)
    # 构造默认逻辑回归模型
    model = LogisticRegression()
    # 训练
    model.fit(x_train, y_train)
    # 预测
    predict_y = model.predict(x_test)
    # 计算测试集上的准确度
    average_precision += np.mean(predict_y == y_test)

average_precision /= iters

# 3. 绘制分类面 - 法1

x1_min, x1_max = score_data['score1'].min() - .5, score_data['score1'].max() + .5

def generate_face(prob):
    y = -np.log(1.0 / prob - 1.0)
    n = 500
    x1 = np.linspace(x1_min, x1_max, n)
    # w1x1+w2x2+b=y
    x2 = (-model.coef_[0][0] / float(model.coef_[0][1])) * x1 + (y - model.intercept_) / float(model.coef_[0][1])
    return x1, x2

pos_data = data[data['result'] == 1]
neg_data = data[data['result'] == 0]
plt.scatter(x = pos_data['score1'], y = pos_data['score2'], color = 'black', marker = 'o')
plt.scatter(x = neg_data['score1'], y = neg_data['score2'], color = 'red', marker = '*')

face_04_x1, face_04_x2 = generate_face(0.4)
face_05_x1, face_05_x2 = generate_face(0.5)
face_06_x1, face_06_x2 = generate_face(0.6)

plt.plot(face_04_x1, face_04_x2)
plt.plot(face_05_x1, face_05_x2)
plt.plot(face_06_x1, face_06_x2)
plt.xlim(score_data['score1'].min(), score_data['score1'].max())
plt.ylim(score_data['score2'].min(), score_data['score2'].max())
plt.xlabel('score1')
plt.ylabel('score2')
plt.legend(['prob_threshold = 0.4', 'prob_threshold = 0.5', 'prob_threshold = 0.6'], loc='center left', bbox_to_anchor=(1, 0.865))
plt.show()
# 5. 评估模型

# 对于测试数据，模型输出1的概率
answer = model.predict_proba(x_test)[:,1]

# 计算不同概率阈值下的P和R
precision, recall, thresholds = precision_recall_curve(y_test, answer)

# prob > 0.5的报告为1
report = answer > 0.5

print(classification_report(y_test, report, target_names = ['neg', 'pos']))
print('average precision: %f'%average_precision)

# 6. 绘制PRC曲线

# step阶跃图，在点(recall[i],precision[i])进行跳变
plt.step(recall, precision, color='b', alpha=0.2, where='post')
# 对PRC下方填充颜色
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve')
plt.show()
