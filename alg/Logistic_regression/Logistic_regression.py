import numpy as np  
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

# 1. ��������

sample_number = 200

# ��һ����˹�ֲ�����
mean1 = [0, 4] # ����ά���ϵľ�ֵ
cov1 = [[5, 3], [3, 10]] # ����ά�ȵ�Э������󣬱�������Գư�����

# �ڶ�����˹�ֲ�����
mean2 = [7, 5]
cov2 = [[7, 2], [2, 15]]

# ��������Ԫ��˹�ֲ�������������ݵ�
class1_x1, class1_x2 = np.random.multivariate_normal(mean1, cov1, sample_number).T # .T��ʾת��
class2_x1, class2_x2 = np.random.multivariate_normal(mean2, cov2, sample_number).T

# ������˹�ֲ���Ӧ��������
data = [[class1_x1[i],class1_x2[i],0] for i in range(sample_number)]+[[class2_x1[i],class2_x2[i],1] for i in range(sample_number)]

# ��䵽pandas��
data = pd.DataFrame(data,columns=['score1','score2','result'])

score_data = data[['score1','score2']]
result_data = data['result']

# 2. ѵ��ģ��

average_precision = 0 # ƽ��׼ȷ��
iters = 10 # ������֤����

for i in xrange(iters):
    # ���ݻ��֣�80%����ѵ����20%����Ԥ��
    x_train, x_test, y_train, y_test = train_test_split(score_data, result_data, test_size = 0.2)
    # ����Ĭ���߼��ع�ģ��
    model = LogisticRegression()
    # ѵ��
    model.fit(x_train, y_train)
    # Ԥ��
    predict_y = model.predict(x_test)
    # ������Լ��ϵ�׼ȷ��
    average_precision += np.mean(predict_y == y_test)

average_precision /= iters

# 3. ���Ʒ����� - ��1

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
# 5. ����ģ��

# ���ڲ������ݣ�ģ�����1�ĸ���
answer = model.predict_proba(x_test)[:,1]

# ���㲻ͬ������ֵ�µ�P��R
precision, recall, thresholds = precision_recall_curve(y_test, answer)

# prob > 0.5�ı���Ϊ1
report = answer > 0.5

print(classification_report(y_test, report, target_names = ['neg', 'pos']))
print('average precision: %f'%average_precision)

# 6. ����PRC����

# step��Ծͼ���ڵ�(recall[i],precision[i])��������
plt.step(recall, precision, color='b', alpha=0.2, where='post')
# ��PRC�·������ɫ
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve')
plt.show()
