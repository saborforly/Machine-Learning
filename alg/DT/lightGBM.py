# coding: utf-8
# pylint: disable = invalid-name, C0111
from __future__ import division
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',   # 目标函数
    #'metric': {'binary_logloss'}, 
    'metric': {'l2', 'auc'},     # 评估函数
    'num_leaves': 63,            # 叶子节点数
    'num_trees': 100,
    'learning_rate': 0.01,       # 学习速率
    'feature_fraction': 0.9,     # 建树的特征选择比例
    'bagging_fraction': 0.8,     # 建树的样本采样比例
    'bagging_freq': 5,           # k 意味着每 k 次迭代执行bagging
    'verbose': 0
}

def junonn_input():
    print("input data...")
    dataset = pd.read_csv("./data/train.csv")  # 训练集
    d_x = dataset.iloc[:, 2:].values
    d_y = dataset['type'].values
    dataset_future = pd.read_csv("./data/test.csv")  # 测试集（用于在线提交结果）
    d_future_x = dataset_future.iloc[:, 2:].values
    
    x_train, x_test, y_train, y_test = train_test_split(
        d_x, d_y, test_size=0.2, random_state=2)  # 将训练集分为训练集+验证集
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    print("Training...")
    #model=lgb.LGBMRegressor()
    #model.fit(X_train,y_train)
    bst = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_eval],
        num_boost_round=500,
        early_stopping_rounds=200)
    print("Saving Model...")
    bst.save_model(model_file)  # 保存模型
    print("Predicting...")
    y_pred = bst.predict(x_test,num_iteration=gbm.best_iteration)  # 预测
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    return predict_result

def main(argv=None):  # pylint: disable=unused-argument
    input_train()


if __name__ == '__main__':
    tf.app.run()