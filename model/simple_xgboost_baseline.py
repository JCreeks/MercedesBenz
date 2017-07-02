#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
basic xgboost baseline with basic original features
@author: MarkLiu
@time  : 17-6-29 上午9:33
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import xgboost as xgb

# my own module
from conf.configure import Configure
from utils import data_util


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()

    y_train_all = train['y']
    del train['ID']
    del train['y']
    id_test = test['ID']
    del test['ID']

    # Convert to numpy values
    X_all = train.values
    # Create a validation set, with last 20% of data
    num_train = int(train.shape[0] * 0.8)
    X_train_all = X_all
    X_train = X_all[:num_train]
    X_val = X_all[num_train:]
    y_train = y_train_all[:num_train]
    y_val = y_train_all[num_train:]
    X_test = test
    print "validate size:", 1.0 * X_val.shape[0] / X_train.shape[0]

    print('X_train_all shape is', X_train_all.shape)
    print('X_train shape is', X_train.shape)
    print('y_train shape is', y_train.shape)
    print('X_val shape is', X_val.shape)
    print('y_val shape is', y_val.shape)
    print('X_test shape is', X_test.shape)

    df_columns = train.columns.values
    dtrain_all = xgb.DMatrix(X_train_all, y_train_all, feature_names=df_columns)
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    dval = xgb.DMatrix(X_val, y_val, feature_names=df_columns)
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)

    y_mean = np.mean(y_train)
    xgb_params = {
        'n_trees': 500,
        'eta': 0.005,
        'max_depth': 4,
        'subsample': 0.95,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'base_score': y_mean,  # base prediction = mean(target)
        'silent': 1
    }

    # xgboost, cross-validation
    cv_result = xgb.cv(xgb_params,
                       dtrain,
                       num_boost_round=500,  # increase to have better results (~700)
                       early_stopping_rounds=50,
                       verbose_eval=50,
                       show_stdv=False
                       )

    num_boost_rounds = len(cv_result)
    print 'num_boost_rounds =', num_boost_rounds
    model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)
    train_r2_score = r2_score(dtrain.get_label(), model.predict(dtrain))
    val_r2_score = r2_score(dval.get_label(), model.predict(dval))
    print 'train r2 score =', train_r2_score, ', validate r2 score =', val_r2_score

    # train model
    model = xgb.train(dict(xgb_params, base_score=np.mean(y_train_all)), dtrain_all, num_boost_round=num_boost_rounds)

    print 'predict submit...'
    y_pred = model.predict(dtest)
    df_sub = pd.DataFrame({'ID': id_test, 'y': y_pred})
    df_sub.to_csv(Configure.submission_path, index=False)


if __name__ == '__main__':
    main()
