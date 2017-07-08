#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-8 下午4:51
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# my own module
from utils import data_util


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()

    y_train_all = train['y']
    del train['ID']
    del train['y']
    id_test = test['ID']
    del test['ID']
    print 'train:', train.shape, ', test:', test.shape

    random_state = 420

    model = RandomForestRegressor(n_estimators=1000, criterion='mse',
                                  max_depth=8, min_samples_split=2,
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                  n_jobs=-1, random_state=random_state)
    model.fit(train.values, y_train_all)

    print 'predict submit...'
    y_pred = model.predict(test.values)
    df_sub = pd.DataFrame({'ID': id_test, 'y': y_pred})
    df_sub.to_csv('rf_regressor_model_result.csv', index=False) # 0.55199


if __name__ == '__main__':
    main()
