#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-8 下午4:42
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from sklearn.linear_model import LassoLarsCV, LassoLars
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

    model = LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, normalize=True, precompute='auto', cv=5,
                        max_n_alphas=1000, n_jobs=-1, eps=2.2204460492503131e-16, copy_X=True, positive=False)
    model.fit(train.values, y_train_all)

    print 'predict submit...'
    y_pred = model.predict(test.values)
    df_sub = pd.DataFrame({'ID': id_test, 'y': y_pred})
    df_sub.to_csv('lassolars_model_result.csv', index=False)  # 0.55827


if __name__ == '__main__':
    main()
