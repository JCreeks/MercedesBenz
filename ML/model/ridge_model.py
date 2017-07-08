#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-8 上午11:40
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

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

    model = Ridge(alpha=1.0, solver='auto', random_state=random_state,
                  fit_intercept=True, normalize=True)
    model.fit(train.values, y_train_all)

    print 'predict submit...'
    y_pred = model.predict(test.values)
    df_sub = pd.DataFrame({'ID': id_test, 'y': y_pred})
    df_sub.to_csv('ridge_model_result.csv', index=False)    # 0.53696


if __name__ == '__main__':
    main()
