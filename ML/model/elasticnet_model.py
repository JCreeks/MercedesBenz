#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-8 下午4:24
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet

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
    cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True,
                            normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5,
                            copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=random_state,
                            selection='cyclic')
    cv_model.fit(train.values, y_train_all)

    print('Optimal alpha: %.8f' % cv_model.alpha_)
    print('Optimal l1_ratio: %.3f' % cv_model.l1_ratio_)
    print('Number of iterations %d' % cv_model.n_iter_)

    print 'train model with best parameters from CV...'
    model = ElasticNet(l1_ratio=cv_model.l1_ratio_, alpha=cv_model.alpha_, max_iter=cv_model.n_iter_,
                       fit_intercept=True, normalize=True)
    model.fit(train.values, y_train_all)

    print 'predict submit...'
    y_pred = model.predict(test.values)
    df_sub = pd.DataFrame({'ID': id_test, 'y': y_pred})
    df_sub.to_csv('elasticnet_model_result.csv', index=False)   # 0.55828

if __name__ == '__main__':
    main()
