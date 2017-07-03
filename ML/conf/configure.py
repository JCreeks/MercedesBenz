#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-26 下午3:14
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import time


class Configure(object):

    original_train_path = '../input/train.csv'
    original_test_path = '../input/test.csv'

    processed_train_path = '../input/train_dataset.pkl'
    processed_test_path = '../input/test_dataset.pkl'

    decomposition_features_train_path = '../input/decomposition_features_train.pkl'
    decomposition_features_test_path = '../input/decomposition_features_test.pkl'

    groupby_features_train_path = '../input/groupby_features_train.pkl'
    groupby_features_test_path = '../input/groupby_features_test.pkl'

    submission_path = '../result/submission_{}.csv'.format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
