#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-3 下午7:07
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
# remove warnings
import warnings

warnings.filterwarnings('ignore')

from utils import data_util


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()
    print 'train:', train.shape, ', test:', test.shape

    train_y = train['y']
    train.drop(['y'], axis=1, inplace=True)
    # 合并训练集和测试集
    conbined_data = pd.concat([train, test])

    dis_features = [c for c in conbined_data.columns if 'pca' in c]
    for c in dis_features:
        mingap = (conbined_data[c].max() - conbined_data[c].min()) / 4000.0
        conbined_data[c] = conbined_data[c].values // mingap

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['y'] = train_y.values
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== perform feature discretize =========='
    main()
