#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-8 下午3:24
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
# remove warnings
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from sklearn import preprocessing
from utils import data_util


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()

    train_y = train['y']
    train.drop(['y'], axis=1, inplace=True)
    # 合并训练集和测试集
    conbined_data = pd.concat([train, test])
    ids = conbined_data['ID']

    remove_features = ['X33', 'X39', 'X42', 'X95', 'X105', 'X124', 'X190', 'X204', 'X207', 'X210', 'X236', 'X252',
                       'X257', 'X259', 'X260', 'X270', 'X278', 'X280', 'X288', 'X295', 'X296', 'X339', 'X372',
                       'label_encoder_X4_median_y', 'label_encoder_X4_mean_y']
    conbined_data.drop(remove_features, axis=1, inplace=True)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['y'] = train_y.values
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== perform other features =========='
    main()

