#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-2 下午12:09
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from utils import data_util


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()

    # 删除 train 中只存在一种值的特征
    removed_features = []
    for c in train.columns:
        if len(set(train[c])) == 1:
            removed_features.append(c)

    train.drop(removed_features, axis=1, inplace=True)
    test.drop(removed_features, axis=1, inplace=True)

    # 去除 train 中的 outlier 数据
    train = train[train.y < 200]
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== perform category features =========='
    main()