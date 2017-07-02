#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-29 下午2:21
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
from utils import data_util


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()
    print 'train:', train.shape, ', test:', test.shape
    train_y = train['y']
    train.drop(['y'], axis=1, inplace=True)
    # 合并训练集和测试集
    conbined_data = pd.concat([train, test])
    ids = conbined_data['ID']
    conbined_data.drop(['ID'], axis=1, inplace=True)

    random_state = 100
    n_jobs = -1

    # PCA
    n_comp = 50
    pca = PCA(n_components=n_comp, random_state=random_state)
    pca_df = pca.fit_transform(conbined_data)
    for i in range(0, n_comp):
        conbined_data['pca_' + str(i)] = pca_df[:, i]

    # IncrementalPCA
    n_comp = 70
    ipca_df = IncrementalPCA(n_components=n_comp, batch_size=conbined_data.shape[0]).fit_transform(conbined_data)
    for i in range(0, n_comp):
        conbined_data['ipca_' + str(i)] = ipca_df[:, i]

    # KernelPCA
    n_comp = 60
    ipca_df = KernelPCA(n_components=n_comp, kernel='linear', random_state=random_state, n_jobs=n_jobs).fit_transform(conbined_data)
    for i in range(0, n_comp):
        conbined_data['kernel_pca_' + str(i)] = ipca_df[:, i]

    conbined_data['ID'] = ids
    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    train['y'] = train_y.values

    print("Save data...")
    print 'train:', train.shape, ', test:', test.shape
    data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== generate decomposition features =========='
    main()
