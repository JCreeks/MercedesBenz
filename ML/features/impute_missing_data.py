#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-3 上午11:19
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
# remove warnings
import warnings

warnings.filterwarnings('ignore')

from utils import data_util
from kmeans_impute_missing_data import KMeansImputeMissingData


def kmeans_impute_datas(conbined_data, num_columns, missing_rates):
    df_numeric = conbined_data[num_columns].copy()
    # 对 str_columns 类别进行编码
    df_obj = conbined_data.select_dtypes(include=['object']).copy()

    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]

    df_data = pd.concat([df_numeric, df_obj], axis=1)

    total_count = conbined_data.shape[0]
    for missing_rate in missing_rates:
        missing_df = conbined_data[num_columns].isnull().sum(axis=0).reset_index()
        missing_df.columns = ['column_name', 'missing_count']
        missing_df = missing_df[missing_df.missing_count > 0]

        if missing_df.shape[0] == 0:  # 不存在缺失数据
            break

        missing_df = missing_df.sort_values(by='missing_count', ascending=False)
        missing_df['missing_rate'] = 1.0 * missing_df['missing_count'] / total_count
        if missing_df['missing_rate'].values[0] < missing_rate:
            continue

        # print '填充缺失率大于{}的数据, 缺失数据属性 {}...'.format(missing_rate, missing_df.shape[0])
        # n_clusters 为超参数！
        impute_model = KMeansImputeMissingData(conbined_data[num_columns], n_clusters=20, max_iter=100)
        kmeans_labels, x_kmeans, centroids, global_labels, x_global_mean = impute_model.impute_missing_data()

        df_data[num_columns] = x_kmeans
        # 填充数据
        big_missing_columns = missing_df[missing_df.missing_rate > missing_rate]['column_name']
        conbined_data[big_missing_columns] = df_data[big_missing_columns]

    return conbined_data


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()

    train_y = train['y']
    train.drop(['y'], axis=1, inplace=True)
    # 合并训练集和测试集
    conbined_data = pd.concat([train, test])
    ids = conbined_data['ID']
    del conbined_data['ID']

    num_columns = conbined_data.select_dtypes(exclude=['object']).columns
    missing_rates = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    conbined_data = kmeans_impute_datas(conbined_data, num_columns, missing_rates)

    conbined_data['ID'] = ids
    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['y'] = train_y.values

    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== impute missing data =========='
    main()
