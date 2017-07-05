#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-5 下午4:16
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
# remove warnings
import warnings

warnings.filterwarnings('ignore')

import cPickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from conf.configure import Configure
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

    perform_clusters = [4, 5, 6, 7, 8, 9, 10]
    for n_clusters in perform_clusters:
        print '>>>> perform kmeans cluser, n_clusters = {}...'.format(n_clusters)
        feature_train_path = Configure.kmeans_feature_distance_train_path.format(n_clusters)
        feature_test_path = Configure.kmeans_feature_distance_test_path.format(n_clusters)
        results_df = pd.DataFrame({'ID': ids})
        if (not os.path.exists(feature_train_path)) or \
                (not os.path.exists(feature_test_path)):
            cls = KMeans(n_clusters=n_clusters, n_jobs=-1)
            kmeans_labels = cls.fit_predict(conbined_data.values)
            conbined_data['cluster_label'] = kmeans_labels
            results_df['cluster_{}_cluster_label'.format(n_clusters)] = kmeans_labels

            # 计算距离
            cluster_centers = cls.cluster_centers_

            def calc_intra_center_distance(data):
                center = cluster_centers[int(data['cluster_label']), :]
                raw_data = data.drop(['cluster_label'])
                return np.linalg.norm(center - raw_data)

            def calc_extra_center_distance(data):
                ci = range(0, n_clusters)
                ci.remove(int(data['cluster_label']))

                distance = 0.0
                raw_data = data.drop(['cluster_label'])
                for i in ci:
                    center = cluster_centers[i, :]
                    dis = np.linalg.norm(center - raw_data)
                    distance += dis
                return distance

            def calc_extra_center_distance2(data):
                ci = range(0, n_clusters)
                ci.remove(int(data['cluster_label']))

                distance = 0.0
                raw_data = data.drop(['cluster_label'])
                for i in ci:
                    center = cluster_centers[i, :]
                    dis = np.linalg.norm(center - raw_data)
                    distance += dis
                return distance / len(ci)

            results_df['intra_center_distance_cluster_{}'.format(n_clusters)] = \
                                            conbined_data.apply(calc_intra_center_distance, axis=1, raw=True)
            results_df['average_extra_center_distance_cluster_{}'.format(n_clusters)] = \
                                            conbined_data.apply(calc_extra_center_distance, axis=1, raw=True)
            results_df['sum_extra_center_distance_cluster_{}'.format(n_clusters)] = \
                                            conbined_data.apply(calc_extra_center_distance2, axis=1, raw=True)

            del conbined_data['cluster_label']

            center_distance_train = results_df.iloc[:train.shape[0], :]
            center_distance_test = results_df.iloc[train.shape[0]:, :]

            with open(feature_train_path, "wb") as f:
                cPickle.dump(center_distance_train, f, -1)
            with open(feature_test_path, "wb") as f:
                cPickle.dump(center_distance_test, f, -1)
        else:
            with open(feature_train_path, "rb") as f:
                center_distance_train = cPickle.load(f)
            with open(feature_test_path, "rb") as f:
                center_distance_test = cPickle.load(f)

        # merge
        train = pd.merge(train, center_distance_train, how='left', on='ID')
        test = pd.merge(test, center_distance_test, how='left', on='ID')

    train['y'] = train_y.values
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== generate feature distance =========='
    main()
