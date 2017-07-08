#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-6 上午11:12
"""
import os
import sys

import cPickle

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
# remove warnings
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from conf.configure import Configure
from utils import data_util


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()
    print 'train:', train.shape, ', test:', test.shape

    print 'perform tsne...'
    if not os.path.exists(Configure.tsne_transformed_data_path):
        features = ['X118', 'X127', 'X47', 'X315', 'X311', 'X179', 'X314', 'X261']
        tsne = TSNE(random_state=2000, perplexity=80)
        tsne_transformed = tsne.fit_transform(pd.concat([train[features], test[features]]))
        tsne_transformed = pd.DataFrame(tsne_transformed)
        tsne_transformed.columns = ['tsne_transform_x', 'tsne_transform_y']

        with open(Configure.tsne_transformed_data_path, "wb") as f:
            cPickle.dump(tsne_transformed, f, -1)
    else:
        with open(Configure.tsne_transformed_data_path, "rb") as f:
            tsne_transformed = cPickle.load(f)

    ids = pd.concat([train.drop(['y'], axis=1), test])['ID']

    perform_clusters = [7]
    for n_clusters in perform_clusters:
        print '>>>> perform kmeans cluser, n_clusters = {}...'.format(n_clusters)
        feature_train_path = Configure.tsne_feature_train_path.format(n_clusters)
        feature_test_path = Configure.tsne_feature_test_path.format(n_clusters)
        if (not os.path.exists(feature_train_path)) or \
                (not os.path.exists(feature_test_path)):

            results_df = pd.DataFrame({'ID': ids})
            conbined_data = tsne_transformed.copy()

            cls = KMeans(n_clusters=n_clusters, n_jobs=-1)
            kmeans_labels = cls.fit_predict(conbined_data.values)
            conbined_data['cluster_label'] = kmeans_labels
            results_df['tsne_cluster_{}_cluster_label'.format(n_clusters)] = kmeans_labels

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

            results_df['tsne_intra_center_distance_cluster_{}'.format(n_clusters)] = \
                conbined_data.apply(calc_intra_center_distance, axis=1, raw=True)
            results_df['tsne_average_extra_center_distance_cluster_{}'.format(n_clusters)] = \
                conbined_data.apply(calc_extra_center_distance, axis=1, raw=True)
            results_df['tsne_sum_extra_center_distance_cluster_{}'.format(n_clusters)] = \
                conbined_data.apply(calc_extra_center_distance2, axis=1, raw=True)

            results_df.drop(['tsne_cluster_{}_cluster_label'.format(n_clusters)],
                            axis=1, inplace=True)

            tsne_df_train = results_df.iloc[:train.shape[0], :]
            tsne_df_test = results_df.iloc[train.shape[0]:, :]
            tsne_df_train['ID'] = train['ID']
            tsne_df_test['ID'] = test['ID']

            with open(feature_train_path, "wb") as f:
                cPickle.dump(tsne_df_train, f, -1)
            with open(feature_test_path, "wb") as f:
                cPickle.dump(tsne_df_test, f, -1)
        else:
            with open(feature_train_path, "rb") as f:
                tsne_df_train = cPickle.load(f)
            with open(feature_test_path, "rb") as f:
                tsne_df_test = cPickle.load(f)

        train = pd.merge(train, tsne_df_train, how='left', on='ID')
        test = pd.merge(test, tsne_df_test, how='left', on='ID')

    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== generate tsne feature =========='
    main()
