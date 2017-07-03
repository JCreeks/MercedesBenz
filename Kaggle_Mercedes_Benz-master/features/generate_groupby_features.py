#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-2 下午8:55
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings

warnings.filterwarnings('ignore')

from utils import data_util


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()

    groupby_features = []
    for c in train.columns:
        if 'label_encoder' in c:
            groupby_features.append(c)

    for c in groupby_features:
        groupby_df = train[[c, 'y']].groupby(c).aggregate('mean')['y'].reset_index()
        def label_encoder_mean_map(data):
            values = groupby_df.loc[groupby_df[c] == data, 'y'].values
            if len(values) == 0:
                return sum(groupby_df.y) / groupby_df.shape[0]
            return values[0]
        train[c + '_mean_y'] = train[c].map(label_encoder_mean_map)
        test[c + '_mean_y'] = test[c].map(label_encoder_mean_map)

        groupby_df = train[[c, 'y']].groupby(c).aggregate('median')['y'].reset_index()
        def label_encoder_median_map(data):
            values = groupby_df.loc[groupby_df[c] == data, 'y'].values
            if len(values) == 0:
                return sum(groupby_df.y) / groupby_df.shape[0]
            return values[0]
        train[c + '_median_y'] = train[c].map(label_encoder_median_map)
        test[c + '_median_y'] = test[c].map(label_encoder_median_map)

        groupby_df = train[[c, 'y']].groupby(c).aggregate('std')['y'].reset_index()
        groupby_df.fillna(0, inplace=True)
        def label_encoder_std_map(data):
            values = groupby_df.loc[groupby_df[c] == data, 'y'].values
            if len(values) == 0:
                return sum(groupby_df.y) / groupby_df.shape[0]
            return values[0]
        train[c + '_std_y'] = train[c].map(label_encoder_std_map)
        test[c + '_std_y'] = test[c].map(label_encoder_std_map)

    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== perform groupby features =========='
    main()
