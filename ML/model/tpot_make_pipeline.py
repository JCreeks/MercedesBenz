#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-4 上午10:49
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from utils import data_util
from conf.configure import Configure

print 'load datas...'
train, test = data_util.load_dataset()

y_train_all = train['y']
del train['ID']
del train['y']
id_test = test['ID']
del test['ID']

print 'train:', train.shape, ', test:', test.shape
random_state = 42
X_train, X_val, y_train, y_val = train_test_split(train, y_train_all, test_size=0.2, random_state=random_state)

pipeline_optimizer = TPOTRegressor(generations=5, population_size=100,
                                   offspring_size=None,
                                   scoring='r2', cv=5,
                                   subsample=0.95, n_jobs=1,
                                   random_state=random_state,
                                   verbosity=2)

pipeline_optimizer.fit(X_train.values, y_train.values)
print(pipeline_optimizer.score(X_val.values, y_val.values))
pipeline_optimizer.export('./tpot_exported_models/tpot_exported_pipeline.py')
predict_y = pipeline_optimizer.predict(test.values)
df_sub = pd.DataFrame({'ID': id_test, 'y': predict_y})
df_sub.to_csv('tpot_pipeline_result.csv', index=False)
