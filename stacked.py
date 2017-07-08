# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

seed=0

class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


train = pd.read_csv('ML/input/train.csv')
test = pd.read_csv('ML/input/test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))



n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

#save columns list before adding the decomposition components

#usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

usable_columns = list(set(train.columns) - set(['y']))

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values

# train.to_csv('ML/input/newFeat_train.csv')
# test.to_csv('ML/input/newFeat_test.csv')
'''Train the xgb model then predict the test data'''

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1,
    'seed': seed
}
# NOTE: Make sure that the class is labeled 'class' in the data file

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

num_boost_rounds = 1250
# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)

'''Train the stacked models then predict the test data'''

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(random_state=seed, learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()

)


stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

# pd.DataFrame(stacked_pipeline.predict(finaltrainset), columns=['stackPip']).to_csv('ML/input/stacked_pipeline_train.csv', index=False)
# pd.DataFrame(results, columns=['stackPip']).to_csv('ML/input/stacked_pipeline_test.csv', index=False)


#####################################
# oof_train = np.zeros((len(finaltrainset),))
# oof_test = np.zeros((len(finaltestset),))
# oof_test_skf = np.empty((5, len(finaltestset)))
# kfold = KFold(n_splits=5, shuffle=True, random_state=0)
# clf = stacked_pipeline

# for i, (train_index, test_index) in enumerate(kfold.split(finaltrainset)):
#     #print 'fold-{}: train: {}, test: {}'.format(i, train_index, test_index)
#     x_tr = finaltrainset[train_index]
#     y_tr = y_train[train_index]
#     x_te = finaltrainset[test_index]

#     clf.fit(x_tr, y_tr)

#     oof_train[test_index] = clf.predict(x_te)
#     oof_test_skf[i, :] = clf.predict(finaltestset)

# oof_test[:] = oof_test_skf.mean(axis=0)

# pd.DataFrame(oof_train.reshape(-1, 1), columns=['stackPip']).to_csv('ML/input/stacked_pipeline_train.csv', index=False)
# pd.DataFrame(oof_test.reshape(-1, 1), columns=['stackPip']).to_csv('ML/input/stacked_pipeline_test.csv', index=False)
#######################################

'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
#print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

n_folds = 10
train_scores = []
val_scores = []
num_boost_roundses = []
metric = r2_score
df_columns = train.columns.values
for i in range(0, n_folds):
    random_state = 42 + i
    X_fit, X_val, y_fit, y_val = train_test_split(train, y_train, test_size=0.25, random_state=random_state)

    y_mean_fit = np.mean(y_fit)
    dtfit = xgb.DMatrix(X_fit.drop('y', axis=1), y_fit)
    dtval = xgb.DMatrix(X_val.drop('y', axis=1), y_fit)
    train_score = metric(y_fit, stacked_pipeline.predict(X_fit[usable_columns].values)*0.2855 + model.predict(dtfit)*0.7145)
    val_score = metric(y_val, stacked_pipeline.predict(X_val[usable_columns].values)*0.2855 + model.predict(dtval)*0.7145)
    print 'perform {} cross-validate: train r2 score = {}, validate r2 score = {}'.format(i + 1, train_score, val_score)
    train_scores.append(train_score)
    val_scores.append(val_score)

print '\naverage train r2 score = {}, average validate r2 score = {}'.format(
    sum(train_scores) / len(train_scores),
    sum(val_scores) / len(val_scores))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('ML/result/stacked-models_moreFeats.csv', index=False)


# Any results you write to the current directory are saved as output.
