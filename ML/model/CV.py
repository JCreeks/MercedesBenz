import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount

from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

R2 = make_scorer(r2_score, greater_is_better=True)

def DeepCV(train, y_train_all, model, metric = r2_score, n_folds = 10): 
    train_scores = []
    val_scores = []
    num_boost_roundses = []
    df_columns = train.columns.values
    for i in range(0, n_folds):
        random_state = 42 + i
        X_train, X_val, y_train, y_val = train_test_split(train, y_train_all, test_size=0.25, random_state=random_state)

        y_mean = np.mean(y_train)
        model.fit(X_train, y_train)
        train_score = metric(y_train, model.predict(X_train))
        val_score = metric(y_val, model.predict(X_val))
        print 'perform {} cross-validate: train r2 score = {}, validate r2 score = {}'.format(i + 1, train_score, val_score)
        train_scores.append(train_score)
        val_scores.append(val_score)

    print '\naverage train r2 score = {}, average validate r2 score = {}'.format(
        sum(train_scores) / len(train_scores),
        sum(val_scores) / len(val_scores))
    
def PrintImportance(X_train, y_train, model):
    model.fit(X_train, y_train)
    feat_imp = pd.Series(model.feature_importances_, X_train.columns).sort_values(ascending=False)[:50]
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.figure(figsize=(50,50))
    
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
    