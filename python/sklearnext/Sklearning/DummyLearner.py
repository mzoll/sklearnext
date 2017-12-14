'''
Created on Dec 8, 2017

@author: marcel.zoll
'''

import numpy as np

from sklearn.base import BaseEstimator

class DummyClassifier(BaseEstimator, object):
    def __init__(self):
        self.classes_ = None
    def fit(self, X, y, **fit_params):
        self.classes_= np.unique(y)
        self.feature_importances_ = np.array([1./len(self.classes_)]*3)
        return self
    def predict(self, X):
        base_l = np.linspace(0, len(self.classes_)-1, len(self.classes_))
        probas = self.predict_proba(X)
        return np.array( [ base_l[[bool(i) for i in p]] for p in probas ] ).flatten()
    def predict_proba(self, X):
        base_l = [0.]*(len(self.classes_)-1) + [1.]
        return np.array( [ np.random.permutation(base_l).tolist() for _ in range(len(X))] )
    

class DummyRegressor(BaseEstimator, object):
    def __init__(self):
        pass
    def fit(self, X, y, **fit_params):
        self.min_= np.min(y)
        self.max_= np.max(y)
        self.feature_importances_ = np.array([1./len(X[1,:])]*3)
        return self
    def predict(self, X):
        return np.linspace(self.min_, self.max_, len(X))