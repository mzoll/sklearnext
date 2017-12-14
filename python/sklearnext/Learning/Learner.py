'''
Created on Dec 8, 2017

@author: marcel.zoll
'''

import pandas as pd

from sklearn.base import MetaEstimatorMixin

class SklearnerWrapper(MetaEstimatorMixin, object):
    """ Wraps around a Learner/Estimator with the interface from sklearn.
    Inputs for X are explicitly an pandas.DataFrame with index
    Input for y is explicitly an pandas.Series with index
    Parameters
    ----------
    estimator : Estimator object
        an sklearn Estimator
    
    Attributes
    ----------
    estimator : Estimator object
        an sklearn Estimator
    """
    def __init__(self, estimator):
        self.estimator = estimator
    def fit(self, X, y, **fit_params):
        print("shapes ", X.shape, y.shape)
        self.estimator.fit(X, y, **fit_params)
        return self
    def predict(self, X):
        return pd.Series( self.estimator.predict(X), index = X.index )
    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_
    def score(self, X, y, sample_weight = None):
        return self.estimator.score(X, y, sample_weight)