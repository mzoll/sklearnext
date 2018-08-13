'''
Created on Dec 8, 2017

@author: marcel.zoll
'''

import pandas as pd
import numpy as np

from sklearn.base import MetaEstimatorMixin, TransformerMixin

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
        #print("shapes ", X.shape, y.shape)
        self.incols = X.columns
        self.estimator.fit(X.values, y.values, **fit_params)
        return self
    def predict(self, X):
        return pd.Series( self.estimator.predict(X.values), index = X.index )
    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_
    def score(self, X, y, sample_weight = None):
        return self.estimator.score(X.values, y.values, sample_weight)
    def predict_dict(self, d):
        v_array = np.array([ np.array( [d[key] for key in self.incols] ) ])
        r = self.estimator.predict(v_array)
        return r.reshape( (1) ) 
    
class SklearnerTransWrapper(MetaEstimatorMixin, TransformerMixin, object):
    """ Wraps around a Learner/Estimator with the interface from sklearn.
    Inputs for X are explicitly an pandas.DataFrame with index
    Input for y is explicitly an pandas.Series with index
    Outputs for predict is a single column pandas.DataFrame with column 'name'
    
    Parameters
    ----------
    estimator : Estimator object
        an sklearn Estimator
    name : string
        an name for the predict output column
    
    Attributes
    ----------
    estimator : Estimator object
        an sklearn Estimator
    """
    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name
    def fit(self, X, y, **fit_params):
        #print("shapes ", X.shape, y.shape)
        self.incols = X.columns        
        self.estimator.fit(X.values, y.values, **fit_params)
        return self
    def transform(self, X):
        return pd.DataFrame( self.estimator.predict(X.values), columns=[self.name], index = X.index )
    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_
    def get_feature_names(self):
        return [self.name]
    def score(self, X, y, sample_weight = None):
        return self.estimator.score(X.values, y.values, sample_weight)
    def transform_dict(self, d):
        v_array = np.array([ np.array([d[key] for key in self.incols]) ])
        return { self.name: self.estimator.predict(v_array) }
