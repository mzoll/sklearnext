'''
Created on Feb 22, 2018

@author: marcel.zoll
'''

import numpy as np
import pandas as pd
from ..base import assert_dfncol
from sklearn.base import TransformerMixin

class SKLTransformerWrapper(TransformerMixin):
    ''' Wraps around a sklearn::Transformer for complience with the interface of sklearnext::Transformer.
    
    It is assumed that the number of in and out features map like this: 
    Either n to n, or 1 to n, more unlikely scenarios are n to m, which are still gonna be covered, but naming can be skewed
    
    Parameters
    ----------
    skltransformer : sklearn.Transformer obj
        must take a single column input and outputs one or multiple columns
    suffix : str
        suffix will be appended to the in-columns names, if multiple columns are output from a single feature an index (0,1,..) is added
    '''
    def __init__(self, skltransformer, suffix):
        self.skltransformer = skltransformer
        self.suffix = suffix
    def fit(self, X, y=None, **fit_params):
        #assert_dfncol(X, 1)
        self.incols = X.columns.values
        self.skltransformer.fit(X.values)
        Xtnp = self.skltransformer.transform(X.values[0:1]) #transform the first row
        #determine feature names
        if len(self.incols)==Xtnp.shape[1]:
            self.feature_names = [ '_'.join([c, self.suffix]) for c in self.incols ]
        elif len(self.incols)==1:
            self.feature_names = [ '_'.join([self.incols[0], self.suffix, str(i)]) for i in range(len(self.incols)) ]
        else:
            self.feature_names = [ '_'.join([self.suffix, str(i)]) for i in range(len(self.incols)) ]
        return self
    def transform(self, X):
        assert_dfncol(X, len(self.incols))
        Xt = self.skltransformer.transform(X.values)
        Xt = pd.DataFrame( data=Xt, columns= self.feature_names, index=X.index)
        return Xt
    def get_feature_names(self):
        return self.feature_names
    def transform_dict(self, d):
        x = np.reshape(np.array([ d.pop(c) for c in self.incols ]), (1,len(self.incols)))
        xt = self.skltransformer.transform(x)
        d.update( dict(zip(self.feature_names, xt.flatten()))  )
        return d
    

from sklearn.base import MetaEstimatorMixin

class SKLEstimatorWrapper(MetaEstimatorMixin, TransformerMixin, object):
    """ Wraps around a sklearn::Estimator for complience with the interface of sklearnext::Transformer.
    
    By definition the output of an sklearn::Estimator has a one-dimensional output
    
    Inputs for X are explicitly an pandas.DataFrame with index
    Input for y is explicitly an pandas.Series with index
    Outputs for predict is a single column pandas.DataFrame with column 'name'
    
    Parameters
    ----------
    estimator : Estimator object
        an sklearn Estimator
    featuren_name : string
        name for the estimated result in result dictionary and dataFrame output
    """
    def __init__(self, estimator, feature_name):
        self.estimator = estimator
        self.feature_name = feature_name
    def fit(self, X, y, **fit_params):
        #print("shapes ", X.shape, y.shape)
        self.incols = X.columns.values
        self.estimator.fit(X.values, y.values, **fit_params)
        return self
    def transform(self, X):
        return pd.DataFrame( self.estimator.predict(X.values), columns=[self.feature_name], index = X.index )
    def transform_dict(self, d):
        v_array = np.array([ np.array([d[key] for key in self.incols]) ])
        predict_val = self.estimator.predict(v_array)[0]
        return {self.feature_name: predict_val}
    #def score(self, X, y, sample_weight = None):
    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_
    def get_feature_names(self):
        return [self.name]
    #    return self.estimator.score(X.values, y.values, sample_weight)