'''
Created on Nov 7, 2017

@author: marcel.zoll
'''

import sys
import pandas as pd
import numpy as np

from ..base import assert_dfncol

from sklearn.base import TransformerMixin 
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

#======================================
class SparseBinarizer(TransformerMixin, object):
    """ take an DataFrame input and just set all non-zero/non-null entries to True, everything else zero """
    def __init__(self):
        pass
    def fit(self, X, y= None, **fit_params):
        self.feature_names_ = list(X.columns)
        return self
    def transform(self, X):
        Xt = pd.DataFrame()
        def chelper(v):
            return not (pd.isnull(v) or v==0) 
        for c in X.columns:
            Xt[c] = X[c].apply(chelper)        
        return Xt.to_sparse(fill_value=False)
    def get_feature_names(self):
        return self.feature_names_

#=======================================
class ObjectLengthTransformer(TransformerMixin, object):
    """ take a singel column input and simply state the lenth of the therein contain objects """
    def __init__(self):
        pass
    def fit(self, X, y= None, **fit_params):
        assert_dfncol(X, 1)
        self.feature_names_ = [ X.columns[0] + '_length' ]
        return self
    def transform(self, X):
        Xt = pd.DataFrame(X.iloc[:,0].apply(lambda v: len(v)))
        Xt.columns = self.feature_names_ 
        return Xt
    def get_feature_names(self):
        return self.feature_names_
    
    