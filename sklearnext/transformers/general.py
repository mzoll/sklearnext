'''
Created on Nov 7, 2017

@author: marcel.zoll
'''

import pandas as pd
import numpy as np

from sklearnext.base import assert_dfncol

from sklearn.base import TransformerMixin 
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

#======================================
class SparseBinarizer(TransformerMixin, object):
    """ take an DataFrame input and just set all non-zero/non-null entries to True, everything else False
    
    Examples
    --------
    df = pandas.DataFrame({'A':[0,1], 'B':['a',None]})
    SparseBinarizer().fit_transform(df) 
    >>> pandas.SparseDataFrame({'A':[False,True], 'B':[True,False]})
    """
    def __init__(self):
        pass
    def fit(self, X, y= None, **fit_params):
        self.incols= X.columns.values
        return self
    def transform(self, X):
        Xt = pd.DataFrame()
        def chelper(v):
            return not (pd.isnull(v) or v==0) 
        for c in X.columns:
            Xt[c] = X[c].apply(chelper)        
        return Xt.f.astype(pd.SparseDtype("bool", False))
    def transform_dict(self, d):
        for k,v in d.items():
            if np.isnan(v) or v is None or v == 0:
                d[k] = False
            d[k] = True
    def get_feature_names(self):
        return self.incols

#=======================================
class ObjectLengthTransformer(TransformerMixin, object):
    """ take a single column input and simply state the lenth of the therein contain objects
    
    Examples
    --------
    df = pandas.DataFrame({'A':[[1,2,3],'ab']})
    ObjectLengthTransformer().fit_transform(df) 
    >>> pandas.DataFrame({'A_length':[3,2]]})
    """
    def __init__(self):
        pass
    def fit(self, X, y= None, **fit_params):
        assert_dfncol(X, 1)
        self.incols = X.columns.values
        self.feature_names_ = [ self.incols[0] + '_length' ]
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        Xt = pd.DataFrame(X.iloc[:,0].apply(lambda v: len(v)))
        Xt.columns = self.feature_names_ 
        return Xt
    def transform_dict(self, d):
        v = d.pop(self.incols[0])
        d[self.incols[0]] = len(v)
        return d
    def get_feature_names(self):
        return self.feature_names_
    