'''
Created on Jan 12, 2018

@author: marcel.zoll
'''

import sys
import pandas as pd
import numpy as np
import datetime as dt

from ..base import assert_dfncol

from sklearn.base import TransformerMixin 
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

class VectorAlignTransformer(TransformerMixin, object):
    """
    Parameters
    ----------
    maxlen : int or None
        limit the output to that many columns
    """
    def __init__(self, maxlen=None):
        self.maxlen = maxlen
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        vec_list = X.iloc[:,0].values
        if self.maxlen is None:
            self.maxlen = max([len(vec) for vec in vec_list])
        incol = X.columns[0]
        self.feature_names = [ "%s_%d"%(incol, i) for i in range(self.maxlen)]
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        vec_list = X.iloc[:,0].values   
        ln = []
        for vec in vec_list:
            if len(vec) > self.maxlen:
                ln.append( vec[:self.maxlen] )
            else:
                ln.append( vec + [None]*(self.maxlen-len(vec)) )
        Xt = pd.DataFrame(data=np.stack(ln), columns=self.feature_names, index = X.index)
        return Xt
        
    def get_feature_names(self):
        return self.feature_names_