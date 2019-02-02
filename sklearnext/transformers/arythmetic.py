'''
Created on Feb 20, 2018

@author: marcel.zoll
'''

import pandas as pd
import numpy as np
import math

from ..base import assert_dfncol

from sklearn.base import TransformerMixin 
from sklearn.utils.validation import check_is_fitted


class Multiplication(TransformerMixin, object):
    """ simply multiply the input with these factors
    
    Parameters
    ----------
    factors : np.array shape(input,)
        factors to be applied on each of the input columns
     """
    def __init__(self, factors):
        self.factors = factors
        if isinstance(self.factors, list):
            self.factors = np.array(self.factors)
    def fit(self, X, y= None, **fit_params):
        self.incols = X.columns
        if len(self.factors)!=X.shape[1]:
            print(len(self.factors), X.shape[1])
            raise Exception("dimensions do not match")
        self.feature_names_ = [ ic+'_norm' for ic in self.incols ]
        return self
    def transform(self, X):
        assert_dfncol(X, len(self.incols))
        def xthelper(vals):
            return vals.values * self.factors
        Xt = X.apply(xthelper, axis=1 )
        Xt.columns = self.feature_names_ 
        return Xt
    def transform_dict(self, d):
        for i in range(len(self.incols)):
            ic = self.incols[i]
            v = d.pop(ic)
            d[self.feature_names_[i]] = v*self.factors[i]
        return d
    def get_feature_names(self):
        return self.feature_names_


class _simpleFuncTransformer(TransformerMixin):
    """ Protoclass for transformation by simple mathematical function  
    
    Parameters
    ----------
    func : callable
        the function that is applied
    suffix : string
        an suffix to be applied
    """
    def __init__(self, func, suffix):
        self.func= func
        self.suffix = suffix
    def fit(self, X, y= None, **fit_params):
        self.incols = X.columns
        self.feature_names_ = [ ic+self.suffix for ic in self.incols ]
        return self
    def transform(self, X):
        assert_dfncol(X, len(self.incols))
        Xt = X.apply(self.func, axis=1, raw=True)
        Xt.columns = self.feature_names_ 
        return Xt
    def transform_dict(self, d):
        for i in range(len(self.incols)):
            ic = self.incols[i]
            v = d.pop(ic)
            d[self.feature_names_[i]] = self.func(v)
        return d
    def get_feature_names(self):
        return self.feature_names_

class LogTransformer(_simpleFuncTransformer, object):
    """ build the (natural) logarithm of these numbers """
    def __init__(self):
        _simpleFuncTransformer.__init__(self, fun=np.log, suffix='_log')
    

class Log10Transformer(TransformerMixin, object):
    """ build the (natural) logarithm of these numbers """
    def __init__(self):
        _simpleFuncTransformer.__init__(self, fun=np.log10, suffix='_log10')
