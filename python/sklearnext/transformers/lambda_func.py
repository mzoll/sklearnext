'''
Created on Dec 21, 2017

@author: marcel.zoll
'''

import pandas as pd

from ..base import assert_dfncol

from sklearn.base import TransformerMixin 
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

class LambdaTransformer(TransformerMixin, object):
    """ specify a lambda function, which can contain named arguments
    
    Parameters
    ----------
    fun : callable 
        function object taking a dataFrame-row (dict) as single argument, returns either a single value, or a panda series
    outcols : list of strings
        designated names of the columns constructed from function output; can be None, than fun should return a named pandas.Series
        otherwise columns are numeric indexes (default None)
                    
    Note
    ----
    This transformer won't be able to pickle; use or write a dedicated transformer for your task
     
    Examples
    --------
    df = pd.DataFrame({'A':[1,2]})
    LambdaTransformer(lambda v: v*v, ['Square']).fit_transform(df)
    >>>    A    Square
    >>> 0  1    1
    >>> 1  2    4
    
    LambdaTransformer(lambda v: pd.Series({'Square': v*v}), None).fit_transform(df)
    >>>    A    Square
    >>> 0  1    1
    >>> 1  2    4
    """
    def __init__(self, fun, outcols=None):
        self.fun = fun
        self.outcols = outcols
    def fit(self, X, y=None, **fit_params):
        self.incols = X.columns
        #probe the lambda function to output the desired outcolum names 
        Xp = pd.DataFrame(X.head(1).apply(self.fun, axis = 1))
        if self.outcols is not None:
            if Xp.shape[1] != len(self.outcols):
                raise ValueError('outnames to short')
            self.feature_names_ = self.outcols
        else:
            self.feature_names_ = list(Xp.columns)
        return self
    def transform(self, X):
        assert_dfncol(X, len(self.incols))
        Xt = pd.DataFrame(X.apply(self.fun, axis = 1))
        Xt.columns = self.feature_names_
        return Xt
    def transform_dict(self, d):
        invals = [d.pop(ic) for ic in self.incols]
        outvals = self.fun(*invals)
        d.update( dict( zip(self.feature_names_, outvals) ) )
        return d    
    def get_feature_names(self):
        return self.feature_names_
        