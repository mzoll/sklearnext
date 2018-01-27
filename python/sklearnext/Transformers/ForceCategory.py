'''
Created on Dec 20, 2017

@author: marcel.zoll
'''

import sys
import pandas as pd
import numpy as np

from ..base import assert_dfncol

from sklearn.base import TransformerMixin 
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

from pandas.api.types import CategoricalDtype

class ForceCategoryTransformer(TransformerMixin, object):
    """ force a single column into a categorical with listed levels, and specify a default level
    Parameters
    ----------
    levels : list of objects or None
        specifies the to be used categorical levels; if None levels are infered from data (default: None)
    ordered : bool   
        specify if the passed on level list is stating hierarchicaly ordered levels (default: False)
    default_level : object or None
        specifies the fallback categorical level if entry is found not to belong to any of the levels (default: None)
        
    Examples
    --------
    df = pd.DataFrame({'A': ['a','b','a','c']})
    ForceCategoryTransformer().fit_transform(df)['A'] #cats ['a','b','c'], vals ['a','b','a','c']
    ForceCategoryTransformer(['a','b'], default_level=None).fit_transform(df)['A'] # cats ['a','b'], vals ['a','b','a', NaN]
    ForceCategoryTransformer(['a','b'], default_level='b').fit_transform(df)['A'] # cats ['a','b'], vals ['a','b','a','b']
    """
    def __init__(self, levels=None, ordered=False, default_level=None):
        self.classes_ = levels
        self.default_level = default_level
        self.ordered = ordered
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.feature_names_ = list(X.columns)
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        x = X.iloc[:,0]
        if self.classes_ is None:
            x = x.astype('category')
            self.classes_ = x.cat.categories.values
        else:
            cat_type = CategoricalDtype(categories=self.classes_, ordered=self.ordered)
            x = x.astype(cat_type)
        if self.default_level is not None:
            if self.default_level not in x.cat.categories:
                x = x.cat.add_categories([self.default_level])
            x = x.fillna(self.default_level)
        Xt = pd.DataFrame(x)
        Xt.columns = X.columns
        return Xt
    def get_feature_names(self):
        return self.feature_names_
        