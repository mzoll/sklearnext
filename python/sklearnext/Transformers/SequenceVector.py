'''
Created on Jan 18, 2018

@author: marcel.zoll
'''

import copy
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

from ..base import assert_dfncol

from sklearn.base import TransformerMixin 
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

class SequenceVectorEncoder(TransformerMixin, object):
    """ encode related columns, which all hold the same levels as numeric-ints. take 0 as the empty placeholder, define a default level
    Parameters
    ----------
    categories_list : list of obj
        a list of the levels, that should be kept intact
    default_level : obj
        the default level to infer if an entry is not found to be contained in the categories list
    leftallign : bool
        vectors are alligned on the left side entry (True) or right side entry (False) [Default: True]
    maxentries : int or None
        maximium number of vector entries; if none the global maximum length of vectors is taken
    integerencode : bool 
        reencode the with integers from 0...n_classes (True) [Default: False]
     """
    def __init__(self, categories_list, default_level='UNKNOWN', padding_level = 'MISSING', leftallign=True, maxentries = None, integerencode=False):
        self.categories = categories_list
        self.default_level = default_level
        self.padding_level = padding_level
        self.leftallign = leftallign
        self.maxentries = maxentries
        self.integerencode = integerencode
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        vec_list = X.iloc[:,0].values
        maxlen = max([len(vec) for vec in vec_list])        
        if self.maxentries is None:
            self.maxentries = maxlen
        incol = X.columns[0]
        if self.leftallign:
            self.feature_names = [ "%s_%d"%(incol, i) for i in range(self.maxentries)]
        else:
            self.feature_names = [ "%s_%d"%(incol, i) for i in range(maxlen-self.maxentries, maxlen)]
            
        if self.categories is None:
            raise ValueError("Implement self extraction of categores")
        
        self.classes_ = list(set([self.padding_level]+self.categories+[self.default_level]))
        if self.integerencode:
            self.classes_ = list(range(len( self.classes_ )))
            
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        vec_list = X.iloc[:,0].values
        ln = []
        if self.leftallign:
            for vec in vec_list:
                if len(vec) > self.maxentries:
                    ln.append( vec[:self.maxentries] )
                else:
                    ln.append( vec + [self.padding_level]*(self.maxentries-len(vec)) )
        else:
            for vec in vec_list:
                if len(vec) > self.maxentries:
                    ln.append( vec[len(vec)-self.maxentries:] )
                else:
                    ln.append( [self.padding_level]*(self.maxentries-len(vec)) + vec )
                   
        Xt = pd.DataFrame(data=np.stack(ln), columns=self.feature_names, index = X.index)
        
        cat_type = CategoricalDtype(categories=set([self.padding_level]+self.categories+[self.default_level]), ordered=True)
        def xt_col_helper(col):
            col = col.astype(cat_type).fillna(self.default_level)
            if self.integerencode:
                col = col.cat.rename_categories(list(range(len( col.cat.categories))))
            return col
        Xtt = Xt.apply( xt_col_helper, axis=0)
        return Xtt
        
    def get_feature_names(self):
        return self.feature_names_
    

class SequenceVectorCheckboxes(TransformerMixin, object):
    """ if defaultname is non not enable this as default column """
    def __init__(self, classes, default_name = None):
        self.default_name = default_name
        if self.default_name is None or self.default_name in self.classes:
            self.classes = classes
        else:
            self.classes = classes + [self.default_name]
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        incol = X.columns[0]
        self.feature_names = [ "%s_%s"%(incol, c) for c in self.classes]
        self.tick_dict = { k:e for e,k in enumerate(self.classes) }
        self.dummy_checkbox = [False] * (len(self.classes))
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        def xthelper(vec):
            cb = copy.copy(self.dummy_checkbox)
            for c in set(vec):
                e = self.tick_dict.get(c)
                if e is None:
                    if self.default_name is not None:
                        cb[self.tick_dict.get(self.default_name)] = True
                    return pd.Series(cb)
                cb[e] = True
            return pd.Series(cb)
        Xt = X.iloc[:,0].apply(xthelper)
        Xt.columns = self.feature_names
        return Xt
        