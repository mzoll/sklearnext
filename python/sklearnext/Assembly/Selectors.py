'''
Created on Dec 8, 2017

@author: marcel.zoll
'''


import sys, copy
import itertools
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin

#=================
# Transformers
#=================
class ColumnsAll(TransformerMixin, object):
    """ only writes out the specified variables """
    def fit(self, X, y=None, **fit_params):
        self.feature_names = X.columns
        return self
    def transform(self, X):
        return X
    def transform_dict(self, d):
        return d
    def get_feature_names(self):
        return self.feature_names

class ColumnsSelect(TransformerMixin, object):
    """ only writes out the specified variables
    
    Parameters
    ----------
    column_names : list of strings
        Names of the columns that oart to be selected """
    def __init__(self, column_names):
        if isinstance(column_names, list): 
            self.feature_names = column_names
        elif isinstance(column_names, str):
            self.feature_names = [column_names]
        else:
            raise TypeError('varname_list needs to be list or str (depricated)')
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X):
        return X[self.feature_names]
    def transform_dict(self, d):
#     del_keys= []
#     for k in d.keys():
#         if k not in self.feature_names:
#             del_keys.append(k)
#     for k in del_keys
#         d.pop(k)
        dt = { k:v for k,v in d.items() if k in self.feature_names } #a little bit faster
        return dt
    def get_feature_names(self):
        return self.feature_names