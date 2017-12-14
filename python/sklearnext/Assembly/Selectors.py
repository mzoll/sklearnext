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
    def get_feature_names(self):
        return self.feature_names