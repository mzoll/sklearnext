'''
Created on Nov 7, 2017

@author: marcel.zoll

(Date)Time Transformers
'''

import pandas as pd
import numpy as np
import datetime as dt

from ..base import assert_dfncol

from sklearn.base import TransformerMixin 
from sklearn.utils.validation import check_is_fitted

class HourWeekdayDayMonthYearTransformer(TransformerMixin):
    """ transform a single column of datetime objects into their components.
    
    Components : 
    hour(float), weekday(uint), day(uint), month(uint), year(uint)
    """ 
    def __init__(self):
        pass
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.incols = X.columns
        self.varname = self.incols[0]
        self.feature_names_ = [self.incols[0]+'_'+suffix for suffix in ['hour','weekday','day','month','year']]
        return self
    def transform(self, X):
        def iterhelper(t):
            return pd.Series([t.hour + t.minute/60., int(t.weekday()), int(t.day), int(t.month), int(t.year)])
        Xt = X[self.varname].apply(iterhelper)
        Xt.columns = self.feature_names_
        Xt[self.varname+'_weekday'] = Xt[self.varname+'_weekday'].astype('uint8')
        Xt[self.varname+'_day'] = Xt[self.varname+'_day'].astype('uint8')
        Xt[self.varname+'_month'] = Xt[self.varname+'_month'].astype('uint8')
        Xt[self.varname+'_year'] = Xt[self.varname+'_year'].astype('uint8')
        return Xt
    def transform_dict(self, d):
        t = d.pop(self.varname)
        d[self.varname+'_hour'] = int(t.hour) + t.minute/60.
        d[self.varname+'_weekday'] = int(t.weekday())
        d[self.varname+'_day'] = int(t.day)
        d[self.varname+'_month'] = int(t.month)
        d[self.varname+'_year'] = int(t.year)
        return d
        
    def get_feature_names(self):
        return self.feature_names_


class DeltaSecTransformer(TransformerMixin, object):
    """ calculate the difference in seconds between two input columns, which need be of format datetime
    
    Parameters
    ----------
    fast_path : bool 
        calculate the value by a faster way. This requires both columns to have only
        valid (non-null) input.
    fill_na : tuple of floats (shape=2)
        fill in these default values in, if left side column respective the right side column value is missing.
        If both are missing the left side default value takes presidence
    """
    def __init__(self, fast_path=False, fill_na=(np.nan, np.nan)):
        self.fast_path = fast_path
        self.fill_na = fill_na
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 2)
        self.incolumns = X.columns
        if len(self.incolumns) != 2:
            raise Exception('Expected to calculate the difference of two datetime columns')
        self.feature_names_ = ['_'.join(X.columns)+'_diffsec']
        return self
    def transform(self, X):
        assert_dfncol(X, 2)
        
        t1 = X[self.incolumns[0]]
        t2 = X[self.incolumns[1]]
        if not t1.isnull().values.any() and not t2.isnull().values.any():
            self.fast_path = True

        if self.fast_path:
            dtime = t2 - t1 
            dtime = dtime.apply(lambda v:v.total_seconds())
            return pd.DataFrame(dtime, columns= self.feature_names_)
        else: #execute line by line; check input
            def xthelper(row):
                t1v = row[self.incolumns[0]]
                t2v = row[self.incolumns[1]]
                if t2v is None:
                    return self.fill_na[1]
                elif t1v is None:
                    return self.fill_na[0]
                
                return (t2v-t1v).total_seconds()
            Xt = X.apply(xthelper, axis=1)
            return pd.DataFrame(Xt, columns= self.feature_names_)
    def transform_dict(self, d):
        t1 = d.pop(self.incolumns[0])
        t2 = d.pop(self.incolumns[1])
        d[self.feature_names_[0]] = (t2 - t1).total_seconds()
        return d            
    def get_feature_names(self):
        return self.feature_names_
