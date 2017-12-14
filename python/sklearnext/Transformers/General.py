'''
Created on Nov 7, 2017

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

class LambdaTransformer(TransformerMixin, object):
    """ specify a lambda function, which can contain named arguments
    Parameters
    ----------
    fun : (function object) taking a dataFrame-row (dict) as single argument,
        returns either a single value, or a panda series
    outcols : (list) designated names of the columns constructed from function output;
            can be None, than fun should return a named pandas.Series otherwise columns are numeric indexes (default None)
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
        """
        @param fun : (function object) taking a dataFrame-row (dict) as single argument,
            returns either a single value, or a panda series
        @param outcols : (list) designated names of the columns constructed from function output;
            can be None, than fun should return a named pandas.Series otherwise columns are numeric indexes (default None)
        """
        self.fun = fun
        self.outcols = outcols
    def fit(self, X, y=None, **fit_params):
        Xp = pd.DataFrame(X.head(1).apply(self.fun, axis = 1))
        if self.outcols is not None:
            if Xp.shape[1] != len(self.outcols):
                raise ValueError('outnames to short')
            self.feature_names_ = self.outcols
        else:
            self.feature_names_ = list(Xp.columns)
        return self
    def transform(self, X):
        Xt = pd.DataFrame(X.apply(self.fun, axis = 1))
        Xt.columns = self.feature_names_
        return Xt
    def get_feature_names(self):
        return self.feature_names_


class LabelDummyTransformer(TransformerMixin, object):
    """ one hot encode label column (encountering labels not present in training raises error)
    
    Parameters
    ----------
    sparse_output : bool
        return a sparse DataFrame (true), else dense DataFrame (false) (default: True)
    
    Examples
    --------
    df = pd.DataFrame({'A':[3,3,3,2,2,1]})
    NLabelDummyTransformer(2, True).fit_transform(df)
    >>>      A_2    A_3   A_1
    >>> 0  False   True  False
    >>> 1  False   True  False
    >>> 2  False   True  False
    >>> 3   True  False  False
    >>> 4   True  False  False
    >>> 5  False  False   True
    """
    def __init__(self, dummy_na = False, sparse_output = True):
        self.dummy_na = dummy_na
        self.sparse_output = sparse_output
    def className_to_dummyName_(self, cn):
        return '_'.join([self.incols[0], str(cn)])    
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.incols = X.columns
        
        self.classes_ = np.unique(X)
        self.feature_names_ = [ self.className_to_dummyName_(c) for c in self.classes_ ]
        #-----------------
        s_dummy = pd.Series( { e:False for e in self.feature_names_ } )
        def xthelper(c):
            sx = s_dummy.copy()
            sx[self.className_to_dummyName_(c) ] = True
            return(sx)
        self.transdict = { c:xthelper(c) for c in self.classes_ }
        if self.dummy_na:
            sx = s_dummy.copy()
            sx[self.className_to_dummyName_('NA') ] = True
            self.transdict['NA'] = sx
        else:
            self.transdict[None] = s_dummy #push in a default non key
        #optimize for single row evals
        self.transdfdict = { k:pd.DataFrame(v).T for k,v in self.transdict.items() }
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'classes_')
        assert_dfncol(X, 1)
        invar_ = self.incols[0]
        
        """
        classes = np.unique(X[invar_].values)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            raise ValueError("y contains new labels: %s" % str(diff))
        """
        if X.shape[0]==1: #optimize for single row evals
            df = self.transdfdict.get(X[invar_].values[0])            
            if df is None:
                if self.dummy_na:
                    df = self.transdfdict['NA']
                else:
                    df = self.transdfdict[None]
            df.index = X.index
            return df
        
        def xthelper(row):
            v = row[invar_]
            r = self.transdict.get(v)
            if r is not None:
                return r    
            if self.dummy_na:
                return self.transdict['NA']
            return self.transdict[None]
        Xt = X.apply(xthelper, axis=1)
        
        if self.sparse_output:
            return Xt.to_sparse(fill_value=False)
        return Xt
   
    def get_feature_names(self):
        check_is_fitted(self, 'classes_')
        self.feature_names_ = [ self.className_to_dummyName_(c) for c in self.classes_ ]
        return self.feature_names_
        
      
class NLabelDummyTransformer(TransformerMixin, object):
    """ one hot encode label column with n most frequent lables
    Parameters
    ----------
    n_many : int
        Number of labelclasses to treat
    dummy_na : bool
        if a encountered label is not with the selected n_many most frequent ones, mark it with a dedicated 'NA'-column (default: False)
    sparse_output : bool
        return a sparse DataFrame (true), else dense DataFrame (false) (default: True)
        
    Examples
    --------
    df = pd.DataFrame({'A':[3,3,3,2,2,1]})
    NLabelDummyTransformer(2, True).fit_transform(df)
    >>>      A_2    A_3   A_NA
    >>> 0  False   True  False
    >>> 1  False   True  False
    >>> 2  False   True  False
    >>> 3   True  False  False
    >>> 4   True  False  False
    >>> 5  False  False   True
    """
    def __init__(self, n_many=sys.maxsize, dummy_na = False, sparse_output = True):
        self.n_many = n_many
        self.dummy_na = dummy_na
        self.sparse_output = sparse_output
    def className_to_dummyName_(self, cn):
        return '_'.join([self.incols[0], str(cn)])    
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.incols = X.columns
        
        uc_ranked = sorted(np.vstack(np.unique(X, return_counts=True)).T, key= lambda uc: uc[1], reverse=True)             
        self.classes_ = [ uc[0] for uc in uc_ranked[:min(self.n_many, len(uc_ranked)+1)] ]
        self.feature_names_ = [ self.className_to_dummyName_(c) for c in self.classes_ ]
        #-----------------
        fd = { e:False for e in self.feature_names_ }
        if self.dummy_na:
            fd.update({ self.className_to_dummyName_('NA'): False })
        s_dummy = pd.Series( fd )     
        def fchelper(c):
            sx = s_dummy.copy()
            sx[self.className_to_dummyName_(c) ] = True
            return(sx)
        self.transdict = { c:fchelper(c) for c in self.classes_ }
        
        if self.dummy_na:
            sx = s_dummy.copy()
            sx[self.className_to_dummyName_('NA') ] = True
            self.transdict['NA'] = sx
        else:
            self.transdict[None] = s_dummy #push in a default non key
        #optimize for single row evals
        self.transdfdict = { k:pd.DataFrame(v).T for k,v in self.transdict.items() }
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'classes_')
        assert_dfncol(X, 1)
        invar_ = self.incols[0]
        
        if X.shape[0]==1: #optimize for single row evals
            df = self.transdfdict.get(X[invar_].values[0])            
            if df is None:
                if self.dummy_na:
                    df = self.transdfdict['NA']
                else:
                    df = self.transdfdict[None]
            df.index = X.index
            return df
        
        def xthelper(v):
            r = self.transdict.get(v)
            if r is not None:
                return r    
            if self.dummy_na:
                return self.transdict['NA']
            return self.transdict[None]
                
        Xt = X[invar_].apply(xthelper)
        
        if self.sparse_output:
            return Xt.to_sparse(fill_value=False)
        return Xt
   
    def get_feature_names(self):
        return self.feature_names_
    
    
class CatLabelsTransformer(TransformerMixin, object):
    """ For a single categorical feature, transform every level not in the `max_levels` most frequent 
    to a default level value
    
    Parameters:
    -----------
    min_coverage: float in (0..1)
        minimal coverage required in each level, otherwise substituted by `default_key` (default: 0.0)
    max_levels: int
        that many levels with highest coverage are retained, all others are grouped under 
        the default key (default: `sys.maxsize`)
    default_name: obj
        key that is substituted for any level which is not in the max_levels most frequent (default: 'DEFAULT')
    force_default_level : bool
        force the default level as part of the possible `levels_`; Notice: setting this to true will most likely
        break the generality of fit/train vs predict/test (default: False)
        
    Attributes
    ----------
    levels_ : list
        list of possible emitted levels
    default_level_ : obj
        obj emitted if default is to be emitted; if obj is None then no default is enabled! (default: 'DEFAULT')
    """
    def __init__(self,
            min_coverage = 0.,
            max_levels = sys.maxsize,
            default_name = 'DEFAULT',
            force_default_level = False):
        self.min_coverage = min_coverage
        self.max_levels = max_levels
        self.default_name = default_name
        self.force_default_level = force_default_level
        #--------------------------
        self.default_key_ = None
        self.default_levels_ = []
    def fit(self, X, y=None, **fit_params):
        assert(X.shape[1]==1)
        self.feature_names_ = [ X.columns[0] ]
        
        
        levels, counts = np.unique( Xg.iloc[:,0], return_counts=True )
        idx = np.array(list(reversed(np.argsort(counts))))
        levels, counts, coverage = [ np.take(x, idx ) for x in [levels, counts, counts / np.sum(counts)] ]
        
        #--- decide which levels to take
        self.levels_ = []
        self.default_levels_ = []
        self.coverage_ = []
        for l,c in zip(levels, coverage):
            if len(self.levels_) < self.max_levels and c >= self.min_coverage:
                self.levels_.append(l)
                self.coverage_.append(c)
            else:
                self.default_levels_.append(l)
        
        #--- insert the default key if neccessary
        if len(self.levels_) < len(levels) or self.force_default_level:
            self.default_key_= self.default_name
        else:
            self.default_key_= None
        
        if self.default_key_ is not None:
            self.levels_.append(self.default_name)
            self.coverage_.append( 1. - sum(self.coverage_) )
        return self
    def transform(self, X):
        assert(X.shape[1]==1)
        def xthelper(v):
            if v in self.levels_:
                return v
            if v in self.default_levels_:
                return self.default_key_
            raise Exception("New level encountered not covered in training")
        xt = X.iloc[:,0].apply(xthelper)
        return pd.DataFrame(xt)
    def get_feature_names(self):
        return self.feature_names_
    
#=============================================================
# Timing transformers
#=============================================================
class HourWeekdayDayMonthYearTransformer(TransformerMixin):
    """ transform a single column of datetimeobjects into its components : 
    hour(float), weekday(uint), day(uint), month(uint), year(uint)
    """ 
    def __init__(self):
        import datetime as dt
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.varname = X.columns[0]
        self.feature_names_ = [self.varname+'_'+suffix for suffix in ['hour','weekday','day','month','year']]
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
        import datetime as dt
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
                
    def get_feature_names(self):
        return self.feature_names_
    

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
    
    