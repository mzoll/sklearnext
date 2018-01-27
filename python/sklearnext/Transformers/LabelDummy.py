'''
Created on Nov 7, 2017

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
    def __init__(self, n_many=sys.maxsize, dummy_na = False, asbool=True, sparse_output = True):
        self.n_many = n_many
        self.dummy_na = dummy_na
        self.asbool= asbool
        self.sparse_output = sparse_output
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.invar_ = X.columns[0]
        vc = X.iloc[:,0].value_counts(sort=True)
        self.classes_ = vc.index[:min(self.n_many, len(vc)+1)]
        self.feature_names = [ str(self.invar_)+'_'+str(c) for c in self.classes_ ]
        return self
    def transform(self, X):
        check_is_fitted(self, 'classes_')
        assert_dfncol(X, 1)
        #assert(self.invar_==X.columns[0])
        Xt = pd.get_dummies(X, dummy_na=False )
        if self.asbool:
            Xt = Xt.astype(bool)
        
        #cut back columns for levels not ranked high enough
        for c in Xt.columns:
            if c not in self.feature_names:
                Xt.drop(columns=c, inplace=True)
        #take vare about dummy na
        if self.dummy_na:
            Xt[self.invar_+'_NaN'] = Xt.apply(lambda r: r.any(), axis=1)
            if not self.asbool:
                Xt[self.invar_+'_NaN'] = Xt[self.invar_+'_NaN'].astype(int)
        
        #add columns for levels which are not present in the current dummies
        for f in self.feature_names:
            if f not in Xt.columns:
                Xt[f] = False if self.asbool else 1
        
        if self.sparse_output:
            if self.asbool:
                return Xt.to_sparse(fill_value=False)
            else:
                return Xt.to_sparse(fill_value=0)
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
            force_default_level = False,
            sparse_output = True):
        self.min_coverage = min_coverage
        self.max_levels = max_levels
        self.default_name = default_name
        self.force_default_level = force_default_level
        self.sparse_output = sparse_output
        #--------------------------
        self.default_key_ = None
        self.default_levels_ = []
    def fit(self, X, y=None, **fit_params):
        assert(X.shape[1]==1)
        self.feature_names_ = [ X.columns[0] ]
        
        levels, counts = np.unique( X.iloc[:,0], return_counts=True )
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
        Xt = pd.DataFrame(X.iloc[:,0].apply(xthelper))
            
        if self.sparse_output:
            return Xt.to_sparse(fill_value=False)
        return Xt
    
    def get_feature_names(self):
        return self.feature_names_


class TheseLabelsTransformer(TransformerMixin, object):
    """ one hot encode categorical column with dummies for these lables 
    
    Parameters
    ----------
    lables : list of str
        list of the names of the to onehot-encode lables
    sparse_output : bool
        return a sparse DataFrame (true), else dense DataFrame (false) (default: True)
    """
    def __init__(self, lables, sparse_output = True):
        self.classes_ = lables
        self.sparse_output = sparse_output
    def className_to_dummyName_(self, cn):
        return '_'.join([self.invar_, str(cn)])    
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.invar_ = X.columns[0]
        
        self.feature_names_ = [ self.className_to_dummyName_(c) for c in self.classes_ ]
        #-----------------
        s_dummy = pd.Series( { e:False for e in self.feature_names_ } )
        def xthelper(c):
            sx = s_dummy.copy()
            sx[self.className_to_dummyName_(c) ] = True
            return(sx)
        self.transdict = { c:xthelper(c) for c in self.classes_ }
        self.transdict[None] = s_dummy.copy()
        #optimize for single row evals
        self.transdfdict = { k:pd.DataFrame(v).T for k,v in self.transdict.items() }
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'classes_')
        assert_dfncol(X, 1)
    
        if X.shape[0]==1: #optimize for single row evals
            df = self.transdfdict.get(X[self.invar_].values[0])            
            if df is None:
                return self.transdfdict[None]
            df.index = X.index
            return df
        
        def xthelper(row):
            v = row[self.invar_]
            r = self.transdict.get(v)
            if r is None:
                return self.transdict[None]
            return r
        Xt = X.apply(xthelper, axis=1)
        
        if self.sparse_output:
            return Xt.to_sparse(fill_value=False)
        return Xt
   
    def get_feature_names(self):
        return self.feature_names_
