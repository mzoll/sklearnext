'''
Created on Nov 7, 2017

@author: marcel.zoll
'''

import sys
import pandas as pd
import numpy as np

from ..base import assert_dfncol

from sklearn.base import TransformerMixin 
from sklearn.utils.validation import check_is_fitted

from pandas.api.types import CategoricalDtype


class OneHotTransformer(TransformerMixin, object):
    """ one hot encode label column analogous to sklearn:preprocessing.OneHotEncoder
    
    Parameters
    ----------
    classes : list or None
        objects which form the set of valid labels; lables not in this set will be mapped to the default_name;
        if this parameter is None the set of labels will be inferred
    default_name : str
        a name for the default level.
        if the default_name is present in classes, by specification or inference, defaulting labels will be written to this same label
        (default: 'DEFAULT') 
    default_dummy : bool
        force a column with superficious dummies, for defaulting labels (default: False)
    sparse_output : bool
        return a sparse DataFrame (True), else dense DataFrame (False) (default: True)
    
    Attributes
    -----------
    categories_ : list
        original categories excluding default_level
    classes_ : list
        classes in the output
    
    Examples
    --------
    df = pd.DataFrame({'A':['a','b','c','d']})
    OneHotTransformer().fit_transform(df)
    >>>      A_a   A_b    A_c
    >>> 0  True    False  False
    >>> 1  False   True   False
    >>> 2  False   False  True
    >>> 3  False   False  False
    
    OneHotTransformer(['a','b'], 'c', False).fit_transform(df)
    >>>      A_a   A_b    
    >>> 0  True    False
    >>> 1  False   True
    >>> 2  False   False
    >>> 3  False   False
    
    OneHotTransformer(['a','b'], 'c', True).fit_transform(df)
    >>>      A_a   A_b    A_c
    >>> 0  True    False  False
    >>> 1  False   True   False
    >>> 2  False   False  True
    >>> 3  False   False  True
    """
    def __init__(self,
            classes = None,
            default_name = 'DEFAULT',
            default_dummy = False,
            sparse_output = True):
        self.categories_ = classes
        self.default_name = default_name
        self.default_dummy = default_dummy
        self.sparse_output = sparse_output
        #---
        self._fit_categories = classes is None
        #--- prep
        if self.categories_ is not None:
            if self.default_name in self.categories_:
                self.default_dummy = True
                self.categories_.remove(self.default_name)
            if self.default_dummy:
                self.classes_ = self.categories_ + [self.default_name]
            else:
                self.classes_ = self.categories_
                
    def _className_to_dummyName(self, cn):
        return '_'.join([self.incols[0], str(cn)])    
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.incols = X.columns.values
        
        if self._fit_categories: 
            self.categories_ = list(X.iloc[:,0].unique())
            if self.default_name in self.categories_:
                self.default_dummy = True
                self.categories_.remove(self.default_name)
            if self.default_dummy:
                self.classes_ = self.categories_ + [self.default_name]
            else:
                self.classes_ = self.categories_
            
        self.feature_names_ = [ self._className_to_dummyName(c) for c in self.classes_ ]
        
        #--- create a translation dictionary holding the punch-card for all classes_
        s_dummy = pd.Series( { e:False for e in self.feature_names_ } )
        def cthelper(c):
            sx = s_dummy.copy()
            sx[self._className_to_dummyName(c) ] = True
            return(sx)
        self.transdict = { c:cthelper(c) for c in self.classes_ }
        self.transdict[None] = s_dummy #push in a default non key
        #optimize for single row evals
        self.transdfdict = { k:pd.DataFrame(v).T for k,v in self.transdict.items() }
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'classes_')
        assert_dfncol(X, 1)
        _invar = self.incols[0]
        
        if X.shape[0]==1: #optimize for single row evals
            df = self.transdfdict.get(X[_invar].values[0])            
            if df is None:
                if self.default_dummy:
                    df = self.transdfdict[self.default_name]
                else:
                    df = self.transdfdict[None]
            df.index = X.index
            return df
        
        def xthelper(row):
            v = row[_invar]
            r = self.transdict.get(v)
            if r is None:
                r = self.transdict.get(self.default_name)
            if r is None:
                r = self.transdict[None]
            return r
        Xt = X.apply(xthelper, axis=1)
        
        #--- if sparse is requested
        if self.sparse_output:
            return Xt.astype(pd.SparseDtype("bool", False))
        return Xt
    def transform_dict(self, d):
        v = d.pop(self.incols[0])
        s = self.transdict.get(v)
        if s is None:
            s = self.transdict.get(self.default_name)
        if s is None:
            s = self.transdict[None]
        d.update( dict(s.items()) )
        return d
    def get_feature_names(self):
        check_is_fitted(self, 'classes_')
        self.feature_names_ = [ self._className_to_dummyName(c) for c in self.classes_ ]
        return self.feature_names_
    

class ForceCategoryTransformer(TransformerMixin, object):
    """ force a single column into a categorical with listed levels, and specify a default level
    
    Parameters
    ----------
    levels : list of objects or None
        specifies the to be used categorical levels; if None levels are infered from data (default: None)
    ordered : bool   
        specify if the passed level list is stating hierarchicaly ordered levels (default: False)
    default_level : object or None
        specifies the fallback categorical level if entry is found not to belong to any of the levels (default: None)
        
    Examples
    --------
    df = pd.DataFrame({'A': ['a','b','a','c']})
    ForceCategoryTransformer().fit_transform(df)['A'] #cats ['a','b','c'], vals ['a','b','a','c']
    ForceCategoryTransformer(['a','b'], default_level=None).fit_transform(df)['A'] # cats ['a','b'], vals ['a','b','a', NaN]
    ForceCategoryTransformer(['a','b'], default_level='b').fit_transform(df)['A'] # cats ['a','b'], vals ['a','b','a','b']
    
    Examples
    --------
    df = pd.DataFrame({'A':['a','b','c']})
    ForceCategoryTransformer().fit_transform(df) #categories= ['a','b','c']
    >>>      A
    >>> 0  'a'
    >>> 1  'b'
    >>> 2  'c'
    
    ForceCategoryTransformer(levels=['a','b'], default_level='DEFAULT').fit_transform(df) #categories= ['a','b','DEFAULT']
    >>>      A
    >>> 0  'a'
    >>> 1  'b'
    >>> 2  'DEFAULT'
    """
    def __init__(self,
                levels = None,
                ordered = False,
                default_level = None):
        self.classes_ = levels
        self.default_level = default_level
        self.ordered = ordered
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.incols = X.columns.values
        self.feature_names_ = self.incols
        if self.classes_ is None:
            x = X.iloc[:,0].astype('category')
            self.classes_ = x.cat.categories.values
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        x = X.iloc[:,0]
        cat_type = CategoricalDtype(categories=self.classes_, ordered=self.ordered)
        x = x.astype(cat_type)
        if self.default_level is not None:
            if self.default_level not in x.cat.categories:
                x = x.cat.add_categories([self.default_level])
            x = x.fillna(self.default_level)
        Xt = pd.DataFrame(x)
        Xt.columns = X.columns
        return Xt
    def transform_dict(self, d):
        v = d.pop(self.feature_names_[0])
        if v not in self.classes_:
            v = self.default_level
        return {self.feature_names_[0]: v}        
    def get_feature_names(self):
        return self.feature_names_


class TopLabelsTransformer(TransformerMixin, object):
    """ For a single categorical column, select only the top most frequent lables and default the others
    
    Parameters:
    -----------
    min_coverage: float in (0..1)
        minimal coverage required in each level, otherwise substituted by `default_key` (default: 0.0)
    max_levels: int >0
        that many levels ordered by highest coverage are retained, all others are grouped under 
        the default key (default: `sys.maxsize`)
    default_name: obj
        key that is substituted for any level which is not in the max_levels most frequent (default: 'DEFAULT')
        
    Attributes
    ----------
    classes_ : list
        list of valid emitted objects.  
    default_key_ : list
        objects emitted if default is to be emitted; if None then no default has ever been emitted!
    default_levels_ : list
        list of levels/lables that are mapped to the default_key; if this is empty the default_key_ never has been emitted!
        
    Examples
    --------
    
    
    
    """
    def __init__(self,
            min_coverage = 0.,
            max_levels = sys.maxsize,
            default_name = 'DEFAULT'):
        self.min_coverage = min_coverage
        self.max_levels = max_levels
        self.default_name = default_name
        #--------------------------
        self.default_key_ = None
        self.default_levels_ = []
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.incols = X.columns.values
        self.feature_names_ = self.incols
        
        levels, counts = np.unique( X.iloc[:,0], return_counts=True )
        idx = np.array(list(reversed(np.argsort(counts))))
        levels, counts, coverage = [ np.take(x, idx ) for x in [levels, counts, counts / np.sum(counts)] ]
        
        #--- decide which levels to take
        self.classes_ = []
        self.default_classes_ = []
        self.coverage_ = []
        for l,c in zip(levels, coverage):
            if len(self.classes_) < self.max_levels and c >= self.min_coverage:
                self.classes_.append(l)
                self.coverage_.append(c)
            else:
                self.default_classes_.append(l)
        
        #--- insert the default key if neccessary
        if len(self.classes_) < len(levels) or self.force_default_level:
            self.default_key_= self.default_name
        else:
            self.default_key_= None
        
        if self.default_key_ is not None:
            self.classes_.append(self.default_name)
            self.coverage_.append( 1. - sum(self.coverage_) )
        return self
    def transform(self, X):
        assert(X.shape[1]==1)
        def xthelper(v):
            if v in self.classes_:
                return v
            if v in self.default_classes_:
                return self.default_key_
            raise Exception("New level encountered not covered in training")
        Xt = pd.DataFrame(X.iloc[:,0].apply(xthelper))
            
        return Xt
    
    def get_feature_names(self):
        return self.feature_names_




