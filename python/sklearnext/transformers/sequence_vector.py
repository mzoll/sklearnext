'''
Created on Jan 18, 2018

@author: marcel.zoll
'''

import copy
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

from sklearnext.base import assert_dfncol

from sklearn.base import TransformerMixin 
from sklearn.utils.validation import check_is_fitted

class SequenceVectorEncoder(TransformerMixin, object):
    """ Encodes a vector of labels as a number rowise aligning columns, all holding the same levels. Truncation and padding is applied. 
    
    Parameters
    ----------
    categories_list : list of obj or None
        a list of the levels, that should be kept intact; if None (default) this list will be inferred by unique elements in the vectors
    default_level : obj
        the default level to infer if an entry is not found to be contained in the categories list
    prioretize_head : bool
        If set to True vectors are first cut to fit specified max length from the tail end and then aligned on the first 
        element with padding for shorter sequences on the right side applied. [Default: False]
        NOTE : if the to processed object is an list that has been appended to, setting this parameter to True will prioretize the head of the sequence, 
        and therefore will preserve the structure of the oldest elements in the series, possibly discarding the newest entries if they do not fit the max length requirement.
    maxentries : int or None
        maximium number of vector entries; if none the global maximum length of vectors is taken
    integerencode : bool 
        reencode the with integers from 0...n_classes (True), where 0 is the padding marker [Default: False]
        
    Examples
    --------
    df= pd.DataFrame({'Vec':[['a','b','c','a'], ['a','b','d']})
    SequenceVectorEncoder(['a','b','c', default_level=['c'], left_allign=True, integerencode=True]).fit_transform(X)
    >>> Pandas.DataFrame({'Vec_0':[1,1], 'Vec_1':[2,2], 'Vec_2':[2,3], 'Vec_3':[1,0]})
    """
    def __init__(self, 
                categories_list = None,
                default_level = 'UNKNOWN',
                padding_level = 'MISSING', 
                prioretize_head = False, 
                maxentries = None, 
                integerencode = False):
        self.categories = categories_list
        self.default_level = default_level
        self.padding_level = padding_level
        self.prioretize_head = prioretize_head
        self.maxentries = maxentries
        self.integerencode = integerencode
        if self.categories is not None and self.padding_level in self.categories:
            raise Exception("Cannot currently handle if padding-level is natively contained in cateories")
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.incols= X.columns
        vec_list = X.iloc[:,0].values
        maxlen = max([len(vec) for vec in vec_list])        
        if self.maxentries is None:
            self.maxentries = maxlen
        incol = X.columns[0]
        if self.prioretize_head:
            self.feature_names = [ "{}_{}".format(incol, i) for i in range(self.maxentries)]
        else:
            self.feature_names = [ "{}_{}".format(incol, i) for i in range(maxlen-self.maxentries, maxlen)]
            
        if self.categories is None:
            s = set()
            for vec in X.iloc[:,0].values:
                s = s | set(vec)
            self.categories = list(s)
            
        self.classes_ = self.categories
        if self.padding_level not in self.classes_:
            self.classes_ = [self.padding_level] + self.classes_
        else:
            raise Exception("Cannot currently handle if padding-level is natively contained in cateories")
        if self.default_level not in self.classes_:
            self.classes_.append(self.default_level)
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
            
        if self.prioretize_head:
            Xt = X.iloc[:,0].apply(_pad_priohead, self.maxentries, self.padding_level)
        else:
            Xt = X.iloc[:,0].apply(_pad_priotail, self.maxentries, self.padding_level)
        
        Xt.columns = self.feature_names
        
        #now resolve defaulting of entries not contained categories
        cat_type = CategoricalDtype(categories=self.classes_, ordered=True)
        def xt_col_helper(col):
            col = col.astype(cat_type).fillna(self.default_level)
            if self.integerencode:
                col = col.cat.rename_categories(list(range(len( col.cat.categories))))
            return col
        Xtt = Xt.apply( xt_col_helper, axis=0)
        return Xtt
        
    def get_feature_names(self):
        return self.feature_names_
    
    def transform_dict(self, d):
        """ transform from a dictionary (needs to hold the incolumn key)"""
        vec = d.pop(self.incols[0])
        # encode default first, do padding latter 
        if self.integerencode:
            trans_dict= {k:i for i,k in enumerate(self.classes_)}
            #figure out the default_integer
            default_en= trans_dict.get(self.default_level)
            def l_helper(e):
                r = trans_dict.get(e)
                return r if r is not None else default_en
            vec = list(map(l_helper, vec))
        else:
            vec = list(map(lambda e: e if e in self.categories else self.default_level), vec)
        
        #now do the padding if neccessary
        padding_level = self.padding_level if not self.integerencode else 0 
        if self.prioretize_head:
            if len(vec) > self.maxentries:
                vec = vec[:self.maxentries]
            else:
                vec = vec + [padding_level]*(self.maxentries-len(vec))
        else:
            if len(vec) > self.maxentries:
                vec = vec[len(vec)-self.maxentries:]
            else:
                vec = [padding_level]*(self.maxentries-len(vec)) + vec
        
        d.update({f:v for f,v in zip(self.feature_names, vec)})
        return d
        

class SequenceVectorCheckboxes(TransformerMixin, object):
    """ Create Checkboxes for all values in classes, for each value in the passed sequence vector the checkbox will be ticked
    if defaultname is not None, the checkbox under that value will be ticked if the encountered value cannot be found in classes
    (default_name might be appended to classes if not contained)
    
    Parameters
    ----------
    classes : list of objects or None
        the classes for which checkboxes are generated
    default_name : string
        specify a default checkbox, which is checked if no other checkbox match for an entry (default: None)
    """
    def __init__(self, classes=None, default_name = None):
        self.default_name = default_name
        self.classes = classes
        
    def _class_to_feature_name(self, classname):
        return "{}_{}".format(self.incols[0], classname)
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.incols = X.columns
        
        if self.classes is None:
            s = set()
            for vec in X.iloc[:,0].values:
                s = s | set(vec)
            self.classes = list(s)
        
        if self.default_name is not None and self.default_name not in self.classes:
            self.classes.append(self.default_name)
        
        self.feature_names = [ self._class_to_feature_name(c) for c in self.classes]
        self.tick_dict = { k:e for e,k in enumerate(self.classes) }
        self._dummy_checkbox = [False] * (len(self.classes))
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        def xthelper(vec):
            cb = copy.copy(self._dummy_checkbox)
            for c in set(vec):
                e = self.tick_dict.get(c)
                if e is None:
                    if self.default_name is not None:
                        cb[self.tick_dict.get(self.default_name)] = True
                    continue
                cb[e] = True
            return pd.Series(cb)
        Xt = X.iloc[:,0].apply(xthelper)
        Xt.columns = self.feature_names
        return Xt
    def transform_dict(self, d):
        val_vec = d.pop(self.incols[0])
        d.update({ k:False for k in self.feature_names})
        for val in val_vec:
            if val in self.classes:
                d[self._class_to_feature_name(val)] = True
            elif self.default_name is not None:
                d[self._class_to_feature_name(self.default_name)]= True
        return d
    
 
#auxilaries   
def _pad_priohead(vec, maxentries, padding_level):
    """ clip and pad a list 'maxentries', so that it fits exactly the size of 'maxentries', prioretize preserving the head of that list """
    if len(vec) > maxentries:
        p= vec[:maxentries]
    else:
        p= vec + [padding_level]*(maxentries-len(vec))
    return pd.Series(p)
        
def _pad_priotail(vec, maxentries, padding_level):
    """ clip and pad a list 'maxentries', so that it fits exactly the size of 'maxentries', prioretize preserving the tail of that list """
    if len(vec) > maxentries:
        p= vec[len(vec)-maxentries:]
    else:
        p= [padding_level]*(maxentries-len(vec)) + vec
    return pd.Series(p)