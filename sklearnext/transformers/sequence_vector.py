'''
Created on Jan 18, 2018

@author: marcel.zoll
'''

import copy
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

import logging
from numpy.polynomial.tests.test_classes import classes
logger = logging.getLogger('seqvec')

from sklearnext.base import assert_dfncol, assert_isfitted

from sklearn.base import TransformerMixin 
from sklearn.utils.validation import check_is_fitted


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
    
    Attributes
    ----------
    categories_ : list of str
        list of all found represented categories in the data, excluding padding level and default_level 
    classes_ : list of str
        list of all classes represented in the output;
        notice that index 0 is reserved for the padding_level and index (size-1) is reserved for the default_level
    maxentries_ : int>0 
        number of entries in the output vector
    
    Examples
    --------
    df= pd.DataFrame({'Vec':[['a','b','c','a'], ['a','b','d']]})
    SequenceVectorEncoder(['a','b','c'], default_level='c', prioretize_head=True, integerencode=True]).fit_transform(X)
    >>> pandas.DataFrame({'Vec_0':[1,1], 'Vec_1':[2,2], 'Vec_2':[2,3], 'Vec_3':[1,0]})
    """
    def __init__(self, 
                categories_list = None,
                default_level = 'UNKNOWN',
                padding_level = 'MISSING', 
                prioretize_head = False, 
                maxentries = None, 
                integerencode = False):
        self.categories_ = categories_list
        self.default_level = default_level
        self.padding_level = padding_level
        self.prioretize_head = prioretize_head
        self.maxentries_ = maxentries
        self.integerencode = integerencode
        #--- fields for save configuration during fit
        self._fit_categories = categories_list is None
        self._fit_maxentries = maxentries is None
        #--- do some preperatory things
        if self.categories_ is not None:
            if self.padding_level in self.categories_:
                raise Exception("Cannot currently handle if padding-level is natively contained in categories")
            if self.default_level in self.categories_:
                self.categories_.remove(self.default_level)
        
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self._incols= X.columns
        if self._fit_maxentries:
            self.maxentries_ = max([len(vec) for vec in X.iloc[:,0].values])
        
        self.feature_names_ = [ "{}_{}".format(X.columns[0], i) for i in range(self.maxentries_)]
            
        if self._fit_categories:
            s = set([])
            for vec in X.iloc[:,0].values:
                s = s | set(vec)
            self.categories_ = list(s)
            #check for problems
            if self.padding_level in self.categories_:
                raise Exception("Cannot currently handle if padding-level is contained in categories")
            if self.default_level in self.categories_:
                self.categories_.remove(self.default_level)
        
        self.classes_ = [self.padding_level] + self.categories_ + [self.default_level]
        logger.info("fit done")
        #a bit of preparation, for speed later
        if self.integerencode:
            self._trans_enum_dict= {k:i for i,k in enumerate(self.classes_)}
        return self
    def transform(self, X):
        assert_isfitted(self)
        assert_dfncol(X, 1)
        #transform to aligned multicolumn
        if self.prioretize_head:
            Xt = X.iloc[:,0].apply(_pad_priohead, maxentries= self.maxentries_, padding_level=self.padding_level)
        else:
            Xt = X.iloc[:,0].apply(_pad_priotail, maxentries= self.maxentries_, padding_level=self.padding_level)
        Xt.columns = self.feature_names_
        
        #now resolve defaulting of entries not contained categories
        cat_type = CategoricalDtype(categories=self.classes_, ordered=True)
        def xt_col_helper(col):
            col = col.astype(cat_type).fillna(self.default_level)
            if self.integerencode:
                col = col.cat.rename_categories(list(range(len( col.cat.categories))))
            return col
        Xtt = Xt.apply( xt_col_helper, axis=0)
        
        logger.info("transform done")
        return Xtt
        
    def get_feature_names(self):
        return self.feature_names_
    
    def transform_dict(self, d):
        """ transform from a dictionary (needs to hold the incolumn key)"""
        vec = d.pop(self._incols[0])
        # encode default first, do padding later 
        if self.integerencode:
            #figure out the default_integer
            default_en= len(self.classes_)-1
            def l_helper(e):
                r = self._trans_enum_dict.get(e)
                return r if r is not None else default_en
            vec = list(map(l_helper, vec))
        else:
            vec = list(map(lambda e: e if e in self.categories_ else self.default_level), vec)
        
        #now do the padding if neccessary
        padding_level = self.padding_level if not self.integerencode else 0 
        if self.prioretize_head:
            if len(vec) > self.maxentries_:
                vec = vec[:self.maxentries_]
            else:
                vec = vec + [padding_level]*(self.maxentries_-len(vec))
        else:
            if len(vec) > self.maxentries_:
                vec = vec[len(vec)-self.maxentries_:]
            else:
                vec = [padding_level]*(self.maxentries_-len(vec)) + vec
        
        d.update({f:v for f,v in zip(self.feature_names_, vec)})
        return d
        

class SequenceVectorEncoderNEXT(TransformerMixin, object):
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
    
    Attributes
    ----------
    classes_ : list of str
        list of all classes represented in the output 
    maxentries_ : int>0 
        number of entries in the output vector
        
    Examples
    --------
    df= pd.DataFrame({'Vec':[['a','b','c','a'], ['a','b','d']})
    SequenceVectorEncoder(['a','b','c', default_level=['b'], left_allign=True, integerencode=True).fit_transform(df)
    >>> Pandas.DataFrame({'Vec_0':[1,1], 'Vec_1':[3,3], 'Vec_2':[2,3], 'Vec_3':[1,0]})
    """
    def __init__(self, 
                categories_list = None,
                default_level = 'UNKNOWN',
                padding_level = 'MISSING', 
                prioretize_head = False, 
                maxentries = None, 
                integerencode = False):
        raise Exception("needs update")
        self.categories = categories_list
        self.default_level = default_level
        self.padding_level = padding_level
        self.prioretize_head = prioretize_head
        self.maxentries_ = maxentries
        self.integerencode = integerencode
        #--- fields for save configuration during fit
        self._fit_categories = categories_list is None
        self._fit_maxentries = maxentries is None
    def fit(self, X, y=None, **fit_params):
        #assert_dfncol(X, 1)
        self._incols= X.columns
        if self._fit_maxentries:
            self.maxentries_ = max([len(vec) for vec in X.iloc[:,0].values])
        
        self.feature_names_ = [ "{}_{}".format(X.columns[0], i) for i in range(self.maxentries_)]
            
        if self._fit_categories:
            s = set()
            for vec in X.iloc[:,0].values:
                s = s | set(vec)
            #self.categories = list(s)
        
        self.categories_ = list( s - set([self.default_level, self.padding_level])) 
        self.classes_ = list( s & set([self.default_level, self.padding_level]))
        
        self.translation_dict = {c:e+1 for e,c in enumerate(self.categories_)} 
        self.translation_dict.update( {self.default_level:-1, self.padding_level:0} )
        
        self.translate_dict_rev =  {e+1:c for e,c in enumerate(self.categories_)}
        self.translate_dict_rev.update( {-1: self.default_level, 0: self.padding_level} )
        logger.info("fit done")
        return self
    def transform(self, X):
        assert_isfitted(self)
        assert_dfncol(X, 1)
        
        def _pad_priohead(vec):
            """ clip and pad a list 'maxentries', so that it fits exactly the size of 'maxentries', prioretize preserving the head of that list """
            if len(vec) > self.maxentries_:
                vec= vec[:self.maxentries_]
            outvec = []
            for v in vec:
                v_code = self.translation_dict.get(v)
                if v_code is None:
                    outvec.append( -1 )
                else:
                    outvec.append( v_code )
            if len(outvec) < self.maxentries_:
                outvec.extend( [0] *(self.maxentries_ - len(vec)) )        
            return pd.Series(outvec)
        
        def _pad_priotail(vec):
            """ clip and pad a list 'maxentries', so that it fits exactly the size of 'maxentries', prioretize preserving the head of that list """
            if len(vec) > self.maxentries_:
                vec= vec[-self.maxentries_:]
            outvec = []
            if len(vec) < self.maxentries_:
                outvec = [0] *(self.maxentries_ - len(vec))
            for v in vec:
                v_code = self.translation_dict.get(v)
                if v_code is None:
                    outvec.append( -1 )
                else:
                    outvec.append( v_code )
            
            return pd.Series(outvec)
      
        if self.prioretize_head:
            Xt = X.iloc[:,0].apply(_pad_priohead)
        else:
            Xt = X.iloc[:,0].apply(_pad_priotail)        
        Xt.columns = self.feature_names_
        
        #now resolve defaulting of entries not contained categories
        if not self.integerencode:
            cat_type = CategoricalDtype(categories=self.translation_dict.values(), ordered=True)

            def xt_col_helper(col):
                col = col.astype(cat_type)
                col = col.cat.rename_categories(self.translate_dict_rev)
                return col
            Xt = Xt.apply( xt_col_helper, axis=0)
            
        logger.info("transform done")
        return Xt
        
    def get_feature_names(self):
        return self.feature_names_
    
    def transform_dict(self, d):
        """ transform from a dictionary (needs to hold the incolumn key)"""
        vec = d.pop(self._incols[0])
        # encode default first, do padding latter
        
        if len(vec) > self.maxentries_:
            if self.prioretize_head:
                vec = vec[:self.maxentries_]
            else:
                vec[-self.maxentries_:]
            
        if self.integerencode:
            default_en= self.translate_dict.get(self.default_level)
            def l_helper(e):
                r = self.translate_dict.get(e)
                return r if r is not None else default_en
            vec = list(map(l_helper, vec))
        else:
            vec = list(map(lambda e: e if e in self.categories_ else self.default_level), vec)
        
        #now do the padding if neccessary
        padding_level = self.padding_level if not self.integerencode else 0 
        if self.prioretize_head:
            if len(vec) > self.maxentries_:
                vec = vec[:self.maxentries_]
            else:
                vec = vec + [padding_level]*(self.maxentries_-len(vec))
        else:
            if len(vec) > self.maxentries_:
                vec = vec[len(vec)-self.maxentries_:]
            else:
                vec = [padding_level]*(self.maxentries_-len(vec)) + vec
        
        d.update({f:v for f,v in zip(self.feature_names_, vec)})
        return d


class SequenceVectorCheckboxes(TransformerMixin, object):
    """ Create Checkboxes for all values in classes, for each value in the passed sequence vector the checkbox will be ticked.
    if _default_name_ is not None, the checkbox under that value will be ticked if the encountered value cannot be found in _classes_
    (_default_name_ might be appended to classes if not contained)
    
    Parameters
    ----------
    classes : list of objects or None
        the classes for which checkboxes are generated (optional)
    default_name : string
        specify a default checkbox, which is ticked if an entry matches no other checkbox (default: None)
        
    Attributes
    ----------
    classes_ : list of str
        list of all classes represented in the output
    
    Examples
    --------
    df= pd.DataFrame({'Vec':[['a','b','c','a'], ['a','b','d']]})
    SequenceVectorCheckboxes().fit_transform(df)
    >>> pandas.DataFrame({'Vec_c':[True,False], 'Vec_b':[True,True], 'Vec_a':[True,True], 'Vec_z':[False,True]})
    SequenceVectorCheckboxes(classes = ['a', 'b', 'c'], default_name = 'z').fit_transform(df)
    >>> pandas.DataFrame({'Vec_a':[True,True], 'Vec_b':[True,True], 'Vec_c':[True,False], 'Vec_z':[False,True]})
    """
    def __init__(self,
            classes = None,
            default_name = None):
        self.classes_ = classes
        self.default_name = default_name
        #--- fields for configuration during fit
        self._fit_classes = classes is None  # classes should be fitted by the module itself
    def _class_to_feature_name(self, classname):
        return "{}_{}".format(self._incols[0], classname)
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self._incols = X.columns
        
        if self._fit_classes:
            s = set()
            for vec in X.iloc[:,0].values:
                #if not isinstance(vec, list):
                #    logger.error("Got unexpected non-list value while processing column: {}".format(vec))
                #    continue
                s = s | set(vec)
            self.classes_ = list(s)
            
        if self.default_name is not None and self.default_name not in self.classes_:
            self.classes_.append(self.default_name)
        
        self.feature_names_ = [ self._class_to_feature_name(c) for c in self.classes_]
        self.tick_dict = { k:e for e,k in enumerate(self.classes_) }
        self._dummy_checkbox = [False] * (len(self.classes_))
        
        logger.info("fit done")
        return self
    def transform(self, X):
        assert_isfitted(self)
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
        Xt.columns = self.feature_names_
        
        logger.info("transform done")
        return Xt
    def transform_dict(self, d):
        val_vec = d.pop(self._incols[0])
        d.update({ k:False for k in self.feature_names_})
        for val in val_vec:
            if val in self.classes_:
                d[self._class_to_feature_name(val)] = True
            elif self.default_name is not None:
                d[self._class_to_feature_name(self.default_name)]= True
        return d
    def get_feature_names(self):
        return self.feature_names_
    

class SequenceVectorElement(TransformerMixin, object):
    """ Extract a specific element from a sequence vector
    
    Parameters
    ----------
    nth : int
        take this element in the vector; native python indexing
    default : obj
        if the required element cannot be obtained, for example because vector has insufficient length, place this value (default: None)
        
    Examples
    --------
    df = pd.DataFrame({'A':[ [1,2,3,4], [5,6,7], [] ]})
    SequenceVectorElement(0, 99).fit_transform(df)
    >>> pd.DataFrame({'A_0':[1,5,99]})
    SequenceVectorElement(-1, 99).fit_transform(df)
    >>> pd.DataFrame({'A_-1':[4,7, 99]})
    SequenceVectorElement(3, 99).fit_transform(df)
    >>> pd.DataFrame({'A_1':[4,99,99]})
    """
    def __init__(self,
            nth,
            default = None):
        self.nth = nth
        self.default = default
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self._incols = X.columns
        self.feature_names_ = [self._incols[0]+'_'+str(self.nth)]
        return self
    def transform(self, X):
        assert_isfitted(self)
        assert_dfncol(X, 1)        
        def xt_helper(val):
            if isinstance(val, list):
                try:
                    el = val[self.nth]
                except:
                    el = self.default
                return el
            return self.default
        xt = X.loc[:,self._incols[0]].apply(xt_helper)
        Xt = pd.DataFrame(xt)
        Xt.columns = self.feature_names_
        return Xt 
    def transform_dict(self, d):
        val_vec = d.pop(self._incols[0])
        if isinstance(val_vec, list):
            try:
                el = val_vec[self.nth]
            except:
                el = self.default
        else:
            el = self.default
        d[self.feature_names_[0]] = el
        return d
    def get_feature_names(self):
        return self.feature_names_
    
    
class SequenceVectorElementRemove(TransformerMixin, object):
    """ Extract an element from a sequence vector
    
    Parameters
    ----------
    n_many : int
        remove that many elements from the vector
    at_front : bool
        remove the elemnts from the vectors front (True) or its back (False)
        
    Examples
    --------
    df = pd.DataFrame({'A':[ [1,2,3,4], [5,6,7] ]})
    SequenceVectorElementRemove(2, False).fit_transform(df)
    >>> DataFrame({'A':[ [1,2], [5] ]})
    SequenceVectorElementRemove(2, True).fit_transform(df)
    >>> DataFrame({'A':[ [3,4], [7] ]})
    SequenceVectorElementRemove(3, True).fit_transform(df)
    >>> DataFrame({'A':[ [4], [] ]})
    """
    def __init__(self,
            n_many,
            at_front = False):
        self.n_many = n_many
        self.at_front = at_front
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self._incols = X.columns
        self.feature_names_ = [self._incols[0]+'_mod']
        return self
    def transform(self, X):
        assert_isfitted(self)
        assert_dfncol(X, 1)        
        
        if self.at_front:
            def xt_helper(val):
                #assert(isinstance(val, list))
                if len(val)<=self.n_many:
                    return []
                return val[self.n_many:]
        else:
            def xt_helper(val):
                #assert(isinstance(val, list))
                if len(val)<=self.n_many:
                    return []
                return val[:-self.n_many]
        xt = X.loc[:,self._incols[0]].apply(xt_helper)
        Xt = pd.DataFrame(xt)
        Xt.columns = self.feature_names_
        return Xt 
    def transform_dict(self, d):
        val_vec = d.pop(self._incols[0])
        #assert(isinstance(val, list))
        if len(val_vec)<= self.n_many:
            el = []
        elif self.at_front:
            el = val_vec[self.n_many:]
        else:
            el = val_vec[:-self.n_many]
        d[self.feature_names_[0]] = el
        return d
    def get_feature_names(self):
        return self.feature_names_
    