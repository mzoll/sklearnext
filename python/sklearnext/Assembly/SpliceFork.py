'''
Created on Dec 8, 2017

@author: marcel.zoll
'''

import sys, copy
import itertools
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn import pipeline 
from sklearn.pipeline import _fit_transform_one, _transform_one, _fit_one_transformer
from sklearn import clone
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.utils import check_array, safe_mask
from sklearn.utils.validation import check_memory


class SpliceFork(TransformerMixin, object): #_BaseComposition
    """ A Pipeline that splices up a subsequent pipeline based on the yield variable of cat_trans, applies preprocessing by
    pre_trans and the applies sub_pipe individually for each so trained segment
    
    Parameters
    ----------
    cat_trans : class Transformer or TransformerPipeline
        a Transformer which yields a single column categorical output feature of finite number of classes
    pre_trans : class Transformer or TransformerPipeline
        a Transformer which is applied to all data before it is passed to the individualized pipelines
    sub_pipe : class Pipeline
        A pipeline that is gonna be copied and trained for each of the classes yielded by cat_trans
    take_pre_only : bool
        only take the features yielded by pre_trans to be parsed into individual training by the sub_pipe's (default: False)
    n_jobs : int
        number of jobs to execute on; effective value min(.,len(self,pipeline_list_)) (default: 1)
    
    Attributes
    ----------
    sub_pipes_ : list of pipelines
        the trained pipelines, one for each levelgroup 
    levels_ : list of strings
        the levels / labels taken by the variable
    coverage_ : list of floats 
        relative coverage in each of the groups determined in fit()
    """
    def __init__(self,
                 cat_trans,
                 pre_trans,
                 sub_pipe,
                 take_pre_only = False,
                 n_jobs=1):
        self.cat_trans = cat_trans
        self.pre_trans = pre_trans
        self.sub_pipe = sub_pipe
        self.take_pre_only = take_pre_only
        self.n_jobs = n_jobs
    def fit(self, X, y, **fit_params):
        ''' determine the factor levels, build the fits '''
        #--- determine groups
        Xg = self.cat_trans.fit_transform(X)        
        assert(Xg.shape[1] == 1)
        self.varname = self.cat_trans.get_feature_names()[0]
        self.levels_ = np.unique(Xg).tolist()
        
        #--- translate labels to group_indexes
        lg_dict = { l:g for g,l in enumerate(self.levels_) }
        def xghelper(v):
            return lg_dict.get(v)
        xgroups = Xg.iloc[:,0].apply(xghelper).values
        
        #--- compute the pre_pipe result and split up into groups
        Xt = self.pre_trans.fit_transform(X, y)
        if not self.take_pre_only:
            if isinstance(Xt, pd.SparseDataFrame):
                Xt = Xt.to_dense()
            Xt = pd.concat([X, Xt], axis=1)
        
        Xtgroups = { gk:df for gk,df in Xt.groupby(xgroups) }
        ygroups = { gk:df for gk,df in y.groupby(xgroups) }
        
        #--- create pipes and fit them for every group 
        self.sub_pipes_ = [ copy.deepcopy(self.sub_pipe) for l in self.levels_ ]
        pls = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_fittable)(self.sub_pipes_[gk], Xtgroups[gk], ygroups[gk])
            for gk in Xtgroups.keys())
        self.sub_pipes_ = pls

        self.coverage_ = np.array([ df.shape[0]/X.shape[0] for gk,df in Xtgroups.items() ])
        return self
    def _transform(self, X):
        ''' transform an input X by applying the grouping, the pre and then the individual pipelines '''
        Xg = self.cat_trans.transform(X)        
        
        #--- translate labels to group_indexes
        lg_dict = { l:g for g,l in enumerate(self.levels_) }
        def xghelper(v):
            res = lg_dict.get(v)
            if res is not None:
                return res
            raise Exception("Unknown level '%s' encountered for variable '%s', and no default enabled" % (v, self.varname))
        xgroups = Xg.iloc[:,0].apply(xghelper)
        
        #--- compute the pre_pipe result and split up into groups           
        Xt = self.pre_trans.transform(X)
        if not self.take_pre_only:
            if isinstance(Xt, pd.SparseDataFrame):
                Xt = Xt.to_dense()
            Xt = pd.concat([X, Xt], axis=1)
        
        results = []
        for gk, Xp in Xt.groupby(xgroups):
            r = self.sub_pipes_[gk].predict(Xp)
            results.append( r )
        return pd.concat(results).reindex(index= X.index)
    def predict(self, X):
        return self._transform(X)
    def transform(self, X):
        return self._transform(X)
    def get_feature_importances_deep(self):
        features = set( fi[0] for p in self.sub_pipes_ for fi in p.get_feature_importances() )
        from collections import defaultdict
        dd = defaultdict(list)
        for p in self.sub_pipes_:
            fi_dict = dict(p.get_feature_importances())
            for f in features:
                dd[f].append( fi_dict[f] if f in fi_dict.keys() else 0. )
        return list(dd.items())
                
    def get_feature_importances(self):    
        fi = []
        for f,il in self.get_feature_importances_deep():
            fi.append( (f, sum(np.array(il) * self.coverage_ )) )
        return fi
    @property
    def feature_importances_(self):
        ''' wrap this as property for chaining to work correctly  '''
        return self.get_feature_importances()

#==================================
# Auxilaries
#=================================
#auxilary
def _fit_one_fittable(fittable, X, y):
    return fittable.fit(X,y)
