'''
Created on Dec 8, 2017

@author: marcel.zoll
'''

import sys, copy
import itertools
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin

from joblib import Parallel, delayed

from sklearnext.transformers import OneHotTransformer

import logging
logger = logging.getLogger('SplitterFork')

class SplitterFork(TransformerMixin, object): #_BaseComposition
    """ A Pipeline that splices up a subsequent pipeline based on the yield variable of cat_trans, applies preprocessing by
    pre_trans and the applies sub_pipe individually for each so trained segment.
    
    In the end of processing there will be one pipeline  per remaining level fulfilling the requirements on min_coverage and
    and max_levels, eventually plus an extra pipeline for defaulting levels, which do not fulfill these requirements 
    
    Parameters
    ----------
    cat_trans : class Transformer or TransformerPipeline
        a Transformer which yields a single column categorical output feature of finite number of classes
    pre_trans : class Transformer or TransformerPipeline
        a Transformer which is applied to all data before it is passed to the individualized pipelines
    sub_pipe : class Pipeline
        A pipeline that is gonna be copied and trained for each of the classes yielded by cat_trans
    min_coverage : float (0..1)
        minimal coverage required in each level, otherwise substituted by `default_key` (default: 0.0)
    max_levels : int >0 
        maximal number of highest ranked original levels that are retained (default: sys.maxsize)
    default_name: obj
        name that is substituted for any level with coverage less than `min_coverage` (default: 'DEFAULT')
    propagate_disc_labels : bool
        propagate dummy lables of the categorical feature from `cat_trans` to the `sub_pipes` (default: False)
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
                 min_coverage = 0.,
                 max_levels = sys.maxsize,
                 propagate_disc_labels =False,
                 take_pre_only = False,
                 default_name = 'DEFAULT',
                 n_jobs=1):
        self.cat_trans = cat_trans
        self.pre_trans = pre_trans
        self.sub_pipe = sub_pipe
        self.min_coverage = min_coverage
        self.max_levels = max_levels
        self.propagate_disc_labels = propagate_disc_labels
        self.take_pre_only = take_pre_only
        self.default_name = default_name
        self.n_jobs = n_jobs
        #-----------------
        self.default_key =None
    def fit(self, X, y, **fit_params):
        ''' determine the factor levels, build the fits '''
        #--- determine groups
        Xg = self.cat_trans.fit_transform(X) 
        assert(Xg.shape[1] == 1)
        self.varname = self.cat_trans.get_feature_names()[0]
        
        levels, counts = np.unique( Xg.iloc[:,0], return_counts=True )
        idx = np.array(list(reversed(np.argsort(counts))))
        levels, counts, coverage = [ np.take(x, idx ) for x in [levels, counts, counts / np.sum(counts)] ]
        
        #--- decide which levels to take
        self.levels_ = [] #regular levels with enough coverage one group/subpipe per
        self.default_levels_ = []  #munched up levels with not enough coverage, all project on the last group/subpipe
        self.coverage_ = [] #the coverage of each group subpipe, where the last entry is for the default_levels if they exist
        for l,c in zip(levels, coverage):
            if len(self.levels_) < self.max_levels and c >= self.min_coverage:
                self.levels_.append(l)
                self.coverage_.append(c)
            else:
                self.default_levels_.append(l)
        
        #--- insert the default key if neccessary
        if len(self.levels_) < len(levels):
            self.default_key_= self.default_name
        else:
            self.default_key_= None
        
        if self.default_key_ is not None:
            self.levels_.append(self.default_name)
            self.coverage_.append( 1. - sum(self.coverage_) )
        
        logger.trace('grouping')
        #--- translate labels to group_indexes
        self.lg_dict = { l:g for g,l in enumerate(self.levels_) }
        def xghelper(v):
            res = self.lg_dict.get(v)
            if res is not None:
                return res
            if v in self.default_levels_:
                return self.lg_dict.get(self.default_key_)
            raise Exception("Unknown level '%s' encountered for variable '%s', and no default enabled" % (v, self.varname))
        xgroups = Xg.iloc[:,0].apply(xghelper).values
        
        logger.trace("pre")
        #--- compute the pre_pipe result and split up into groups
        Xt = self.pre_trans.fit_transform(X, y)
        if not self.take_pre_only:
            if isinstance(Xt, pd.SparseDataFrame):
                Xt = Xt.to_dense()
            Xt = pd.concat([X, Xt], axis=1)
        if self.propagate_disc_labels:
            self.level_encoder_ = OneHotTransformer(sparse_output=False).fit(Xg)
            Xgt = self.level_encoder_.transform(Xg)
            #from sklearn.preprocessing import LabelEncoder
            #self.level_encoder_ = LabelEncoder().
            #self.level_encoder_.classes_ = np.array(levels)
            #Xgt = Xg.apply(self.level_encoder_.transform, axis=1)
            Xt = pd.concat([Xt, Xgt], axis=1)
        
        Xtgroups = { gk:df for gk,df in Xt.groupby(xgroups) }
        ygroups = { gk:df for gk,df in y.groupby(xgroups) }
        
        logger.trace("segment fit")
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
        def xghelper(v):
            res = self.lg_dict.get(v)
            if res is not None:
                return res
            if v in self.default_levels_:
                return self.lg_dict.get(self.default_key_)
            raise Exception("Unknown level '%s' encountered for variable '%s', and no default enabled" % (v, self.varname))
        xgroups = Xg.iloc[:,0].apply(xghelper)
        
        #--- compute the pre_pipe result and split up into groups           
        Xt = self.pre_trans.transform(X)
        if not self.take_pre_only:
            if isinstance(Xt, pd.SparseDataFrame):
                Xt = Xt.to_dense()
            Xt = pd.concat([X, Xt], axis=1)
        if self.propagate_disc_labels:
            Xgt = self.level_encoder_.transform(Xg)
            Xt = pd.concat([Xt, Xgt], axis=1)
        
        results = []
        for gk, Xp in Xt.groupby(xgroups):
            r = self.sub_pipes_[gk].predict(Xp)
            results.append( r )
        return pd.concat(results).reindex(index= X.index)
#    def predict(self, X):
#        return self._transform(X)
#    def transform(self, X):
#        return self._transform(X)
    #------------
    def _get_tpPre(self,X):
        ''' transform an input X by applying the grouping, the pre and then the individual pipelines '''
        Xg = self.cat_trans.transform(X)        
        
        #--- translate labels to group_indexes
        def xghelper(v):
            res = self.lg_dict.get(v)
            if res is not None:
                return res
            if v in self.default_levels_:
                return self.lg_dict.get(self.default_key_)
            raise Exception("Unknown level '%s' encountered for variable '%s', and no default enabled" % (v, self.varname))
        xgroups = Xg.iloc[:,0].apply(xghelper)
    
        #--- compute the pre_pipe result and split up into groups           
        Xt = self.pre_trans.transform(X)
        if not self.take_pre_only:
            if isinstance(Xt, pd.SparseDataFrame):
                Xt = Xt.to_dense()
            Xt = pd.concat([X, Xt], axis=1)
        if self.propagate_disc_labels:
            Xgt = self.level_encoder_.transform(Xg)
            Xt = pd.concat([Xt, Xgt], axis=1)
            
        return Xt, xgroups
        
    def predict(self, X):
        Xt, xgroups = self._get_tpPre(X)
        
        results = []
        for gk, Xp in Xt.groupby(xgroups):
            r = self.sub_pipes_[gk].predict(Xp)
            results.append( r )
        return pd.concat(results).reindex(index= X.index)
    
    def transform(self, X):
        Xt, xgroups = self._get_tpPre(X)
        
        results = []
        for gk, Xp in Xt.groupby(xgroups):
            r = self.sub_pipes_[gk].transform(Xp)
            results.append( r )
        return pd.concat(results).reindex(index= X.index)    
    
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
    
    def predict_dict(self, d):
        _d = d.copy()
        k,v = list(self.cat_trans.transform_dict(_d).items())[0]
    
        g = self.lg_dict.get(v)
        
        if g is None:
            if v in self.default_levels_:
                g = self.lg_dict.get(self.default_key_)
            else:
                raise Exception("Unrecognized level: %s" % str(v))
        
        _dt = self.pre_trans.transform_dict(d)
        if not self.take_pre_only:
            _dt.update(_d)
        if self.propagate_disc_labels:
            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FIXME
            _g = self.level_encoder_.transform_dict(g)
            _dt.update(_g)
        
        dt = self.sub_pipes_[g].predict_dict(_dt)
        return dt



#==================================
# Auxilaries
#=================================
#auxilary to CategoryFork
def _fit_one_fittable(fittable, X, y):
    return fittable.fit(X,y)