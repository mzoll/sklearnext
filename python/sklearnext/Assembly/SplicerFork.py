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

from ..Transformers.LabelDummy import LabelDummyTransformer
from ..base import assert_dfncol


class SplitterFork(TransformerMixin, object): #_BaseComposition
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
    class TheseLabelsDummyTransformer(object):
        """ one hot encode categorical column with dummies for these lables 
        
        Parameters
        ----------
        lables : list of str
            list of the names of the to onehot-encode lables
        sparse_output : bool
            return a sparse DataFrame (true), else dense DataFrame (false) (default: True)
        """
        def __init__(self, lables, sparse_output = False):
            self.classes_ = lables
            self.sparse_output = sparse_output
        def className_to_dummyName_(self, cn):
            return '_'.join([self.invar_, str(cn)])
        def constructTransdict(self):
            s_dummy = pd.Series( { e:False for e in self.feature_names_ } )
            def xthelper(c):
                sx = s_dummy.copy()
                sx[self.className_to_dummyName_(c) ] = True
                return(sx)
            self.transdict = { c:xthelper(c) for c in self.classes_ }
            self.transdict[None] = s_dummy
            #optimize for single row evals
            self.transdfdict = { k:pd.DataFrame(v).T for k,v in self.transdict.items() }
               
        def fit(self, X, y=None, **fit_params):
            self.invar_ = X.columns[0]
            
            self.feature_names_ = [ self.className_to_dummyName_(c) for c in self.classes_ ]
            self.constructTransdict()
            
            return self
        
        def transform(self, X):
            check_is_fitted(self, 'classes_')
            assert_dfncol(X, 1)
        
            if X.shape[0]==1: #optimize for single row evals
                df = self.transdfdict.get(X[invar_].values[0])            
                df.index = X.index
                return df
            
            def xthelper(row):
                v = row[self.invar_]
                r = self.transdict.get(v)
                return r
            Xt = X.apply(xthelper, axis=1)
            
            if self.sparse_output:
                return Xt.to_sparse(fill_value=False)
            return Xt
    
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
        if len(self.levels_) < len(levels):
            self.default_key_= self.default_name
        else:
            self.default_key_= None
        
        if self.default_key_ is not None:
            self.levels_.append(self.default_name)
            self.coverage_.append( 1. - sum(self.coverage_) )
        
        print('grouping')
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
        
        
        print("pre")
        #--- compute the pre_pipe result and split up into groups
        Xt = self.pre_trans.fit_transform(X, y)
        if isinstance(Xt, pd.SparseDataFrame):
            Xt = Xt.to_dense()
    
        
        print("segment fit")
        #--- create pipes and fit them for every group
        #fit the easy ones : for all levels with enough coverage the sub_pipe is copied locally (DANGER or so I hope) and trained
        #print(Xt.columns)
        gk_idx_groups = y.index.groupby(xgroups)
        gkd = len(self.lg_dict)-1
        
        #self.sub_pipes_ = [ copy.deepcopy(self.sub_pipe) for l in self.levels_ ]
        pls = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_fittable)(self.sub_pipe_, Xtgroups.loc[idx,:], ygroups.loc[idx]) for gk, idx in gk_idx_groups if gk!=gkd)
        self.sub_pipes_ = pls
        
        #do the hard one: Add the one hot encoding to the grouped block
        #then ovesample from rest until min_coverage is reached  
        
        gkd_idx = gk_idx_groups[gkd]
        gkx_idx = y.index[~ y.index.isin(gkd_idx)]
        
        oversample_size = int(len(y)*self.min_coverage- len(gkd_idx))
        
        gkx_idx_os = gkx_idx( np.random.permutation( [True]*oversample_size + [False]*(len(gkx_idx)-oversample_size) ) )
        
        #we know which indexes to take, so now put it all together:
        Xg_gkd = Xg.loc[gkd_idx]
        self.level_encoder_ = TheseLabelsDummyTransformer(self.default_levels_, sparse_output=False).fit(Xg)
        Xgt_gkd = self.level_encoder_.transform(Xgt_gkd)
        
        Xt_gkd = Xt.loc[gkd_idx]
        
        XtXgt_gkd = pd.concat(Xt_gkd, Xgt_gkd, axis=1) 
        
        XtXgt_gkdgkxos = pd.concat(XtXgt_gkd, Xt.loc[gkx_idx_os])
        
        y_gkdgkxos = pd.concat(y.loc[gkd_idx], y.loc[gkx_idx_os])
        
        #now fit the last pipe
        self.sub_pipes_.append(_fit_one_fittable(self.sub_pipe_, XtXgt_gkdgkxos, y_gkdgkxos))

        return self
    def _transform(self, X):
        ''' transform an input X by applying the grouping, the pre and then the individual pipelines '''
        Xg = self.cat_trans.fit_transform(X)        
        
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
        Xt = self.pre_trans.fit_transform(X)
        if isinstance(Xt, pd.SparseDataFrame):
            Xt = Xt.to_dense()
        
        gk_idx_groups = y.index.groupby(xgroups)
        gkd = len(self.lg_dict)-1
        
        #self.sub_pipes_ = [ copy.deepcopy(self.sub_pipe) for l in self.levels_ ]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_one)(self.sub_pipes_[gk], Xtgroups.loc[idx,:], ygroups.loc[idx]) for gk, idx in gk_idx_groups if gk!=gkd)
        
        #do the hard one: Add the one hot encoding to the grouped block and predict  
        gkd_idx = gk_idx_groups[gkd]
        Xt_gkd = Xt.loc[gkd_idx]
        Xg_gkd = Xg.loc[gkd_idx]
        Xgt_gkd = self.level_encoder_.transform(Xg)
        
        XtXgt_gkd = pd.concat(XtXgt_gkd, Xt.loc[gkx_idx_os])
        
        results.append(self.sub_pipes_[gkd_idx].predict(XtXgt_gkd))
        
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
#auxilary to CategoryFork
def _fit_one_fittable(fittable, X, y):
    return fittable.fit(X,y)

def _predict_one(est, X, y):
    return est.predict(X,y)