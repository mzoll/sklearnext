'''
Created on Dec 8, 2017

@author: marcel.zoll
'''

import sys, copy
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.utils.metaestimators import _BaseComposition

from sklearn.externals.joblib import Parallel, delayed

import logging
logger = logging.getLogger('CategoryFork')


class CategoryFork(TransformerMixin, object): #_BaseComposition
    """ A Pipeline that forks to dedicated pipelines on basis of factor-levels found for a specified categorical variable
    
    Parameters
    ----------
    pipeline : class Pipeline
        A pipeline that is gonna be copied and trained for each of the 
    varname : string
        name of the categorical variable
    level_list : None or list of tuples or single values
        list of tuples of levels, that should form a group with one dedicated pipeline each.
        Using an entry 'DUMMY' as a dedicated group, will sum up all levels other than the ones specified in this group.
        If unspecified with None, each encountered level will be treated as a seperate group (default: None)
    n_jobs : int
        number of jobs to execute on; effective value min(.,len(self,pipeline_list_)) (default: 1)
    
    Attributes
    ----------
    pipeline_list_ : list of pipelines
        the trained pipelines, one for each levelgroup 
    levels_ : list of strings
        the levels / labels taken by the variable
    levelgroups_ : list of ints
        the groups that each level in levels_ is to be associated to
    coverage_ : list of floats 
        relative coverage in each of the groups determined in fit()
    """
    def __init__(self, pipeline, varname, level_lists = None, n_jobs=1):
        """ use None in level_lists to signal arbitrary level (all none specified)
        e.g. level_lists = [(level0), (level1, level2), (None)]
        """
        self.pipeline = pipeline
        self.varname = varname
        self.n_jobs = n_jobs

        self.pipeline_list_ = None
        self.levelgroups_ = None
        self.levels_ = None
        
        if level_lists is not None:
            #check and construct levelgroups_
            self.levels_ = []
            self.levelgroups_ = []
            
            self.default_enabled_ = False
            for i,lg in enumerate(level_lists):
                if isinstance(lg, list) or isinstance(lg, tuple):
                    for l in lg:
                        if not isinstance(l, str):
                            raise Exception('too deep nested list in parameter levelgroups')
                        if l is None:
                            raise Exception("not allowed NONE in nested list")
                        if l=='DEFAULT':
                            if self.default_enabled_:
                                raise Exception("'Dummy' specified twice")
                            self.default_enabled_ = True
                        self.levels_.append(l)
                        self.levelgroups_.append(i)
                else:
                    if lg=='DEFAULT':
                        if self.default_enabled_:
                            raise Exception("'DEFAULT' specified twice")
                        self.default_enabled_ = True
                    self.levels_.append(lg)
                    self.levelgroups_.append(i)
    
    def fit(self, X, y, **fit_params):
        ''' determine the factor levels if not already done, build the fits if not already done '''
        if self.levels_ is None:
            #only fit if we have no predefined
            self.levels_=[]
            for u in X[self.varname].unique():
                self.levels_.append(u)
            self.levelgroups_ = [i for i in range(len(self.levels_))]
        
        #instantize a pipeline for each level group now
        logger.debug('copy init')
        self.pipeline_list_ = [copy.deepcopy(self.pipeline) for lg in self.levelgroups_ for i in np.unique(self.levelgroups_)]
        logger.debug('done')
        
        #segment the dataframe
        xgroups = self._segmentX(X).values
        logger.debug('group done')
        if len(np.unique(xgroups)) != len(np.unique(self.levelgroups_)):
            raise Exception("Not all pipelines can be fitted because of insufficent data coverage")
        
        Xgroups = { gk:df for gk,df in X.groupby(xgroups) }
        ygroups = { gk:df for gk,df in y.groupby(xgroups) }
        logger.debug('fit')
        pls = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_fittable)(self.pipeline_list_[gk], Xgroups[gk], ygroups[gk])
            for gk in Xgroups.keys())
        self.pipeline_list_ = pls

        self.coverage_ = np.array([ df.shape[0]/X.shape[0] for gk,df in Xgroups.items() ])

        return self
    
    def _segmentX(self, X):
        #NOTE this could be done also row-wise, maybe its even better to do for rowwise evaluation
        """
        xfactors = X[self.varname].unique()
        if len(set(xfactors).difference(set(self.levels_))) != 0:
            raise Exception('Some Factors occuring for %s not encountered in fit(); have trained list: %s' % \
                            (self.varname, str(self.levels_)))
        """
        lg_dict = dict(zip(self.levels_, self.levelgroups_))
        
        def xghelper(v):
            res = lg_dict.get(v)
            if res is not None:
                return res
            if self.default_enabled_:
                return lg_dict['DEFAULT']
            raise Exception("Unknown level '%s' encountered for variable '%s', and no default enabled" % (v, self.varname))
        xgroups = X[self.varname].apply(xghelper)
        return xgroups
        
    def predict(self, X):
        ''' predict for X where predictions are individual for each factor level,
        in the end splice it all together again
        '''
        xgroups = self._segmentX(X)
        
        results = []
        for gk, Xp in X.groupby(xgroups):
            r = self.pipeline_list_[gk].predict(Xp)
            results.append( r )
        return pd.concat(results).reindex(index= X.index)
    
    def transform(self, X):
        """ depricated """
        xgroups = self._segmentX(X)
        
        results = []
        for gk, Xp in X.groupby(xgroups):
            r = self.pipeline_list_[gk].transform(Xp)
            results.append( r )
        return pd.concat(results).reindex(index= X.index)
    
    def get_feature_importances_deep(self):
        features = set( fi[0] for p in self.pipeline_list_ for fi in p.get_feature_importances() )
        from collections import defaultdict
        dd = defaultdict(list)
        for p in self.pipeline_list_:
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

