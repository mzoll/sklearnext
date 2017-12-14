'''
Created on Dec 8, 2017

@author: marcel.zoll
'''


import sys, copy
import itertools
import numpy as np
import pandas as pd

from sklearn.utils.metaestimators import _BaseComposition
from sklearn import pipeline 
from sklearn.pipeline import _fit_transform_one, _transform_one, _fit_one_transformer
from sklearn import clone
from sklearn.utils import check_array, safe_mask
from sklearn.utils.validation import check_memory

from .Pipeline import FeatureUnion


class FeatureSelectPipeline(_BaseComposition, object):
    """
    Selects Features with most importance and shoots down those with least significance
    in an interative process of elimination one at a time;
    eliminates whole transformators if their features are of no importance.
    
    Paramters
    ---------
    feature_union : instance of FeatureUnion
        A FeatureUnion generating named features (needs to follow the sklearn.FeatureUnion interface)
    estimator : instane of Estimator
        A Estimator that can be repeatatly retrained and has attribute feature_importances_ . will be used to iteratively elimnate
        the weakest features with a final round of fitting 
    threshold : float greater 0.
        threshold on the importance of a feature below which this feature can possibly be eliminated
    elimnate_nmax_iter : int
        eliminate at most nmax features with each iteration before a reevaluation by fitting, importance evaluation and scoring is required; choose
        smaller stepsize to eliminate bunches of features at a time at the expense of time spend in the estimator reevaluation (default: sys.maxsize)
    """
    def __init__(self,
                 feature_union,
                 estimator,
                 threshold= 1e-5,
                 elimnate_nmax_iter = sys.maxsize):
        self.feature_union_null = feature_union
        self.estimator = estimator
        self.threshold = threshold #we support only float for now
        self.elimnate_nmax_iter = elimnate_nmax_iter
        
    def fit(self, X, y, **fit_params):
        """Fit X and y by every transformer and perform one training with the estimator
        Eliminate features with zero importance right away and train estimator again.
        Then iteratively remove elimnate_nmax_iter fatures below the threshold at each iteration,
        train, score and validate the improvement. If at any point of iterative removal the score
        gets worse, this step is reverted and the last good estimator is taken.
        At last with the final estimator the transformers in the FeatureUnion are evaluated if their
        produced features partake positively
        
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        #--- fit all transformers
        self.feature_union_null.fit(X, y, **fit_params)
        
        #--- get the exposed features by every transformer
        self.feature_names_null = self.feature_union_null.get_feature_names()
        
        #--- transform and compose catanated output from transformers
        Xs = self.feature_union_null.transform(X)
        
        #--- one round of estimator fitting
        print("Fitting first (naive) estimator")
        self.estimator.fit(Xs, y)
        
        #--- get the feature importances and shoot down a first round of features
        self.feature_importances_null = copy.copy(self.estimator.feature_importances_)
        named_importances_null = list(zip(self.feature_names_null, self.feature_importances_null))
        fmask = np.array([ imp > 0. for imp in self.feature_importances_null ])
        for f,i in itertools.compress(named_importances_null, fmask):
            print("eliminating feature %s with zero importance" % (f))
        
        #---rescore and prepare for iterative removal
        print("Fitting reduced feature estimator")
        Xp = Xs.loc[:, safe_mask(Xs, fmask)]
        self.estimator.fit(Xp, y, **fit_params)
        imp = copy.copy(self.estimator.feature_importances_)
        fnames = itertools.compress(self.feature_names_null, fmask)
        named_imp = list(zip(fnames, imp))
        fimp_ranked = list(sorted(named_imp, key= lambda e: e[1]))
        elim = list(map(lambda e: e[1]<self.threshold, fimp_ranked))
        
        #--- iteratively remove features until all above threshold and test for improvement
        def can_improve():
            return sum(fmask)>1 and any(elim)
        if can_improve():
            score = self.estimator.score(Xp, y)                            
        while can_improve():
            print("Iterative removal and refit")
            
            fmask_next = copy.copy(fmask)
            for i in range(min(self.elimnate_nmax_iter, sum(elim))):
                print("trying to eliminate feature '%s' with importance %e" % (fimp_ranked[i][0], fimp_ranked[i][1]))
                fmask_next[ self.feature_names_null.index(fimp_ranked[i][0]) ] = False
            
            est_next = copy.deepcopy(self.estimator) #DANGER sklearn.clone
            Xp = Xs.loc[:, safe_mask(Xs, fmask_next)]
            est_next.fit(Xp, y)
            score_next = est_next.score(Xp, y)

            if score_next < score:
                print("no improvement")
                break
                
            print("improvement: score %f -> %f" % (score, score_next))
            self.estimator = est_next
            fmask = fmask_next
            fnames = itertools.compress(self.feature_names_null, fmask)
            imp = copy.copy(self.estimator.feature_importances_)
            named_imp = list(zip(fnames, imp))
            fimp_ranked = list(sorted(named_imp, key= lambda e: e[1]))
            elim = list(map(lambda e: e[1]<self.threshold, fimp_ranked))
            score = score_next
            
        self.fmask_null = fmask 
        
        #--- determine remaining features and what is needed to remain in action of the transformers in pipeline
        self.fnames_apt = list(itertools.compress(self.feature_names_null, self.fmask_null))
        
        def fnames_helper(prefix, fn):
            return '__'.join([prefix, fn])
        def fnameslist_helper(prefix, fn_list):
            return [fnames_helper(prefix, fn) for fn in fn_list ]
        self.tmask_apt = np.array(
            [ len(set(self.fnames_apt) & set(fnameslist_helper(n, trans.get_feature_names()))) >0 \
                 for n,trans in self.feature_union_null.transformer_list])
        
        if all(self.tmask_apt):
        #if sum( [v==0 for v in self.get_transformer_importances_apt().values()] )==0:
            print('All transformers contribute')
            self.feature_union_apt = self.feature_union_null
            self.fmask_apt = self.fmask_null
        else:
            names, t = zip(*self.feature_union_null.transformer_list)
            print("Eliminating not contributing transformers:\n\t", list(itertools.compress( names, ~self.tmask_apt)))
            self.feature_union_apt = FeatureUnion(
                list(itertools.compress( self.feature_union_null.transformer_list, self.tmask_apt)))
        
            self.fmask_apt = np.array([ fnames_helper(n,f) in self.fnames_apt \
                for n,trans in self.feature_union_apt.transformer_list for f in trans.get_feature_names() ])
        
        return self
    
    def predict(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = self.feature_union_apt.transform(X)
        Xp = Xs.loc[:, safe_mask(Xs, self.fmask_apt)]
        return self.estimator.predict(Xp)
    # estimator reach-thruough
    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_
    def get_feature_importances_null(self):
        return list(zip(self.feature_names_null, self.feature_importances_null))
    def get_feature_importances_apt(self):
        imps = self.estimator.feature_importances_
        return list(zip( self.fnames_apt, imps ))
    def get_feature_importances(self):
        return self.get_feature_importances_apt()
    # transformer info
    def get_transformer_importances_null(self):
        #trans_imp = []
        #for names,trans in zip(*self.feature_union_null.transformer_list) for n in names:
        #    imp_sum = 0
        #    for f,i in self.feature_importances_null:
        #        if f.split('__')[0] == n:
        #            imp_sum += i
        #trans_imp.append( tuple(n, imp_sum) )     
        trans_imp = {}
        for f,i in self.get_feature_importances_null():
            n = f.split('__')[0]
            trans_imp[n] = (trans_imp[n] if n in trans_imp else 0) + i
        return list(trans_imp.items())
    def get_transformer_importances_apt(self):
        trans_imp = {}
        for f,i in self.get_feature_importances():
            n = f.split('__')[0]
            trans_imp[n] = (trans_imp[n] if n in trans_imp else 0) + i
        return list(trans_imp.items())
