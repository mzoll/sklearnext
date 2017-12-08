'''
Created on Nov 3, 2017

@author: marcel.zoll

classes to assemble a complex and especially nested model by chaining Transformers and Estimators in FeatureUnions and Pipelines   
These classes superseed the sklearn.* implementations by explicitly taking an DataFrame as input and producing (for most cases)
a DataFrame as output. This requires that the input as well as all intermediate steps handle indexed Dataframes and write their results
as columns of such.

The input-output scheme is as follows:
Transformers-- In: DataFrame Out: FataFrame
Estimator-- In: DataFrame Out: numpy-array
EstimatorWrapper-- wrap Estimator or Pipeline-- In: nparray Out: DataFrame
FeatureUnions-- assembly of Tranformers-- In: DataFrame Out: DataFrame
TransformerPipe-- assembly of Transformers-- In: DataFrame Out: DataFrame
ForkingPipeline-- a pipeline-- In: DataFrame Out: nparray
Pipeline-- assembly of Transformers plus Estimator-- In: DataFrame Out:nparray
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

#=================
# Transformers
#=================
class ColumnsAll(TransformerMixin, object):
    """ only writes out the specified variables """
    def fit(self, X, y=None, **fit_params):
        self.feature_names = X.columns
        return self
    def transform(self, X):
        return X
    def get_feature_names(self):
        return self.feature_names

class ColumnsSelect(TransformerMixin, object):
    """ only writes out the specified variables
    
    Parameters
    ----------
    column_names : list of strings
        Names of the columns that oart to be selected """
    def __init__(self, column_names):
        if isinstance(column_names, list): 
            self.feature_names = column_names
        elif isinstance(column_names, str):
            self.feature_names = [column_names]
        else:
            raise TypeError('varname_list needs to be list or str (depricated)')
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X):
        return X[self.feature_names]
    def get_feature_names(self):
        return self.feature_names


#=================
# Estimator
#=================
from sklearn.base import is_classifier
class OneProbClassifierWrapper(MetaEstimatorMixin, object):
    """ wraps around a Classifier on returning only the outcome of one predictive class
    
    Parameters
    ----------
    estimator : instance of Classifier
        The estimator, which needs to be an classifier following the native sklearn implementation
    predictClass : int
        The ordeal index of the class to predict in the estimators predict output matrix;
        if all predictive classes are populated and ordered starting at 0 this is identical to the predict class label name (default: 1)  
    """
    def __init__(self, estimator, predictClass = 1):
        if not is_classifier(estimator):
            raise Exception("not a classifier as estimatorimator")
        self.estimator = estimator
        self.predictClass = predictClass
    def fit(self, X, y, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        if len(self.estimator.classes_) <= self.predictClass or self.predictClass not in self.estimator.classes_ :
            raise Exception('impossible to select predictClass %s from estimatorimated classes %s' % (self.predictClass, self.estimator.classes_))
        return self
    def predict(self, X):
        """ predict the probability for the predictClass outcome """
        idx = next(i for i,v in enumerate(self.estimator.classes_) if v==self.predictClass)
        return np.array( [e[idx] for e in self.estimator.predict_proba(X) ] )
    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_
    def score(self, X, y, sample_weight = None):
        return self.estimator.score(X, y, sample_weight)


#from sklearn.base import MetaEstimatorMixin
class EstimatorWrapper(TransformerMixin, MetaEstimatorMixin, object):
    """ warps around a estimator making it behave like a Transformer; this can explicitly be applied on Pipelines 
    
    Parameters
    ----------
    estimator : instance of Classifier
        The estimator, which needs to be an classifier following the native sklearn implementation
    feature_names : list of strings
        a list of feature_names which will be assigned to the estimator generated columns of the predict matrix
    """   
    def __init__(self, estimator, feature_names=None):
        if not hasattr(estimator, 'predict'):
            raise TypeError("Estimator object has no attribute predict")
        if hasattr(estimator, 'transform'):
            raise TypeError("Estimator object already implements 'transform'")
        self.estimator = estimator
        self.feature_names = feature_names
    def fit(self, X, y, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        self.estimator.predict(X)
        Xt = pd.DataFrame(self.estimator.predict(X), index = X.index)
        if Xt.shape[1] != len(self.feature_names):
            raise ValueError('outnames to short')
        Xt.columns = self.feature_names
        return Xt
    def get_feature_names(self):
        return self.feature_names


#=================
# FeatureUnion
#=================
class FeatureUnion(pipeline.FeatureUnion):
    """ Like sklearn.FeatureUnion but requires output and all Transformers to handle indexed DataFrames"""
    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X, y,
                                        **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            raise Exception('possible internal error in %s; All transformers are None' % (str(self)))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        
        Xs = pd.concat(Xs, axis=1) 
        return Xs
    
    def transform(self, X):
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
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            raise Exception('possible internal error in %s; All transformers are None' % (str(self)))
            
        Xs = pd.concat(Xs, axis=1) 
        
        return Xs
    

#=================
# Pipeline
#=================
class Pipeline(pipeline.Pipeline):
    """ convenience wrap the sklearn.Pipeline and provide some additional functionality """
    def get_feature_importances(self):
        imps = self.steps[-1][1].feature_importances_
        feats = self.steps[-2][1].get_feature_names()
        return list(zip(feats, imps))
    

class TransformerPipe(TransformerMixin, object):
    """ a pipeline only consistent of transformers; provides convenience when transforming
    by a sequential chain of Transformers
    Parameters
    ----------
    steps : list of pairs (string, transformer instance)
        named steps to process the data through
    memory : UNKONWN
        UNKOWN
    """
    def __init__(self, steps, memory=None):
        self.steps = steps
        self._validate_steps()
        self.memory = memory
        
    def _validate_steps(self):
        names, transformers = zip(*self.steps)
        # validate names
        #self._validate_names(names)

        # validate transformers
        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All steps should be transformers and implement fit and transform."
                                " '%s' (type %s) doesn't" % (t, type(t)))
    
    def _fit_transform(self, X, y=None, **fit_params):
        """ fit and transform X by transforming it by every step in sequence """
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for step_idx, (name, transformer) in enumerate(self.steps):
            if transformer is None:
                pass
            else:
                if hasattr(memory, 'cachedir') and memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve
                    # backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transfomer
                Xt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, Xt, y,
                    **fit_params_steps[name])
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)
        
        return Xt
           
    def fit(self, X, y, **fit_params):
        self._fit_transform(X, y, **fit_params)
        return self
    def transform(self, X):
        Xt = X
        for name, transform in self.steps:
            #if transform is not None:
                Xt = transform.transform(Xt)
        return Xt
    def fit_transform(self, X, y= None, **fit_params):
        return self._fit_transform(X)
    def get_feature_names(self):
        return self.steps[-1][-1].get_feature_names()


class CategoryFork(TransformerMixin, object): #_BaseComposition
    """ A Pipeline that forks to dedicated pipelines on factor-levels of a certain categorical variable
    
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
        self.pipeline_list_ = [copy.deepcopy(self.pipeline) for lg in self.levelgroups_ for i in np.unique(self.levelgroups_)]
        
        #segment the dataframe
        xgroups = self._segmentX(X)
        
        if len(np.unique(xgroups)) != len(np.unique(self.levelgroups_)):
            raise Exception("Not all pipelines can be fitted because of insufficent data coverage")
        
        Xgroups = { gk:df for gk,df in X.groupby(xgroups.values) }
        ygroups = { gk:df for gk,df in y.groupby(xgroups.values) }
        
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
            try:
                return lg_dict[v]
            except:
                pass
            if self.default_enabled_:
                return lg_dict['DEFAULT']
            else:
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
            results.append( pd.Series(r, index = Xp.index) )
        return pd.concat(results).reindex(X.index)
    
    def transform(self, X):
        """ depricated """
        xgroups = self._segmentX(X)
        
        results = []
        for gk, Xp in X.groupby(xgroups):
            r = self.pipeline_list_[gk].transform(Xp)
            results.append( r )
        return pd.concat(results).reindex(X.index)
    
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
        
#auxilary to CategoryFork
def _fit_one_fittable(fittable, X, y):
    return fittable.fit(X,y)


#========================
# FeatureSelectPipeline
#========================

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
                 elimnate_nmax_iter = sys.maxsize,
                 n_jobs=1):
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
