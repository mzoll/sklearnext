'''
Created on Dec 8, 2017

@author: marcel.zoll
'''

__all__ = ['FeatureUnion', 'Pipeline', 'TransformerPipe']


import sys, copy
import itertools
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, MetaEstimatorMixin

from sklearn.pipeline import _fit_transform_one, _transform_one, _fit_one_transformer
from sklearn import clone
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.utils.validation import check_memory

from sklearn import pipeline

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
            delayed(_fit_transform_one)(trans, weight, X, y, **fit_params)
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
    def transform_dict(self, d):
        result_dicts = [trans.transform_dict(d.copy()) for name, trans, weight in self._iter()]
        return { k:v for _d in result_dicts for k,v in _d.items() }


#=================
# Pipeline
#=================
class Pipeline(pipeline.Pipeline):
    """ wrap the sklearn.Pipeline and provide some additional functionality """
    #def fit(self, X, y, **fit_params):
    #    ''' make y non optional '''
    #    pipeline.Pipeline(self, X, y, **fit_params)
    def get_feature_importances(self):
        imps = self.steps[-1][1].feature_importances_
        feats = self.steps[-2][1].get_feature_names()
        return list(zip(feats, imps))
    def predict_dict(self, d):
        dt = d
        for n,t in self.steps[:-1]:
            dt = t.transform_dict(dt)
        if self._final_estimator is not None: #self._final_estimator
            #feats = self.steps[-2][1].get_feature_names()
            #valvec = np.array([ dt.pop(f) for f in feats ])
            #dt = self._final_estimator.predict_dict(valvec)
            dt = self._final_estimator.predict_dict(dt)
        return dt

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
           
    def fit(self, X, y= None, **fit_params):
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
    def transform_dict(self, d):
        dt = d
        for name, transform in self.steps:
            dt = transform.transform_dict(dt)
        return dt
    def get_feature_names(self):
        return self.steps[-1][-1].get_feature_names()
