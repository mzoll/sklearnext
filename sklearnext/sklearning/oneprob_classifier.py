'''
Created on Dec 8, 2017

@author: marcel.zoll
'''

import numpy as np

from sklearn.base import MetaEstimatorMixin
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
        
    Attributes
    ----------
    estimator : Estimator object
        The Estimator object
    predictClass : int
        the class that is predicted and `predict()` returns the probability of
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
        idx = next(i for i,v in enumerate(self.estimator.classes_) if v==self.predictClass)
        return np.array( [e[idx] for e in self.estimator.predict_proba(X) ] )
    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_
    def score(self, X, y, sample_weight = None):
        return self.estimator.score(X, y, sample_weight)