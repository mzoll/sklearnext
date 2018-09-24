'''
Created on Dec 8, 2017

@author: marcel.zoll
'''

import numpy as np

from sklearn.base import BaseEstimator

class DummyClassifier(BaseEstimator, object):
    """ A dummy Classifier which just randomly generates an outcome from its attribute classes_ 
    
    Parameters
    ----------
    classes : list of obj
        the class labels that are to be rnadomly sampled sampled
    """
    def __init__(self, classes):
        self.classes_ = classes
    def fit(self, X, y, **fit_params):
        self.feature_importances_ = np.array([1./len(self.classes_)]*3)
        return self
    def predict(self, X):
        base_l = np.linspace(0, len(self.classes_)-1, len(self.classes_))
        probas = self.predict_proba(X)
        return np.array( [ base_l[[bool(i) for i in p]] for p in probas ] ).flatten()
    def predict_proba(self, X):
        base_l = [0.]*(len(self.classes_)-1) + [1.]
        return np.array( [ np.random.permutation(base_l).tolist() for _ in range(len(X))] )
    

class DummyRegressor(BaseEstimator, object):
    """ A dummy Regressor generating outcomes on the interval (minval,maxval) 
    
    Parameters
    ----------
    minval, maxval : float (with minval <= maxval)
        lower/upper bound on the output range interval that will be samples (currently an ascending series)
    """
    def __init__(self, minval, maxval):
        self.min_= minval
        self.max_= maxval
    def fit(self, X, y, **fit_params):
        self.feature_importances_ = np.array([1./len(X[1,:])]*3)
        return self
    def predict(self, X):
        return np.linspace(self.min_, self.max_, len(X))
    