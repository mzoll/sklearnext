'''
Created on Nov 21, 2017

@author: marcel.zoll
'''
from sklearn.base import TransformerMixin, MetaEstimatorMixin

class FreezeEstimator(MetaEstimatorMixin, object):
    """ prohibits the call on fit of an underlaying mutable object, so freezing it """
    def __init__(self, estimator):
        self.estimator = estimator
    def fit(self, X, y, **fit_params):
        return self
    def predict(self, X):
        return self.estimator.predict(X)
    
class FreezeTransformer(TransformerMixin, object):
    """ prohibits the call on fit of an underlaying mutable object, so freezing it """
    def __init__(self, transformer):
        self.transformer = transformer
    def fit(self, X, y, **fit_params):
        return self
    def transform(self, X):
        return self.transformer.transform(X)