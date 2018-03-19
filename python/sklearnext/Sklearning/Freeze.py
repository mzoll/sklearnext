'''
Created on Nov 21, 2017

@author: marcel.zoll
'''

from sklearn.base import TransformerMixin, MetaEstimatorMixin

class FreezeEstimator(MetaEstimatorMixin, object):
    """ prohibits the call on fit of an underlaying mutable estimator object, so freezing it 
    
    Parameters
    ----------
    estimator: estimator obj
        the already fitted estimator, needs to support predict and predict_dict (optional)
    """
    def __init__(self, estimator):
        self.estimator = estimator
    def fit(self, X, y, **fit_params):
        return self
    def predict(self, X):
        return self.estimator.predict(X)
    def predict_dict(self, d):
        return self.estimator.predict_dict(d)
    

class FreezeTransformer(TransformerMixin, object):
    """ prohibits the call on fit of an underlaying mutable transformer object, so freezing it
    
    Paraneters
    ----------
    transformer : transformer obj
        the already fitted transformer, needs to support transform and transform_dict (optional) 
     """
    def __init__(self, transformer):
        self.transformer = transformer
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X):
        return self.transformer.transform(X)
    def transform_dict(self, d):
        self.transformer.transform_dict(d)
    def get_feature_names(self):
        return self.transformer.get_feature_names()
    