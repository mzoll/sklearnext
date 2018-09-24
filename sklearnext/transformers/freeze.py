'''
Created on Nov 21, 2017

@author: marcel.zoll
'''

from sklearn.base import TransformerMixin
    

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
    