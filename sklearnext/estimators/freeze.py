'''
Created on Nov 21, 2017

@author: marcel.zoll
'''

from sklearn.base import MetaEstimatorMixin


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
    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_