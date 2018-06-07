'''
Created on May 14, 2018

@author: marcel.zoll
'''


import pandas as pd
import numpy as np
import math

from sklearnext.base import assert_dfncol

from sklearn.base import TransformerMixin 

class CyclicSineCosineTransformer(TransformerMixin, object):
    """ take a single column that has a certain constant periodicity, and express it as its sine and cosine transformatives
    on the interval [-1...1], alternatively onto the interval [0...1]
    
    Parameters
    ----------
    periodicity : float >0
        the period of the input value
    pure_positive : bool
        picture to the interval [0...1] instead of [-1...1] (default: False)    
        
    Examples
    --------
    df = pandas.DataFrame({'A':[0,1,2,3,4],)
    t = CyclicSineCosineTransformer(periodicity=4).fit(df) 
    t.transform(df)
    >>> pandas.DataFrame({'A_cyclicsin': [0., 1., 0., -1., 0.], 'A_cycliccos': [1., 0., -1., 0., 1.] })
    d = {'A': 0}
    t.transform_dict(d)
    >>>{'A_cyclicsin': 0., 'A_cyclicsin': 1.}
    t = CyclicSineCosineTransformer(periodicity=4, pure_positive=1).fit(df)
    t.transform(df)
    >>> pandas.DataFrame({'A_cyclicsin': [0.5, 1., 0.5, 0., 0.5], 'A_cycliccos': [1., 0.5, 0., 0.5, 1.] })
    d = {'A': 0}
    t.transform_dict(d)
    >>>{'A_cyclicsin': 0.5, 'A_cyclicsin': 1.0}
    
    """
    def __init__(self, periodicity, pure_positive=False):
        self.periodicity = periodicity
        self.pure_positive = pure_positive
    def fit(self, X, y= None, **fit_params):
        assert_dfncol(X, 1)
        self.incols= list(X.columns)
        self.feature_names = [self.incols[0]+'_cyclicsin', self.incols[0]+'_cycliccos']
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        def xthelper(val):
            t = val/self.periodicity *2.*math.pi
            return pd.Series([math.sin(t), math.cos(t)])
        Xt = X.iloc[:,0].apply(xthelper)
        if self.pure_positive:
            Xt = Xt.apply(lambda t: 0.5*(t+1.), axis=1)
        Xt.columns= self.feature_names        
        return Xt
    def transform_dict(self, d):
        val = d.pop(self.incols[0])
        p = val/self.periodicity *2.*math.pi
        t = np.array([math.sin(p), math.cos(p)])
        if self.pure_positive:
            t = 0.5*(t+1.)
        d.update( dict(zip(self.feature_names, t)) )
    def get_feature_names(self):
        return self.feature_names