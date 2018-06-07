'''
Created on Feb 22, 2018

@author: marcel.zoll
'''

import numpy as np
import pandas as pd
from ..base import assert_dfncol
from sklearn.base import TransformerMixin

class SKLTransformerWrapper(TransformerMixin):
    ''' Wraps around a sklearn Transformer, taking a single column input and transforms to one or multiple columns output
    
    Parameters
    ----------
    skltransformer : sklearn.Transformer obj
        must take a single column input and outputs one or multiple columns
    suffix : str
        suffix will be appended to the in-column name, if multiple columns are output an index (0,1,..) is added
    '''
    def __init__(self, skltransformer, suffix):
        self.skltransformer = skltransformer
        self.suffix = suffix
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.incols = X.columns
        self.skltransformer.fit(X.values)
        
        Xtnp = self.skltransformer.transform(X.values[0:1])
        
        #determine feature names
        if Xtnp.shape[1] > 1:
            self.feature_names = []
            for i in range(Xtnp.shape[1]):
                self.feature_names.append(self.incols[0]+self.suffix+str(i))             
        else:
            self.feature_names = [self.incols[0]+self.suffix]
            
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        Xtnp = self.skltransformer.transform(X.values)
        
        Xt = pd.DataFrame( data=Xtnp, columns = self.feature_names, index=X.index)
        return Xt
    def get_feature_names(self):
        return self.feature_names
    def transform_dict(self, d):
        x = d.pop(self.incols[0])
        xt = self.skltransformer.transform(np.array(x))
        for i in range(len(self.feature_names)):
            d[self.feature_names[i]] = xt[i][0]
        return d