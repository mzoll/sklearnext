'''
Created on Oct 26, 2017

@author: marcel.zoll
'''

import pandas
import pandas as pd

def XyMuxer(X, y):
    """ given two sets X, y by the same index (WeblogId), return the overlap in each """
    Xt = X.loc[ X.index.isin(y.index)]
    yt = y.loc[ y.index.isin(Xt.index) ]
    return Xt, yt
