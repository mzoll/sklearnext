'''
Created on Oct 26, 2017

@author: marcel.zoll
'''

import pandas

def XyMuxer(X, y):
    """ given two sets X, y by the same index (WeblogId), return the overlap in each """
    common_index = X.index.intersection(y.index)
    return X.loc[common_index], y.loc[common_index]

