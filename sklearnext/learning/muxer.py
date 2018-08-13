'''
Created on Oct 26, 2017

@author: marcel.zoll
'''

def XyMuxer(X, y):
    """ given two sets X, y by the same index, return the intersection in index for each """
    common_index = X.index.intersection(y.index)
    return X.loc[common_index], y.loc[common_index]

