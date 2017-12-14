'''
Created on Nov 17, 2017

@author: marcel.zoll
'''

import pandas

def assert_dfncol(X, ncolumns):
    assert( isinstance(X, pandas.DataFrame) )
    assert( X.shape[1] == ncolumns )