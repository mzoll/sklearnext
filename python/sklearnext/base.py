'''
Created on Nov 17, 2017

@author: marcel.zoll
'''

import pandas

def assert_dfncol(X, ncolumns):
    ''' checks the dimensionality of a dataframe; raises an assertation error
    
    Parameters
    ----------
    X : pandas.DataFrame obj
        the dataframe to ckeck
    ncolumns : int > 0
        number of expected columns to be present
    '''
    assert( isinstance(X, pandas.DataFrame) )
    assert( X.shape[1] == ncolumns )
    