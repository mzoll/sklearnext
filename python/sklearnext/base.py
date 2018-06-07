'''
Created on Nov 17, 2017

@author: marcel.zoll
'''

import pandas

def assert_dfncol(X, ncolumns, verbose=False):
    ''' checks the dimensionality of a dataframe; raises an assertation error
    
    Parameters
    ----------
    X : pandas.DataFrame obj
        the dataframe to ckeck
    ncolumns : int > 0
        number of expected columns to be present
    '''
    if not isinstance(X, pandas.DataFrame):
        raise AssertionError("object passed is not a pandas.DataFrame, but {}".format(type(X)))
    
    ncols= X.shape[1]
    if not ncols == ncolumns:
        raise AssertionError("DataFrame has not the required number of columns of {}, but {}".format(ncolumns, ncols))
    
def assert_isfitted(transformer_inst):
    """ checks for the existence of the attribute 'feature_names_', taht should be created when transformers are fitted
    
    Parameters
    ----------
    transformer_inst : inst
        a transformer instance
    """
    if not hasattr(transformer_inst, 'feature_names_'):
        raise AssertionError("The Transformer seems to have not been fitted yet")
