
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts

def train_test_split(X, y, test_size=.2, random_state=0, stratify=None, checks=False):
    """ do the same as the sklearn implementation, but return dataframes with sorted indexes """ 
    if checks:
        #assert( np.all(X.index.isin(y.index)) )
        X.index == y.index
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size, random_state, stratify= stratify)

    X_train.sort_index(inplace = True)
    y_train.sort_index(inplace = True) #y_train.reindex(index= X_train.index)
    X_test.sort_index(inplace = True)
    y_test.sort_index(inplace = True) #y_test.reindex(index= X_test.index)
    
    return X_train, X_test, y_train, y_test