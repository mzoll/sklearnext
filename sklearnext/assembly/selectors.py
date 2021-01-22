"""
Created on Dec 8, 2017

@author: marcel.zoll
"""


from sklearn.base import TransformerMixin


class ColumnsAll(TransformerMixin, object):
    """ passes through all columns unaltered, a unitary operation (checks are in place) """
    def fit(self, X, y=None, **fit_params):
        self.feature_names = X.columns.values
        return self
    def transform(self, X):
        assert( set(self.feature_names) == set(X.columns.values) )
        return X
    def transform_dict(self, d):
        #assert( set(self.feature_names) == set(d.keys()))  # NOTE disabled for speed
        return d
    def get_feature_names(self):
        return self.feature_names


class ColumnsSelect(TransformerMixin, object):
    """ passes through only the specified columns, a unitary operation
    
    Parameters
    ----------
    column_names : list of strings
        Names of the columns that oart to be selected 
    """
    def __init__(self, column_names):
        if isinstance(column_names, list): 
            self.feature_names = column_names
        elif isinstance(column_names, str):
            self.feature_names = [column_names]
        else:
            raise TypeError('varname_list needs to be list or str (deprecated)')
    def fit(self, X, y=None, **fit_params):
        assert( set(self.feature_names).issubset(set(X.columns.values)))
        return self
    def transform(self, X):
        """ inert operation: just return all selected variables """
        return X[self.feature_names]
    def transform_dict(self, d):
#     del_keys= []
#     for k in d.keys():
#         if k not in self.feature_names:
#             del_keys.append(k)
#     for k in del_keys
#         d.pop(k)
        dt = { k:v for k,v in d.items() if k in self.feature_names } #a little bit faster
        return dt
    def get_feature_names(self):
        return self.feature_names

    
class Tettletale(TransformerMixin, object):
    """ For diagnostic: is noisy about the things it encounters  
    passes through all columns unaltered, a unitary operation;
    
    Parameters
    ----------
    fit_verbose : bool
        be verbose about what is encountered during `fit()` (default: True)
    transform_verbose : bool
        be verbose about what is encountered during `transform()` (default: False)
    """
    def __init__(self, fit_verbose=True, transform_verbose=False):
        self.fit_verbose=fit_verbose
        self.transform_verbose=transform_verbose
    def fit(self, X, y=None, **fit_params):
        if self.fit_verbose:
            print(X.info())
            if y is not None:
                print("pandas Series: ", y.name, y.shape)
        self.feature_names = X.columns.values
        return self
    def transform(self, X):
        if self.transform_verbose:
            print(X.info())
        return X
    def transform_dict(self, d):
        if self.transform_verbose:
            print(d.keys())
        return d
    def get_feature_names(self):
        return self.feature_names
