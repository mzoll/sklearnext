
import sys, copy
import itertools
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin


class ColumnsNone(TransformerMixin, object):
    """ passes through nothing """
    def fit(self, X, y=None, **fit_params):
        self.incols = X.columns
        self.feature_names = []
        return self
    def transform(self, X):
        return pd.DataFrame(index= X.index)
    def transform_dict(self, d):
        return {}
    def get_feature_names(self):
        return self.feature_names


class FeatureEliminator(TransformerMixin, object):
    """ eliminate all collumns which have not diversive values """
    def fit(self, X, y=None, **fit_params):
        self.incols = X.columns

        def checkEqualIvo(lst):
            return not lst or lst.count(lst[0]) == len(lst)
        keep_col_idx = []
        for i in range( len(X.columns)):
            if not checkEqualIvo(X.iloc[:, i]):
                keep_col_idx.append(i)
            """
            self._stride_mult = 10
            self._max_stride_len = 1000
            
            # fetch first value
            val = X.iloc[0, i]

            test_range_len = 1
            test_idx_start = 1
            test_idx_end = 1
            while test_idx_start < len(X):
                test_range = X.iloc[ test_idx_start:test_idx_end, i]
                if any( test_range != val ):
                    self._keep_col_idx
                    break
                test_range_len = min(test_range_len*self._stride_mult, self._max_stride_len)
                test_idx_start = test_idx_end + 1
                test_inx_end = min(test_inx_end + test_range_len + 1, len(X))
            """

        self.feature_names = [ ic for cidx, ic in enumerate(self.incols) if cidx in keep_col_idx ]
        return self

    def transform(self, X):
        return X(self.feature_names)

    def transform_dict(self, d):
        return {fn:d[fn] for fn in self.feature_names}

    def get_feature_names(self):
        return self.feature_names
