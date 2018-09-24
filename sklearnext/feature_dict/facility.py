'''
Created on Jun 8, 2018

@author: marcel.zoll
'''

import numpy as np

from  joblib import Parallel, delayed

import logging
logger = logging.getLogger('FeatureTransformFacility')

class FeatureTransformFacility(object):
    """ A collection of Pipelines, that are nested into a dictionary. Effectively allows to directly fit and transform multiple pipelines
    in parallel, why keeping their results separated.
    This is really just a parallel processing facility for Pipelines that do not interact and can be for themselfes very deeply structured.
    NOTE: Do not confuse this with a sklearnext::Transformer, even when methods and interfaces look alike 
    
    Parameters
    ----------
    pipeline_dict : dict(str: TransformerPipe)
        a dictionary with a featurename as key and a TransformerPipe generating this feature from input
    is_fitted : bool
        is this FeatureTransformationFacility fitted, aka can the transform routines be called
    """
    def __init__(self, pipeline_dict = {}, is_fitted = False):
        self.pipeline_dict = pipeline_dict
        self._is_fitted = is_fitted
    def fit(self, X, y=None, n_jobs=1, **fit_params):
        """ fit the transformers in the transformer dictionary to the data
        
        Parameters
        ----------
        X : pandas.DataFrame
        n_jobs : int
        """
        transformers = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_transformer)(ik, trans, X, y)
            for ik, trans in self.pipeline_dict.items())
        self.pipeline_dict = dict( zip(self.pipeline_dict.keys(), transformers) )
        self._is_fitted = True
        return self
        
    def transform(self, X, n_jobs=1):
        """ transform a dataframe by the (already fitted) transformers
        
        Parameters
        ----------
        X : pandas.DataFrame
        n_jobs : int
        
        Returns
        -------
        dict(str:pandas.DataFrame) : the input_dict
        """
        Xt_vec = Parallel(n_jobs=n_jobs)(
            delayed(_transform_one)(ik, trans, X)
            for ik, trans in self.pipeline_dict.items())
        return dict( zip(self.pipeline_dict.keys(), Xt_vec) )
    
    def fit_transform(self, X, y=None, n_jobs=1, **fit_params):
        """ fit and transform a dataframe by the transformers
        
        Parameters
        ----------
        X : pandas.DataFrame
        n_jobs : int
        
        Returns
        -------
        dict(str:pandas.DataFrame) : the input_dict
        """
        result = Parallel(n_jobs)(
            delayed(_fit_transform_one)(ik, trans, X, y, **fit_params)
            for ik, trans in self.pipeline_dict.items())
        Xt_vec, transformers = zip(*result)
        
        self.pipeline_dict = dict( zip(self.pipeline_dict.keys(), transformers) )
        self._is_fitted = True
        
        return dict( zip(self.pipeline_dict.keys(), Xt_vec) )
        
    def transform_dict(self, d):
        """ transfrom a single dictionary
        
        Returns
        -------
        dictionary of featurenames and arrays
        """
        inputs_dict = {k:np.array(list(p.transform_dict(d.copy()).values())) for k,p in self.pipeline_dict.items() }
        inputs_dict = {k:v.reshape( (1,len(v)) ) for k,v in inputs_dict.items()}
        return inputs_dict

    def get_feature_keys(self):
        return list(self.pipeline_dict.keys())
    
    def derive_dimensions_dict(self):
        """ get the ouput shape for each generated feature """
        assert( self._is_fitted )
        dim_dict = {}    
        for k,p in self.pipeline_dict.items():
            dim_dict[k+'_len'] = len(p.get_feature_names())
            last_trans = p.steps[-1][1]
            if hasattr(last_trans, 'classes_'):
                dim_dict[k+'_depth'] = len(last_trans.classes_)
            if hasattr(last_trans, 'classes'):
                print(last_trans)
                dim_dict[k+'_depth'] = len(last_trans.classes)
        return dim_dict 
    
        
    
def _fit_one_transformer(ik, transformer, X, y):
    logger.debug("fit {}".format(ik))
    return transformer.fit(X, y)

def _transform_one(ik, transformer, X):
    logger.debug("transform {}".format(ik))
    res = transformer.transform(X)
    return res

def _fit_transform_one(ik, transformer, X, y, **fit_params):
    logger.debug("fit_transform {}".format(ik))
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    return res, transformer
    