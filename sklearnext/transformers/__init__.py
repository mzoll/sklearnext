'''
Created on Jan 19, 2018

@author: marcel.zoll
'''

""" 
sklearnext::Transformers work in a similar way as the sklearn::Transformers, however they are rigged to take an DataFrame as input
and convolute it into another DataFrame.

They need to support the calls to

Methods
-------
fit() :
transform() :
fit_transform() :
transform_dict() :
get_feature_names() : returns the names of columns in the output dataFrame
get_infeature_names() : returns the names of the to be concoluted columns in the input dataFrame
""" 

from .arythmetic import *
from .categorical import *
from .general import *
from .lambda_func import *
from .sequence_vector import *
from .time import *
from .cyclic import *
