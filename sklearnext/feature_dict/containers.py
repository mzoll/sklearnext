'''
Created on Jun 12, 2018

@author: marcel.zoll
'''


import pandas as pd
import numpy as np
import copy


class LearningCollection(object):
    """ Holds the learnable information of a FeatureDict and the optionally the connected outcome;
    the internal base representation are pandas objects (indexed DataFrame and Series)
    
    Attributes
    ----------
    feature_names : list of str
        an ordered list, one entry per entry in _features_ parameter
    features : list of pandas.DataFrames
        an ordered list of pandas.DataFrames, each entry having the same primary length;
        if possible, each dataframe at each column should only contain numeric-type objects like bool, int, float
    outcome : pandas.Series or None
        a optional pandas.Series or pandas.DataFrame, of same length as the primary length of entries in features
        
    Example
    -------
    ```
    sample_size = 100
    feature_one = np.random.rand(sample_size, 1)
    feature_two = np.random.rand(sample_size, 2)
    outcome = np.random.sample([0,1], sample_size)
    lc = LearningCollection(['feature_one', 'feature_two'], [feature_one, feature_two], outcome)
    subsample = lc.sample(int(sample_size*0.2))
    test_split, train_split = lc.split(frac=0.2, shuffle=True)
    ```
    """
    def __init__(self):
        self.feature_names = None
        self.features = None
        self.outcome = None
    @staticmethod
    def fromFeatureDict(feature_dict, outcome=None):
        """ construct object from a FeatureDictionary, ie the output of the FeatureTransformFacility
        
        Parameters
        ----------
        feature_dict : dict(str: pandas.DataFrame)
        outcome : pandas.Series or None
        """
        lc = LearningCollection()
        lc.feature_names = list(feature_dict.keys())
        lc.features = list(feature_dict.values())
        lc.outcome = outcome
        return lc
    @property
    def nsamples(self):
        return len(self.features[0])
    @property
    def index(self):
        return self.features[0].index
    def sample(self, nmany=None, shuffle=False):
        """ make a subsample, with or without randomization in sampling;
        if used with default parameters this returns a (deep)copy of the object
        
        Parameters
        ----------
        nmany : int >0
            size of the output (sub)sample; if None is specified, it is substituted for self.nsamples (default: None)
        shuffle : bool
            shuffle before drawing samples, thus making a random sample (default: False)
            
        Returns
        -------
        LearningCollection containing a subsample of samples in this object
        """ 
        if nmany == self.nsamples:
            return self
        if nmany is None:
            nmany = self.nsamples
        
        if not shuffle:
            mask = np.random.permutation( [True]*nmany + [False]*(self.nsamples-nmany) )
            
            lc = LearningCollection()
            lc.feature_names = copy.copy(self.feature_names)
            lc.features = [ df[mask] for df in self.features ]
            if self.outcome is not None:
                lc.outcome = self.outcome[mask]
            return lc 
        else:
            idx_values = np.random.choice(self.features[0].index, size=nmany, replace=False)
            idx = pd.Index(idx_values)
            
            lc = LearningCollection()
            lc.feature_names = copy.copy(self.feature_names)
            lc.features = [ df.loc[idx,] for df in self.features ]
            if self.outcome is not None:
                lc.outcome = self.outcome.loc[idx]
            return lc 
    
    def split(self, frac=0.5, shuffle=False):
        """ part up the data into two portions returned as a tuple; this can be use to make perform split of the data
        into train and test data
        
        Parameters
        ----------
        frac : float in (0...1)
            the fraction of samples attributed to the left side of the split (default: 0.5)
        shuffle : bool
            shuffle the samples before split is performed (default: False)
        
        Returns
        -------
        tuple of LearningCollection of split-sample and LearningCollection of the rest
        """
        split_pos = int(len(self.index)*frac)
        if shuffle:
            idx_values = np.random.permutation(self.index)
            
            idx0 = pd.Index(idx_values[:split_pos])            
            c0 = LearningCollection()
            c0.feature_names = copy.copy(self.feature_names) 
            c0.features = [ f.loc[idx0] for f in self.features ]
            if self.outcome is not None:
                c0.outcome = self.outcome.loc[idx0]
            
            idx1 = pd.Index(idx_values[split_pos:])
            c1 = LearningCollection()
            c1.feature_names = copy.copy(self.feature_names) 
            c1.features = [ f.loc[idx1] for f in self.features ]
            if self.outcome is not None:
                c0.outcome = self.outcome.loc[idx1]
            
            return c0, c1 
        else:
            mask0 = [True]*split_pos + [False]*(self.nsamples-split_pos)
            
            lc0 = LearningCollection()
            lc0.feature_names = copy.copy(self.feature_names)
            lc0.features = [ df[mask0] for df in self.features ]
            if self.outcome is not None:
                lc0.outcome = self.outcome[mask0]
             
            lc1 = LearningCollection()
            lc1.feature_names = copy.copy(self.feature_names)
            lc1.features = [ df[~mask0] for df in self.features ]
            if self.outcome is not None:
                lc1.outcome = self.outcome[~mask0]
                 
            return lc0, lc1
    def to_compact(self):
        """ transform this object into a CompactLearningCollection, effectivly stripping away the pandas dependency
        """
        clc = CompactLearningCollection()
        clc.index = self.index
        clc.feature_names = copy.copy(self.feature_names)
        clc.features = [ df.values for df in self.features ]
        if self.outcome is None:
            clc.outcome = None
        else:
            clc.outcome = self.outcome.values
        return clc
        

class CompactLearningCollection(object):
    """ Holds the learnable information of a FeatureDict and the connected outcome, while reducing dedundant data;
    the internal base representation are numpy.arrays 
    
    Attributes
    ----------
    feature_names : [ str ] #vector of feature names
    features : [ np.array() ] #vector of feature content
    outcome : np.array() #vector of outcome
    index : np.array() #vector of index
    """
    def __init__(self):
        self.feature_names = None
        self.features = None
        self.outcome = None
        self.index = None
    @staticmethod
    def fromFeatureDict(feature_dict, outcome):
        """ construct object from a FeatureDictionary, ie the output of the FeatureTransformFacility
        
        Parameters
        ----------
        feature_dict : dict(str: pandas.DataFrame)
        outcome : pandas.Series or None
        """
        clc = CompactLearningCollection()
        clc.feature_names = list(feature_dict.keys())
        clc.features = [ df.values for df in feature_dict.values() ]
        clc.outcome = outcome.values #vector of outcome
        clc.index = outcome.index.values  #vector of index
        #TODO ---assertments: all indexes are the same
        return clc
    def sample(self, nmany=None, shuffle=False):
        """ make a subsample, with or without randomization in sampling;
        if used with default parameters this returns a (deep)copy of the object
        
        Parameters
        ----------
        nmany : int >0
            size of the output (sub)sample; if None is specified, it is substituted for self.nsamples (default: None)
        shuffle : bool
            shuffle before drawing samples, thus making a random sample (default: False)
            
        Returns
        -------
        LearningCollection containing a subsample of samples in this object
        """
        if nmany is None:
            nmany = len(self.index)
        i = np.random.choice(nmany, size=nmany, replace=False)
        if not shuffle:
            i.sort()
        
        c0 = CompactLearningCollection()
        c0.feature_names = copy.copy(self.feature_names) 
        c0.features = [ np.take(f, i) for f in self.features ]
        c0.outcome = np.take(self.outcome, i)
        c0.index = np.take(self.index, i)
        return c0
    
    def split(self, frac=0.5, shuffle=False):
        """ part up the data into two portions returned as a tuple; this can be use to make perform split of the data
        into train and test data
        
        Parameters
        ----------
        frac : float in (0...1)
            the fraction of samples attributed to the left side of the split (default: 0.5)
        shuffle : bool
            shuffle the samples before split is performed (default: False)
        
        Returns
        -------
        tuple of LearningCollection of split-sample and LearningCollection of the rest
        """
        split_pos = int(len(self.index)*frac)
        if shuffle:
            i = np.random.permutation(len(self.index))
            
            i0 = i[:split_pos]
            i1 = i[split_pos:]
            
            c0 = CompactLearningCollection()
            c0.feature_names = copy.copy(self.feature_names) 
            c0.features = [ np.take(f, i0) for f in self.features ]
            c0.outcome = np.take(self.outcome, i0)
            c0.index = np.take(self.index, i0)
            
            c1 = CompactLearningCollection()
            c1.feature_names = copy.copy(self.feature_names) 
            c1.features = [ np.take(f, i1) for f in self.features ]
            c1.outcome = np.take(self.outcome, i1)
            c1.index = np.take(self.index, i1)
            
            return c0, c1 
        else:
            c0 = CompactLearningCollection()
            c0.feature_names = copy.copy(self.feature_names) 
            c0.features = [ f[:split_pos] for f in self.features ]
            c0.outcome = self.outcome[:split_pos]
            c0.index = self.index[:split_pos]
            
            c1 = CompactLearningCollection()
            c1.feature_names = copy.copy(self.feature_names) 
            c1.features = [ f[split_pos:] for f in self.features ]
            c1.outcome = self.outcome[split_pos:]
            c1.index = self.index[split_pos:]
            
            return c0, c1
        def to_regular(self):
            """ transform this object into a LearningCollection, effectivly enriching the data
            """
            lc = LearningCollection()
            lc.feature_names = copy.copy(self.feature_names)
            lc.features = [ pd.DataFrame(data=f, index=self.index, columns=['_'.join(fn,str(i)) for i in range(f.shape[1])])
                            for fn,f in zip(self.feature_names, self.features) ] 
            if self.outcome is None:
                lc.outcome = None
            else:
                lc.outcome = pd.DataFrame(data=self.outcome, index=self.index, columns=['_'.join('outcome',str(i)) for i in range(self.outcome.shape[1])])
            return lc
                
                
                
            