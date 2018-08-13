'''
Created on Apr 3, 2018

@author: marcel.zoll
'''
import unittest

import numpy as np
import pandas as pd

from sklearnext.sklearning.dummy import DummyClassifier, DummyRegressor
from sklearnext.sklearning.oneprob_classifier import OneProbClassifierWrapper

class Test(unittest.TestCase):

    def testDummyClassifier(self):
        dc = DummyClassifier([0,1])
        
        dc.fit(np.array([0,1,2,3]).reshape((1,4)), np.array([0,0,1,1]))
        dc.predict(np.array([0,1,2,3]))
        dc.predict_proba(np.array([0,1,2,3]))
        
    def testDummyRegressor(self):
        dr = DummyRegressor(0.,1.)
        
        dr.fit(np.array([0,1,2,3]).reshape((1,4)), np.array([0.,0.,1.,1.]))
        dr.predict(np.array([0,1,2,3]))
        
    def OneProbClassifierWrapper(self):
        pass

        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()