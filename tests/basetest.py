"""
Created on Apr 3, 2018

@author: marcel.zoll
"""

import unittest
from sklearnext.base import assert_dfncol

import pandas as pd


class Test(unittest.TestCase):
    def testDfassert(self):
        df = pd.DataFrame({'A': [0,1]})
        assert_dfncol(df, 1)
        
        df = pd.DataFrame({'A': [0,1], 'B':[0,1]})
        assert_dfncol(df, 2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
