'''
Created on Apr 3, 2018

@author: marcel.zoll
'''
import unittest
from sklearnext.base import assert_dfncol


class Test(unittest.TestCase):
    def testDfassert(self):
        import pandas as pd
        
        df = pd.DataFrame({'A': [0,1]})
        assert(assert_dfncol(df, 1))
        
        df = pd.DataFrame({'A': [0,1], 'B':[0,1]})
        assert(assert_dfncol(df, 2))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()