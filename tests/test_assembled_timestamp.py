import unittest
from sklearnext.transformers.assembled.timestamp import TimestampTransformer
import pandas as pd
import datetime as dt
import numpy as np


class TimestampTransformer_Test(unittest.TestCase):
    def test_transformer(self):
        d = dt.datetime.utcnow()

        dth_vec = np.linspace(0, 24*31*2, 24*31*2+1)

        ds = [d + dt.timedelta(hours=dth) for dth in dth_vec]

        X = pd.DataFrame({'ts': ds})

        t = TimestampTransformer('ts')
        t.fit(X)
        t.transform(X)


if __name__ == '__main__':
    unittest.main()
