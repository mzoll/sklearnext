"""
Assemble a pipeline for feature extraction from time-like features
"""

from sklearnext.assembly import *
from sklearnext.transformers.time import *
from sklearnext.transformers.cyclic import CyclicSineCosineTransformer
from sklearnext.transformers.categorical import OneHotTransformer


def TimestampTransformer(colname):
    """
    Assemble a pipeline for feature extraction from time-like features

    :param colname: str
        name of column for the timestamp
    :return: TransformerPipe
    """
    tf1 = TransformerPipe([
        ('starttimeExtr', ColumnsSelect(colname)),
        ('native_time', FeatureUnion([
            # do things for the hour of the day as an periodic
            ('', TransformerPipe([
                ('', WeekdayExtractor(discrete=False, normalize=True)),
                ('sincosTrans', CyclicSineCosineTransformer(periodicity=1, pure_positive=False))
            ])),

            # do things for the weekday as an absolute label and periodic
            ('', FeatureUnion([
                ('', TransformerPipe([
                    ('', WeekdayExtractor(discrete=True)),
                    ('labelEnc', OneHotTransformer())
                ])),
                ('', TransformerPipe([
                    ('', WeekdayExtractor(discrete=False, normalize=True)),
                    ('sincosTrans', CyclicSineCosineTransformer(periodicity=1, pure_positive=False))
                ]))
            ])),

            # do things for the progress of the month
            ('dayofmonth_relative', TransformerPipe([
                ('monthday', MonthfracExtractor()),
                ('sincosTrans', CyclicSineCosineTransformer(periodicity=1, pure_positive=False))  # periodic
            ])),

            # do things for the day of the month as an absolute label
            ('dayofmonth_absolute', TransformerPipe([
                ('monthday', MonthdayExtractor(discrete=True, normalize=false)),
                ('labelEnc', OneHotTransformer())  # label
            ])),

            # do things for the fraction of the year
            ('dayofyear_relative', TransformerPipe([
                ('', YeardayFracExtractor(discrete=False, normalize=1)),
                ('', FeatureUnion([

                    ('sincosTrans', CyclicSineCosineTransformer(periodicity=1, pure_positive=False)) #periodic
                ])),

            ])),

            # do things for the year as an absolute label
            ('dayofyear_relative', TransformerPipe([
                ('', HourWeekdayDayMonthYearTransformer(False, False, False, False, True)),
                ('labelEnc', OneHotTransformer())
            ])),

            # take the absolute range of the timestamp and picture it on [0, 1]
            ('', TimeMinMaxTransformer())



        ]))
    ])

    return tf1