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
    tp = TransformerPipe([
        ('timestamp_select', ColumnsSelect(colname)),
        ('native_time', FeatureUnion([
            # do things for the hour of the day as an periodic
            ('hour_of_day', TransformerPipe([
                ('extract', WeekdayExtractor(discrete=False, normalize=True)),
                ('sincosTrans', CyclicSineCosineTransformer(periodicity=1, pure_positive=False))
            ])),

            # do things for the weekday as an absolute label and periodic
            ('week', FeatureUnion([
                ('weekday', TransformerPipe([
                    ('extract', WeekdayExtractor(discrete=True)),
                    ('labelEnc', OneHotTransformer())
                ])),
                ('day_of_week', TransformerPipe([
                    ('extract', WeekdayExtractor(discrete=False, normalize=True)),
                    ('sincosTrans', CyclicSineCosineTransformer(periodicity=1, pure_positive=False))
                ]))
            ])),

            # do things for the progress of the month
            ('day_in_month', TransformerPipe([
                ('extract', MonthfracExtractor()),
                ('sincosTrans', CyclicSineCosineTransformer(periodicity=1, pure_positive=False))  # periodic
            ])),

            # do things for the day of the month as an absolute label
            ('monthday', TransformerPipe([
                ('extract', MonthdayExtractor(discrete=True, normalize=False)),
                ('labelEnc', OneHotTransformer())  # label
            ])),

            # do things for the fraction of the year
            ('time_in_year', TransformerPipe([
                ('extract', YearfracExtractor(discrete=False)),
                ('sincosTrans', CyclicSineCosineTransformer(periodicity=1, pure_positive=False)) #periodic
            ])),

            # do things for the year as an absolute label
            ('year', TransformerPipe([
                ('extract', HourWeekdayDayMonthYearTransformer(False, False, False, False, True)),
                ('labelEnc', OneHotTransformer())
            ])),

            # take the absolute range of the timestamp and picture it on [0, 1]
            ('abs_time', TimeMinMaxTransformer())
        ]))
    ])

    return tp
