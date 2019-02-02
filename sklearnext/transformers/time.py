'''
Created on Nov 7, 2017

@author: marcel.zoll

(Date)Time Transformers
'''

import pandas as pd
import numpy as np
import datetime as dt

from ..base import assert_dfncol

from sklearn.base import TransformerMixin 

import calendar


class DeltaSecTransformer(TransformerMixin, object):
    """ calculate the difference in seconds between two input columns, which need be of format datetime
    
    Parameters
    ----------
    fast_path : bool 
        calculate the value by a faster way. This requires both columns to have only
        valid (non-null) input (default: False).
    fill_na : tuple of floats (shape=2)
        fill in these default values, if left side column respective the right side column value is missing.
        If both are missing the left side default value takes presidence (default: (na, na))
        
    Example
    -------
    df = pandas.DataFrame({'A': [datetime.datetime(2018,1,1,0,0)], 'B': [datetime.datetime(2018,1,1,0,1)]})
    DeltaSecTransformer().fit_transform(df)
    #>>> pandas.DataFrame({'A_B_diffsec': [60.]})
    """
    def __init__(self,
            fast_path = False,
            fill_na = (np.nan, np.nan)):
        self.fast_path = fast_path
        self.fill_na = fill_na
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 2)
        self.incolumns = X.columns
        if len(self.incolumns) != 2:
            raise Exception('Expected to calculate the difference of two datetime columns')
        self.feature_names_ = ['_'.join(X.columns)+'_diffsec']
        return self
    def transform(self, X):
        assert_dfncol(X, 2)
        
        t1 = X[self.incolumns[0]]
        t2 = X[self.incolumns[1]]
        if not t1.isnull().values.any() and not t2.isnull().values.any():
            self.fast_path = True

        if self.fast_path:
            dtime = t2 - t1 
            dtime = dtime.apply(lambda v:v.total_seconds())
            return pd.DataFrame(dtime, columns= self.feature_names_)
        else: #execute line by line; check input
            def xthelper(row):
                t1v = row[self.incolumns[0]]
                t2v = row[self.incolumns[1]]
                if t2v is None:
                    return self.fill_na[1]
                elif t1v is None:
                    return self.fill_na[0]
                
                return (t2v-t1v).total_seconds()
            Xt = X.apply(xthelper, axis=1)
            return pd.DataFrame(Xt, columns= self.feature_names_)
    def transform_dict(self, d):
        t1 = d.pop(self.incolumns[0])
        t2 = d.pop(self.incolumns[1])
        d[self.feature_names_[0]] = (t2 - t1).total_seconds()
        return d            
    def get_feature_names(self):
        return self.feature_names_
    
class HourWeekdayDayMonthYearTransformer(TransformerMixin):
    """ transform a single column with a datetime object into its principle time-components.
    
    Components : 
    hour(float), weekday(uint), day(uint), month(uint), year(uint)

    Parameters
    ----------
    hour : bool
        should the hour of the day be extracted as an float (default: True)
    weekday : bool
        should the weekday be extracted as an integer label (default: True)
    day : bool
        should the day of the month bve extracted as an absolute label (default: True)
    month : bool
        should the month of the year be extracted as an absolute label (default: True)
    year : bool
        should the year be extracted as an absolute label (default: True)

    Example
    -------
    df = pandas.DataFrame({'A': [datetime.datetime(2018,1,2,3,4,5)],]})
    HourWeekdayDayMonthYearTransformer().fit_transform(df)
    #>>> pandas.DataFrame({'A_hour': [3.068056], 'A_weekday': [0], 'A_day': [2], 'A_month': [1], 'A_year': [2018] })
    """ 
    def __init__(self, hour=True, weekday=True, day=True, month=True, year=True):
        self.ext_hour = hour
        self.ext_weekday = weekday
        self.ext_day = day
        self.ext_month = month
        self.ext_year = year
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        self.incols = X.columns
        self.varname = self.incols[0]
        self.feature_names_ = []
        if self.ext_hour:
            self.feature_names_.append(self.incols[0]+'_'+'hour')
        if self.ext_weekday:
            self.feature_names_.append(self.incols[0]+'_'+'weekday')
        if self.ext_day:
            self.feature_names_.append(self.incols[0]+'_'+'day')
        if self.ext_month:
            self.feature_names_.append(self.incols[0]+'_'+'month')
        if self.ext_year:
            self.feature_names_.append(self.incols[0]+'_'+'year')
        return self
    def transform(self, X):
        if self.ext_hour and self.ext_weekday and self.ext_day and self.ext_month and self.ext_year:
            def iterhelper(t):
                return pd.Series([t.hour + t.minute/60., int(t.weekday()), int(t.day), int(t.month), int(t.year)])
            Xt = X[self.varname].apply(iterhelper)
            Xt.columns = self.feature_names_
            Xt[self.varname+'_weekday'] = Xt[self.varname+'_weekday'].astype('uint8')
            Xt[self.varname+'_day'] = Xt[self.varname+'_day'].astype('uint8')
            Xt[self.varname+'_month'] = Xt[self.varname+'_month'].astype('uint8')
            Xt[self.varname+'_year'] = Xt[self.varname+'_year'].astype('uint8')
            return Xt
        else:
            dfs = []
            if self.ext_hour:
                s = X[self.varname].apply(lambda t: t.hour + t.minute /60.)
                s.name = self.varname + '_hour'
                dfs.append(s)
            if self.ext_weekday:
                s = X[self.varname].apply(lambda t: t.weekday)
                s.name = self.varname + '_weekday'
                s = s.astype('uint8')
                dfs.append(s)
            if self.ext_day:
                s = X[self.varname].apply(lambda t: t.day)
                s.name = self.varname + '_day'
                s = s.astype('uint8')
                dfs.append(s)
            if self.ext_month:
                s = X[self.varname].apply(lambda t: t.month)
                s.name = self.varname + '_month'
                s = s.astype('uint8')
                dfs.append(s)
            if self.ext_year:
                s = X[self.varname].apply(lambda t: t.year)  )
                s.name = self.varname + '_year'
                s = s.astype('uint8')
                dfs.append(s)
            return pd.concat(dfs, axis=1)

    def transform_dict(self, d):
        t = d.pop(self.varname)
        if self.ext_hour:
            d[self.varname+'_hour'] = int(t.hour) + t.minute/60.
        if self.ext_weekday:
            d[self.varname + '_weekday'] = int(t.weekday)
        if self.ext_day:
            d[self.varname+'_day'] = int(t.day)
        if self.ext_month:
            d[self.varname+'_month'] = int(t.month)
        if self.ext_year:
            d[self.varname+'_year'] = int(t.year)
        return d
        
    def get_feature_names(self):
        return self.feature_names_


class HourExtractor(TransformerMixin, object):
    """ Transformer giving the hours in this day
    
    Parameters
    ----------
    discrete : bool
        use descrete (non-fractional) values instead of continious floats (default: False)
    normalize : bool
        the output will be expressed in the interval [0..1]
        
    Examples
    --------
    df = pandas.DataFrame({'A': [datetime.datetime(2018,1,2,3,4,5)],]})
    HourExtractor().fit_transform(df)
    #>>>pandas.DataFrame({'A_hour': [3.068056]})
    """
    @staticmethod
    def dt_hour_disc(dtval):
            return dtval.hour
    @staticmethod
    def dt_hour_cont(dtval):
        return dtval.hour + dtval.minute/60. + dtval.second/3600.
      
    def __init__(self, discrete= False, normalize= False):
        self.discrete = discrete
        self.normalize = normalize
        
        if self.discrete:
            self._transfunc = self.dt_hour_disc
        else:
            self._transfunc = self.dt_hour_cont
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type 
        self.incols= list(X.columns)
        self.feature_names_ = [self.incols[0]+'_hour']
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        Xt = pd.DataFrame( X.iloc[:,0].apply(self._transfunc) )
        if self.normalize:
            Xt = Xt.apply(lambda v: v/24., axis=1)
        Xt.columns = self.feature_names_    
        return Xt    
    def transform_dict(self, d):    
        dtval = d.pop(self.incols[0])
        t = self.transfunc(dtval)
        if self.normalize:
            t /= 24.
        d.update( {self.feature_names_[0]: t} )
    def get_feature_names_(self):
        return self.feature_names_
    

class WeekdayExtractor(TransformerMixin, object):
    """ Transformer giving the days from the Monday this week and the next week
    
    Parameters
    ----------
    discrete : bool
        use discrete (non-fractional) values instead of continuous floats (default: False)
    normalize : bool
        the output will be expressed in the interval [0..1]
    """
    @staticmethod
    def dt_weekday_disc(dtval):
        return dtval.weekday()
    @staticmethod
    def dt_weekday_cont(dtval):
        return dtval.weekday() + (dtval.hour + dtval.minute/60. + dtval.second/3600.) / 24.
    
    def __init__(self, discrete= False, normalize=False, week_starts_so=False):
        self.discrete = discrete
        self.normalize = normalize
        self.week_starts_so = week_starts_so
        
        if self.discrete:
            self._transfunc = self.dt_weekday_disc
        else:
            self._transfunc = self.dt_weekday_cont
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        self.incols= list(X.columns)
        self.feature_names_ = [self.incols[0]+'_weekday']
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        Xt = pd.DataFrame( X.iloc[:,0].apply(self._transfunc) )
        if self.week_starts_so:
            Xt = pd.DataFrame(Xt.iloc[:,0].apply(lambda v: (v+6)%7))
        if self.normalize:
            Xt = pd.DataFrame(Xt.iloc[:,0].apply(lambda v: v/7.))
        Xt.columns = self.feature_names_    
        return Xt    
    def transform_dict(self, d):
        dtval = d.pop(self.incols[0])
        t = self.transfunc(dtval)
        if self.normalize:
            t /= 7.
        d.update( {self.feature_names_[0]: t} )
    def get_feature_names_(self):
        return self.feature_names_


class MonthdayExtractor(TransformerMixin, object):
    """ Transformer giving the days from the Monday this week and the next week
    
    Note: This Transformer fixes the number of days in each month to 31 and deducts 1 , so effectively pictures values on the interval (0...30).
        Thus this Transformer is giving an absolute on the monthday, not a relative!.
        (Explanation: Monthday is a bit special in the sense that the days in each month are varying and so do not reflect 
        how far along into a month the time in question is, or how much time/days are left to the beginning of the next month.
        For this purpose use the MonthfracExtractor instead!)
    
    Parameters
    ----------
    discrete : bool
        use descrete (non-fractional) values instead of continious floats (default: False)
    normalize : bool
        the output will be expressed in the interval [0..1]
    """
    @staticmethod
    def dt_monthday_disc(dtval):
        return dtval.day - 1.
    @staticmethod
    def dt_monthday_cont(dtval):
        return dtval.day + (dtval.hour + dtval.minute/60. + dtval.second/3600.) / 24. - 1.
    
    def __init__(self, discrete = False, normalize=False):
        self.discrete = discrete
        self.normalize = normalize
        
        if self.discrete:
            self._transfunc = self.dt_monthday_disc
        else:
            self._transfunc = self.dt_monthday_cont
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        self.incols= list(X.columns)
        self.feature_names_ = [self.incols[0]+'_monthday']
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        Xt = pd.DataFrame( X.iloc[:,0].apply(self._transfunc) )
        if self.normalize:
            Xt = Xt.apply(lambda v: v/7., axis=1)
        Xt.columns = self.feature_names_    
        return Xt    
    def transform_dict(self, d):    
        dtval = d.pop(self.incols[0])
        t = self._transfunc(dtval)
        if self.normalize:
            t /= 31.
        d.update( {self.feature_names_[0]: t} )
    def get_feature_names_(self):
        return self.feature_names_
    

class MonthfracExtractor(TransformerMixin, object):
    """ Transformer giving the days from the Monday this week and the next week
    
    Like MonthdayExtractor, but taking the actual days in each month into account and with 
    the option 'normalize' fixed to True, so effectively ONLY picturing values onto the interval (0...1)
    
    Parameters
    ----------
    discrete : bool
        use descrete (non-fractional) values instead of continious floats (default: False)
    """
    @staticmethod
    def dt_monthfrac_disc(dtval):
        daysthismonth = calendar.monthrange(dtval.year, dtval.month)[1]
        return (dtval.day - 1.) / daysthismonth
    @staticmethod
    def dt_monthfrac_cont(dtval):
        daysthismonth = calendar.monthrange(dtval.year, dtval.month)[1]
        return (dtval.day + (dtval.hour + dtval.minute/60. + dtval.second/3600.) / 24. - 1.) / daysthismonth
    
    def __init__(self, discrete = False):
        self.discrete = discrete
        
        if self.discrete:
            self._transfunc = self.dt_monthfrac_disc
        else:
            self._transfunc = self.dt_monthfrac_cont
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        self.incols= list(X.columns)
        self.feature_names_ = [self.incols[0]+'_monthfrac']
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        Xt = pd.DataFrame( X.iloc[:,0].apply(self._transfunc) )
        Xt.columns = self.feature_names_    
        return Xt    
    def transform_dict(self, d):    
        dtval = d.pop(self.incols[0])
        t = self._transfunc(dtval)
        d.update( {self.feature_names_[0]: t} )
    def get_feature_names_(self):
        return self.feature_names_   
    
    
    
class YeardayExtractor(TransformerMixin, object):
    """ Transformer giving the days since the since the beginning of the year
    
    Note: This Transformer fixes the number of days in each year to 366 and deducts 1, so it effectively pictures values on the interval (0...365).
        Thus this Transformer is giving an absolute values, not a relative!.
        (Explanation: Yearday is a bit special in the sense that the days in each year are varying (leapyears) and so do not reflect 
        how far along into a year the time in question is, or how muchg time/days are left to the beginning of the next year.
        For this purpose use the YearfracExtractor instead!)
    
    Parameters
    ----------
    discrete : bool
        use descrete (non-fractional) values instead of continious floats (default: False)
    normalize : bool
        the output will be expressed in the interval [0..1]
    """
    @staticmethod
    def dt_yearday_disc(dtval):
        days_to_now = 0
        for i in range(dtval.month-1):
            days_to_now += calendar.monthrange(i+1)
        return days_to_now + dtval.day -1
    @staticmethod
    def dt_yearhday_cont(dtval):
        days_to_now = 0
        for i in range(dtval.month-1):
            days_to_now += calendar.monthrange(i+1)
        return days_to_now + dtval.day -1. + (dtval.hour + dtval.minute/60. + dtval.second/3600.) / 24.
        
    def __init__(self, discrete = False, normalize=False):
        self.discrete = discrete
        self.normalize = normalize
        
        if self.discrete:
            self._transfunc = self.dt_yearday_disc
        else:
            self._transfunc = self.dt_yearday_cont
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        self.incols= list(X.columns)
        self.feature_names_ = [self.incols[0]+'_yearday']
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        Xt = pd.DataFrame( X.iloc[:,0].apply(self._transfunc) )
        if self.normalize:
            Xt = Xt.apply(lambda v: v/366., axis=1)
        Xt.columns = self.feature_names_    
        return Xt    
    def transform_dict(self, d):    
        dtval = d.pop(self.incols[0])
        t = self._transfunc(dtval)
        if self.normalize:
            t /= 366.
        d.update( {self.feature_names_[0]: t} )
    def get_feature_names_(self):
        return self.feature_names_  
    
    
class YearfracExtractor(TransformerMixin, object):
    """ Transformer giving the fractional time into this year 
    
    Like YeardayExtractor, but taking the actual days in each month into account and with 
    the option 'normalize' fixed to True, so effectively ONLY picturing values onto the interval (0...1)
    
    Parameters
    ----------
    discrete : bool
        use descrete (non-fractional) values instead of continious floats (default: False)
    """
    @staticmethod
    def dt_yearfrac_disc(dtval):
        days_to_now = 0
        for i in range(dtval.month-1):
            days_to_now += calendar.monthrange(i+1)
        days_in_year= 366 if calendar.isleap(dtval.year) else 365
        return (days_to_now + dtval.day -1) / days_in_year
    @staticmethod
    def dt_yearhfrac_cont(dtval):
        days_to_now = 0
        for i in range(dtval.month-1):
            days_to_now += calendar.monthrange(i+1)
        days_in_year= 366 if calendar.isleap(dtval.year) else 365
        return (days_to_now + dtval.day -1. + (dtval.hour + dtval.minute/60. + dtval.second/3600.) / 24. ) / days_in_year
    
    def __init__(self, discrete = False):
        self.discrete = discrete
        
        if self.discrete:
            self._transfunc = self.dt_yearfrac_disc
        else:
            self._transfunc = self.dt_yearfrac_cont
    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        self.incols= list(X.columns)
        self.feature_names_ = [self.incols[0]+'_yearfrac']
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        #assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        Xt = pd.DataFrame( X.iloc[:,0].apply(self._transfunc) )
        Xt.columns = self.feature_names_    
        return Xt    
    def transform_dict(self, d):    
        dtval = d.pop(self.incols[0])
        t = self._transfunc(dtval)
        d.update( {self.feature_names_[0]: t} )
    def get_feature_names_(self):
        return self.feature_names_


class TimeMinMaxTransformer(TransformerMixin, object):
    """ inspect for the lowest and highest value in the time column and scale to the interval [0,1]
    """
    def __init__(self):
        self._min_time = None
        self._max_time = None

    def _scalefkt(self, dt_val):
        val = (dt_val - self._min_time).total_seconds() / self._time_dt_sec
        return np.clip(val, 0., 1.)

    def fit(self, X, y=None, **fit_params):
        assert_dfncol(X, 1)
        # assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        self.incols = list(X.columns)
        self.feature_names_ = [self.incols[0] + '_minmaxscaled']
        self._min_time = np.min(X.iloc[:,0])
        self._max_time = np.min(X.iloc[:, 0])
        self._time_dt_sec = (self._max_time - self._min_time).total_seconds()
        return self
    def transform(self, X):
        assert_dfncol(X, 1)
        # assert( isinstance(X.iloc[:,0].dtype, dt.datetime) ) #FIXME simple check for type
        Xt = pd.DataFrame( X.iloc[:,0].apply( self._scalefkt )
        Xt.columns = self.feature_names_
        return Xt
    def transform_dict(self, d):
        dtval = d.pop(self.incols[0])
        t = self._transfunc(dtval)
        d.update({self.feature_names_[0]: t})
    def get_feature_names_(self):
        return self.feature_names_
