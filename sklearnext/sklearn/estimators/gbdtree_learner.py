'''
Created on Oct 20, 2017

@author: marcel.zoll
'''

import sys
import copy

from sklearn.base import RegressorMixin, ClassifierMixin, MetaEstimatorMixin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from sklearn.ensemble import GradientBoostingClassifier


import logging
logger = logging.getLogger('GrowingTreeEstimator')

class _GrowingTreeEstimator(MetaEstimatorMixin, object):
    """ Base for growing tree estimators; grows additional trees as long as the score improves
    
    Parameters
    ----------
    treeEstimatorClass : class (Estimator)
        An ensamble estimator based on decision-trees
    est_params : dict
        Parameters on the estimator
    ntrees_start : int
        number of trees to initially construct
    ntress_increment : int
        number of trees to grow the estimator with each iteration
    scoring : callable
        Callable object that returns a scalar score; greater is better.   
    min_score_improvement : float
        minimal required difference in score, so that an training with additional trees is considered an improvement (default: 0.)
    nrounds_noimp : int
        number of rounds without an improvement to end search iteration;
        then take the estimator with the last best estimator (default: 2)
    nrounds_stop : int
        number of maximal iterations (thus growing maximal ntrees_increment*nrounds_stop additional trees) until ending search iteration;
        then take the estimator with the last best estimator (default: sys.maxsize (nostop))
    cv_test_frac : float ( in interval (0..1) )
        the fraction (ratio) of the test sample size for the internal cross-validation (default: 0.2) 
    cv_n_splits : int
        number of cross_validation rounds to generate scoring in each iteration round (default: 3):
        
    Attributes
    ----------
    est : Estimator object
        the estimator which is trained up
    """
    def __init__(self,
            treeEstimatorClass, 
            est_params,
            ntrees_start,
            ntress_increment,
            score_func = None,                    
            min_score_improvement = 0.,
            nrounds_noimp = 2,
            nrounds_stop = sys.maxsize,
            cv_test_frac = 0.2, 
            cv_n_splits = 3,
            warm_start = False):
        for attr in ['fit', 'predict', 'score']:
            if not hasattr(treeEstimatorClass, attr):
                raise TypeError("passed attribute does not seem to be a proper estimator")
        self._treeEstimatorClass = treeEstimatorClass
        self._est_params = est_params
        self._ntrees_start = ntrees_start
        self._ntress_increment = ntress_increment
        self._score_func = score_func
        self._min_score_improvement = min_score_improvement
        self._nrounds_noimp = nrounds_noimp
        self._nrounds_stop = nrounds_stop
        self._cv_test_frac = cv_test_frac
        self._cv_n_splits = cv_n_splits 
        self._warm_start = warm_start
        self.est = self._treeEstimatorClass(n_estimators = self._ntrees_start, **self._est_params)
        
    def fit(self, X, y, **fit_params):
        #reset the estimator to zero state
        if not self._warm_start:
            logger.info("resetting estimator to null-state")
            self.est = self._treeEstimatorClass(n_estimators = self._ntrees_start, **self._est_params)

        logger.debug('call from %s' % (str(self)))
        n_est = self._ntrees_start
        
        cv = ShuffleSplit(n_splits=self._cv_n_splits, test_size=self._cv_test_frac, random_state=0)
        
        meter = cross_val_score(self.est, X, y, scoring=self._score_func, cv=cv).mean()
        
        logger.debug("round n_est meter")
        logger.debug("%d %d %f" % (0, n_est, meter))
        
        est = copy.deepcopy(self.est)
        
        meter_prev_max = meter
        n_est_best = n_est
        niter_best = 0
        niter = 0
        while niter < self._nrounds_stop and self._ntress_increment > 0:
            niter += 1
            n_est += self._ntress_increment
            est.set_params(n_estimators=n_est, warm_start=True)
            
            meter = cross_val_score(est, X, y, scoring=self._score_func, cv=cv).mean()
            
            logger.debug("%d %d %f" % (niter, n_est, meter))

            if meter - meter_prev_max <= self._min_score_improvement :
                #print("score diff %e" % (meter - meter_prev_max))
                #no improvement
                if niter >= niter_best + self._nrounds_noimp :
                    logger.debug('no improvement for %d rounds' % (self._nrounds_noimp))
                    break
                continue
            
            #improvement
            self.est = copy.deepcopy(est)
            meter_prev_max = meter
            n_est_best = n_est
            niter_best = niter
            
        if niter == self._nrounds_stop:
            logger.debug('Stopped after %d iterations' % (self._nrounds_stop))

        self.est.set_params(warm_start=True) # n_estimators=n_est_best, 
        #self.est.estimators_ = self.est.estimators_[:n_est_best] 
        self.est.fit(X, y, **fit_params)
        logger.info("With %d estimators:: TrainScore(cv-test): %f , TrainScore(all): %f" % (n_est_best, meter_prev_max, self._score_func(self.est, X, y)))
        return self.est
    def predict(self, X):
        return self.est.predict(X)
    def predict_proba(self, X):
        return self.est.predict_proba(X)
    #def transform(self, X):
        #return self.predict(X)
    @property
    def feature_importances_(self):
        return self.est.feature_importances_


from sklearn.ensemble import GradientBoostingRegressor
class GrowingGBRegressor(_GrowingTreeEstimator, RegressorMixin):
    """ A GradientBoostingRegressor that adds more trees as long as the score improves """
    def __init__(self,
            ntrees_start = 100,
            ntress_increment = 10,
            scoring = None,
            min_score_improvement = 0.,
            nrounds_noimp = 2, #number of rounds no improvement in scores is seen
            nrounds_stop = 20,
            est_params = {
                'learning_rate':0.1,
                'max_depth':3,
                'random_state':0,
                'loss':'ls'            
            },
            cv_test_frac = 0.2,
            cv_n_splits = 2,
            warm_start= False):
        score_func = scoring_to_score_func(scoring, 'r2_score')
        
        _GrowingTreeEstimator.__init__(self, 
            GradientBoostingRegressor,
            ntrees_start = ntrees_start,
            ntress_increment = ntress_increment,
            score_func = score_func,
            min_score_improvement = min_score_improvement,
            nrounds_noimp = nrounds_noimp,
            nrounds_stop = nrounds_stop,
            est_params = est_params,
            cv_test_frac = cv_test_frac,
            cv_n_splits = cv_n_splits,
            warm_start = warm_start)



class GrowingGBClassifier(_GrowingTreeEstimator, ClassifierMixin):
    """ A GradientBoostingClassifier that adds more trees as long as the score improves """
    def __init__(self,
            ntrees_start = 100,
            ntress_increment = 10,
            scoring = None,
            min_score_improvement = 0.,
            nrounds_noimp = 2, #number of rounds no improvement in scores is seen
            nrounds_stop = 20,
            est_params = {
                'learning_rate':0.1,
                'max_depth':3,
                'random_state':0,
                'loss':'deviance'            
            },
            cv_test_frac = 0.2,
            cv_n_splits = 2,
            warm_start = False):
        score_func = scoring_to_score_func(scoring, 'log_loss')
        
        _GrowingTreeEstimator.__init__(self, 
            GradientBoostingClassifier,
            ntrees_start = ntrees_start,
            ntress_increment = ntress_increment,
            score_func = score_func,
            min_score_improvement = min_score_improvement,
            nrounds_noimp = nrounds_noimp,
            nrounds_stop = nrounds_stop,
            est_params = est_params,
            cv_test_frac = cv_test_frac,
            cv_n_splits = cv_n_splits,
            warm_start = warm_start)
    @property
    def classes_(self):
        return self.est.classes_


class GrowingGBBinaryProbClassifier(_GrowingTreeEstimator, ClassifierMixin):
    """ A GradientBoostingRegressor for binary classification tasks that adds more trees as long as the score improves """
    def __init__(self,
            ntrees_start = 100,
            ntress_increment = 10,
            scoring = None,
            min_score_improvement = 0.,
            nrounds_noimp = 2, #number of rounds no improvement in scores is seen
            nrounds_stop = 20,
            est_params = {
                'learning_rate':0.1,
                'max_depth':3,
                'random_state':0,
                'loss':'deviance'            
            },
            cv_test_frac = 0.2,
            cv_n_splits = 2,
            warm_start = False):
        score_func = scoring_to_score_func(scoring, 'log_loss')
        
        _GrowingTreeEstimator.__init__(self, 
            GradientBoostingClassifier,
            ntrees_start = ntrees_start,
            ntress_increment = ntress_increment,
            score_func = score_func,
            min_score_improvement = min_score_improvement,
            nrounds_noimp = nrounds_noimp,
            nrounds_stop = nrounds_stop,
            est_params = est_params,
            cv_test_frac = cv_test_frac,
            cv_n_splits = cv_n_splits,
            warm_start = warm_start)
    def fit(self, X, y, **fit_params):
        _GrowingTreeEstimator.fit(self, X, y, **fit_params)
        return self
    def predict(self, X):
        ''' predict only the positive component '''
        return self.est.predict_proba(X)[:,1]
    @property
    def classes_(self):
        return self.est.classes_
    

#=====================================
# Scoring auxilary
#=====================================

from sklearn.metrics import make_scorer
from sklearn.utils import column_or_1d
from sklearn import metrics

def scoring_to_score_func(scoring, default):
    if callable(scoring):
        return scoring
    if scoring is None:
        if callable(default):
            return default
        scoring = default
    
    if scoring=='binary_roc_auc_score':
        return make_binary_scorer(metrics.roc_auc_score, greater_is_better=True, needs_proba=True)
    if scoring == 'binary_log_loss':
        return make_binary_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)
     
    if scoring=='accuracy_score':
        return make_scorer(metrics.accuracy_score, greater_is_better=True, needs_proba=False)
    if scoring=='precision_score':
        return make_scorer(metrics.precision_score, greater_is_better=True, needs_proba=False)
    if scoring=='f1_score':
        return make_scorer(metrics.f1_score, greater_is_better=True, needs_proba=False)
    if scoring=='roc_auc_score':
        return make_scorer(metrics.roc_auc_score, greater_is_better=True, needs_proba=True)
    if scoring == 'log_loss':
        return make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)
    
    if scoring == 'mean_squared_error':
        return make_scorer(metrics.mean_squared_error, greater_is_better=False, needs_proba=False)
    if scoring == 'r2_score':
        return make_scorer(metrics.r2_score, greater_is_better=False, needs_proba=False)
    
    raise Exception('Cannot comprehend scoring')

class binary_scorer(object):
    ''' auxilary to make_binary_scorer '''
    def __init__(self, score_fct):
        self.score_fct = score_fct
    def __call__(self, est, X, y_true):
        column_or_1d(y_true)
        y_pred = est.predict_proba(X)[:,1]
        return self.score_fct(y_true, y_pred)
        
def make_binary_scorer(score_fct, greater_is_better=False, needs_proba=False):
    ''' auxilary to scoring to score_fct '''
    if not needs_proba:
        return make_scorer(score_fct, greater_is_better=greater_is_better, needs_proba=needs_proba)
    else:
        return binary_scorer(score_fct)


class _binary_scoring_wrap():
    def __init__(self, multimetric):
        self.multimetric = multimetric
    def __call__(self, y_true, y_score):
        return self.multimetric( y_true, y_score[:,1] )
