'''
DO stuff
'''

import numpy as np

def RocAucScore(y_true, y_pred_proba):
    """ compute and print the ROC AUC score """
    from sklearn.metrics import roc_auc_score
    score = roc_auc_score(y_true, y_pred_proba)
    #print("RocAuc score : %f" % (score))
    return score

def LogLossScore(y_true, y_pred_proba):
    from sklearn.metrics import log_loss
    score = log_loss(y_true, y_pred_proba)
    #print("LogLoss score : %f" % (score))
    return score

def LogLossScore_adapted(y_true, y_pred_proba):
    from sklearn.metrics import log_loss
    score = log_loss(y_true, y_pred_proba) - log_loss(y_true, np.repeat( np.mean(y_true), len(y_true) )) 
    #print("Adapted LogLoss score : %f" % (score))
    return score


def ClassifierOvertrainCheck(y_train, y_train_pred_proba, y_test, y_test_pred_proba):
    """ print out measures for possible overtraining """
    #--- RocAuc
    score_train = RocAucScore(y_train, y_train_pred_proba)
    score_test = RocAucScore(y_test, y_test_pred_proba)
    print("RocAuc score :: train: %f  test: %f  diff: %e" % (score_train, score_test, score_train-score_test))
    
    #--- RocAuc
    score_train = LogLossScore(y_train, y_train_pred_proba)
    score_test = LogLossScore(y_test, y_test_pred_proba)
    print("LogLoss :: train: %f  test: %f  diff: %e" % (score_train, score_test, score_train-score_test))
    
    #--- RocAuc
    score_train = LogLossScore_adapted(y_train, y_train_pred_proba)
    score_test = LogLossScore_adapted(y_test, y_test_pred_proba)
    print("Adapted LogLoss :: train: %f  test: %f  diff: %e" % (score_train, score_test, score_train-score_test))


#========================
# Regressor task
#========================

def R2Score(y_true, y_pred):
    from sklearn.metrics import r2_score
    score = r2_score(y_true, y_pred) 
    #print("Adapted R2 score : %f" % (score))
    return score

def RegressorOvertrainCheck(y_train, y_train_pred, y_test, y_test_pred):
    """ print out measures for possible overtraining """
    #--- RocAuc
    score_train = R2Score(y_train, y_train_pred)
    score_test = R2Score(y_test, y_test_pred)
    print("R2 score :: train: %f  test: %f  diff: %e" % (score_train, score_test, score_train-score_test))
    
    