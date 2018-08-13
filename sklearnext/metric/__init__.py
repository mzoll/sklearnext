'''
Specify some common performance indicators for classification and regression
'''

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, r2_score, precision_recall_curve, auc
    

def RocAucScore(y_true, y_pred_proba):
    """ compute and print the ROC AUC score """
    score = roc_auc_score(y_true, y_pred_proba)
    #print("RocAuc score : %f" % (score))
    return score

def LogLossScore(y_true, y_pred_proba):
    score = log_loss(y_true, y_pred_proba)
    #print("LogLoss score : %f" % (score))
    return score

def LogLossScore_adapted(y_true, y_pred_proba):
    score = log_loss(y_true, y_pred_proba) / log_loss(y_true, np.repeat( np.mean(y_true), len(y_true) )) 
    #print("Adapted LogLoss score : %f" % (score))
    return score

def PrcAucScore(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(precision, recall, 1)


def ClassifierOvertrainCheck(y_train, y_train_pred_proba, y_test, y_test_pred_proba):
    """ print out measures for possible overtraining """
    #--- RocAuc
    score_train = RocAucScore(y_train, y_train_pred_proba)
    score_test = RocAucScore(y_test, y_test_pred_proba)
    print("RocAuc score :: train: %f  test: %f  diff: %e" % (score_train, score_test, score_train-score_test))
    #--- PrcAuc
    score_train = PrcAucScore(y_train, y_train_pred_proba)
    score_test = PrcAucScore(y_test, y_test_pred_proba)
    print("PrcAuc score :: train: %f  test: %f  diff: %e" % (score_train, score_test, score_train-score_test))
    #--- LogLoss
    score_train = LogLossScore(y_train, y_train_pred_proba)
    score_test = LogLossScore(y_test, y_test_pred_proba)
    print("LogLoss :: train: %f  test: %f  diff: %e" % (score_train, score_test, score_train-score_test))
    #--- LogLoss_adapted
    score_train = LogLossScore_adapted(y_train, y_train_pred_proba)
    score_test = LogLossScore_adapted(y_test, y_test_pred_proba)
    print("Adapted LogLoss :: train: %f  test: %f  diff: %e" % (score_train, score_test, score_train-score_test))
    


#========================
# Regressor task
#========================

def R2Score(y_true, y_pred):
    score = r2_score(y_true, y_pred) 
    #print("Adapted R2 score : %f" % (score))
    return score

def RegressorOvertrainCheck(y_train, y_train_pred, y_test, y_test_pred):
    """ print out measures for possible overtraining """
    #--- R2
    score_train = R2Score(y_train, y_train_pred)
    score_test = R2Score(y_test, y_test_pred)
    print("R2 score :: train: %f  test: %f  diff: %e" % (score_train, score_test, score_train-score_test))
    
    