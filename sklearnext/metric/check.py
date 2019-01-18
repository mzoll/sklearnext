import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, r2_score, precision_recall_curve, auc

_CLIP = 1E-6

def RocAucScore(y_true, y_pred_proba):
    """ compute and print the ROC AUC score """
    try:
        score = roc_auc_score(y_true, y_pred_proba)
        #print("RocAuc score : %f" % (score))
        return score
    except:
        return float('nan')

def LogLossScore(y_true, y_pred_proba):
    try:
        y_pred_proba = np.clip(y_pred_proba, _CLIP, 1.-_CLIP)
        score = log_loss(y_true, y_pred_proba)
        #print("LogLoss score : %f" % (score))
        return score
    except:
        return float('nan')
def LogLossScore_adapted(y_true, y_pred_proba):
    try:
        y_pred_proba = np.clip(y_pred_proba, _CLIP, 1.-_CLIP)
        score = log_loss(y_true, y_pred_proba) / log_loss(y_true, np.repeat( np.mean(y_true), len(y_true) ))
        #print("Adapted LogLoss score : %f" % (score))
        return score
    except:
        return float('nan')

def PrcAucScore(y_true, y_pred_proba):
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return auc(precision, recall, 1)
    except:
        return float('nan')


def ClassifierOvertrainCheck(y_train, y_train_pred_proba, y_test, y_test_pred_proba):
    """ print out measures for possible overtraining """
    #---General
    print("Support :: train %d  test %d" % (len(y_train), len(y_test)) )
    print("Coverage :: train %f  test %f" % (np.mean(y_train), np.mean(y_test)) )
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
    try:
        score = r2_score(y_true, y_pred)
        #print("Adapted R2 score : %f" % (score))
        return score
    except:
        return float('nan')

def RegressorOvertrainCheck(y_train, y_train_pred, y_test, y_test_pred):
    """ print out measures for possible overtraining """
    #---General
    print("Support :: train %d  test %d" % (len(y_train), len(y_test)) )
    #--- R2
    score_train = R2Score(y_train, y_train_pred)
    score_test = R2Score(y_test, y_test_pred)
    print("R2 score :: train: %f  test: %f  diff: %e" % (score_train, score_test, score_train-score_test))
    
    