'''
Created on Mar 8, 2018

@author: marcel.zoll
'''

import sys
import pandas as pd
import numpy as np

import matplotlib
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import average_precision_score


def plot_BinaryRocCurve( y, y_proba):
    fpr_1, tpr_1, _ = roc_curve(y, y_proba)
    roc_auc_1 = auc(fpr_1, tpr_1)
    
    fig = matplotlib.pyplot.figure()
    plt1 = fig.add_subplot(1,1,1)
    lw = 2
    plt.plot(fpr_1, tpr_1, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Binary Receiver operating characteristic')
    #plt.legend(loc="lower right")
    return fig


def plot_BinaryPrCurve(y, y_proba):
    average_precision = average_precision_score(y, y_proba)
    precision, recall, _ = precision_recall_curve(y, y_proba)
    prc_auc = auc(precision, recall, 1)
    
    fig = matplotlib.pyplot.figure()
    plt1 = fig.add_subplot(1,1,1)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Binary Precision-Recall curve: AP={0:0.2f} PRC_AUC={1:0.2f}'.format(average_precision, prc_auc))
    plt.hlines(1, 0., 1.)
    return fig