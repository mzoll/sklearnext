'''
Created on Nov 20, 2017

@author: marcel.zoll
'''

import sys
import pandas as pd
import numpy as np

import matplotlib
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_FeatureImportances(est, n_many=sys.maxsize):
    """
    Parameters
    ----------
    est : estimator instance
        needs to support call `get_feature_importances`
    """
    fi_list = sorted(est.get_feature_importances(), key= lambda e: e[1], reverse=True)    
    l,i = zip(*fi_list)
    
    if n_many < sys.maxsize:
        l = l[:n_many]
        i = i[:n_many]
    
    fig = matplotlib.pyplot.figure(figsize=(20, len(fi_list)/3.), dpi=80)
    
    plt.title("Feature importances")
    plt.barh(range(len(l)), i, color="r", align="center")
    plt.yticks(range(len(l)), l, ha = 'left')
    plt.ylim([-1, len(l)])
    
    plt1 = fig.add_subplot(1,1,1)
    for i in plt1.get_yticklabels():
        i.set_fontsize(12)
    yax = plt1.get_yaxis()
    yax.set_tick_params(pad=-40)
    return fig


def plot_Predictions(y_pred):
    """
    Parameters
    ----------
    y_pred : narray
        predictions in arbitray interval
    """
    fig = matplotlib.pyplot.figure()
    plt1 = fig.add_subplot(1,1,1)
    plt1.hist(y_pred, 100, alpha=0.8, histtype=u'bar')
    plt1.set_yscale("log", nonposx='clip')
    return fig




def plot_BinaryOutcomeDist(X, y, varname, n_many=sys.maxsize):
    df =  pd.concat([X[[varname]], y], axis=1)
    ''' plots the distribution of binary outcomes given a categorical variable
    Parameters
    ----------
    X : pandas.DataFrame shape(n,:)
        dataframe containing the categorical variable
    y : pandas.Series shape(n,)   
        the positive outcome
    varname : string
        name of the categorical variable to indicate
    n_many : int > 0
        plot only the first `n_mnay` most frequent categorical lables
    '''
    
    def xthelper(df):
        s = df.iloc[:,1]
        n = np.sum(s.values)
        l = df.shape[0]
        return pd.Series([l-n, n, l], index=['neg', 'pos', 'all'])
    dx = df.copy()
    dx = dx.groupby(dx.iloc[:,0]).apply(xthelper)
    dx.sort_values('all', inplace=True)
    dx = dx.tail(n_many)
    
    # Data
    lables = dx.index.values
    rights = dx['pos'].values
    lefts = dx['neg'].values    
    sums = dx['all'].values
    
    max_x = max( np.max(lefts) , np.max(rights) )
    ratios = np.array(lefts/rights)
    ratios = [ '%f'%(n)   if np.isfinite(n) else 'NaN' for n in ratios  ]
    ratiolabels = [ '%s :: %d'%(r,s) for r,s in zip(ratios,sums) ]
    
    # Sort by number of sales staff
    idx = sums.argsort()
    lables, lefts, rights = [np.take(x, idx) for x in [lables, lefts, rights]]
    
    ys = np.arange(lables.size)
    
    fig, axes = plt.subplots(figsize=(20, len(lables)/3.), dpi=80, ncols=2, sharey=True)
    axes[0].barh(ys, lefts, align='center', color='r', alpha=0.2) #, zorder=10)
    axes[0].set(title='negatives')
    axes[1].barh(ys, rights, align='center', color='r', alpha=0.2) # zorder=10)
    axes[1].set(title='positives')
    
    axes[0].invert_xaxis()
    #axes[0].set(yticks=y, yticklabels=states)
    axes[0].set(yticks=ys, yticklabels='')
    axes[0].yaxis.tick_right()
    
    for ax in axes.flat:
        ax.margins(0.03)
        #ax.grid(True)
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.09)
        
    axes[0].set_xlim([0.9, max_x*1.05])
    axes[1].set_xlim([0.9, max_x*1.05])
    
    axes[0].set_xscale("log", nonposx='clip')
    axes[1].set_xscale("log", nonposx='clip')
    axes[0].invert_xaxis()
    
    for y_pos,r in zip(ys,ratiolabels):
        axes[0].text(1, y_pos, r, horizontalalignment='right', verticalalignment='center', fontsize =8, color ='k' )
    
    for y_pos,label in zip(ys,lables):
        axes[1].text(1, y_pos, label, horizontalalignment='left', verticalalignment='center', fontsize =8, color ='k' )
        
    return fig

