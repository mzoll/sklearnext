'''
Created on Nov 20, 2017

@author: marcel.zoll
'''

import sys, os
import pandas as pd
import numpy as np

import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_FeatureImportances(est):
    """
    Parameters
    ----------
    est : estimator instance
        needs to support call `get_feature_importances`
    """
    fi_list = sorted(est.get_feature_importances(), key= lambda e: e[1], reverse=True)    
    l,i = zip(*fi_list)
    
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

  
def plot_CategoryFork_FeatureImportances(cf, coverage_weighted=True, n_many=sys.maxsize):
    """
    Parameters
    ----------
    cf : CathegoryFork instance
        used to obtain feature importances and stuff
    coverage_weighted : bool
        weight in the coverage for the feature impotance to get a ballanced impression (default: True)
    """
    
    fi = cf.get_feature_importances_deep()
    c = cf.coverage_
    cov = np.array(c) if coverage_weighted else np.ones(len(c))
    
    fsi = [(f, sum(np.array(i)*cov)) for f,i in fi]
    fsi = sorted(fsi, key= lambda e: e[1], reverse=True)
    f = list(zip(*fsi))[0]
    
    fdict = { ff:p for p,ff in enumerate(f)  }
    
    sfi = sorted(fi, key= lambda e: fdict[e[0]])
    l,i = zip(*sfi)
    
    if coverage_weighted:
        cov = np.array(c)
        i = [np.array(ii)*cov for ii in i]
    
    if n_many < sys.maxsize:
        l = l[-n_many:]
        i = i[-n_many:]
        cov = cov[-n_many:]
       
    # generate some multi-dimensional data & arbitrary labels
    y_pos = np.arange(len(l))
    data = np.asarray(np.matrix(i).T)
    
    
    fig = matplotlib.pyplot.figure(figsize=(20, len(l)/3.), dpi=80)
    plt1 = fig.add_subplot(1,1,1)
    
    colors ='rgbmc'
    
    patch_handles = []
    left = np.zeros(len(l)) # left alignment of data starts at zero
    for i, d in enumerate(data):
        patch_handles.append(plt1.barh(y_pos, d, 
          color=colors[i%len(colors)], align='center', 
          left=left))
        # accumulate the left-hand offsets
        left += d
    
    plt.title("Feature importances")
    if coverage_weighted:
        plt.xlabel('weighted importance')
    else:
        plt.xlabel('importance')
    
    plt.yticks(range(len(l)), l, ha = 'left')
    plt.ylim([-1, len(l)])
    
    for i in plt1.get_yticklabels():
        i.set_fontsize(12)
    yax = plt1.get_yaxis()
    yax.set_tick_params(pad=-40)
  
    #--- legend
    l_handles = []
    for l,c in zip(cf.levels_, cf.coverage_):
        print(l,c)
        l_handles.append(mpatches.Patch(color=colors[len(l_handles)%len(colors)], label="%s (cov: %f)"%(l, c)))
        
    plt.legend(handles=l_handles, loc='upper right')
  
    return fig  


def plot_CategoryFork_prediction(cf, X):
    """
    Parameters
    ----------
    cf : CathegoryFork instance
        used to obtain feature importances and stuff
    X : pandas.DataFrame
        input data to obtain predictions of
    """
    Xp = pd.DataFrame(X[cf.varname])
    Xp['y'] = cf.predict(X)

    fig = matplotlib.pyplot.figure()
    plt1 = fig.add_subplot(1,1,1)
    
    y_plot = []
    for gk,df in Xp.groupby(cf._segmentX(Xp)):
        plt1.hist(df['y'], 100, alpha=0.8, label=str(gk), histtype=u'step') #barstacked #step
        y_plot.append(df['y'].values)
        
    plt.yscale("log", nonposx='clip')
    plt.legend(loc='upper right')

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


def plot_BinaryRocAucCurve( y, y_proba):
    from sklearn.metrics import roc_curve, auc
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
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return fig
