'''
Created on Jun 15, 2018

@author: marcel.zoll
'''


import sys
import pandas as pd
import numpy as np

import matplotlib
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

  
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
        l = l[:n_many]
        i = i[:n_many]
        cov = cov[:n_many]
       
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