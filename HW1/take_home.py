# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 22:17:13 2019

@author: yifan
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp



# shuffle and split training and test sets
nrocs = 100
y_score_list = []
y_test_list = []
for i in range(nrocs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=i)
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                     random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    y_score_list.append(y_score[:, 2])
    y_test_list.append(y_test[:, 2])

# using methord in paper for in class
import math
def createROC(f, L):
    L_sorted = L.copy()
    L_sorted = [L[i] for i in np.argsort(f)[::-1]]
    TP = 0
    FP = 0
    R = []
    fprev = -math.inf
    P = len(L[L == 1])
    N = len(L[L == 0])
    i = 0
    while i <= len(L_sorted) - 1: 
        if f[i] != fprev:
            R.append((FP/N, TP/P))
            fprev = f[i]
        if L_sorted[i] == 1: 
            TP = TP + 1
        else:
            FP = FP + 1
        i = i + 1
    R.append((FP/N, TP/P))    
    fpr = [j[0] for j in R]
    tpr = [j[1] for j in R]
    
    AUC = 0
    for i in range(len(fpr)-1):
        AUC = AUC + (fpr[i+1] - fpr[i]) * tpr[i+1]
        
    return R, AUC


def INTERPOLATE(ROCP1, ROCP2, X):
    slope = (ROCP2[1] - ROCP1[1]) / (ROCP2[0] - ROCP1[0])
    return ROCP1[1] + slope * (X - ROCP1[0])

def TPR_FOR_FPR(fprsample, ROC, npts):
    i = 0
    while i < (npts -1) and ROC[i + 1][0] <= fprsample:
        i = i + 1
        if ROC[i + 1][0] == fprsample:
            return ROC[i + 1][1]
        else:
            return INTERPOLATE(ROC[i], ROC[i+1], fprsample)

def createROCtpr(predictedClass, actualClass):
    
    ROCS = []
    AUCS = []
    for i in range(len(y_score_list)):
        y_test_i = y_test_list[i]
        y_score_i = y_score_list[i] 
        ROC, AUC = createROC(y_score_i, y_test_i)
        ROCS.append(ROC)
        AUCS.append(AUC)
        
    sample = 50
    #nrocs = len(y_score_list)
    npts = [len(i) for i in y_score_list]
    
    tpravg = []
    fpr_list = list(np.linspace(0, 1, sample))
    R = []
    for i in fpr_list:
        tprsum = 0
        for j in range(nrocs):
            #print(len(npts[j]))
            result = TPR_FOR_FPR(i, ROCS[j], npts[j])
            print(result)
            tprsum = tprsum + result
        tpravg_j = tprsum/nrocs
        tpravg.append(tpravg_j)
        R.append((i, tpravg_j))
    AUC = 0
    for n in range(len(tpravg - 1)):
            AUC = AUC + (fpr_list[n+1] - fpr_list[n]) * tpravg[n+1]
    
    return R, AUC 

ROC, AUC = createROCtpr(y_score_list, y_test_list)

plt.figure()
lw = 2
class_num = 0
ROC, AUC = createROC(y_score[:, class_num], y_test[:, class_num])
#plt.plot(fpr[class_num], tpr[class_num], color='black',
         #lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[class_num])
plt.plot([j[0] for j in ROC], [j[1] for j in ROC], color='blue',
         lw=lw, label='ROC curve (area = %0.2f)' % AUC)

plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()