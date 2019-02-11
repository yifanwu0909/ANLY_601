
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:10:29 2018


In Class Exercise (submit before end of class for +10 extra credit).
******************************************************
Goal: Implement function 
ROC, AUC = createROC(predictedClass, actualClass)
******************************************************

which computes a ROC given list of predicted classes and list of actualClasses.
Specifically construct ROC as a dictionary with entries for "tpr" and "fpr": the 
arrays of the true positive rates and false positive rates.

This function will also compute the area under the curve, AUC using simple Riemann 
estimate.

Compare your results to the existing methods roc_curve and auc in sklearn, by overlaying
the ROCs computed by your function (in blue) and the ROCs computed by sklearn (in black). 


Take-Home Assignment (Extra Credit +20).
******************************************************
Read: ROC Analysis paper posted on course site.
To Do: Implement two more ROC curve creating functions with error bars
******************************************************

Specifically, the creation of a ROC curve can be seen as a trial, the results of which
may vary based on the variability of data. The ROC analysis paper acknowledges this 
concern and addresses it by designing two algorithms which result in ROCs with error bars:
one algorithm addressing variability in TPR and one addressing variability in FPR.

Goals: Implement functions
ROC, AUC = createROCtpr(predictedClass, actualClass)
ROC, AUC = createROCfpr(predictedClass, actualClass)


Resource(s) that may help:
    https://matplotlib.org/
    

@author: jerem
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



# 

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test) #predict_proba
y_prob = classifier.fit(X_train, y_train).predict_proba(X_test)
y_class = classifier.fit(X_train, y_train).predict(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 2
class_num = 1
plt.plot(fpr[class_num], tpr[class_num], color='black',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[class_num])

# Display your results here.
# plt.plot(..., ..., color='black',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])

plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#%%

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



plt.figure()
lw = 2
class_num = 0
R, AUC = createROC(y_score[:, class_num], y_test[:, class_num])
plt.plot(fpr[class_num], tpr[class_num], color='black',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[class_num])
plt.plot([j[0] for j in R], [j[1] for j in R], color='blue',
         lw=lw, label='ROC curve (area = %0.2f)' % AUC)

plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#They are exactly the same!





























