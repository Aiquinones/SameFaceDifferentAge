#%%
from time import time
from numpy import load
from scipy.spatial.distance import cosine
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.externals.joblib import dump

#%%
# The data is load
M = load("use_data_normalized.npy")

Xtrain = M.item().get('X_train')
Xtest = M.item().get('X_test')
ytrain = M.item().get('y_train')
ytest = M.item().get('y_test')

#%%
"""
from sklearn.metrics import confusion_matrix

ypred = clf.predict(Xtest)

cm = confusion_matrix(ytest, ypred)

print(f"FMR: {100*cm[0][1]/(cm[0][1] + cm[0][0])}%")
print(f"FNMR: {100*cm[1][0]/(cm[1][0] + cm[1][1])}%")
"""

#%%

import numpy as np
from math import sqrt
from numpy import linalg as LA


def get_d_prime(clf, Xtest, ytest):
    
    
    values_genuines = []
    values_impostors = []
    
    for pr, real in zip(Xtest, ytest):
        v1, v2 = pr[:512], pr[512:]
        v1 = v1/LA.norm(v1)
        v2 = v2/LA.norm(v2)
        distance = cosine(v1, v2)
        distance = LA.norm(distance)
        
        if real == 0:
            values_impostors.append(distance)
        else:
            values_genuines.append(distance)
            
    impostors = np.array(values_impostors)
    genuines = np.array(values_genuines)
    
    std_impostors = np.std(impostors)
    std_genuines = np.std(genuines)
    
    print(f"std genuinos: {std_genuines}")
    print(f"std impostores: {std_impostors}")
    
    mean_impostors = np.mean(impostors)
    mean_genuines = np.mean(genuines)
    
    print(f"mean genuinos: {mean_genuines}")
    print(f"mean impostores: {mean_impostors}")
    
    d_prime = abs(mean_genuines-mean_impostors)/(sqrt(0.5*(std_impostors+std_genuines)))
    
    print(d_prime)
    return d_prime

get_d_prime(None, Xtest, ytest)
    
print("\n\n\n\n\n")