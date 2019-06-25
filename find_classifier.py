#%%
from time import time
from numpy import load
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.externals.joblib import dump

#%%
# The data is load
M = load("use_data.npy")

Xtrain = M.item().get('X_train')
Xtest = M.item().get('X_test')
ytrain = M.item().get('y_train')
ytest = M.item().get('y_test')

#%%
# This is for capturing the best value
start_time = time()

param_grid = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (8, 8, 8), (8, 10, 8)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

clf = GridSearchCV(MLP(max_iter=1000, early_stopping=True, validation_fraction=1.0 / 6), param_grid, cv=10, verbose=1, n_jobs=-1)
clf.fit(Xtrain, ytrain)

elapsed_time = time() - start_time
print("Time final: {}".format(elapsed_time))
dump(clf, 'archivo_grey.joblib')

#%%
nn_acc = clf.score(Xtest, ytest)
print(f"NN: El accuracy encontrado fue {nn_acc * 100.0}%")

#%%
from sklearn.metrics import confusion_matrix

ypred = clf.predict(Xtest)

cm = confusion_matrix(ytest, ypred)

print(f"FMR: {100*cm[0][1]/(cm[0][1] + cm[0][0])}%")
print(f"FNMR: {100*cm[1][0]/(cm[1][0] + cm[1][1])}%")

#%%
from scipy.stats import norm
import math
Z = norm.ppf
 
def SDT(hits, misses, fas, crs):
    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)
 
    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1: 
        hit_rate = 1 - half_hit
    if hit_rate == 0: 
        hit_rate = half_hit
 
    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1: 
        fa_rate = 1 - half_fa
    if fa_rate == 0: 
        fa_rate = half_fa
 
    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate)
    
    return  out
d_prime = SDT(sum(ypred), len(ypred)-sum(ypred), cm[1][0], cm[0][0])

print(d_prime)


#%%

import numpy as np
from math import sqrt

def get_d_prime(clf, Xtest, ytest):
    proba = clf.predict_proba(Xtest)
    values_genuines = []
    values_impostors = []
    
    for pr, real in zip(proba, ytest):
        if real == 0:
            values_impostors.append(pr[0])
        else:
            values_genuines.append(pr[0])
            
    impostors = np.array(values_impostors)
    genuines = np.array(values_genuines)
    
    std_impostors = np.std(impostors)
    std_genuines = np.std(genuines)
    
    print(f"std genuinos: {std_genuines**2}")
    print(f"std impostores: {std_impostors**2}")
    
    mean_impostors = np.mean(impostors)
    mean_genuines = np.mean(genuines)
    
    print(f"mean genuinos: {mean_genuines}")
    print(f"mean impostores: {mean_impostors}")
    
    d_prime = abs(mean_genuines-mean_impostors)/(sqrt(0.5*(std_impostors**2+std_genuines**2)))
    
    print(d_prime)
    return d_prime

get_d_prime(clf, Xtest, ytest)
    
print("\n\n\n\n\n")

#%%

from sklearn.externals.joblib import load as sk_load

clf = sk_load('archivo_grey.joblib')

#%%
proba = clf.predict_proba(Xtest[:3])
pred = clf.predict(Xtest[:3])
real = ytest[:3]

print(proba)
print(pred)
print(real)

#%%
