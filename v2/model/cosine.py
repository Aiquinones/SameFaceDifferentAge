#%%
from time import time
from numpy import load
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.externals.joblib import dump
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from numpy import linalg as LA
from tqdm import tqdm
from scipy.spatial.distance import cosine

#%%

graphs_folder = "graphs"
dataset = "filtered"
mode = "cosine"
dataset_path = f"dataset/npys/KerasFaceNet/{dataset}.npy"

# The data is load
M = load(dataset_path)

Xtrain = M.item().get('X_train')
Xtest = M.item().get('X_test')
ytrain = M.item().get('y_train')
ytest = M.item().get('y_test')

#%%
def graph_impostors_and_genuines(impostors, genuines, d_prime, folder, mode="eucledian", dataset="" ,style='seaborn-deep'):
    plt.style.use(style) 
    
    ratio = len(impostors)/len(genuines)
    print(f"Proporción: {ratio}. Ajustando tamaño...")
    
    genuines_graph = [genuines for _ in range(int(ratio))]
    genuines_graph = np.array(genuines_graph)

    bins = np.linspace(0, 2, 30)

    plt.hist([impostors, genuines_graph], bins, label=['Impostores', 'Genuinos'])
    plt.legend(loc='upper right')
    
    filename = f"{mode}_{dataset}_{'%.3f'%(d_prime)}.png"
    
    plt.savefig(f"{folder}/{filename}")

"""
metrics:
    from sklearn.metrics import confusion_matrix

    ypred = clf.predict(Xtest)

    cm = confusion_matrix(ytest, ypred)

    print(f"FMR: {100*cm[0][1]/(cm[0][1] + cm[0][0])}%")
    print(f"FNMR: {100*cm[1][0]/(cm[1][0] + cm[1][1])}%")
"""

def get_d_prime(clf, Xtest, ytest):
    
    values_genuines = []
    values_impostors = []
    
    pbar = tqdm(total=len(Xtest))
    for pr, real in zip(Xtest, ytest):
        pbar.update()
        
        v1, v2 = pr[:512], pr[512:]
        v1 = v1/LA.norm(v1)
        v2 = v2/LA.norm(v2)
        distance = cosine(v1, v2)
        distance = LA.norm(distance)
        
        if real == 0:
            values_impostors.append(distance)
        else:
            values_genuines.append(distance)
            
    pbar.close()
            
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
    
    return d_prime, impostors, genuines

d_prime, impostors, genuines = get_d_prime(None, Xtest, ytest)
    
print(d_prime)

#%%
graph_impostors_and_genuines(impostors, genuines, d_prime, graphs_folder, mode=mode, dataset=dataset)
