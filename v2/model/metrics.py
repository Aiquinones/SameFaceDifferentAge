import numpy as np
from math import sqrt

def get_d_prime(impostors, genuines):
    """
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
    """
    std_impostors = np.std(impostors)
    std_genuines = np.std(genuines)
    
    print(f"std genuinos: {std_genuines**2}")
    print(f"std impostores: {std_impostors**2}")
    
    mean_impostors = np.mean(impostors)
    mean_genuines = np.mean(genuines)
    
    print(f"mean genuinos: {mean_genuines}")
    print(f"mean impostores: {mean_impostors}")
    
    d_prime = abs(mean_genuines-mean_impostors)/(sqrt(0.5*(std_impostors**2+std_genuines**2)))
    return d_prime    