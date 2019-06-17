from sklearn.metrics import confusion_matrix

from sklearn.externals.joblib import load as sk_load
from numpy import load

clf = sk_load('archivo.joblib')

M = load("use_data.npy")

Xtest = M.item().get('X_test')
ytest = M.item().get('y_test')

ypred = clf.predict(Xtest)

print(confusion_matrix(ytest, ypred))
