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
nn_acc = clf.score(Xtest, ytest)

print(f"NN: El accuracy encontrado fue {nn_acc * 100.0}%")

elapsed_time = time() - start_time
print("Time final: {}".format(elapsed_time))
dump(clf, 'archivo.joblib')


#%%
from sklearn.metrics import confusion_matrix

ypred = clf.predict(Xtest)

cm = confusion_matrix(ytest, ypred)

print(f"FMR: {100*cm[0][1]/len(ytest)}%")
print(f"FNMR: {100*cm[1][0]/len(ytest)}%")

#%%
