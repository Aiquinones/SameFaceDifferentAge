from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier as MLP

from numpy import load

# The data is load
# Xtest, Xtrain, ytest, ytrain = # Here have to be put the data

# Is this nesesary?
# ytrain = ytrain.ravel()

# This is for capturing the best value
start_time = time()

param_grid = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (8, 8, 8), (8, 10, 8)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

clf = GridSearchCV(MLP(early_stopping=True, validation_fraction=5.0 / 6), param_grid, cv=10)
clf.fit(Xtrain, ytrain)
nn_acc = clf.score(Xtest, ytest)

print(f"NN: El accuracy encontrado fue {nn_acc * 100.0}%")


elapsed_time = time() - start_time
print("Time final: {}".format(elapsed_time))
