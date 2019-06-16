from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from layers import ONE_LAYER, TWO_LAYER, TREE_LAYER

# The data is load
Xtest, Xtrain, ytest, ytrain = loadData()  # This one have to be changed

# Is this nesesary?
ytrain = ytrain.ravel()

# This is for capturing the best value
start_time = time()

parameter_space = {
    'hidden_layer_sizes': ONE_LAYER + TWO_LAYER + TREE_LAYER,
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

cv = [(slice(None), slice(None))]  # Hack para no hacer cv
mlp = MLPClassifier(early_stopping=True, validation_fraction=1.0 / 4)
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=cv)
clf.fit(Xtrain, ytrain)
nn_acc = clf.score(Xtest, ytest)

print(f"NN: El accuracy encontrado fue {nn_acc * 100.0}%")

elapsed_time = time() - start_time
print("Time final: {}".format(elapsed_time))
