from json import dump
from os import makedirs
from os.path import exists, isfile
from statistics import mean
from time import time

from pybalu.performance_eval import performance
from sklearn.neural_network import MLPClassifier

from layers import ONE_LAYER, TWO_LAYER, TREE_LAYER

# The folder where the results are going to be saved
FOLDER_NAME = 'neural_network'

# If the folder don't exist, is created first
if not exists("./{}".format(FOLDER_NAME)):
    makedirs("./{}".format(FOLDER_NAME))

# The data is load
Xtest, Xtrain, ytest, ytrain = loadData()  # This one have to be changed

# Is this nesesary?
ytrain = ytrain.ravel()

# This is for capturing the best value
start_time = time()
best_accuracy = 0
best_name = ""

# For every convination of 1 to 3 layer  of 1 to 10 neurons
for layer_n in [ONE_LAYER, TWO_LAYER, TREE_LAYER]:
    for layers in layer_n:
        # For every type of activation
        for activation in ['identity', 'logistic', 'tanh', 'relu']:
            # For every type of solver
            for solver in ['lbfgs', 'sgd', 'adam']:
                file_name = f"./{FOLDER_NAME}/{layers}_{activation}_{solver}.json"

                if not isfile(file_name):
                    total_accuracy = []
                    # 8 times to find the average
                    for time in range(8):
                        # The predictor and created
                        predictor = MLPClassifier(
                            hidden_layer_sizes=layers,
                            activation=activation,
                            solver=solver,
                            max_iter=10000
                        )
                        # Is trained
                        predictor.fit(Xtrain, ytrain)
                        # Is tested
                        Y_pred = predictor.predict(Xtrain)
                        accuracy = str(performance(Y_pred, ytrain))
                        # And saved
                        print(f"File: {file_name}. {time + 1}.- Accuracy =  {accuracy}")
                    average = mean(total_accuracy)
                    # If is the best is used for later
                    with open(file_name, "w") as file:
                        dump({
                            "average": average,
                            "accuracies": total_accuracy
                        }, file)
                    if float(average) > best_accuracy:
                        best_accuracy = float(average)
                        best_name = file_name

# And is used the best
print(f"\nInicial:\nAccuracy:{best_accuracy} , name:{best_name}\n")

elapsed_time = time() - start_time
print("Time final: {}".format(elapsed_time))
