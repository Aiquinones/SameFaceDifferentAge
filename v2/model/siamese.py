#%%
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np
from math import sqrt

epochs = 5
FEATURES_SIZE = 512

def get_d_prime(impostors, genuines):
    std_impostors = np.std(impostors)
    std_genuines = np.std(genuines)
    
    print(f"std genuinos: {std_genuines}")
    print(f"std impostores: {std_impostors}")
    
    mean_impostors = np.mean(impostors)
    mean_genuines = np.mean(genuines)
    
    print(f"mean genuinos: {mean_genuines}")
    print(f"mean impostores: {mean_impostors}")
    
    d_prime = abs(mean_genuines-mean_impostors)/(sqrt(0.5*(std_impostors+std_genuines)))
    return d_prime 

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    #TODO: loss you possibly be modified to be d-prime (?)
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    for x_i in x:
        pairs += [[x_i[:FEATURES_SIZE], x_i[FEATURES_SIZE:]]]
    return np.array(pairs)


def create_base_network(input_shape, dimensions=[128,128,128]):
    '''Base network to be shared (eq. to feature extraction).
    '''
    inp = Input(shape=input_shape)
    x = inp
    for i, dim in enumerate(dimensions):
        x = Dense(dim, activation='relu')(x)
        if i != len(dimensions) - 1:
            x = Dropout(0.1)(x)
    return Model(inp, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


#%%
# The data is load
M = np.load("dataset/npys/KerasFaceNet/main.npy")

Xtrain = np.array(M.item().get('X_train'))
Xtest = np.array(M.item().get('X_test'))
ytrain = np.array(M.item().get('y_train'))
ytest = np.array(M.item().get('y_test'))
#%%

print(f"X train: {Xtrain.shape}")
print(f"X test: {Xtest.shape}")

#%%
"""
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
"""
# /2 since X has two vectors concatenated
input_shape = Xtrain.shape[1:]
input_shape = (int(input_shape[0]/2),)
print(f"input shape is {Xtrain.shape} -> {input_shape}")

#%%

# create training + test positive and negative pairs
tr_pairs = create_pairs(Xtrain)
te_pairs = create_pairs(Xtest)

# network definition
young_mlp = create_base_network(input_shape)
old_mlp = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# we use different instances of the mlp,
# the weights of the network
# will be different between the two branches
processed_a = young_mlp(input_a)
processed_b = old_mlp(input_b)

# TODO: modify with wanted "distance"
distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
# TODO: replace accuracy with "accuracy"
model.compile(loss=contrastive_loss, optimizer=rms, metrics=["accuracy"])

model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], ytrain,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], ytest))
# TODO: validation data

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(ytrain, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(ytest, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

#%%
tr_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
te_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
predictions = np.concatenate([tr_pred, te_pred])
ys = np.concatenate([ytrain, ytest])

impostors = []
genuines = []

for pr, real in zip(predictions, ys):
    if real == 0:
        impostors.append(pr)
    else:
        genuines.append(pr)

impostors = np.array(impostors)
genuines = np.array(genuines)

d_prime = get_d_prime(impostors, genuines)
print(d_prime)
#%%
d_prime

#%%
