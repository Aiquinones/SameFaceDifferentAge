import pickle
from numpy import save, concatenate
import random
from numpy import linalg as LA
import numpy as np
import tqdm

#040888A
def getXy(dic, file_type="png", fiftyfifty=True):
    X = []
    Y = []
    
    keys = [key for key in dic]
    
    progress_bar = tqdm.tqdm(total=len(dic))
    for x in dic:
        path = x[:-4]
        number = path[:-1]
        AB_coding = path[-1:]
        
        if AB_coding == "B":
            continue
        if dic[x] is None:
            continue 
        
        other_coding = "A" if AB_coding == "B" else "B"
        y = number + other_coding + "." + file_type
        
        vector_x = dic[x]
        vector_x = vector_x / LA.norm(vector_x)
        
        if dic[y] is not None:
            vector_y = dic[y]
            vector_y = vector_y / LA.norm(vector_y)
            
            equal = concatenate((vector_x, vector_y))
            X.append(equal)
            Y.append(1)
        
        if fiftyfifty:
            not_y = random.choice(keys)
            while not_y != y:
                not_y = random.choice(keys)
            
            if dic[not_y] is not None:
                vector_y = dic[not_y]
                vector_y = vector_y / LA.norm(vector_y)
                
                not_equal = concatenate((vector_x, vector_y))
                X.append(not_equal)
                Y.append(0)
                
        else:
            for not_y in dic:
                AB = not_y[-5:-4]
                if AB == "A":
                    continue
                if dic[not_y] is None:
                    continue 
                if not_y == y:
                    continue
               
                vector_y = dic[not_y]
                vector_y = vector_y / LA.norm(vector_y)
                
                not_equal = concatenate((vector_x, vector_y)) 
                X.append(not_equal)
                Y.append(0)
                
        progress_bar.update()
    
    progress_bar.close()
    return X, Y


def split_data(training_pickle, testing_pickle, destination):
    
    with open(training_pickle, 'rb') as f:
        dataset_train = pickle.load(f)
    with open(testing_pickle, 'rb') as fp:
        dataset_test = pickle.load(fp)

    print(dataset_train)

    X_train, y_train = getXy(dataset_train)
    X_test, y_test = getXy(dataset_test, fiftyfifty=False)

    save(destination, {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    })


training_pickle = 'dataset/pickles/KerasFaceNet/filtered/training.p'
testing_pickle = 'dataset/pickles/KerasFaceNet/filtered/testing.p'
destination = "dataset/npys/KerasFaceNet/filtered.npy"

if __name__ == "__main__":
    split_data(training_pickle, testing_pickle, destination)