import pickle
from numpy import save, concatenate
import random

def getXy(dic, fiftyfifty=True):
    X = []
    Y = []
    
    keys = [key for key in dic]
        
    for x in dic:
        path = x[:-4]
        number = path[:-1]
        AB_coding = path[-1:]
        
        if AB_coding == "B":
            continue
        if dic[x] is None:
            continue 
        
        other_coding = "A" if AB_coding == "B" else "B"
        y = number + other_coding + ".jpg"
        
        if dic[y] is not None:
            equal = concatenate((dic[x], dic[y]))
            X.append(equal)
            Y.append(1)
        
        if fiftyfifty:
            not_y = random.choice(keys)
            while not_y != y:
                not_y = random.choice(keys)
            
            if dic[not_y] is not None:
                not_equal = concatenate((dic[x], dic[not_y]))
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
                not_equal = concatenate((dic[x], dic[not_y]))
                X.append(not_equal)
                Y.append(0)
            
    return X, Y


with open('training_grey.p', 'rb') as f:
    dataset_train = pickle.load(f)
with open('testing_grey.p', 'rb') as fp:
    dataset_test = pickle.load(fp)

# print(dataset_train)
X_train, y_train = getXy(dataset_train)
X_test, y_test = getXy(dataset_test, fiftyfifty=False)

save("use_data_not_grey.npy", {
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test,
})
