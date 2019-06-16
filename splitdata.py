import random
import numpy as np
import pickle

def getXy(dic, training):
	X = []
	Y = []
	for x in dic:
		xp = x[:-4]
		numA = xp[:-1]
		abA = xp[-1:]
		if abA == "A":
			done1 = 0
			done2 = 0
			done = 0
			equal = []
			notequal = []
			for y in dic:
				yp = y[:-4]
				numB = yp[:-1]
				abB = yp[-1:]
				if abB == "B":
					if not training and numA == numB and dic[x] is not None and dic[y] is not None:
						X.append(dic[x]+dic[y])
						Y.append(1)
					elif not training and numA != numB and dic[x] is not None and dic[y] is not None:
						X.append(dic[x]+dic[y])
						Y.append(0)
					if training and numA == numB and dic[x] is not None and dic[y] is not None and not done1:
						equal = dic[x]+dic[y]
						done1 = 1
					if training and numA != numB and not done2 and dic[x] is not None and dic[y] is not None:
						notequal = dic[x]+dic[y]
						done2 = 1
				if done1 and done2 and not done and training:
					X.append(equal)
					Y.append(1)
					X.append(notequal)
					Y.append(0)
					done = 1
					break
	return X, Y

with open('training.p', 'rb') as f:
	dataset_train = pickle.load(f)
with open('testing.p', 'rb') as fp:
	dataset_test = pickle.load(fp)

# print(dataset_train)
X_train, y_train = getXy(dataset_train,1)
X_test, y_test= getXy(dataset_test,0)
