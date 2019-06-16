import random
import numpy as np

dataset = {}

versions = ["A", "B"]
people = [str(i) for i in range(300, 310)]
feature_size = 2048

for p in people:
        for v in versions:
                if random.random() < 0.1: # algunos son None
                        dataset[p+v] = None
                else:
                        dataset[p+v] = np.array([random.random() for _ in range(feature_size)])
print(dataset)

def getXy(dic, training):
	X_names = []
	X = []
	Y = []
	for x in dataset:
		numA = x[:-1]
		abA = x[-1:]
		if abA == "A":
			done1 = 0
			done2 = 0
			done = 0
			equal = []
			notequal = []
			equalnames = ""
			notequalnames = ""
			for y in dataset:
				numB = y[:-1]
				abB = y[-1:]
				if abB == "B":
					if not training and numA == numB and dataset[x] is not None and dataset[y] is not None:
						X.append(dataset[x]+dataset[y])
						Y.append(1)
						X_names.append(x+y)
					elif not training and numA != numB and dataset[x] is not None and dataset[y] is not None:
						X.append(dataset[x]+dataset[y])
						Y.append(0)
						X_names.append(x+y)
					if training and numA == numB and dataset[x] is not None and dataset[y] is not None and not done1:
						equalnames = x+y
						equal = dataset[x]+dataset[y]
						done1 = 1
					if training and numA != numB and not done2 and dataset[x] is not None and dataset[y] is not None:
						notequalnames = x+y
						notequal = dataset[x]+dataset[y]
						done2 = 1
				if done1 and done2 and not done and training:
					X.append(equal)
					Y.append(1)
					X.append(notequal)
					Y.append(0)
					X_names.append(equalnames)
					X_names.append(notequalnames)
					done = 1
					break
	return X, Y, X_names

X_train, y_train, xnames_train = getXy(dataset,1)
X_test, y_test, xnames_test = getXy(dataset,0)
