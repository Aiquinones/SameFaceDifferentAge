import pickle

with open('testing.p', 'rb') as fp:
    d = pickle.load(fp)

print(len(d.keys()))

with open('training.p', 'rb') as fp:
    d = pickle.load(fp)

print(len(d.keys()))