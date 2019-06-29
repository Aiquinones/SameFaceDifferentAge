
#%%
from easyfacenet.simple import facenet
import pickle
import os

"""
images = ['dataset/testing/036981A.jpg', 'dataset/testing/036981B.jpg', 'dataset/testing/036987B.jpg']
aligned = facenet.align_face(images)
#comparisons = facenet.compare(aligned)

embs = facenet.embedding(aligned)

print(embs)
"""

#%%
training_path = "dataset/training"
testing_path = "dataset/testing"

training_pickle_filename = 'training_grey.p'
testing_pickle_filename = 'testing_grey.p'


#%%

print("Procesando testing...")
testing_paths = [testing_path + "/" + path for path in os.listdir(testing_path) if path.split(".")[1] == "jpg"]
removed_testing = []
aligned_testing = facenet.align_face(testing_paths, removed_testing)
embs_testing = facenet.embedding(aligned_testing)

print("Testing:")
if len(removed_testing) == 0:
    print(f"Todas las caras fueron procesadas! ({len(embs_testing)}/{len(testing_paths)})")
else:
    print(f"Ciertas caras ({len(removed_testing)}) no pudieron ser definidas y fueron pobladas con un None.\nEl resto ({len(embs_testing)}/{len(testing_paths)}) fue procesado correctamente")

print("Procesando training...")
training_paths = [training_path + "/" + path for path in os.listdir(training_path) if path.split(".")[1] == "jpg"]
removed_training = []
aligned_training = facenet.align_face(training_paths, removed_training)
embs_training = facenet.embedding(aligned_training)

print("Training:")
if len(removed_testing) == 0:
    print(f"Todas las caras fueron procesadas! ({len(embs_training)}/{len(training_paths)})")
else:
    print(f"Ciertas caras ({len(removed_training)}) no pudieron ser definidas y fueron pobladas con un None.\nEl resto ({len(embs_training)}/{len(training_paths)}) fue procesado correctamente")


#%%

print("Escribiendo en memoria...")
training_dict = {}
i = 0
for path in training_paths:
    if path in removed_training:
        training_dict[path.split("/")[-1]] = None
    else:
        training_dict[path.split("/")[-1]] = embs_training[i]
        i += 1      

#%%
testing_dict = {}
i = 0
for path in testing_paths:
    if path in removed_testing:
        testing_dict[path.split("/")[-1]] = None
    else:
        testing_dict[path.split("/")[-1]] = embs_testing[i]  
        i += 1

with open(training_pickle_filename, 'wb') as fp:
    pickle.dump(training_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(testing_pickle_filename, 'wb') as fp:
    pickle.dump(testing_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
print(f"Los archivos fueron guardados exitosamente! Se encuentran en\nTraining: {training_pickle_filename}\nTesting: {testing_pickle_filename}")


"""
para leer corremos

with open('testing.p', 'rb') as fp:
    d = pickle.load(fp)

key: nombre del archivo
value: numpy.ndarray con las features
"""

#%%
