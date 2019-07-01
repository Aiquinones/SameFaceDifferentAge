from ntpath import basename
from os import listdir

from keras_facenet import FaceNet
from numpy import linalg as LA, expand_dims
from scipy.spatial.distance import cosine

# from create_dataset.create_filtered_dataset import testing_path, filter_faces
from v2.model.facenet_from_keras import get_images_from_filepaths
from v2.model.splitdata_modified import getXy


if __name__ == "__main__":
    # Create filtered dataset
    testing_path = "./dataset/filtered/testing"
    sorted_path = listdir(testing_path)
    sorted_path.sort()
    testing_paths = [f"{testing_path}/{path}" for path in sorted_path if path.split(".")[1] == "png"]
    # cropped_testing, cropped_paths_testing = filter_faces(testing_paths)
    #for img in cropped_testing:
     #   img = expand_dims(img, axis=3)

    #print("Getting images...")

    images = get_images_from_filepaths(testing_paths)
    print("Getting embeddings...")

    # FaceNet From keras
    embedder = FaceNet()

    try:
        embeddings = embedder.embeddings(images, verbose=1)
    except:
        embeddings = embedder.embeddings(images)

    dictionary_vectors = {path: embedding for path, embedding in zip(testing_paths, embeddings)}

    # Split Data
    names = []
    for name_index in range(len(testing_paths) // 2):
        names.append(basename(testing_paths[name_index * 2])[:-5])

    # Distance
    for name1 in names:
        for name2 in names:
            v1, v2 = \
                dictionary_vectors[f"./dataset/filtered/testing/{name1}A.png"],\
                dictionary_vectors[f"./dataset/filtered/testing/{name2}B.png"]
            v1 = v1 / LA.norm(v1)
            v2 = v2 / LA.norm(v2)
            distance = cosine(v1, v2)
            distance = LA.norm(distance)

            if name1 != name2 and distance < 0.5:
                print(f"Wrong impostor {name1}A-{name2}B: {distance}")

            if name1 != name2 and distance > 1.25:
                print(f"Rigth impostor {name1}A-{name2}B: {distance}")

            if name1 == name2 and distance > 0.75:
                print(f"Wrong genuine {name1}A-{name2}B: {distance}")

            if name1 == name2 and distance < 0.25:
                print(f"Rigth genuine {name1}A-{name2}B: {distance}")
