from os import listdir, mkdir
from os.path import exists

from keras_facenet import FaceNet
from numpy import linalg as LA
from scipy.spatial.distance import cosine

from create_dataset.create_filtered_dataset import filter_faces, save_images
from v2.model.facenet_from_keras import get_images_from_filepaths


def do_examples(image_path, embedder):
    filtered_path = f"{image_path}_filtered"
    if not exists(filtered_path):
        mkdir(filtered_path)

    # Create filtered dataset
    sorted_path = listdir(image_path)
    sorted_path.sort()
    testing_paths = [f"{image_path}/{path}" for path in sorted_path if path.split(".")[1] == "jpg"]
    cropped_testing, cropped_paths_testing = filter_faces(testing_paths)

    save_images(cropped_testing, filtered_path, cropped_paths_testing)

    # FaceNet From keras
    filtered_paths = [filtered_path + "/" + path for path in listdir(filtered_path) if path.split(".")[1] == "png"]
    print("Getting images...")
    images = get_images_from_filepaths(filtered_paths)
    print("Getting embeddings...")

    try:
        embeddings = embedder.embeddings(images, verbose=1)
    except:
        embeddings = embedder.embeddings(images)

    dictionary_vectors = {path: embedding for path, embedding in zip(filtered_paths, embeddings)}

    # Distance
    for image in ["1", "2", "3"]:
        v1 = dictionary_vectors[f"{filtered_path}/{image}A.png"]
        v2 = dictionary_vectors[f"{filtered_path}/{image}B.png"]
        v1 = v1 / LA.norm(v1)
        v2 = v2 / LA.norm(v2)
        distance = cosine(v1, v2)
        distance = LA.norm(distance)
        print(distance)


if __name__ == "__main__":
    genral_embedder = FaceNet()

    paths = ["./demo/wrong_impostor",
             "./demo/right_impostor",
             "./demo/wrong_genuine",
             "./demo/right_genuine"]
    for act in paths:
        do_examples(act, genral_embedder)
        input()

"""
wrong_impostor: impostor clasificado como genuino
1.- 036981A.jpg-046217B.jpg
2.- 038004A.jpg-040757B.jpg
3.- 037087A.jpg-109451B.jpg

right impostor: impostor clasificado como impostor
1.- 091700A.jpg-054799B.jpg
2.- 094788A.jpg-101925B.jpg
3.- 106940A.jpg-105934B.jpg

wrong genuine: genuino clasificado como impostor
1.- 111955A.jpg-111955B.jpg
2.- 099230A.jpg-099230B.jpg
3.- 085596A.jpg-085596B.jpg

right genuine: genuino clasificado como genunio
1.- 040551A.jpg-040551B.jpg
2.- 110586A.jpg-110586B.jpg
3.- 072098A.jpg-072098B.jpg
"""
