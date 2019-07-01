#%%
from keras_facenet import FaceNet
import os
import tqdm
import cv2
import pickle


#%% 
def get_images_from_filepaths(filepaths):
    imgs = []
    pbar = tqdm.tqdm(total=len(filepaths))
    for filepath in filepaths:
        pbar.update()
        imgs.append(cv2.imread(os.path.expanduser(filepath)))
    pbar.close()
        
    return imgs


#%%
def save_imgs(paths, embeddings, destination):
    d = {path: embedding for path, embedding in zip(paths, embeddings)}
    with open(destination, 'wb') as fp:
        pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)


#%%
if __name__ == "__main__":
    # %%
    dataset = "filtered"

    training_destination = f"dataset/pickles/KerasFaceNet/{dataset}/training.p"
    testing_destination = f"dataset/pickles/KerasFaceNet/{dataset}/testing.p"
    training_path = f"dataset/{dataset}/training"
    testing_path = f"dataset/{dataset}/testing"
    testing_paths = [testing_path + "/" + path for path in os.listdir(testing_path) if path.split(".")[1] == "png"]
    training_paths = [training_path + "/" + path for path in os.listdir(training_path) if path.split(".")[1] == "png"]

    embedder = FaceNet()
    # images is a list of images, each as an
    # np.ndarray of shape (H, W, 3).

    for paths, destination in [
            (training_paths, training_destination),
            (testing_paths,  testing_destination)]:

        print("Getting images...")
        images = get_images_from_filepaths(paths)
        print("Getting embeddings...")

        try:
            embeddings = embedder.embeddings(images, verbose=1)
        except:
            embeddings = embedder.embeddings(images)

        print("Saving embeddings...")
        save_imgs(paths, embeddings, destination)